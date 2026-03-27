import os
import re
import sqlite3
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path

import requests

# ============================================================
# CONFIG
# ============================================================
UPLOAD_DIR = "data/uploads"
NL2SQL_DB = "app.db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "meta-llama/llama-3-8b-instruct"


# ============================================================
# DATABASE — query log table
# ============================================================
def init_nl2sql_log_table():
    conn = sqlite3.connect(NL2SQL_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nl2sql_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            session_id  TEXT    NOT NULL,
            user_query  TEXT    NOT NULL,
            sql_query   TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_nl2sql_log_table()


def log_nl2sql(user_id: int, session_id: str, user_query: str, sql_query: str):
    conn = sqlite3.connect(NL2SQL_DB)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO nl2sql_logs
           (user_id, session_id, user_query, sql_query, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, session_id, user_query, sql_query, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


# ============================================================
# GUARDRAILS — allow SELECT only
# ============================================================
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "MERGE", "EXEC",
    "EXECUTE", "PRAGMA", "ATTACH", "DETACH"
]

def is_safe_sql(sql: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    Blocks anything that isn't a pure SELECT statement.
    """
    cleaned = sql.strip().upper()

    # Must start with SELECT
    if not cleaned.startswith("SELECT"):
        return False, "Only SELECT queries are allowed. Write operations are blocked."

    # Block forbidden keywords anywhere in the query
    for keyword in FORBIDDEN_KEYWORDS:
        # Word boundary check to avoid matching e.g. "SELECTED"
        pattern = rf"\b{keyword}\b"
        if re.search(pattern, cleaned):
            return False, f"Query contains forbidden keyword: {keyword}"

    # Block SQL comments (used to bypass filters)
    if "--" in sql or "/*" in sql:
        return False, "SQL comments are not allowed."

    return True, ""


# ============================================================
# FILE LOADER — load CSV/Excel into in-memory SQLite
# ============================================================
def load_files_to_sqlite(file_paths: list[str]) -> tuple[sqlite3.Connection, dict]:
    """
    Load each CSV/Excel file into an in-memory SQLite DB.
    Returns (connection, schema_info) where schema_info maps
    table_name -> list of column names.
    """
    conn = sqlite3.connect(":memory:")
    schema_info = {}

    for path in file_paths:
        suffix = Path(path).suffix.lower()
        table_name = Path(path).stem.lower()
        # Sanitize table name — remove non-alphanumeric chars
        table_name = re.sub(r"[^a-z0-9_]", "_", table_name)

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            continue  # skip unsupported files silently

        df.to_sql(table_name, conn, index=False, if_exists="replace")
        schema_info[table_name] = list(df.columns)

    return conn, schema_info


# ============================================================
# SCHEMA BUILDER — format schema for LLM prompt
# ============================================================
def build_schema_string(schema_info: dict) -> str:
    """
    Format schema dict as readable text for the LLM.
    Example:
        Table: sales
        Columns: product, quantity, price, date
    """
    lines = []
    for table, columns in schema_info.items():
        lines.append(f"Table: {table}")
        lines.append(f"Columns: {', '.join(columns)}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# LLM CALLS
# ============================================================
def call_llm(prompt: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_sql(user_query: str, schema_string: str) -> str:
    """
    Ask LLM to generate a SQL SELECT query from the schema + question.
    """
    prompt = f"""You are an expert SQL assistant.

You are given the following database schema:
{schema_string}

The user asks:
"{user_query}"

Rules:
- Write ONLY a valid SQLite SELECT query.
- Do NOT use INSERT, UPDATE, DELETE, DROP, ALTER, or any write operations.
- Do NOT include markdown formatting, code blocks, or any explanation.
- Output ONLY the raw SQL query and nothing else.

SQL Query:"""

    raw = call_llm(prompt)

    # Strip any accidental markdown code fences the LLM might add
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip().rstrip("```").strip()
    return raw


def generate_natural_answer(user_query: str, sql_query: str, result_markdown: str) -> str:
    """
    Ask LLM to convert raw query results into a friendly natural language answer.
    """
    prompt = f"""You are a helpful data analyst assistant.

The user asked: "{user_query}"

The SQL query run was:
{sql_query}

The result of the query is:
{result_markdown}

Write a clear, concise, friendly natural language answer summarizing the result.
Do not repeat the SQL query. Just answer the user's question directly."""

    return call_llm(prompt)


# ============================================================
# RESULT FORMATTER — DataFrame → Markdown table
# ============================================================
def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No results found._"
    return df.to_markdown(index=False)


# ============================================================
# MAIN PIPELINE
# ============================================================
def nl2sql_pipeline(
    user_query: str,
    user_id: int,
    file_paths: list[str],
    session_id: str = None
) -> dict:
    """
    Full NL2SQL pipeline.

    Returns:
    {
        "user_query": str,
        "sql_query": str,
        "natural_answer": str,
        "result_markdown": str,
        "error": str | None
    }
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    # ----------------------
    # 1. Load files into SQLite
    # ----------------------
    try:
        conn, schema_info = load_files_to_sqlite(file_paths)
    except Exception as e:
        return _error_response(user_query, f"Failed to load data files: {str(e)}")

    if not schema_info:
        return _error_response(user_query, "No valid CSV or Excel files found to query.")

    schema_string = build_schema_string(schema_info)

    # ----------------------
    # 2. Generate SQL via LLM
    # ----------------------
    try:
        sql_query = generate_sql(user_query, schema_string)
    except Exception as e:
        return _error_response(user_query, f"Failed to generate SQL: {str(e)}")

    # ----------------------
    # 3. Guardrail check
    # ----------------------
    is_safe, reason = is_safe_sql(sql_query)
    if not is_safe:
        return _error_response(user_query, f"Unsafe query blocked: {reason}", sql_query)

    # ----------------------
    # 4. Execute SQL
    # ----------------------
    try:
        result_df = pd.read_sql_query(sql_query, conn)
    except Exception as e:
        return _error_response(user_query, f"SQL execution failed: {str(e)}", sql_query)
    finally:
        conn.close()

    result_markdown = dataframe_to_markdown(result_df)

    # ----------------------
    # 5. Generate natural language answer
    # ----------------------
    try:
        natural_answer = generate_natural_answer(user_query, sql_query, result_markdown)
    except Exception as e:
        natural_answer = "Could not generate a natural language summary."

    # ----------------------
    # 6. Log to DB
    # ----------------------
    try:
        log_nl2sql(
            user_id=user_id,
            session_id=session_id,
            user_query=user_query,
            sql_query=sql_query
        )
    except Exception:
        pass  # Don't fail the response just because logging failed

    return {
        "user_query": user_query,
        "sql_query": sql_query,
        "natural_answer": natural_answer,
        "result_markdown": result_markdown,
        "error": None
    }


def _error_response(user_query: str, error: str, sql_query: str = "N/A") -> dict:
    return {
        "user_query": user_query,
        "sql_query": sql_query,
        "natural_answer": None,
        "result_markdown": None,
        "error": error
    }