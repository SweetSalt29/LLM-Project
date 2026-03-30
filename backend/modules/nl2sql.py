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
    cleaned = sql.strip().upper()

    if not cleaned.startswith("SELECT"):
        return False, "Only SELECT queries are allowed. Write operations are blocked."

    for keyword in FORBIDDEN_KEYWORDS:
        pattern = rf"\b{keyword}\b"
        if re.search(pattern, cleaned):
            return False, f"Query contains forbidden keyword: {keyword}"

    if "--" in sql or "/*" in sql:
        return False, "SQL comments are not allowed."

    return True, ""


# ============================================================
# COLUMN NAME SANITIZER
# Renames columns with spaces/special chars so SQLite handles
# them cleanly, and keeps a mapping back to original names.
# ============================================================
def sanitize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Replace spaces and special characters in column names with underscores.
    Returns sanitized DataFrame and a mapping {new_name: original_name}.
    """
    mapping = {}
    new_columns = []

    for col in df.columns:
        new_col = re.sub(r"[^a-zA-Z0-9]", "_", str(col)).strip("_")
        # Handle duplicates after sanitization
        base = new_col
        count = 1
        while new_col in mapping:
            new_col = f"{base}_{count}"
            count += 1
        mapping[new_col] = col
        new_columns.append(new_col)

    df = df.copy()
    df.columns = new_columns
    return df, mapping


# ============================================================
# FILE LOADER — load CSV/Excel into in-memory SQLite
# ============================================================
def load_files_to_sqlite(file_paths: list[str]) -> tuple[sqlite3.Connection, dict]:
    """
    Load each file into an in-memory SQLite DB.
    Columns are sanitized so SQLite never sees spaces or slashes.
    Returns (connection, schema_info) where schema_info maps
    table_name -> list of sanitized column names.
    """
    conn = sqlite3.connect(":memory:")
    schema_info = {}

    for path in file_paths:
        suffix = Path(path).suffix.lower()
        table_name = Path(path).stem.lower()
        table_name = re.sub(r"[^a-z0-9_]", "_", table_name)

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            continue  # skip unsupported silently

        # Sanitize column names before loading into SQLite
        df, col_mapping = sanitize_columns(df)

        df.to_sql(table_name, conn, index=False, if_exists="replace")

        # Store sanitized column names in schema (what LLM must use)
        schema_info[table_name] = list(df.columns)

    return conn, schema_info


# ============================================================
# SCHEMA BUILDER
# Shows exact column names the LLM must use, with a sample row
# ============================================================
def build_schema_string(schema_info: dict, conn: sqlite3.Connection) -> str:
    """
    Build a detailed schema string with:
    - Exact sanitized column names
    - Column data types inferred from actual data
    - 3 sample rows so LLM understands content and can identify numeric columns
    """
    lines = []
    for table, columns in schema_info.items():
        lines.append(f"Table: `{table}`")
        lines.append("Columns (use EXACT names as shown — already sanitized, no spaces):")

        try:
            sample_df = pd.read_sql_query(f"SELECT * FROM `{table}` LIMIT 3", conn)

            for col in columns:
                dtype = sample_df[col].dtype if col in sample_df.columns else "unknown"
                # Map pandas dtype to human-readable type
                if pd.api.types.is_integer_dtype(dtype):
                    col_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    col_type = "FLOAT"
                elif pd.api.types.is_bool_dtype(dtype):
                    col_type = "BOOLEAN"
                else:
                    col_type = "TEXT"
                lines.append(f"  - `{col}` ({col_type})")

            if not sample_df.empty:
                lines.append(f"Sample data ({min(3, len(sample_df))} rows):")
                lines.append(sample_df.to_string(index=False))

        except Exception:
            # Fallback: just list column names without types
            for col in columns:
                lines.append(f"  - `{col}`")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# LLM CALL
# ============================================================
def call_llm(messages: list) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": messages
        }
    )
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ============================================================
# SQL GENERATION — with conversation history for context
# ============================================================
def generate_sql(user_query: str, schema_string: str, history: list) -> str:
    history_text = ""
    if history:
        recent = history[-12:]
        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        ])

    prompt_text = (
        f"You are an expert SQLite assistant.\n\n"
        f"Database schema:\n{schema_string}\n"
    )

    if history_text:
        prompt_text += f"\nConversation so far:\n{history_text}\n"

    prompt_text += (
        f"\nUser now asks: \"{user_query}\"\n\n"
        "Rules:\n"
        "- Write ONLY a valid SQLite SELECT query.\n"
        "- Use the EXACT column names from the schema — do not guess, rename, or derive new ones.\n"
        "- Column names are already sanitized (no spaces or special characters).\n"
        "- Use the sample data rows to identify which columns hold numeric values for aggregations.\n"
        "- For AVG, SUM, MIN, MAX — only use columns shown as INTEGER or FLOAT in the schema.\n"
        "- Never use CAST, SUBSTR, or string manipulation to extract numbers from text columns.\n"
        "- Do NOT use INSERT, UPDATE, DELETE, DROP, ALTER, or any write operations.\n"
        "- Do NOT include markdown, code blocks, backtick fences, or any explanation.\n"
        "- Use conversation history to resolve references like 'that column', 'same filter'.\n"
        "- Output ONLY the raw SQL query and nothing else.\n\n"
        "SQL Query:"
    )

    raw = call_llm([{"role": "user", "content": prompt_text}])

    # Strip any accidental markdown fences
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip().rstrip("```").strip()
    return raw


# ============================================================
# NATURAL LANGUAGE ANSWER
# ============================================================
def generate_natural_answer(
    user_query: str,
    sql_query: str,
    result_markdown: str,
    history: list
) -> str:
    history_text = ""
    if history:
        recent = history[-12:]
        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        ])

    prompt_text = "You are a helpful data analyst assistant.\n\n"

    if history_text:
        prompt_text += f"Conversation so far:\n{history_text}\n\n"

    prompt_text += (
        f"The user asked: \"{user_query}\"\n"
        f"SQL run: {sql_query}\n"
        f"Result:\n{result_markdown}\n\n"
        "Write a clear, concise, friendly natural language answer. "
        "Reference prior conversation context if relevant. "
        "Do not repeat the SQL query."
    )

    return call_llm([{"role": "user", "content": prompt_text}])


# ============================================================
# CONVERSATION SUMMARIZER
# ============================================================
def summarize_conversation(history: list) -> str:
    if not history:
        return "No conversation to summarize yet."

    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history
    ])

    prompt_text = (
        "Summarize the following data analysis conversation. "
        "Highlight: key questions asked, SQL queries run, and important findings.\n\n"
        f"Conversation:\n{history_text}\n\nSummary:"
    )

    return call_llm([{"role": "user", "content": prompt_text}])


# ============================================================
# RESULT FORMATTER
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
    session_id: str = None,
    history: list = None
) -> dict:
    if session_id is None:
        session_id = str(uuid.uuid4())
    if history is None:
        history = []

    # 1. Load files (columns sanitized inside)
    try:
        conn, schema_info = load_files_to_sqlite(file_paths)
    except Exception as e:
        return _error_response(user_query, f"Failed to load data files: {str(e)}")

    if not schema_info:
        return _error_response(user_query, "No valid CSV or Excel files found to query.")

    # 2. Build schema string (passes conn for sample rows)
    schema_string = build_schema_string(schema_info, conn)

    # 3. Generate SQL
    try:
        sql_query = generate_sql(user_query, schema_string, history)
    except Exception as e:
        return _error_response(user_query, f"Failed to generate SQL: {str(e)}")

    # 4. Guardrail check
    is_safe, reason = is_safe_sql(sql_query)
    if not is_safe:
        return _error_response(user_query, f"Unsafe query blocked: {reason}", sql_query)

    # 5. Execute SQL
    try:
        result_df = pd.read_sql_query(sql_query, conn)
    except Exception as e:
        return _error_response(user_query, f"SQL execution failed: {str(e)}", sql_query)
    finally:
        conn.close()

    result_markdown = dataframe_to_markdown(result_df)

    # 6. Natural language answer
    try:
        natural_answer = generate_natural_answer(user_query, sql_query, result_markdown, history)
    except Exception:
        natural_answer = "Could not generate a natural language summary."

    # 7. Log
    try:
        log_nl2sql(
            user_id=user_id,
            session_id=session_id,
            user_query=user_query,
            sql_query=sql_query
        )
    except Exception:
        pass

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