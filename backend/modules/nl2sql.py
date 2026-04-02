import os
import re
import sqlite3
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path

import requests

from backend.modules.chat_memory import (
    get_standalone_context,
    save_standalone_message
)

# ============================================================
# CONFIG
# ============================================================
NL2SQL_DB          = "app.db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
REWRITE_MODEL      = "meta-llama/llama-3.2-3b-instruct"
MODEL              = "meta-llama/llama-3-8b-instruct"


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
# GUARDRAILS
# ============================================================
FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "MERGE", "EXEC",
    "EXECUTE", "PRAGMA", "ATTACH", "DETACH"
]

def is_safe_sql(sql: str) -> tuple[bool, str]:
    cleaned = sql.strip().upper()
    if not cleaned.startswith("SELECT"):
        return False, "Only SELECT queries are allowed."
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", cleaned):
            return False, f"Query contains forbidden keyword: {keyword}"
    if "--" in sql or "/*" in sql:
        return False, "SQL comments are not allowed."
    return True, ""


# ============================================================
# COLUMN SANITIZER
# ============================================================
def sanitize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mapping     = {}
    new_columns = []
    for col in df.columns:
        new_col = re.sub(r"[^a-zA-Z0-9]", "_", str(col)).strip("_")
        base, count = new_col, 1
        while new_col in mapping:
            new_col = f"{base}_{count}"
            count  += 1
        mapping[new_col] = col
        new_columns.append(new_col)
    df = df.copy()
    df.columns = new_columns
    return df, mapping


# ============================================================
# FILE LOADER — load all session files into in-memory SQLite
# Each file becomes its own table so multi-file queries work.
# ============================================================
def load_files_to_sqlite(file_paths: list[str]) -> tuple[sqlite3.Connection, dict]:
    """
    Load each CSV/Excel file as a separate table in one in-memory SQLite DB.
    schema_info: {table_name: [sanitized_col, ...]}
    file_map:    {table_name: original_file_name}  — for attribution in prompts.
    """
    conn        = sqlite3.connect(":memory:")
    schema_info = {}
    file_map    = {}

    for path in file_paths:
        suffix     = Path(path).suffix.lower()
        table_name = Path(path).stem.lower()
        table_name = re.sub(r"[^a-z0-9_]", "_", table_name)

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            continue

        df, _ = sanitize_columns(df)
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        schema_info[table_name] = list(df.columns)
        file_map[table_name]    = Path(path).name

    return conn, schema_info, file_map


# ============================================================
# SCHEMA BUILDER
# Includes file name per table so LLM knows which file = which table.
# ============================================================
def build_schema_string(
    schema_info: dict,
    conn: sqlite3.Connection,
    file_map: dict
) -> str:
    lines = []
    for table, columns in schema_info.items():
        original_name = file_map.get(table, table)
        lines.append(f"Table: `{table}` (from file: '{original_name}')")
        lines.append("Columns (sanitized — use EXACT names):")

        try:
            sample_df = pd.read_sql_query(f"SELECT * FROM `{table}` LIMIT 3", conn)
            for col in columns:
                dtype = sample_df[col].dtype if col in sample_df.columns else "unknown"
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
            for col in columns:
                lines.append(f"  - `{col}`")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# LLM CALLS
# ============================================================
def call_llm(messages: list, model: str = MODEL) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json"
        },
        json={"model": model, "messages": messages}
    )
    return response.json()["choices"][0]["message"]["content"].strip()


def call_rewrite_llm(messages: list) -> str:
    return call_llm(messages, model=REWRITE_MODEL)


# ============================================================
# CONTEXT BUILDER
# ============================================================
def build_context_str(context_rows: list) -> str:
    if not context_rows:
        return ""
    lines = []
    for i, row in enumerate(context_rows, 1):
        lines.append(f"Turn {i}:")
        lines.append(f"  Q: {row['standalone_query']}")
        lines.append(f"  A: {row['answer_summary']}")
    return "\n".join(lines)


# ============================================================
# STANDALONE REWRITE
# ============================================================
def rewrite_as_standalone(user_query: str, context_str: str) -> str:
    if not context_str.strip():
        return user_query

    messages = [
        {
            "role": "user",
            "content": (
                "You are a query rewriting assistant.\n\n"
                "Given prior conversation context and a follow-up question, "
                "rewrite the follow-up into a fully self-contained, explicit question "
                "that can be understood WITHOUT any prior context.\n\n"
                "Rules:\n"
                "- Replace ALL pronouns (it, they, this, that, its, their) "
                "with the explicit noun they refer to.\n"
                "- Expand vague references like 'the table', 'the same filter', "
                "'that column', 'the above' to their specific subject.\n"
                "- If the question is already standalone, return it unchanged.\n"
                "- Output ONLY the rewritten question — no explanation.\n\n"
                f"Prior conversation context:\n{context_str}\n\n"
                f"Follow-up question: {user_query}\n\n"
                "Rewritten standalone question:"
            )
        }
    ]
    try:
        rewritten = call_rewrite_llm(messages).strip()
        return rewritten if rewritten and len(rewritten) <= 500 else user_query
    except Exception:
        return user_query


# ============================================================
# ANSWER SUMMARIZER
# ============================================================
def summarize_answer(standalone_query: str, natural_answer: str) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "Summarize the following answer in 2-3 sentences only. "
                "Be concise and preserve the key facts.\n\n"
                f"Question: {standalone_query}\n"
                f"Answer: {natural_answer}\n\n"
                "2-3 sentence summary:"
            )
        }
    ]
    try:
        summary = call_rewrite_llm(messages).strip()
        return summary if summary else natural_answer[:300]
    except Exception:
        return natural_answer[:300]


# ============================================================
# SQL GENERATION
# file_map injected into prompt so LLM knows table → file mapping.
# ============================================================
def generate_sql(
    user_query: str,
    schema_string: str,
    context_str: str,
    file_map: dict
) -> str:
    file_note = "\n".join([
        f"  - Table `{t}` comes from file '{f}'"
        for t, f in file_map.items()
    ])

    prompt = (
        f"You are an expert SQLite assistant.\n\n"
        f"Database schema:\n{schema_string}\n"
        f"File-to-table mapping:\n{file_note}\n"
    )

    if context_str:
        prompt += f"\nPrior conversation context:\n{context_str}\n"

    prompt += (
        f"\nUser asks: \"{user_query}\"\n\n"
        "Rules:\n"
        "- Write ONLY a valid SQLite SELECT query.\n"
        "- Use EXACT column names from the schema.\n"
        "- For cross-file questions, JOIN or UNION tables as appropriate.\n"
        "- For AVG, SUM, MIN, MAX — only use INTEGER or FLOAT columns.\n"
        "- Do NOT use INSERT, UPDATE, DELETE, DROP, ALTER, or write operations.\n"
        "- Do NOT include markdown, code blocks, or explanation.\n"
        "- Output ONLY the raw SQL query.\n\n"
        "SQL Query:"
    )

    raw = call_llm([{"role": "user", "content": prompt}])
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip().rstrip("```").strip()
    return raw


# ============================================================
# NATURAL LANGUAGE ANSWER
# ============================================================
def generate_natural_answer(
    user_query: str,
    sql_query: str,
    result_markdown: str,
    context_str: str,
    file_map: dict
) -> str:
    file_list = ", ".join(f"'{v}'" for v in file_map.values())

    prompt = (
        f"You are a helpful data analyst assistant.\n"
        f"The data comes from these file(s): {file_list}\n\n"
    )

    if context_str:
        prompt += f"Prior conversation context:\n{context_str}\n\n"

    prompt += (
        f"User asked: \"{user_query}\"\n"
        f"SQL run: {sql_query}\n"
        f"Result:\n{result_markdown}\n\n"
        "Write a clear, concise, friendly natural language answer. "
        "When the result spans multiple files/tables, clearly state which file each insight comes from. "
        "Do not repeat the SQL query."
    )

    return call_llm([{"role": "user", "content": prompt}])


# ============================================================
# CONVERSATION SUMMARIZER
# ============================================================
def summarize_conversation(session_id: str) -> str:
    context_rows = get_standalone_context(session_id, mode="sql", limit=5)
    if not context_rows:
        return "No conversation to summarize yet."

    context_str = build_context_str(context_rows)
    prompt = (
        "Summarize the following data analysis conversation. "
        "Highlight: key questions asked, SQL queries run, and important findings.\n\n"
        f"Conversation:\n{context_str}\n\nSummary:"
    )
    return call_llm([{"role": "user", "content": prompt}])


# ============================================================
# RESULT FORMATTER
# ============================================================
def dataframe_to_markdown(df: pd.DataFrame) -> str:
    return "_No results found._" if df.empty else df.to_markdown(index=False)


# ============================================================
# VISUALIZATION DECIDER
# Uses a lightweight LLM call to decide chart type + axis columns.
# Returns a viz_config dict consumed by the Streamlit frontend.
#
# chart_type options: "bar" | "line" | "pie" | "scatter" | "none"
# "none" means: single-row detail query, or non-comparable result.
# ============================================================
def decide_visualization(
    user_query: str,
    result_df: pd.DataFrame,
) -> dict:
    """
    Decide whether and how to visualize the query result.

    Rules applied by the LLM:
    - Single-row results (details about one item) → none
    - Aggregates comparing multiple categories/items → bar
    - Time-series or ordered numeric sequences → line
    - Part-of-whole (percentages, shares) → pie
    - Correlation between two numeric columns → scatter
    - Text-only or ambiguous results → none

    Returns:
        {
            "chart_type": "bar" | "line" | "pie" | "scatter" | "none",
            "x_col":      "<column name or null>",
            "y_col":      "<column name or null>",
            "title":      "<short chart title or null>"
        }
    """
    # Fast-path: empty or single-row — no chart needed
    if result_df is None or result_df.empty or len(result_df) <= 1:
        return {"chart_type": "none", "x_col": None, "y_col": None, "title": None}

    columns      = list(result_df.columns)
    num_rows     = len(result_df)
    num_cols     = len(columns)
    col_types    = {c: str(result_df[c].dtype) for c in columns}
    sample_rows  = result_df.head(3).to_dict(orient="records")

    prompt = (
        "You are a data visualization advisor. Given a user's query and the structure of "
        "its SQL result, decide the best chart type to visualize it.\n\n"
        f"User query: \"{user_query}\"\n"
        f"Result columns: {columns}\n"
        f"Column dtypes: {col_types}\n"
        f"Number of rows: {num_rows}\n"
        f"Sample rows (first 3): {sample_rows}\n\n"
        "Decision rules:\n"
        "- If the result is a single row describing one specific item/entity → chart_type: none\n"
        "- If comparing aggregates (counts, totals, averages) across categories → chart_type: bar\n"
        "- If data is ordered over time or sequence → chart_type: line\n"
        "- If showing proportions or percentages of a whole → chart_type: pie\n"
        "- If showing correlation between two numeric columns → chart_type: scatter\n"
        "- If result is mostly text, IDs, or not meaningfully comparable → chart_type: none\n\n"
        "Pick the SINGLE best x_col (categorical/label axis) and y_col (numeric/value axis) "
        "from the actual column names above. Use exact column names.\n\n"
        "Respond ONLY with a valid JSON object — no markdown, no explanation:\n"
        '{"chart_type": "<bar|line|pie|scatter|none>", "x_col": "<col_name or null>", '
        '"y_col": "<col_name or null>", "title": "<short descriptive chart title or null>"}'
    )

    try:
        raw = call_rewrite_llm([{"role": "user", "content": prompt}])
        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip().rstrip("```").strip()
        import json
        config = json.loads(raw)

        # Validate chart_type
        valid_types = {"bar", "line", "pie", "scatter", "none"}
        if config.get("chart_type") not in valid_types:
            config["chart_type"] = "none"

        # Validate column names exist in DataFrame
        if config.get("x_col") and config["x_col"] not in columns:
            config["x_col"] = None
        if config.get("y_col") and config["y_col"] not in columns:
            config["y_col"] = None

        # If chart type needs axes but columns are missing → fall back to none
        if config["chart_type"] in ("bar", "line", "scatter") and (
            not config.get("x_col") or not config.get("y_col")
        ):
            config["chart_type"] = "none"

        if config["chart_type"] == "pie" and not config.get("x_col"):
            config["chart_type"] = "none"

        return config

    except Exception as e:
        print(f"[decide_visualization] Failed: {e}")
        return {"chart_type": "none", "x_col": None, "y_col": None, "title": None}


# ============================================================
# MAIN PIPELINE
# ============================================================
def nl2sql_pipeline(
    user_query: str,
    user_id: int,
    file_paths: list[str],
    session_id: str = None,
) -> dict:
    if session_id is None:
        session_id = str(uuid.uuid4())

    # 1. Standalone context
    context_rows = get_standalone_context(session_id, mode="sql", limit=5)
    context_str  = build_context_str(context_rows)

    # 2. Rewrite query
    standalone_query = rewrite_as_standalone(user_query, context_str)

    # 3. Load files (session-scoped)
    try:
        conn, schema_info, file_map = load_files_to_sqlite(file_paths)
    except Exception as e:
        return _error_response(user_query, f"Failed to load data files: {str(e)}")

    if not schema_info:
        return _error_response(user_query, "No valid CSV or Excel files found to query.")

    # 4. Build schema
    schema_string = build_schema_string(schema_info, conn, file_map)

    # 5. Generate SQL
    try:
        sql_query = generate_sql(standalone_query, schema_string, context_str, file_map)
    except Exception as e:
        return _error_response(user_query, f"Failed to generate SQL: {str(e)}")

    # 6. Guardrail
    is_safe, reason = is_safe_sql(sql_query)
    if not is_safe:
        return _error_response(user_query, f"Unsafe query blocked: {reason}", sql_query)

    # 7. Execute
    try:
        result_df = pd.read_sql_query(sql_query, conn)
    except Exception as e:
        return _error_response(user_query, f"SQL execution failed: {str(e)}", sql_query)
    finally:
        conn.close()

    result_markdown = dataframe_to_markdown(result_df)

    # 8. Natural answer
    try:
        natural_answer = generate_natural_answer(
            user_query, sql_query, result_markdown, context_str, file_map
        )
    except Exception:
        natural_answer = "Could not generate a natural language summary."

    # 9. Summarize for storage
    answer_summary = summarize_answer(standalone_query, natural_answer)

    # 10. Save standalone turn
    save_standalone_message(
        session_id=session_id,
        mode="sql",
        user_query=user_query,
        standalone_query=standalone_query,
        llm_answer=natural_answer,
        answer_summary=answer_summary
    )

    # 11. Log SQL
    try:
        log_nl2sql(user_id, session_id, user_query, sql_query)
    except Exception:
        pass

    # 12. Decide visualization (non-blocking — never fails the pipeline)
    try:
        viz_config = decide_visualization(user_query, result_df)
    except Exception:
        viz_config = {"chart_type": "none", "x_col": None, "y_col": None, "title": None}

    # Serialize DataFrame rows for frontend chart rendering
    result_data = result_df.to_dict(orient="records") if not result_df.empty else []

    return {
        "user_query":      user_query,
        "sql_query":       sql_query,
        "natural_answer":  natural_answer,
        "result_markdown": result_markdown,
        "result_data":     result_data,
        "viz_config":      viz_config,
        "error":           None
    }


def _error_response(user_query: str, error: str, sql_query: str = "N/A") -> dict:
    return {
        "user_query":      user_query,
        "sql_query":       sql_query,
        "natural_answer":  None,
        "result_markdown": None,
        "result_data":     [],
        "viz_config":      {"chart_type": "none", "x_col": None, "y_col": None, "title": None},
        "error":           error
    }