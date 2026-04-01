"""
chat_memory.py
Handles all persistent storage for chat sessions, messages, summaries,
file library, and standalone LLM context.
"""

import sqlite3
import uuid
import json
from datetime import datetime, timezone

DB_PATH = "app.db"


# ============================================================
# INIT TABLES
# ============================================================
def init_chat_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── File library — one row per uploaded file per user ──────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_library (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            file_name    TEXT    NOT NULL,
            file_path    TEXT    NOT NULL,
            pipeline     TEXT    NOT NULL,
            indexed      INTEGER NOT NULL DEFAULT 0,
            uploaded_at  TEXT    NOT NULL
        )
    """)

    # ── Chat sessions ───────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id   TEXT PRIMARY KEY,
            user_id      INTEGER NOT NULL,
            title        TEXT,
            mode         TEXT,
            file_paths   TEXT,
            summary      TEXT,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        )
    """)

    # Migrate existing tables that lack file_paths column
    existing = [r[1] for r in cursor.execute("PRAGMA table_info(chat_sessions)").fetchall()]
    if "file_paths" not in existing:
        cursor.execute("ALTER TABLE chat_sessions ADD COLUMN file_paths TEXT")

    # ── Chat messages — UI display only ────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            role         TEXT NOT NULL,
            content      TEXT NOT NULL,
            timestamp    TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
    """)

    # ── Standalone messages — LLM context only ─────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS standalone_messages (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT NOT NULL,
            mode             TEXT NOT NULL,
            user_query       TEXT NOT NULL,
            standalone_query TEXT NOT NULL,
            llm_answer       TEXT NOT NULL,
            answer_summary   TEXT NOT NULL,
            timestamp        TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
    """)

    conn.commit()
    conn.close()


init_chat_tables()


# ============================================================
# FILE LIBRARY OPERATIONS
# ============================================================
def file_exists_in_library(user_id: int, file_path: str) -> bool:
    """
    Returns True if this exact file_path is already registered for this user.
    Used to prevent duplicate uploads and re-embedding.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM file_library WHERE user_id = ? AND file_path = ? LIMIT 1",
        (user_id, file_path)
    )
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def register_file(
    user_id: int,
    file_name: str,
    file_path: str,
    pipeline: str
) -> int:
    """
    Register an uploaded file in the library.
    SQL files are marked indexed=1 immediately (no embedding needed).
    RAG files start at indexed=0 and are updated after FAISS embedding.
    Returns the new file id.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO file_library
           (user_id, file_name, file_path, pipeline, indexed, uploaded_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            file_name,
            file_path,
            pipeline,
            1 if pipeline == "sql" else 0,
            datetime.now(timezone.utc).isoformat()
        )
    )
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id


def mark_file_indexed(file_path: str, user_id: int):
    """Mark a RAG file as fully indexed after FAISS embedding completes."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE file_library SET indexed = 1 WHERE file_path = ? AND user_id = ?",
        (file_path, user_id)
    )
    conn.commit()
    conn.close()


def get_user_library(user_id: int, pipeline: str) -> list:
    """
    Return all ready (indexed=1) files for a user filtered by pipeline.
    Newest first.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, file_name, file_path, pipeline, indexed, uploaded_at
           FROM file_library
           WHERE user_id = ? AND pipeline = ? AND indexed = 1
           ORDER BY uploaded_at DESC""",
        (user_id, pipeline)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id":          r[0],
            "file_name":   r[1],
            "file_path":   r[2],
            "pipeline":    r[3],
            "indexed":     bool(r[4]),
            "uploaded_at": r[5]
        }
        for r in rows
    ]


def get_pending_files(user_id: int) -> list:
    """Return RAG files still being embedded (indexed=0)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT file_name, file_path
           FROM file_library
           WHERE user_id = ? AND indexed = 0""",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"file_name": r[0], "file_path": r[1]} for r in rows]


# ============================================================
# SESSION OPERATIONS
# ============================================================
def create_session(
    user_id: int,
    mode: str,
    file_paths: list,
    title: str = None
) -> str:
    """
    Create a new chat session with file_paths locked at creation time.
    file_paths stored as JSON — immutable for the session lifetime.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO chat_sessions
           (session_id, user_id, title, mode, file_paths, summary, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            user_id,
            title or "New Chat",
            mode,
            json.dumps(file_paths),
            None,
            now,
            now
        )
    )
    conn.commit()
    conn.close()
    return session_id


def get_session_file_paths(session_id: str) -> list:
    """Return the file_paths locked to a session as a Python list."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_paths FROM chat_sessions WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if not row or not row[0]:
        return []
    return json.loads(row[0])


def get_user_sessions(user_id: int, mode: str) -> list:
    """Get all sessions for a user in a given mode, newest first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT session_id, title, summary, file_paths, created_at, updated_at
           FROM chat_sessions
           WHERE user_id = ? AND mode = ?
           ORDER BY updated_at DESC""",
        (user_id, mode)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "session_id": r[0],
            "title":      r[1],
            "summary":    r[2],
            "file_paths": json.loads(r[3]) if r[3] else [],
            "created_at": r[4],
            "updated_at": r[5]
        }
        for r in rows
    ]


def update_session_title(session_id: str, title: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
        (title[:60], datetime.now(timezone.utc).isoformat(), session_id)
    )
    conn.commit()
    conn.close()


def update_session_summary(session_id: str, summary: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
        (summary, datetime.now(timezone.utc).isoformat(), session_id)
    )
    conn.commit()
    conn.close()


def touch_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
        (datetime.now(timezone.utc).isoformat(), session_id)
    )
    conn.commit()
    conn.close()


# ============================================================
# CHAT MESSAGE OPERATIONS (UI display only)
# ============================================================
def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO chat_messages (session_id, role, content, timestamp)
           VALUES (?, ?, ?, ?)""",
        (session_id, role, content, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()
    touch_session(session_id)


def get_session_messages(session_id: str) -> list:
    """Load full message history for a session, oldest first. UI display only."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT role, content, timestamp
           FROM chat_messages
           WHERE session_id = ?
           ORDER BY id ASC""",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]


# ============================================================
# STANDALONE MESSAGE OPERATIONS (LLM context)
# ============================================================
def save_standalone_message(
    session_id: str,
    mode: str,
    user_query: str,
    standalone_query: str,
    llm_answer: str,
    answer_summary: str
):
    """Save a resolved turn — used as LLM context on future turns."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO standalone_messages
           (session_id, mode, user_query, standalone_query, llm_answer, answer_summary, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id, mode, user_query, standalone_query,
            llm_answer, answer_summary,
            datetime.now(timezone.utc).isoformat()
        )
    )
    conn.commit()
    conn.close()


def get_standalone_context(session_id: str, mode: str, limit: int = 5) -> list:
    """
    Fetch last N standalone turns oldest-first.
    Returns standalone_query + answer_summary only — compact for LLM context.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT standalone_query, answer_summary
           FROM standalone_messages
           WHERE session_id = ? AND mode = ?
           ORDER BY id DESC
           LIMIT ?""",
        (session_id, mode, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"standalone_query": r[0], "answer_summary": r[1]}
        for r in reversed(rows)
    ]