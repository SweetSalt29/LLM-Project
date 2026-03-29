"""
chat_memory.py
Handles all persistent storage for chat sessions, messages, and summaries.
"""

import sqlite3
import uuid
from datetime import datetime

DB_PATH = "app.db"


# ============================================================
# INIT TABLES
# ============================================================
def init_chat_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id   TEXT PRIMARY KEY,
            user_id      INTEGER NOT NULL,
            title        TEXT,
            mode         TEXT,
            summary      TEXT,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        )
    """)

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

    conn.commit()
    conn.close()


init_chat_tables()


# ============================================================
# SESSION OPERATIONS
# ============================================================
def create_session(user_id: int, mode: str, title: str = None) -> str:
    """Create a new chat session. Returns session_id."""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO chat_sessions
           (session_id, user_id, title, mode, summary, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session_id, user_id, title or "New Chat", mode, None, now, now)
    )
    conn.commit()
    conn.close()
    return session_id


def get_user_sessions(user_id: int, mode: str) -> list:
    """Get all sessions for a user in a given mode, newest first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT session_id, title, summary, created_at, updated_at
           FROM chat_sessions
           WHERE user_id = ? AND mode = ?
           ORDER BY updated_at DESC""",
        (user_id, mode)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"session_id": r[0], "title": r[1], "summary": r[2],
         "created_at": r[3], "updated_at": r[4]}
        for r in rows
    ]


def update_session_title(session_id: str, title: str):
    """Set session title (called after first user message)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
        (title[:60], datetime.utcnow().isoformat(), session_id)
    )
    conn.commit()
    conn.close()


def update_session_summary(session_id: str, summary: str):
    """Store a generated summary for the session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
        (summary, datetime.utcnow().isoformat(), session_id)
    )
    conn.commit()
    conn.close()


def touch_session(session_id: str):
    """Bump updated_at on every new message."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
        (datetime.utcnow().isoformat(), session_id)
    )
    conn.commit()
    conn.close()


# ============================================================
# MESSAGE OPERATIONS
# ============================================================
def save_message(session_id: str, role: str, content: str):
    """Persist a single message and touch the session timestamp."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO chat_messages (session_id, role, content, timestamp)
           VALUES (?, ?, ?, ?)""",
        (session_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    touch_session(session_id)


def get_session_messages(session_id: str) -> list:
    """Load full message history for a session, oldest first."""
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


def build_history_string(messages: list, max_turns: int = 10) -> str:
    """
    Format the last N conversation turns into a string for LLM context.
    Each turn = 1 user message + 1 assistant message.
    """
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines)