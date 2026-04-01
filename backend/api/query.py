from fastapi import APIRouter, HTTPException, Depends, status
from backend.modules.rag.rag_pipeline import RAGPipeline
from backend.modules.nl2sql import nl2sql_pipeline, summarize_conversation as nl2sql_summarize
from backend.modules.chat_memory import (
    create_session, get_user_sessions, get_session_messages,
    get_session_file_paths, save_message,
    update_session_title, update_session_summary
)
from backend.core.security import get_current_user
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
from datetime import datetime, timezone
import json

router = APIRouter(prefix="/query", tags=["query"])


# ========================
# SCHEMAS
# ========================
class ChatRequest(BaseModel):
    query:      str
    session_id: Optional[str]       = None
    mode:       str                  = "rag"
    # file_paths required only when creating a NEW session (session_id is None)
    file_paths: Optional[List[str]] = None


class SummarizeRequest(BaseModel):
    session_id: str
    mode:       str = "rag"


# ========================
# DATABASE — query log
# ========================
def get_db():
    return sqlite3.connect("app.db")


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id        INTEGER PRIMARY KEY,
        user_id   INTEGER,
        query     TEXT,
        response  TEXT,
        mode      TEXT,
        timestamp TEXT
    )
    """)
    existing = [r[1] for r in cursor.execute("PRAGMA table_info(queries)").fetchall()]
    if "mode" not in existing:
        cursor.execute("ALTER TABLE queries ADD COLUMN mode TEXT")
    conn.commit()
    conn.close()


init_db()


def log_query(user_id: int, query: str, response: dict, mode: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO queries (user_id, query, response, mode, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, query, json.dumps(response), mode, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


# ========================
# CHAT ENDPOINT
# ========================
@router.post("/chat")
def chat(req: ChatRequest, user_id: int = Depends(get_current_user)):
    """
    Multi-turn chat endpoint for RAG and NL2SQL.

    Session lifecycle:
    - New session (session_id=None): file_paths MUST be provided.
      Session is created with those file_paths locked permanently.
    - Existing session (session_id provided): file_paths are read from DB.
      Any file_paths in the request body are ignored.

    Each pipeline fetches its own standalone context internally via session_id.
    No history is passed explicitly.
    """
    try:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if req.mode not in ("rag", "sql"):
            raise HTTPException(status_code=400, detail="mode must be 'rag' or 'sql'")

        # ========================
        # SESSION — create or load
        # ========================
        session_id = req.session_id

        if not session_id:
            # New session — file_paths required
            if not req.file_paths:
                raise HTTPException(
                    status_code=400,
                    detail="file_paths is required when starting a new session."
                )
            session_id = create_session(
                user_id=user_id,
                mode=req.mode,
                file_paths=req.file_paths,
                title="New Chat"
            )
            locked_file_paths = req.file_paths
        else:
            # Existing session — read locked file_paths from DB
            locked_file_paths = get_session_file_paths(session_id)
            if not locked_file_paths:
                raise HTTPException(
                    status_code=400,
                    detail="Session has no files associated. Please start a new session."
                )

        # Check if first message (for title)
        full_history = get_session_messages(session_id)

        # ========================
        # EXECUTION
        # ========================
        if req.mode == "rag":
            pipeline = RAGPipeline(user_id)
            result   = pipeline.query(
                user_query=req.query,
                session_id=session_id,
                file_paths=locked_file_paths,
                mode=req.mode
            )
            assistant_content = result["answer"]
            response_payload  = {
                "answer":           result["answer"],
                "sources":          result["sources"],
                "session_id":       session_id,
                "standalone_query": result.get("retrieval_query", req.query)
            }

        else:  # sql
            result = nl2sql_pipeline(
                user_query=req.query,
                user_id=user_id,
                file_paths=locked_file_paths,
                session_id=session_id,
            )
            if result.get("error"):
                raise HTTPException(status_code=500, detail=result["error"])

            assistant_content = result["natural_answer"]
            response_payload  = {**result, "session_id": session_id}

        # ========================
        # SAVE MESSAGES (UI display)
        # ========================
        save_message(session_id, "user", req.query)
        save_message(session_id, "assistant", assistant_content)

        if not full_history:
            update_session_title(session_id, req.query[:60])

        log_query(user_id=user_id, query=req.query, response=response_payload, mode=req.mode)

        return {"mode": req.mode, "response": response_payload}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# ========================
# SUMMARIZE
# ========================
@router.post("/summarize")
def summarize(req: SummarizeRequest, user_id: int = Depends(get_current_user)):
    try:
        if req.mode == "sql":
            summary = nl2sql_summarize(req.session_id)
        else:
            pipeline = RAGPipeline(user_id)
            summary  = pipeline.summarize_conversation(req.session_id, req.mode)

        update_session_summary(req.session_id, summary)
        return {"summary": summary, "session_id": req.session_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


# ========================
# GET ALL SESSIONS (sidebar)
# ========================
@router.get("/sessions")
def get_sessions(mode: str = "rag", user_id: int = Depends(get_current_user)):
    sessions = get_user_sessions(user_id, mode)
    return {"sessions": sessions}


# ========================
# GET SESSION HISTORY (load past chat)
# ========================
@router.get("/sessions/{session_id}")
def get_history(session_id: str, user_id: int = Depends(get_current_user)):
    messages = get_session_messages(session_id)
    return {"messages": messages, "session_id": session_id}