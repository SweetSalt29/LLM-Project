from fastapi import APIRouter, HTTPException, Depends, status
from backend.state.session_manager import get_active_file
from backend.router.dispatcher import route_query
from backend.modules.rag.rag_pipeline import RAGPipeline
from backend.modules.nl2sql import nl2sql_pipeline, summarize_conversation as nl2sql_summarize
from backend.modules.chat_memory import (
    create_session, get_user_sessions, get_session_messages,
    save_message, update_session_title, update_session_summary
)
from backend.models.schemas import QueryRequest
from backend.core.security import get_current_user
from pydantic import BaseModel
from typing import Optional
import sqlite3
from datetime import datetime
import uuid
import json

router = APIRouter(prefix="/query", tags=["query"])


# ========================
# SCHEMAS
# ========================
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None   # None = start new session
    mode: str = "rag"                  # "rag" | "sql"


class SummarizeRequest(BaseModel):
    session_id: str
    mode: str = "rag"


# ========================
# DATABASE — legacy query log
# ========================
def get_db():
    return sqlite3.connect("app.db")


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        query TEXT,
        response TEXT,
        route TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()


init_db()


def log_query(user_id: int, query: str, response: dict, route: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO queries (user_id, query, response, route, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, query, json.dumps(response), route, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


# ========================
# CHAT ENDPOINT
# Replaces the old single-turn /query/ endpoint
# ========================
@router.post("/chat")
def chat(req: ChatRequest, user_id: int = Depends(get_current_user)):
    """
    Multi-turn chat endpoint for both RAG and NL2SQL modes.
    Creates a new session if session_id is not provided.
    Loads history, runs pipeline, saves messages, returns response.
    """
    try:
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # ========================
        # GET USER FILE STATE
        # ========================
        state = get_active_file(user_id)
        if not state:
            raise HTTPException(status_code=400, detail="No files uploaded")

        ingestion_status = state.get("status", "processing")
        if ingestion_status == "processing":
            raise HTTPException(
                status_code=202,
                detail="Documents are still being processed. Please wait."
            )
        if ingestion_status.startswith("failed"):
            raise HTTPException(
                status_code=500,
                detail=f"Document ingestion failed: {ingestion_status}. Please re-upload."
            )

        # ========================
        # SESSION — create or load
        # ========================
        session_id = req.session_id
        if not session_id:
            session_id = create_session(
                user_id=user_id,
                mode=req.mode,
                title="New Chat"
            )

        # Load full conversation history from DB
        history = get_session_messages(session_id)

        # ========================
        # ROUTING
        # ========================
        file_types = state.get("types", [])
        if any(ft in ["csv", "xlsx", "db"] for ft in file_types):
            primary_type = "csv"
        else:
            primary_type = "pdf"

        route = route_query(primary_type, req.query)

        # ========================
        # EXECUTION
        # ========================
        if route in ["rag", "rag_with_warning"]:
            pipeline = RAGPipeline(user_id)
            result = pipeline.query(req.query, history=history)
            assistant_content = result["answer"]

            response_payload = {
                "answer": result["answer"],
                "sources": result["sources"],
                "session_id": session_id
            }
            if route == "rag_with_warning":
                response_payload["warning"] = (
                    "Data is unstructured. Numerical results may be approximate."
                )

        elif route == "nl2sql":
            file_paths = state.get("files", [])
            result = nl2sql_pipeline(
                user_query=req.query,
                user_id=user_id,
                file_paths=file_paths,
                session_id=session_id,
                history=history
            )
            if result.get("error"):
                raise HTTPException(status_code=500, detail=result["error"])

            assistant_content = result["natural_answer"]
            response_payload = {**result, "session_id": session_id}

        elif route == "hybrid":
            pipeline = RAGPipeline(user_id)
            file_paths = state.get("files", [])
            rag_result = pipeline.query(req.query, history=history)
            sql_result = nl2sql_pipeline(
                user_query=req.query,
                user_id=user_id,
                file_paths=file_paths,
                session_id=session_id,
                history=history
            )
            assistant_content = rag_result["answer"]
            response_payload = {
                "rag": rag_result,
                "sql": sql_result,
                "session_id": session_id
            }

        else:
            raise HTTPException(status_code=500, detail="Invalid routing decision")

        # ========================
        # SAVE MESSAGES TO DB
        # ========================
        save_message(session_id, "user", req.query)
        save_message(session_id, "assistant", assistant_content)

        # Set session title from first user message
        if not history:
            title = req.query[:60]
            update_session_title(session_id, title)

        # ========================
        # LOG
        # ========================
        log_query(user_id=user_id, query=req.query, response=response_payload, route=route)

        return {"route": route, "response": response_payload}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


# ========================
# SUMMARIZE CONVERSATION
# ========================
@router.post("/summarize")
def summarize(req: SummarizeRequest, user_id: int = Depends(get_current_user)):
    """
    Summarizes the conversation of a given session.
    Stores the summary in DB and returns it.
    """
    try:
        history = get_session_messages(req.session_id)

        if not history:
            raise HTTPException(status_code=400, detail="No messages in this session yet.")

        if req.mode == "sql":
            summary = nl2sql_summarize(history)
        else:
            # Use RAG pipeline summarizer
            pipeline = RAGPipeline(user_id)
            summary = pipeline.summarize_conversation(history)

        # Persist summary
        update_session_summary(req.session_id, summary)

        return {"summary": summary, "session_id": req.session_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


# ========================
# GET ALL SESSIONS (for sidebar)
# ========================
@router.get("/sessions")
def get_sessions(mode: str = "rag", user_id: int = Depends(get_current_user)):
    """
    Returns all chat sessions for the user in a given mode.
    Used to populate the sidebar.
    """
    sessions = get_user_sessions(user_id, mode)
    return {"sessions": sessions}


# ========================
# GET SESSION HISTORY (load a past chat)
# ========================
@router.get("/sessions/{session_id}")
def get_history(session_id: str, user_id: int = Depends(get_current_user)):
    """
    Returns full message history for a session.
    Used when user clicks a past chat in the sidebar.
    """
    messages = get_session_messages(session_id)
    return {"messages": messages, "session_id": session_id}