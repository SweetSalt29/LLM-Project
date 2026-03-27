from fastapi import APIRouter, HTTPException, Depends, status
from backend.state.session_manager import get_active_file
from backend.router.dispatcher import route_query
from backend.modules.rag.rag_pipeline import RAGPipeline
from backend.modules.nl2sql import nl2sql_pipeline
import uuid
from backend.models.schemas import QueryRequest
from backend.core.security import get_current_user
import sqlite3
from datetime import datetime
import json

router = APIRouter(prefix="/query", tags=["query"])


# ========================
# DATABASE
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
    """
    Store query safely as JSON
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO queries (user_id, query, response, route, timestamp) VALUES (?, ?, ?, ?, ?)",
        (
            user_id,
            query,
            json.dumps(response),
            route,
            datetime.utcnow().isoformat()
        )
    )

    conn.commit()
    conn.close()


# ========================
# MAIN QUERY ENDPOINT
# ========================
@router.post("/")
def query(req: QueryRequest, user_id: int = Depends(get_current_user)):
    """
    Process user query with multi-file RAG support
    """

    try:
        # ========================
        # VALIDATE INPUT
        # ========================
        if not req.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )

        # ========================
        # GET USER STATE
        # ========================
        state = get_active_file(user_id)

        if not state:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files uploaded"
            )

        # ========================
        # CHECK INGESTION STATUS
        # Block query if background ingestion isn't done yet
        # ========================
        ingestion_status = state.get("status", "processing")

        if ingestion_status == "processing":
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Documents are still being processed. Please wait a moment and try again."
            )

        if ingestion_status.startswith("failed"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document ingestion failed: {ingestion_status}. Please re-upload your file."
            )

        # ========================
        # FILE TYPE ROUTING
        # ========================
        file_types = state.get("types", [])

        if not file_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid file types found"
            )

        if any(ft in ["csv", "xlsx", "db"] for ft in file_types):
            primary_type = "csv"
        else:
            primary_type = "pdf"

        # ========================
        # ROUTING
        # ========================
        route = route_query(primary_type, req.query)

        # ========================
        # EXECUTION
        # ========================
        if route == "rag":
            pipeline = RAGPipeline(user_id)
            result = pipeline.query(req.query)

        elif route == "nl2sql":
            file_paths = state.get("files", [])
            session_id = str(uuid.uuid4())
            result = nl2sql_pipeline(
                user_query=req.query,
                user_id=user_id,
                file_paths=file_paths,
                session_id=session_id
            )

        elif route == "rag_with_warning":
            pipeline = RAGPipeline(user_id)
            result = {
                "warning": "Data is unstructured. Numerical results may be approximate.",
                "answer": pipeline.query(req.query)
            }

        elif route == "hybrid":
            pipeline = RAGPipeline(user_id)
            file_paths = state.get("files", [])
            session_id = str(uuid.uuid4())
            result = {
                "rag": pipeline.query(req.query),
                "sql": nl2sql_pipeline(
                    user_query=req.query,
                    user_id=user_id,
                    file_paths=file_paths,
                    session_id=session_id
                )
            }

        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid routing decision"
            )

        # ========================
        # LOG QUERY
        # ========================
        log_query(
            user_id=user_id,
            query=req.query,
            response=result,
            route=route
        )

        # ========================
        # RESPONSE
        # ========================
        return {
            "route": route,
            "response": result
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )