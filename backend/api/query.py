from fastapi import APIRouter, HTTPException, Depends, status
from backend.state.session_manager import get_active_file
from backend.router.dispatcher import route_query
from backend.modules.rag import rag_pipeline
from backend.modules.nl2sql import nl2sql_pipeline
from backend.models.schemas import QueryRequest
from backend.core.security import get_current_user
import sqlite3
from datetime import datetime

router = APIRouter(prefix="/query", tags=["query"])


# ========================
# DATABASE CONNECTION
# ========================
def get_db():
    return sqlite3.connect("app.db")


def init_db():
    """
    Initialize queries table
    """
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


# Initialize once
init_db()


def log_query(user_id: int, query: str, response: str, route: str):
    """
    Store query and response in database
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO queries (user_id, query, response, route, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, query, response, route, datetime.utcnow().isoformat())
    )

    conn.commit()
    conn.close()


# ========================
# MAIN QUERY ENDPOINT
# ========================
@router.post("/")
def query(req: QueryRequest, user_id: int = Depends(get_current_user)):
    """
    Process user query:
    1. Get active file
    2. Route query (dispatcher)
    3. Execute module
    4. Log result
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
                detail="No active file uploaded"
            )

        file_type = state.get("file_type")

        # ========================
        # ROUTING
        # ========================
        route = route_query(file_type, req.query)

        # ========================
        # EXECUTION
        # ========================
        if route == "rag":
            result = rag_pipeline(req.query)

        elif route == "nl2sql":
            result = nl2sql_pipeline(req.query)

        elif route == "rag_with_warning":
            result = {
                "warning": "Data is unstructured. Numerical results may be approximate.",
                "answer": rag_pipeline(req.query)
            }

        elif route == "hybrid":
            # Future-ready (simple version)
            result = {
                "rag": rag_pipeline(req.query),
                "sql": nl2sql_pipeline(req.query)
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
            response=str(result),
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