from fastapi import APIRouter, UploadFile, Depends, HTTPException, status, File, BackgroundTasks
from typing import List
from backend.modules.file_handler import save_files
from backend.state.session_manager import set_active_file, update_active_file_status, get_active_file
from backend.core.security import get_current_user
from backend.modules.rag.rag_pipeline import RAGPipeline
from backend.modules.rag.rag_loader import prepare_documents
import os

router = APIRouter(prefix="/upload", tags=["upload"])

# ========================
# PIPELINE-AWARE EXTENSION SETS
# ========================
RAG_EXTENSIONS  = {"pdf", "doc", "docx", "txt"}
SQL_EXTENSIONS  = {"csv", "xlsx", "xls", "db", "sql"}
ALL_EXTENSIONS  = RAG_EXTENSIONS | SQL_EXTENSIONS

def get_pipeline(suffix: str) -> str:
    """Return 'rag' or 'sql' based on file extension."""
    if suffix in RAG_EXTENSIONS:
        return "rag"
    return "sql"


# ========================
# BACKGROUND INGESTION (RAG only)
# SQL files are loaded at query time — no embedding needed.
# ========================
def run_ingestion(user_id: int, paths: List[str]):
    """
    Embed RAG documents into FAISS for the user.
    Only called for RAG-type files.
    """
    try:
        pipeline = RAGPipeline(user_id)
        pipeline.embedder.load_or_create()

        for path in paths:
            docs = pipeline.loader.load(path)
            prepared = prepare_documents(docs)
            pipeline.embedder.add_documents(prepared)

        update_active_file_status(user_id, "ready")

    except Exception as e:
        update_active_file_status(user_id, f"failed: {str(e)}")


# ========================
# UPLOAD ENDPOINT
# ========================
@router.post("/", status_code=status.HTTP_202_ACCEPTED)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: int = Depends(get_current_user)
):
    """
    Upload one or more files.

    Accepted types:
    - RAG  : pdf, doc, docx, txt
    - NL2SQL: csv, xlsx, xls, db, sql

    Mixing RAG and SQL files in a single upload is not allowed —
    each upload should be one pipeline type.
    """
    try:
        # ========================
        # VALIDATE EXTENSIONS
        # ========================
        validated_files = []
        file_types      = []
        pipeline_types  = set()

        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="One or more files has no filename."
                )

            suffix = file.filename.rsplit(".", 1)[-1].lower()

            if suffix not in ALL_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Unsupported file type: .{suffix}. "
                        f"Accepted for documents: {', '.join(sorted(RAG_EXTENSIONS))}. "
                        f"Accepted for data: {', '.join(sorted(SQL_EXTENSIONS))}."
                    )
                )

            pipeline_types.add(get_pipeline(suffix))
            validated_files.append(file)
            file_types.append(suffix)

        # ========================
        # BLOCK MIXED UPLOADS
        # RAG + SQL files in one batch makes routing ambiguous
        # ========================
        if len(pipeline_types) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Mixed upload not allowed. "
                    "Please upload document files (pdf, doc, docx, txt) "
                    "and data files (csv, xlsx, xls, db, sql) separately."
                )
            )

        primary_pipeline = pipeline_types.pop()  # "rag" or "sql"

        # ========================
        # READ FILE BYTES (async)
        # ========================
        file_contents = []
        for file in validated_files:
            data = await file.read()
            if not data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{file.filename}' is empty."
                )
            file_contents.append((file.filename, data))

        # ========================
        # SAVE FILES TO DISK
        # ========================
        file_paths = save_files(file_contents, user_id)

        # ========================
        # UPDATE SESSION STATE
        # ========================
        set_active_file(user_id, {
            "files":    file_paths,
            "types":    file_types,
            "pipeline": primary_pipeline,   # "rag" or "sql"
            "status":   "processing" if primary_pipeline == "rag" else "ready"
            # SQL files need no background processing — they're ready immediately
        })

        # ========================
        # BACKGROUND INGESTION (RAG only)
        # ========================
        if primary_pipeline == "rag":
            background_tasks.add_task(run_ingestion, user_id, file_paths)
            message = "Upload successful. Embedding documents in background."
        else:
            message = "Upload successful. Data file is ready to query."

        return {
            "message":  message,
            "pipeline": primary_pipeline,
            "count":    len(file_paths)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


# ========================
# STATUS ENDPOINT
# ========================
@router.get("/status")
def get_status(user_id: int = Depends(get_current_user)):
    """
    Returns current ingestion status.
    SQL files are always 'ready' immediately after upload.
    RAG files are 'processing' until embedding completes.
    """
    state = get_active_file(user_id)

    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No files uploaded yet"
        )

    return {
        "status":   state.get("status", "processing"),
        "pipeline": state.get("pipeline", "rag")
    }