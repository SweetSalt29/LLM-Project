from fastapi import APIRouter, UploadFile, Depends, HTTPException, status, File, BackgroundTasks
from typing import List
from backend.modules.file_handler import save_files
from backend.state.session_manager import set_active_file, update_active_file_status, get_active_file
from backend.core.security import get_current_user
from backend.modules.rag.rag_pipeline import RAGPipeline
from backend.modules.rag.rag_loader import prepare_documents
import os

router = APIRouter(prefix="/upload", tags=["upload"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "csv", "xlsx", "msg", "chm"}


# ========================
# BACKGROUND INGESTION
# ========================
def run_ingestion(user_id: int, paths: List[str]):
    """
    Load, prepare and embed documents into FAISS for the user.
    Runs in background to avoid frontend timeouts.
    """
    try:
        pipeline = RAGPipeline(user_id)
        pipeline.embedder.load_or_create()

        for path in paths:
            docs = pipeline.loader.load(path)
            prepared = prepare_documents(docs)
            pipeline.embedder.add_documents(prepared)

        # Mark ingestion as done
        update_active_file_status(user_id, "ready")

    except Exception as e:
        # Mark ingestion as failed so query endpoint can warn user
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
    Upload one or more files. Validates extensions, saves to disk,
    updates session state, then kicks off background RAG ingestion.
    """
    try:
        # ========================
        # VALIDATE EXTENSIONS
        # ========================
        validated_files = []
        file_types = []

        for file in files:
            suffix = file.filename.split(".")[-1].lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: .{suffix}"
                )
            validated_files.append(file)
            file_types.append(suffix)

        # ========================
        # READ FILE BYTES (async)
        # Must await — UploadFile.read() is a coroutine.
        # Skipping await returns empty bytes → 0-byte saved files.
        # ========================
        file_contents = []
        for file in validated_files:
            data = await file.read()
            file_contents.append((file.filename, data))

        # ========================
        # SAVE FILES TO DISK
        # ========================
        file_paths = save_files(file_contents, user_id)

        # ========================
        # UPDATE SESSION STATE
        # ========================
        set_active_file(user_id, {
            "files": file_paths,
            "types": file_types,
            "status": "processing"
        })

        # ========================
        # BACKGROUND INGESTION
        # ========================
        background_tasks.add_task(run_ingestion, user_id, file_paths)

        return {
            "message": "Upload successful. Processing documents in background.",
            "count": len(file_paths)
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
# Polled by frontend to know when ingestion is complete
# ========================
@router.get("/status")
def get_status(user_id: int = Depends(get_current_user)):
    """
    Returns current ingestion status for the user's uploaded files.
    Possible values: 'processing' | 'ready' | 'failed: <reason>'
    """
    state = get_active_file(user_id)

    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No files uploaded yet"
        )

    return {"status": state.get("status", "processing")}