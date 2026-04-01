from fastapi import APIRouter, UploadFile, Depends, HTTPException, status, File, BackgroundTasks
from typing import List
from backend.modules.file_handler import save_files
from backend.modules.chat_memory import (
    register_file, mark_file_indexed,
    get_user_library, get_pending_files,
    file_exists_in_library
)
from backend.core.security import get_current_user
from backend.modules.rag.rag_pipeline import RAGPipeline
from backend.modules.rag.rag_loader import prepare_documents

router = APIRouter(prefix="/upload", tags=["upload"])

# ========================
# EXTENSION SETS
# ========================
RAG_EXTENSIONS = {"pdf", "doc", "docx", "txt", "msg", "chm"}
SQL_EXTENSIONS = {"csv", "xlsx", "xls", "db", "sql"}
ALL_EXTENSIONS = RAG_EXTENSIONS | SQL_EXTENSIONS


def get_pipeline(suffix: str) -> str:
    return "rag" if suffix in RAG_EXTENSIONS else "sql"


# ========================
# BACKGROUND INGESTION (RAG only)
# Embeds one file at a time and marks it indexed on completion.
# ========================
def run_ingestion(user_id: int, file_path: str):
    """
    Embed a single RAG file into FAISS and mark it indexed in file_library.
    Called as a background task per file so partial failures don't block others.
    """
    try:
        pipeline = RAGPipeline(user_id)
        pipeline.embedder.load_or_create()

        docs     = pipeline.loader.load(file_path)
        prepared = prepare_documents(docs)
        pipeline.embedder.add_documents(prepared)

        mark_file_indexed(file_path, user_id)

    except Exception as e:
        # Log failure — file stays indexed=0 in library (visible as pending)
        print(f"[run_ingestion] Failed for {file_path}: {e}")


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
    Upload one or more files to the user's file library.

    - RAG files (pdf, doc, docx, txt, msg, chm): embedded into FAISS in background.
    - SQL files (csv, xlsx, xls, db, sql): registered immediately, ready to query.
    - Mixed RAG+SQL uploads in one batch are blocked.

    Each file is registered individually in file_library.
    Sessions choose which files to query at session creation time.
    """
    try:
        # ── Validate extensions ────────────────────────────────────
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

        # ── Block mixed RAG+SQL uploads ────────────────────────────
        if len(pipeline_types) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Mixed upload not allowed. "
                    "Please upload document files (pdf, doc, docx, txt, msg, chm) "
                    "and data files (csv, xlsx, xls, db, sql) separately."
                )
            )

        primary_pipeline = pipeline_types.pop()

        # ── Read file bytes ────────────────────────────────────────
        file_contents = []
        for file in validated_files:
            data = await file.read()
            if not data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{file.filename}' is empty."
                )
            file_contents.append((file.filename, data))

        # ── Save to disk ───────────────────────────────────────────
        file_paths = save_files(file_contents, user_id)

        # ── Register each file in library + schedule ingestion ─────
        registered = []
        skipped    = []
        for file_path, (file_name, _) in zip(file_paths, file_contents):

            # Duplicate guard — same file_path already in library
            if file_exists_in_library(user_id, file_path):
                skipped.append(file_name)
                continue

            file_id = register_file(
                user_id=user_id,
                file_name=file_name,
                file_path=file_path,
                pipeline=primary_pipeline
            )
            registered.append({"file_id": file_id, "file_name": file_name, "file_path": file_path})

            if primary_pipeline == "rag":
                background_tasks.add_task(run_ingestion, user_id, file_path)

        if not registered and skipped:
            return {
                "message":  f"All {len(skipped)} file(s) already exist in your library. Nothing uploaded.",
                "pipeline": primary_pipeline,
                "files":    [],
                "skipped":  skipped
            }

        if primary_pipeline == "rag":
            message = (
                f"{len(registered)} document(s) uploaded. "
                "Embedding in background — they will appear in your library when ready."
            )
        else:
            message = f"{len(registered)} data file(s) uploaded and ready to query."

        if skipped:
            message += f" ({len(skipped)} duplicate(s) skipped: {', '.join(skipped)})"

        return {
            "message":  message,
            "pipeline": primary_pipeline,
            "files":    registered,
            "skipped":  skipped
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


# ========================
# LIBRARY ENDPOINT
# Returns all ready files for the user filtered by pipeline.
# ========================
@router.get("/library")
def get_library(pipeline: str = "rag", user_id: int = Depends(get_current_user)):
    """
    Returns all indexed (ready) files in the user's library for a given pipeline.
    Used to populate the file selector when starting a new chat.
    """
    files = get_user_library(user_id, pipeline)
    return {"files": files, "pipeline": pipeline}


# ========================
# PENDING ENDPOINT
# Returns files still being embedded (indexed=0).
# ========================
@router.get("/pending")
def get_pending(user_id: int = Depends(get_current_user)):
    """Returns RAG files still being processed."""
    pending = get_pending_files(user_id)
    return {"pending": pending}