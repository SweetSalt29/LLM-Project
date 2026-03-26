from fastapi import APIRouter, UploadFile, Depends, HTTPException, status
from backend.modules.file_handler import save_file
from backend.state.session_manager import set_active_file
from backend.core.security import get_current_user
import os

router = APIRouter(prefix="/upload", tags=["upload"])

# Allowed file types (extend later)
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "csv", "xlsx", "msg", "chm"}


@router.post("/")
def upload_file(file: UploadFile, user_id: int = Depends(get_current_user)):
    """
    Upload a file and set it as active data source for the user.

    Steps:
    1. Validate file
    2. Save file to disk
    3. Update session state
    4. Return metadata
    """
    try:
        # ========================
        # VALIDATION
        # ========================
        if not file or not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )

        suffix = file.filename.split(".")[-1].lower()

        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: .{suffix}"
            )

        # ========================
        # SAVE FILE
        # ========================
        file_path = save_file(file)

        # ========================
        # UPDATE SESSION STATE
        # ========================
        set_active_file(user_id, file.filename, suffix)

        # ========================
        # RESPONSE
        # ========================
        return {
            "message": "File uploaded successfully",
            "file_name": file.filename,
            "file_type": suffix,
            "path": file_path
        }

    except HTTPException:
        # Re-raise known HTTP errors
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )