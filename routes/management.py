from fastapi import APIRouter, HTTPException, Depends
from models import ResumeInfo, HealthResponse
from services.vector_store import VectorStoreService
from config import get_settings
from typing import List
import os

router = APIRouter(prefix="/management", tags=["Management"])


async def get_vector_store():
    settings = get_settings()
    return VectorStoreService(
        persist_directory=settings.chroma_dir,
        embedding_model=settings.embedding_model
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(
        vector_store: VectorStoreService = Depends(get_vector_store)
):
    """Health check endpoint"""

    settings = get_settings()

    # Count resumes
    upload_dir = settings.upload_dir
    total_resumes = len([f for f in os.listdir(upload_dir) if f.endswith(('.pdf', '.docx'))]) if os.path.exists(
        upload_dir) else 0

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        total_resumes=total_resumes
    )


@router.get("/resumes", response_model=List[str])
async def list_resumes():
    """List all uploaded resumes"""

    settings = get_settings()
    upload_dir = settings.upload_dir

    if not os.path.exists(upload_dir):
        return []

    files = [f for f in os.listdir(upload_dir) if f.endswith(('.pdf', '.docx'))]
    return files


@router.delete("/resume/{filename}")
async def delete_resume(
        filename: str,
        vector_store: VectorStoreService = Depends(get_vector_store)
):
    """Delete a resume and its embeddings"""

    settings = get_settings()
    file_path = os.path.join(settings.upload_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(404, "Resume not found")

    try:
        # Delete from vector store
        vector_store.delete_by_filename(filename)

        # Delete file
        os.remove(file_path)

        return {"message": f"Resume {filename} deleted successfully"}

    except Exception as e:
        raise HTTPException(500, f"Error deleting resume: {str(e)}")