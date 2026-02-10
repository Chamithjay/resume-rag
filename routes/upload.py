from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from models import ResumeUploadResponse
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from config import get_settings
import aiofiles
import os
from datetime import datetime
import traceback

router = APIRouter(prefix="/upload", tags=["Upload"])


async def get_document_processor():
    settings = get_settings()
    return DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )


async def get_vector_store():
    settings = get_settings()
    return VectorStoreService(
        persist_directory=settings.chroma_dir,
        api_key=settings.google_api_key
    )


@router.post("/resume", response_model=ResumeUploadResponse)
async def upload_resume(
        file: UploadFile = File(...),
        doc_processor: DocumentProcessor = Depends(get_document_processor),
        vector_store: VectorStoreService = Depends(get_vector_store)
):
    """Upload and process a resume"""

    print(f"📤 Received file: {file.filename}")  # Debug log

    # Validate file type
    if not file.filename.endswith(('.pdf', '.docx', '.doc')):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported"
        )

    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Save file
    file_path = os.path.join(settings.upload_dir, file.filename)

    try:
        # Read and save file
        print(f"💾 Saving file to: {file_path}")
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        print(f"📄 Extracting text from: {file.filename}")
        # Extract text
        text = doc_processor.extract_text(file_path)
        candidate_name = doc_processor.extract_candidate_name(text)

        print(f"✂️ Creating chunks for: {candidate_name}")
        # Create chunks
        metadata = {
            "filename": file.filename,
            "candidate_name": candidate_name,
            "upload_date": datetime.now().isoformat()
        }

        chunks = doc_processor.chunk_text(text, metadata)

        print(f"💾 Storing {len(chunks)} chunks in vector database...")
        # Store in vector DB
        chunk_count = vector_store.add_documents(chunks)

        print(f"✅ Successfully processed: {file.filename}")

        return ResumeUploadResponse(
            success=True,
            message="Resume processed successfully",
            filename=file.filename,
            candidate_name=candidate_name,
            chunks_created=chunk_count
        )

    except Exception as e:
        # Print full error for debugging
        print(f"❌ Error processing resume: {str(e)}")
        print(traceback.format_exc())

        # Cleanup on error
        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}"
        )


@router.get("/test")
async def test_upload():
    """Test endpoint to verify upload route is accessible"""
    return {"message": "Upload route is working!"}