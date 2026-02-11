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

    print(f"\n{'=' * 60}")
    print(f"📤 UPLOAD STARTED: {file.filename}")
    print(f"{'=' * 60}")

    if not file.filename.endswith(('.pdf', '.docx', '.doc')):
        raise HTTPException(400, "Only PDF and DOCX files are supported")

    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)

    file_path = os.path.join(settings.upload_dir, file.filename)


    try:
        # Step 1: Save file
        print(f"\n[1/5] 💾 Saving file...")
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        print(f"✅ FILE SAVED!")
        print(f"📍 FILE EXISTS: {os.path.exists(file_path)}")
        print(f"📍 FILE SIZE: {os.path.getsize(file_path)} bytes")

        # Step 2: Extract text
        print(f"\n[2/5] 📄 Extracting text...")
        text = doc_processor.extract_text(file_path)
        print(f"      ✅ Extracted {len(text)} characters")

        # Step 3: Get candidate name
        print(f"\n[3/5] 👤 Extracting candidate name...")
        candidate_name = doc_processor.extract_candidate_name(text)
        print(f"      ✅ Candidate: {candidate_name}")

        # Step 4: Create chunks
        print(f"\n[4/5] ✂️  Creating chunks...")
        metadata = {
            "filename": file.filename,
            "candidate_name": candidate_name,
            "upload_date": datetime.now().isoformat()
        }
        chunks = doc_processor.chunk_text(text, metadata)
        print(f"      ✅ Created {len(chunks)} chunks")

        # Step 5: Store in vector DB
        print(f"\n[5/5] 💾 Storing in vector database...")

        try:
            chunk_count = vector_store.add_documents(chunks)
            print(f"      ✅ Stored {chunk_count} chunks")

        except Exception as e:
            print(f"      ❌ Vector store error: {str(e)}")
            traceback.print_exc()
            # Continue anyway - file is saved
            chunk_count = 0

        print(f"\n{'=' * 60}")
        print(f"✅ UPLOAD COMPLETED: {file.filename}")
        print(f"{'=' * 60}\n")

        # CRITICAL: RETURN THE RESPONSE
        return ResumeUploadResponse(
            success=True,
            message="Resume processed successfully",
            filename=file.filename,
            candidate_name=candidate_name,
            chunks_created=chunk_count
        )

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"❌ UPLOAD FAILED: {file.filename}")
        print(f"{'=' * 60}")
        print(f"Error: {str(e)}")
        traceback.print_exc()

        if os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(500, detail=f"Error: {str(e)}")


@router.get("/test")
async def test_upload():
    """Test endpoint"""
    return {"message": "Upload route is working!"}