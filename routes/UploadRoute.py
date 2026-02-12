import os
from fastapi import UploadFile, APIRouter, Depends
from services.FileService import FileService
from services.PdfService import PDFService
from services.ChunkService import ChunkService
from services.EmbeddingSerivce import EmbeddingService
from services.VectorStoreService import VectorStoreService
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="", tags=["Upload"])

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")


def get_file_service():
    return FileService()


def get_pdf_service():
    return PDFService()


def get_chunks_service():
    return ChunkService()


def get_embeddings_service():
    return EmbeddingService(api_key=api_key)


def get_vector_store_service():
    return VectorStoreService()


@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile, file_service: FileService = Depends(get_file_service),
                             pdf_service: PDFService = Depends(get_pdf_service),
                             chunk_service: ChunkService = Depends(get_chunks_service),
                             embed_service: EmbeddingService = Depends(get_embeddings_service),
                             vector_store: VectorStoreService = Depends(get_vector_store_service)):
    file_path = await file_service.save_file(file)

    text = pdf_service.extract_text(file_path)

    chunks = chunk_service.split_text(text)
    embeddings_list = embed_service.embed_chunks(chunks)

    # Add to Pinecone
    vector_store.store_embeddings( embeddings_list, chunks, file.filename)
    count = vector_store.count()
    print(f"Total vectors in Pinecone: {count}")

    return {
        "message": "File processed",
        "filename": file.filename,
        "char_count": len(text),
        "num_chunks": len(chunks),
        "embedding_dim": len(embeddings_list[0]) if embeddings_list else 0
    }
