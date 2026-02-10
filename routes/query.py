from fastapi import APIRouter, HTTPException, Depends
from models import QueryRequest, QueryResponse, SourceDocument
from services.rag_service import RAGService
from services.vector_store import VectorStoreService
from config import get_settings

router = APIRouter(prefix="/query", tags=["Query"])


async def get_rag_service():
    settings = get_settings()
    vector_store = VectorStoreService(
        persist_directory=settings.chroma_dir,
        api_key=settings.google_api_key  # CHANGED: Pass API key
    )

    return RAGService(
        vector_store_service=vector_store,
        llm_model=settings.llm_model,
        api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
        max_tokens=settings.gemini_max_tokens
    )


@router.post("/ask", response_model=QueryResponse)
async def query_resumes(
        request: QueryRequest,
        rag_service: RAGService = Depends(get_rag_service)
):
    """Query the resume database using Gemini"""

    try:
        result = rag_service.query(
            question=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata
        )

        sources = [
            SourceDocument(**source)
            for source in result["sources"]
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")