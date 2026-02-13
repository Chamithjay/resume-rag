from fastapi import APIRouter, HTTPException, Depends
from models.QueryModel import QueryRequest, QueryResponse
from services.VectorStoreService import VectorStoreService
from services.EmbeddingSerivce import EmbeddingService
from services.RagService import RAGService  # NEW IMPORT
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

MIN_SCORE = 0.5
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")


def get_embeddings_service():
    return EmbeddingService(api_key=api_key)


vector_store = VectorStoreService()
rag_service = RAGService()  # NEW: Initialize RAG service


@router.post("/query", response_model=QueryResponse)
async def query_embeddings(
        req: QueryRequest,
        embedding_service: EmbeddingService = Depends(get_embeddings_service)
):
    try:

        query_embedding = embedding_service.embed_chunks([req.query])[0]


        response = vector_store.index.query(
            vector=query_embedding,
            top_k=20,
            include_metadata=True
        )

        seen_candidates = {}

        for match in response["matches"]:
            if match["score"] < MIN_SCORE:
                continue

            candidate_name = match["metadata"].get("candidate_name", "Unknown")

            if candidate_name not in seen_candidates:
                seen_candidates[candidate_name] = {
                    "candidate_name": candidate_name,
                    "filename": match["metadata"].get("filename", "unknown"),
                    "text": match["metadata"].get("text", ""),
                    "score": match["score"],
                    "chunk_index": match["metadata"].get("chunk_index", None)
                }

        top_candidates = list(seen_candidates.values())[:5]

        if top_candidates:
            llm_response = rag_service.generate_structured_response(
                query=req.query,
                matched_chunks=top_candidates
            )

            return {
                "query": req.query,
                "answer": llm_response["answer"],
                "candidates": llm_response["candidates"],
                "total_candidates": len(llm_response["candidates"])
            }
        else:
            # No matches found
            return {
                "query": req.query,
                "answer": "No candidates found matching your criteria. Try adjusting your query or upload more resumes.",
                "candidates": [],
                "total_candidates": 0
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))