from fastapi import APIRouter, HTTPException, Depends
from models.QueryModel import QueryRequest, QueryResponse
from services.VectorStoreService import VectorStoreService
from services.EmbeddingSerivce import EmbeddingService
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


@router.post("/query", response_model=QueryResponse)
async def query_embeddings(req: QueryRequest, embedding_service: EmbeddingService = Depends(get_embeddings_service)):
    try:
        query_embedding = embedding_service.embed_chunks([req.query])[0]

        response = vector_store.index.query(
            vector=query_embedding,
            top_k=20,
            include_metadata=True
        )

        seen_files = {}

        for match in response["matches"]:
            if match["score"] < MIN_SCORE:
                continue
            filename = match["metadata"].get("filename", "unknown")

            if filename not in seen_files:
                seen_files[filename] = {
                    "score": match["score"],
                    "text": match["metadata"]["text"],
                    "filename": filename,
                    "chunk_index": match["metadata"].get("chunk_index", None)
                }

        results = list(seen_files.values())[:5]

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
