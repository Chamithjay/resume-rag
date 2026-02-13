from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class MatchingExcerpt(BaseModel):
    text: str
    score: float


class Candidate(BaseModel):
    name: str
    filename: str
    relevance_score: float
    matching_excerpts: List[MatchingExcerpt]


class QueryResponse(BaseModel):
    query: str
    answer: str
    candidates: List[Candidate]
    total_candidates: int
