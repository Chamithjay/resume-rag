from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ResumeUploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    candidate_name: Optional[str] = None
    chunks_created: int

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    filter_metadata: Optional[dict] = None

class SourceDocument(BaseModel):
    content: str
    candidate_name: str
    filename: str
    page: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    confidence: Optional[float] = None

class ResumeInfo(BaseModel):
    filename: str
    candidate_name: Optional[str]
    upload_date: datetime
    chunk_count: int

class HealthResponse(BaseModel):
    status: str
    version: str
    total_resumes: int