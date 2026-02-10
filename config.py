from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    google_api_key: str

    # Application
    app_name: str = "Resume RAG Analyzer API"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    upload_dir: str = "../data/uploads"
    chroma_dir: str = "../data/chroma_db"

    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 5

    # LLM Settings (Gemini)
    llm_model: str = "gemini-1.5-flash"
    gemini_temperature: float = 0.0
    gemini_max_tokens: int = 2048

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()