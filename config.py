from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # API Keys
    google_api_key: str

    # Application
    app_name: str = "Resume RAG Analyzer API"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths - Now from environment
    upload_dir: str
    chroma_dir: str

    # RAG Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 5

    # Gemini Settings
    llm_model: str = "gemini-1.5-flash"
    gemini_temperature: float = 0.0
    gemini_max_tokens: int = 2048

    class Config:
        env_file = ".env"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)


@lru_cache()
def get_settings():
    return Settings()