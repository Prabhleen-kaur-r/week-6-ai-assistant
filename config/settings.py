"""Application configuration management."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google Gemini
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    llm_model: str = Field("gemini-1.5-flash", env="LLM_MODEL")
    embedding_model: str = Field("embedding-001", env="EMBEDDING_MODEL")
    
    # Security
    api_key: str = Field(..., env="API_KEY")
    
    # RAG Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE", ge=100, le=2000)
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP", ge=0, le=500)
    retrieval_top_k: int = Field(8, env="RETRIEVAL_TOP_K", ge=1, le=20)
    final_top_k: int = Field(5, env="FINAL_TOP_K", ge=1, le=10)
    semantic_weight: float = Field(0.6, env="SEMANTIC_WEIGHT", ge=0.0, le=1.0)
    keyword_weight: float = Field(0.4, env="KEYWORD_WEIGHT", ge=0.0, le=1.0)
    
    # Memory
    max_memory_messages: int = Field(10, env="MAX_MEMORY_MESSAGES", ge=1, le=50)
    
    # File Upload
    max_upload_size_mb: int = Field(20, env="MAX_UPLOAD_SIZE_MB", ge=1, le=100)
    
    # Cost Optimization
    max_output_tokens: int = Field(1024, env="MAX_OUTPUT_TOKENS", ge=100, le=8192)
    temperature: float = Field(0.2, env="TEMPERATURE", ge=0.0, le=1.0)
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Application
    app_name: str = "AI Knowledge Base Assistant"
    app_version: str = "1.0.0"
    
    # Paths
    data_dir: str = Field("data", env="DATA_DIR")
    uploads_dir: str = Field("data/uploads", env="UPLOADS_DIR")
    processed_dir: str = Field("data/processed", env="PROCESSED_DIR")
    
    # Backend URL
    backend_url: str = "http://localhost:8000"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for directory in [self.data_dir, self.uploads_dir, self.processed_dir]:
            os.makedirs(directory, exist_ok=True)


settings = Settings()