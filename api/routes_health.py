"""Health check routes."""

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from typing import Optional
import logging

from config.settings import settings
from api.dependencies import get_rag_pipeline
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model: str
    embedding_model: str
    documents: int = 0
    chunks: int = 0
    version: str


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint"
)
async def health_check(
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> HealthResponse:
    """
    Check the health status of the application.
    
    Args:
        rag_pipeline: RAG pipeline instance
        
    Returns:
        HealthResponse: System health information
    """
    try:
        stats = rag_pipeline.get_document_stats()
        return HealthResponse(
            status="healthy",
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            documents=stats["total_documents"],
            chunks=stats["total_chunks"],
            version=settings.app_version
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="degraded",
            model=settings.llm_model,
            embedding_model=settings.embedding_model,
            documents=0,
            chunks=0,
            version=settings.app_version
        )