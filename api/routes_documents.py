"""Document management routes."""

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from typing import List, Dict, Any
import logging

from api.dependencies import verify_api_key, get_rag_pipeline
from rag.pipeline import RAGPipeline
from utils.error_handler import DocumentProcessingError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> Dict[str, Any]:
    """Upload and process a document."""
    try:
        content = await file.read()
        result = rag_pipeline.process_document(content, file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "message": "Document processed successfully",
            "chunks": result["chunks"],
            "embeddings": result["embeddings"]
        }
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": e.message, "details": e.details}
        )
    except Exception as e:
        error_msg = str(e)
        # Clean up error message for user
        if "404" in error_msg and "embedding" in error_msg.lower():
            user_msg = "Embedding model not found. Please check your EMBEDDING_MODEL setting."
        elif "429" in error_msg:
            user_msg = "API quota exceeded. Please wait and try again."
        else:
            user_msg = error_msg[:200]
        
        logger.error(f"Unexpected error during upload: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": user_msg}
        )


@router.get("/")
async def list_documents(
    api_key: str = Depends(verify_api_key),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> List[str]:
    """List all processed documents."""
    try:
        return rag_pipeline.list_documents()
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to list documents"}
        )


@router.delete("/")
async def clear_documents(
    api_key: str = Depends(verify_api_key),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> Dict[str, str]:
    """Clear all documents from the knowledge base."""
    try:
        result = rag_pipeline.clear_knowledge_base()
        return result
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to clear documents"}
        )


@router.get("/stats")
async def get_document_stats(
    api_key: str = Depends(verify_api_key),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> Dict[str, Any]:
    """Get document statistics."""
    try:
        return rag_pipeline.get_document_stats()
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Failed to get document statistics"}
        )