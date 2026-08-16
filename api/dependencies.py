"""FastAPI dependency injection."""

from fastapi import Header, HTTPException, status, Request, Depends
from typing import Optional
import logging

from config.settings import settings
from utils.error_handler import AuthenticationError
from memory.conversation_memory import MemoryManager
from rag.pipeline import RAGPipeline
from tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Global instances - START FRESH (no persistence loaded)
_rag_pipeline = None
_tool_registry = None
memory_manager = MemoryManager()


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """Verify the API key for authentication."""
    if not x_api_key:
        logger.warning("Missing API key in request")
        raise AuthenticationError("Missing API key")
    
    if x_api_key != settings.api_key:
        logger.warning(f"Invalid API key attempt: {x_api_key[:8]}...")
        raise AuthenticationError("Invalid API key")
    
    return x_api_key


async def get_api_key_optional(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """Get API key optionally (for health endpoints)."""
    return x_api_key


def require_api_key(func):
    """Decorator to require API key authentication."""
    async def wrapper(*args, **kwargs):
        await verify_api_key(kwargs.get("x_api_key"))
        return await func(*args, **kwargs)
    return wrapper


def get_memory_manager() -> MemoryManager:
    """Get memory manager instance."""
    return memory_manager


def get_rag_pipeline() -> RAGPipeline:
    """
    Get or create RAG pipeline instance.
    
    IMPORTANT: load_persisted=False ensures fresh start with empty knowledge base.
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        # Start fresh - no persistence loaded
        _rag_pipeline = RAGPipeline(load_persisted=False)
        logger.info("Created new RAG pipeline (fresh, no persisted data)")
    return _rag_pipeline


def set_rag_pipeline(pipeline: RAGPipeline) -> None:
    """Set RAG pipeline instance (for testing)."""
    global _rag_pipeline
    _rag_pipeline = pipeline


def get_tool_registry() -> ToolRegistry:
    """Get or create tool registry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry(get_rag_pipeline())
    return _tool_registry