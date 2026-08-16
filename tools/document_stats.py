"""Document statistics tool."""

import logging
from typing import Dict, Any

from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class DocumentStatsTool:
    """Tool for getting document statistics."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the stats tool.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.name = "get_document_stats"
        self.description = "Get statistics about the knowledge base including document count, chunk count, and sources"
        self.parameters = {}
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the stats tool.
        
        Returns:
            Dictionary with document statistics
        """
        try:
            stats = self.rag_pipeline.get_document_stats()
            
            return {
                "status": "success",
                "total_documents": stats["total_documents"],
                "total_chunks": stats["total_chunks"],
                "sources": stats["sources"],
                "chunks_per_document": stats.get("chunks_per_document", {})
            }
        
        except Exception as e:
            logger.error(f"Document stats failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to get stats: {str(e)}",
                "total_documents": 0,
                "total_chunks": 0,
                "sources": []
            }