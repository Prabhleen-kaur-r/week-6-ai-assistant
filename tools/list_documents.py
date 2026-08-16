"""List documents tool."""

import logging
from typing import List, Dict, Any

from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ListDocumentsTool:
    """Tool for listing all uploaded documents."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the list documents tool.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.name = "list_documents"
        self.description = "List all uploaded and processed documents"
        self.parameters = {}
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the list documents tool.
        
        Returns:
            Dictionary with document list
        """
        try:
            documents = self.rag_pipeline.list_documents()
            
            return {
                "status": "success",
                "documents": documents,
                "total": len(documents)
            }
        
        except Exception as e:
            logger.error(f"List documents failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to list documents: {str(e)}",
                "documents": [],
                "total": 0
            }