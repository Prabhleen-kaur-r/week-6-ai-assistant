"""Document search tool."""

import logging
from typing import List, Dict, Any, Optional

from rag.pipeline import RAGPipeline
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentSearchTool:
    """Tool for searching documents."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the search tool.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.name = "search_documents"
        self.description = "Search for relevant documents and chunks based on a query"
        self.parameters = {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5
            }
        }
    
    def execute(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the search tool.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Dictionary with search results
        """
        try:
            # Retrieve chunks
            results = self.rag_pipeline.retrieve(
                query=query,
                top_k=top_k
            )
            
            if not results:
                return {
                    "status": "success",
                    "message": "No results found",
                    "results": [],
                    "total": 0
                }
            
            # Format results
            formatted_results = []
            for result in results:
                chunk = result.get("chunk", {})
                metadata = chunk.get("metadata", {})
                formatted_results.append({
                    "text": chunk.get("text", ""),
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page"),
                    "score": result.get("combined_score", 0),
                    "chunk_id": metadata.get("chunk_id", "")
                })
            
            return {
                "status": "success",
                "results": formatted_results,
                "total": len(formatted_results)
            }
        
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "results": [],
                "total": 0
            }