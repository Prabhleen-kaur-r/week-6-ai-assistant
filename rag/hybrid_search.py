"""Hybrid search combining semantic and keyword retrieval."""

from typing import List, Dict, Any, Tuple, Optional
import logging

from config.settings import settings
from rag.vector_store import VectorStore
from rag.bm25_index import BM25Index
from rag.embeddings import EmbeddingGenerator
from utils.error_handler import RetrievalError

logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid search combining semantic (FAISS) and keyword (BM25) retrieval."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize hybrid search.
        
        Args:
            vector_store: FAISS vector store
            bm25_index: BM25 keyword index
            embedding_generator: Embedding generator
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_generator = embedding_generator
        
        self.semantic_weight = settings.semantic_weight
        self.keyword_weight = settings.keyword_weight
        self.retrieval_top_k = settings.retrieval_top_k
    
    def search(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = None,
        keyword_weight: float = None,
        min_score: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
            min_score: Minimum score threshold
            
        Returns:
            List of result dictionaries with chunk and scores
        """
        if top_k is None:
            top_k = self.retrieval_top_k
        
        if semantic_weight is None:
            semantic_weight = self.semantic_weight
        
        if keyword_weight is None:
            keyword_weight = self.keyword_weight
        
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        if total_weight == 0:
            semantic_weight = 0.5
            keyword_weight = 0.5
        else:
            semantic_weight = semantic_weight / total_weight
            keyword_weight = keyword_weight / total_weight
        
        # Get semantic results
        semantic_results = self._semantic_search(query, top_k * 2)
        
        # Get keyword results
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine and normalize scores
        combined_results = self._combine_results(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        # Filter and sort
        filtered_results = [
            r for r in combined_results
            if r["combined_score"] >= min_score
        ]
        
        # Sort by combined score
        filtered_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Return top-k
        final_results = filtered_results[:top_k]
        
        logger.info(
            f"Hybrid search returned {len(final_results)} results "
            f"(semantic: {len(semantic_results)}, keyword: {len(keyword_results)})"
        )
        
        return final_results
    
    def _semantic_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform semantic search using FAISS.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.vector_store.count() == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, top_k)
            
            return results
        
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    def _keyword_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform keyword search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.bm25_index.count() == 0:
            return []
        
        try:
            results = self.bm25_index.search(query, top_k)
            return results
        
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def _combine_results(
        self,
        semantic_results: List[Tuple[Dict[str, Any], float]],
        keyword_results: List[Tuple[Dict[str, Any], float]],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine and normalize scores from both search methods.
        
        Args:
            semantic_results: Semantic search results
            keyword_results: Keyword search results
            semantic_weight: Weight for semantic scores
            keyword_weight: Weight for keyword scores
            
        Returns:
            List of combined result dictionaries
        """
        # Map results by chunk ID
        combined = {}
        
        # Process semantic results
        max_semantic = max([score for _, score in semantic_results], default=1)
        for chunk, score in semantic_results:
            chunk_id = chunk.get("metadata", {}).get("chunk_id", str(id(chunk)))
            normalized_score = score / max_semantic if max_semantic > 0 else 0
            
            combined[chunk_id] = {
                "chunk": chunk,
                "semantic_score": normalized_score,
                "keyword_score": 0,
                "combined_score": 0
            }
        
        # Process keyword results
        max_keyword = max([score for _, score in keyword_results], default=1)
        for chunk, score in keyword_results:
            chunk_id = chunk.get("metadata", {}).get("chunk_id", str(id(chunk)))
            normalized_score = score / max_keyword if max_keyword > 0 else 0
            
            if chunk_id in combined:
                combined[chunk_id]["keyword_score"] = normalized_score
            else:
                combined[chunk_id] = {
                    "chunk": chunk,
                    "semantic_score": 0,
                    "keyword_score": normalized_score,
                    "combined_score": 0
                }
        
        # Calculate combined scores
        for chunk_id, data in combined.items():
            data["combined_score"] = (
                data["semantic_score"] * semantic_weight +
                data["keyword_score"] * keyword_weight
            )
        
        return list(combined.values())
    
    def update_weights(
        self,
        semantic_weight: float,
        keyword_weight: float
    ) -> None:
        """
        Update search weights.
        
        Args:
            semantic_weight: New semantic weight
            keyword_weight: New keyword weight
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        logger.info(f"Updated search weights: semantic={semantic_weight}, keyword={keyword_weight}")