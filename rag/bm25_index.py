"""BM25 keyword search implementation."""

import os
import pickle
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
import logging
import re

from config.settings import settings

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 keyword search index for document chunks."""
    
    def __init__(self, load_persisted: bool = False):
        """
        Initialize the BM25 index.
        
        Args:
            load_persisted: Whether to load persisted data from disk (default: False)
        """
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []
        self._initialized = False
        self.load_persisted = load_persisted
        
        # Only load from disk if explicitly requested
        if load_persisted:
            self._load_from_disk()
        else:
            logger.info("BM25 index starting fresh (no persistence loaded)")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to the BM25 index."""
        if not chunks:
            return
        
        # Filter out duplicate chunks
        existing_ids = {c.get("metadata", {}).get("chunk_id") for c in self.chunks}
        new_chunks = []
        new_texts = []
        
        for chunk in chunks:
            chunk_id = chunk.get("metadata", {}).get("chunk_id")
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_texts.append(chunk["text"])
                existing_ids.add(chunk_id)
        
        if not new_chunks:
            logger.info("No new chunks to add to BM25 (all duplicates)")
            return
        
        tokenized_texts = [self._tokenize(text) for text in new_texts]
        
        self.chunks.extend(new_chunks)
        self.tokenized_chunks.extend(tokenized_texts)
        
        if not self._initialized:
            self.bm25 = BM25Okapi(self.tokenized_chunks)
            self._initialized = True
        else:
            self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        logger.info(f"Added {len(new_chunks)} chunks to BM25 index (total: {self.count()})")
    
    def search(
        self,
        query: str,
        top_k: int = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for chunks matching the query."""
        if top_k is None:
            top_k = settings.retrieval_top_k
        
        if not self._initialized or self.bm25 is None or self.count() == 0:
            return []
        
        try:
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:min(top_k, len(scores))]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append((self.chunks[idx], float(scores[idx])))
            
            return results
        
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return []
    
    def clear(self) -> None:
        """Clear all data from the BM25 index."""
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []
        self._initialized = False
        self._delete_from_disk()
        logger.info("BM25 index cleared")
    
    def count(self) -> int:
        """Get the number of chunks in the index."""
        return len(self.chunks)
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks in the index."""
        return self.chunks
    
    def save_to_disk(self) -> None:
        """Save the BM25 index to disk."""
        try:
            save_dir = settings.processed_dir
            os.makedirs(save_dir, exist_ok=True)
            
            data_path = os.path.join(save_dir, "bm25_data.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "tokenized_chunks": self.tokenized_chunks,
                    "initialized": self._initialized
                }, f)
            
            logger.info(f"BM25 index saved to {save_dir}")
        
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {str(e)}")
    
    def _load_from_disk(self) -> None:
        """Load the BM25 index from disk."""
        try:
            save_dir = settings.processed_dir
            data_path = os.path.join(save_dir, "bm25_data.pkl")
            
            if not os.path.exists(data_path):
                logger.info("No existing BM25 index found")
                return
            
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data.get("chunks", [])
            self.tokenized_chunks = data.get("tokenized_chunks", [])
            self._initialized = data.get("initialized", False)
            
            if self._initialized and self.tokenized_chunks:
                self.bm25 = BM25Okapi(self.tokenized_chunks)
                logger.info(f"Loaded BM25 index with {self.count()} chunks")
        
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {str(e)}")
            self._initialized = False
    
    def _delete_from_disk(self) -> None:
        """Delete BM25 index files from disk."""
        try:
            save_dir = settings.processed_dir
            data_path = os.path.join(save_dir, "bm25_data.pkl")
            if os.path.exists(data_path):
                os.remove(data_path)
            logger.info("BM25 index files deleted from disk")
        
        except Exception as e:
            logger.warning(f"Failed to delete BM25 index files: {str(e)}")
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [t for t in text.split() if t]
        return tokens