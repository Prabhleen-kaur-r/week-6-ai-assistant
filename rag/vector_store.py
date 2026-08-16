"""FAISS vector store implementation."""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
import logging

from config.settings import settings
from utils.error_handler import RetrievalError

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS vector store for document chunks."""
    
    def __init__(self, dimension: int = 768, load_persisted: bool = False):
        """
        Initialize the vector store.
        
        Args:
            dimension: Embedding dimension
            load_persisted: Whether to load persisted data from disk (default: False)
        """
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.embeddings = []
        self._initialized = False
        self.load_persisted = load_persisted
        
        # Only load from disk if explicitly requested
        if load_persisted:
            self._load_from_disk()
        else:
            logger.info("Vector store starting fresh (no persistence loaded)")
            # Ensure data directory is clean
            self._clean_data_directory()
    
    def _clean_data_directory(self) -> None:
        """Clean the data directory to ensure fresh start."""
        try:
            save_dir = settings.processed_dir
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleaned data directory for fresh start")
        except Exception as e:
            logger.warning(f"Failed to clean data directory: {str(e)}")
    
    def initialize(self, dimension: Optional[int] = None) -> None:
        """Initialize the FAISS index."""
        if dimension:
            self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self._initialized = True
        logger.info(f"Initialized FAISS index with dimension {self.dimension}")
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        """Add chunks and their embeddings to the store."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) doesn't match embeddings count ({len(embeddings)})")
        
        if not chunks:
            return
        
        if not self._initialized:
            embedding_dim = len(embeddings[0]) if embeddings else self.dimension
            self.initialize(embedding_dim)
        
        # Check for duplicate chunks
        existing_ids = {c.get("metadata", {}).get("chunk_id") for c in self.chunks}
        new_chunks = []
        new_embeddings = []
        
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = chunk.get("metadata", {}).get("chunk_id")
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_embeddings.append(emb)
                existing_ids.add(chunk_id)
        
        if not new_chunks:
            logger.info("No new chunks to add (all duplicates)")
            return
        
        embedding_array = np.array(new_embeddings, dtype=np.float32)
        self.chunks.extend(new_chunks)
        self.embeddings.extend(new_embeddings)
        self.index.add(embedding_array)
        
        logger.info(f"Added {len(new_chunks)} chunks to vector store (total: {self.count()})")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar chunks."""
        if top_k is None:
            top_k = settings.retrieval_top_k
        
        if not self._initialized or self.index is None:
            return []
        
        if self.count() == 0:
            return []
        
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_array, min(top_k, self.count()))
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]
                similarity = max(0, 1 - distance / 10)
                results.append((chunk, float(similarity)))
            
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise RetrievalError(f"Vector search failed: {str(e)}")
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.index = None
        self.chunks = []
        self.embeddings = []
        self._initialized = False
        self._delete_from_disk()
        logger.info("Vector store cleared")
    
    def count(self) -> int:
        """Get the number of chunks in the store."""
        return len(self.chunks)
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks in the store."""
        return self.chunks
    
    def save_to_disk(self) -> None:
        """Save the vector store to disk."""
        try:
            save_dir = settings.processed_dir
            os.makedirs(save_dir, exist_ok=True)
            
            if self.index:
                index_path = os.path.join(save_dir, "faiss_index.index")
                faiss.write_index(self.index, index_path)
            
            data_path = os.path.join(save_dir, "vector_data.pkl")
            with open(data_path, 'wb') as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "embeddings": self.embeddings,
                    "dimension": self.dimension,
                    "initialized": self._initialized
                }, f)
            
            logger.info(f"Vector store saved to {save_dir}")
        
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
    
    def _load_from_disk(self) -> None:
        """Load the vector store from disk."""
        try:
            save_dir = settings.processed_dir
            data_path = os.path.join(save_dir, "vector_data.pkl")
            index_path = os.path.join(save_dir, "faiss_index.index")
            
            if not os.path.exists(data_path) or not os.path.exists(index_path):
                logger.info("No existing vector store found")
                return
            
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data.get("chunks", [])
            self.embeddings = data.get("embeddings", [])
            self.dimension = data.get("dimension", self.dimension)
            self._initialized = data.get("initialized", False)
            
            if self._initialized:
                self.index = faiss.read_index(index_path)
                logger.info(f"Loaded vector store with {self.count()} chunks")
        
        except Exception as e:
            logger.warning(f"Failed to load vector store: {str(e)}")
            self._initialized = False
    
    def _delete_from_disk(self) -> None:
        """Delete vector store files from disk."""
        try:
            save_dir = settings.processed_dir
            data_path = os.path.join(save_dir, "vector_data.pkl")
            index_path = os.path.join(save_dir, "faiss_index.index")
            
            for path in [data_path, index_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            logger.info("Vector store files deleted from disk")
        
        except Exception as e:
            logger.warning(f"Failed to delete vector store files: {str(e)}")