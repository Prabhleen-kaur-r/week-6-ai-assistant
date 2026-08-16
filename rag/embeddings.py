"""Embedding generation using Gemini."""

import time
from typing import List, Dict, Any, Optional
import logging
from google import genai
from google.genai.types import EmbedContentConfig

from config.settings import settings
from utils.error_handler import LLMError

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Gemini Embeddings."""
    
    def __init__(self):
        self.model = settings.embedding_model
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.batch_size = 5  # reduced to avoid quota issues
        self.cache = {}
        self.max_retries = 3
        self.retry_delay = 5
        
        logger.info(f"Initialized embedding generator with model: {self.model}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        
        texts = [str(t).strip() for t in texts if t and str(t).strip()]
        if not texts:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    embeddings = self._embed_batch(batch, task_type)
                    all_embeddings.extend(embeddings)
                    logger.debug(f"Batch {i//self.batch_size + 1}: {len(batch)} texts")
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        wait_time = self.retry_delay * (attempt + 1)
                        logger.warning(f"Quota exceeded, retrying in {wait_time}s (attempt {attempt + 1})")
                        time.sleep(wait_time)
                        if attempt == self.max_retries - 1:
                            raise LLMError(f"Embedding failed after {self.max_retries} retries: {str(e)}")
                    elif "404" in error_str or "NOT_FOUND" in error_str:
                        logger.error(f"Embedding model not found: {self.model}")
                        raise LLMError(f"Embedding model '{self.model}' not found. Please check your EMBEDDING_MODEL setting.")
                    else:
                        raise
        
        return all_embeddings
    
    def _embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """Embed a batch of texts using Gemini, with robust parsing and fallback."""
        if not texts:
            return []
        
        try:
            model_path = f"models/{self.model}" if not self.model.startswith("models/") else self.model
            response = self.client.models.embed_content(
                model=model_path,
                contents=texts,
                config=EmbedContentConfig(task_type=task_type)
            )
            
            # Extract embeddings from response
            embeddings = []
            if hasattr(response, 'embeddings'):
                for emb in response.embeddings:
                    if hasattr(emb, 'values'):
                        embeddings.append(emb.values)
                    else:
                        embeddings.append(emb)
            else:
                # Fallback: response might be a list of embedding objects
                for emb in response:
                    if hasattr(emb, 'values'):
                        embeddings.append(emb.values)
                    else:
                        embeddings.append(emb)
            
            # If the returned count doesn't match the input, fallback to individual requests.
            if len(embeddings) != len(texts):
                logger.warning(
                    f"Batch embedding returned {len(embeddings)} embeddings for {len(texts)} texts. "
                    "Falling back to individual requests."
                )
                individual_embeddings = []
                for text in texts:
                    single_response = self.client.models.embed_content(
                        model=model_path,
                        contents=[text],
                        config=EmbedContentConfig(task_type=task_type)
                    )
                    if hasattr(single_response, 'embeddings'):
                        emb = single_response.embeddings[0]
                        individual_embeddings.append(emb.values if hasattr(emb, 'values') else emb)
                    else:
                        emb = single_response[0]
                        individual_embeddings.append(emb.values if hasattr(emb, 'values') else emb)
                return individual_embeddings
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Gemini embedding API error: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        if not query or not query.strip():
            raise LLMError("Empty query provided for embedding")
        
        cache_key = query.lower().strip()
        if cache_key in self.cache:
            logger.debug("Using cached query embedding")
            return self.cache[cache_key]
        
        for attempt in range(self.max_retries):
            try:
                embeddings = self.generate_embeddings(
                    [query],
                    task_type="RETRIEVAL_QUERY"
                )
                
                if embeddings and len(embeddings) > 0:
                    embedding = embeddings[0]
                    self.cache[cache_key] = embedding
                    return embedding
                else:
                    raise LLMError("No embedding generated for query")
                    
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(f"Quota exceeded for query, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    if attempt == self.max_retries - 1:
                        raise
                elif "404" in error_str or "NOT_FOUND" in error_str:
                    raise LLMError(f"Embedding model '{self.model}' not found. Please check your EMBEDDING_MODEL setting.")
                else:
                    raise
        
        raise LLMError("Failed to generate query embedding after retries")
    
    def clear_cache(self) -> None:
        self.cache.clear()
        logger.info("Embedding cache cleared")