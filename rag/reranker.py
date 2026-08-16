"""LLM-based reranking for retrieved chunks."""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from google import genai
from google.genai.types import GenerateContentConfig

from config.settings import settings
from utils.error_handler import LLMError

logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieved chunks using Gemini LLM."""
    
    def __init__(self):
        """Initialize the reranker."""
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.llm_model
        self.temperature = 0.0  # Deterministic for ranking
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on relevance to query.
        
        Args:
            query: User query
            chunks: List of chunks with scores
            top_k: Number of chunks to return
            
        Returns:
            Reranked list of chunks with updated scores
        """
        if not chunks:
            return []
        
        if len(chunks) <= top_k:
            # Keep original order if we have fewer chunks
            return chunks[:top_k]
        
        try:
            # Prepare chunk texts for reranking (limit to reasonable number)
            max_chunks_for_rerank = min(len(chunks), 10)  # Cap at 10 for efficiency
            chunks_to_rank = chunks[:max_chunks_for_rerank]
            
            chunk_texts = []
            for i, chunk in enumerate(chunks_to_rank):
                text = chunk["chunk"]["text"]
                # Truncate if too long
                if len(text) > 500:
                    text = text[:500] + "..."
                chunk_texts.append(f"[{i}] {text}")
            
            # Create prompt for reranking
            prompt = self._create_rerank_prompt(query, chunk_texts)
            
            # Get LLM response with system instruction for deterministic output
            system_instruction = "You are a precise ranking expert. Return ONLY numbers separated by commas."
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=200,
                    system_instruction=system_instruction
                )
            )
            
            # Parse ranking
            ranked_indices = self._parse_ranking(response.text, len(chunks_to_rank))
            
            # Reorder chunks
            reranked_chunks = []
            for idx in ranked_indices:
                if idx < len(chunks_to_rank):
                    chunk = chunks_to_rank[idx].copy()
                    # Update score to reflect reranking position
                    chunk["rerank_score"] = 1.0 - (len(reranked_chunks) / len(ranked_indices))
                    reranked_chunks.append(chunk)
            
            # Add remaining chunks at the end if any
            remaining_indices = set(range(len(chunks_to_rank))) - set(ranked_indices)
            for idx in remaining_indices:
                chunk = chunks_to_rank[idx].copy()
                chunk["rerank_score"] = 0.0
                reranked_chunks.append(chunk)
            
            # Add any chunks beyond the reranked set (keep original order)
            if len(chunks) > max_chunks_for_rerank:
                for chunk in chunks[max_chunks_for_rerank:]:
                    chunk_copy = chunk.copy()
                    chunk_copy["rerank_score"] = 0.0
                    reranked_chunks.append(chunk_copy)
            
            logger.info(f"Reranked {len(reranked_chunks)} chunks, returning top {min(top_k, len(reranked_chunks))}")
            return reranked_chunks[:top_k]
        
        except Exception as e:
            logger.warning(f"Reranking failed, falling back to original ranking: {str(e)}")
            return chunks[:top_k]
    
    def _create_rerank_prompt(self, query: str, chunk_texts: List[str]) -> str:
        """
        Create prompt for reranking.
        
        Args:
            query: User query
            chunk_texts: List of chunk texts with indices
            
        Returns:
            Prompt string
        """
        chunks_text = "\n".join(chunk_texts)
        
        return f"""You are a relevance ranking expert. Given a query and a list of text chunks,
rank the chunks by their relevance to the query.

IMPORTANT: Rank based on answer relevance, not just keyword matching.
Consider whether each chunk contains information needed to answer the query accurately.

Query: {query}

Chunks:
{chunks_text}

Task: Rank the chunks by relevance. Return ONLY a comma-separated list of indices (0-based)
in order of relevance (most relevant first).

Example output: 3, 1, 5, 2, 4

Ranking:"""
    
    def _parse_ranking(self, response: str, num_chunks: int) -> List[int]:
        """
        Parse the LLM response to extract ranking.
        
        Args:
            response: LLM response text
            num_chunks: Total number of chunks
            
        Returns:
            List of indices in ranked order
        """
        try:
            # Clean response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if "```" in response:
                response = re.sub(r'```.*?\n', '', response)
                response = re.sub(r'```', '', response)
            
            # Remove any text before the first number
            match = re.search(r'(\d[\d,\s]*)', response)
            if match:
                response = match.group(1)
            
            # Extract numbers
            numbers = re.findall(r'\d+', response)
            
            # Convert to integers and filter valid indices
            indices = [int(n) for n in numbers if int(n) < num_chunks]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            
            # If we got valid indices, return them
            if unique_indices:
                return unique_indices
            
            # Fallback: return all indices in order
            logger.warning("No valid indices parsed from reranker response, using fallback order")
            return list(range(num_chunks))
        
        except Exception as e:
            logger.warning(f"Failed to parse reranking response: {str(e)}")
            return list(range(num_chunks))