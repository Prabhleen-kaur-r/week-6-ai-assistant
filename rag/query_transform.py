"""Query transformation for improved retrieval."""

import logging
from typing import List, Optional
from google import genai

from config.settings import settings
from utils.error_handler import LLMError

logger = logging.getLogger(__name__)


class QueryTransformer:
    """Transform queries to improve retrieval."""
    
    def __init__(self):
        """Initialize the query transformer."""
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.llm_model
        self.temperature = 0.2
        self.max_alternatives = 3
    
    def transform(
        self,
        query: str,
        num_alternatives: int = 2,
        force_transform: bool = False
    ) -> List[str]:
        """
        Generate alternative query formulations.
        
        Args:
            query: Original query
            num_alternatives: Number of alternative queries to generate
            force_transform: If True, force generation even for short queries
            
        Returns:
            List of query variants including original
        """
        # By default, do NOT generate alternatives for normal queries
        # Only transform if explicitly requested with force_transform=True
        if not force_transform:
            logger.debug("Query transformation disabled by default, using original query")
            return [query]
        
        # If query is short, skip transformation even when forced
        if len(query.split()) < 3:
            logger.debug("Query too short for transformation, using original")
            return [query]
        
        try:
            # Generate alternatives (this makes an API call)
            alternatives = self._generate_alternatives(query, num_alternatives)
            
            # Combine with original, remove duplicates
            all_queries = [query]
            for alt in alternatives:
                if alt and alt.strip() and alt != query:
                    all_queries.append(alt.strip())
            
            # Limit number of queries
            all_queries = all_queries[:self.max_alternatives]
            
            logger.info(f"Transformed query into {len(all_queries)} variants")
            return all_queries
        
        except Exception as e:
            logger.warning(f"Query transformation failed: {str(e)}")
            return [query]
    
    def _generate_alternatives(
        self,
        query: str,
        num_alternatives: int
    ) -> List[str]:
        """
        Generate alternative query formulations using LLM.
        
        Args:
            query: Original query
            num_alternatives: Number of alternatives
            
        Returns:
            List of alternative queries
        """
        prompt = self._create_transform_prompt(query, num_alternatives)
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": 300,
                }
            )
            
            # Parse response
            alternatives = self._parse_alternatives(response.text)
            return alternatives
        
        except Exception as e:
            logger.warning(f"Alternative generation failed: {str(e)}")
            return []
    
    def _create_transform_prompt(self, query: str, num: int) -> str:
        """
        Create prompt for query transformation.
        
        Args:
            query: Original query
            num: Number of alternatives
            
        Returns:
            Prompt string
        """
        return f"""Generate {num} alternative search queries for the following question. 
The alternatives should capture different aspects or phrasings of the original question 
to improve document retrieval.

Original Query: {query}

Task: Generate {num} alternative queries that:
1. Use different wording
2. Include key terms from the original
3. Might help find relevant documents

Return ONLY the alternative queries, one per line, with no numbers or bullets.

Alternative Queries:"""
    
    def _parse_alternatives(self, response: str) -> List[str]:
        """
        Parse alternatives from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            List of alternative queries
        """
        lines = response.strip().split('\n')
        alternatives = []
        
        for line in lines:
            # Clean line
            line = line.strip()
            
            # Remove numbers/bullets
            if line and line[0].isdigit():
                line = line.split('.', 1)[-1].strip()
            elif line and line[0] in ['-', '*', '•']:
                line = line[1:].strip()
            
            if line and len(line) > 10:  # Filter out very short lines
                alternatives.append(line)
        
        return alternatives[:self.max_alternatives - 1]