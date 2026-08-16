"""Input and output guardrails for the application."""

import re
import logging
from typing import Tuple, Optional, List, Dict, Any

from config.settings import settings
from utils.error_handler import ValidationError

logger = logging.getLogger(__name__)


class InputGuardrails:
    """Validate and sanitize user inputs."""
    
    MAX_QUERY_LENGTH = 500
    MIN_QUERY_LENGTH = 3
    MAX_UPLOAD_SIZE = settings.max_upload_size_mb * 1024 * 1024
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore previous instructions',
        r'ignore all previous',
        r'you are now',
        r'you are a',
        r'pretend to be',
        r'act as',
        r'system prompt',
        r'system instruction',
        r'forget previous',
        r'disregard previous',
        r'override',
        r'bypass',
        r'jailbreak',
        r'override all',
    ]
    
    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a user query.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        query = query.strip()
        
        # Check length
        if len(query) < cls.MIN_QUERY_LENGTH:
            return False, f"Query must meet the minimum length of {cls.MIN_QUERY_LENGTH} characters"
        
        if len(query) > cls.MAX_QUERY_LENGTH:
            return False, f"Query exceeds the maximum length of {cls.MAX_QUERY_LENGTH} characters"
        
        # Check for prompt injection
        lower_query = query.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, lower_query, re.IGNORECASE):
                logger.warning(f"Potential prompt injection detected: {query[:50]}...")
                return False, "Query contains prohibited content"
        
        return True, None
    
    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """
        Sanitize a query for safe processing.
        
        Args:
            query: Raw query string
            
        Returns:
            Sanitized query
        """
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove control characters
        query = re.sub(r'[\x00-\x1f\x7f]', '', query)
        
        return query.strip()
    
    @classmethod
    def validate_upload(cls, filename: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate file upload.
        
        Args:
            filename: File name
            file_size: File size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if file_size > cls.MAX_UPLOAD_SIZE:
            return False, f"File size exceeds maximum of {settings.max_upload_size_mb}MB"
        
        if file_size == 0:
            return False, "File is empty"
        
        # Check file extension
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        if ext not in ['pdf', 'txt']:
            return False, f"Unsupported file type: {ext}. Supported: pdf, txt"
        
        return True, None


class OutputGuardrails:
    """Validate and verify LLM outputs."""
    
    @classmethod
    def verify_grounding(
        cls,
        answer: str,
        sources: List[str],
        chunks: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that the answer is grounded in the provided chunks.
        Now more flexible: accepts if cited source is contained in any valid source.
        
        Args:
            answer: Generated answer
            sources: List of source filenames cited by the LLM
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (is_grounded, warning_message)
        """
        if not chunks:
            if "I don't have enough information" in answer:
                return True, None
            return False, "Answer provided without any source documents"
        
        if not sources:
            return False, "No sources cited in the answer"
        
        # Get valid sources from chunks
        valid_sources = set()
        for chunk in chunks:
            chunk_data = chunk.get("chunk", chunk)
            source = chunk_data.get("metadata", {}).get("source")
            if source:
                valid_sources.add(source)
        
        if not valid_sources:
            return False, "No valid sources found in chunks"
        
        # Check each cited source against valid sources (substring match)
        invalid_sources = []
        for cited in sources:
            # Clean up the cited source: remove extra text, keep only filename-like part
            # Attempt to extract filename pattern (e.g., "file.pdf")
            filename_match = re.search(r'([^/\s,;]+\.(?:pdf|txt|docx))', cited, re.IGNORECASE)
            if filename_match:
                cited_clean = filename_match.group(1)
            else:
                cited_clean = cited.strip()
            
            # Check if cited_clean is contained in any valid source (case-insensitive)
            matched = False
            for valid in valid_sources:
                if cited_clean.lower() in valid.lower() or valid.lower() in cited_clean.lower():
                    matched = True
                    break
            if not matched:
                invalid_sources.append(cited_clean)
        
        if invalid_sources:
            logger.warning(f"Invalid sources cited: {invalid_sources}")
            return False, f"Invalid sources cited in answer: {', '.join(invalid_sources)}"
        
        return True, None
    
    @classmethod
    def detect_hallucination(
        cls,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect potential hallucinations in the answer.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (is_safe, warning_message)
        """
        if not chunks:
            if "I don't have enough information" in answer:
                return True, None
            return False, "Answer generated without document support"
        
        # Extract source documents from the answer
        source_pattern = r'according to ([^\s,;.]+)'
        cited_sources = re.findall(source_pattern, answer, re.IGNORECASE)
        
        valid_sources = set()
        for chunk in chunks:
            chunk_data = chunk.get("chunk", chunk)
            source = chunk_data.get("metadata", {}).get("source")
            if source:
                valid_sources.add(source)
        
        for source in cited_sources:
            if source not in valid_sources:
                logger.warning(f"Answer cites non-existent source: {source}")
                return False, f"Answer cites non-existent source: {source}"
        
        return True, None
    
    @classmethod
    def validate_confidence(cls, confidence: str) -> bool:
        """
        Validate confidence level.
        
        Args:
            confidence: Confidence string
            
        Returns:
            True if valid, False otherwise
        """
        return confidence.lower() in ['high', 'medium', 'low']
    
    @classmethod
    def generate_no_context_response(cls, query: str) -> Dict[str, Any]:
        """
        Generate response when no context is available.
        
        Args:
            query: User query
            
        Returns:
            Structured response with no-context message
        """
        return {
            "answer": "I don't have enough information in the uploaded documents to answer this question confidently. Please upload relevant documents first.",
            "key_points": [
                "No documents have been uploaded",
                "Upload documents to enable question answering"
            ],
            "confidence": "low",
            "sources": []
        }