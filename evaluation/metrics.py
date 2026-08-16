"""RAG evaluation metrics."""

from typing import List, Dict, Any, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class RAGMetrics:
    """Calculate RAG evaluation metrics."""
    
    @staticmethod
    def retrieval_accuracy(
        retrieved_sources: List[str],
        expected_sources: List[str]
    ) -> float:
        """
        Calculate retrieval accuracy based on source overlap.
        
        Args:
            retrieved_sources: Sources retrieved by the system
            expected_sources: Expected sources for the question
            
        Returns:
            Accuracy score (0-1)
        """
        if not expected_sources:
            return 1.0
        
        retrieved_set = set(retrieved_sources)
        expected_set = set(expected_sources)
        
        if not retrieved_set:
            return 0.0
        
        # Precision: retrieved sources that are expected
        precision = len(retrieved_set.intersection(expected_set)) / len(retrieved_set) if retrieved_set else 0
        
        # Recall: expected sources that were retrieved
        recall = len(retrieved_set.intersection(expected_set)) / len(expected_set) if expected_set else 0
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def source_accuracy(
        cited_sources: List[str],
        retrieved_sources: List[str]
    ) -> float:
        """
        Check if cited sources are from retrieved sources.
        
        Args:
            cited_sources: Sources cited in the answer
            retrieved_sources: Sources that were retrieved
            
        Returns:
            Accuracy score (0-1)
        """
        if not cited_sources:
            return 0.0
        
        if not retrieved_sources:
            return 0.0
        
        cited_set = set(cited_sources)
        retrieved_set = set(retrieved_sources)
        
        # Check if all cited sources were retrieved
        valid_citations = cited_set.intersection(retrieved_set)
        
        return len(valid_citations) / len(cited_set) if cited_set else 0.0
    
    @staticmethod
    def keyword_coverage(
        answer: str,
        expected_keywords: List[str]
    ) -> float:
        """
        Check if expected keywords appear in the answer.
        
        Args:
            answer: Generated answer
            expected_keywords: Expected keywords
            
        Returns:
            Coverage score (0-1)
        """
        if not expected_keywords:
            return 1.0
        
        if not answer:
            return 0.0
        
        answer_lower = answer.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        
        return found / len(expected_keywords) if expected_keywords else 0.0
    
    @staticmethod
    def answer_grounding(
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate if answer is grounded in retrieved chunks.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            Grounding score (0-1)
        """
        if not chunks:
            return 0.0
        
        if not answer:
            return 0.0
        
        # Check if answer contains content from chunks
        chunk_texts = []
        for chunk_data in chunks:
            chunk = chunk_data.get("chunk", chunk_data)
            text = chunk.get("text", "")
            if text:
                chunk_texts.append(text.lower())
        
        if not chunk_texts:
            return 0.0
        
        # Simple overlap check
        answer_lower = answer.lower()
        answer_words = set(answer_lower.split())
        
        # Check for significant overlap
        chunk_words = set()
        for text in chunk_texts:
            chunk_words.update(text.split())
        
        if not chunk_words:
            return 0.0
        
        overlap = len(answer_words.intersection(chunk_words))
        total = len(answer_words)
        
        return overlap / total if total > 0 else 0.0
    
    @staticmethod
    def confidence_accuracy(
        confidence: str,
        is_correct: bool
    ) -> float:
        """
        Evaluate confidence calibration.
        
        Args:
            confidence: Confidence level (high/medium/low)
            is_correct: Whether the answer is correct
            
        Returns:
            Calibration score (0-1)
        """
        confidence_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
        confidence_score = confidence_map.get(confidence.lower(), 0.5)
        
        # If confident but wrong, or not confident but right, penalize
        if is_correct:
            return confidence_score
        else:
            return 1.0 - confidence_score
    
    @staticmethod
    def calculate_all(
        answer: str,
        retrieved_sources: List[str],
        cited_sources: List[str],
        expected_sources: List[str],
        expected_keywords: List[str],
        confidence: str,
        chunks: List[Dict[str, Any]],
        is_correct: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "retrieval_accuracy": RAGMetrics.retrieval_accuracy(
                retrieved_sources, expected_sources
            ),
            "source_accuracy": RAGMetrics.source_accuracy(
                cited_sources, retrieved_sources
            ),
            "keyword_coverage": RAGMetrics.keyword_coverage(
                answer, expected_keywords
            ),
            "answer_grounding": RAGMetrics.answer_grounding(
                answer, chunks
            ),
            "confidence_accuracy": RAGMetrics.confidence_accuracy(
                confidence, is_correct
            )
        }