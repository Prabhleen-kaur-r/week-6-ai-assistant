"""RAG evaluation system."""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from evaluation.metrics import RAGMetrics
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG pipeline performance."""
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        test_questions_file: Optional[str] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            rag_pipeline: RAG pipeline instance
            test_questions_file: Path to test questions JSON file
        """
        self.rag_pipeline = rag_pipeline
        self.test_questions = []
        self.results = []
        
        if test_questions_file:
            self.load_test_questions(test_questions_file)
    
    def load_test_questions(self, file_path: str) -> None:
        """
        Load test questions from JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.test_questions = json.load(f)
            logger.info(f"Loaded {len(self.test_questions)} test questions")
        except Exception as e:
            logger.error(f"Failed to load test questions: {str(e)}")
            self.test_questions = []
    
    def add_test_question(
        self,
        question: str,
        expected_sources: List[str],
        expected_keywords: List[str],
        category: str = "general"
    ) -> None:
        """
        Add a test question.
        
        Args:
            question: Test question
            expected_sources: Expected source documents
            expected_keywords: Expected keywords in answer
            category: Question category
        """
        self.test_questions.append({
            "id": f"q{len(self.test_questions) + 1}",
            "question": question,
            "expected_sources": expected_sources,
            "expected_keywords": expected_keywords,
            "category": category
        })
    
    def run_evaluation(
        self,
        use_reranking: bool = True,
        use_query_transformation: bool = True,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on all test questions.
        
        Args:
            use_reranking: Whether to use reranking
            use_query_transformation: Whether to use query transformation
            top_k: Number of chunks to retrieve
            
        Returns:
            Evaluation results summary
        """
        if not self.test_questions:
            return {
                "status": "error",
                "message": "No test questions loaded",
                "results": []
            }
        
        self.results = []
        
        for question_data in self.test_questions:
            try:
                result = self._evaluate_question(
                    question_data,
                    use_reranking,
                    use_query_transformation,
                    top_k
                )
                self.results.append(result)
            except Exception as e:
                logger.error(f"Evaluation failed for question {question_data.get('id')}: {str(e)}")
                self.results.append({
                    "question": question_data,
                    "error": str(e)
                })
        
        summary = self._generate_summary()
        
        logger.info(f"Evaluation completed: {len(self.results)} questions, "
                   f"average accuracy: {summary.get('average_retrieval_accuracy', 0):.2f}")
        
        return summary
    
    def _evaluate_question(
        self,
        question_data: Dict[str, Any],
        use_reranking: bool,
        use_query_transformation: bool,
        top_k: Optional[int]
    ) -> Dict[str, Any]:
        """
        Evaluate a single question.
        
        Args:
            question_data: Question data
            use_reranking: Whether to use reranking
            use_query_transformation: Whether to use query transformation
            top_k: Number of chunks to retrieve
            
        Returns:
            Evaluation result
        """
        question = question_data["question"]
        
        # Generate answer
        result = self.rag_pipeline.answer_question(
            question=question,
            history=[],
            use_reranking=use_reranking,
            use_query_transformation=use_query_transformation,
            top_k=top_k
        )
        
        # Extract sources from result
        cited_sources = result.get("sources", [])
        confidence = result.get("confidence", "medium")
        answer = result.get("answer", "")
        chunks = result.get("chunks", [])
        
        # Get retrieved sources from chunks
        retrieved_sources = []
        for chunk_data in chunks:
            chunk = chunk_data.get("chunk", chunk_data)
            source = chunk.get("metadata", {}).get("source")
            if source:
                retrieved_sources.append(source)
        retrieved_sources = list(set(retrieved_sources))
        
        # Calculate metrics
        metrics = RAGMetrics.calculate_all(
            answer=answer,
            retrieved_sources=retrieved_sources,
            cited_sources=cited_sources,
            expected_sources=question_data.get("expected_sources", []),
            expected_keywords=question_data.get("expected_keywords", []),
            confidence=confidence,
            chunks=chunks
        )
        
        return {
            "question_id": question_data.get("id"),
            "question": question,
            "answer": answer,
            "sources": cited_sources,
            "confidence": confidence,
            "metrics": metrics,
            "chunks_used": len(chunks)
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics from results.
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {
                "total_questions": 0,
                "average_retrieval_accuracy": 0,
                "average_source_accuracy": 0,
                "average_keyword_coverage": 0,
                "average_grounding": 0,
                "average_confidence_accuracy": 0,
                "results": []
            }
        
        # Calculate averages
        metrics_sum = {
            "retrieval_accuracy": 0,
            "source_accuracy": 0,
            "keyword_coverage": 0,
            "answer_grounding": 0,
            "confidence_accuracy": 0
        }
        
        valid_results = [r for r in self.results if "metrics" in r]
        
        for result in valid_results:
            metrics = result["metrics"]
            for key in metrics_sum:
                metrics_sum[key] += metrics.get(key, 0)
        
        count = len(valid_results) or 1
        averages = {
            key: value / count for key, value in metrics_sum.items()
        }
        
        # Categorize results
        category_results = {}
        for result in valid_results:
            category = result.get("question", {}).get("category", "general")
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        category_summaries = {}
        for category, results in category_results.items():
            cat_metrics = {key: 0 for key in metrics_sum}
            for r in results:
                for key in cat_metrics:
                    cat_metrics[key] += r["metrics"].get(key, 0)
            cat_count = len(results) or 1
            category_summaries[category] = {
                key: value / cat_count for key, value in cat_metrics.items()
            }
        
        return {
            "total_questions": len(self.results),
            "valid_responses": len(valid_results),
            "average_retrieval_accuracy": averages["retrieval_accuracy"],
            "average_source_accuracy": averages["source_accuracy"],
            "average_keyword_coverage": averages["keyword_coverage"],
            "average_grounding": averages["answer_grounding"],
            "average_confidence_accuracy": averages["confidence_accuracy"],
            "category_summaries": category_summaries,
            "results": self.results
        }
    
    def save_results(self, output_path: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
    
    def get_best_question(self) -> Optional[Dict[str, Any]]:
        """
        Get the best performing question.
        
        Returns:
            Best question result
        """
        if not self.results:
            return None
        
        valid_results = [r for r in self.results if "metrics" in r]
        if not valid_results:
            return None
        
        # Score by average of metrics
        best_result = None
        best_score = -1
        
        for result in valid_results:
            metrics = result["metrics"]
            score = sum(metrics.values()) / len(metrics)
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def get_worst_question(self) -> Optional[Dict[str, Any]]:
        """
        Get the worst performing question.
        
        Returns:
            Worst question result
        """
        if not self.results:
            return None
        
        valid_results = [r for r in self.results if "metrics" in r]
        if not valid_results:
            return None
        
        worst_result = None
        worst_score = float('inf')
        
        for result in valid_results:
            metrics = result["metrics"]
            score = sum(metrics.values()) / len(metrics)
            if score < worst_score:
                worst_score = score
                worst_result = result
        
        return worst_result