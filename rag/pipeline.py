"""Complete RAG pipeline integration."""

import os
from typing import List, Dict, Any, Optional, Tuple
import logging

from config.settings import settings
from rag.ingestion import DocumentIngestor
from rag.chunking import DocumentChunker
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.bm25_index import BM25Index
from rag.hybrid_search import HybridSearch
from rag.reranker import Reranker
from rag.query_transform import QueryTransformer
from llm.client import GeminiClient
from llm.prompts import SYSTEM_PROMPT, format_context, format_history
from llm.structured_output import StructuredOutputParser, ResearchAnswer
from guardrails.safety import InputGuardrails, OutputGuardrails
from utils.error_handler import DocumentProcessingError, RetrievalError

logger = logging.getLogger(__name__)

# Threshold to filter out low‑quality matches – set to 0.6
MIN_RELEVANCE_SCORE = 0.9


class RAGPipeline:
    """Complete RAG pipeline for document processing and retrieval."""
    
    def __init__(self, load_persisted: bool = False):
        """
        Initialize the RAG pipeline.
        
        Args:
            load_persisted: Whether to load persisted data from disk (default: False)
        """
        self.ingestor = DocumentIngestor()
        self.chunker = DocumentChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(load_persisted=load_persisted)
        self.bm25_index = BM25Index(load_persisted=load_persisted)
        self.hybrid_search = HybridSearch(
            self.vector_store,
            self.bm25_index,
            self.embedding_generator
        )
        self.reranker = Reranker()
        self.query_transformer = QueryTransformer()
        self.gemini_client = GeminiClient()
        self.structured_parser = StructuredOutputParser()
        
        self.processed_documents = set()
        self._update_processed_documents()
        
        self.llm_calls = 0
        self.embedding_calls = 0
        self.total_tokens = 0
        
        if load_persisted:
            logger.info(f"RAG pipeline initialized with persisted data ({self.vector_store.count()} chunks)")
        else:
            logger.info("RAG pipeline initialized fresh (no persisted data)")
    
    def process_document(
        self,
        file_content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """Process and index a document."""
        try:
            chunks = self.ingestor.process_document(file_content, filename)
            
            if not chunks:
                raise DocumentProcessingError(
                    "No text extracted from document",
                    filename=filename
                )
            
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)
            self.embedding_calls += 1
            
            if not embeddings:
                raise DocumentProcessingError(
                    "Failed to generate embeddings",
                    filename=filename
                )
            
            self.vector_store.add_chunks(chunks, embeddings)
            self.bm25_index.add_chunks(chunks)
            
            self.vector_store.save_to_disk()
            self.bm25_index.save_to_disk()
            
            self.processed_documents.add(filename)
            
            logger.info(f"Successfully processed document: {filename} ({len(chunks)} chunks)")
            
            return {
                "status": "success",
                "filename": filename,
                "chunks": len(chunks),
                "embeddings": len(embeddings)
            }
        
        except Exception as e:
            logger.error(f"Document processing failed for {filename}: {str(e)}")
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = None,
        keyword_weight: float = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        if self.vector_store.count() == 0:
            return []
        
        try:
            results = self.hybrid_search.search(
                query=query,
                top_k=top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            return results
        
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []
    
    def answer_question(
        self,
        question: str,
        history: List[Dict[str, str]] = None,
        top_k: int = None,
        final_k: int = None,
        semantic_weight: float = None,
        keyword_weight: float = None,
        use_reranking: bool = True,
        use_query_transformation: bool = False
    ) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline."""
        is_valid, error = InputGuardrails.validate_query(question)
        if not is_valid:
            return {
                "answer": f"I cannot process this question: {error}",
                "key_points": [],
                "confidence": "low",
                "sources": [],
                "error": error
            }

        question = InputGuardrails.sanitize_query(question)

        # --- Greeting detection ---
        greetings = {"hello", "hi", "hey", "howdy", "good morning", "good afternoon", "good evening", "yo", "sup"}
        if question.lower().strip() in greetings:
            return {
                "answer": "Hello! I'm your document assistant. I can answer questions about the documents you've uploaded. What would you like to know?",
                "key_points": ["Greeting detected"],
                "confidence": "high",
                "sources": []
            }
        # --- End greeting detection ---

        # Check if knowledge base has documents
        if self.vector_store.count() == 0:
            return {
                "answer": "I can only answer questions about the uploaded documents. Please upload relevant documents first.",
                "key_points": ["No documents uploaded"],
                "confidence": "low",
                "sources": []
            }

        history = history or []
        top_k = top_k or settings.retrieval_top_k
        final_k = final_k or settings.final_top_k

        queries = [question]
        if use_query_transformation:
            try:
                alternatives = self.query_transformer.transform(question, force_transform=True)
                if len(alternatives) > 1:
                    queries = alternatives
                    logger.info(f"Using {len(queries)} query variants")
            except Exception as e:
                logger.warning(f"Query transformation failed: {str(e)}")

        all_results = []
        for q in queries:
            results = self.retrieve(
                query=q,
                top_k=top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            all_results.extend(results)

        seen_chunk_ids = set()
        unique_results = []
        for result in all_results:
            chunk_id = result["chunk"].get("metadata", {}).get("chunk_id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                unique_results.append(result)
            elif not chunk_id:
                text = result["chunk"].get("text", "")[:100]
                if text not in [r["chunk"].get("text", "")[:100] for r in unique_results]:
                    unique_results.append(result)

        unique_results.sort(key=lambda x: x["combined_score"], reverse=True)

        if use_reranking and len(unique_results) > final_k:
            try:
                reranked = self.reranker.rerank(
                    query=question,
                    chunks=unique_results,
                    top_k=min(final_k * 2, len(unique_results))
                )
                unique_results = reranked
            except Exception as e:
                logger.warning(f"Reranking failed: {str(e)}")
                unique_results = unique_results[:final_k * 2]
        else:
            unique_results = unique_results[:final_k * 2]

        # --- Debug: log top score ---
        if unique_results:
            logger.info(f"Top combined score for question '{question[:50]}...': {unique_results[0].get('combined_score', 0):.3f}")
        else:
            logger.info(f"No unique results for question '{question[:50]}...'")

        # --- Apply relevance threshold (now 0.6) ---
        if unique_results and unique_results[0].get("combined_score", 0) < MIN_RELEVANCE_SCORE:
            final_chunks = []
        else:
            final_chunks = unique_results[:final_k]

        # --- If no chunks, reject ---
        if not final_chunks:
            return {
                "answer": "I can only answer questions about the uploaded documents. Please ask a question related to the content of your documents.",
                "key_points": ["No relevant chunks found"],
                "confidence": "low",
                "sources": []
            }

        context = format_context(final_chunks)
        history_text = format_history(history, settings.max_memory_messages)

        prompt = SYSTEM_PROMPT.format(
            context=context,
            history=history_text,
            question=question
        )

        try:
            response_text = self.gemini_client.generate(
                prompt=prompt,
                system_prompt=None,
                temperature=settings.temperature
            )
            self.llm_calls += 1

            structured = self.structured_parser.parse(response_text)

            if structured:
                # --- Check for any rejection phrase ---
                answer_lower = structured.answer.lower()
                rejection_phrases = [
                    "only answer questions about the uploaded documents",
                    "i don't have enough information",
                    "upload documents first",
                    "upload relevant documents",
                    "question related to the content of your documents"
                ]
                if any(phrase in answer_lower for phrase in rejection_phrases):
                    return {
                        "answer": structured.answer,
                        "key_points": structured.key_points,
                        "confidence": structured.confidence.value,
                        "sources": []
                    }

                # Ensure sources are populated from chunks if missing
                if not structured.sources:
                    structured.sources = self._extract_sources_from_chunks(final_chunks)

                # Verify grounding
                is_grounded, warning = OutputGuardrails.verify_grounding(
                    structured.answer,
                    structured.sources,
                    final_chunks
                )

                if not is_grounded:
                    logger.warning(f"Grounding verification failed: {warning}")
                    # Still return the answer, but mark the warning
                    return {
                        "answer": structured.answer,
                        "key_points": structured.key_points,
                        "confidence": structured.confidence.value,
                        "sources": structured.sources,
                        "warning": warning
                    }

                return {
                    "answer": structured.answer,
                    "key_points": structured.key_points,
                    "confidence": structured.confidence.value,
                    "sources": structured.sources,
                    "chunks": final_chunks
                }
            else:
                return {
                    "answer": "I can only answer questions about the uploaded documents. Please ask a question related to the content of your documents.",
                    "key_points": ["Unable to parse response"],
                    "confidence": "low",
                    "sources": []
                }

        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return {
                "answer": "I can only answer questions about the uploaded documents. Please ask a question related to the content of your documents.",
                "key_points": [],
                "confidence": "low",
                "sources": []
            }
    
    def _extract_sources_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source names from chunks."""
        sources = set()
        for chunk_data in chunks:
            chunk = chunk_data.get("chunk", chunk_data)
            source = chunk.get("metadata", {}).get("source")
            if source:
                sources.add(source)
        return sorted(list(sources))
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        chunks = self.vector_store.get_all_chunks()
        sources = set()
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            if source:
                sources.add(source)
        
        return {
            "total_documents": len(sources),
            "total_chunks": len(chunks),
            "sources": sorted(list(sources)),
            "chunks_per_document": {
                source: sum(1 for c in chunks 
                           if c.get("metadata", {}).get("source") == source)
                for source in sources
            }
        }
    
    def list_documents(self) -> List[str]:
        """List all processed documents."""
        return sorted(list(self.processed_documents))
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear all documents from the knowledge base."""
        self.vector_store.clear()
        self.bm25_index.clear()
        self.processed_documents.clear()
        self.embedding_generator.clear_cache()
        self._cleanup_data_directory()
        logger.info("Knowledge base cleared")
        return {
            "status": "success",
            "message": "Knowledge base cleared"
        }
    
    def _update_processed_documents(self) -> None:
        """Update the list of processed documents from the vector store."""
        chunks = self.vector_store.get_all_chunks()
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source")
            if source:
                self.processed_documents.add(source)
    
    def _cleanup_data_directory(self) -> None:
        """Clean up data directory files."""
        try:
            for filename in os.listdir(settings.processed_dir):
                file_path = os.path.join(settings.processed_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Data directory cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup data directory: {str(e)}")
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """Get cost optimization statistics."""
        return {
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
            "total_tokens": self.total_tokens,
            "chunks": self.vector_store.count(),
            "documents": len(self.processed_documents)
        }