"""RAG pipeline tests."""

import pytest
from rag.chunking import DocumentChunker
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.bm25_index import BM25Index
from rag.hybrid_search import HybridSearch
from rag.pipeline import RAGPipeline
from utils.error_handler import DocumentProcessingError


class TestChunking:
    """Test document chunking."""
    
    def test_chunk_text(self, temp_data_dir):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test document. " * 10
        chunks = chunker.chunk_text(text, "test.txt", "txt")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["source"] == "test.txt"
            assert chunk["metadata"]["document_type"] == "txt"
    
    def test_chunk_empty_text(self, temp_data_dir):
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("", "empty.txt", "txt")
        assert chunks == []
    
    def test_chunk_overlap(self, temp_data_dir):
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "a" * 100
        chunks = chunker.chunk_text(text, "test.txt", "txt")
        
        # Check that chunks overlap
        if len(chunks) > 1:
            first_chunk = chunks[0]["text"]
            second_chunk = chunks[1]["text"]
            # Should have some overlap
            assert any(char in second_chunk for char in first_chunk[-10:])


class TestVectorStore:
    """Test FAISS vector store."""
    
    def test_add_and_search(self, temp_data_dir):
        store = VectorStore(dimension=10)
        store.initialize(10)
        
        chunks = [
            {"text": "First chunk", "metadata": {"chunk_id": "1"}},
            {"text": "Second chunk", "metadata": {"chunk_id": "2"}},
            {"text": "Third chunk", "metadata": {"chunk_id": "3"}}
        ]
        embeddings = [
            [0.1] * 10,
            [0.2] * 10,
            [0.3] * 10
        ]
        
        store.add_chunks(chunks, embeddings)
        assert store.count() == 3
        
        results = store.search([0.1] * 10, top_k=2)
        assert len(results) == 2
    
    def test_clear(self, temp_data_dir):
        store = VectorStore(dimension=10)
        store.initialize(10)
        store.add_chunks(
            [{"text": "test", "metadata": {}}],
            [[0.1] * 10]
        )
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
    
    def test_get_all_chunks(self, temp_data_dir):
        store = VectorStore(dimension=10)
        store.initialize(10)
        chunks = [{"text": f"chunk{i}", "metadata": {"id": i}} for i in range(3)]
        embeddings = [[0.1] * 10 for _ in range(3)]
        store.add_chunks(chunks, embeddings)
        
        all_chunks = store.get_all_chunks()
        assert len(all_chunks) == 3


class TestBM25Index:
    """Test BM25 index."""
    
    def test_add_and_search(self, temp_data_dir):
        bm25 = BM25Index()
        chunks = [
            {"text": "working hours are 9 to 6", "metadata": {"id": "1"}},
            {"text": "remote work policy", "metadata": {"id": "2"}},
            {"text": "vacation days policy", "metadata": {"id": "3"}}
        ]
        bm25.add_chunks(chunks)
        assert bm25.count() == 3
        
        results = bm25.search("working hours", top_k=2)
        assert len(results) > 0
    
    def test_clear(self, temp_data_dir):
        bm25 = BM25Index()
        bm25.add_chunks([{"text": "test", "metadata": {}}])
        assert bm25.count() == 1
        bm25.clear()
        assert bm25.count() == 0


class TestHybridSearch:
    """Test hybrid search."""
    
    def test_search(self, temp_data_dir):
        # Setup
        vector_store = VectorStore(dimension=10)
        vector_store.initialize(10)
        
        bm25_index = BM25Index()
        
        chunks = [
            {"text": "working hours 9 to 6", "metadata": {"chunk_id": "1", "source": "doc1.txt"}},
            {"text": "remote work allowed", "metadata": {"chunk_id": "2", "source": "doc2.txt"}},
        ]
        embeddings = [[0.1] * 10, [0.2] * 10]
        
        vector_store.add_chunks(chunks, embeddings)
        bm25_index.add_chunks(chunks)
        
        # We need to mock embedding generator
        class MockEmbeddingGenerator:
            def generate_query_embedding(self, query):
                return [0.1] * 10
        
        hybrid = HybridSearch(vector_store, bm25_index, MockEmbeddingGenerator())
        results = hybrid.search("working hours", top_k=2)
        
        # Results should be returned
        assert len(results) <= 2


class TestRAGPipeline:
    """Test complete RAG pipeline."""
    
    def test_process_document(self, rag_pipeline, sample_txt_content):
        result = rag_pipeline.process_document(
            sample_txt_content.encode(),
            "test.txt"
        )
        
        assert result["status"] == "success"
        assert result["filename"] == "test.txt"
        assert result["chunks"] > 0
    
    def test_process_empty_document(self, rag_pipeline):
        with pytest.raises(DocumentProcessingError):
            rag_pipeline.process_document(b"", "empty.txt")
    
    def test_get_document_stats(self, rag_pipeline, sample_txt_content):
        rag_pipeline.process_document(sample_txt_content.encode(), "test.txt")
        stats = rag_pipeline.get_document_stats()
        
        assert stats["total_documents"] >= 1
        assert stats["total_chunks"] > 0
        assert "test.txt" in stats["sources"]
    
    def test_list_documents(self, rag_pipeline, sample_txt_content):
        rag_pipeline.process_document(sample_txt_content.encode(), "test1.txt")
        rag_pipeline.process_document(sample_txt_content.encode(), "test2.txt")
        
        docs = rag_pipeline.list_documents()
        assert len(docs) >= 2
        assert "test1.txt" in docs
    
    def test_clear_knowledge_base(self, rag_pipeline, sample_txt_content):
        rag_pipeline.process_document(sample_txt_content.encode(), "test.txt")
        assert rag_pipeline.get_document_stats()["total_documents"] >= 1
        
        rag_pipeline.clear_knowledge_base()
        stats = rag_pipeline.get_document_stats()
        assert stats["total_documents"] == 0
        assert stats["total_chunks"] == 0