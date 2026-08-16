"""Pytest configuration and fixtures."""

import pytest
import os
import tempfile
import shutil
import json
from fastapi.testclient import TestClient
from typing import Generator, Dict, Any, List, Optional
from unittest.mock import MagicMock, patch

from api.main import app
from config.settings import settings
from rag.pipeline import RAGPipeline
from rag.embeddings import EmbeddingGenerator
from memory.conversation_memory import MemoryManager
from tools.tool_registry import ToolRegistry
from api.dependencies import get_rag_pipeline, get_tool_registry, get_memory_manager


class MockEmbeddingGenerator:
    """Mock embedding generator for testing."""
    
    def __init__(self):
        self.cache = {}
        self.model = "mock-embedding-model"
    
    def generate_embeddings(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """Generate deterministic mock embeddings."""
        return [self.generate_query_embedding(text) for text in texts]
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate deterministic mock query embedding."""
        import hashlib
        hash_val = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        embedding = []
        for i in range(768):
            embedding.append(0.1 + (hash_val % 100) / 1000)
        return embedding
    
    def clear_cache(self):
        self.cache.clear()


class MockGeminiClient:
    """Mock Gemini client for testing."""
    
    def __init__(self):
        self.model = "mock-gemini-model"
        self.temperature = 0.3
        self.max_output_tokens = 2048
    
    def generate(self, prompt: str, system_prompt: str = None, temperature: float = None, 
                 max_tokens: int = None, response_format: Dict[str, Any] = None) -> str:
        """Generate mock response."""
        if "working hours" in prompt.lower() or "hours" in prompt.lower():
            return json.dumps({
                "answer": "According to company_policy.pdf, standard working hours are 9:00 AM to 6:00 PM, Monday through Friday.",
                "key_points": ["Standard working hours: 9:00 AM to 6:00 PM", "Monday through Friday"],
                "confidence": "high",
                "sources": ["company_policy.pdf"]
            })
        elif "remote work" in prompt.lower() or "remote" in prompt.lower():
            return json.dumps({
                "answer": "According to remote_work_policy.pdf, employees may work remotely up to 2 days per week with manager approval.",
                "key_points": ["Remote work allowed up to 2 days per week", "Requires manager approval"],
                "confidence": "high",
                "sources": ["remote_work_policy.pdf"]
            })
        elif "vacation" in prompt.lower() or "leave" in prompt.lower():
            return json.dumps({
                "answer": "According to employee_handbook.pdf, employees are entitled to 20 days of paid vacation per year.",
                "key_points": ["20 paid vacation days per year", "Must be approved by manager"],
                "confidence": "high",
                "sources": ["employee_handbook.pdf"]
            })
        elif "I don't have enough information" in prompt or "no documents" in prompt.lower():
            return json.dumps({
                "answer": "I don't have enough information in the uploaded documents to answer this question confidently.",
                "key_points": ["No relevant documents found", "Please upload documents first"],
                "confidence": "low",
                "sources": []
            })
        else:
            return json.dumps({
                "answer": "I found relevant information in the documents.",
                "key_points": ["Information found in documents"],
                "confidence": "medium",
                "sources": ["document.pdf"]
            })
    
    def generate_stream(self, prompt: str, system_prompt: str = None, temperature: float = None,
                        max_tokens: int = None):
        response = self.generate(prompt, system_prompt, temperature, max_tokens)
        try:
            data = json.loads(response)
            answer = data.get("answer", "")
        except:
            answer = response
        for word in answer.split():
            yield word + " "
    
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class MockReranker:
    """Mock reranker for testing."""
    
    def __init__(self):
        self.temperature = 0.0
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        for i, chunk in enumerate(chunks[:top_k]):
            chunk["rerank_score"] = 1.0 - (i / len(chunks))
        return chunks[:top_k]


class MockQueryTransformer:
    """Mock query transformer for testing."""
    
    def transform(self, query: str, num_alternatives: int = 2, force_transform: bool = False) -> List[str]:
        return [query]


class FakeRAGPipeline:
    """Fake RAG pipeline for API testing - no real Gemini/FAISS/BM25 calls."""
    
    def __init__(self):
        self.documents = {}
        self.chunks = []
        self.processed_documents = set()
        self.llm_calls = 0
        self.embedding_calls = 0
        self.total_tokens = 0
    
    def process_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        try:
            text = file_content.decode('utf-8') if isinstance(file_content, bytes) else str(file_content)
        except UnicodeDecodeError:
            text = "PDF content placeholder"
        
        num_chunks = max(1, len(text) // 100) if text else 1
        chunks = []
        for i in range(num_chunks):
            start = i * 100
            end = min(start + 100, len(text))
            chunk_text = text[start:end] if text else f"Chunk {i} content"
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "chunk_id": f"{filename}_{i}",
                    "chunk_index": i,
                    "document_type": filename.split('.')[-1] if '.' in filename else "txt"
                }
            })
        
        self.chunks.extend(chunks)
        self.documents[filename] = chunks
        self.processed_documents.add(filename)
        
        return {
            "status": "success",
            "filename": filename,
            "chunks": len(chunks),
            "embeddings": len(chunks)
        }
    
    def retrieve(self, query: str, top_k: int = None, semantic_weight: float = None, keyword_weight: float = None) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        top_k = top_k or 8
        results = []
        for i, chunk in enumerate(self.chunks[:min(top_k, len(self.chunks))]):
            results.append({
                "chunk": chunk,
                "combined_score": 0.8 - (i * 0.1),
                "semantic_score": 0.8 - (i * 0.1),
                "keyword_score": 0.8 - (i * 0.1)
            })
        return results
    
    def answer_question(self, question: str, history: List[Dict[str, str]] = None, 
                        top_k: int = None, final_k: int = None, semantic_weight: float = None,
                        keyword_weight: float = None, use_reranking: bool = True,
                        use_query_transformation: bool = False) -> Dict[str, Any]:
        self.llm_calls += 1
        if not self.processed_documents:
            return {
                "answer": "I don't have enough information in the uploaded documents to answer this question confidently. Please upload relevant documents first.",
                "key_points": ["No documents have been uploaded", "Upload documents to enable question answering"],
                "confidence": "low",
                "sources": []
            }
        question_lower = question.lower()
        if "working hours" in question_lower or "hours" in question_lower:
            return {
                "answer": "According to company_policy.pdf, standard working hours are 9:00 AM to 6:00 PM, Monday through Friday.",
                "key_points": ["Standard working hours: 9:00 AM to 6:00 PM", "Monday through Friday"],
                "confidence": "high",
                "sources": ["company_policy.pdf"],
                "chunks": self.chunks[:5] if self.chunks else []
            }
        elif "remote work" in question_lower or "remote" in question_lower:
            return {
                "answer": "According to remote_work_policy.pdf, employees may work remotely up to 2 days per week with manager approval.",
                "key_points": ["Remote work allowed up to 2 days per week", "Requires manager approval"],
                "confidence": "high",
                "sources": ["remote_work_policy.pdf"],
                "chunks": self.chunks[:5] if self.chunks else []
            }
        elif "vacation" in question_lower or "leave" in question_lower:
            return {
                "answer": "According to employee_handbook.pdf, employees are entitled to 20 days of paid vacation per year.",
                "key_points": ["20 paid vacation days per year", "Must be approved by manager"],
                "confidence": "high",
                "sources": ["employee_handbook.pdf"],
                "chunks": self.chunks[:5] if self.chunks else []
            }
        else:
            return {
                "answer": "I found relevant information in the documents.",
                "key_points": ["Information found in documents"],
                "confidence": "medium",
                "sources": list(self.processed_documents)[:3],
                "chunks": self.chunks[:5] if self.chunks else []
            }
    
    def list_documents(self) -> List[str]:
        return sorted(list(self.processed_documents))
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        self.documents.clear()
        self.chunks = []
        self.processed_documents.clear()
        return {"status": "success", "message": "Knowledge base cleared"}
    
    def get_document_stats(self) -> Dict[str, Any]:
        sources = sorted(list(self.processed_documents))
        return {
            "total_documents": len(sources),
            "total_chunks": len(self.chunks),
            "sources": sources,
            "chunks_per_document": {
                source: len(self.documents.get(source, []))
                for source in sources
            }
        }
    
    def get_cost_stats(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
            "total_tokens": self.total_tokens,
            "chunks": len(self.chunks),
            "documents": len(self.processed_documents)
        }


@pytest.fixture(autouse=True)
def mock_all_gemini_calls():
    """Automatically mock all Gemini calls during tests."""
    with patch('rag.embeddings.genai.Client') as mock_embed_client, \
         patch('rag.reranker.genai.Client') as mock_rerank_client, \
         patch('rag.query_transform.genai.Client') as mock_transform_client, \
         patch('llm.client.genai.Client') as mock_llm_client:
        
        mock_embed_client.return_value = MagicMock()
        mock_rerank_client.return_value = MagicMock()
        mock_transform_client.return_value = MagicMock()
        mock_llm_client.return_value = MagicMock()
        
        yield


@pytest.fixture
def test_client() -> Generator:
    """Create test client for FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_key_header() -> Dict[str, str]:
    """Get API key header for authenticated requests."""
    return {"X-API-Key": settings.api_key}


@pytest.fixture
def temp_data_dir() -> Generator[str, None, None]:
    """Create temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    old_data_dir = settings.data_dir
    old_uploads_dir = settings.uploads_dir
    old_processed_dir = settings.processed_dir
    
    settings.data_dir = temp_dir
    settings.uploads_dir = os.path.join(temp_dir, "uploads")
    settings.processed_dir = os.path.join(temp_dir, "processed")
    
    os.makedirs(settings.uploads_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)
    
    yield temp_dir
    
    shutil.rmtree(temp_dir)
    settings.data_dir = old_data_dir
    settings.uploads_dir = old_uploads_dir
    settings.processed_dir = old_processed_dir


@pytest.fixture(autouse=True)
def override_dependencies(temp_data_dir):
    """
    Override FastAPI dependencies with fake pipeline for all tests.
    
    This fixture runs automatically for every test.
    """
    fake_pipeline = FakeRAGPipeline()
    fake_memory_manager = MemoryManager()
    fake_tool_registry = ToolRegistry(fake_pipeline)
    
    app.dependency_overrides[get_rag_pipeline] = lambda: fake_pipeline
    app.dependency_overrides[get_tool_registry] = lambda: fake_tool_registry
    app.dependency_overrides[get_memory_manager] = lambda: fake_memory_manager
    
    yield
    
    app.dependency_overrides.clear()


@pytest.fixture
def rag_pipeline(temp_data_dir) -> Generator[RAGPipeline, None, None]:
    """
    Create a real RAGPipeline instance with mocks for testing.
    This fixture is used by test_rag.py and test_tools.py.
    """
    with patch('rag.pipeline.EmbeddingGenerator', MockEmbeddingGenerator), \
         patch('rag.pipeline.Reranker', MockReranker), \
         patch('rag.pipeline.QueryTransformer', MockQueryTransformer), \
         patch('rag.pipeline.GeminiClient', MockGeminiClient), \
         patch('rag.embeddings.genai.Client') as mock_embed_client, \
         patch('rag.reranker.genai.Client') as mock_rerank_client, \
         patch('rag.query_transform.genai.Client') as mock_transform_client, \
         patch('llm.client.genai.Client') as mock_llm_client:
        
        mock_embed_client.return_value = MagicMock()
        mock_rerank_client.return_value = MagicMock()
        mock_transform_client.return_value = MagicMock()
        mock_llm_client.return_value = MagicMock()
        
        pipeline = RAGPipeline()
        
        # Replace components with mocks
        pipeline.embedding_generator = MockEmbeddingGenerator()
        pipeline.reranker = MockReranker()
        pipeline.query_transformer = MockQueryTransformer()
        pipeline.gemini_client = MockGeminiClient()
        
        yield pipeline


@pytest.fixture
def sample_txt_content() -> str:
    """Get sample text content."""
    return """
    Company Policy Manual
    
    Working Hours
    Standard working hours are 9:00 AM to 6:00 PM, Monday through Friday.
    Core hours are 10:00 AM to 4:00 PM.
    Flexible start time is available between 8:00 AM and 10:00 AM.
    
    Remote Work Policy
    Employees may work remotely up to 2 days per week.
    Remote work requires manager approval.
    
    Vacation Policy
    Employees are entitled to 20 days of paid vacation per year.
    Vacation days must be approved by the manager.
    """


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Get sample PDF content (minimal valid PDF)."""
    return b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 72 712 Td (Test PDF) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000015 00000 n \n0000000067 00000 n \n0000000119 00000 n \n0000000196 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n287\n%%EOF'


@pytest.fixture
def fake_pipeline():
    """Provide direct access to the fake pipeline for tests."""
    return FakeRAGPipeline()