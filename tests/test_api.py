"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "version" in data
    
    def test_health_check_authenticated(self, test_client, api_key_header):
        response = test_client.get("/health", headers=api_key_header)
        assert response.status_code == 200


class TestDocumentEndpoints:
    """Test document management endpoints."""
    
    def test_upload_document(self, test_client, api_key_header, sample_txt_content):
        files = {"file": ("test.txt", sample_txt_content.encode())}
        response = test_client.post(
            "/api/documents/upload",
            files=files,
            headers=api_key_header
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["filename"] == "test.txt"
        assert data["chunks"] > 0
        assert data["embeddings"] > 0
    
    def test_upload_document_unauthorized(self, test_client, sample_txt_content):
        files = {"file": ("test.txt", sample_txt_content.encode())}
        response = test_client.post("/api/documents/upload", files=files)
        assert response.status_code in [401, 403]
    
    def test_list_documents(self, test_client, api_key_header):
        response = test_client.get("/api/documents", headers=api_key_header)
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_document_stats(self, test_client, api_key_header):
        response = test_client.get("/api/documents/stats", headers=api_key_header)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "sources" in data
    
    def test_clear_documents(self, test_client, api_key_header, sample_txt_content):
        # Upload first
        files = {"file": ("test.txt", sample_txt_content.encode())}
        test_client.post("/api/documents/upload", files=files, headers=api_key_header)
        
        # Clear
        response = test_client.delete("/api/documents", headers=api_key_header)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        
        # Check cleared
        stats = test_client.get("/api/documents/stats", headers=api_key_header)
        assert stats.json()["total_documents"] == 0


class TestChatEndpoints:
    """Test chat endpoints."""
    
    def test_chat_no_documents(self, test_client, api_key_header):
        response = test_client.post(
            "/api/chat/",
            json={"message": "What are working hours?"},
            headers=api_key_header
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert "I don't have enough information" in data["answer"] or "upload documents" in data["answer"].lower()
        assert data["confidence"] == "low"
    
    def test_chat_invalid_query(self, test_client, api_key_header):
        response = test_client.post(
            "/api/chat/",
            json={"message": ""},
            headers=api_key_header
        )
        assert response.status_code == 400
    
    def test_chat_with_documents(self, test_client, api_key_header, sample_txt_content):
        # Upload document
        files = {"file": ("test.txt", sample_txt_content.encode())}
        test_client.post("/api/documents/upload", files=files, headers=api_key_header)
        
        # Ask question
        response = test_client.post(
            "/api/chat/",
            json={"message": "What are the working hours?"},
            headers=api_key_header
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert "9:00 AM" in data["answer"] or "hours" in data["answer"].lower()
    
    def test_chat_stream(self, test_client, api_key_header):
        response = test_client.post(
            "/api/chat/stream",
            json={"message": "Hello"},
            headers=api_key_header
        )
        assert response.status_code == 200
        
        content = response.text
        assert len(content) > 0
        assert "chunk" in content


class TestToolEndpoints:
    """Test tool endpoints."""
    
    def test_list_tools(self, test_client, api_key_header):
        response = test_client.get("/api/tools", headers=api_key_header)
        assert response.status_code == 200
        
        tools = response.json()
        assert len(tools) >= 3
        
        tool_names = [t["name"] for t in tools]
        assert "search_documents" in tool_names
        assert "get_document_stats" in tool_names
        assert "list_documents" in tool_names
    
    def test_tool_search(self, test_client, api_key_header):
        response = test_client.post(
            "/api/tools/search",
            json={"query": "test", "top_k": 3},
            headers=api_key_header
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "total" in data
    
    def test_tool_execute_search(self, test_client, api_key_header):
        response = test_client.post(
            "/api/tools/execute",
            json={
                "tool_name": "list_documents",
                "parameters": {}
            },
            headers=api_key_header
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["tool"] == "list_documents"
    
    def test_tool_execute_not_found(self, test_client, api_key_header):
        response = test_client.post(
            "/api/tools/execute",
            json={
                "tool_name": "nonexistent_tool",
                "parameters": {}
            },
            headers=api_key_header
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "not_found"