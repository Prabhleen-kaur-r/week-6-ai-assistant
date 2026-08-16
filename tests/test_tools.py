"""Tool tests."""

import pytest
from tools.document_search import DocumentSearchTool
from tools.document_stats import DocumentStatsTool
from tools.list_documents import ListDocumentsTool
from tools.tool_registry import ToolRegistry


class TestDocumentSearchTool:
    """Test document search tool."""
    
    def test_init(self, rag_pipeline):
        tool = DocumentSearchTool(rag_pipeline)
        assert tool.name == "search_documents"
        assert "query" in tool.parameters
    
    def test_execute_no_documents(self, rag_pipeline):
        tool = DocumentSearchTool(rag_pipeline)
        result = tool.execute("test query")
        
        assert result["status"] == "success"
        assert result["total"] == 0
        assert result["results"] == []
    
    def test_execute_with_documents(self, rag_pipeline, sample_txt_content):
        # Process document
        rag_pipeline.process_document(sample_txt_content.encode(), "test.txt")
        
        tool = DocumentSearchTool(rag_pipeline)
        result = tool.execute("working hours", top_k=3)
        
        assert result["status"] == "success"
        assert result["total"] > 0
        assert "results" in result


class TestDocumentStatsTool:
    """Test document statistics tool."""
    
    def test_init(self, rag_pipeline):
        tool = DocumentStatsTool(rag_pipeline)
        assert tool.name == "get_document_stats"
    
    def test_execute_no_documents(self, rag_pipeline):
        tool = DocumentStatsTool(rag_pipeline)
        result = tool.execute()
        
        assert result["status"] == "success"
        assert result["total_documents"] == 0
        assert result["total_chunks"] == 0
        assert result["sources"] == []
    
    def test_execute_with_documents(self, rag_pipeline, sample_txt_content):
        rag_pipeline.process_document(sample_txt_content.encode(), "test.txt")
        
        tool = DocumentStatsTool(rag_pipeline)
        result = tool.execute()
        
        assert result["status"] == "success"
        assert result["total_documents"] == 1
        assert result["total_chunks"] > 0
        assert "test.txt" in result["sources"]


class TestListDocumentsTool:
    """Test list documents tool."""
    
    def test_init(self, rag_pipeline):
        tool = ListDocumentsTool(rag_pipeline)
        assert tool.name == "list_documents"
    
    def test_execute_no_documents(self, rag_pipeline):
        tool = ListDocumentsTool(rag_pipeline)
        result = tool.execute()
        
        assert result["status"] == "success"
        assert result["total"] == 0
        assert result["documents"] == []
    
    def test_execute_with_documents(self, rag_pipeline, sample_txt_content):
        rag_pipeline.process_document(sample_txt_content.encode(), "test1.txt")
        rag_pipeline.process_document(sample_txt_content.encode(), "test2.txt")
        
        tool = ListDocumentsTool(rag_pipeline)
        result = tool.execute()
        
        assert result["status"] == "success"
        assert result["total"] >= 2
        assert "test1.txt" in result["documents"]
        assert "test2.txt" in result["documents"]


class TestToolRegistry:
    """Test tool registry."""
    
    def test_init(self, rag_pipeline):
        registry = ToolRegistry(rag_pipeline)
        tools = registry.list_tools()
        assert len(tools) >= 3
        
        tool_names = [t["name"] for t in tools]
        assert "search_documents" in tool_names
        assert "get_document_stats" in tool_names
        assert "list_documents" in tool_names
    
    def test_execute_tool(self, rag_pipeline):
        registry = ToolRegistry(rag_pipeline)
        
        result = registry.execute_tool(
            "list_documents",
            {}
        )
        assert result["status"] == "success"
        assert result["tool"] == "list_documents"
    
    def test_execute_not_found(self, rag_pipeline):
        registry = ToolRegistry(rag_pipeline)
        
        result = registry.execute_tool(
            "nonexistent",
            {}
        )
        assert result["status"] == "not_found"
        assert "available_tools" in result
    
    def test_validate_parameters(self, rag_pipeline):
        registry = ToolRegistry(rag_pipeline)
        
        # Should handle invalid parameters gracefully
        result = registry.execute_tool(
            "search_documents",
            {"invalid_param": "test"}
        )
        # Should still execute but might fail due to missing required params
        assert result["status"] in ["success", "error"]