"""Tool routes for the API."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from api.dependencies import verify_api_key, get_tool_registry
from tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tools", tags=["tools"])


class ToolInfo(BaseModel):
    """Tool information model."""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolSearchRequest(BaseModel):
    """Tool search request model."""
    query: str
    top_k: Optional[int] = 5


class ToolSearchResponse(BaseModel):
    """Tool search response model."""
    results: List[Dict[str, Any]]
    total: int


class ToolExecuteRequest(BaseModel):
    """Tool execution request model."""
    tool_name: str
    parameters: Dict[str, Any]


class ToolExecuteResponse(BaseModel):
    """Tool execution response model."""
    status: str
    tool: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.get("/")
async def list_tools(
    api_key: str = Depends(verify_api_key),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
) -> List[ToolInfo]:
    """
    List all available tools.
    
    Args:
        api_key: Validated API key
        tool_registry: Tool registry instance
        
    Returns:
        List[ToolInfo]: List of available tools
    """
    try:
        tools = tool_registry.list_tools()
        return [ToolInfo(**tool) for tool in tools]
    
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to list tools: {str(e)}"}
        )


@router.post("/search")
async def tool_search(
    request: ToolSearchRequest,
    api_key: str = Depends(verify_api_key),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
) -> ToolSearchResponse:
    """
    Execute a document search tool.
    
    Args:
        request: Search request
        api_key: Validated API key
        tool_registry: Tool registry instance
        
    Returns:
        ToolSearchResponse: Search results
    """
    try:
        result = tool_registry.execute_tool(
            "search_documents",
            {"query": request.query, "top_k": request.top_k}
        )
        
        if result["status"] == "success":
            search_result = result.get("result", {})
            return ToolSearchResponse(
                results=search_result.get("results", []),
                total=search_result.get("total", 0)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": result.get("error", "Search failed")}
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tool search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Search failed: {str(e)}"}
        )


@router.post("/execute")
async def execute_tool(
    request: ToolExecuteRequest,
    api_key: str = Depends(verify_api_key),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
) -> ToolExecuteResponse:
    """
    Execute a specific tool.
    
    Args:
        request: Tool execution request
        api_key: Validated API key
        tool_registry: Tool registry instance
        
    Returns:
        ToolExecuteResponse: Tool execution result
    """
    try:
        result = tool_registry.execute_tool(
            request.tool_name,
            request.parameters
        )
        
        return ToolExecuteResponse(
            status=result["status"],
            tool=request.tool_name,
            result=result.get("result"),
            error=result.get("error")
        )
    
    except Exception as e:
        logger.error(f"Tool execution error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Tool execution failed: {str(e)}"}
        )