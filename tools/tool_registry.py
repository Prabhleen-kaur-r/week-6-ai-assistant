"""Tool registry for managing tool calls."""

import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

from tools.document_search import DocumentSearchTool
from tools.document_stats import DocumentStatsTool
from tools.list_documents import ListDocumentsTool
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ToolCallStatus(str, Enum):
    """Status of tool call execution."""
    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the tool registry.
        
        Args:
            rag_pipeline: RAG pipeline instance
        """
        self.rag_pipeline = rag_pipeline
        self.tools: Dict[str, Any] = {}
        self._register_default_tools()
        logger.info(f"Tool registry initialized with {len(self.tools)} tools")
    
    def _register_default_tools(self) -> None:
        """Register the default tools."""
        # Register search tool
        search_tool = DocumentSearchTool(self.rag_pipeline)
        self.register_tool(search_tool.name, search_tool)
        
        # Register stats tool
        stats_tool = DocumentStatsTool(self.rag_pipeline)
        self.register_tool(stats_tool.name, stats_tool)
        
        # Register list documents tool
        list_tool = ListDocumentsTool(self.rag_pipeline)
        self.register_tool(list_tool.name, list_tool)
    
    def register_tool(self, name: str, tool_instance: Any) -> None:
        """
        Register a tool.
        
        Args:
            name: Tool name
            tool_instance: Tool instance with execute method
        """
        if not hasattr(tool_instance, 'execute'):
            raise ValueError(f"Tool {name} must have an execute method")
        
        self.tools[name] = tool_instance
        logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[Any]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None
        """
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools with their metadata.
        
        Returns:
            List of tool information dictionaries
        """
        tool_list = []
        for name, tool in self.tools.items():
            tool_info = {
                "name": name,
                "description": getattr(tool, 'description', 'No description provided'),
                "parameters": getattr(tool, 'parameters', {})
            }
            tool_list.append(tool_info)
        return tool_list
    
    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found or invalid parameters
        """
        # Validate tool exists
        if tool_name not in self.tools:
            logger.warning(f"Tool not found: {tool_name}")
            return {
                "status": ToolCallStatus.NOT_FOUND.value,
                "message": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }
        
        try:
            tool = self.tools[tool_name]
            
            # Validate parameters
            if hasattr(tool, 'parameters'):
                self._validate_parameters(tool.parameters, parameters)
            
            # Execute tool
            logger.info(f"Executing tool: {tool_name} with params: {parameters}")
            result = tool.execute(**parameters)
            
            return {
                "status": ToolCallStatus.SUCCESS.value,
                "tool": tool_name,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {str(e)}")
            return {
                "status": ToolCallStatus.ERROR.value,
                "tool": tool_name,
                "error": str(e)
            }
    
    def _validate_parameters(
        self,
        param_schema: Dict[str, Any],
        provided_params: Dict[str, Any]
    ) -> None:
        """
        Validate tool parameters against schema.
        
        Args:
            param_schema: Parameter schema
            provided_params: Provided parameters
            
        Raises:
            ValueError: If validation fails
        """
        # Check required parameters
        required = param_schema.get('required', [])
        for req in required:
            if req not in provided_params:
                raise ValueError(f"Missing required parameter: {req}")
        
        # Check parameter types (basic validation)
        for param_name, param_value in provided_params.items():
            if param_name not in param_schema.get('properties', {}):
                continue
            
            schema = param_schema['properties'][param_name]
            param_type = schema.get('type')
            
            if param_type == 'string' and not isinstance(param_value, str):
                raise ValueError(f"Parameter {param_name} must be a string")
            elif param_type == 'integer' and not isinstance(param_value, int):
                raise ValueError(f"Parameter {param_name} must be an integer")
            elif param_type == 'number' and not isinstance(param_value, (int, float)):
                raise ValueError(f"Parameter {param_name} must be a number")
            elif param_type == 'boolean' and not isinstance(param_value, bool):
                raise ValueError(f"Parameter {param_name} must be a boolean")
            elif param_type == 'array' and not isinstance(param_value, list):
                raise ValueError(f"Parameter {param_name} must be an array")
            elif param_type == 'object' and not isinstance(param_value, dict):
                raise ValueError(f"Parameter {param_name} must be an object")
    
    def get_tool_schemas(self) -> Dict[str, Any]:
        """
        Get all tool schemas for LLM function calling.
        
        Returns:
            Dictionary of tool schemas
        """
        schemas = {}
        for name, tool in self.tools.items():
            schemas[name] = {
                "name": name,
                "description": getattr(tool, 'description', ''),
                "parameters": getattr(tool, 'parameters', {})
            }
        return schemas