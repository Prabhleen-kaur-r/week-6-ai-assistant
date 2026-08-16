"""Chat routes for the API."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
import asyncio

from api.dependencies import (
    verify_api_key,
    get_memory_manager,
    get_rag_pipeline,
    get_tool_registry
)
from memory.conversation_memory import MemoryManager
from rag.pipeline import RAGPipeline
from tools.tool_registry import ToolRegistry
from guardrails.safety import InputGuardrails

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False
    use_reranking: bool = True
    use_query_transformation: bool = False


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: str
    session_id: Optional[str] = None
    key_points: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@router.post("/")
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    tool_registry: ToolRegistry = Depends(get_tool_registry)
) -> ChatResponse:
    try:
        is_valid, error = InputGuardrails.validate_query(request.message)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": error}
            )
        
        memory = memory_manager.get_or_create_memory(request.session_id)
        session_id = memory.session_id
        
        memory.add_message("user", request.message)
        
        tool_result = None
        if "search for" in request.message.lower() or "find" in request.message.lower():
            try:
                tool_result = tool_registry.execute_tool(
                    "search_documents",
                    {"query": request.message, "top_k": 5}
                )
            except Exception as e:
                logger.warning(f"Tool execution attempt failed: {str(e)}")
        
        history = memory.build_context(include_metadata=False)
        
        result = rag_pipeline.answer_question(
            question=request.message,
            history=history[:-1],
            use_reranking=request.use_reranking,
            use_query_transformation=request.use_query_transformation
        )
        
        memory.add_message(
            "assistant",
            result["answer"],
            metadata={
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", "medium")
            }
        )
        
        response = ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", "medium"),
            session_id=session_id,
            key_points=result.get("key_points")
        )
        
        if tool_result:
            response.tool_calls = [{"tool": "search_documents", "result": tool_result}]
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to process chat: {str(e)}"}
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> StreamingResponse:
    async def generate():
        try:
            is_valid, error = InputGuardrails.validate_query(request.message)
            if not is_valid:
                yield json.dumps({"error": error}) + "\n"
                return
            
            memory = memory_manager.get_or_create_memory(request.session_id)
            session_id = memory.session_id
            
            memory.add_message("user", request.message)
            history = memory.build_context(include_metadata=False)
            
            result = rag_pipeline.answer_question(
                question=request.message,
                history=history[:-1],
                use_reranking=request.use_reranking,
                use_query_transformation=request.use_query_transformation
            )
            
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            confidence = result.get("confidence", "medium")
            key_points = result.get("key_points", [])
            
            memory.add_message(
                "assistant",
                answer,
                metadata={"sources": sources, "confidence": confidence}
            )
            
            words = answer.split()
            
            for i, word in enumerate(words):
                chunk_data = {
                    "chunk": word + (" " if i < len(words) - 1 else ""),
                    "session_id": session_id
                }
                if i == len(words) - 1:
                    chunk_data.update({
                        "sources": sources,
                        "confidence": confidence,
                        "key_points": key_points,
                        "complete": True
                    })
                
                yield json.dumps(chunk_data) + "\n"
                await asyncio.sleep(0.02)
        
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield json.dumps({"error": str(e)}) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )


@router.delete("/{session_id}")
async def clear_chat_history(
    session_id: str,
    api_key: str = Depends(verify_api_key),
    memory_manager: MemoryManager = Depends(get_memory_manager)
) -> Dict[str, str]:
    try:
        if memory_manager.clear_memory(session_id):
            return {"status": "success", "message": f"Cleared history for session {session_id}"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"message": f"Session {session_id} not found"}
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Failed to clear history: {str(e)}"}
        )