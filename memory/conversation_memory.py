"""Conversation memory management."""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manage conversation history with configurable window."""
    
    def __init__(self, max_messages: int = None):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to retain
        """
        self.max_messages = max_messages or settings.max_memory_messages
        self.messages: List[Dict[str, Any]] = []
        self.session_id: str = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Additional metadata (sources, confidence, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        self.last_updated = datetime.now()
        
        # Trim if exceeding max
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        logger.debug(f"Added {role} message (total: {len(self.messages)})")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get all conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.messages.copy()
    
    def get_recent_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            n: Number of recent messages to return
            
        Returns:
            List of recent message dictionaries
        """
        n = n or self.max_messages
        return self.messages[-n:] if self.messages else []
    
    def get_user_messages(self) -> List[str]:
        """
        Get all user messages.
        
        Returns:
            List of user message contents
        """
        return [msg["content"] for msg in self.messages if msg["role"] == "user"]
    
    def get_assistant_messages(self) -> List[str]:
        """
        Get all assistant messages.
        
        Returns:
            List of assistant message contents
        """
        return [msg["content"] for msg in self.messages if msg["role"] == "assistant"]
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []
        self.last_updated = datetime.now()
        logger.info("Conversation memory cleared")
    
    def build_context(self, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Build context for LLM prompt.
        
        Args:
            include_metadata: Whether to include metadata in context
            
        Returns:
            List of message dictionaries formatted for LLM
        """
        context = []
        for msg in self.messages:
            entry = {
                "role": msg["role"],
                "content": msg["content"]
            }
            if include_metadata and msg.get("metadata"):
                entry["metadata"] = msg["metadata"]
            context.append(entry)
        return context
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "user_messages": len([m for m in self.messages if m["role"] == "user"]),
            "assistant_messages": len([m for m in self.messages if m["role"] == "assistant"]),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "max_messages": self.max_messages
        }
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message in the conversation.
        
        Returns:
            Last message or None
        """
        return self.messages[-1] if self.messages else None
    
    def get_last_user_message(self) -> Optional[str]:
        """
        Get the last user message.
        
        Returns:
            Last user message content or None
        """
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert memory to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "max_messages": self.max_messages
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """
        Create memory from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ConversationMemory instance
        """
        memory = cls(max_messages=data.get("max_messages", settings.max_memory_messages))
        memory.session_id = data.get("session_id", str(uuid.uuid4()))
        memory.messages = data.get("messages", [])
        memory.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        memory.last_updated = datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now()
        return memory


class MemoryManager:
    """Manage multiple conversation memories."""
    
    def __init__(self):
        """Initialize the memory manager."""
        self.memories: Dict[str, ConversationMemory] = {}
        logger.info("Memory manager initialized")
    
    def get_or_create_memory(self, session_id: Optional[str] = None) -> ConversationMemory:
        """
        Get existing memory or create a new one.
        
        Args:
            session_id: Session ID for the conversation
            
        Returns:
            ConversationMemory instance
        """
        if session_id and session_id in self.memories:
            return self.memories[session_id]
        
        memory = ConversationMemory()
        if session_id:
            memory.session_id = session_id
        self.memories[memory.session_id] = memory
        return memory
    
    def clear_memory(self, session_id: str) -> bool:
        """
        Clear a specific conversation memory.
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            True if cleared, False if not found
        """
        if session_id in self.memories:
            self.memories[session_id].clear()
            return True
        return False
    
    def delete_memory(self, session_id: str) -> bool:
        """
        Delete a specific conversation memory.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.memories:
            del self.memories[session_id]
            return True
        return False
    
    def get_all_memories(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summaries of all memories.
        
        Returns:
            Dictionary mapping session IDs to summaries
        """
        return {
            session_id: memory.get_summary()
            for session_id, memory in self.memories.items()
        }
    
    def cleanup_old_memories(self, max_age_hours: int = 24) -> int:
        """
        Remove memories older than max_age_hours.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of memories cleaned up
        """
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_delete = []
        
        for session_id, memory in self.memories.items():
            if memory.last_updated < cutoff:
                to_delete.append(session_id)
        
        for session_id in to_delete:
            del self.memories[session_id]
        
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old memories")
        
        return len(to_delete)