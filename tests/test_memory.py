"""Conversation memory tests."""

import pytest
from memory.conversation_memory import ConversationMemory, MemoryManager


class TestConversationMemory:
    """Test conversation memory."""
    
    def test_add_message(self):
        memory = ConversationMemory(max_messages=5)
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there")
        
        history = memory.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
    
    def test_max_messages_limit(self):
        memory = ConversationMemory(max_messages=3)
        for i in range(5):
            memory.add_message("user", f"Message {i}")
        
        assert len(memory.get_history()) == 3
        assert memory.get_history()[0]["content"] == "Message 2"
    
    def test_get_recent_history(self):
        memory = ConversationMemory()
        memory.add_message("user", "First")
        memory.add_message("user", "Second")
        memory.add_message("user", "Third")
        
        recent = memory.get_recent_history(2)
        assert len(recent) == 2
        assert recent[0]["content"] == "Second"
        assert recent[1]["content"] == "Third"
    
    def test_clear(self):
        memory = ConversationMemory()
        memory.add_message("user", "Hello")
        memory.clear()
        assert len(memory.get_history()) == 0
    
    def test_build_context(self):
        memory = ConversationMemory()
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi")
        
        context = memory.build_context()
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"
    
    def test_get_summary(self):
        memory = ConversationMemory()
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi")
        memory.add_message("user", "How are you?")
        
        summary = memory.get_summary()
        assert summary["total_messages"] == 3
        assert summary["user_messages"] == 2
        assert summary["assistant_messages"] == 1
        assert "session_id" in summary
    
    def test_serialization(self):
        memory = ConversationMemory()
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi")
        
        data = memory.to_dict()
        assert data["session_id"] == memory.session_id
        assert len(data["messages"]) == 2
        
        restored = ConversationMemory.from_dict(data)
        assert restored.session_id == memory.session_id
        assert len(restored.get_history()) == 2


class TestMemoryManager:
    """Test memory manager."""
    
    def test_get_or_create_memory(self):
        manager = MemoryManager()
        
        memory1 = manager.get_or_create_memory("session1")
        memory2 = manager.get_or_create_memory("session1")
        
        assert memory1.session_id == memory2.session_id
        assert memory1 is memory2
    
    def test_create_new_session(self):
        manager = MemoryManager()
        
        memory1 = manager.get_or_create_memory("session1")
        memory2 = manager.get_or_create_memory("session2")
        
        assert memory1.session_id != memory2.session_id
        assert len(manager.memories) == 2
    
    def test_clear_memory(self):
        manager = MemoryManager()
        memory = manager.get_or_create_memory("session1")
        memory.add_message("user", "Hello")
        
        assert manager.clear_memory("session1")
        assert len(memory.get_history()) == 0
    
    def test_delete_memory(self):
        manager = MemoryManager()
        manager.get_or_create_memory("session1")
        
        assert manager.delete_memory("session1")
        assert "session1" not in manager.memories
    
    def test_get_all_memories(self):
        manager = MemoryManager()
        manager.get_or_create_memory("session1")
        manager.get_or_create_memory("session2")
        
        all_memories = manager.get_all_memories()
        assert len(all_memories) == 2
        assert "session1" in all_memories
        assert "session2" in all_memories