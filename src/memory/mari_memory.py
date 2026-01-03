"""
Mari's Long-term Memory System
Local persistent memory storage for Mari to remember important information
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading


class MariMemory:
    """Local memory storage for Mari"""
    
    def __init__(self, memory_dir: Optional[str] = None):
        if memory_dir:
            self.memory_dir = Path(memory_dir)
        else:
            self.memory_dir = Path(__file__).parent.parent.parent / "data" / "mari_memory"
        
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "memories.json"
        self.preferences_file = self.memory_dir / "preferences.json"
        self.conversation_file = self.memory_dir / "conversation_history.json"
        
        self._lock = threading.Lock()
        self._memories: List[Dict[str, Any]] = []
        self._preferences: Dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Load memories from disk"""
        with self._lock:
            if self.memory_file.exists():
                try:
                    self._memories = json.loads(self.memory_file.read_text(encoding="utf-8"))
                except Exception:
                    self._memories = []
            
            if self.preferences_file.exists():
                try:
                    self._preferences = json.loads(self.preferences_file.read_text(encoding="utf-8"))
                except Exception:
                    self._preferences = {}
    
    def _save(self) -> None:
        """Save memories to disk"""
        with self._lock:
            self.memory_file.write_text(
                json.dumps(self._memories, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            self.preferences_file.write_text(
                json.dumps(self._preferences, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
    
    def remember(self, content: str, category: str = "general", importance: int = 1) -> str:
        """
        Store a new memory
        
        Args:
            content: The content to remember
            category: Category (general, preference, fact, instruction)
            importance: 1-5, higher = more important
        
        Returns:
            Memory ID
        """
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._memories)}"
        memory = {
            "id": memory_id,
            "content": content,
            "category": category,
            "importance": importance,
            "created_at": datetime.now().isoformat(),
            "accessed_count": 0,
            "last_accessed": None
        }
        
        with self._lock:
            self._memories.append(memory)
        self._save()
        
        return memory_id
    
    def recall(self, query: str = "", category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recall memories matching query
        
        Args:
            query: Search query (simple substring match)
            category: Filter by category
            limit: Max number of results
        
        Returns:
            List of matching memories
        """
        results = []
        query_lower = query.lower()
        
        with self._lock:
            for mem in self._memories:
                if category and mem.get("category") != category:
                    continue
                if query and query_lower not in mem.get("content", "").lower():
                    continue
                results.append(mem)
        
        # Sort by importance and recency
        results.sort(key=lambda x: (x.get("importance", 0), x.get("created_at", "")), reverse=True)
        
        # Update access counts
        for mem in results[:limit]:
            mem["accessed_count"] = mem.get("accessed_count", 0) + 1
            mem["last_accessed"] = datetime.now().isoformat()
        
        self._save()
        return results[:limit]
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories"""
        with self._lock:
            return list(self._memories)
    
    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        with self._lock:
            for i, mem in enumerate(self._memories):
                if mem.get("id") == memory_id:
                    self._memories.pop(i)
                    self._save()
                    return True
        return False
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference"""
        with self._lock:
            self._preferences[key] = {
                "value": value,
                "updated_at": datetime.now().isoformat()
            }
        self._save()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference"""
        with self._lock:
            pref = self._preferences.get(key)
            if pref:
                return pref.get("value", default)
            return default
    
    def get_context_for_llm(self, limit: int = 5) -> str:
        """
        Get formatted memory context for LLM
        
        Returns:
            Formatted string with relevant memories
        """
        important_memories = self.recall(limit=limit)
        
        if not important_memories:
            return ""
        
        lines = ["[Mari's Memories]"]
        for mem in important_memories:
            cat = mem.get("category", "general")
            content = mem.get("content", "")
            lines.append(f"- [{cat}] {content}")
        
        prefs = []
        with self._lock:
            for k, v in self._preferences.items():
                prefs.append(f"- {k}: {v.get('value')}")
        
        if prefs:
            lines.append("\n[User Preferences]")
            lines.extend(prefs)
        
        return "\n".join(lines)
    
    def save_conversation(self, role: str, content: str) -> None:
        """Save a conversation turn"""
        history = []
        if self.conversation_file.exists():
            try:
                history = json.loads(self.conversation_file.read_text(encoding="utf-8"))
            except Exception:
                history = []
        
        history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep last 100 messages
        history = history[-100:]
        
        self.conversation_file.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def get_recent_conversation(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        if not self.conversation_file.exists():
            return []
        try:
            history = json.loads(self.conversation_file.read_text(encoding="utf-8"))
            return history[-limit:]
        except Exception:
            return []


# Global instance
_mari_memory: Optional[MariMemory] = None


def get_mari_memory() -> MariMemory:
    """Get the global Mari memory instance"""
    global _mari_memory
    if _mari_memory is None:
        _mari_memory = MariMemory()
    return _mari_memory


def parse_memory_command(user_input: str) -> Optional[Dict[str, Any]]:
    """
    Parse user input for memory commands
    
    Supported commands:
    - "记住: ..." or "remember: ..." - Save a memory
    - "忘记: ..." or "forget: ..." - Delete a memory
    - "回忆" or "memories" - List memories
    
    Returns:
        Dict with action and params, or None if not a command
    """
    input_lower = user_input.lower().strip()
    
    # Remember commands
    for prefix in ["记住:", "记住：", "remember:", "请记住:", "请记住："]:
        if input_lower.startswith(prefix):
            content = user_input[len(prefix):].strip()
            if content:
                return {"action": "remember", "content": content}
    
    # Forget commands
    for prefix in ["忘记:", "忘记：", "forget:", "删除记忆:"]:
        if input_lower.startswith(prefix):
            content = user_input[len(prefix):].strip()
            if content:
                return {"action": "forget", "query": content}
    
    # List memories
    if input_lower in ["回忆", "memories", "记忆", "我的记忆", "你记得什么"]:
        return {"action": "list"}
    
    return None
