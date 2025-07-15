"""Simple SQLite storage for memories and conversation history."""

import aiosqlite
import json
import uuid
from datetime import datetime
from typing import List, Optional
from pathlib import Path

class SimpleStore:
    """Simple SQLite-based storage for memories and conversation history."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def init_db(self):
        """Initialize the database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    thread_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            await db.commit()
    
    async def save_memory(self, content: str, user_id: str = "myself") -> str:
        """Save a memory for a user."""
        await self.init_db()
        memory_id = uuid.uuid4().hex
        created_at = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO memories (id, user_id, content, created_at) VALUES (?, ?, ?, ?)",
                (memory_id, user_id, content, created_at)
            )
            await db.commit()
        
        return f"Saved memory: {content}"
    
    async def get_memories(self, user_id: str = "myself") -> List[str]:
        """Get all memories for a user."""
        await self.init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT content FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
    
    async def save_conversation(self, thread_id: str):
        """Save or update a conversation thread."""
        await self.init_db()
        now = datetime.now().isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO conversations (thread_id, created_at, updated_at) 
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET updated_at = ?
            """, (thread_id, now, now, now))
            await db.commit()
    
    async def get_last_conversation_id(self) -> Optional[str]:
        """Get the most recent conversation thread ID."""
        await self.init_db()
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT thread_id FROM conversations ORDER BY updated_at DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None

class ConversationManager:
    """Manages conversation persistence."""
    
    def __init__(self, db_path: str):
        self.store = SimpleStore(db_path)
    
    async def save_id(self, thread_id: str, conn=None):
        """Save a conversation thread ID."""
        await self.store.save_conversation(thread_id)
    
    async def get_last_id(self) -> Optional[str]:
        """Get the last conversation thread ID."""
        return await self.store.get_last_conversation_id()