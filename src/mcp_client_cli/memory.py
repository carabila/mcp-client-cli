"""Memory management for the MCP client CLI."""

from typing import List
from .simple_storage import SimpleStore

async def save_memory_function(memory: str, store: SimpleStore, user_id: str = "myself") -> str:
    """Save a memory for the current user."""
    return await store.save_memory(memory, user_id)

async def get_memories(store: SimpleStore, user_id: str = "myself") -> List[str]:
    """Get memories for a user."""
    return await store.get_memories(user_id)