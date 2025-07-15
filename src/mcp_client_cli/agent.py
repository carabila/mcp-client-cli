"""Agent implementation for tool-using conversations."""

import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from datetime import datetime
import uuid
from pydantic import BaseModel

from .llm import BaseLLM, Message, LLMResponse, ToolCall
from .tool import McpToolManager, McpTool, McpToolResult
from .simple_storage import SimpleStore

class AgentState(BaseModel):
    """State of the agent during conversation."""
    messages: List[Message]
    today_datetime: str
    memories: str = "no memories"
    remaining_steps: int = 5
    
class AgentStep(BaseModel):
    """Represents a single step in the agent's execution."""
    step_type: str  # "message", "tool_call", "tool_result", "error"
    content: str = ""
    tool_calls: List[ToolCall] = []
    tool_results: List[Dict[str, Any]] = []
    is_final: bool = False

class McpAgent:
    """Agent that can use MCP tools to assist with user queries."""
    
    def __init__(self, 
                 llm: BaseLLM, 
                 tool_manager: McpToolManager,
                 system_prompt: str = "",
                 max_iterations: int = 10,
                 confirmation_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
                 memory_store: Optional[SimpleStore] = None):
        self.llm = llm
        self.tool_manager = tool_manager
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.confirmation_callback = confirmation_callback
        self.memory_store = memory_store

    def _create_tool_schemas(self) -> List[Dict[str, Any]]:
        """Create tool schemas for the LLM."""
        tools = self.tool_manager.get_all_tools()
        tool_schemas = []
        
        for tool in tools:
            tool_schemas.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            })
        
        # Add save_memory tool
        tool_schemas.append({
            "name": "save_memory",
            "description": "Save a memory about the user for future conversations. Use this to remember important information about the user's preferences, context, or ongoing projects.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string",
                        "description": "The memory to save about the user"
                    }
                },
                "required": ["memory"]
            }
        })
        
        return tool_schemas

    def _should_confirm_tool(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Check if a tool call should be confirmed by the user."""
        if not self.confirmation_callback:
            return False
        return self.confirmation_callback(tool_name, arguments)

    async def _execute_tool_call(self, tool_call: ToolCall) -> McpToolResult:
        """Execute a single tool call."""
        if tool_call.name == "save_memory":
            # Handle memory saving locally
            try:
                memory = tool_call.arguments.get("memory", "")
                if memory and self.memory_store:
                    result = await self.memory_store.save_memory(memory)
                    return McpToolResult(content=result)
                else:
                    return McpToolResult(content="No memory provided or memory store not available", is_error=True)
            except Exception as e:
                return McpToolResult(content=f"Error saving memory: {e}", is_error=True)
        else:
            # Execute MCP tool
            return await self.tool_manager.execute_tool(tool_call.name, tool_call.arguments)

    async def run(self, user_message: str, memories: str = "") -> AsyncGenerator[AgentStep, None]:
        """Run the agent with a user message."""
        # Initialize conversation
        messages = []
        
        if self.system_prompt:
            system_content = self.system_prompt
            if memories:
                system_content += f"\n\nUser memories:\n{memories}"
            messages.append(Message(role="system", content=system_content))
        
        messages.append(Message(role="user", content=user_message))
        
        tool_schemas = self._create_tool_schemas()
        
        for iteration in range(self.max_iterations):
            # Get LLM response
            try:
                response = await self.llm.generate(messages, tool_schemas)
                
                # Yield the message content if any
                if response.content:
                    yield AgentStep(
                        step_type="message",
                        content=response.content,
                        is_final=not response.tool_calls
                    )
                
                # If no tool calls, we're done
                if not response.tool_calls:
                    break
                
                # Process tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    # Check if confirmation is needed
                    if self._should_confirm_tool(tool_call.name, tool_call.arguments):
                        yield AgentStep(
                            step_type="tool_call",
                            content=f"About to call tool '{tool_call.name}' with arguments: {json.dumps(tool_call.arguments, indent=2)}",
                            tool_calls=[tool_call]
                        )
                        # In a real implementation, we'd wait for user confirmation here
                        # For now, assume confirmation is given
                    
                    # Execute the tool
                    try:
                        result = await self._execute_tool_call(tool_call)
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "result": result.content,
                            "is_error": result.is_error
                        })
                        
                        yield AgentStep(
                            step_type="tool_result",
                            content=f"Tool '{tool_call.name}' result: {result.content}",
                            tool_results=[{
                                "name": tool_call.name,
                                "result": result.content,
                                "is_error": result.is_error
                            }]
                        )
                        
                    except Exception as e:
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "result": f"Error executing tool: {e}",
                            "is_error": True
                        })
                        
                        yield AgentStep(
                            step_type="error",
                            content=f"Error executing tool '{tool_call.name}': {e}"
                        )
                
                # Add assistant message with tool calls
                messages.append(Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=[{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    } for tc in response.tool_calls]
                ))
                
                # Add tool results as messages
                for result in tool_results:
                    messages.append(Message(
                        role="tool",
                        content=result["result"],
                        tool_call_id=result["tool_call_id"]
                    ))
                
            except Exception as e:
                yield AgentStep(
                    step_type="error",
                    content=f"Error in agent execution: {e}",
                    is_final=True
                )
                break

    async def run_conversation(self, 
                             state: AgentState,
                             confirmation_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None) -> AsyncGenerator[AgentStep, None]:
        """Run a conversation with the given state."""
        if confirmation_callback:
            self.confirmation_callback = confirmation_callback
            
        # Extract the last user message
        user_messages = [msg for msg in state.messages if msg.role == "user"]
        if not user_messages:
            yield AgentStep(
                step_type="error",
                content="No user message found in conversation state",
                is_final=True
            )
            return
        
        last_user_message = user_messages[-1].content
        if isinstance(last_user_message, list):
            # Handle multimodal content (text + images)
            text_content = ""
            for item in last_user_message:
                if item.get("type") == "text":
                    text_content = item.get("text", "")
                    break
            last_user_message = text_content
        
        async for step in self.run(str(last_user_message), state.memories):
            yield step