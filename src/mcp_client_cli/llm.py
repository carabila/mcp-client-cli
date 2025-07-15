"""LLM abstraction layer for different providers."""

import json
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from pydantic import BaseModel
import httpx

class Message(BaseModel):
    """Represents a message in a conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: Union[str, List[Dict[str, Any]]]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ToolCall(BaseModel):
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]

class LLMResponse(BaseModel):
    """Response from an LLM."""
    content: str
    tool_calls: List[ToolCall] = []
    finish_reason: str = "stop"

class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.extra_params = kwargs

    @abstractmethod
    async def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def generate_stream(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response from the LLM."""
        pass

class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation."""
    
    def __init__(self, model: str, api_key: str, temperature: float = 0.7, **kwargs):
        super().__init__(model, temperature, **kwargs)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package is required for Anthropic LLM")

    def _convert_messages(self, messages: List[Message]) -> tuple[str, List[Dict]]:
        """Convert our message format to Anthropic's format."""
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                anthropic_messages.append({
                    "role": msg.role if msg.role != "tool" else "user",
                    "content": msg.content
                })
        
        return system_message, anthropic_messages

    def _convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict]]:
        """Convert tools to Anthropic's format."""
        if not tools:
            return None
        
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            })
        return anthropic_tools

    async def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        """Generate a response from Claude."""
        system_message, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        
        kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": 4096
        }
        
        if system_message:
            kwargs["system"] = system_message
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        
        response = await self.client.messages.create(**kwargs)
        
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input
                ))
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop"
        )

    async def generate_stream(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response from Claude."""
        system_message, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        
        kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": 4096
        }
        
        if system_message:
            kwargs["system"] = system_message
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        
        current_content = ""
        current_tool_calls = []
        
        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        current_content += event.delta.text
                        yield LLMResponse(content=current_content, tool_calls=current_tool_calls)
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_calls.append(ToolCall(
                            id=event.content_block.id,
                            name=event.content_block.name,
                            arguments={}
                        ))
                elif event.type == "content_block_stop":
                    if hasattr(event, 'content_block') and event.content_block.type == "tool_use":
                        # Update the last tool call with final arguments
                        if current_tool_calls:
                            current_tool_calls[-1].arguments = event.content_block.input

class OpenAILLM(BaseLLM):
    """OpenAI GPT LLM implementation."""
    
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None, temperature: float = 0.7, **kwargs):
        super().__init__(model, temperature, **kwargs)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("openai package is required for OpenAI LLM")

    def _convert_messages(self, messages: List[Message]) -> List[Dict]:
        """Convert our message format to OpenAI's format."""
        openai_messages = []
        for msg in messages:
            openai_msg = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            openai_messages.append(openai_msg)
        return openai_messages

    def _convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict]]:
        """Convert tools to OpenAI's format."""
        if not tools:
            return None
        
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })
        return openai_tools

    async def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        """Generate a response from OpenAI."""
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools)
        
        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature
        }
        
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"
        
        response = await self.client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content or ""
        tool_calls = []
        
        if response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop"
        )

    async def generate_stream(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response from OpenAI."""
        openai_messages = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools)
        
        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": True
        }
        
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"
        
        current_content = ""
        current_tool_calls = {}
        
        async for chunk in await self.client.chat.completions.create(**kwargs):
            choice = chunk.choices[0]
            
            if choice.delta.content:
                current_content += choice.delta.content
                yield LLMResponse(content=current_content)
            
            if choice.delta.tool_calls:
                for tc_delta in choice.delta.tool_calls:
                    if tc_delta.index not in current_tool_calls:
                        current_tool_calls[tc_delta.index] = {
                            "id": tc_delta.id,
                            "name": tc_delta.function.name if tc_delta.function else "",
                            "arguments": ""
                        }
                    
                    if tc_delta.function and tc_delta.function.arguments:
                        current_tool_calls[tc_delta.index]["arguments"] += tc_delta.function.arguments

class GoogleLLM(BaseLLM):
    """Google Gemini LLM implementation."""
    
    def __init__(self, model: str, api_key: str, temperature: float = 0.7, **kwargs):
        super().__init__(model, temperature, **kwargs)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai package is required for Google LLM")

    async def generate(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        """Generate a response from Gemini."""
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            if msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
            elif msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
        
        prompt = "\n".join(prompt_parts)
        
        # Note: Google's library might not have full async support, 
        # this is a simplified implementation
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.client.generate_content, prompt
        )
        
        return LLMResponse(
            content=response.text,
            tool_calls=[],  # Tool calls not implemented for Google yet
            finish_reason="stop"
        )

    async def generate_stream(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response from Gemini."""
        # Simplified implementation - would need proper streaming support
        response = await self.generate(messages, tools)
        yield response

def create_llm(provider: str, model: str, api_key: str, temperature: float = 0.7, 
               base_url: Optional[str] = None, **kwargs) -> BaseLLM:
    """Factory function to create an LLM instance."""
    if provider.lower() == "anthropic":
        return AnthropicLLM(model, api_key, temperature, **kwargs)
    elif provider.lower() == "openai":
        return OpenAILLM(model, api_key, base_url, temperature, **kwargs)
    elif provider.lower() == "google":
        return GoogleLLM(model, api_key, temperature, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")