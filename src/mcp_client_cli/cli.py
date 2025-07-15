#!/usr/bin/env python3

"""
Simple llm CLI that acts as MCP client.
"""

from datetime import datetime
import argparse
import asyncio
import os
from typing import Dict, Any
import uuid
import sys
import re
import base64
import mimetypes

# Replacement for deprecated imghdr module
def what(file, h=None):
    """Detect image format from file or header bytes."""
    if h is None:
        with open(file, 'rb') as f:
            h = f.read(32)
    else:
        h = h[:32]
    
    # Check for common image formats
    if h.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    elif h.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    elif h.startswith(b'GIF87a') or h.startswith(b'GIF89a'):
        return 'gif'
    elif h.startswith(b'BM'):
        return 'bmp'
    elif h.startswith(b'RIFF') and b'WEBP' in h:
        return 'webp'
    return None
from rich.console import Console
from rich.table import Table

from .input import *
from .const import *
from .output import *
from .simple_storage import SimpleStore, ConversationManager
from .tool import McpServerConfig, create_mcp_tool_manager
from mcp import StdioServerParameters
from .prompt import *
from .memory import get_memories
from .config import AppConfig
from .llm import create_llm, Message
from .agent import McpAgent, AgentState, AgentStep

async def run() -> None:
    """Run the LLM agent."""
    args = setup_argument_parser()
    query, is_conversation_continuation = parse_query(args)
    app_config = AppConfig.load()
    
    if args.list_tools:
        await handle_list_tools(app_config, args)
        return
    
    if args.show_memories:
        await handle_show_memories()
        return
        
    if args.list_prompts:
        handle_list_prompts()
        return
        
    await handle_conversation(args, query, is_conversation_continuation, app_config)

def setup_argument_parser() -> argparse.Namespace:
    """Setup and return the argument parser."""
    parser = argparse.ArgumentParser(
        description='Run MCP client with LLM tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm "What is the capital of France?"     Ask a simple question
  llm c "tell me more"                     Continue previous conversation
  llm p review                             Use a prompt template
  cat file.txt | llm                       Process input from a file
  llm --list-tools                         Show available tools
  llm --list-prompts                       Show available prompt templates
  llm --no-confirmations "search web"      Run tools without confirmation
        """
    )
    parser.add_argument('query', nargs='*', default=[],
                       help='The query to process (default: read from stdin). '
                            'Special prefixes:\n'
                            '  c: Continue previous conversation\n'
                            '  p: Use prompt template')
    parser.add_argument('--list-tools', action='store_true',
                       help='List all available LLM tools')
    parser.add_argument('--list-prompts', action='store_true',
                       help='List all available prompts')
    parser.add_argument('--no-confirmations', action='store_true',
                       help='Bypass tool confirmation requirements')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of tools capabilities')
    parser.add_argument('--text-only', action='store_true',
                       help='Print output as raw text instead of parsing markdown')
    parser.add_argument('--no-tools', action='store_true',
                       help='Do not add any tools')
    parser.add_argument('--no-intermediates', action='store_true',
                       help='Only print the final message')
    parser.add_argument('--show-memories', action='store_true',
                       help='Show user memories')
    parser.add_argument('--model',
                       help='Override the model specified in config')
    return parser.parse_args()

async def handle_list_tools(app_config: AppConfig, args: argparse.Namespace) -> None:
    """Handle the --list-tools command."""
    server_configs = [
        McpServerConfig(
            server_name=name,
            server_param=StdioServerParameters(
                command=config.command,
                args=config.args or [],
                env={**(config.env or {}), **os.environ}
            ),
            exclude_tools=config.exclude_tools or []
        )
        for name, config in app_config.get_enabled_servers().items()
    ]
    
    if not args.no_tools:
        tool_manager = await create_mcp_tool_manager(server_configs, args.force_refresh)
        
        console = Console()
        table = Table(title="Available LLM Tools")
        table.add_column("Server", style="cyan")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Description", style="green")

        for tool in tool_manager.get_all_tools():
            table.add_row(tool.server_name, tool.name, tool.description)

        # Add save_memory tool
        table.add_row("built-in", "save_memory", "Save a memory about the user for future conversations")
        
        console.print(table)
        await tool_manager.close_all()
    else:
        console = Console()
        console.print("No tools available (--no-tools flag used)")

async def handle_show_memories() -> None:
    """Handle the --show-memories command."""
    store = SimpleStore(SQLITE_DB)
    memories = await get_memories(store)
    console = Console()
    table = Table(title="My LLM Memories")
    table.add_column("Memory", style="green")
    for memory in memories:
        table.add_row(memory)
    console.print(table)

def handle_list_prompts() -> None:
    """Handle the --list-prompts command."""
    console = Console()
    table = Table(title="Available Prompt Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Template")
    table.add_column("Arguments")
    
    for name, template in prompt_templates.items():
        table.add_row(name, template, ", ".join(re.findall(r'\{(\w+)\}', template)))
        
    console.print(table)

async def handle_conversation(args: argparse.Namespace, query: Message, 
                            is_conversation_continuation: bool, app_config: AppConfig) -> None:
    """Handle the main conversation flow."""
    
    # Create tool manager
    tool_manager = None
    if not args.no_tools:
        server_configs = [
            McpServerConfig(
                server_name=name,
                server_param=StdioServerParameters(
                    command=config.command,
                    args=config.args or [],
                    env={**(config.env or {}), **os.environ}
                ),
                exclude_tools=config.exclude_tools or []
            )
            for name, config in app_config.get_enabled_servers().items()
        ]
        tool_manager = await create_mcp_tool_manager(server_configs, args.force_refresh)
    else:
        from .tool import McpToolManager
        tool_manager = McpToolManager()
    
    # Override model if specified in command line
    if args.model:
        app_config.llm.model = args.model
    
    # Create LLM instance
    llm = create_llm(
        provider=app_config.llm.provider,
        model=app_config.llm.model,
        api_key=app_config.llm.api_key,
        temperature=app_config.llm.temperature,
        base_url=app_config.llm.base_url
    )
    
    # Create confirmation callback
    def confirmation_callback(tool_name: str, arguments: Dict[str, Any]) -> bool:
        if args.no_confirmations:
            return False
        
        # Check if this tool requires confirmation from config
        for server_name, server_config in app_config.get_enabled_servers().items():
            if hasattr(server_config, 'requires_confirmation'):
                if tool_name in server_config.requires_confirmation:
                    # In a real implementation, we'd prompt the user here
                    # For now, we'll just show the tool call and proceed
                    console = Console()
                    console.print(f"[yellow]Tool confirmation required for {tool_name}[/yellow]")
                    console.print(f"Arguments: {arguments}")
                    return True
        return False
    
    # Handle conversation continuation
    conversation_manager = ConversationManager(SQLITE_DB)
    store = SimpleStore(SQLITE_DB)
    memories = await get_memories(store)
    formatted_memories = "\n".join(f"- {memory}" for memory in memories)
    
    # Create agent
    agent = McpAgent(
        llm=llm,
        tool_manager=tool_manager,
        system_prompt=app_config.system_prompt,
        confirmation_callback=confirmation_callback,
        memory_store=store
    )
    
    thread_id = (await conversation_manager.get_last_id() if is_conversation_continuation 
                else uuid.uuid4().hex)
    
    # Create agent state
    state = AgentState(
        messages=[query],
        today_datetime=datetime.now().isoformat(),
        memories=formatted_memories,
        remaining_steps=3
    )
    
    # Initialize output handler
    output = OutputHandler(text_only=args.text_only, only_last_message=args.no_intermediates)
    output.start()
    
    try:
        # Run the agent
        async for step in agent.run_conversation(state, confirmation_callback):
            # Convert agent steps to output format
            if step.step_type == "message":
                # Simulate the chunk format expected by OutputHandler
                chunk = {
                    "messages": [{"content": step.content, "type": "ai"}]
                }
                output.update(chunk)
            elif step.step_type == "tool_call":
                # Show tool calls if not hiding intermediates
                if not args.no_intermediates:
                    chunk = {
                        "messages": [{"content": f"🔧 {step.content}", "type": "ai"}]
                    }
                    output.update(chunk)
            elif step.step_type == "tool_result":
                # Show tool results if not hiding intermediates
                if not args.no_intermediates:
                    chunk = {
                        "messages": [{"content": f"📋 {step.content}", "type": "ai"}]
                    }
                    output.update(chunk)
            elif step.step_type == "error":
                output.update_error(Exception(step.content))
                break
                
    except Exception as e:
        output.update_error(e)
    finally:
        output.finish()
        
        # Save conversation thread (simplified since we don't have checkpointer)
        await conversation_manager.save_id(thread_id, None)
        
        # Cleanup
        if tool_manager:
            await tool_manager.close_all()

def parse_query(args: argparse.Namespace) -> tuple[Message, bool]:
    """
    Parse the query from command line arguments.
    Returns a tuple of (Message, is_conversation_continuation).
    """
    query_parts = ' '.join(args.query).split()
    stdin_content = ""
    stdin_image = None
    is_continuation = False

    # Handle clipboard content if requested
    if query_parts and query_parts[0] == 'cb':
        # Remove 'cb' from query parts
        query_parts = query_parts[1:]
        # Try to get content from clipboard
        clipboard_result = get_clipboard_content()
        if clipboard_result:
            content, mime_type = clipboard_result
            if mime_type:  # It's an image
                stdin_image = base64.b64encode(content).decode('utf-8')
            else:  # It's text
                stdin_content = content
        else:
            print("No content found in clipboard")
            raise Exception("Clipboard is empty")
    # Check if there's input from pipe
    elif not sys.stdin.isatty():
        stdin_data = sys.stdin.buffer.read()
        # Try to detect if it's an image
        image_type = what(None, h=stdin_data)
        if image_type:
            # It's an image, encode it as base64
            stdin_image = base64.b64encode(stdin_data).decode('utf-8')
            mime_type = mimetypes.guess_type(f"dummy.{image_type}")[0] or f"image/{image_type}"
        else:
            # It's text
            stdin_content = stdin_data.decode('utf-8').strip()

    # Process the query text
    query_text = ""
    if query_parts:
        if query_parts[0] == 'c':
            is_continuation = True
            query_text = ' '.join(query_parts[1:])
        elif query_parts[0] == 'p' and len(query_parts) >= 2:
            template_name = query_parts[1]
            if template_name not in prompt_templates:
                print(f"Error: Prompt template '{template_name}' not found.")
                print("Available templates:", ", ".join(prompt_templates.keys()))
                return Message(role="user", content=""), False

            template = prompt_templates[template_name]
            template_args = query_parts[2:]
            try:
                # Extract variable names from the template
                var_names = re.findall(r'\{(\w+)\}', template)
                # Create dict mapping parameter names to arguments
                template_vars = dict(zip(var_names, template_args))
                query_text = template.format(**template_vars)
            except KeyError as e:
                print(f"Error: Missing argument {e}")
                return Message(role="user", content=""), False
        else:
            query_text = ' '.join(query_parts)

    # Combine stdin content with query text if both exist
    if stdin_content and query_text:
        query_text = f"{stdin_content}\n\n{query_text}"
    elif stdin_content:
        query_text = stdin_content
    elif not query_text and not stdin_image:
        return Message(role="user", content=""), False

    # Create the message content
    if stdin_image:
        content = [
            {"type": "text", "text": query_text or "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{stdin_image}"}}
        ]
    else:
        content = query_text

    return Message(role="user", content=content), is_continuation

def main() -> None:
    """Entry point of the script."""
    asyncio.run(run())


if __name__ == "__main__":
    main()