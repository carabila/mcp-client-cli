# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python CLI application that implements a Model Context Protocol (MCP) client for LLM interactions. It allows users to run LLM prompts while leveraging MCP-compatible servers for enhanced functionality like web search, file operations, and more.

## Core Architecture

### Main Components
- **CLI Interface** (`cli.py`): Main entry point handling command-line arguments and orchestrating LLM interactions
- **Configuration Management** (`config.py`): Handles JSON-based configuration for LLM providers and MCP servers
- **Tool Integration** (`tool.py`): Manages MCP server connections and tool execution
- **Memory System** (`memory.py`, `storage.py`): Provides conversation continuity and user memory
- **Input/Output** (`input.py`, `output.py`): Handles various input sources (clipboard, stdin, files) and formatted output
- **Prompt Templates** (`prompt.py`): Manages predefined prompt templates

### Technology Stack
- **Python 3.12+** with modern async/await patterns
- **LangChain** for LLM orchestration and tool integration
- **Model Context Protocol (MCP)** for tool server communication
- **Rich** for terminal output formatting
- **CommentJSON** for configuration file parsing

## Development Commands

### Installation
```bash
# Install for development with clipboard support
pipx install -e ".[clipboard]"

# Standard installation
pip install mcp-client-cli
```

### Running the Application
```bash
# Basic usage
llm "Your prompt here"

# Development entry point
python -m mcp_client_cli.cli "Your prompt here"
```

### Configuration
- Configuration file locations: `~/.llm/config.json` or `$PWD/.llm/config.json`
- See `CONFIG.md` for complete configuration documentation
- Use `mcp-server-config-example.json` as a template

## Key Features to Understand

### MCP Server Integration
The application connects to external MCP servers for extended functionality. Each server runs as a separate process and communicates via JSON-RPC over stdio.

### Memory and Continuity
- Uses SQLite for conversation storage
- Supports conversation continuation with `c ` prefix
- Implements user memory for personalization

### Input Modes
- Direct command-line arguments
- Stdin piping (text and images)
- Clipboard integration (`cb` command)
- Prompt templates (`p template_name`)

### Tool Confirmation System
Some tools require user confirmation before execution, configured via `requires_confirmation` in the MCP server config.

## Code Style Guidelines

### From .cursorrule
- Use modern Python 3.12+ features with proper type annotations
- Implement async/await patterns for I/O operations
- Use Pydantic models for data validation
- Follow modular design for testability
- Cache expensive operations appropriately
- Maintain backward compatibility with existing MCP tools

### Key Principles
- Separate concerns between tool management, LLM interaction, and CLI interface
- Implement proper error handling for LLM and tool interactions
- Use clear docstrings and maintain API documentation
- Handle sensitive data through proper environment configuration