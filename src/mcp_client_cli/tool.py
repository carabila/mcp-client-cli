from typing import List, Optional, Any, Dict
from pydantic import BaseModel
from mcp import StdioServerParameters, types, ClientSession
from mcp.client.stdio import stdio_client
import pydantic
from pydantic_core import to_json
import asyncio
import json

# Tool caching functionality will be implemented later
def get_cached_tools(server_param):
    return None

def save_tools_cache(server_param, tools):
    pass

class McpServerConfig(BaseModel):
    """Configuration for an MCP server.
    
    This class represents the configuration needed to connect to and identify an MCP server,
    containing both the server's name and its connection parameters.

    Attributes:
        server_name (str): The name identifier for this MCP server
        server_param (StdioServerParameters): Connection parameters for the server, including
            command, arguments and environment variables
        exclude_tools (list[str]): List of tool names to exclude from this server
    """
    
    server_name: str
    server_param: StdioServerParameters
    exclude_tools: list[str] = []

class McpToolResult(BaseModel):
    """Result from executing an MCP tool."""
    content: str
    is_error: bool = False
    error_message: Optional[str] = None

class McpTool(BaseModel):
    """Represents an MCP tool that can be executed."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

class McpClient:
    """Client for managing MCP server connections and tool execution."""
    
    def __init__(self, server_config: McpServerConfig):
        self.server_config = server_config
        self._session: Optional[ClientSession] = None
        self._client = None
        self._tools: List[McpTool] = []
        self._init_lock = asyncio.Lock()

    async def _start_session(self):
        """Start the MCP client session."""
        async with self._init_lock:
            if self._session:
                return self._session

            self._client = stdio_client(self.server_config.server_param)
            read, write = await self._client.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
            return self._session

    async def initialize(self, force_refresh: bool = False):
        """Initialize the client and load available tools."""
        if self._tools and not force_refresh:
            return

        cached_tools = get_cached_tools(self.server_config.server_param)
        if cached_tools and not force_refresh:
            for tool in cached_tools:
                if tool.name in self.server_config.exclude_tools:
                    continue
                self._tools.append(McpTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    server_name=self.server_config.server_name
                ))
            return

        try:
            await self._start_session()
            tools_result: types.ListToolsResult = await self._session.list_tools()
            save_tools_cache(self.server_config.server_param, tools_result.tools)
            
            for tool in tools_result.tools:
                if tool.name in self.server_config.exclude_tools:
                    continue
                self._tools.append(McpTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    server_name=self.server_config.server_name
                ))
        except Exception as e:
            print(f"Error gathering tools for {self.server_config.server_param.command} {' '.join(self.server_config.server_param.args)}: {e}")
            raise e

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> McpToolResult:
        """Execute a specific tool with the given arguments."""
        if not self._session:
            await self._start_session()

        try:
            result = await self._session.call_tool(tool_name, arguments=arguments)
            content = to_json(result.content).decode()
            
            if result.isError:
                return McpToolResult(
                    content=content,
                    is_error=True,
                    error_message=content
                )
            
            return McpToolResult(content=content)
            
        except Exception as e:
            return McpToolResult(
                content=str(e),
                is_error=True,
                error_message=str(e)
            )

    async def close(self):
        """Close the MCP client session."""
        try:
            if self._session:
                try:
                    await self._session.__aexit__(None, None, None)
                except Exception:
                    pass
                finally:
                    self._session = None
        except:
            pass
        
        try:
            if self._client:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception:
                    pass
                finally:
                    self._client = None
        except:
            pass

    def get_tools(self) -> List[McpTool]:
        """Get the list of available tools."""
        return self._tools

class McpToolManager:
    """Manages multiple MCP clients and provides unified tool access."""
    
    def __init__(self):
        self.clients: List[McpClient] = []
        self._tools_by_name: Dict[str, tuple[McpClient, McpTool]] = {}

    async def add_server(self, server_config: McpServerConfig, force_refresh: bool = False):
        """Add an MCP server to the manager."""
        client = McpClient(server_config)
        await client.initialize(force_refresh=force_refresh)
        self.clients.append(client)
        
        # Index tools by name for quick lookup
        for tool in client.get_tools():
            self._tools_by_name[tool.name] = (client, tool)

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> McpToolResult:
        """Execute a tool by name."""
        if tool_name not in self._tools_by_name:
            return McpToolResult(
                content=f"Tool '{tool_name}' not found",
                is_error=True,
                error_message=f"Tool '{tool_name}' not found"
            )
        
        client, tool = self._tools_by_name[tool_name]
        return await client.execute_tool(tool_name, arguments)

    def get_all_tools(self) -> List[McpTool]:
        """Get all available tools from all clients."""
        all_tools = []
        for client in self.clients:
            all_tools.extend(client.get_tools())
        return all_tools

    def get_tool_by_name(self, tool_name: str) -> Optional[McpTool]:
        """Get a specific tool by name."""
        if tool_name in self._tools_by_name:
            return self._tools_by_name[tool_name][1]
        return None

    async def close_all(self):
        """Close all MCP client connections."""
        for client in self.clients:
            await client.close()

async def create_mcp_tool_manager(server_configs: List[McpServerConfig], force_refresh: bool = False) -> McpToolManager:
    """Create and initialize an MCP tool manager with the given server configurations."""
    manager = McpToolManager()
    
    for server_config in server_configs:
        try:
            await manager.add_server(server_config, force_refresh=force_refresh)
        except Exception as e:
            print(f"Failed to add server {server_config.server_name}: {e}")
    
    return manager