import asyncio
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentType(str, Enum):
    """Agent execution type."""
    LLM = "llm"           # Standard LLM agent (default)
    SEQUENTIAL = "sequential"  # Chain of agents
    TOOL = "tool"         # Direct tool execution


class StopProcessing(Exception):
    """
    Raise this in middleware to gracefully stop agent execution and return a custom response.
    
    Example:
        async def before(content, context: dict):
            # Check first text part for authentication
            if content.parts and content.parts[0].text:
                if "api_key" not in content.parts[0].text:
                    raise StopProcessing("Authentication required: missing api_key")
            return content
    
    The message will be returned as the response without running the agent.
    """
    def __init__(self, response: str):
        self.response = response
        super().__init__(response)


class Middleware(BaseModel):
    """
    Middleware functions that run before and after agent execution.
    
    Middleware files are Python modules in the middleware/ directory,
    named after the agent they apply to (e.g., middleware/assistant.py).
    
    The 'before' middleware receives a dict representing the user message,
    allowing you to inspect/modify the content and attach documents or images.
    
    Example middleware file:
        async def before(content: dict, context: dict) -> dict:
            # content = {"role": "user", "parts": [...]}
            # Each part is either {"text": "..."} or {"data": bytes, "mime_type": "..."}
            pdf_bytes = open("context.pdf", "rb").read()
            content["parts"].append({"data": pdf_bytes, "mime_type": "application/pdf"})
            return content
        
        async def after(response: str, context: dict) -> str:
            # Process response after agent completes
            return response
    """
    before: Optional[Callable[..., Any]] = None
    after: Optional[Callable[..., Any]] = None
    
    class Config:
        arbitrary_types_allowed = True


class RetryOptions(BaseModel):
    """Configuration for automatic retries on failures."""
    attempts: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts (max: 10)")
    max_delay: int = Field(default=30, ge=1, le=300, description="Maximum seconds between retries (max: 300s)")


class McpServerConfig(BaseModel):
    """
    Configuration for an MCP (Model Context Protocol) server connection.
    
    MCP servers provide external tools that agents can use. The connection
    is established from within the deployed agent container via HTTP/SSE.
    
    Example YAML:
        mcp_servers:
          - name: filesystem
            url: https://mcp.example.com/filesystem
            tools:
              - read_file
              - write_file
            headers:
              Authorization: "Bearer ${API_TOKEN}"
    """
    name: str = Field(..., description="Unique identifier for this MCP server connection")
    url: str = Field(..., description="URL of the MCP server endpoint")
    tools: Optional[List[str]] = Field(
        default=None, 
        description="Optional list of tool names to include. If omitted, all tools from the server are available."
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional HTTP headers for authentication (supports ${VAR} substitution)"
    )


class Tool(BaseModel):
    """
    Represents a tool that an agent can use.
    In the SDK, this wraps a Python function.
    
    For predefined tools (is_predefined=True), the func may be None
    and will be injected by the runner at build time.
    """
    name: str
    description: str = ""
    func: Optional[Callable[..., Any]] = None  # None for predefined tools until injection
    is_async: bool = False
    
    # JSON Schema for LLM function calling
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # True for tools that will be injected by runner (trigger_agent, query_knowledge, etc.)
    is_predefined: bool = False

    class Config:
        arbitrary_types_allowed = True
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool, handling both sync and async functions."""
        if self.func is None:
            raise ValueError(f"Tool '{self.name}' has no function. Predefined tools must be injected by runner.")
        if self.is_async:
            return await self.func(**kwargs)
        else:
            # Run sync function in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.func(**kwargs))
    
    def execute_sync(self, **kwargs) -> Any:
        """Execute the tool synchronously."""
        if self.func is None:
            raise ValueError(f"Tool '{self.name}' has no function. Predefined tools must be injected by runner.")
        if self.is_async:
            return asyncio.run(self.func(**kwargs))
        return self.func(**kwargs)


class AgentConfig(BaseModel):
    """Configuration for an Agent as defined in YAML."""
    # Required fields
    version: str = Field(..., description="Configuration schema version. Currently only '1.0' is supported.")
    name: str = Field(..., description="Unique identifier for the agent")
    description: str = Field(..., description="Human-readable description of what the agent does")
    
    # Agent type - determines execution mode
    type: AgentType = Field(default=AgentType.LLM, description="Agent execution type")
    
    # LLM agent fields (required when type=llm)
    model: Optional[str] = Field(default=None, description="The AI model to use (required for LLM agents)")
    system_prompt: Optional[str] = Field(default=None, description="Instructions that define the agent's behavior")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Controls randomness in output")
    tools: List[str] = Field(default_factory=list, description="List of tools the agent can use")
    
    # Sequential agent fields (required when type=sequential)
    agents: List[str] = Field(default_factory=list, description="List of agent names to execute in sequence")
    
    # Tool agent fields (required when type=tool)
    tool_name: Optional[str] = Field(default=None, description="Tool to execute for tool-type agents")
    
    # MCP server connections
    mcp_servers: List[McpServerConfig] = Field(
        default_factory=list, 
        description="List of MCP servers to connect to for external tools (max: 50 per agent)",
        max_length=50
    )
    
    # Output schema for structured output (LLM agents only)
    output_schema: Optional[str] = Field(
        default=None, 
        description="Name of the output schema file (without .json extension) from schemas/ directory. LLM agents only."
    )
    
    # Runtime field - populated by loader, not from YAML
    output_schema_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The loaded JSON Schema dict (populated at runtime by loader)"
    )
    
    # Common optional fields
    max_concurrent_runs: int = Field(default=1, ge=1, description="Maximum simultaneous runs allowed")
    retry_options: Optional[RetryOptions] = Field(default=None, description="Configuration for automatic retries")
    timeout: Optional[int] = Field(default=None, ge=5, description="Max execution time in seconds (min: 5). Capped by subscription.")
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v: str) -> str:
        if v != "1.0":
            raise ValueError(f"Unsupported version '{v}'. Currently only '1.0' is supported.")
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        import re
        if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$', v):
            raise ValueError(
                f"Invalid agent name '{v}'. Use lowercase letters, numbers, and hyphens. "
                "Must start and end with a letter or number."
            )
        return v
    
    @field_validator('mcp_servers')
    @classmethod
    def validate_mcp_servers_limit(cls, v: List[McpServerConfig]) -> List[McpServerConfig]:
        """Validate MCP servers limit (max 50 per agent)."""
        if len(v) > 50:
            raise ValueError(
                f"Too many MCP servers: {len(v)} configured (max: 50 per agent). "
                "Consider consolidating servers or removing unused ones."
            )
        return v
    
    @model_validator(mode='after')
    def validate_type_requirements(self):
        """Validate that required fields are present based on agent type."""
        if self.type == AgentType.LLM:
            if not self.model:
                raise ValueError("LLM agents require 'model' to be specified")
            if not self.system_prompt:
                raise ValueError("LLM agents require 'system_prompt' to be specified")
        elif self.type == AgentType.SEQUENTIAL:
            if not self.agents:
                raise ValueError("Sequential agents require 'agents' list to be defined")
        elif self.type == AgentType.TOOL:
            if not self.tool_name:
                raise ValueError("Tool agents require 'tool_name' to be specified")
        return self


class Agent(BaseModel):
    """
    The runtime representation of an Agent.
    """
    config: AgentConfig
    tools: List[Tool] = Field(default_factory=list)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get the JSON schema for all tools, formatted for LLM function calling."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools
        ]
