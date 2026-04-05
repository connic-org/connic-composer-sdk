import asyncio
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentType(str, Enum):
    """Agent execution type."""
    LLM = "llm"           # Standard LLM agent (default)
    SEQUENTIAL = "sequential"  # Chain of agents
    TOOL = "tool"         # Direct tool execution


class StopProcessing(Exception):
    """
    Raise this in middleware or in tools to gracefully stop the run
    and return a custom response string. The run is marked completed, not failed.

    Middleware example:
        async def before(content, context: dict):
            if not _is_authenticated(content):
                raise StopProcessing("Authentication required")
            return content

    Tool example:
        from connic import StopProcessing

        def refund_order(order_id: str, context: dict) -> str:
            if not context.get("refunds_enabled"):
                raise StopProcessing("Refunds are disabled for this account")
            return "Refund processed"
    """
    def __init__(self, response: str):
        self.response = response
        super().__init__(response)


class AbortTool(Exception):
    """
    Raise this in a tool hook's before() to skip tool execution and return
    a custom result. Unlike StopProcessing (which aborts the entire run),
    AbortTool only skips the current tool call. The returned result is passed
    back to the LLM as if the tool had executed normally, but the tool
    execution is marked as an error in traces.

    Hook example:
        from connic import AbortTool

        async def before(tool_name: str, params: dict, context: dict) -> dict:
            if tool_name == "db_delete" and not context.get("is_admin"):
                raise AbortTool({"error": "Only admins can delete records"})
            return params
    """
    def __init__(self, result: str | dict):
        self.result = result
        super().__init__(str(result))


class Middleware(BaseModel):
    """
    Middleware functions that run before and after agent execution.
    
    Middleware files are Python modules in the middleware/ directory,
    named after the agent they apply to (e.g., middleware/assistant.py).
    
    The 'before' middleware receives a dict representing the user message
    and a shared context dict. You can inspect/modify the content and attach
    documents or images. The context dict is pre-populated with system metadata
    (run_id, agent_name, connector_id, timestamp) and you can add your own
    values. Values set on context are available in prompts via {var} syntax
    and in tools that declare a ``context`` parameter.
    
    Example middleware file:
        async def before(content: dict, context: dict) -> dict:
            # content = {"role": "user", "parts": [...]}
            # Each part is either {"text": "..."} or {"data": bytes, "mime_type": "..."}
            #
            # context is a shared mutable dict for this run:
            #   - Pre-populated: run_id, agent_name, connector_id, timestamp
            #   - Add your own values to pass data to prompts and tools
            context["user_name"] = "Peter"
            context["user_id"] = 123
            return content
        
        async def after(response: str, context: dict) -> str:
            # context contains everything: system fields, your values from before(),
            # any values set by tools, plus token_usage and duration_ms
            return response
    """
    before: Optional[Callable[..., Any]] = None
    after: Optional[Callable[..., Any]] = None

    class Config:
        arbitrary_types_allowed = True


class ToolHook(BaseModel):
    """
    Hook functions that run before and after tool execution for an agent.

    Hook files are Python modules in the hooks/ directory,
    named after the agent they apply to (e.g., hooks/assistant.py).

    The 'before' hook receives the tool name, a dict of parameters that will
    be passed to the tool, and an optional shared context dict. It can modify
    params, raise AbortTool to skip the tool, or raise StopProcessing to
    abort the entire run.

    The 'after' hook receives the tool name, the original params, the tool's
    result, and an optional context dict. It can modify the result.

    Example hook file (hooks/assistant.py):
        from connic import AbortTool

        async def before(tool_name: str, params: dict, context: dict) -> dict:
            if tool_name == "db_delete" and not context.get("is_admin"):
                raise AbortTool({"error": "Only admins can delete records"})
            return params

        async def after(tool_name: str, params: dict, result, context: dict):
            return result
    """
    before: Optional[Callable[..., Any]] = None
    after: Optional[Callable[..., Any]] = None

    class Config:
        arbitrary_types_allowed = True


class RetryOptions(BaseModel):
    """Configuration for automatic retries on failures."""
    attempts: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts (max: 10)")
    max_delay: int = Field(default=30, ge=1, le=300, description="Maximum seconds between retries (max: 300s)")
    rerun_middleware: bool = Field(default=False, description="Re-execute the 'before' middleware on each retry attempt. Useful when middleware enriches the prompt with external state that may change between retries.")


class SessionConfig(BaseModel):
    """
    Persistent session configuration for maintaining conversation history across requests.

    When configured, the agent reuses sessions keyed by a resolved value,
    enabling multi-turn conversations that survive restarts and redeployments.

    Example YAML:
        session:
          key: context.telegram_chat_id
          ttl: 86400
    """
    key: str = Field(
        ..., min_length=1,
        description="Dot-path expression to resolve session ID. "
                    "Prefix with 'context.' to read from middleware context, "
                    "or 'input.' to read from the raw payload."
    )
    ttl: Optional[int] = Field(
        default=None, ge=60,
        description="Session time-to-live in seconds. Sessions not updated within "
                    "this period are considered expired. "
                    "When not set, sessions never expire."
    )

    @field_validator('key')
    @classmethod
    def validate_key_prefix(cls, v: str) -> str:
        if not v.startswith("context.") and not v.startswith("input."):
            raise ValueError(
                f"Invalid session key '{v}'. Must start with 'context.' or 'input.' "
                "(e.g., 'context.chat_id' or 'input.user_id')."
            )
        parts = v.split(".", 1)
        if len(parts) < 2 or not parts[1]:
            raise ValueError(
                f"Invalid session key '{v}'. Must specify a field after the prefix "
                "(e.g., 'context.chat_id')."
            )
        return v


class ConcurrencyConfig(BaseModel):
    """
    Key-based concurrency control for agent runs.
    
    Ensures only one run per unique key value is active at a time.
    The key is extracted from the trigger payload using dot-notation.
    
    Example YAML:
        concurrency:
          key: "process_id"
          on_conflict: queue
    """
    key: str = Field(..., min_length=1, description="Dot-path to extract concurrency key from trigger payload (e.g., 'process_id', 'data.customer_id')")
    on_conflict: Literal["queue", "drop"] = Field(
        default="queue",
        description="'queue' waits for active run to finish; 'drop' cancels the new run immediately"
    )


class CollectionPermissions(BaseModel):
    """Per-collection permission overrides for database access control."""
    prevent_delete: Optional[bool] = Field(
        default=None,
        description="If true, db_delete is blocked for this collection. Inherits global setting if None."
    )
    prevent_write: Optional[bool] = Field(
        default=None,
        description="If true, db_insert and db_update are blocked for this collection. Inherits global setting if None."
    )


class DatabaseAccessConfig(BaseModel):
    """
    Access control configuration for database tools.

    Example YAML (simple - flat list, global flags apply to all):
        database:
          collections: [orders, customers]
          prevent_delete: true
          prevent_write: false

    Example YAML (advanced - per-collection overrides):
        database:
          prevent_delete: true
          prevent_write: false
          collections:
            orders:
              prevent_write: true
            customers: {}
            logs:
              prevent_delete: false
    """
    collections: Optional[Dict[str, CollectionPermissions]] = Field(
        default=None,
        description="Restrict access to these collections only. None means all collections are accessible. "
                    "After YAML parsing, this is always a dict mapping collection names to their permissions."
    )
    prevent_delete: bool = Field(
        default=False,
        description="If true, db_delete is blocked for all collections (can be overridden per collection)."
    )
    prevent_write: bool = Field(
        default=False,
        description="If true, db_insert and db_update are blocked for all collections (can be overridden per collection)."
    )


class NamespacePermissions(BaseModel):
    """Per-namespace permission overrides for knowledge access control."""
    prevent_delete: Optional[bool] = Field(
        default=None,
        description="If true, delete_knowledge is blocked for this namespace. Inherits global setting if None."
    )
    prevent_write: Optional[bool] = Field(
        default=None,
        description="If true, store_knowledge is blocked for this namespace. Inherits global setting if None."
    )


class KnowledgeAccessConfig(BaseModel):
    """
    Access control configuration for knowledge tools.

    Example YAML (simple - flat list, global flags apply to all):
        knowledge:
          namespaces: [products, faq]
          prevent_delete: true
          prevent_write: false

    Example YAML (advanced - per-namespace overrides):
        knowledge:
          prevent_delete: true
          prevent_write: false
          namespaces:
            products:
              prevent_write: true
            faq: {}
    """
    namespaces: Optional[Dict[str, NamespacePermissions]] = Field(
        default=None,
        description="Restrict access to these namespaces only. None means all namespaces are accessible. "
                    "After YAML parsing, this is always a dict mapping namespace names to their permissions."
    )
    prevent_delete: bool = Field(
        default=False,
        description="If true, delete_knowledge is blocked for all namespaces (can be overridden per namespace)."
    )
    prevent_write: bool = Field(
        default=False,
        description="If true, store_knowledge is blocked for all namespaces (can be overridden per namespace)."
    )


class GuardrailResult(BaseModel):
    """
    Result returned by a guardrail check function.

    Custom guardrails return this from their ``check()`` function.

    Example::

        from connic import GuardrailResult

        def check(content: str, context: dict) -> GuardrailResult:
            if "bad" in content:
                return GuardrailResult(passed=False, message="Content policy violation")
            return GuardrailResult(passed=True)
    """
    passed: bool
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class GuardrailRule(BaseModel):
    """
    A single guardrail rule within a guardrails configuration.

    Example YAML::

        - type: prompt_injection
          mode: block
        - type: pii
          mode: redact
          config:
            entities: [email, phone, ssn]
        - type: custom
          name: my-check
          mode: warn
    """
    type: str = Field(..., description="Guardrail type (e.g., prompt_injection, pii, moderation, topic_restriction, regex, custom)")
    mode: str = Field(default="block", description="Action on violation: block, warn, or redact (PII only)")
    name: Optional[str] = Field(default=None, description="Name of the custom guardrail file (only for type=custom)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Type-specific configuration options")

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"block", "warn", "redact"}
        if v not in allowed:
            raise ValueError(f"Invalid guardrail mode '{v}'. Must be one of: {', '.join(sorted(allowed))}")
        return v


BUILTIN_GUARDRAIL_TYPES = {
    "prompt_injection", "pii", "moderation", "topic_restriction", "regex", "custom",
    "pii_leakage", "system_prompt_leakage", "relevance", "data_exfiltration",
}


class GuardrailsConfig(BaseModel):
    """
    Input and output guardrails configuration for an agent.

    Example YAML::

        guardrails:
          input:
            - type: prompt_injection
              mode: block
            - type: pii
              mode: redact
          output:
            - type: moderation
              mode: block
            - type: system_prompt_leakage
              mode: block
    """
    input: List[GuardrailRule] = Field(default_factory=list, description="Guardrails applied to agent input before execution")
    output: List[GuardrailRule] = Field(default_factory=list, description="Guardrails applied to agent output after execution")

    @model_validator(mode='after')
    def validate_rules(self):
        for rule in self.input + self.output:
            if rule.type == "custom" and not rule.name:
                raise ValueError("Custom guardrails require a 'name' field pointing to the guardrail file")
            if rule.type not in BUILTIN_GUARDRAIL_TYPES:
                raise ValueError(
                    f"Unknown guardrail type '{rule.type}'. "
                    f"Available types: {', '.join(sorted(BUILTIN_GUARDRAIL_TYPES))}"
                )
            if rule.mode == "redact" and rule.type not in ("pii", "pii_leakage"):
                raise ValueError(f"Mode 'redact' is only supported for 'pii' and 'pii_leakage' guardrails, not '{rule.type}'")
        return self


class CustomGuardrail(BaseModel):
    """A loaded custom guardrail check function."""
    name: str
    check: Callable[..., Any]
    is_async: bool = False

    class Config:
        arbitrary_types_allowed = True


class ApprovalConfig(BaseModel):
    """
    Human-in-the-loop approval configuration.

    When set on an agent, the specified tools will require human approval
    before execution. The agent run pauses until a human approves or rejects
    the tool call (or the timeout expires).

    Each tool entry is either a plain string (always requires approval) or a
    mapping ``{tool_ref: condition}`` where the condition is evaluated at call
    time using ``param.*`` (tool parameters) and ``context.*`` (middleware
    context). Approval is only required when the condition evaluates to true.

    Example YAML:
        approval:
          tools:
            - order_tools.delete_order
            - order_tools.process_refund: param.amount > 50 and not context.is_admin
          timeout: 300
          message: "This action requires human approval before proceeding."
    """
    tools: List[Union[str, Dict[str, str]]] = Field(..., description="Tool names (or {name: condition} mappings) requiring human approval before execution")
    timeout: int = Field(default=300, ge=30, le=604800, description="Seconds to wait for approval before timing out")
    message: Optional[str] = Field(default=None, description="Custom message shown to human approvers")


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
    
    # Condition expression for conditional tool availability (evaluated per-request)
    condition: Optional[str] = None

    # Canonical tool reference (e.g. calculator.add or billing.calculator.add)
    ref: Optional[str] = None

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
    fallback_model: Optional[str] = Field(default=None, description="Fallback AI model to use when the primary model's provider is unavailable (LLM agents only)")
    system_prompt: Optional[str] = Field(default=None, description="Instructions that define the agent's behavior")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Controls randomness in output")
    tools: List[Any] = Field(default_factory=list, description="List of tools the agent can use. Each is a string (always available) or a mapping {tool_ref: condition_expression}. Tool refs must use the exact module path under tools/, such as module.function or directory.module.function.")
    
    # Sequential agent fields (required when type=sequential)
    agents: List[str] = Field(default_factory=list, description="List of agent names to execute in sequence")
    
    # Tool agent fields (required when type=tool)
    tool_name: Optional[str] = Field(default=None, description="Tool to execute for tool-type agents using the exact module path under tools/ (module.function or directory.module.function)")
    
    # Database access control
    database: Optional[DatabaseAccessConfig] = Field(
        default=None,
        description="Access control configuration for database tools. "
                    "If omitted, the agent has unrestricted database access."
    )

    # Knowledge access control
    knowledge: Optional[KnowledgeAccessConfig] = Field(
        default=None,
        description="Access control configuration for knowledge tools. "
                    "If omitted, the agent has unrestricted knowledge access."
    )

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
    source_path: Optional[str] = Field(
        default=None,
        description="The relative path to the agent YAML file (populated at runtime by loader)"
    )

    # A/B test fields - populated by loader, not from YAML
    is_test_variant: bool = Field(
        default=False,
        description="True if this agent is a test variant (name matches {base}-test-{name} pattern)"
    )
    base_agent_name: Optional[str] = Field(
        default=None,
        description="Name of the base agent this is a test variant of (populated at runtime by loader)"
    )
    test_name: Optional[str] = Field(
        default=None,
        description="Test identifier extracted from the agent name (part after -test-). Populated at runtime by loader."
    )
    
    # Reasoning configuration (LLM agents only)
    reasoning: Optional[bool] = Field(
        default=True,
        description="Include the model's reasoning/thinking in the response. When enabled, reasoning content is captured in run traces. Defaults to true."
    )
    reasoning_budget: Optional[int] = Field(
        default=None, ge=0,
        description="Maximum number of tokens the model may use for reasoning. 0 disables reasoning, -1 lets the model decide automatically."
    )
    
    # Common optional fields
    max_concurrent_runs: int = Field(default=1, ge=1, description="Maximum simultaneous runs allowed")
    retry_options: Optional[RetryOptions] = Field(default=None, description="Configuration for automatic retries")
    timeout: Optional[int] = Field(default=None, ge=5, description="Max execution time in seconds (min: 5). Capped by subscription.")
    max_iterations: int = Field(
        default=100, ge=1,
        description="Maximum number of agent loop iterations per run. Each iteration is one LLM call (e.g. agent output, tool call, next output). Prevents infinite loops and excessive resource consumption."
    )
    concurrency: Optional[ConcurrencyConfig] = Field(default=None, description="Key-based concurrency control. Ensures only one run per unique key value at a time.")
    
    # Persistent session configuration
    session: Optional[SessionConfig] = Field(
        default=None,
        description="Persistent session configuration. When set, the agent maintains "
                    "conversation history across requests, keyed by the resolved value."
    )
    
    # Guardrails configuration
    guardrails: Optional[GuardrailsConfig] = Field(
        default=None,
        description="Input and output guardrails for safety and compliance. "
                    "Guardrails run before/after agent execution to enforce content policies."
    )

    # Human-in-the-loop approval configuration
    approval: Optional[ApprovalConfig] = Field(
        default=None,
        description="Human-in-the-loop approval configuration. "
                    "Specifies which tools require human approval before execution."
    )
    
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
