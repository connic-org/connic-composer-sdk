"""Tests for connic.core – models, validators, and tool execution."""
import asyncio
import warnings

import pytest

from connic.core import (
    AbortTool,
    AgentConfig,
    AgentType,
    ApprovalConfig,
    ConcurrencyConfig,
    CollectionPermissions,
    DatabaseAccessConfig,
    GuardrailResult,
    GuardrailRule,
    GuardrailsConfig,
    KnowledgeAccessConfig,
    McpServerConfig,
    Middleware,
    NamespacePermissions,
    RetryOptions,
    SessionConfig,
    StopProcessing,
    Tool,
    Agent,
    ToolHook,
    BUILTIN_GUARDRAIL_TYPES,
)


# ---------------------------------------------------------------------------
# StopProcessing / AbortTool exceptions
# ---------------------------------------------------------------------------


def test_stop_processing_stores_response():
    exc = StopProcessing("halt now")
    assert exc.response == "halt now"
    assert str(exc) == "halt now"


def test_abort_tool_stores_result_string():
    exc = AbortTool("denied")
    assert exc.result == "denied"
    assert str(exc) == "denied"


def test_abort_tool_stores_result_dict():
    result = {"error": "permission denied"}
    exc = AbortTool(result)
    assert exc.result == result


# ---------------------------------------------------------------------------
# SessionConfig validation
# ---------------------------------------------------------------------------


def test_session_config_valid_context_key():
    cfg = SessionConfig(key="context.chat_id")
    assert cfg.key == "context.chat_id"


def test_session_config_valid_input_key():
    cfg = SessionConfig(key="input.user_id")
    assert cfg.key == "input.user_id"


def test_session_config_rejects_invalid_prefix():
    with pytest.raises(ValueError, match="Must start with 'context.' or 'input.'"):
        SessionConfig(key="payload.chat_id")


def test_session_config_rejects_empty_field_after_prefix():
    with pytest.raises(ValueError, match="Must specify a field after the prefix"):
        SessionConfig(key="context.")


# ---------------------------------------------------------------------------
# GuardrailRule / GuardrailsConfig validation
# ---------------------------------------------------------------------------


def test_guardrail_rule_valid():
    rule = GuardrailRule(type="prompt_injection", mode="block")
    assert rule.type == "prompt_injection"
    assert rule.mode == "block"


def test_guardrail_rule_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Invalid guardrail mode"):
        GuardrailRule(type="pii", mode="ignore")


def test_guardrails_config_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown guardrail type"):
        GuardrailsConfig(input=[GuardrailRule(type="nonexistent_type", mode="block")])


def test_guardrails_config_rejects_custom_without_name():
    with pytest.raises(ValueError, match="Custom guardrails require a 'name' field"):
        GuardrailsConfig(input=[GuardrailRule(type="custom", mode="block")])


def test_guardrails_config_rejects_redact_on_non_pii():
    with pytest.raises(ValueError, match="Mode 'redact' is only supported"):
        GuardrailsConfig(input=[GuardrailRule(type="prompt_injection", mode="redact")])


def test_guardrails_config_allows_redact_on_pii():
    cfg = GuardrailsConfig(input=[GuardrailRule(type="pii", mode="redact")])
    assert cfg.input[0].mode == "redact"


def test_guardrails_config_allows_redact_on_pii_leakage():
    cfg = GuardrailsConfig(output=[GuardrailRule(type="pii_leakage", mode="redact")])
    assert cfg.output[0].mode == "redact"


def test_guardrails_config_custom_with_name():
    cfg = GuardrailsConfig(input=[GuardrailRule(type="custom", mode="warn", name="my-check")])
    assert cfg.input[0].name == "my-check"


# ---------------------------------------------------------------------------
# AgentConfig validation
# ---------------------------------------------------------------------------


def _llm_agent(**overrides) -> dict:
    base = {
        "version": "1.0",
        "name": "test-agent",
        "description": "test",
        "type": "llm",
        "model": "openai/gpt-4o",
        "system_prompt": "You are helpful.",
    }
    base.update(overrides)
    return base


def test_agent_config_valid_llm():
    cfg = AgentConfig(**_llm_agent())
    assert cfg.type == AgentType.LLM
    assert cfg.model == "openai/gpt-4o"


def test_agent_config_rejects_unsupported_version():
    with pytest.raises(ValueError, match="Unsupported version"):
        AgentConfig(**_llm_agent(version="2.0"))


def test_agent_config_rejects_invalid_name():
    with pytest.raises(ValueError, match="Invalid agent name"):
        AgentConfig(**_llm_agent(name="My Agent"))


def test_agent_config_rejects_name_starting_with_hyphen():
    with pytest.raises(ValueError, match="Invalid agent name"):
        AgentConfig(**_llm_agent(name="-invalid"))


def test_agent_config_allows_single_char_name():
    cfg = AgentConfig(**_llm_agent(name="a"))
    assert cfg.name == "a"


def test_agent_config_llm_requires_model():
    with pytest.raises(ValueError, match="LLM agents require 'model'"):
        AgentConfig(**_llm_agent(model=None))


def test_agent_config_llm_requires_system_prompt():
    with pytest.raises(ValueError, match="LLM agents require 'system_prompt'"):
        AgentConfig(**_llm_agent(system_prompt=None))


def test_agent_config_sequential_requires_agents():
    with pytest.raises(ValueError, match="Sequential agents require 'agents'"):
        AgentConfig(
            version="1.0", name="seq", description="seq", type="sequential",
        )


def test_agent_config_sequential_valid():
    cfg = AgentConfig(
        version="1.0", name="seq", description="seq", type="sequential",
        agents=["step-1", "step-2"],
    )
    assert cfg.type == AgentType.SEQUENTIAL
    assert cfg.agents == ["step-1", "step-2"]


def test_agent_config_tool_requires_tool_name():
    with pytest.raises(ValueError, match="Tool agents require 'tool_name'"):
        AgentConfig(
            version="1.0", name="t", description="t", type="tool",
        )


def test_agent_config_tool_valid():
    cfg = AgentConfig(
        version="1.0", name="t", description="t", type="tool",
        tool_name="utils.calculate",
    )
    assert cfg.type == AgentType.TOOL


def test_agent_config_mcp_servers_limit():
    with pytest.raises(ValueError, match="Too many MCP servers"):
        AgentConfig(**_llm_agent(
            mcp_servers=[
                {"name": f"srv-{i}", "url": f"http://localhost:{8000+i}"}
                for i in range(51)
            ],
        ))


@pytest.mark.parametrize(("reasoning", "expected_effort"), [(True, "auto"), (False, "off")])
def test_agent_config_migrates_legacy_reasoning_flag(reasoning, expected_effort):
    with pytest.warns(DeprecationWarning, match="AgentConfig.reasoning is deprecated"):
        cfg = AgentConfig(**_llm_agent(reasoning=reasoning))

    assert cfg.reasoning_effort == expected_effort
    assert cfg.reasoning is None


def test_agent_config_reasoning_effort_takes_precedence_over_legacy_reasoning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = AgentConfig(**_llm_agent(reasoning=False, reasoning_effort="high"))

    assert caught == []
    assert cfg.reasoning_effort == "high"
    assert cfg.reasoning is None


def test_mcp_server_config_bridge_default_none():
    cfg = McpServerConfig(name="srv", url="https://mcp.example.com/mcp")
    assert cfg.bridge is None


def test_mcp_server_config_bridge_field():
    cfg = McpServerConfig(
        name="internal",
        url="http://mcp.internal:8080/mcp",
        bridge="abc123",
    )
    assert cfg.bridge == "abc123"


def test_mcp_server_config_bridge_supports_var_placeholder():
    # Substitution itself happens at runtime in the runner, but the schema
    # must accept a ${VAR} string verbatim (the same shape used for headers/url).
    cfg = McpServerConfig(
        name="internal",
        url="http://mcp.internal:8080/mcp",
        bridge="${INTERNAL_BRIDGE_ID}",
    )
    assert cfg.bridge == "${INTERNAL_BRIDGE_ID}"


def test_mcp_server_config_headers_support_context_placeholder():
    cfg = McpServerConfig(
        name="compliance",
        url="https://mcp.example.com/mcp",
        headers={"Authorization": "Bearer ${MCP_TOKEN}", "X-User-Id": "${context.user_id}"},
    )

    assert cfg.headers == {"Authorization": "Bearer ${MCP_TOKEN}", "X-User-Id": "${context.user_id}"}


# ---------------------------------------------------------------------------
# Tool.execute / execute_sync
# ---------------------------------------------------------------------------


def test_tool_execute_sync_function():
    def add(a: int, b: int) -> int:
        return a + b

    tool = Tool(name="add", func=add)
    result = asyncio.run(tool.execute(a=2, b=3))
    assert result == 5


def test_tool_execute_async_function():
    async def greet(name: str) -> str:
        return f"Hello, {name}"

    tool = Tool(name="greet", func=greet, is_async=True)
    result = asyncio.run(tool.execute(name="World"))
    assert result == "Hello, World"


def test_tool_execute_raises_when_no_func():
    tool = Tool(name="ghost", is_predefined=True)
    with pytest.raises(ValueError, match="has no function"):
        asyncio.run(tool.execute())


def test_tool_execute_sync_method():
    def multiply(x: int, y: int) -> int:
        return x * y

    tool = Tool(name="multiply", func=multiply)
    assert tool.execute_sync(x=4, y=5) == 20


def test_tool_execute_sync_raises_when_no_func():
    tool = Tool(name="ghost", is_predefined=True)
    with pytest.raises(ValueError, match="has no function"):
        tool.execute_sync()


def test_tool_execute_sync_runs_async_func():
    async def async_double(n: int) -> int:
        return n * 2

    tool = Tool(name="double", func=async_double, is_async=True)
    result = tool.execute_sync(n=7)
    assert result == 14


# ---------------------------------------------------------------------------
# Agent helper methods
# ---------------------------------------------------------------------------


def test_agent_get_tool_found():
    tool_a = Tool(name="search", description="Search docs")
    tool_b = Tool(name="calc", description="Calculator")
    agent = Agent(
        config=AgentConfig(**_llm_agent()),
        tools=[tool_a, tool_b],
    )
    assert agent.get_tool("calc") is tool_b


def test_agent_get_tool_not_found():
    agent = Agent(
        config=AgentConfig(**_llm_agent()),
        tools=[Tool(name="search", description="Search")],
    )
    assert agent.get_tool("missing") is None


def test_agent_get_tools_schema():
    tool = Tool(name="search", description="Search docs", parameters={"type": "object", "properties": {"q": {"type": "string"}}})
    agent = Agent(config=AgentConfig(**_llm_agent()), tools=[tool])
    schema = agent.get_tools_schema()
    assert schema == [
        {
            "name": "search",
            "description": "Search docs",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]


# ---------------------------------------------------------------------------
# Other model instantiation smoke tests
# ---------------------------------------------------------------------------


def test_retry_options_defaults():
    opts = RetryOptions()
    assert opts.attempts == 3
    assert opts.max_delay == 30
    assert opts.rerun_middleware is False


def test_concurrency_config_defaults():
    cfg = ConcurrencyConfig(key="user_id")
    assert cfg.on_conflict == "queue"


def test_guardrail_result_basic():
    result = GuardrailResult(passed=True)
    assert result.passed is True
    assert result.message is None


def test_middleware_arbitrary_callables():
    mw = Middleware(before=lambda c, ctx: c, after=lambda r, ctx: r)
    assert mw.before is not None
    assert mw.after is not None


def test_tool_hook_arbitrary_callables():
    hook = ToolHook(before=lambda t, p, ctx: p, after=lambda t, p, r, ctx: r)
    assert hook.before is not None


def test_approval_config_defaults():
    cfg = ApprovalConfig(tools=["db_delete"])
    assert cfg.timeout == 3600
    assert cfg.on_rejection == "fail"


def test_database_access_config_defaults():
    cfg = DatabaseAccessConfig()
    assert cfg.prevent_delete is False
    assert cfg.prevent_write is False
    assert cfg.collections is None


def test_knowledge_access_config_defaults():
    cfg = KnowledgeAccessConfig()
    assert cfg.prevent_delete is False
    assert cfg.prevent_write is False
    assert cfg.namespaces is None
