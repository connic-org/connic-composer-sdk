"""Tests for connic.core – models, validators, and tool execution."""
import asyncio
import warnings

import pytest

from connic.core import (
    AbortTool,
    AgentConfig,
    AgentType,
    ApprovalConfig,
    ApprovalInput,
    ConcurrencyConfig,
    CollectionPermissions,
    DatabaseAccessConfig,
    GuardrailResult,
    GuardrailRule,
    GuardrailsConfig,
    McpServerConfig,
    Middleware,
    NamespacePermissions,
    RetryOptions,
    RetrievalAccessConfig,
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
    assert exc.publish_outbound is True
    assert str(exc) == "halt now"


def test_stop_processing_can_disable_outbound_publish():
    exc = StopProcessing("halt now", publish_outbound=False)
    assert exc.response == "halt now"
    assert exc.publish_outbound is False
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


@pytest.mark.parametrize("value", [True, {}, {"key": None}, SessionConfig()])
def test_agent_session_without_key_normalizes_to_shared_persistence(value):
    cfg = AgentConfig(**_llm_agent(session=value))

    assert isinstance(cfg.session, SessionConfig)
    assert cfg.model_dump()["session"] == {"key": None, "ttl": None, "history": True, "browser": True}
    assert AgentConfig.model_validate_json(cfg.model_dump_json()).session == cfg.session


@pytest.mark.parametrize("value", [False, None])
def test_agent_session_can_be_disabled(value):
    cfg = AgentConfig(**_llm_agent(session=value))

    assert cfg.session is None
    assert cfg.model_dump()["session"] is None
    assert AgentConfig(**_llm_agent()).session is None


@pytest.mark.parametrize("history,browser", [(True, True), (True, False), (False, True), (False, False)])
def test_session_persistence_switches_are_independent(history, browser):
    cfg = AgentConfig(**_llm_agent(session={"history": history, "browser": browser, "ttl": 3600}))

    assert cfg.session.key is None
    assert cfg.session.ttl == 3600
    assert cfg.session.history is history
    assert cfg.session.browser is browser


@pytest.mark.parametrize("key", ["context.chat_id", "input.user_id"])
def test_keyed_session_keeps_key_and_ttl_with_optional_persistence(key):
    cfg = AgentConfig(**_llm_agent(session={"key": key, "ttl": 86400, "history": False}))

    assert cfg.model_dump()["session"] == {"key": key, "ttl": 86400, "history": False, "browser": True}


@pytest.mark.parametrize("value", ["true", "false", 0, 1, [], "input.user_id"])
def test_agent_session_rejects_non_boolean_shorthand(value):
    with pytest.raises(ValueError):
        AgentConfig(**_llm_agent(session=value))


@pytest.mark.parametrize("field", ["history", "browser"])
@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_session_persistence_switches_require_booleans(field, value):
    with pytest.raises(ValueError):
        SessionConfig(**{field: value})


def test_agent_session_json_schema_supports_shorthand_and_optional_key():
    schema = AgentConfig.model_json_schema()

    assert {"type": "boolean"} in schema["properties"]["session"]["anyOf"]
    assert {"$ref": "#/$defs/SessionConfig"} in schema["properties"]["session"]["anyOf"]
    session_schema = schema["$defs"]["SessionConfig"]
    assert "key" not in session_schema.get("required", [])
    assert session_schema["properties"]["key"]["default"] is None
    assert session_schema["properties"]["ttl"]["default"] is None
    for field in ("history", "browser"):
        assert session_schema["properties"][field]["type"] == "boolean"
        assert session_schema["properties"][field]["default"] is True


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


def test_agent_config_warns_for_deprecated_reasoning_budget():
    with pytest.warns(DeprecationWarning, match="AgentConfig.reasoning_budget is deprecated"):
        cfg = AgentConfig(**_llm_agent(reasoning_budget=4096))

    assert cfg.reasoning_budget == 4096


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
    assert opts.initial_delay == 10
    assert opts.max_delay == 30
    assert opts.rerun_middleware is False


def test_retry_options_allow_max_delay_to_cap_the_initial_delay():
    opts = RetryOptions(initial_delay=30, max_delay=10)

    assert opts.initial_delay == 30
    assert opts.max_delay == 10


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
    assert cfg.inputs == []


def test_approval_config_accepts_conditions_and_human_input():
    cfg = ApprovalConfig(
        tools=["auth.just_needs_approval", {"billing.refund": "param.amount > 50"}],
        inputs=[
            {"get_mfa": {"prompt": "Request an MFA code.", "label": "MFA code", "sensitive": True}},
            {"get_details": {"prompt": "Ask for missing details.", "label": "Additional details"}},
        ],
    )

    assert cfg.tools[1] == {"billing.refund": "param.amount > 50"}
    assert cfg.inputs[0]["get_mfa"] == ApprovalInput(prompt="Request an MFA code.", label="MFA code", sensitive=True)
    assert cfg.inputs[1]["get_details"].sensitive is False
    assert cfg.model_dump()["inputs"][0] == {
        "get_mfa": {"prompt": "Request an MFA code.", "label": "MFA code", "sensitive": True, "params": []},
    }


def test_approval_input_without_params_has_no_arguments():
    config = ApprovalInput(prompt="Request an MFA code.", label="MFA code")

    assert config.params == []
    assert config.parameters_schema() == {
        "type": "object", "properties": {}, "additionalProperties": False,
    }


def test_approval_input_params_generate_required_scalar_schema():
    params = [{"reason": "str"}, {"attempt": "int"}, {"amount": "float"}, {"urgent": "bool"}]
    config = ApprovalInput(prompt="Request a code.", label="Code", params=params)

    assert config.params == params
    assert config.model_dump()["params"] == params
    assert config.parameters_schema() == {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "attempt": {"type": "integer"},
            "amount": {"type": "number"},
            "urgent": {"type": "boolean"},
        },
        "required": ["reason", "attempt", "amount", "urgent"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize("params", [
    None,
    {"reason": "str"},
    ["reason"],
    [{}],
    [{"reason": "str", "account_email": "str"}],
    [{"reason": "str"}, {"reason": "int"}],
    [{"reason": "string"}],
    [{"reason": "list"}],
    [{"reason": "dict"}],
    [{"reason": "str | None"}],
    [{"reason": {"type": "str"}}],
    [{"reason": None}],
    [{"reason": True}],
    [{"reason": 1}],
])
def test_approval_input_rejects_invalid_params(params):
    with pytest.raises(ValueError):
        ApprovalInput(prompt="Request a code.", label="Code", params=params)


@pytest.mark.parametrize("name", ["", " ", "auth.reason", "account-email", "2fa", "réason", "class", "context"])
def test_approval_input_rejects_invalid_param_names(name):
    with pytest.raises(ValueError):
        ApprovalInput(prompt="Request a code.", label="Code", params=[{name: "str"}])


@pytest.mark.parametrize("entry", [
    {},
    {"auth.get_mfa": "True", "auth.other": "True"},
    {"auth.get_mfa": {"label": "MFA code", "sensitive": True}},
])
def test_approval_tools_reject_invalid_entries(entry):
    with pytest.raises(ValueError):
        ApprovalConfig(tools=[entry])


@pytest.mark.parametrize("entry", [
    {},
    "get_mfa",
    {"get_mfa": "param.enabled"},
    {"get_mfa": {}},
    {"get_mfa": {"label": "MFA code"}},
    {"get_mfa": {"prompt": "Request an MFA code."}},
    {"get_mfa": {"prompt": "", "label": "MFA code"}},
    {"get_mfa": {"prompt": "   ", "label": "MFA code"}},
    {"get_mfa": {"prompt": "Request an MFA code.", "label": ""}},
    {"get_mfa": {"prompt": "Request an MFA code.", "label": "   "}},
    {"get_mfa": {"prompt": "Request an MFA code.", "label": "a" * 201}},
    {"get_mfa": {"prompt": "Request an MFA code.", "label": "MFA code", "unknown": True}},
    {"get_mfa": {"prompt": "Request an MFA code.", "label": "MFA code", "sensitive": "true"}},
    {" ": {"prompt": "Request an MFA code.", "label": "MFA code"}},
    {
        "get_mfa": {"prompt": "Request an MFA code.", "label": "MFA code"},
        "auth.other": {"prompt": "Request another code.", "label": "Other code"},
    },
])
def test_approval_config_rejects_invalid_input(entry):
    with pytest.raises(ValueError):
        ApprovalConfig(inputs=[entry])


@pytest.mark.parametrize("name", ["auth.get_mfa", "auth.*", "auth..get_mfa", "auth/get_mfa", "get-mfa", "2fa", "gét_mfa"])
def test_approval_inputs_reject_invalid_names(name):
    with pytest.raises(ValueError, match="identifiers without dots"):
        ApprovalConfig(inputs=[{name: {"prompt": "Request a code.", "label": "Code"}}])


def test_approval_inputs_reject_duplicate_names():
    entry = {"get_mfa": {"prompt": "Request a code.", "label": "Code"}}
    with pytest.raises(ValueError, match="Duplicate approval input name"):
        ApprovalConfig(inputs=[entry, entry])


@pytest.mark.parametrize("name", ["search_tools", "use_tool"])
def test_approval_inputs_reject_discovery_helper_names(name):
    with pytest.raises(ValueError, match="reserved for tool discovery"):
        ApprovalConfig(inputs=[{name: {"prompt": "Request input.", "label": "Input"}}])


def test_approval_inputs_limit_model_tool_name_length():
    input_config = {"prompt": "Request input.", "label": "Input"}
    ApprovalConfig(inputs=[{"a" * 64: input_config}])
    with pytest.raises(ValueError, match="at most 64 characters"):
        ApprovalConfig(inputs=[{"a" * 65: input_config}])


@pytest.mark.parametrize("tool", ["get_mfa", {"get_mfa": "param.enabled"}])
def test_approval_inputs_cannot_also_be_tool_approvals(tool):
    with pytest.raises(ValueError, match="both approval.tools and approval.inputs"):
        ApprovalConfig(tools=[tool], inputs=[{"get_mfa": {"prompt": "Request a code.", "label": "Code"}}])


@pytest.mark.parametrize("agent_type", ["tool", "sequential"])
def test_approval_inputs_require_an_llm_agent(agent_type):
    with pytest.raises(ValueError, match="approval.inputs is only supported for LLM agents"):
        AgentConfig(
            version="1.0", name="assistant", description="Collect input.", type=agent_type,
            tool_name="auth.authenticate", agents=["auth"],
            approval={"inputs": [{"get_mfa": {"prompt": "Request a code.", "label": "Code"}}]},
        )


def test_database_access_config_defaults():
    cfg = DatabaseAccessConfig()
    assert cfg.prevent_delete is False
    assert cfg.prevent_write is False
    assert cfg.collections is None


def test_retrieval_access_config_defaults():
    cfg = RetrievalAccessConfig()
    assert cfg.prevent_delete is False
    assert cfg.prevent_write is False
    assert cfg.namespaces is None
