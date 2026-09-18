"""Voice configuration validation and compatibility with the existing agent loader."""

import pytest
from pydantic import ValidationError

from connic import ProjectLoader, VoiceConfig
from connic.core import AgentConfig, resolve_voice_model


def voice_agent(**overrides):
    return AgentConfig(**{
        "version": "1.0",
        "name": "phone-agent",
        "description": "Phone assistant",
        "model": "openai/gpt-realtime",
        "system_prompt": "Help the caller using the available tools.",
        "voice_config": {},
        **overrides,
    })


@pytest.mark.parametrize(("model", "provider", "native_model"), [
    ("openai/gpt-realtime", "openai", "gpt-realtime"),
    ("openai/gpt-realtime-mini", "openai", "gpt-realtime-mini"),
    ("openai/gpt-4o-realtime-preview", "openai", "gpt-4o-realtime-preview"),
    ("openai/gpt-4o-mini-realtime-preview", "openai", "gpt-4o-mini-realtime-preview"),
    ("azure/gpt-realtime-2.1", "azure", "gpt-realtime-2.1"),
    ("azure/gpt-4o-realtime-preview", "azure", "gpt-4o-realtime-preview"),
    ("azure/gpt-4o-mini-realtime-preview", "azure", "gpt-4o-mini-realtime-preview"),
    ("gemini/gemini-3.1-flash-live-preview", "gemini", "gemini-3.1-flash-live-preview"),
    ("gemini/gemini-2.5-flash-native-audio-preview-12-2025", "gemini", "gemini-2.5-flash-native-audio-preview-12-2025"),
    ("vertex_ai/gemini-live-2.5-flash-native-audio", "vertex_ai", "gemini-live-2.5-flash-native-audio"),
    ("openai/new-model", "openai", "new-model"),
    ("azure/support-phone-deployment", "azure", "support-phone-deployment"),
    ("gemini/new-model", "gemini", "new-model"),
    ("vertex_ai/new-model", "vertex_ai", "new-model"),
    (
        "vertex_ai/projects/my-project/locations/europe-west4/publishers/google/models/custom-model",
        "vertex_ai",
        "projects/my-project/locations/europe-west4/publishers/google/models/custom-model",
    ),
    ("openai/gpt-4o", "openai", "gpt-4o"),
    ("gemini/gemini-2.5-flash", "gemini", "gemini-2.5-flash"),
])
def test_voice_model_resolution(model, provider, native_model):
    assert resolve_voice_model(model) == (provider, native_model)
    assert voice_agent(model=model).model == model


@pytest.mark.parametrize("model", [
    "connic/gpt-realtime",
    "openai/",
    "azure/",
    "gemini/",
    "vertex_ai/",
    "google/gemini-3.1-flash-live-preview",
    "google/gemini-2.5-flash-native-audio-preview-12-2025",
    "anthropic/claude-sonnet-4",
    "gpt-realtime",
])
def test_voice_rejects_unsupported_model_routes(model):
    with pytest.raises(ValidationError, match="voice_config requires"):
        voice_agent(model=model)


def test_empty_voice_config_preserves_provider_defaults():
    assert voice_agent().voice_config.model_dump(exclude_none=True) == {"thinking_sound": True, "hang_up_allowed": True}


def test_voice_thinking_sound_can_be_disabled():
    assert voice_agent(voice_config={"thinking_sound": False}).voice_config.thinking_sound is False


def test_voice_hang_up_can_be_disabled():
    assert voice_agent(voice_config={"hang_up_allowed": False}).voice_config.hang_up_allowed is False


@pytest.mark.parametrize("provider", ["openai", "azure", "gemini", "vertex_ai"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "minimal", "xhigh", "off", "provider-specific-level"])
def test_voice_reasoning_level_is_validated_by_the_provider(provider, effort):
    assert voice_agent(model=f"{provider}/native-model", reasoning_effort=effort).reasoning_effort == effort


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_voice_preserves_explicit_transcription_model(provider):
    agent = voice_agent(
        model=f"{provider}/realtime-model",
        voice_config={"transcription_model": "custom-transcription-model"},
    )
    assert agent.voice_config.transcription_model == "custom-transcription-model"


@pytest.mark.parametrize("options", [
    {"voice": ""},
    {"transcription_model": ""},
    {"thinking_sound": {}},
    {"hang_up_allowed": {}},
    {"turn_detection": {"mode": "semantic"}},
    {"turn_detection": {"mode": "automatic"}},
    {"turn_detection": {"silence_ms": 0}},
    {"interruptions": {}},
    {"idle_timeout_seconds": 0},
    {"idle_timeout_seconds": -1},
    {"twilio_account_sid": "AC123"},
    {"turn_detection": {"threshold": 0.5}},
])
def test_voice_rejects_invalid_or_unknown_options(options):
    with pytest.raises(ValidationError):
        VoiceConfig(**options)


@pytest.mark.parametrize("agent_fields", [
    {"type": "tool", "tool_name": "support.lookup"},
    {"type": "sequential", "agents": ["support"]},
])
def test_voice_requires_an_llm_agent(agent_fields):
    with pytest.raises(ValidationError, match="voice_config is only supported for LLM agents"):
        voice_agent(**agent_fields)


def test_voice_rejects_output_guardrails():
    with pytest.raises(ValidationError, match="Output guardrails are not supported for voice agents"):
        voice_agent(guardrails={"output": [{"type": "pii_leakage", "mode": "block"}]})


def test_voice_rejects_input_guardrails():
    with pytest.raises(ValidationError, match="Input guardrails are not supported for voice agents"):
        voice_agent(guardrails={"input": [{"type": "prompt_injection", "mode": "block"}]})


@pytest.mark.parametrize(("field", "value"), [
    ("approval", {"tools": ["trigger_agent"]}),
    ("output_schema", "result"),
    ("output_schema_dict", {"type": "object"}),
    ("fallback_model", "openai/gpt-realtime-mini"),
    ("context_compression", {}),
])
def test_voice_rejects_unsupported_execution_settings(field, value):
    with pytest.raises(ValidationError, match=f"{field} is not supported for voice agents"):
        voice_agent(**{field: value})


def test_text_agents_keep_output_guardrails():
    agent = voice_agent(
        model="openai/gpt-4o",
        voice_config=None,
        guardrails={"output": [{"type": "pii_leakage", "mode": "block"}]},
    )
    assert agent.guardrails.output[0].type == "pii_leakage"


def test_voice_yaml_loads_with_existing_tools_and_session(tmp_path):
    agents_path = tmp_path / "agents"
    agents_path.mkdir()
    (agents_path / "phone-agent.yaml").write_text('''
version: "1.0"
name: phone-agent
description: Phone assistant
model: gemini/gemini-3.1-flash-live-preview
system_prompt: Help the caller using the available tools.
voice_config:
  voice: Kore
  language: de-DE
  greeting: Hallo, wie kann ich dir helfen?
  thinking_sound: true
  hang_up_allowed: false
  turn_detection:
    silence_ms: 500
  interruptions:
    enabled: true
  idle_timeout_seconds: 30
tools:
  - retrieval_query
  - trigger_agent
session:
  key: context.customer_id
''')

    agent = ProjectLoader(str(tmp_path)).load_agent("phone-agent")

    assert agent.config.voice_config.voice == "Kore"
    assert agent.config.voice_config.language == "de-DE"
    assert agent.config.voice_config.greeting == "Hallo, wie kann ich dir helfen?"
    assert agent.config.voice_config.thinking_sound is True
    assert agent.config.voice_config.hang_up_allowed is False
    assert agent.config.voice_config.turn_detection.model_dump(exclude_none=True) == {"silence_ms": 500}
    assert agent.config.voice_config.interruptions.enabled is True
    assert agent.config.voice_config.idle_timeout_seconds == 30
    assert agent.config.session.key == "context.customer_id"
    assert {tool.name for tool in agent.tools} == {"retrieval_query", "trigger_agent"}
    assert all(tool.is_predefined for tool in agent.tools)
