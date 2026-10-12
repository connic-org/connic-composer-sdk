"""Pydantic models for connectors declared in code.

Each ``connectors/<key>.yaml`` file declares one connector. The filename stem
is the connector's key and identity; renaming the file replaces the connector.

Example::

    type: kafka
    mode: inbound
    name: Order events                # optional, defaults to the key
    connection: orders-kafka-staging  # saved Connection, by name
    bridge: staging-cluster           # Bridge name or ID
    config:                           # connector fields, same names as the REST API
      topic: ${ORDER_EVENTS_TOPIC}
    agents:
      - order-processor
      - name: audit-logger
        files_required: true
    environments:                     # omit to deploy to every environment
      Staging: {}
      Production:
        connection: orders-kafka-prod
        config:
          group_id: connic-orders-prod

``${VAR}`` references are resolved at deploy time from the target
environment's variables, in ``config``, ``connection``, ``bridge`` and
``domain`` only. Values that Connic generates, such as webhook secrets, are
never part of the file.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

CONNECTOR_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
_VARIABLE_PATTERN = re.compile(r"\$\{([^}]*)\}")
_VARIABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ACTION_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PHONE_NUMBER_PATTERN = re.compile(r"^\+[1-9][0-9]{6,14}$")

CONNECTOR_MODES: Dict[str, Tuple[str, ...]] = {
    "http": ("inbound", "outbound", "sync"),
    "cron": ("inbound",),
    "kafka": ("inbound", "outbound"),
    "websocket": ("sync",),
    "mcp": ("inbound", "sync"),
    "sqs": ("inbound", "outbound"),
    "s3": ("inbound",),
    "postgres": ("inbound",),
    "email": ("inbound", "outbound"),
    "stripe": ("inbound",),
    "telegram": ("inbound", "outbound"),
    "twilio": ("sync",),
    "twilio_messaging": ("inbound", "outbound"),
    "telnyx": ("sync",),
    "telnyx_messaging": ("inbound", "outbound"),
}
_TYPE_ALIASES = {"webhook": "http"}

CONNECTION_TYPES = frozenset({
    "kafka", "sqs", "s3", "postgres", "email", "twilio", "twilio_messaging", "telnyx", "telnyx_messaging",
})
BRIDGE_TYPES = frozenset({"http", "kafka", "sqs", "s3", "postgres", "email"})
DOMAIN_TYPES = frozenset({"http", "mcp", "s3", "stripe", "telegram", "websocket"})
VOICE_TYPES = frozenset({"twilio", "telnyx"})

GENERATED_FIELDS: Dict[str, frozenset[str]] = {
    "http": frozenset({"secret", "webhook_secret"}),
    "websocket": frozenset({"secret"}),
    "mcp": frozenset({"secret"}),
    "s3": frozenset({"webhook_secret"}),
    "telegram": frozenset({"secret_token"}),
    "twilio": frozenset({"webhook_url"}),
    "twilio_messaging": frozenset({"webhook_url", "sender_sid"}),
    "telnyx": frozenset({"webhook_url", "application_id", "application_name"}),
    "telnyx_messaging": frozenset({"webhook_url", "sender_id", "messaging_profile_id", "setup_mode"}),
}

# Config keys that have a dedicated place in the file.
_RELOCATED_CONFIG_KEYS = {
    "mode": "mode",
    "connection_id": "connection",
    "bridge_id": "bridge",
    "phone_number_sid": "config.phone_number",
    "phone_number_id": "config.phone_number",
}


def _check_variables(value: Any, label: str) -> None:
    if isinstance(value, str):
        for match in _VARIABLE_PATTERN.finditer(value):
            if not _VARIABLE_NAME_PATTERN.fullmatch(match.group(1)):
                raise ValueError(
                    f"{label}: '${{{match.group(1)}}}' is not a variable reference; "
                    "use ${NAME} with letters, digits and underscores"
                )
        if "${" in _VARIABLE_PATTERN.sub("", value):
            raise ValueError(f"{label}: unterminated '${{' variable reference")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_variables(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_variables(item, f"{label}[{index}]")


def _reject_variables(value: Optional[str], label: str) -> None:
    if value is not None and "${" in value:
        raise ValueError(f"{label} can't use ${{...}}; variables only work in config, connection, bridge and domain")


def _normalize_agent_links(value: Any) -> Any:
    if isinstance(value, list):
        return [{"name": item} if isinstance(item, str) else item for item in value]
    return value


class OutboundLink(BaseModel):
    """How a linked agent uses an outbound connector."""

    model_config = ConfigDict(extra="forbid")

    usage: Literal["automatic", "agent_tool", "middleware"]
    action_name: Optional[str] = None
    source: Optional[List[str]] = Field(
        default=None,
        description="Connector keys whose runs trigger this automatic outbound connector. Omit for all runs.",
    )

    @model_validator(mode="after")
    def _check_usage(self) -> "OutboundLink":
        if self.usage == "automatic":
            if self.action_name is not None:
                raise ValueError("automatic outbound links can't set action_name")
            if self.source is not None and not self.source:
                raise ValueError("source must list at least one connector key; omit it to use every run")
            return self
        if self.source is not None:
            raise ValueError(f"{self.usage} outbound links can't set source")
        if self.action_name is None or not _ACTION_NAME_PATTERN.fullmatch(self.action_name):
            raise ValueError(
                "action_name must start with a lowercase letter and contain only lowercase letters, "
                "numbers or underscores (64 characters max)"
            )
        return self


class AgentLink(BaseModel):
    """An agent linked to the connector."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    files_required: StrictBool = False
    outbound: Optional[OutboundLink] = None


class ConnectorOverride(BaseModel):
    """Per-environment changes to a connector."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1)
    connection: Optional[str] = Field(default=None, min_length=1)
    bridge: Optional[str] = Field(default=None, min_length=1)
    domain: Optional[str] = Field(default=None, min_length=1)
    agents: Optional[List[AgentLink]] = None
    config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("agents", mode="before")
    @classmethod
    def _normalize_agents(cls, v: Any) -> Any:
        return _normalize_agent_links(v)


class ConnectorFile(BaseModel):
    """Parsed contents of one ``connectors/<key>.yaml`` file."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="Taken from the filename stem.")
    type: str
    mode: Optional[str] = None
    name: Optional[str] = Field(default=None, min_length=1, description="Defaults to the key.")
    connection: Optional[str] = Field(default=None, min_length=1)
    bridge: Optional[str] = Field(default=None, min_length=1)
    domain: Optional[str] = Field(default=None, min_length=1)
    config: Dict[str, Any] = Field(default_factory=dict)
    agents: List[AgentLink] = Field(default_factory=list)
    environments: Optional[Dict[str, Optional[ConnectorOverride]]] = Field(
        default=None,
        description="Environment names the connector is deployed to, with optional overrides. Omit for all.",
    )

    @field_validator("agents", mode="before")
    @classmethod
    def _normalize_agents(cls, v: Any) -> Any:
        return _normalize_agent_links(v)

    @field_validator("key")
    @classmethod
    def _validate_key(cls, v: str) -> str:
        if not CONNECTOR_KEY_PATTERN.fullmatch(v):
            raise ValueError(
                f"file name '{v}' must start with a letter or digit and use only letters, digits, '-' or '_' "
                "(100 characters max)"
            )
        return v

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        v = _TYPE_ALIASES.get(v, v)
        if v == "slack":
            raise ValueError("Slack connectors can't be declared in code; create them in the dashboard")
        if v not in CONNECTOR_MODES:
            known = sorted({*CONNECTOR_MODES, *_TYPE_ALIASES} - {"http"})
            raise ValueError(f"unknown connector type '{v}'. Known types: {', '.join(known)}")
        return v

    @model_validator(mode="after")
    def _validate_connector(self) -> "ConnectorFile":
        modes = CONNECTOR_MODES[self.type]
        if self.mode is None:
            if len(modes) > 1:
                raise ValueError(f"mode is required for {self.type} connectors: {', '.join(modes)}")
            self.mode = modes[0]
        elif self.mode not in modes:
            raise ValueError(f"{self.type} connectors support mode {', '.join(modes)}, not '{self.mode}'")

        if self.environments is not None:
            if not self.environments:
                raise ValueError("environments must list at least one environment; omit it to deploy everywhere")
            for environment in list(self.environments):
                _reject_variables(environment, f"environment '{environment}'")
                if self.environments[environment] is None:
                    self.environments[environment] = ConnectorOverride()

        for environment, spec in self.specs():
            where = f"environment '{environment}': " if environment else ""
            try:
                self._validate_spec(spec)
            except ValueError as exc:
                raise ValueError(f"{where}{exc}") from None
        return self

    def _validate_spec(self, spec: Dict[str, Any]) -> None:
        _reject_variables(spec["name"], "name")
        for field in ("connection", "bridge", "domain"):
            _check_variables(spec[field], field)
        _check_variables(spec["config"], "config")

        if self.type in CONNECTION_TYPES and spec["connection"] is None:
            raise ValueError(f"{self.type} connectors need a connection")
        if self.type not in CONNECTION_TYPES and spec["connection"] is not None:
            raise ValueError(f"{self.type} connectors don't use a connection")
        if self.type not in BRIDGE_TYPES and spec["bridge"] is not None:
            raise ValueError(f"{self.type} connectors don't support a bridge")
        if self.type not in DOMAIN_TYPES and spec["domain"] is not None:
            raise ValueError(f"{self.type} connectors don't support a custom domain")

        generated = GENERATED_FIELDS.get(self.type, frozenset())
        for key in spec["config"]:
            if key in generated:
                raise ValueError(f"config.{key} is generated by Connic; remove it from the file")
            if key in _RELOCATED_CONFIG_KEYS:
                raise ValueError(f"config.{key} isn't allowed; use {_RELOCATED_CONFIG_KEYS[key]} instead")
            if key.startswith("_"):
                raise ValueError(f"config.{key} is reserved for Connic")

        if self.type in VOICE_TYPES:
            phone_number = spec["config"].get("phone_number")
            if not isinstance(phone_number, str):
                raise ValueError("config.phone_number is required, in E.164 format such as +4930123456")
            if "${" not in phone_number and not _PHONE_NUMBER_PATTERN.fullmatch(phone_number):
                raise ValueError(f"config.phone_number '{phone_number}' must be in E.164 format such as +4930123456")

        agents: List[AgentLink] = spec["agents"]
        if not agents:
            raise ValueError("link at least one agent")
        if self.type in VOICE_TYPES and len(agents) != 1:
            raise ValueError(f"{self.type} connectors link exactly one voice agent")
        seen = set()
        for link in agents:
            _reject_variables(link.name, "agent name")
            if link.name in seen:
                raise ValueError(f"agent '{link.name}' is linked twice")
            seen.add(link.name)
            if link.outbound is not None and self.mode != "outbound":
                raise ValueError(f"agent '{link.name}': outbound settings need an outbound connector")

    def specs(self) -> List[Tuple[Optional[str], Dict[str, Any]]]:
        """Every environment variant of the connector. The environment is None when it deploys everywhere."""
        if self.environments is None:
            return [(None, self._merge(ConnectorOverride()))]
        return [(environment, self._merge(override)) for environment, override in self.environments.items()]

    def for_environment(self, environment: str) -> Optional[Dict[str, Any]]:
        """The connector as deployed to ``environment``, or None when it isn't declared there."""
        if self.environments is None:
            return self._merge(ConnectorOverride())
        override = self.environments.get(environment)
        return self._merge(override) if override is not None else None

    def _merge(self, override: ConnectorOverride) -> Dict[str, Any]:
        return {
            "key": self.key,
            "type": self.type,
            "mode": self.mode,
            "name": override.name or self.name or self.key,
            "connection": override.connection or self.connection,
            "bridge": override.bridge or self.bridge,
            "domain": override.domain or self.domain,
            "config": {**self.config, **override.config},
            "agents": override.agents if override.agents is not None else self.agents,
        }
