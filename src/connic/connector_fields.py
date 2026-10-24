"""Config fields of connectors declared in code.

Mirrors connic's connector schemas for the keys a ``connectors/<key>.yaml`` file
can set under ``config``; a backend test keeps the two in sync. Credentials
belong to the Connection, and ``mode``, ``connection``, ``bridge`` and
``domain`` have their own place in the file.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

_VARIABLE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_INTEGER = re.compile(r"-?[0-9]+")

_INBOUND = {"mode": ["inbound"]}
_OUTBOUND = {"mode": ["outbound"]}
_RECEIVING = {"mode": ["inbound", "sync"]}

# connectors/*.yaml groups the rate limit fields under one ``rate_limit`` block.
RATE_LIMIT_BLOCK = {
    "total": "rate_limit_per_minute",
    "per_ip": "rate_limit_per_ip_per_minute",
    "key": "rate_limit_key",
    "per_key": "rate_limit_per_key_per_minute",
}
_RATE_LIMIT_LABELS = {name: f"rate_limit.{key}" for key, name in RATE_LIMIT_BLOCK.items()}


def _rate_limits(show_when: Optional[Dict[str, list]] = None) -> Dict[str, Dict[str, Any]]:
    condition = {"show_when": show_when} if show_when else {}
    limit = {"type": "number", "default": 0, "min": 0, "max": 1000, **condition}
    return {
        "rate_limit_per_minute": limit,
        "rate_limit_per_ip_per_minute": limit,
        "rate_limit_key": {"type": "text", **condition},
        "rate_limit_per_key_per_minute": limit,
    }


CONFIG_FIELDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "http": {
        "outbound_url": {"type": "url", "required": True, "show_when": _OUTBOUND},
        "require_auth": {"type": "boolean", "default": True, "show_when": _RECEIVING},
        **_rate_limits(_RECEIVING),
    },
    "cron": {
        # The dashboard suggests a daily schedule; a file sets its own.
        "schedule": {"type": "cron", "required": True},
        "timezone": {"type": "timezone", "default": "UTC"},
        "prompt": {"type": "textarea"},
    },
    "kafka": {
        "topic": {"type": "text", "required": True},
        "group_id": {"type": "text", "show_when": _INBOUND},
        "auto_offset_reset": {
            "type": "select", "default": "latest", "options": ["latest", "earliest"], "show_when": _INBOUND,
        },
    },
    "websocket": {
        "streaming": {"type": "boolean", "default": True},
        "progress_events": {"type": "boolean", "default": False, "show_when": {"streaming": [True]}},
        "require_auth": {"type": "boolean", "default": True},
        "session_timeout": {"type": "number", "default": 3600, "min": 60, "max": 86400},
        "max_messages_per_session": {"type": "number", "default": 100, "min": 1, "max": 10000},
        **_rate_limits(),
    },
    "mcp": {
        "require_auth": {"type": "boolean", "default": True},
        **_rate_limits(),
    },
    "sqs": {
        "queue_url": {"type": "url", "required": True},
        "max_messages": {"type": "number", "default": 10, "min": 1, "max": 10, "show_when": _INBOUND},
        "wait_time_seconds": {"type": "number", "default": 20, "min": 0, "max": 20, "show_when": _INBOUND},
        "visibility_timeout": {"type": "number", "default": 300, "min": 6, "max": 43200, "show_when": _INBOUND},
        "message_group_id": {"type": "text", "show_when": _OUTBOUND},
    },
    "s3": {
        "bucket_name": {"type": "text", "required": True},
        "prefix": {"type": "text"},
        "suffix": {"type": "text"},
        "event_types": {"type": "select", "default": "created", "options": ["created", "all"]},
        "include_content": {"type": "boolean", "default": False},
        "region": {"type": "text", "required": True, "show_when": {"include_content": [True]}},
        "max_file_size_mb": {"type": "number", "default": 10, "min": 1, "max": 10},
    },
    "postgres": {
        "channel": {"type": "text", "required": True},
        "parse_json_payload": {"type": "boolean", "default": True},
    },
    "email": {
        "mailbox": {"type": "text", "default": "INBOX", "show_when": _INBOUND},
        "mark_as_read": {"type": "boolean", "default": True, "show_when": _INBOUND},
        "filter_unread_only": {"type": "boolean", "default": True, "show_when": _INBOUND},
        "filter_from": {"type": "text", "show_when": _INBOUND},
        "filter_subject": {"type": "text", "show_when": _INBOUND},
        "from_address": {"type": "text", "required": True, "show_when": _OUTBOUND},
        "from_name": {"type": "text", "show_when": _OUTBOUND},
        "default_to": {"type": "text", "show_when": _OUTBOUND},
    },
    "stripe": {
        "webhook_secret": {"type": "password", "sensitive": True},
    },
    "telegram": {
        "bot_token": {"type": "password", "required": True, "sensitive": True},
        "allowed_user_ids": {"type": "textarea", "show_when": _INBOUND},
        "chat_id": {"type": "text", "accepts_integer": True, "show_when": _OUTBOUND},
    },
    "twilio": {
        "phone_number": {"type": "text", "required": True},
        "transfer_numbers": {"type": "textarea"},
        "fallback_number": {"type": "text"},
    },
    "twilio_messaging": {
        "channel": {"type": "select", "required": True, "default": "sms", "options": ["sms", "whatsapp", "rcs"]},
        "sender": {"type": "text", "required": True},
        "to": {"type": "text", "show_when": _OUTBOUND},
        "messaging_service_sid": {"type": "text", "show_when": _OUTBOUND},
        "content_sid": {"type": "text", "show_when": _OUTBOUND},
        "content_variables": {"type": "textarea", "show_when": _OUTBOUND},
    },
    "telnyx": {
        "phone_number": {"type": "text", "required": True},
        "transfer_numbers": {"type": "textarea"},
        "fallback_number": {"type": "text"},
    },
    "telnyx_messaging": {
        "channel": {"type": "select", "required": True, "default": "sms", "options": ["sms", "whatsapp", "rcs"]},
        "sender": {"type": "text", "required": True},
        "whatsapp_profile_id": {"type": "text", "show_when": {"channel": ["whatsapp"]}},
        "to": {"type": "text", "show_when": _OUTBOUND},
        "whatsapp_template": {"type": "textarea", "show_when": {"mode": ["outbound"], "channel": ["whatsapp"]}},
    },
}

_AWS = frozenset({"aws_access_key_id", "aws_secret_access_key", "endpoint_url", "bridge_id"})
_TWILIO = frozenset({"region", "account_sid", "auth_token"})
_TELNYX = frozenset({"api_key", "public_key"})
CONNECTION_FIELDS: Dict[str, frozenset[str]] = {
    "kafka": frozenset({
        "bootstrap_servers", "security_protocol", "sasl_mechanism", "sasl_username", "sasl_password",
        "ssl_ca_pem", "ssl_cert_pem", "ssl_key_pem", "ssl_password", "ssl_check_hostname", "bridge_id",
    }),
    "sqs": _AWS,
    "s3": _AWS,
    "postgres": frozenset({"host", "port", "database", "username", "password", "ssl_mode", "bridge_id"}),
    "email": frozenset({
        "imap_host", "imap_port", "imap_use_ssl", "imap_username", "imap_password",
        "smtp_host", "smtp_port", "smtp_use_tls", "smtp_username", "smtp_password", "ssl_ca_pem", "bridge_id",
    }),
    "twilio": _TWILIO,
    "twilio_messaging": _TWILIO,
    "telnyx": _TELNYX,
    "telnyx_messaging": _TELNYX,
}

# Settings that moved to the Connection. Until April 30, 2027 a file may keep them when they match the
# Connection's; the deploy checks that.
DEPRECATED_CONFIG_FIELDS: Dict[str, frozenset[str]] = {"s3": frozenset({"endpoint_url"})}

# Modes that can use a bridge, for types that allow it only in some modes.
BRIDGE_MODES: Dict[str, tuple[str, ...]] = {"http": ("outbound",)}

# Types whose inbound and sync events can carry files, so their agent links can set files_required.
FILE_TYPES = frozenset({
    "http", "mcp", "websocket", "email", "s3", "kafka", "sqs", "telegram", "twilio_messaging", "telnyx_messaging",
})

_CRON_MACROS = frozenset({"@yearly", "@annually", "@monthly", "@weekly", "@daily", "@midnight", "@hourly"})


def check_routes(connector_type: str, mode: str, *, bridge: Optional[str], domain: Optional[str]) -> None:
    """Bridges and custom domains apply only to some modes of the types that support them."""
    if bridge is not None and mode not in BRIDGE_MODES.get(connector_type, (mode,)):
        raise ValueError(f"{connector_type} connectors use a bridge only in {' or '.join(BRIDGE_MODES[connector_type])} mode")
    if domain is not None and mode == "outbound":
        raise ValueError("outbound connectors have no URL, so they can't use a custom domain")


def receives_files(connector_type: str, mode: str) -> bool:
    return connector_type in FILE_TYPES and mode != "outbound"


def check_destination(connector_type: str, mode: str, config: Dict[str, Any], *, bridge: Optional[str]) -> None:
    """Without a bridge, an outbound HTTP connector sends to an HTTPS URL on port 443.

    Whether its host resolves to a public address is checked when the connector is saved.
    """
    url = config.get("outbound_url")
    if connector_type != "http" or mode != "outbound" or bridge or not isinstance(url, str) or _has_variable(url):
        return
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("config.outbound_url must be an https:// URL; use a bridge to reach other destinations")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("config.outbound_url must not include credentials")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("config.outbound_url has an invalid port") from None
    if port not in (None, 443):
        raise ValueError("config.outbound_url must use port 443; use a bridge to reach other ports")


def _has_variable(value: Any) -> bool:
    if isinstance(value, str):
        return "${" in value
    if isinstance(value, dict):
        return any(_has_variable(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_variable(item) for item in value)
    return False


def _flatten(connector_type: str, config: Dict[str, Any], ignored: Optional[List[str]]) -> Dict[str, Any]:
    fields = CONFIG_FIELDS[connector_type]
    connection = CONNECTION_FIELDS.get(connector_type, frozenset())
    values: Dict[str, Any] = {}
    for name, value in config.items():
        if name == "rate_limit" and "rate_limit_per_minute" in fields:
            if not isinstance(value, dict):
                raise ValueError("config.rate_limit must be a mapping with total, per_ip, key or per_key")
            for key, item in value.items():
                if key not in RATE_LIMIT_BLOCK:
                    raise ValueError(f"config.rate_limit.{key} isn't a rate limit field; use total, per_ip, key or per_key")
                if item is not None:
                    values[RATE_LIMIT_BLOCK[key]] = item
        elif name in _RATE_LIMIT_LABELS and name in fields:
            raise ValueError(f"config.{name} isn't allowed; use config.{_RATE_LIMIT_LABELS[name]} instead")
        elif name in DEPRECATED_CONFIG_FIELDS.get(connector_type, ()):
            continue
        elif name in connection:
            raise ValueError(f"config.{name} belongs to the connection; set it on the Connection instead")
        elif name not in fields:
            if ignored is None:
                raise ValueError(f"config.{name} isn't a {connector_type} connector field")
            ignored.append(f"config.{name}")
        else:
            values[name] = value
    return values


def _condition_failure(field: Dict[str, Any], mode: str, values: Dict[str, Any], fields: Dict[str, Any]) -> Optional[str]:
    """Why the field doesn't apply, or None. Conditions that still hold a ${NAME} reference count as met."""
    for key, expected in (field.get("show_when") or {}).items():
        actual = mode if key == "mode" else values.get(key, fields[key].get("default"))
        if _has_variable(actual) or actual in expected:
            continue
        if key == "mode":
            return f"only applies to {' or '.join(expected)} connectors"
        allowed = (str(item).lower() if isinstance(item, bool) else item for item in expected)
        return f"only applies when config.{key} is {' or '.join(allowed)}"
    return None


def _is_cron(value: str) -> bool:
    """Five fields or a macro; a sixth field would be read as seconds at the end, not the start."""
    value = value.strip()
    return value in _CRON_MACROS if value.startswith("@") else len(value.split()) == 5


def _value(label: str, field: Dict[str, Any], value: Any) -> Any:
    kind = field["type"]
    if kind == "number":
        if isinstance(value, str) and _INTEGER.fullmatch(value.strip()):
            value = int(value)
        if type(value) is not int:
            raise ValueError(f"{label} must be a whole number")
        if not field["min"] <= value <= field["max"]:
            raise ValueError(f"{label} must be between {field['min']} and {field['max']}")
    elif kind == "boolean":
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            value = value.strip().lower() == "true"
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be true or false")
    elif kind == "select":
        if value not in field["options"]:
            raise ValueError(f"{label} must be one of {', '.join(field['options'])}")
    elif kind == "textarea":
        if isinstance(value, list):
            for index, item in enumerate(value):
                if not isinstance(item, str):
                    raise ValueError(f"{label}[{index}] must be a string; put it in quotes")
        elif not isinstance(value, (str, dict)):
            raise ValueError(f"{label} must be a string, a list or a mapping")
    elif field.get("accepts_integer") and type(value) is int:
        value = str(value)
    elif not isinstance(value, str):
        raise ValueError(f"{label} must be a string; put values such as phone numbers and IDs in quotes")
    elif kind == "cron" and not _is_cron(value):
        raise ValueError(
            f"{label} must have five fields (minute, hour, day of month, month, day of week) "
            "or be a macro such as @daily"
        )
    return value


def check_config(
    connector_type: str,
    mode: str,
    config: Dict[str, Any],
    *,
    resolved: bool = False,
    ignored: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check a connector's ``config`` and return the settings it declares, with defaults for the rest.

    Rate limits come back as flat fields and values as their field type. Until ``resolved``, values
    may still hold ``${NAME}`` references; the deploy checks them again once they are substituted.
    With ``ignored``, unknown fields and fields that don't apply to the mode are left out of the
    result and listed there instead of failing the check.
    """
    fields = CONFIG_FIELDS[connector_type]
    values = _flatten(connector_type, config, ignored)
    for name, value in list(values.items()):
        label = f"config.{_RATE_LIMIT_LABELS.get(name, name)}"
        field = fields[name]
        failure = _condition_failure(field, mode, values, fields)
        if failure and ignored is not None:
            ignored.append(label)
            del values[name]
            continue
        if failure:
            raise ValueError(f"{label} {failure}")
        if field.get("sensitive") and not resolved:
            if not (isinstance(value, str) and _VARIABLE.fullmatch(value)):
                raise ValueError(f"{label} is a secret; reference a variable such as ${{{name.upper()}}} instead")
        elif resolved or not (isinstance(value, str) and _has_variable(value)):
            values[name] = _value(label, field, value)

    for name, field in fields.items():
        if name in values or _condition_failure(field, mode, values, fields):
            continue
        if "default" in field:
            values[name] = field["default"]
        elif field.get("required"):
            raise ValueError(f"config.{_RATE_LIMIT_LABELS.get(name, name)} is required")

    per_key, key = values.get("rate_limit_per_key_per_minute"), values.get("rate_limit_key")
    if type(per_key) is int and per_key > 0 and not (isinstance(key, str) and key.strip()):
        raise ValueError("config.rate_limit.key is required when config.rate_limit.per_key is set")
    return values
