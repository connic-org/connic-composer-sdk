"""
Bridge helpers for use inside custom tools and middlewares.

The agent runtime intercepts hostnames matching `<target>.cnc-bridge-<bridge_id>`
and routes them through the named bridge. You can either build the hostname
inline or use this helper:

    from connic import bridge_host
    psycopg.connect(host=bridge_host("abc123", "postgres-primary"), port=5432, ...)
"""


def bridge_host(bridge_id: str, target: str) -> str:
    """Return the hostname that tunnels to `target` via `bridge_id`."""
    return f"{target}.cnc-bridge-{bridge_id}"
