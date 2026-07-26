"""
Bridge helpers for use inside custom tools, middleware, hooks, and guardrails.

The agent runtime intercepts hostnames matching `<target>.cnc-bridge-<bridge_id>`
and routes them through the named bridge. You can either build the hostname
inline or use this helper:

    from connic import bridge_host
    psycopg.connect(host=bridge_host("abc123", "postgres-primary"), port=5432, ...)

For protocols that discover another endpoint at runtime, bridge settings can
define exact or safe anchored-regex destination routes for one TCP port. Those
routes are limited to 32 per bridge and 256 per project; a regex may contain at
most one quantified character class. Routes do not bypass the bridge agent's
exact ALLOWED_HOSTS check. Custom native resolvers and later background-thread
connections should continue to use the explicit hostname returned by this
helper.
"""


def bridge_host(bridge_id: str, target: str) -> str:
    """Return the hostname that tunnels to `target` via `bridge_id`."""
    return f"{target}.cnc-bridge-{bridge_id}"
