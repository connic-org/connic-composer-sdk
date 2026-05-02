"""
Predefined tools are stubs at import time; the runtime injects real implementations.

These tests document the developer-facing contract: calling a stub outside the CLI
or deployment yields a clear RuntimeError so agent code fails fast during local edits.
"""
import asyncio
import re

import pytest

from connic import tools


async def _expect_stub(coro):
    with pytest.raises(RuntimeError, match=re.escape("Run 'connic test'")):
        await coro


@pytest.mark.parametrize(
    "tool_fn, args, kwargs",
    [
        (tools.trigger_agent, ("summarizer", {"text": "hello"}), {}),
        (
            tools.trigger_agent_at,
            ("report-job", {"k": 1}),
            {"delay": {"h": 1}},
        ),
        (tools.query_knowledge, ("refund policy",), {}),
        (tools.store_knowledge, ("some content",), {}),
        (tools.delete_knowledge, ("entry-1",), {}),
        (tools.kb_list_namespaces, (), {}),
        (tools.web_search, ("latest AI news",), {}),
        (tools.web_read_page, ("https://example.com",), {}),
        (tools.db_find, ("orders",), {}),
        (tools.db_insert, ("orders", {"a": 1}), {}),
        (tools.db_update, ("orders", {"id": 1}, {"ok": True}), {}),
        (tools.db_delete, ("orders", {"id": 1}), {}),
        (tools.db_count, ("orders",), {}),
        (tools.db_list_collections, (), {}),
    ],
)
def test_predefined_tool_raises_until_injected(tool_fn, args, kwargs):
    asyncio.run(_expect_stub(tool_fn(*args, **kwargs)))
