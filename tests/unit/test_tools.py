"""
Predefined tools are stubs at import time; the runtime injects real implementations.

These tests document the developer-facing contract: calling a stub outside the CLI
or deployment yields a clear RuntimeError so agent code fails fast during local edits.
"""
import asyncio
import inspect
import re
from typing import Any, Dict, List, get_args, get_origin, get_type_hints

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
        (tools.retrieval_query, ("refund policy",), {}),
        (tools.retrieval_store, ("some content",), {}),
        (tools.retrieval_delete, ("entry-1",), {}),
        (tools.retrieval_list_namespaces, (), {}),
        (tools.web_search, ("latest AI news",), {}),
        (tools.web_read_page, ("https://example.com",), {}),
        (tools.db_find, ("orders",), {}),
        (tools.db_insert, ("orders", {"a": 1}), {}),
        (tools.db_update, ("orders", {"id": 1}, {"ok": True}), {}),
        (
            tools.db_upsert,
            ("orders", {"order_id": "ORD-1"}, {"status": "shipped"}),
            {"insert_only": {"source": "etl"}},
        ),
        (tools.db_delete, ("orders", {"id": 1}), {}),
        (tools.db_count, ("orders",), {}),
        (tools.db_list_collections, (), {}),
    ],
)
def test_predefined_tool_raises_until_injected(tool_fn, args, kwargs):
    asyncio.run(_expect_stub(tool_fn(*args, **kwargs)))


@pytest.mark.parametrize("tool_fn", [tools.trigger_agent, tools.trigger_agent_at])
def test_orchestration_payload_type_supports_structured_and_text_inputs(tool_fn):
    payload_type = get_type_hints(tool_fn)["payload"]

    assert get_origin(payload_type) is not None
    assert set(get_args(payload_type)) == {Dict[str, Any], List[Any], str}


def test_retrieval_tools_preserve_public_default_score():
    assert inspect.signature(tools.retrieval_query).parameters["min_score"].default == 0.3
    assert inspect.signature(tools.query_knowledge).parameters["min_score"].default == 0.3


@pytest.mark.parametrize(
    "alias, args",
    [
        ("query_knowledge", ("refund policy",)),
        ("store_knowledge", ("some content",)),
        ("delete_knowledge", ("entry-1",)),
        ("kb_list_namespaces", ()),
    ],
)
def test_legacy_retrieval_aliases_remain_callable_but_are_not_exported(alias, args):
    asyncio.run(_expect_stub(getattr(tools, alias)(*args)))
    assert alias not in tools.__all__


@pytest.mark.parametrize(
    "alias, canonical, args, kwargs, expected_args",
    [
        (
            "query_knowledge",
            "retrieval_query",
            ("refund policy",),
            {
                "namespace": "policies",
                "min_score": 0.5,
                "max_results": 7,
                "metadata_filter": {"status": "active"},
            },
            ("refund policy", "policies", 0.5, 7, {"status": "active"}),
        ),
        (
            "store_knowledge",
            "retrieval_store",
            ("Refunds are available for 30 days.",),
            {
                "entry_id": "refund-policy",
                "namespace": "policies",
                "metadata": {"owner": "support"},
            },
            (
                "Refunds are available for 30 days.",
                "refund-policy",
                "policies",
                {"owner": "support"},
            ),
        ),
        (
            "delete_knowledge",
            "retrieval_delete",
            (),
            {
                "entry_id": "refund-policy",
                "namespace": "policies",
                "metadata_filter": {"status": "stale"},
            },
            ("refund-policy", "policies", {"status": "stale"}),
        ),
        (
            "kb_list_namespaces",
            "retrieval_list_namespaces",
            (),
            {"parent": "policies", "depth": 0},
            ("policies", 0),
        ),
    ],
)
def test_legacy_retrieval_aliases_forward_to_canonical_tools(
    monkeypatch,
    alias,
    canonical,
    args,
    kwargs,
    expected_args,
):
    calls = []

    async def canonical_tool(*received_args, **received_kwargs):
        calls.append((received_args, received_kwargs))
        return {"canonical": canonical}

    monkeypatch.setattr(tools, canonical, canonical_tool)

    result = asyncio.run(getattr(tools, alias)(*args, **kwargs))

    assert result == {"canonical": canonical}
    assert calls == [(expected_args, {})]
