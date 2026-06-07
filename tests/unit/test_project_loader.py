import asyncio
import inspect
from textwrap import dedent
from typing import Any, Dict, List, Optional, Union

import pytest

from connic import tools as connic_tools
from connic.loader import ProjectLoader

DOCUMENTED_PREDEFINED_TOOLS = [
    "trigger_agent",
    "trigger_agent_at",
    "query_knowledge",
    "store_knowledge",
    "delete_knowledge",
    "kb_list_namespaces",
    "web_search",
    "web_read_page",
    "db_find",
    "db_insert",
    "db_update",
    "db_delete",
    "db_count",
    "db_list_collections",
]


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n")


def test_documented_predefined_tools_can_be_referenced_from_agent_yaml(tmp_path):
    write_file(
        tmp_path / "agents" / "assistant.yaml",
        """
        version: "1.0"
        name: assistant
        type: llm
        model: openai/gpt-5.2
        description: "Assistant using Connic-managed tools"
        system_prompt: "Use the managed tools when they are relevant."
        tools:
          - trigger_agent
          - trigger_agent_at
          - query_knowledge
          - store_knowledge
          - delete_knowledge
          - kb_list_namespaces
          - web_search
          - web_read_page
          - db_find
          - db_insert
          - db_update
          - db_delete
          - db_count
          - db_list_collections
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("assistant")

    assert {tool.name for tool in agent.tools} == set(DOCUMENTED_PREDEFINED_TOOLS)
    assert all(tool.is_predefined for tool in agent.tools)
    assert set(DOCUMENTED_PREDEFINED_TOOLS).issubset(set(connic_tools.__all__))


def test_loads_realistic_support_agent_config(tmp_path):
    write_file(
        tmp_path / "schemas" / "support-response.json",
        """
        {
          "type": "object",
          "properties": {
            "answer": {"type": "string"},
            "needs_follow_up": {"type": "boolean"}
          },
          "required": ["answer"]
        }
        """,
    )
    write_file(
        tmp_path / "tools" / "support.py",
        '''
        from typing import Any, Dict, Optional


        def lookup_customer(
            customer_id: str,
            include_orders: bool = False,
            context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Look up a customer profile for support triage.

            Args:
                customer_id: Stable customer identifier from the support ticket.
                include_orders: Include recent order history when useful.
                context: Runtime context injected by Connic.
            """
            return {"customer_id": customer_id, "include_orders": include_orders}


        async def issue_refund(order_id: str, amount: float, reason: str = "customer_request") -> Dict[str, Any]:
            """Issue a customer refund after policy checks.

            Args:
                order_id: Order identifier to refund.
                amount: Refund amount in the order currency.
                reason: Internal reason code for the refund.
            """
            return {"order_id": order_id, "amount": amount, "reason": reason}


        def search_policy(query: str, limit: int = 3) -> Dict[str, Any]:
            """Search support policies without loading every policy tool upfront.

            Args:
                query: Plain-language policy search query.
                limit: Maximum policy snippets to return.
            """
            return {"query": query, "limit": limit, "results": []}
        ''',
    )
    refund_condition = "context.account_tier == 'enterprise' and input.refund_amount > 0"
    approval_condition = "param.amount > 100 and not context.manager_approved"
    write_file(
        tmp_path / "agents" / "support" / "support-agent.yaml",
        f"""
        version: "1.0"
        name: support-agent
        type: llm
        model: openai/gpt-5.2
        description: "Customer support agent with policy and refund tools"
        system_prompt: |
          Answer support tickets using customer context and policy data.
          Escalate refund work when approval is required.
        tools:
          - support.lookup_customer
          - support.issue_refund: {refund_condition}
          - query_knowledge
          - web_read_page
        discoverable_tools:
          - support.search_policy
        database:
          prevent_delete: true
          prevent_write: false
          collections:
            tickets:
              prevent_write: true
            audit-log: {{}}
        knowledge:
          prevent_delete: true
          namespaces:
            - policies
            - support.faq
        approval:
          tools:
            - support.issue_refund: {approval_condition}
            - db_delete
          timeout: 600
          message: "Approve refund before executing."
          on_rejection: continue
        mcp_servers:
          - name: internal-docs
            url: https://mcp.example.com/sse
            discoverable: true
        output_schema: support-response
        session:
          key: input.user_id
          ttl: 86400
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agent = loader.load_agent("support-agent")

    assert loader._load_errors == []
    assert agent.config.source_path == "agents/support/support-agent.yaml"
    assert agent.config.output_schema_dict["required"] == ["answer"]
    assert agent.config.session.key == "input.user_id"
    assert agent.config.approval.timeout == 600
    assert agent.config.approval.on_rejection == "continue"
    assert agent.config.approval.tools == [
        {"support.issue_refund": approval_condition},
        "db_delete",
    ]

    assert agent.config.database.prevent_delete is True
    assert agent.config.database.collections["tickets"].prevent_write is True
    assert agent.config.database.collections["audit-log"].prevent_write is None
    assert agent.config.knowledge.prevent_delete is True
    assert sorted(agent.config.knowledge.namespaces) == ["policies", "support.faq"]

    tools_by_name = {tool.name: tool for tool in agent.tools}
    assert {
        "lookup_customer",
        "issue_refund",
        "query_knowledge",
        "web_read_page",
        "search_tools",
        "use_tool",
    }.issubset(tools_by_name)
    assert tools_by_name["issue_refund"].condition == refund_condition
    assert tools_by_name["web_read_page"].is_predefined is True

    lookup_schema = tools_by_name["lookup_customer"].parameters
    assert lookup_schema["required"] == ["customer_id"]
    assert lookup_schema["properties"]["customer_id"]["type"] == "string"
    assert lookup_schema["properties"]["include_orders"]["type"] == "boolean"
    assert "context" not in lookup_schema["properties"]

    assert [tool.name for tool in agent.discoverable_tools] == ["search_policy"]
    assert agent.discoverable_tools[0].parameters["required"] == ["query"]


def test_loads_mcp_server_with_bridge(tmp_path):
    write_file(
        tmp_path / "agents" / "internal-mcp-agent.yaml",
        """
        version: "1.0"
        name: internal-mcp-agent
        type: llm
        model: openai/gpt-5.2
        description: "Agent that talks to a private MCP server via bridge"
        system_prompt: "Use the internal MCP tools to answer."
        mcp_servers:
          - name: internal-mcp
            url: http://mcp.internal:8080/mcp
            bridge: ${INTERNAL_BRIDGE_ID}
          - name: public-mcp
            url: https://mcp.example.com/mcp
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("internal-mcp-agent")

    servers_by_name = {s.name: s for s in agent.config.mcp_servers}
    assert servers_by_name["internal-mcp"].bridge == "${INTERNAL_BRIDGE_ID}"
    assert servers_by_name["public-mcp"].bridge is None


def test_validation_only_builds_tool_schema_without_importing_module(tmp_path):
    write_file(
        tmp_path / "tools" / "inventory.py",
        '''
        from typing import Dict, Optional
        import dependency_that_is_not_installed


        def reserve_item(
            sku: str,
            quantity: int,
            metadata: Optional[Dict[str, str]] = None,
            context: dict = {},
        ) -> dict:
            """Reserve stock for an order.

            Args:
                sku: Stock keeping unit to reserve.
                quantity: Number of units to reserve.
                metadata: Optional request metadata for auditing.
                context: Runtime context injected by Connic.
            """
            client = dependency_that_is_not_installed.Client()
            return client.reserve(sku, quantity, metadata)
        ''',
    )
    write_file(
        tmp_path / "agents" / "inventory-agent.yaml",
        """
        version: "1.0"
        name: inventory-agent
        type: llm
        model: openai/gpt-5.2
        description: "Reserves stock for paid orders"
        system_prompt: "Reserve stock only after payment is confirmed."
        tools:
          - inventory.reserve_item
        """,
    )

    agent = ProjectLoader(str(tmp_path), validation_only=True).load_agent("inventory-agent")

    reserve_item = agent.get_tool("reserve_item")
    assert reserve_item is not None
    assert reserve_item.description.startswith("Reserve stock for an order.")
    assert reserve_item.parameters["required"] == ["sku", "quantity"]
    assert reserve_item.parameters["properties"]["sku"]["type"] == "string"
    assert reserve_item.parameters["properties"]["quantity"]["type"] == "integer"
    assert reserve_item.parameters["properties"]["metadata"]["type"] == "object"
    assert "context" not in reserve_item.parameters["properties"]


def test_validation_only_infers_schema_from_ast_annotations_and_literal_defaults(tmp_path):
    write_file(
        tmp_path / "tools" / "support" / "enrichment.py",
        '''
        import dependency_that_is_not_installed
        from datetime import datetime


        async def enrich_ticket(
            ticket_id: "str",
            tags: list[str] = [],
            attributes: dict[str, int] = {},
            scheduled_for: datetime | None = None,
            context: dict = {},
        ) -> dict:
            """Enrich a support ticket before routing.

            Args:
                ticket_id: External ticket identifier from the helpdesk.
                tags: Routing tags already attached by upstream automation.
                attributes: Numeric attributes used for queue selection.
                scheduled_for: Optional follow-up time from the helpdesk.
                context: Runtime context injected by Connic.
            """
            client = dependency_that_is_not_installed.Client()
            return await client.enrich(ticket_id, tags, attributes, scheduled_for)
        ''',
    )
    write_file(
        tmp_path / "agents" / "support-router.yaml",
        """
        version: "1.0"
        name: support-router
        type: llm
        model: openai/gpt-5.2
        description: "Routes enriched support tickets"
        system_prompt: "Use ticket metadata to choose the right support queue."
        tools:
          - support.enrichment.enrich_ticket
        """,
    )

    agent = ProjectLoader(str(tmp_path), validation_only=True).load_agent("support-router")

    tool = agent.get_tool("enrich_ticket")
    assert tool is not None
    assert tool.is_async is True
    assert tool.parameters["required"] == ["ticket_id"]
    assert tool.parameters["properties"]["ticket_id"]["type"] == "string"
    assert tool.parameters["properties"]["tags"]["type"] == "array"
    assert tool.parameters["properties"]["attributes"]["type"] == "object"
    assert tool.parameters["properties"]["scheduled_for"]["type"] == "string"
    assert "context" not in tool.parameters["properties"]


def test_duplicate_tool_function_names_are_reported_with_resolved_refs(tmp_path):
    write_file(
        tmp_path / "tools" / "billing" / "notifications.py",
        """
        def send_receipt(invoice_id: str) -> dict:
            return {"invoice_id": invoice_id}
        """,
    )
    write_file(
        tmp_path / "tools" / "crm" / "notifications.py",
        """
        def send_receipt(customer_id: str) -> dict:
            return {"customer_id": customer_id}
        """,
    )
    write_file(
        tmp_path / "agents" / "billing-agent.yaml",
        """
        version: "1.0"
        name: billing-agent
        type: llm
        model: openai/gpt-5.2
        description: "Sends customer billing updates"
        system_prompt: "Send the right notification for the billing workflow."
        tools:
          - billing.notifications.send_receipt
          - crm.notifications.send_receipt
        """,
    )

    loader = ProjectLoader(str(tmp_path))

    assert loader.load_agents() == []
    assert len(loader._load_errors) == 1
    assert "Agent exposes duplicate tool names" in loader._load_errors[0]
    assert "billing.notifications.send_receipt" in loader._load_errors[0]
    assert "crm.notifications.send_receipt" in loader._load_errors[0]


def test_api_spec_tools_can_be_referenced_exactly_and_with_wildcards(tmp_path):
    billing_condition = "input.billing_enabled and context.account_tier == 'enterprise'"
    write_file(
        tmp_path / "agents" / "billing-assistant.yaml",
        f"""
        version: "1.0"
        name: billing-assistant
        type: llm
        model: openai/gpt-5.2
        description: "Uses generated API tools for billing support."
        system_prompt: "Use the billing API when customer context is available."
        tools:
          - api:stripe.customers_create
          - "api:stripe.invoices_*": {billing_condition}
        discoverable_tools:
          - api:internal.support_*
        """,
    )
    api_spec_tools = {
        "stripe": {
            "customers_create": {
                "description": "Create a customer in Stripe.",
                "method": "POST",
                "path": "/v1/customers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["email"],
                },
            },
            "invoices_get": {
                "method": "GET",
                "path": "/v1/invoices/{invoice_id}",
                "parameters": {
                    "type": "object",
                    "properties": {"invoice_id": {"type": "string"}},
                    "required": ["invoice_id"],
                },
            },
            "invoices_void": {
                "description": "Void a draft invoice.",
                "method": "POST",
                "path": "/v1/invoices/{invoice_id}/void",
            },
            "payment_methods_list": {
                "method": "GET",
                "path": "/v1/payment_methods",
            },
        },
        "internal": {
            "support_ticket_get": {
                "description": "Fetch the support ticket that triggered this run.",
                "parameters": {
                    "type": "object",
                    "properties": {"ticket_id": {"type": "string"}},
                    "required": ["ticket_id"],
                },
            },
            "users_get": {
                "description": "Fetch an internal user profile.",
            },
        },
    }

    agent = ProjectLoader(str(tmp_path), api_spec_tools=api_spec_tools).load_agent("billing-assistant")

    tools_by_name = {tool.name: tool for tool in agent.tools}
    assert set(tools_by_name) == {
        "customers_create",
        "invoices_get",
        "invoices_void",
        "search_tools",
        "use_tool",
    }

    customer_tool = tools_by_name["customers_create"]
    assert customer_tool.is_predefined is True
    assert customer_tool.func is None
    assert customer_tool.ref == "api:stripe.customers_create"
    assert customer_tool.description == "Create a customer in Stripe."
    assert customer_tool.parameters["required"] == ["email"]

    assert tools_by_name["invoices_get"].description == "GET /v1/invoices/{invoice_id}"
    assert tools_by_name["invoices_get"].condition == billing_condition
    assert tools_by_name["invoices_void"].condition == billing_condition
    assert "payment_methods_list" not in tools_by_name

    assert [tool.name for tool in agent.discoverable_tools] == ["support_ticket_get"]
    assert agent.discoverable_tools[0].ref == "api:internal.support_ticket_get"
    assert agent.discoverable_tools[0].parameters["required"] == ["ticket_id"]


def test_api_spec_tools_without_local_registry_are_recorded_as_warnings(tmp_path):
    write_file(
        tmp_path / "agents" / "billing-assistant.yaml",
        """
        version: "1.0"
        name: billing-assistant
        type: llm
        model: openai/gpt-5.2
        description: "References cloud-managed API spec tools during local validation."
        system_prompt: "Use the generated API tools when available."
        tools:
          - api:stripe.customers_get
          - api:internal.users_*
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agent = loader.load_agent("billing-assistant")

    assert agent.tools == []
    assert loader._load_errors == []
    assert loader._api_spec_warnings == [
        "api:stripe.customers_get",
        "api:internal.users_*",
    ]


@pytest.mark.parametrize(
    ("tool_ref", "expected_error"),
    [
        ("api:*", "Invalid API spec wildcard 'api:*'"),
        ("api:missing.users_*", "API spec 'missing' not found in registry"),
        ("api:stripe.refunds_*", "Wildcard 'api:stripe.refunds_*' matched no tools"),
        ("api:stripe", "Invalid API spec tool reference 'api:stripe'"),
        ("api:missing.customers_get", "API spec 'missing' not found in registry"),
        ("api:stripe.refunds_create", "Tool 'refunds_create' not found in API spec 'stripe'"),
    ],
)
def test_api_spec_tool_reference_errors_are_reported_with_actionable_messages(tmp_path, tool_ref, expected_error):
    write_file(
        tmp_path / "agents" / "billing-assistant.yaml",
        f"""
        version: "1.0"
        name: billing-assistant
        type: llm
        model: openai/gpt-5.2
        description: "Uses generated API tools for billing support."
        system_prompt: "Use the billing API when customer context is available."
        tools:
          - {tool_ref}
        """,
    )
    api_spec_tools = {
        "stripe": {
            "customers_get": {
                "method": "GET",
                "path": "/v1/customers/{{customer_id}}",
            },
            "invoices_get": {
                "method": "GET",
                "path": "/v1/invoices/{{invoice_id}}",
            },
        }
    }

    loader = ProjectLoader(str(tmp_path), api_spec_tools=api_spec_tools)
    agents = loader.load_agents()

    assert len(agents) == 1
    assert agents[0].config.name == "billing-assistant"
    assert agents[0].tools == []
    assert len(loader._load_errors) == 1
    assert expected_error in loader._load_errors[0]
    assert "billing-assistant" in loader._load_errors[0]


def test_guardrail_config_loads_documented_support_baseline(tmp_path):
    write_file(
        tmp_path / "agents" / "support-agent.yaml",
        """
        version: "1.0"
        name: support-agent
        type: llm
        model: openai/gpt-5.2
        description: "Protects customer support conversations."
        system_prompt: "Answer account questions without exposing private data."
        guardrails:
          input:
            - type: prompt_injection
              mode: block
            - type: pii
              mode: redact
              config:
                entities: [email, phone, ssn]
            - type: custom
              name: validate-ticket-id
              mode: block
          output:
            - type: moderation
              mode: block
            - type: pii_leakage
              mode: redact
              config:
                entities: [ssn, credit_card, api_key]
            - type: system_prompt_leakage
              mode: warn
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("support-agent")

    assert [(rule.type, rule.mode) for rule in agent.config.guardrails.input] == [
        ("prompt_injection", "block"),
        ("pii", "redact"),
        ("custom", "block"),
    ]
    assert agent.config.guardrails.input[1].config == {"entities": ["email", "phone", "ssn"]}
    assert agent.config.guardrails.input[2].name == "validate-ticket-id"
    assert [(rule.type, rule.mode) for rule in agent.config.guardrails.output] == [
        ("moderation", "block"),
        ("pii_leakage", "redact"),
        ("system_prompt_leakage", "warn"),
    ]


def test_custom_guardrail_with_documented_hyphenated_name_is_loaded_and_cached(tmp_path):
    write_file(
        tmp_path / "guardrails" / "validate-ticket-id.py",
        r'''
        import re
        from connic import GuardrailResult


        def check(content: str, context: dict) -> GuardrailResult:
            """Require a support ticket identifier before account data is used."""
            prefix = context.get("ticket_prefix", "TICKET")
            if re.search(rf"{prefix}-\d{{4,8}}", content):
                return GuardrailResult(passed=True)
            return GuardrailResult(
                passed=False,
                message=f"Please include a valid {prefix} ticket ID.",
                details={"prefix": prefix},
            )
        ''',
    )
    write_file(
        tmp_path / "guardrails" / "_draft.py",
        """
        def check(content: str, context: dict) -> bool:
            return False
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    guardrail = loader.load_guardrail("validate-ticket-id")

    assert loader.discover_guardrails() == ["validate-ticket-id"]
    assert loader.load_guardrail("validate-ticket-id") is guardrail
    assert guardrail.name == "validate-ticket-id"
    assert guardrail.is_async is False

    passed = guardrail.check("Customer references CASE-12345", {"ticket_prefix": "CASE"})
    failed = guardrail.check("Customer cannot find their invoice.", {"ticket_prefix": "CASE"})

    assert passed.passed is True
    assert failed.passed is False
    assert failed.message == "Please include a valid CASE ticket ID."
    assert failed.details == {"prefix": "CASE"}


def test_async_custom_guardrail_is_detected_and_executable(tmp_path):
    write_file(
        tmp_path / "guardrails" / "check-user-permissions.py",
        """
        from connic import GuardrailResult


        async def check(content: str, context: dict) -> GuardrailResult:
            if context.get("plan") == "enterprise":
                return GuardrailResult(passed=True)
            return GuardrailResult(passed=False, message="Enterprise plan required.")
        """,
    )

    guardrail = ProjectLoader(str(tmp_path)).load_guardrail("check-user-permissions")

    assert guardrail.is_async is True
    result = asyncio.run(guardrail.check("Show restricted account data.", {"plan": "free"}))
    assert result.passed is False
    assert result.message == "Enterprise plan required."


def test_invalid_guardrail_configuration_is_reported_with_agent_context(tmp_path):
    write_file(
        tmp_path / "agents" / "unsafe-agent.yaml",
        """
        version: "1.0"
        name: unsafe-agent
        type: llm
        model: openai/gpt-5.2
        description: "Invalid safety configuration."
        system_prompt: "Answer customer questions."
        guardrails:
          input:
            - type: prompt_injection
              mode: redact
          output:
            - type: custom
              mode: block
        """,
    )

    loader = ProjectLoader(str(tmp_path))

    assert loader.load_agents() == []
    assert len(loader._load_errors) == 1
    assert "agents/unsafe-agent.yaml" in loader._load_errors[0]
    assert "Mode 'redact' is only supported for 'pii' and 'pii_leakage'" in loader._load_errors[0]


@pytest.mark.parametrize(
    ("guardrail_yaml", "expected_message"),
    [
        (
            """
            input:
              - type: custom
                mode: block
            """,
            "Custom guardrails require a 'name' field pointing to the guardrail file",
        ),
        (
            """
            output:
              - type: account_takeover
                mode: block
            """,
            "Unknown guardrail type 'account_takeover'",
        ),
    ],
)
def test_invalid_custom_guardrail_rules_are_reported_with_agent_context(
    tmp_path,
    guardrail_yaml,
    expected_message,
):
    write_file(
        tmp_path / "agents" / "support-agent.yaml",
        f"""
        version: "1.0"
        name: support-agent
        type: llm
        model: openai/gpt-5.2
        description: "Invalid custom guardrail configuration."
        system_prompt: "Answer customer questions."
        guardrails:
        {guardrail_yaml}
        """,
    )

    loader = ProjectLoader(str(tmp_path))

    assert loader.load_agents() == []
    assert len(loader._load_errors) == 1
    assert "agents/support-agent.yaml" in loader._load_errors[0]
    assert expected_message in loader._load_errors[0]


def test_guardrail_without_callable_check_is_reported(tmp_path):
    write_file(
        tmp_path / "guardrails" / "broken-policy.py",
        """
        check = "not callable"
        """,
    )

    loader = ProjectLoader(str(tmp_path))

    assert loader.load_guardrail("broken-policy") is None
    assert loader._load_errors == [
        "Custom guardrail 'broken-policy': missing or non-callable 'check' function"
    ]


def test_missing_or_crashing_custom_guardrail_returns_none_without_poisoning_discovery(tmp_path):
    write_file(
        tmp_path / "guardrails" / "crashing-policy.py",
        """
        raise RuntimeError("missing vendor SDK")
        """,
    )

    loader = ProjectLoader(str(tmp_path))

    assert loader.load_guardrail("missing-policy") is None
    assert loader.load_guardrail("crashing-policy") is None
    assert loader.discover_guardrails() == ["crashing-policy"]


# ---------------------------------------------------------------------------
# A/B test variant detection
# ---------------------------------------------------------------------------

def test_ab_test_variant_is_detected_when_base_agent_exists(tmp_path):
    write_file(
        tmp_path / "agents" / "support.yaml",
        """
        version: "1.0"
        name: support
        type: llm
        model: openai/gpt-5.2
        description: "Base support agent"
        system_prompt: "Help customers."
        """,
    )
    write_file(
        tmp_path / "agents" / "support-test-fast.yaml",
        """
        version: "1.0"
        name: support-test-fast
        type: llm
        model: openai/gpt-5.2
        description: "Fast variant of support"
        system_prompt: "Help customers quickly."
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    by_name = {a.config.name: a for a in agents}

    assert "support" in by_name
    assert "support-test-fast" in by_name
    assert by_name["support-test-fast"].config.is_test_variant is True
    assert by_name["support-test-fast"].config.base_agent_name == "support"
    assert by_name["support-test-fast"].config.test_name == "fast"
    assert by_name["support"].config.is_test_variant is False


def test_ab_test_variant_without_base_agent_is_a_load_error(tmp_path):
    write_file(
        tmp_path / "agents" / "orphan-test-v2.yaml",
        """
        version: "1.0"
        name: orphan-test-v2
        type: llm
        model: openai/gpt-5.2
        description: "Orphan variant"
        system_prompt: "No base."
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    loader.load_agents()
    assert any("orphan-test-v2" in e and "no matching base agent" in e for e in loader._load_errors)


# ---------------------------------------------------------------------------
# Tool agent type
# ---------------------------------------------------------------------------

def test_tool_agent_resolves_single_tool(tmp_path):
    write_file(
        tmp_path / "tools" / "notifier.py",
        """
        def send_email(to: str, subject: str, body: str) -> dict:
            \"\"\"Send an email notification.

            Args:
                to: Recipient email address.
                subject: Email subject line.
                body: Email body content.
            \"\"\"
            return {"to": to, "subject": subject, "body": body}
        """,
    )
    write_file(
        tmp_path / "agents" / "email-sender.yaml",
        """
        version: "1.0"
        name: email-sender
        type: tool
        tool_name: notifier.send_email
        description: "Send an email"
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("email-sender")
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "send_email"
    assert agent.tools[0].is_async is False


def test_tool_agent_with_invalid_tool_reports_error(tmp_path):
    write_file(
        tmp_path / "agents" / "broken-tool.yaml",
        """
        version: "1.0"
        name: broken-tool
        type: tool
        tool_name: nonexistent.function
        description: "Invalid tool reference"
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    assert len(agents) == 1
    assert any("cannot resolve tool" in e for e in loader._load_errors)


# ---------------------------------------------------------------------------
# Sequential agent
# ---------------------------------------------------------------------------

def test_sequential_agent_loads_without_tools(tmp_path):
    write_file(
        tmp_path / "agents" / "pipeline.yaml",
        """
        version: "1.0"
        name: pipeline
        type: sequential
        description: "Orchestration pipeline"
        agents:
          - classify
          - respond
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("pipeline")
    assert agent.config.type.value == "sequential"
    assert agent.tools == []
    assert agent.config.agents == ["classify", "respond"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_load_agents_raises_when_agents_dir_missing(tmp_path):
    loader = ProjectLoader(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Agents directory not found"):
        loader.load_agents()


def test_load_agent_raises_when_agent_not_found(tmp_path):
    (tmp_path / "agents").mkdir()
    loader = ProjectLoader(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not found"):
        loader.load_agent("nonexistent")


def test_empty_agent_yaml_is_reported_as_error(tmp_path):
    write_file(tmp_path / "agents" / "empty.yaml", "")

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    assert agents == []
    assert any("empty" in e.lower() or "Empty" in e for e in loader._load_errors)


def test_duplicate_agent_names_across_files_are_reported(tmp_path):
    for subdir in ("a", "b"):
        write_file(
            tmp_path / "agents" / subdir / "agent.yaml",
            """
            version: "1.0"
            name: shared-name
            type: llm
            model: openai/gpt-5.2
            description: "Duplicate"
            system_prompt: "Hello."
            """,
        )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    assert len(agents) == 1
    assert any("duplicate agent name" in e for e in loader._load_errors)


def test_agent_discovery_skips_yaml_under_hidden_directory_segments(tmp_path):
    write_file(
        tmp_path / "agents" / "prod" / "agent.yaml",
        """
        version: "1.0"
        name: prod-agent
        type: llm
        model: openai/gpt-5.2
        description: "Production"
        system_prompt: "Hello."
        """,
    )
    write_file(
        tmp_path / "agents" / ".scratch" / "draft.yaml",
        """
        version: "1.0"
        name: should-not-load
        type: llm
        model: openai/gpt-5.2
        description: "Draft"
        system_prompt: "Hello."
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()

    assert len(agents) == 1
    assert agents[0].config.name == "prod-agent"
    assert not any("should-not-load" in e for e in loader._load_errors)


def test_symlinked_agent_directory_does_not_load_same_agent_twice(tmp_path):
    import os

    real = tmp_path / "agents" / "real"
    real.mkdir(parents=True)
    write_file(
        real / "agent.yaml",
        """
        version: "1.0"
        name: once-only
        type: llm
        model: openai/gpt-5.2
        description: "Single logical agent"
        system_prompt: "Hello."
        """,
    )
    try:
        os.symlink(real, tmp_path / "agents" / "alias", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks not available in this environment")

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()

    assert len(agents) == 1
    assert agents[0].config.name == "once-only"


# ---------------------------------------------------------------------------
# Tool overlap between tools and discoverable_tools
# ---------------------------------------------------------------------------

def test_overlap_between_tools_and_discoverable_tools_is_an_error(tmp_path):
    write_file(
        tmp_path / "tools" / "calc.py",
        """
        def add(a: int, b: int) -> int:
            return a + b
        """,
    )
    write_file(
        tmp_path / "agents" / "overlapper.yaml",
        """
        version: "1.0"
        name: overlapper
        type: llm
        model: openai/gpt-5.2
        description: "Agent with overlapping tool lists"
        system_prompt: "Do math."
        tools:
          - calc.add
        discoverable_tools:
          - calc.add
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    assert loader.load_agents() == []
    assert any("both 'tools' and 'discoverable_tools'" in e for e in loader._load_errors)


# ---------------------------------------------------------------------------
# Wildcard tool resolution
# ---------------------------------------------------------------------------

def test_wildcard_resolves_matching_functions(tmp_path):
    write_file(
        tmp_path / "tools" / "billing.py",
        """
        def create_invoice(customer: str) -> dict:
            \"\"\"Create a new invoice.\"\"\"
            return {}

        def send_invoice(invoice_id: str) -> dict:
            \"\"\"Send an invoice.\"\"\"
            return {}

        def _private_helper():
            pass
        """,
    )
    write_file(
        tmp_path / "agents" / "biller.yaml",
        """
        version: "1.0"
        name: biller
        type: llm
        model: openai/gpt-5.2
        description: "Billing agent"
        system_prompt: "Handle billing."
        tools:
          - "billing.*"
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("biller")
    tool_names = {t.name for t in agent.tools}
    assert "create_invoice" in tool_names
    assert "send_invoice" in tool_names
    assert "_private_helper" not in tool_names


def test_wildcard_no_matches_is_a_load_error(tmp_path):
    write_file(
        tmp_path / "tools" / "billing.py",
        """
        def create_invoice(customer: str) -> dict:
            return {}
        """,
    )
    write_file(
        tmp_path / "agents" / "biller.yaml",
        """
        version: "1.0"
        name: biller
        type: llm
        model: openai/gpt-5.2
        description: "Billing agent"
        system_prompt: "Handle billing."
        tools:
          - "billing.zzz_*"
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    assert any("matched no tools" in e for e in loader._load_errors)


# ---------------------------------------------------------------------------
# Invalid tool references
# ---------------------------------------------------------------------------

def test_invalid_tool_ref_single_part(tmp_path):
    write_file(
        tmp_path / "agents" / "bad.yaml",
        """
        version: "1.0"
        name: bad
        type: llm
        model: openai/gpt-5.2
        description: "Bad tool ref"
        system_prompt: "Hello."
        tools:
          - noseparator
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    assert any("Invalid tool reference" in e for e in loader._load_errors)


def test_tool_function_not_found_in_module(tmp_path):
    write_file(
        tmp_path / "tools" / "calc.py",
        """
        def add(a: int, b: int) -> int:
            return a + b
        """,
    )
    write_file(
        tmp_path / "agents" / "bad.yaml",
        """
        version: "1.0"
        name: bad
        type: llm
        model: openai/gpt-5.2
        description: "Missing function"
        system_prompt: "Hello."
        tools:
          - calc.nonexistent_function
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    assert any("not found in module" in e for e in loader._load_errors)


def test_tool_reference_to_module_variable_is_reported_as_non_callable(tmp_path):
    write_file(
        tmp_path / "tools" / "billing.py",
        """
        refund_limits = {"default": 100}
        """,
    )
    write_file(
        tmp_path / "agents" / "bad.yaml",
        """
        version: "1.0"
        name: bad
        type: llm
        model: openai/gpt-5.2
        description: "Invalid callable reference"
        system_prompt: "Hello."
        tools:
          - billing.refund_limits
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    assert len(agents) == 1
    assert agents[0].tools == []
    assert any("is not callable" in e for e in loader._load_errors)


def test_imported_helper_function_is_not_exposed_as_project_tool(tmp_path):
    write_file(
        tmp_path / "tools" / "billing.py",
        """
        from math import sqrt


        def calculate_total(amount: float) -> float:
            return amount
        """,
    )
    write_file(
        tmp_path / "agents" / "bad.yaml",
        """
        version: "1.0"
        name: bad
        type: llm
        model: openai/gpt-5.2
        description: "Invalid imported tool reference"
        system_prompt: "Hello."
        tools:
          - billing.sqrt
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    assert len(agents) == 1
    assert agents[0].tools == []
    assert any("is not defined in module" in e for e in loader._load_errors)


# ---------------------------------------------------------------------------
# Conditional tool with bad syntax
# ---------------------------------------------------------------------------

def test_conditional_tool_with_invalid_syntax_raises(tmp_path):
    write_file(
        tmp_path / "agents" / "bad-cond.yaml",
        """
        version: "1.0"
        name: bad-cond
        type: llm
        model: openai/gpt-5.2
        description: "Bad condition"
        system_prompt: "Hello."
        tools:
          - trigger_agent: "context. == "
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()
    assert agents == []
    assert any("Invalid condition syntax" in e or "SyntaxError" in e for e in loader._load_errors)


# ---------------------------------------------------------------------------
# Schema loading and discovery
# ---------------------------------------------------------------------------

def test_load_schema_and_discovery(tmp_path):
    write_file(
        tmp_path / "schemas" / "order.json",
        """
        {
          "type": "object",
          "properties": {"total": {"type": "number"}},
          "required": ["total"]
        }
        """,
    )
    write_file(
        tmp_path / "schemas" / "_draft.json",
        """{"type": "string"}""",
    )

    loader = ProjectLoader(str(tmp_path))
    schema = loader._load_schema("order")
    assert schema["properties"]["total"]["type"] == "number"
    # Cached
    assert loader._load_schema("order") is schema
    # Discovery skips underscore-prefixed
    assert loader.discover_schemas() == ["order"]


def test_load_schema_missing():
    loader = ProjectLoader("/tmp/nonexistent")
    with pytest.raises(FileNotFoundError, match="Schema"):
        loader._load_schema("missing")


def test_load_schema_invalid_json(tmp_path):
    write_file(tmp_path / "schemas" / "bad.json", '"just a string"')
    loader = ProjectLoader(str(tmp_path))
    with pytest.raises(ValueError, match="must be a JSON object"):
        loader._load_schema("bad")


def test_load_schema_missing_type(tmp_path):
    write_file(tmp_path / "schemas" / "notype.json", '{"properties": {}}')
    loader = ProjectLoader(str(tmp_path))
    with pytest.raises(ValueError, match="must have a 'type' field"):
        loader._load_schema("notype")


def test_discover_schemas_empty(tmp_path):
    loader = ProjectLoader(str(tmp_path))
    assert loader.discover_schemas() == []


# ---------------------------------------------------------------------------
# Middleware loading and discovery
# ---------------------------------------------------------------------------

def test_load_middleware_with_before_and_after(tmp_path):
    write_file(
        tmp_path / "middleware" / "support.py",
        """
        async def before(content, context):
            context["enriched"] = True
            return content

        async def after(response, context):
            return response + " [reviewed]"
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    mw = loader.load_middleware("support")
    assert mw is not None
    assert mw.before is not None
    assert mw.after is not None
    # Cached
    assert loader.load_middleware("support") is mw


def test_load_middleware_missing_returns_none(tmp_path):
    loader = ProjectLoader(str(tmp_path))
    assert loader.load_middleware("nonexistent") is None


def test_load_middleware_no_hooks_returns_none(tmp_path):
    write_file(
        tmp_path / "middleware" / "empty.py",
        """
        # No before or after functions
        helper = "not a hook"
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    assert loader.load_middleware("empty") is None


def test_load_middleware_ignores_non_callable_hook_attributes(tmp_path):
    write_file(
        tmp_path / "middleware" / "support.py",
        """
        before = "disabled by configuration"

        def after(response, context):
            return response
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    middleware = loader.load_middleware("support")

    assert middleware is not None
    assert middleware.before is None
    assert middleware.after is not None
    assert loader.discover_middlewares() == {"support": ["after"]}


def test_load_middleware_import_failure_returns_none_and_logs_warning(tmp_path, capsys):
    write_file(
        tmp_path / "middleware" / "support.py",
        """
        raise RuntimeError("missing CRM SDK")
        """,
    )

    loader = ProjectLoader(str(tmp_path))

    assert loader.load_middleware("support") is None
    assert loader.discover_middlewares() == {}
    assert "Failed to load middleware for support" in capsys.readouterr().out


def test_discover_middlewares(tmp_path):
    write_file(
        tmp_path / "middleware" / "support.py",
        """
        async def before(content, context):
            return content
        """,
    )
    write_file(
        tmp_path / "middleware" / "_draft.py",
        """
        async def before(content, context):
            return content
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    middlewares = loader.discover_middlewares()
    assert "support" in middlewares
    assert "before" in middlewares["support"]
    assert "_draft" not in middlewares


def test_discover_middlewares_empty(tmp_path):
    loader = ProjectLoader(str(tmp_path))
    assert loader.discover_middlewares() == {}


# ---------------------------------------------------------------------------
# Tool hooks loading and discovery
# ---------------------------------------------------------------------------

def test_load_tool_hooks_with_before_and_after(tmp_path):
    write_file(
        tmp_path / "hooks" / "assistant.py",
        """
        async def before(tool_name, params, context):
            return params

        async def after(tool_name, params, result, context):
            return result
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    hook = loader.load_tool_hooks("assistant")
    assert hook is not None
    assert hook.before is not None
    assert hook.after is not None
    # Cached
    assert loader.load_tool_hooks("assistant") is hook


def test_load_tool_hooks_missing_returns_none(tmp_path):
    loader = ProjectLoader(str(tmp_path))
    assert loader.load_tool_hooks("nonexistent") is None


def test_load_tool_hooks_no_functions_returns_none(tmp_path):
    write_file(
        tmp_path / "hooks" / "bare.py",
        """
        helper_var = 42
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    assert loader.load_tool_hooks("bare") is None


def test_load_tool_hooks_ignores_non_callable_hook_attributes(tmp_path):
    write_file(
        tmp_path / "hooks" / "billing.py",
        """
        before = {"disabled": True}

        def after(tool_name, params, result, context):
            return result
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    hook = loader.load_tool_hooks("billing")

    assert hook is not None
    assert hook.before is None
    assert hook.after is not None
    assert loader.discover_tool_hooks() == {"billing": ["after"]}


def test_load_tool_hooks_import_failure_returns_none_and_logs_warning(tmp_path, capsys):
    write_file(
        tmp_path / "hooks" / "billing.py",
        """
        raise RuntimeError("missing audit SDK")
        """,
    )

    loader = ProjectLoader(str(tmp_path))

    assert loader.load_tool_hooks("billing") is None
    assert loader.discover_tool_hooks() == {}
    assert "Failed to load tool hooks for billing" in capsys.readouterr().out


def test_discover_tool_hooks(tmp_path):
    write_file(
        tmp_path / "hooks" / "billing.py",
        """
        async def before(tool_name, params, context):
            return params
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    hooks = loader.discover_tool_hooks()
    assert "billing" in hooks
    assert "before" in hooks["billing"]


def test_discover_tool_hooks_empty(tmp_path):
    loader = ProjectLoader(str(tmp_path))
    assert loader.discover_tool_hooks() == {}


# ---------------------------------------------------------------------------
# discover_tools
# ---------------------------------------------------------------------------

def test_discover_tools(tmp_path):
    write_file(
        tmp_path / "tools" / "math.py",
        """
        def add(a: int, b: int) -> int:
            return a + b

        def subtract(a: int, b: int) -> int:
            return a - b

        def _private():
            pass
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    tools = loader.discover_tools()
    assert "math" in tools
    assert sorted(tools["math"]) == ["add", "subtract"]


def test_discover_tools_empty(tmp_path):
    loader = ProjectLoader(str(tmp_path))
    assert loader.discover_tools() == {}


# ---------------------------------------------------------------------------
# _type_to_schema – generic types
# ---------------------------------------------------------------------------

def test_type_to_schema_optional():
    loader = ProjectLoader("/tmp")
    schema = loader._type_to_schema(Optional[str])
    assert schema["type"] == "string"


def test_type_to_schema_list_of_int():
    loader = ProjectLoader("/tmp")
    schema = loader._type_to_schema(List[int])
    assert schema["type"] == "array"
    assert schema["items"]["type"] == "integer"


def test_type_to_schema_dict_str_any():
    loader = ProjectLoader("/tmp")
    schema = loader._type_to_schema(Dict[str, Any])
    assert schema["type"] == "object"


def test_type_to_schema_plain_list():
    loader = ProjectLoader("/tmp")
    schema = loader._type_to_schema(list)
    assert schema["type"] == "array"


def test_type_to_schema_none_type():
    loader = ProjectLoader("/tmp")
    schema = loader._type_to_schema(type(None))
    assert schema["type"] == "null"


def test_type_to_schema_union_non_optional():
    loader = ProjectLoader("/tmp")
    schema = loader._type_to_schema(Union[str, int])
    # Falls through to first non-None arg
    assert schema["type"] == "string"


def test_type_to_schema_unknown_type():
    loader = ProjectLoader("/tmp")
    schema = loader._type_to_schema(object)
    assert schema["type"] == "string"


def test_type_to_schema_no_annotation():
    loader = ProjectLoader("/tmp")
    schema = loader._type_to_schema(inspect.Parameter.empty)
    assert schema["type"] == "string"


# ---------------------------------------------------------------------------
# _parse_docstring_params
# ---------------------------------------------------------------------------

def test_parse_docstring_params_google_style():
    loader = ProjectLoader("/tmp")
    doc = """Do something.

    Args:
        name: The user's name.
        age (int): User age in years.
            This is a multi-line description.

    Returns:
        dict: result
    """
    params = loader._parse_docstring_params(doc)
    assert params["name"] == "The user's name."
    assert "User age" in params["age"]
    assert "multi-line" in params["age"]


def test_parse_docstring_params_empty():
    loader = ProjectLoader("/tmp")
    assert loader._parse_docstring_params("") == {}
    assert loader._parse_docstring_params(None) == {}


def test_parse_docstring_params_no_args_section():
    loader = ProjectLoader("/tmp")
    doc = """A simple function with no documented args."""
    assert loader._parse_docstring_params(doc) == {}


# ---------------------------------------------------------------------------
# output_schema for non-LLM agent prints warning
# ---------------------------------------------------------------------------

def test_output_schema_on_tool_agent_prints_warning(tmp_path, capsys):
    write_file(
        tmp_path / "schemas" / "out.json",
        '{"type": "object", "properties": {}}',
    )
    write_file(
        tmp_path / "tools" / "echo.py",
        """
        def echo(text: str) -> str:
            return text
        """,
    )
    write_file(
        tmp_path / "agents" / "tool-with-schema.yaml",
        """
        version: "1.0"
        name: tool-with-schema
        type: tool
        tool_name: echo.echo
        description: "Tool agent with output schema"
        output_schema: out
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("tool-with-schema")
    assert agent.config.output_schema_dict is None
    captured = capsys.readouterr()
    assert "only supported for LLM agents" in captured.out


# ---------------------------------------------------------------------------
# Database config – advanced dict format with per-collection overrides
# ---------------------------------------------------------------------------

def test_database_advanced_dict_format(tmp_path):
    write_file(
        tmp_path / "agents" / "db-agent.yaml",
        """
        version: "1.0"
        name: db-agent
        type: llm
        model: openai/gpt-5.2
        description: "Agent with advanced DB config"
        system_prompt: "Query data."
        database:
          prevent_delete: true
          collections:
            orders:
              prevent_write: true
            logs:
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("db-agent")
    assert agent.config.database.collections["orders"].prevent_write is True
    assert agent.config.database.collections["logs"].prevent_write is None


# ---------------------------------------------------------------------------
# Knowledge config – advanced dict format
# ---------------------------------------------------------------------------

def test_knowledge_advanced_dict_format(tmp_path):
    write_file(
        tmp_path / "agents" / "kb-agent.yaml",
        """
        version: "1.0"
        name: kb-agent
        type: llm
        model: openai/gpt-5.2
        description: "Agent with advanced knowledge config"
        system_prompt: "Search knowledge."
        knowledge:
          prevent_delete: true
          namespaces:
            policies:
              prevent_write: true
            faq:
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("kb-agent")
    assert agent.config.knowledge.namespaces["policies"].prevent_write is True
    assert agent.config.knowledge.namespaces["faq"].prevent_write is None


# ---------------------------------------------------------------------------
# _parse_tool_entry edge cases
# ---------------------------------------------------------------------------

def test_parse_tool_entry_dict_with_multiple_keys_raises(tmp_path):
    loader = ProjectLoader(str(tmp_path))
    with pytest.raises(ValueError, match="exactly one key"):
        loader._parse_tool_entry({"a": "cond1", "b": "cond2"}, "tool")


def test_parse_tool_entry_invalid_type_raises(tmp_path):
    loader = ProjectLoader(str(tmp_path))
    with pytest.raises(ValueError, match="Invalid"):
        loader._parse_tool_entry(42, "tool")


# ---------------------------------------------------------------------------
# Cascading _defaults.yaml
# ---------------------------------------------------------------------------

def test_root_defaults_apply_to_flat_agent(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        type: llm
        model: openai/gpt-5.2
        temperature: 0.4
        system_prompt: "Default prompt."
        """,
    )
    write_file(
        tmp_path / "agents" / "assistant.yaml",
        """
        version: "1.0"
        name: assistant
        description: "Inherits model and temperature from root defaults."
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("assistant")

    assert agent.config.model == "openai/gpt-5.2"
    assert agent.config.temperature == 0.4
    assert agent.config.system_prompt == "Default prompt."


def test_deeper_defaults_override_shallower_scalars(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        type: llm
        model: openai/gpt-5.2
        temperature: 0.4
        system_prompt: "Inherited prompt."
        """,
    )
    write_file(
        tmp_path / "agents" / "process" / "_defaults.yaml",
        """
        version: "1.0"
        model: anthropic/claude-sonnet-4-6
        """,
    )
    write_file(
        tmp_path / "agents" / "process" / "ingest" / "loader.yaml",
        """
        version: "1.0"
        name: loader
        description: "Inherits root temperature and process model."
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("loader")

    assert agent.config.model == "anthropic/claude-sonnet-4-6"
    assert agent.config.temperature == 0.4


def test_agent_overrides_inherited_scalar(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        type: llm
        model: openai/gpt-5.2
        temperature: 0.4
        system_prompt: "Default prompt."
        """,
    )
    write_file(
        tmp_path / "agents" / "assistant.yaml",
        """
        version: "1.0"
        name: assistant
        description: "Overrides model."
        model: anthropic/claude-haiku-4-5
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("assistant")

    assert agent.config.model == "anthropic/claude-haiku-4-5"
    assert agent.config.temperature == 0.4


def test_tool_list_concat_with_dedup_by_ref(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        type: llm
        model: openai/gpt-5.2
        system_prompt: "Default prompt."
        tools:
          - web_search
          - db_find
        """,
    )
    write_file(
        tmp_path / "agents" / "assistant.yaml",
        """
        version: "1.0"
        name: assistant
        description: "Adds and re-declares tools."
        tools:
          - db_insert
          - db_find: param.collection == "public"
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("assistant")

    tool_names = [t.name for t in agent.tools]
    # web_search inherited; db_find re-declared (so dropped from base, added by overlay with condition);
    # db_insert added by overlay.
    assert tool_names == ["web_search", "db_insert", "db_find"]
    db_find = next(t for t in agent.tools if t.name == "db_find")
    assert db_find.condition == 'param.collection == "public"'


def test_mcp_servers_dedup_by_name(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        type: llm
        model: openai/gpt-5.2
        system_prompt: "Default prompt."
        mcp_servers:
          - name: shared
            url: https://mcp.example.com/shared
          - name: kept
            url: https://mcp.example.com/kept
        """,
    )
    write_file(
        tmp_path / "agents" / "assistant.yaml",
        """
        version: "1.0"
        name: assistant
        description: "Overrides shared MCP server URL."
        mcp_servers:
          - name: shared
            url: https://mcp.example.com/shared-override
          - name: extra
            url: https://mcp.example.com/extra
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("assistant")

    by_name = {s.name: s for s in agent.config.mcp_servers}
    assert set(by_name) == {"shared", "kept", "extra"}
    assert by_name["shared"].url == "https://mcp.example.com/shared-override"


def test_database_collections_dict_deep_merge(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        type: llm
        model: openai/gpt-5.2
        system_prompt: "Default prompt."
        database:
          collections:
            audit_log:
              prevent_delete: true
              prevent_write: false
        """,
    )
    write_file(
        tmp_path / "agents" / "assistant.yaml",
        """
        version: "1.0"
        name: assistant
        description: "Adds its own collection."
        database:
          collections:
            orders:
              prevent_write: true
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("assistant")

    collections = agent.config.database.collections
    assert set(collections) == {"audit_log", "orders"}
    assert collections["audit_log"].prevent_delete is True
    assert collections["audit_log"].prevent_write is False
    assert collections["orders"].prevent_write is True


def test_defaults_file_is_not_loaded_as_agent(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        type: llm
        model: openai/gpt-5.2
        system_prompt: "Default prompt."
        """,
    )
    write_file(
        tmp_path / "agents" / "assistant.yaml",
        """
        version: "1.0"
        name: assistant
        description: "Only real agent."
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()

    assert [a.config.name for a in agents] == ["assistant"]
    assert loader._load_errors == []


def test_defaults_with_agent_identity_fields_is_load_error(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        name: should-not-be-here
        type: llm
        model: openai/gpt-5.2
        """,
    )
    write_file(
        tmp_path / "agents" / "assistant.yaml",
        """
        version: "1.0"
        name: assistant
        description: "Defaults file is broken."
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()

    assert agents == []
    assert len(loader._load_errors) == 1
    err = loader._load_errors[0]
    assert "agents/_defaults.yaml" in err
    assert "agent-identity fields" in err
    assert "name" in err


def test_agent_missing_required_fields_errors_even_when_defaults_supply_them(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        type: llm
        model: openai/gpt-5.2
        """,
    )
    write_file(
        tmp_path / "agents" / "broken.yaml",
        """
        version: "1.0"
        description: "Missing name."
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = loader.load_agents()

    assert agents == []
    assert len(loader._load_errors) == 1
    assert "missing required field" in loader._load_errors[0]
    assert "name" in loader._load_errors[0]


def test_defaults_apply_across_multiple_agents_in_same_dir(tmp_path):
    write_file(
        tmp_path / "agents" / "process" / "_defaults.yaml",
        """
        version: "1.0"
        type: llm
        model: openai/gpt-5.2
        system_prompt: "Default prompt."
        """,
    )
    write_file(
        tmp_path / "agents" / "process" / "a.yaml",
        """
        version: "1.0"
        name: process-a
        description: "Agent A."
        """,
    )
    write_file(
        tmp_path / "agents" / "process" / "b.yaml",
        """
        version: "1.0"
        name: process-b
        description: "Agent B."
        """,
    )

    loader = ProjectLoader(str(tmp_path))
    agents = {a.config.name: a for a in loader.load_agents()}

    assert agents["process-a"].config.model == "openai/gpt-5.2"
    assert agents["process-b"].config.model == "openai/gpt-5.2"
    assert loader._load_errors == []


def test_sequential_agents_list_concat_dedup(tmp_path):
    write_file(
        tmp_path / "agents" / "_defaults.yaml",
        """
        version: "1.0"
        type: sequential
        agents:
          - first
          - shared
        """,
    )
    write_file(
        tmp_path / "agents" / "pipeline.yaml",
        """
        version: "1.0"
        name: pipeline
        description: "Pipeline appends to defaults."
        agents:
          - shared
          - last
        """,
    )

    agent = ProjectLoader(str(tmp_path)).load_agent("pipeline")

    assert agent.config.agents == ["first", "shared", "last"]
