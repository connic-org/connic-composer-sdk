import pytest
import yaml
from pydantic import ValidationError

from connic.test_schema import TestDefaults as ConnicTestDefaults
from connic.test_schema import TestFile as ConnicTestFile


def test_test_file_parses_realistic_yaml_and_resolves_defaults():
    doc = yaml.safe_load(
        """
        version: "1.0"
        agent: stress-tester
        defaults:
          runs: 5
          success_threshold: 80
          timeout_s: 60
        tests:
          - name: adds_two_numbers
            payload: '{"message": "add 4 and 6", "a": 4, "b": 6}'
            expected_result: status == "completed" and output.id == 10
            expected_tool_calls:
              - math.calculator.add
              - math.calculator.add: invocations >= 1
            expected_tool_call_order:
              - math.calculator.add
              - notifications.send
            expected_no_tool_calls:
              - email.send
          - name: high_concurrency_smoke
            payload: '{"message": "stress ping"}'
            runs: 20
            success_threshold: 95
            timeout_s: 90
            expected_result: status == "completed"
        """
    )

    test_file = ConnicTestFile.model_validate(doc)

    assert test_file.version == "1.0"
    assert test_file.agent == "stress-tester"
    assert test_file.defaults == ConnicTestDefaults(runs=5, success_threshold=80, timeout_s=60)

    first = test_file.resolved(test_file.tests[0])
    assert first == {
        "name": "adds_two_numbers",
        "payload": '{"message": "add 4 and 6", "a": 4, "b": 6}',
        "files": [],
        "builder": None,
        "builder_args": None,
        "mocks": None,
        "strict_mocks": False,
        "strict_hook_mocks": False,
        "strict_middleware_mocks": False,
        "strict_guardrail_mocks": False,
        "approval_decisions": [],
        "strict_approval_decisions": False,
        "runs": 5,
        "success_threshold": 80,
        "timeout_s": 60,
        "expected_result": 'status == "completed" and output.id == 10',
        "expected_tool_calls": ["math.calculator.add", {"math.calculator.add": "invocations >= 1"}],
        "expected_tool_call_order": ["math.calculator.add", "notifications.send"],
        "expected_no_tool_calls": ["email.send"],
        "expected_child_agents": None,
    }

    second = test_file.resolved(test_file.tests[1])
    assert second["runs"] == 20
    assert second["success_threshold"] == 95
    assert second["timeout_s"] == 90
    assert second["expected_tool_calls"] == []
    assert second["expected_tool_call_order"] == []
    assert second["expected_no_tool_calls"] == []


def test_test_file_resolves_nested_child_agent_expectations():
    doc = yaml.safe_load(
        """
        tests:
          - name: routes_refund_to_child_agents
            builder: create_refund_case
            builder_args:
              amount_cents: 4200
            expected_child_agents:
              refund-specialist:
                expected_triggered: 2
                expected_payload: payload.charge_id == context.charge_id
                expected_result: output.status == "refunded"
                expected_tool_calls:
                  - billing.refund
                  - billing.refund: params.charge_id == context.charge_id
                expected_tool_call_order:
                  - billing.refund
                  - ledger.record_refund
                expected_no_tool_calls:
                  - email.send
                expected_child_agents:
                  ledger-writer:
                    expected_payload: payload.refund_id == output.refund_id
                    expected_tool_calls:
                      - ledger.record_refund
                    expected_tool_call_order:
                      - ledger.record_refund
              telemetry:
                expected_triggered: 1
                expected_payload: payload.event == "refund"
        """
    )

    test_file = ConnicTestFile.model_validate(doc)

    resolved = test_file.resolved(test_file.tests[0])
    assert resolved["expected_child_agents"] == {
        "refund-specialist": {
            "expected_triggered": 2,
            "expected_payload": "payload.charge_id == context.charge_id",
            "expected_result": 'output.status == "refunded"',
            "expected_tool_calls": [
                "billing.refund",
                {"billing.refund": "params.charge_id == context.charge_id"},
            ],
            "expected_tool_call_order": ["billing.refund", "ledger.record_refund"],
            "expected_no_tool_calls": ["email.send"],
            "expected_child_agents": {
                "ledger-writer": {
                    "expected_triggered": 1,
                    "expected_payload": "payload.refund_id == output.refund_id",
                    "expected_result": None,
                    "expected_tool_calls": ["ledger.record_refund"],
                    "expected_tool_call_order": ["ledger.record_refund"],
                    "expected_no_tool_calls": [],
                    "expected_child_agents": None,
                }
            },
        },
        "telemetry": {
            "expected_triggered": 1,
            "expected_payload": 'payload.event == "refund"',
            "expected_result": None,
            "expected_tool_calls": [],
            "expected_tool_call_order": [],
            "expected_no_tool_calls": [],
            "expected_child_agents": None,
        },
    }


def test_test_file_uses_schema_defaults_for_minimal_suite():
    test_file = ConnicTestFile.model_validate(
        {
            "tests": [
                {
                    "name": "plain_message_no_tools",
                    "payload": "say hello",
                }
            ]
        }
    )

    assert test_file.version == "1.0"
    assert test_file.agent is None
    assert test_file.resolved(test_file.tests[0]) == {
        "name": "plain_message_no_tools",
        "payload": "say hello",
        "files": [],
        "builder": None,
        "builder_args": None,
        "mocks": None,
        "strict_mocks": False,
        "strict_hook_mocks": False,
        "strict_middleware_mocks": False,
        "strict_guardrail_mocks": False,
        "approval_decisions": [],
        "strict_approval_decisions": False,
        "runs": 1,
        "success_threshold": 100,
        "timeout_s": 120,
        "expected_result": None,
        "expected_tool_calls": [],
        "expected_tool_call_order": [],
        "expected_no_tool_calls": [],
        "expected_child_agents": None,
    }


def test_test_file_rejects_duplicate_case_names():
    with pytest.raises(ValidationError, match="Duplicate test name 'same_name'"):
        ConnicTestFile.model_validate(
            {
                "tests": [
                    {"name": "same_name", "payload": "first"},
                    {"name": "same_name", "payload": "second"},
                ]
            }
        )


def test_test_case_parses_and_resolves_approval_decisions():
    test_file = ConnicTestFile.model_validate(
        yaml.safe_load(
            """
            tests:
              - name: exercises_hitl
                payload: refund and notify
                approval_decisions:
                  - tool: billing.refund
                    params: params.charge_id == context.charge_id
                    decision: approve
                  - tool: notifications.send
                    decision: reject
                    reason: Keep the test isolated
                  - tool: billing.capture
                    decision: timeout
            """
        )
    )

    assert test_file.resolved(test_file.tests[0])["approval_decisions"] == [
        {
            "tool": "billing.refund",
            "params": "params.charge_id == context.charge_id",
            "decision": "approve",
            "reason": None,
        },
        {
            "tool": "notifications.send",
            "params": None,
            "decision": "reject",
            "reason": "Keep the test isolated",
        },
        {
            "tool": "billing.capture",
            "params": None,
            "decision": "timeout",
            "reason": None,
        },
    ]


def test_strict_approval_decisions_defaults_to_false():
    test_file = ConnicTestFile.model_validate(
        {"tests": [{"name": "t", "payload": "p"}]}
    )

    assert (
        test_file.resolved(test_file.tests[0])["strict_approval_decisions"]
        is False
    )


def test_strict_approval_decisions_inherits_from_defaults():
    test_file = ConnicTestFile.model_validate(
        {
            "defaults": {"strict_approval_decisions": True},
            "tests": [
                {"name": "inherits", "payload": "p"},
                {
                    "name": "opts_out",
                    "payload": "p",
                    "strict_approval_decisions": False,
                },
            ],
        }
    )

    by_name = {case.name: test_file.resolved(case) for case in test_file.tests}
    assert by_name["inherits"]["strict_approval_decisions"] is True
    assert by_name["opts_out"]["strict_approval_decisions"] is False


@pytest.mark.parametrize(
    "decision",
    [
        {"tool": "", "decision": "approve"},
        {"tool": "   ", "decision": "approve"},
        {"tool": "billing.refund", "decision": "approved"},
        {"tool": "billing.refund", "decision": "approve", "params": ""},
        {"tool": "billing.refund", "decision": "approve", "params": "   "},
        {"tool": "billing.refund", "decision": "approve", "unexpected": True},
    ],
)
def test_approval_decisions_reject_invalid_entries(decision):
    with pytest.raises(ValidationError):
        ConnicTestFile.model_validate(
            {
                "tests": [
                    {
                        "name": "invalid_hitl",
                        "payload": "refund",
                        "approval_decisions": [decision],
                    }
                ]
            }
        )


def test_resolved_isolates_approval_decisions_from_caller():
    test_file = ConnicTestFile.model_validate(
        {
            "tests": [
                {
                    "name": "hitl",
                    "payload": "refund",
                    "approval_decisions": [
                        {"tool": "billing.refund", "decision": "approve"}
                    ],
                }
            ]
        }
    )

    resolved = test_file.resolved(test_file.tests[0])
    resolved["approval_decisions"][0]["tool"] = "billing.capture"

    assert test_file.tests[0].approval_decisions[0].tool == "billing.refund"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runs", 0),
        ("runs", 101),
        ("success_threshold", 0),
        ("success_threshold", 101),
        ("timeout_s", 0),
        ("timeout_s", 3601),
    ],
)
def test_defaults_enforce_runner_limits(field, value):
    with pytest.raises(ValidationError):
        ConnicTestDefaults.model_validate({field: value})


@pytest.mark.parametrize(
    "expected_tool_calls",
    [
        [{"math.add": "invocations >= 1", "math.subtract": "invocations == 0"}],
        [{"": "invocations >= 1"}],
        [{"math.add": "   "}],
        [""],
    ],
)
def test_expected_tool_calls_reject_invalid_entries(expected_tool_calls):
    with pytest.raises(ValidationError):
        ConnicTestFile.model_validate(
            {
                "tests": [
                    {
                        "name": "uses_calculator",
                        "payload": '{"a": 4, "b": 6}',
                        "expected_tool_calls": expected_tool_calls,
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "expected_tool_calls",
    [
        [{"billing.refund": "invocations >= 1", "ledger.record": "invocations >= 1"}],
        [{"": "invocations >= 1"}],
        [{"billing.refund": "   "}],
        [""],
    ],
)
def test_child_agent_expected_tool_calls_reject_invalid_entries(expected_tool_calls):
    with pytest.raises(ValidationError):
        ConnicTestFile.model_validate(
            {
                "tests": [
                    {
                        "name": "routes_refund",
                        "payload": '{"charge_id": "ch_123"}',
                        "expected_child_agents": {
                            "refund-specialist": {
                                "expected_tool_calls": expected_tool_calls,
                            }
                        },
                    }
                ]
            }
        )


@pytest.mark.parametrize("expected_tool_call_order", [[""], [123]])
def test_expected_tool_call_order_rejects_invalid_entries(expected_tool_call_order):
    with pytest.raises(ValidationError):
        ConnicTestFile.model_validate(
            {
                "tests": [
                    {
                        "name": "uses_calculator",
                        "payload": '{"a": 4, "b": 6}',
                        "expected_tool_call_order": expected_tool_call_order,
                    }
                ]
            }
        )


@pytest.mark.parametrize("expected_tool_call_order", [[""], [123]])
def test_child_agent_expected_tool_call_order_rejects_invalid_entries(expected_tool_call_order):
    with pytest.raises(ValidationError):
        ConnicTestFile.model_validate(
            {
                "tests": [
                    {
                        "name": "routes_refund",
                        "payload": '{"charge_id": "ch_123"}',
                        "expected_child_agents": {
                            "refund-specialist": {
                                "expected_tool_call_order": expected_tool_call_order,
                            }
                        },
                    }
                ]
            }
        )


def test_tests_list_must_not_be_empty():
    with pytest.raises(ValidationError):
        ConnicTestFile.model_validate({"tests": []})


def test_test_case_accepts_files_list_alongside_payload():
    test_file = ConnicTestFile.model_validate(
        {
            "tests": [
                {
                    "name": "extracts_invoice_total",
                    "payload": "extract the total",
                    "files": ["invoice_a.pdf", "invoice_b.png"],
                }
            ]
        }
    )

    resolved = test_file.resolved(test_file.tests[0])
    assert resolved["files"] == ["invoice_a.pdf", "invoice_b.png"]
    assert resolved["builder"] is None
    assert resolved["builder_args"] is None
    assert resolved["payload"] == "extract the total"


def test_test_case_accepts_builder_without_payload():
    test_file = ConnicTestFile.model_validate(
        {
            "tests": [
                {
                    "name": "uses_builder",
                    "builder": "create_scenario",
                    "builder_args": {"amount_cents": 4200, "currency": "eur"},
                }
            ]
        }
    )

    resolved = test_file.resolved(test_file.tests[0])
    assert resolved["payload"] is None
    assert resolved["builder"] == "create_scenario"
    assert resolved["builder_args"] == {"amount_cents": 4200, "currency": "eur"}


def test_builder_strips_trailing_py_suffix():
    test_file = ConnicTestFile.model_validate(
        {"tests": [{"name": "t", "builder": "make_payload.py"}]}
    )
    assert test_file.tests[0].builder == "make_payload"


def test_explicit_null_builder_is_allowed_with_payload():
    test_file = ConnicTestFile.model_validate(
        {"tests": [{"name": "t", "payload": "classify this", "builder": None}]}
    )
    assert test_file.tests[0].builder is None


def test_test_case_requires_payload_or_builder():
    with pytest.raises(ValidationError, match="must specify either `payload` or `builder`"):
        ConnicTestFile.model_validate({"tests": [{"name": "empty"}]})


@pytest.mark.parametrize(
    "bad_filename",
    [
        "../escape.pdf",
        "subdir/file.pdf",
        "back\\slash.pdf",
        "..hidden.pdf",
        "with space.pdf",
        "weird?char.pdf",
    ],
)
def test_files_entries_reject_unsafe_names(bad_filename):
    with pytest.raises(ValidationError):
        ConnicTestFile.model_validate(
            {
                "tests": [
                    {"name": "t", "payload": "p", "files": [bad_filename]},
                ]
            }
        )


def test_files_entries_reject_duplicates():
    with pytest.raises(ValidationError, match="duplicate file"):
        ConnicTestFile.model_validate(
            {
                "tests": [
                    {
                        "name": "t",
                        "payload": "p",
                        "files": ["a.pdf", "a.pdf"],
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "bad_builder",
    [
        "../escape",
        "subdir/builder",
        "back\\slash",
        "with space",
    ],
)
def test_builder_rejects_unsafe_names(bad_builder):
    with pytest.raises(ValidationError):
        ConnicTestFile.model_validate(
            {"tests": [{"name": "t", "builder": bad_builder}]}
        )


def test_mocks_parses_and_strips_trailing_py_suffix():
    test_file = ConnicTestFile.model_validate(
        {"tests": [{"name": "t", "payload": "p", "mocks": "customer_mocks.py"}]}
    )
    assert test_file.tests[0].mocks == "customer_mocks"
    assert test_file.resolved(test_file.tests[0])["mocks"] == "customer_mocks"


def test_mocks_inherits_from_defaults_and_allows_case_override():
    test_file = ConnicTestFile.model_validate(
        {
            "defaults": {"mocks": "safe_billing_tools.py"},
            "tests": [
                {"name": "refunds_without_charging", "payload": '{"charge_id": "ch_123"}'},
                {
                    "name": "emails_receipt",
                    "payload": '{"customer_id": "cus_123"}',
                    "mocks": "customer_notifications",
                },
            ],
        }
    )

    by_name = {case.name: test_file.resolved(case) for case in test_file.tests}
    assert by_name["refunds_without_charging"]["mocks"] == "safe_billing_tools"
    assert by_name["emails_receipt"]["mocks"] == "customer_notifications"


def test_explicit_null_mocks_is_allowed_with_payload():
    test_file = ConnicTestFile.model_validate(
        {"tests": [{"name": "t", "payload": "p", "mocks": None}]}
    )
    assert test_file.tests[0].mocks is None
    assert test_file.resolved(test_file.tests[0])["mocks"] is None


@pytest.mark.parametrize(
    "bad_mocks",
    [
        "../escape",
        "subdir/mocks",
        "back\\slash",
        "with space",
    ],
)
def test_mocks_rejects_unsafe_names(bad_mocks):
    with pytest.raises(ValidationError):
        ConnicTestFile.model_validate(
            {"tests": [{"name": "t", "payload": "p", "mocks": bad_mocks}]}
        )


def test_strict_mocks_defaults_to_false():
    test_file = ConnicTestFile.model_validate(
        {"tests": [{"name": "t", "payload": "p"}]}
    )
    assert test_file.resolved(test_file.tests[0])["strict_mocks"] is False


def test_strict_mocks_inherits_from_defaults():
    test_file = ConnicTestFile.model_validate(
        {
            "defaults": {"strict_mocks": True},
            "tests": [
                {"name": "inherits", "payload": "p"},
                {"name": "opts_out", "payload": "p", "strict_mocks": False},
            ],
        }
    )
    by_name = {c.name: test_file.resolved(c) for c in test_file.tests}
    assert by_name["inherits"]["strict_mocks"] is True
    assert by_name["opts_out"]["strict_mocks"] is False


@pytest.mark.parametrize(
    "strict_field",
    [
        "strict_hook_mocks",
        "strict_middleware_mocks",
        "strict_guardrail_mocks",
    ],
)
def test_strict_lifecycle_mock_flags_default_false_and_resolve_independently(strict_field):
    strict_fields = {
        "strict_hook_mocks",
        "strict_middleware_mocks",
        "strict_guardrail_mocks",
    }
    test_file = ConnicTestFile.model_validate(
        {
            "defaults": {strict_field: True},
            "tests": [
                {"name": "inherits", "payload": "p"},
                {"name": "opts_out", "payload": "p", strict_field: False},
            ],
        }
    )

    inherited = test_file.resolved(test_file.tests[0])
    opted_out = test_file.resolved(test_file.tests[1])
    for field in strict_fields:
        assert inherited[field] is (field == strict_field)
        assert opted_out[field] is False

    opted_in = ConnicTestFile.model_validate(
        {"tests": [{"name": "opts_in", "payload": "p", strict_field: True}]}
    )
    assert opted_in.resolved(opted_in.tests[0])[strict_field] is True


def test_resolved_isolates_builder_args_from_caller():
    test_file = ConnicTestFile.model_validate(
        {
            "tests": [
                {
                    "name": "t",
                    "builder": "b",
                    "builder_args": {"k": "v"},
                }
            ]
        }
    )

    resolved = test_file.resolved(test_file.tests[0])
    resolved["builder_args"]["k"] = "mutated"
    # Original case is untouched -- resolved() returns a copy.
    assert test_file.tests[0].builder_args == {"k": "v"}
