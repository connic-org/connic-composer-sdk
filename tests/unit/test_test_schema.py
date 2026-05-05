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
        "runs": 5,
        "success_threshold": 80,
        "timeout_s": 60,
        "expected_result": 'status == "completed" and output.id == 10',
        "expected_tool_calls": ["math.calculator.add", {"math.calculator.add": "invocations >= 1"}],
        "expected_no_tool_calls": ["email.send"],
    }

    second = test_file.resolved(test_file.tests[1])
    assert second["runs"] == 20
    assert second["success_threshold"] == 95
    assert second["timeout_s"] == 90
    assert second["expected_tool_calls"] == []
    assert second["expected_no_tool_calls"] == []


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
        "runs": 1,
        "success_threshold": 100,
        "timeout_s": 120,
        "expected_result": None,
        "expected_tool_calls": [],
        "expected_no_tool_calls": [],
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
