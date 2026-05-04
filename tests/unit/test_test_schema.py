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
        ("runs", 1001),
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
