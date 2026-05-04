"""Pydantic models for the customer-facing test framework.

A test file lives at ``tests/<agent-name>.yaml`` (flat, mirroring how
``middleware/`` works). It declares one or more test cases that run the named
agent N times with a payload, then assert on output and tool-call traces.

Example::

    version: "1.0"
    agent: stress-tester               # optional, defaults to filename stem
    defaults:
      runs: 1
      success_threshold: 100
      timeout_s: 60

    tests:
      - name: returns_id_10
        payload: '{"a": 4, "b": 6}'
        runs: 10
        success_threshold: 90
        expected_result: output.id == 10
        expected_tool_calls:
          - math.calculator.add                      # bare → called >= 1
          - math.calculator.add: invocations >= 5    # mapping → expression
        expected_no_tool_calls:
          - email.send

The ``expected_result`` and ``expected_tool_calls`` mapping expressions are
evaluated by ``shared.expression_filter.safe_eval`` server-side, with
bindings ``output``, ``error``, ``status``, ``invocations``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# Tool-call expectation: either a bare tool name (= called at least once) or
# a one-key mapping ``{tool_name: <expression>}`` (mirrors the YAML pattern
# used by ``approval.tools``). YAML parses the mapping form into a dict.
ToolCallExpectation = Union[str, Dict[str, str]]


class TestDefaults(BaseModel):
    """Defaults applied to every test case in the file."""

    runs: int = Field(default=1, ge=1, le=1000, description="Number of agent runs per test case.")
    success_threshold: int = Field(
        default=100,
        ge=1,
        le=100,
        description="Percent of runs that must pass for the test to pass.",
    )
    timeout_s: int = Field(
        default=120,
        ge=1,
        le=3600,
        description="Per-run wall-clock timeout in seconds.",
    )


class TestCase(BaseModel):
    """A single test case."""

    name: str = Field(..., min_length=1, max_length=120, description="Stable identifier within the file.")
    payload: str = Field(
        ...,
        description=(
            "Agent input. Always a string, mirroring normal Connic payloads. "
            "If parseable as JSON, the runner converts it before binding "
            "`output`/`input`."
        ),
    )
    runs: Optional[int] = Field(default=None, ge=1, le=1000)
    success_threshold: Optional[int] = Field(default=None, ge=1, le=100)
    timeout_s: Optional[int] = Field(default=None, ge=1, le=3600)

    expected_result: Optional[str] = Field(
        default=None,
        description=(
            "Expression evaluated against bindings: output, error, status. "
            "If omitted, only the run's terminal status (`completed`) is "
            "checked."
        ),
    )
    expected_tool_calls: List[ToolCallExpectation] = Field(default_factory=list)
    expected_no_tool_calls: List[str] = Field(
        default_factory=list,
        description="Tools that must NOT be called during the run.",
    )

    @field_validator("expected_tool_calls")
    @classmethod
    def _validate_tool_call_mappings(cls, v: List[ToolCallExpectation]) -> List[ToolCallExpectation]:
        for entry in v:
            if isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(
                        "expected_tool_calls mapping entries must have exactly one key, "
                        f"got {len(entry)}: {list(entry.keys())}"
                    )
                ((tool_name, expr),) = entry.items()
                if not isinstance(tool_name, str) or not tool_name:
                    raise ValueError("Tool name must be a non-empty string.")
                if not isinstance(expr, str) or not expr.strip():
                    raise ValueError(f"Expression for '{tool_name}' must be a non-empty string.")
            elif not isinstance(entry, str) or not entry:
                raise ValueError("expected_tool_calls entries must be a string or a single-key mapping.")
        return v


class TestFile(BaseModel):
    """Parsed contents of one ``tests/*.yaml`` file."""

    version: str = Field(default="1.0")
    agent: Optional[str] = Field(
        default=None,
        description="Agent name. Defaults to the filename stem (e.g. tests/foo.yaml → foo).",
    )
    defaults: TestDefaults = Field(default_factory=TestDefaults)
    tests: List[TestCase] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _names_are_unique(self) -> "TestFile":
        seen: set[str] = set()
        for case in self.tests:
            if case.name in seen:
                raise ValueError(f"Duplicate test name '{case.name}' in file.")
            seen.add(case.name)
        return self

    def resolved(self, case: TestCase) -> Dict[str, Any]:
        """Merge defaults into a single test case for execution."""
        return {
            "name": case.name,
            "payload": case.payload,
            "runs": case.runs if case.runs is not None else self.defaults.runs,
            "success_threshold": (
                case.success_threshold if case.success_threshold is not None else self.defaults.success_threshold
            ),
            "timeout_s": case.timeout_s if case.timeout_s is not None else self.defaults.timeout_s,
            "expected_result": case.expected_result,
            "expected_tool_calls": case.expected_tool_calls,
            "expected_no_tool_calls": case.expected_no_tool_calls,
        }
