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

      # Attach files (located under tests/files/). The runner base64-encodes
      # them and delivers a multimodal payload to the agent.
      - name: extract_invoice_total
        payload: "extract the total amount"
        files:
          - invoice_a.pdf
          - invoice_b.pdf
        expected_result: output.total > 0

      # Build a dynamic payload at run time. tests/builders/<name>.py
      # exposes:
      #   build(context, builder_args, test_name, payload, files)
      #     -> str | dict                                     (required)
      #   cleanup(run, context, builder_args) -> bool | None  (optional)
      # build() produces the agent input and may stash state in
      # ``context`` for cleanup() to read back. The same ``context``
      # dict is also bound as ``context`` in ``expected_result`` and
      # ``expected_tool_calls`` expressions, so a builder can stash a
      # freshly minted uuid (or any other fixture id) and the
      # assertions can compare agent output / tool params against it.
      # cleanup() additionally receives ``run`` -- a dict with
      # ``input``, ``output``, and ``context`` (the real run_context
      # the runner produced, the same dict middleware/hooks see) --
      # and runs after the agent finishes (pass or fail) so fixtures
      # get torn down. May return False to mark the case as failed in
      # addition to the yaml-defined checks.
      - name: refunds_a_real_charge
        builder: create_charge_then_refund
        builder_args:
          amount_cents: 4200
        expected_result: output.status == "refunded"
        expected_tool_calls:
          - billing.refund: params.charge_id == context.charge_id

The ``expected_result`` and ``expected_tool_calls`` mapping expressions are
evaluated server-side with bindings ``output``, ``error``, ``status``,
``context`` (in ``expected_result``) and ``params``, ``invocations``,
``context`` (in ``expected_tool_calls``). ``context`` is the builder's
mutable dict; for tests with no builder it is empty.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# Tool-call expectation: either a bare tool name (= called at least once) or
# a one-key mapping ``{tool_name: <expression>}`` (mirrors the YAML pattern
# used by ``approval.tools``). YAML parses the mapping form into a dict.
ToolCallExpectation = Union[str, Dict[str, str]]


# Filenames referenced by ``files:`` and ``builder:`` are resolved relative
# to ``tests/files/`` and ``tests/builders/`` respectively. Reject anything
# that looks like a path so a malicious yaml can't escape those directories.
_SAFE_REF_RE = re.compile(r"^[A-Za-z0-9._-]+(?:\.[A-Za-z0-9]+)?$")


def _validate_safe_ref(value: str, field_label: str) -> str:
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError(
            f"{field_label} '{value}' must be a bare filename (no path separators)"
        )
    if not _SAFE_REF_RE.fullmatch(value):
        raise ValueError(
            f"{field_label} '{value}' contains disallowed characters; "
            "only letters, digits, dot, dash, underscore are allowed"
        )
    return value


class TestDefaults(BaseModel):
    """Defaults applied to every test case in the file."""

    runs: int = Field(default=1, ge=1, le=100, description="Number of agent runs per test case.")
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
    payload: Optional[str] = Field(
        default=None,
        description=(
            "Static agent input. If parseable as JSON, the runner converts it "
            "before binding `output`/`input`. May be omitted when `builder` is "
            "set; may be combined with `files` to attach binary content."
        ),
    )
    files: List[str] = Field(
        default_factory=list,
        description=(
            "Filenames (no path separators) located in ``tests/files/``. "
            "The runner reads each file, base64-encodes it, and sends the "
            "agent a multimodal payload of the form "
            "``{message: <payload-or-builder-output>, files: [{name, mime_type, data}]}``."
        ),
    )
    builder: Optional[str] = Field(
        default=None,
        description=(
            "Name of a Python module under ``tests/builders/`` (with or "
            "without the ``.py`` suffix). Must expose "
            "``build(context, builder_args, test_name, payload, files)``; "
            "may expose ``cleanup(run, context, builder_args)``. "
            "build()'s return value replaces any static `payload`."
        ),
    )
    builder_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary kwargs forwarded to the builder via ``test_details['builder_args']``.",
    )
    runs: Optional[int] = Field(default=None, ge=1, le=100)
    success_threshold: Optional[int] = Field(default=None, ge=1, le=100)
    timeout_s: Optional[int] = Field(default=None, ge=1, le=3600)

    expected_result: Optional[str] = Field(
        default=None,
        description=(
            "Expression evaluated against bindings: output, error, status, "
            "context. ``context`` is the builder's mutable dict (empty when "
            "no builder is set), so assertions can reference fixture state "
            "stashed by ``build()`` (e.g. ``output.id == context.row_uuid``). "
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

    @field_validator("files")
    @classmethod
    def _validate_files(cls, v: List[str]) -> List[str]:
        seen: set[str] = set()
        for entry in v:
            _validate_safe_ref(entry, "files entry")
            if entry in seen:
                raise ValueError(f"duplicate file '{entry}' in files list")
            seen.add(entry)
        return v

    @field_validator("builder")
    @classmethod
    def _validate_builder(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        _validate_safe_ref(v, "builder")
        # Strip a trailing .py so the runner can look up the module name.
        return v[:-3] if v.endswith(".py") else v

    @model_validator(mode="after")
    def _payload_or_builder(self) -> "TestCase":
        if self.payload is None and self.builder is None:
            raise ValueError(
                f"test '{self.name}': must specify either `payload` or `builder`."
            )
        return self


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
            "files": list(case.files),
            "builder": case.builder,
            "builder_args": dict(case.builder_args) if case.builder_args else None,
            "runs": case.runs if case.runs is not None else self.defaults.runs,
            "success_threshold": (
                case.success_threshold if case.success_threshold is not None else self.defaults.success_threshold
            ),
            "timeout_s": case.timeout_s if case.timeout_s is not None else self.defaults.timeout_s,
            "expected_result": case.expected_result,
            "expected_tool_calls": case.expected_tool_calls,
            "expected_no_tool_calls": case.expected_no_tool_calls,
        }
