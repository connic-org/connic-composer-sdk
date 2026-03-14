from __future__ import annotations

import ast
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import click
import yaml

from .loader import PREDEFINED_TOOL_NAMES

MIGRATION_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
}
MIGRATION_DEFAULT_MODEL = "gemini/gemini-2.5-flash"
MIGRATION_DEFAULT_PROMPT = (
    "You are a migrated agent. Review MIGRATION_REPORT.md and update this prompt before production use."
)
MIGRATION_MODEL_PROVIDER_MAP = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "gpt": "openai",
    "azure_openai": "azure",
    "azure": "azure",
    "google_genai": "gemini",
    "google": "gemini",
    "gemini": "gemini",
    "openrouter": "openrouter",
}
ADK_YAML_AGENT_KEYS = {
    "instruction",
    "system_instruction",
    "system_prompt",
    "model",
    "tools",
    "sub_agents",
    "agents",
}


@dataclass
class ImportBinding:
    kind: str
    module: str | None
    name: str | None = None
    level: int = 0


@dataclass
class ModuleInfo:
    path: Path
    source: str
    tree: ast.Module
    imports: dict[str, ImportBinding] = field(default_factory=dict)
    assignments: dict[str, ast.AST] = field(default_factory=dict)
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = field(default_factory=dict)
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)


@dataclass
class ToolCandidate:
    function_name: str
    source_file: Path | None = None
    ref: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class AgentCandidate:
    source_id: str
    framework: str
    source_file: Path | None
    name: str
    agent_type: str
    description: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    tool_candidates: list[ToolCandidate] = field(default_factory=list)
    agent_ref_keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


WriteEssentialFiles = Callable[[Path, bool], None]
RunLint = Callable[..., bool]


def _is_hidden_or_skipped(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts
    return any(part.startswith(".") or part in MIGRATION_SKIP_DIRS for part in relative_parts)


def _collect_python_files(root: Path) -> list[Path]:
    files = []
    for file_path in root.rglob("*.py"):
        if _is_hidden_or_skipped(file_path, root):
            continue
        files.append(file_path)
    return sorted(files)


def _collect_yaml_files(root: Path) -> list[Path]:
    files = []
    for pattern in ("*.yaml", "*.yml"):
        for file_path in root.rglob(pattern):
            if _is_hidden_or_skipped(file_path, root):
                continue
            files.append(file_path)
    return sorted(set(files))


def _build_module_name_variants(root: Path, file_path: Path) -> list[str]:
    relative = file_path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    variants: list[list[str]] = [parts]
    if parts and parts[0] == "src":
        variants.append(parts[1:])

    names = []
    for variant in variants:
        if not variant:
            continue
        current = list(variant)
        if current[-1] == "__init__":
            current = current[:-1]
        if current:
            dotted = ".".join(current)
            if dotted not in names:
                names.append(dotted)
    return names


def _build_module_lookup(root: Path, python_files: list[Path]) -> dict[str, Path]:
    module_lookup: dict[str, Path] = {}
    for file_path in python_files:
        for module_name in _build_module_name_variants(root, file_path):
            module_lookup.setdefault(module_name, file_path)
    return module_lookup


def _parse_module_info(file_path: Path) -> ModuleInfo | None:
    try:
        source = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    info = ModuleInfo(path=file_path, source=source, tree=tree)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias_name = alias.asname or alias.name.split(".")[-1]
                info.imports[alias_name] = ImportBinding(kind="import", module=alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                alias_name = alias.asname or alias.name
                info.imports[alias_name] = ImportBinding(
                    kind="from",
                    module=node.module,
                    name=alias.name,
                    level=node.level,
                )
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            info.assignments[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            info.assignments[node.target.id] = node.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            info.classes[node.name] = node
    return info


def _get_cached_module_info(file_path: Path, module_cache: dict[Path, ModuleInfo | None]) -> ModuleInfo | None:
    if file_path not in module_cache:
        module_cache[file_path] = _parse_module_info(file_path)
    return module_cache[file_path]


def _get_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _get_full_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _get_full_attr_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _resolve_function_return_string(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_info: ModuleInfo,
    module_lookup: dict[str, Path],
    module_cache: dict[Path, ModuleInfo | None],
    seen: set[str],
) -> str | None:
    for child in ast.walk(function_node):
        if isinstance(child, ast.Return) and child.value is not None:
            resolved = _resolve_string_expr(child.value, module_info, module_lookup, module_cache, seen.copy())
            if resolved:
                return resolved
    return None


def _resolve_literal_expr(
    expr: ast.AST | None,
    module_info: ModuleInfo,
    module_lookup: dict[str, Path],
    module_cache: dict[Path, ModuleInfo | None],
    seen: set[str],
) -> str | int | float | None:
    if expr is None:
        return None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (str, int, float)):
        return expr.value
    if isinstance(expr, ast.Name):
        if expr.id in seen:
            return None
        seen.add(expr.id)
        assigned = module_info.assignments.get(expr.id)
        if assigned is not None:
            return _resolve_literal_expr(assigned, module_info, module_lookup, module_cache, seen)
        binding = module_info.imports.get(expr.id)
        if binding:
            source_file, symbol_name = _resolve_imported_symbol_source(binding, module_info.path, module_lookup)
            other_module = _get_cached_module_info(source_file, module_cache) if source_file else None
            if other_module and symbol_name:
                assigned = other_module.assignments.get(symbol_name)
                if assigned is not None:
                    return _resolve_literal_expr(assigned, other_module, module_lookup, module_cache, seen)
    return None


def _resolve_string_from_binding(
    binding: ImportBinding,
    current_file: Path,
    module_lookup: dict[str, Path],
    module_cache: dict[Path, ModuleInfo | None],
    seen: set[str],
) -> str | None:
    source_file, symbol_name = _resolve_imported_symbol_source(binding, current_file, module_lookup)
    if not source_file or not symbol_name:
        return None
    other_module = _get_cached_module_info(source_file, module_cache)
    if other_module is None:
        return None
    if symbol_name in seen:
        return None
    seen.add(symbol_name)
    assigned = other_module.assignments.get(symbol_name)
    if assigned is not None:
        return _resolve_string_expr(assigned, other_module, module_lookup, module_cache, seen)
    function_node = other_module.functions.get(symbol_name)
    if function_node is not None:
        return _resolve_function_return_string(function_node, other_module, module_lookup, module_cache, seen)
    return None


def _resolve_string_expr(
    expr: ast.AST | None,
    module_info: ModuleInfo,
    module_lookup: dict[str, Path],
    module_cache: dict[Path, ModuleInfo | None],
    seen: set[str] | None = None,
) -> str | None:
    if expr is None:
        return None
    if seen is None:
        seen = set()
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value.strip()
    if isinstance(expr, ast.JoinedStr):
        parts = []
        for value in expr.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts).strip()
    if isinstance(expr, ast.Name):
        if expr.id in seen:
            return None
        seen.add(expr.id)
        assigned = module_info.assignments.get(expr.id)
        if assigned is not None:
            return _resolve_string_expr(assigned, module_info, module_lookup, module_cache, seen)
        binding = module_info.imports.get(expr.id)
        if binding:
            return _resolve_string_from_binding(binding, module_info.path, module_lookup, module_cache, seen)
        return None
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        binding = module_info.imports.get(expr.value.id)
        if binding:
            source_file = _resolve_module_alias_source(binding, module_info.path, module_lookup)
            other_module = _get_cached_module_info(source_file, module_cache) if source_file else None
            if other_module is None or expr.attr in seen:
                return None
            seen.add(expr.attr)
            assigned = other_module.assignments.get(expr.attr)
            if assigned is not None:
                return _resolve_string_expr(assigned, other_module, module_lookup, module_cache, seen)
            function_node = other_module.functions.get(expr.attr)
            if function_node is not None:
                return _resolve_function_return_string(function_node, other_module, module_lookup, module_cache, seen)
    if isinstance(expr, ast.Call):
        call_name = _get_call_name(expr.func)
        if call_name == "init_chat_model" and expr.args:
            return _resolve_string_expr(expr.args[0], module_info, module_lookup, module_cache, seen)
        if call_name == "getenv" and len(expr.args) >= 2:
            return _resolve_string_expr(expr.args[1], module_info, module_lookup, module_cache, seen)
        if isinstance(expr.func, ast.Attribute) and expr.func.attr == "format":
            base = _resolve_string_expr(expr.func.value, module_info, module_lookup, module_cache, seen.copy())
            if base is not None:
                format_args = [
                    _resolve_literal_expr(arg, module_info, module_lookup, module_cache, seen.copy())
                    for arg in expr.args
                ]
                format_kwargs = {
                    keyword.arg: _resolve_literal_expr(keyword.value, module_info, module_lookup, module_cache, seen.copy())
                    for keyword in expr.keywords
                    if keyword.arg
                }
                if all(arg is not None for arg in format_args) and all(value is not None for value in format_kwargs.values()):
                    try:
                        return base.format(*format_args, **format_kwargs)
                    except Exception:
                        return base
                return base
        if not expr.args and not expr.keywords:
            if isinstance(expr.func, ast.Name):
                if expr.func.id in module_info.functions:
                    return _resolve_function_return_string(module_info.functions[expr.func.id], module_info, module_lookup, module_cache, seen)
                binding = module_info.imports.get(expr.func.id)
                if binding:
                    return _resolve_string_from_binding(binding, module_info.path, module_lookup, module_cache, seen)
            if isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name):
                binding = module_info.imports.get(expr.func.value.id)
                if binding:
                    source_file = _resolve_module_alias_source(binding, module_info.path, module_lookup)
                    other_module = _get_cached_module_info(source_file, module_cache) if source_file else None
                    if other_module and expr.func.attr in other_module.functions:
                        return _resolve_function_return_string(other_module.functions[expr.func.attr], other_module, module_lookup, module_cache, seen)
        for keyword in expr.keywords:
            if keyword.arg == "model":
                return _resolve_string_expr(keyword.value, module_info, module_lookup, module_cache, seen)
    return None


def _normalize_model_name(raw_model: str | None) -> str | None:
    if not raw_model:
        return None
    model = raw_model.strip()
    if not model:
        return None
    if "/" in model and model.split("/", 1)[0] in {"openai", "azure", "anthropic", "gemini", "openrouter"}:
        return model
    if ":" in model:
        provider, model_name = model.split(":", 1)
        mapped_provider = MIGRATION_MODEL_PROVIDER_MAP.get(provider, provider)
        return f"{mapped_provider}/{model_name}"
    lowered = model.lower()
    for key, provider in MIGRATION_MODEL_PROVIDER_MAP.items():
        if lowered.startswith(key):
            if provider == key:
                return model
            return f"{provider}/{model}"
    if lowered.startswith("gpt-"):
        return f"openai/{model}"
    if lowered.startswith("claude"):
        return f"anthropic/{model}"
    if lowered.startswith("gemini"):
        return f"gemini/{model}"
    return model


def _sanitize_agent_name(value: str | None, fallback: str) -> str:
    raw = value or fallback
    slug = re.sub(r"[^a-z0-9-]+", "-", raw.lower().replace("_", "-"))
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        slug = fallback
    if not re.match(r"^[a-z0-9]", slug):
        slug = f"agent-{slug}"
    if not re.search(r"[a-z0-9]$", slug):
        slug = f"{slug}0"
    return slug


def _dedupe_agent_names(agents: list[AgentCandidate]) -> dict[str, str]:
    used: set[str] = set()
    source_key_map: dict[str, str] = {}
    for index, agent in enumerate(agents, start=1):
        base_name = _sanitize_agent_name(agent.name, f"migrated-agent-{index}")
        unique_name = base_name
        suffix = 2
        while unique_name in used:
            unique_name = f"{base_name}-{suffix}"
            suffix += 1
        agent.name = unique_name
        used.add(unique_name)
        source_key_map[agent.source_id] = unique_name
        source_key_map.setdefault(base_name, unique_name)
    for agent in agents:
        resolved_refs = []
        for ref in agent.agent_ref_keys:
            resolved_refs.append(source_key_map.get(ref, _sanitize_agent_name(ref, ref)))
        agent.agent_ref_keys = resolved_refs
    return source_key_map


def _resolve_absolute_module_path(module_name: str | None, module_lookup: dict[str, Path]) -> Path | None:
    if not module_name:
        return None
    return module_lookup.get(module_name)


def _resolve_neighbor_module_path(current_file: Path, module_name: str | None) -> Path | None:
    if not module_name:
        return None
    target = current_file.parent.joinpath(*module_name.split("."))
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _resolve_relative_module_path(current_file: Path, level: int, module_name: str | None) -> Path | None:
    base_dir = current_file.parent
    for _ in range(max(level - 1, 0)):
        base_dir = base_dir.parent
    target = base_dir
    if module_name:
        target = target.joinpath(*module_name.split("."))
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _resolve_imported_symbol_source(
    binding: ImportBinding,
    current_file: Path,
    module_lookup: dict[str, Path],
) -> tuple[Path | None, str | None]:
    if binding.kind != "from":
        return None, None
    if binding.level:
        module_path = _resolve_relative_module_path(current_file, binding.level, binding.module)
        return module_path, binding.name
    module_path = _resolve_absolute_module_path(binding.module, module_lookup)
    if module_path is None:
        module_path = _resolve_neighbor_module_path(current_file, binding.module)
    return module_path, binding.name


def _resolve_module_alias_source(
    binding: ImportBinding,
    current_file: Path,
    module_lookup: dict[str, Path],
) -> Path | None:
    if binding.kind == "import":
        return _resolve_absolute_module_path(binding.module, module_lookup) or _resolve_neighbor_module_path(current_file, binding.module)
    if binding.level:
        target = binding.name if not binding.module else f"{binding.module}.{binding.name}"
        return _resolve_relative_module_path(current_file, binding.level, target)
    if binding.module and binding.name:
        return _resolve_absolute_module_path(f"{binding.module}.{binding.name}", module_lookup)
    return None


def _predefined_tool_candidate(name: str) -> ToolCandidate | None:
    alias_map = {"google_search": "web_search"}
    mapped = alias_map.get(name, name)
    if mapped in PREDEFINED_TOOL_NAMES:
        return ToolCandidate(function_name=mapped, ref=mapped)
    return None


def _unique_tool_candidates(candidates: list[ToolCandidate]) -> list[ToolCandidate]:
    unique = []
    seen = set()
    for candidate in candidates:
        key = (candidate.source_file, candidate.function_name, candidate.ref)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _resolve_tool_candidates(
    expr: ast.AST | None,
    module_info: ModuleInfo,
    module_lookup: dict[str, Path],
    notes: list[str],
    seen_names: set[str] | None = None,
) -> list[ToolCandidate]:
    if expr is None:
        return []
    if seen_names is None:
        seen_names = set()

    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        tools = []
        for item in expr.elts:
            tools.extend(_resolve_tool_candidates(item, module_info, module_lookup, notes, seen_names))
        return _unique_tool_candidates(tools)

    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        predefined = _predefined_tool_candidate(expr.value)
        if predefined:
            return [predefined]
        notes.append(f"Could not migrate string-based tool reference '{expr.value}'.")
        return []

    if isinstance(expr, ast.Name):
        if expr.id in seen_names:
            return []
        seen_names.add(expr.id)
        predefined = _predefined_tool_candidate(expr.id)
        if predefined:
            return [predefined]
        if expr.id in module_info.functions:
            return [ToolCandidate(function_name=expr.id, source_file=module_info.path)]
        assigned = module_info.assignments.get(expr.id)
        if assigned is not None:
            return _resolve_tool_candidates(assigned, module_info, module_lookup, notes, seen_names)
        binding = module_info.imports.get(expr.id)
        if binding:
            source_file, function_name = _resolve_imported_symbol_source(binding, module_info.path, module_lookup)
            if source_file and function_name:
                return [ToolCandidate(function_name=function_name, source_file=source_file)]
            predefined = _predefined_tool_candidate(binding.name or expr.id)
            if predefined:
                return [predefined]
        notes.append(f"Could not resolve tool reference '{expr.id}' in {module_info.path.name}.")
        return []

    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        binding = module_info.imports.get(expr.value.id)
        if binding:
            source_file = _resolve_module_alias_source(binding, module_info.path, module_lookup)
            if source_file:
                return [ToolCandidate(function_name=expr.attr, source_file=source_file)]
        predefined = _predefined_tool_candidate(expr.attr)
        if predefined:
            return [predefined]
        notes.append(f"Could not resolve tool attribute '{expr.value.id}.{expr.attr}'.")
        return []

    if isinstance(expr, ast.Call):
        call_name = _get_call_name(expr.func)
        if call_name == "FunctionTool":
            wrapped = _get_keyword(expr, "func") or (expr.args[0] if expr.args else None)
            return _resolve_tool_candidates(wrapped, module_info, module_lookup, notes, seen_names)
        if call_name == "AgentTool":
            notes.append("Skipped ADK AgentTool wrapper because Connic does not have a direct agent-as-tool equivalent.")
            return []
        if call_name == "MCPToolset":
            notes.append("Skipped ADK MCPToolset because it requires manual MCP server mapping in Connic.")
            return []
        notes.append(
            f"Could not safely migrate dynamic tool expression '{ast.unparse(expr)}'. Replace it with a Connic tool manually."
        )
        return []

    return []


def _get_keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _extract_description(prompt: str | None, fallback: str) -> str:
    if not prompt:
        return fallback
    first_line = next((line.strip() for line in prompt.splitlines() if line.strip()), "")
    if not first_line:
        return fallback
    return first_line[:140]


def _extract_langchain_agents(module_info: ModuleInfo, module_lookup: dict[str, Path]) -> list[AgentCandidate]:
    agents: list[AgentCandidate] = []
    module_cache: dict[Path, ModuleInfo | None] = {module_info.path: module_info}
    supported_calls = {"create_agent", "create_react_agent"}
    for node in module_info.tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call_name = _get_call_name(node.value.func)
        if call_name not in supported_calls:
            continue
        target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        target_name = target_names[0] if target_names else module_info.path.stem
        tools_expr = _get_keyword(node.value, "tools")
        if tools_expr is None and len(node.value.args) >= 2:
            tools_expr = node.value.args[1]
        model_expr = _get_keyword(node.value, "model") or (node.value.args[0] if node.value.args else None)
        prompt_expr = _get_keyword(node.value, "system_prompt") or _get_keyword(node.value, "prompt")
        configured_name = _resolve_string_expr(_get_keyword(node.value, "name"), module_info, module_lookup, module_cache)

        notes: list[str] = []
        system_prompt = _resolve_string_expr(prompt_expr, module_info, module_lookup, module_cache)
        model = _normalize_model_name(_resolve_string_expr(model_expr, module_info, module_lookup, module_cache))
        tools = _resolve_tool_candidates(tools_expr, module_info, module_lookup, notes)
        fallback_name = configured_name or target_name
        fallback_description = f"Migrated from LangChain agent '{fallback_name}'."
        agents.append(
            AgentCandidate(
                source_id=target_name,
                framework="langchain",
                source_file=module_info.path,
                name=fallback_name,
                agent_type="llm",
                description=_extract_description(system_prompt, fallback_description),
                model=model,
                system_prompt=system_prompt,
                tool_candidates=tools,
                notes=notes,
            )
        )
    return agents


def _resolve_name_list(expr: ast.AST | None, module_info: ModuleInfo, seen: set[str] | None = None) -> list[str]:
    if expr is None:
        return []
    if seen is None:
        seen = set()
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        values = []
        for item in expr.elts:
            if isinstance(item, ast.Name):
                values.append(item.id)
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                values.append(item.value)
        return values
    if isinstance(expr, ast.Name):
        if expr.id in seen:
            return []
        seen.add(expr.id)
        assigned = module_info.assignments.get(expr.id)
        if assigned is not None:
            return _resolve_name_list(assigned, module_info, seen)
    return []


def _extract_adk_agents(module_info: ModuleInfo, module_lookup: dict[str, Path]) -> list[AgentCandidate]:
    agents: list[AgentCandidate] = []
    module_cache: dict[Path, ModuleInfo | None] = {module_info.path: module_info}
    supported_calls = {"Agent", "LlmAgent", "SequentialAgent", "ParallelAgent", "LoopAgent"}
    for node in module_info.tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call_name = _get_call_name(node.value.func)
        if call_name not in supported_calls:
            continue
        target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        target_name = target_names[0] if target_names else module_info.path.stem
        source_name = _resolve_string_expr(_get_keyword(node.value, "name"), module_info, module_lookup, module_cache) or target_name
        instruction = _resolve_string_expr(
            _get_keyword(node.value, "instruction") or _get_keyword(node.value, "system_instruction"),
            module_info,
            module_lookup,
            module_cache,
        )
        description = _resolve_string_expr(_get_keyword(node.value, "description"), module_info, module_lookup, module_cache)
        model = _normalize_model_name(_resolve_string_expr(_get_keyword(node.value, "model"), module_info, module_lookup, module_cache))
        notes: list[str] = []
        tools = _resolve_tool_candidates(_get_keyword(node.value, "tools"), module_info, module_lookup, notes)
        sub_agents = _resolve_name_list(_get_keyword(node.value, "sub_agents") or _get_keyword(node.value, "agents"), module_info)
        agent_type = "llm"
        if call_name in {"SequentialAgent", "ParallelAgent", "LoopAgent"} and sub_agents:
            agent_type = "sequential"
            if call_name != "SequentialAgent":
                notes.append(f"Original ADK agent used {call_name}; migrated as a sequential Connic agent for review.")
        elif sub_agents and not model:
            agent_type = "sequential"
            notes.append("ADK sub_agents were migrated as a sequential Connic workflow.")
        elif sub_agents:
            notes.append(
                "ADK agent references sub_agents. Connic does not have a direct multi-agent router equivalent, so review this agent manually."
            )

        fallback_description = description or f"Migrated from ADK agent '{source_name}'."
        agents.append(
            AgentCandidate(
                source_id=target_name,
                framework="adk",
                source_file=module_info.path,
                name=source_name,
                agent_type=agent_type,
                description=fallback_description,
                model=model,
                system_prompt=instruction,
                tool_candidates=tools,
                agent_ref_keys=sub_agents,
                notes=notes,
            )
        )
    return agents


def _extract_adk_yaml_agents(yaml_files: list[Path]) -> list[AgentCandidate]:
    agents: list[AgentCandidate] = []
    for yaml_file in yaml_files:
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        lower_name = yaml_file.name.lower()
        has_signature_key = any(key in data for key in ADK_YAML_AGENT_KEYS)
        looks_like_agent_file = "agent" in lower_name or lower_name in {"root_agent.yaml", "root_agent.yml"}
        if not has_signature_key and not looks_like_agent_file:
            continue
        if not has_signature_key:
            continue
        name = data.get("name") or data.get("agent_name") or data.get("id") or yaml_file.stem
        instruction = data.get("instruction") or data.get("system_instruction") or data.get("system_prompt")
        description = data.get("description") or f"Migrated from ADK YAML agent '{name}'."
        model = _normalize_model_name(data.get("model"))
        tool_candidates = []
        notes = []
        raw_tools = data.get("tools")
        if isinstance(raw_tools, list):
            for item in raw_tools:
                if isinstance(item, str):
                    predefined = _predefined_tool_candidate(item)
                    if predefined:
                        tool_candidates.append(predefined)
                    else:
                        notes.append(f"Could not migrate YAML tool reference '{item}'.")
        raw_agent_refs = data.get("sub_agents") or data.get("agents") or []
        agent_refs = [item for item in raw_agent_refs if isinstance(item, str)] if isinstance(raw_agent_refs, list) else []
        agent_type = "sequential" if agent_refs else "llm"
        agents.append(
            AgentCandidate(
                source_id=f"yaml:{yaml_file.stem}",
                framework="adk",
                source_file=yaml_file,
                name=name,
                agent_type=agent_type,
                description=description,
                model=model,
                system_prompt=instruction,
                tool_candidates=_unique_tool_candidates(tool_candidates),
                agent_ref_keys=agent_refs,
                notes=notes,
            )
        )
    return agents


def _detect_custom_adk_patterns(module_infos: dict[Path, ModuleInfo]) -> list[str]:
    custom_agent_classes: set[str] = set()
    dynamic_registration_files: set[str] = set()
    custom_wrapper_calls: set[str] = set()

    for module_info in module_infos.values():
        for class_name, class_node in module_info.classes.items():
            for base in class_node.bases:
                base_name = _get_full_attr_name(base)
                if base_name in {"LlmAgent", "Agent", "google.adk.agents.LlmAgent", "google.adk.agents.Agent"}:
                    custom_agent_classes.add(class_name)

    if not custom_agent_classes:
        return []

    for module_info in module_infos.values():
        for node in ast.walk(module_info.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "register_agent":
                if any(
                    isinstance(child, ast.Call) and _get_call_name(child.func) in custom_agent_classes
                    for child in ast.walk(node)
                ):
                    dynamic_registration_files.add(module_info.path.name)

            if isinstance(node, ast.Call):
                call_name = _get_call_name(node.func)
                if call_name in custom_agent_classes:
                    custom_wrapper_calls.add(call_name)

    notes: list[str] = []
    if dynamic_registration_files:
        wrapper_list = ", ".join(sorted(custom_wrapper_calls)) or "custom ADK wrappers"
        file_list = ", ".join(sorted(dynamic_registration_files))
        notes.append(
            f"Detected dynamically registered custom ADK agent wrappers ({wrapper_list}) in {file_list}. "
            "`connic migrate` currently auto-migrates only explicit top-level ADK agent definitions."
        )
    return notes


def _detect_framework(source_root: Path, python_files: list[Path], yaml_files: list[Path]) -> tuple[str, list[str]]:
    notes: list[str] = []
    has_adk = any(path.name in {"root_agent.yaml", "root_agent.yml"} for path in yaml_files)
    has_langchain = False
    for file_path in python_files:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "google.adk" in source or "adk " in source or "root_agent" in source and "Agent(" in source:
            has_adk = True
        if "langchain" in source or "langgraph" in source or "langsmith" in source or "create_agent(" in source:
            has_langchain = True
    if has_adk and not has_langchain:
        notes.append("Detected Google ADK project structure and imports.")
        return "adk", notes
    if has_langchain and not has_adk:
        notes.append("Detected LangChain/LangGraph/LangSmith imports.")
        return "langchain", notes
    if has_adk and has_langchain:
        notes.append("Detected both ADK and LangChain signals. Defaulting to LangChain migration heuristics.")
        return "langchain", notes
    notes.append("Could not confidently detect the source framework. Using generic LangChain-style heuristics.")
    return "unknown", notes


def _gather_local_dependency_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
    return names


def _extract_module_subset(module_info: ModuleInfo, function_names: set[str]) -> tuple[str, list[str]]:
    selected_functions: set[str] = set()
    selected_assignments: set[str] = set()
    selected_classes: set[str] = set()
    used_import_names: set[str] = set()
    notes: list[str] = []

    def collect_assignment(name: str):
        if name in selected_assignments:
            return
        assignment = module_info.assignments.get(name)
        if assignment is None:
            return
        selected_assignments.add(name)
        for dependency in _gather_local_dependency_names(assignment):
            if dependency in module_info.functions:
                collect_function(dependency)
            elif dependency in module_info.assignments:
                collect_assignment(dependency)
            elif dependency in module_info.classes:
                collect_class(dependency)
            elif dependency in module_info.imports:
                used_import_names.add(dependency)

    def collect_function(name: str):
        if name in selected_functions:
            return
        function_node = module_info.functions.get(name)
        if function_node is None:
            notes.append(f"Missing source for tool function '{name}'.")
            return
        selected_functions.add(name)
        for dependency in _gather_local_dependency_names(function_node):
            if dependency in module_info.functions:
                collect_function(dependency)
            elif dependency in module_info.assignments:
                collect_assignment(dependency)
            elif dependency in module_info.classes:
                collect_class(dependency)
            elif dependency in module_info.imports:
                used_import_names.add(dependency)

    def collect_class(name: str):
        if name in selected_classes:
            return
        class_node = module_info.classes.get(name)
        if class_node is None:
            return
        selected_classes.add(name)
        for dependency in _gather_local_dependency_names(class_node):
            if dependency in module_info.functions:
                collect_function(dependency)
            elif dependency in module_info.assignments:
                collect_assignment(dependency)
            elif dependency in module_info.classes:
                collect_class(dependency)
            elif dependency in module_info.imports:
                used_import_names.add(dependency)

    for function_name in function_names:
        collect_function(function_name)

    import_nodes = []
    for node in module_info.tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        aliases = [alias.asname or alias.name.split(".")[-1] for alias in node.names]
        if used_import_names.intersection(aliases):
            import_nodes.append(node)

    ordered_nodes: list[tuple[int, str]] = []
    for node in import_nodes:
        if isinstance(node, ast.ImportFrom) and node.level:
            notes.append(
                f"Relative import '{ast.get_source_segment(module_info.source, node) or ''}' may need manual review."
            )
        segment = ast.get_source_segment(module_info.source, node)
        if segment:
            ordered_nodes.append((node.lineno, segment))

    for name in selected_assignments:
        assignment = module_info.assignments.get(name)
        if assignment is None:
            continue
        parent = next(
            (
                node
                for node in module_info.tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and ((isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
                     or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name))
            ),
            None,
        )
        if parent is None:
            continue
        segment = ast.get_source_segment(module_info.source, parent)
        if segment:
            ordered_nodes.append((parent.lineno, segment))

    for name in selected_functions:
        function_node = module_info.functions.get(name)
        if function_node is None:
            continue
        segment = ast.get_source_segment(module_info.source, function_node)
        if segment:
            ordered_nodes.append((function_node.lineno, segment))

    for name in selected_classes:
        class_node = module_info.classes.get(name)
        if class_node is None:
            continue
        segment = ast.get_source_segment(module_info.source, class_node)
        if segment:
            ordered_nodes.append((class_node.lineno, segment))

    ordered_nodes.sort(key=lambda item: item[0])
    unique_segments = []
    seen_segments = set()
    for _, segment in ordered_nodes:
        normalized = segment.strip()
        if not normalized or normalized in seen_segments:
            continue
        seen_segments.add(normalized)
        unique_segments.append(segment.rstrip())

    module_text = "\n\n".join(unique_segments).strip() + "\n"
    return module_text, notes


def _tool_destination_relative(source_root: Path, source_file: Path) -> Path:
    relative = source_file.relative_to(source_root)
    if relative.name == "__init__.py":
        relative = relative.parent / "module.py"
    return relative


def _tool_module_name_from_relative(relative: Path) -> str:
    return ".".join(relative.with_suffix("").parts)


def _collect_local_module_dependencies(
    source_file: Path,
    source_root: Path,
    module_lookup: dict[str, Path],
    module_cache: dict[Path, ModuleInfo | None],
    seen: set[Path] | None = None,
) -> set[Path]:
    if seen is None:
        seen = set()
    if source_file in seen:
        return set()
    seen.add(source_file)

    module_info = _get_cached_module_info(source_file, module_cache)
    if module_info is None:
        return set()

    dependencies: set[Path] = set()
    for binding in module_info.imports.values():
        dependency_path: Path | None = None
        if binding.kind == "import":
            dependency_path = _resolve_absolute_module_path(binding.module, module_lookup)
        else:
            dependency_path, _ = _resolve_imported_symbol_source(binding, source_file, module_lookup)
        if dependency_path is None:
            continue
        try:
            dependency_path.relative_to(source_root)
        except ValueError:
            continue
        if dependency_path == source_file:
            continue
        dependencies.add(dependency_path)
        dependencies.update(
            _collect_local_module_dependencies(dependency_path, source_root, module_lookup, module_cache, seen)
        )
    return dependencies


def _write_migration_readme(destination_root: Path, framework: str, agents: list[AgentCandidate]) -> None:
    readme = textwrap.dedent(
        f"""\
        # Migrated Connic Project

        This project was generated by `connic migrate` from a {framework} codebase.

        ## Structure

        ```
        ├── agents/        # Migrated Connic agent YAML files
        ├── tools/         # Migrated Python tools
        ├── middleware/    # Add Connic middleware here if needed
        ├── schemas/       # Add structured output schemas here if needed
        ├── requirements.txt
        └── MIGRATION_REPORT.md
        ```

        ## Next steps

        1. Review `MIGRATION_REPORT.md` for anything that still needs manual work.
        2. Inspect the generated agent YAML files under `agents/`.
        3. Run `connic lint` to validate the migrated project.
        4. Add credentials and deploy with `connic test` or `connic deploy` when ready.

        ## Migrated agents

        {chr(10).join(f"- {agent.name}" for agent in agents) or "- None"}
        """
    )
    (destination_root / "README.md").write_text(readme)


def _write_requirements_file(source_root: Path, destination_root: Path, report_notes: list[str]) -> None:
    source_requirements = source_root / "requirements.txt"
    if source_requirements.exists():
        shutil.copy2(source_requirements, destination_root / "requirements.txt")
        return
    (destination_root / "requirements.txt").write_text(
        "# Review and add the dependencies required by your migrated tools\n"
    )
    report_notes.append("No source requirements.txt found. Review dependencies manually.")


def _write_migration_report(
    destination_root: Path,
    framework: str,
    detection_notes: list[str],
    agents: list[AgentCandidate],
    report_notes: list[str],
) -> None:
    lines = [
        "# Migration Report",
        "",
        f"- Source framework: `{framework}`",
        f"- Generated agents: {len(agents)}",
        "",
        "## Detection Notes",
    ]
    for note in detection_notes:
        lines.append(f"- {note}")

    lines.extend(["", "## Agents"])
    for agent in agents:
        lines.append(f"- `{agent.name}` ({agent.agent_type})")
        if agent.source_file:
            lines.append(f"  - Source: `{agent.source_file}`")
        if agent.model:
            lines.append(f"  - Model: `{agent.model}`")
        if agent.tool_candidates:
            tool_refs = [tool.ref or tool.function_name for tool in agent.tool_candidates]
            lines.append(f"  - Tools: {', '.join(tool_refs)}")
        if agent.agent_ref_keys:
            lines.append(f"  - Agent refs: {', '.join(agent.agent_ref_keys)}")
        for note in agent.notes:
            lines.append(f"  - Note: {note}")

    lines.extend(["", "## Follow-up Items"])
    if report_notes or any(agent.notes for agent in agents):
        for note in report_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No additional follow-up items were detected.")

    (destination_root / "MIGRATION_REPORT.md").write_text("\n".join(lines).rstrip() + "\n")


def _render_agent_yaml(agent: AgentCandidate) -> str:
    data: dict[str, object] = {
        "version": "1.0",
        "name": agent.name,
        "type": agent.agent_type,
        "description": agent.description or f"Migrated from {agent.framework}.",
    }
    if agent.agent_type == "llm":
        data["model"] = agent.model or MIGRATION_DEFAULT_MODEL
        data["system_prompt"] = agent.system_prompt or MIGRATION_DEFAULT_PROMPT
        if agent.tool_candidates:
            tool_refs = [tool.ref for tool in agent.tool_candidates if tool.ref]
            if tool_refs:
                data["tools"] = tool_refs
    elif agent.agent_type == "sequential":
        data["agents"] = agent.agent_ref_keys or []
    else:
        data["model"] = agent.model or MIGRATION_DEFAULT_MODEL
        data["system_prompt"] = agent.system_prompt or MIGRATION_DEFAULT_PROMPT
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)


def _is_same_or_nested_path(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _build_migration_candidates(source_root: Path) -> tuple[str, list[str], list[AgentCandidate], dict[Path, ModuleInfo]]:
    python_files = _collect_python_files(source_root)
    yaml_files = _collect_yaml_files(source_root)
    module_lookup = _build_module_lookup(source_root, python_files)
    module_infos = {
        file_path: module_info
        for file_path in python_files
        if (module_info := _parse_module_info(file_path)) is not None
    }
    framework, detection_notes = _detect_framework(source_root, python_files, yaml_files)

    candidates: list[AgentCandidate] = []
    if framework in {"langchain", "unknown"}:
        for module_info in module_infos.values():
            candidates.extend(_extract_langchain_agents(module_info, module_lookup))

    if framework == "adk" or (framework == "unknown" and not candidates):
        for module_info in module_infos.values():
            candidates.extend(_extract_adk_agents(module_info, module_lookup))
        if not candidates:
            candidates.extend(_extract_adk_yaml_agents(yaml_files))
        if not candidates:
            detection_notes.extend(_detect_custom_adk_patterns(module_infos))

    _dedupe_agent_names(candidates)
    valid_names = {agent.name for agent in candidates}
    for agent in candidates:
        if agent.agent_type == "sequential":
            filtered_refs = []
            for ref in agent.agent_ref_keys:
                if ref in valid_names and ref != agent.name:
                    filtered_refs.append(ref)
                else:
                    agent.notes.append(f"Removed unresolved workflow reference '{ref}'.")
            agent.agent_ref_keys = filtered_refs
            if not agent.agent_ref_keys:
                agent.agent_type = "llm"
                agent.notes.append("Converted unsupported workflow to an LLM placeholder because no valid Connic agent chain was recovered.")
        if agent.agent_type == "llm" and not agent.model:
            agent.notes.append(f"No model could be extracted. Defaulted to {MIGRATION_DEFAULT_MODEL}.")
        if agent.agent_type == "llm" and not agent.system_prompt:
            agent.notes.append("No static system prompt could be extracted. Added a placeholder prompt.")

    return framework, detection_notes, candidates, module_infos


def _write_migrated_tools(
    source_root: Path,
    destination_root: Path,
    agents: list[AgentCandidate],
    module_infos: dict[Path, ModuleInfo],
    report_notes: list[str],
) -> None:
    python_files = _collect_python_files(source_root)
    module_lookup = _build_module_lookup(source_root, python_files)
    module_cache: dict[Path, ModuleInfo | None] = dict(module_infos)
    source_to_functions: dict[Path, set[str]] = {}
    for agent in agents:
        for tool in agent.tool_candidates:
            if tool.ref and tool.source_file is None:
                continue
            if tool.source_file is None:
                continue
            source_to_functions.setdefault(tool.source_file, set()).add(tool.function_name)

    module_name_map: dict[Path, str] = {}
    for source_file, function_names in source_to_functions.items():
        module_info = module_infos.get(source_file)
        if module_info is None:
            report_notes.append(f"Could not parse tool module '{source_file}'.")
            continue
        module_text, extraction_notes = _extract_module_subset(module_info, function_names)
        report_notes.extend(f"{source_file.name}: {note}" for note in extraction_notes)
        if not module_text.strip():
            report_notes.append(f"No tool code was extracted from '{source_file}'.")
            continue
        destination_relative = _tool_destination_relative(source_root, source_file)
        destination_path = destination_root / "tools" / destination_relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(module_text)
        module_name_map[source_file] = _tool_module_name_from_relative(destination_relative)

        support_dependencies = _collect_local_module_dependencies(source_file, source_root, module_lookup, module_cache)
        for dependency_path in sorted(support_dependencies):
            if dependency_path in source_to_functions:
                continue
            dependency_relative = _tool_destination_relative(source_root, dependency_path)
            dependency_destination = destination_root / "tools" / dependency_relative
            if dependency_destination.exists():
                continue
            dependency_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dependency_path, dependency_destination)

    for agent in agents:
        resolved_tools = []
        for tool in agent.tool_candidates:
            if tool.ref and tool.source_file is None:
                resolved_tools.append(tool)
                continue
            if tool.source_file is None:
                continue
            module_name = module_name_map.get(tool.source_file)
            if not module_name:
                report_notes.append(
                    f"Tool '{tool.function_name}' from '{tool.source_file}' could not be written to the Connic project."
                )
                continue
            tool.ref = f"{module_name}.{tool.function_name}"
            resolved_tools.append(tool)
        agent.tool_candidates = _unique_tool_candidates(resolved_tools)


def _write_migrated_agents(destination_root: Path, agents: list[AgentCandidate]) -> None:
    for agent in agents:
        agent_yaml = _render_agent_yaml(agent)
        (destination_root / "agents" / f"{agent.name}.yaml").write_text(agent_yaml)


def _generate_migrated_project(
    source_root: Path,
    destination_root: Path,
    framework: str,
    detection_notes: list[str],
    agents: list[AgentCandidate],
    module_infos: dict[Path, ModuleInfo],
    write_essential_files: WriteEssentialFiles,
) -> list[str]:
    report_notes: list[str] = []
    destination_root.mkdir(parents=True, exist_ok=True)
    for directory in ("agents", "tools", "middleware", "schemas"):
        (destination_root / directory).mkdir(exist_ok=True)

    write_essential_files(destination_root, quiet=True)
    _write_requirements_file(source_root, destination_root, report_notes)
    _write_migrated_tools(source_root, destination_root, agents, module_infos, report_notes)
    _write_migrated_agents(destination_root, agents)
    _write_migration_readme(destination_root, framework, agents)
    report_notes.append("Custom guardrails are not migrated automatically. Recreate them manually if needed.")
    report_notes.append("Review migrated dependencies in requirements.txt before testing or deployment.")
    _write_migration_report(destination_root, framework, detection_notes, agents, report_notes)
    return report_notes


def register_migrate_command(main: click.Group, write_essential_files: WriteEssentialFiles, run_lint: RunLint) -> None:
    @main.command()
    @click.option("--source", "source_path", type=click.Path(path_type=Path, file_okay=False), default=None, help="Path to the existing LangChain or ADK project")
    @click.option("--dest", "destination_path", type=click.Path(path_type=Path, file_okay=False), default=None, help="Path for the new Connic project")
    def migrate(source_path: Path | None, destination_path: Path | None):
        """Migrate a LangChain or ADK project into a Connic project."""
        click.echo()
        click.secho("  Connic Migrate", fg="cyan", bold=True)
        click.echo("  " + "─" * 30)
        click.echo()

        if source_path is None:
            source_path = click.prompt(click.style("  Existing project path", fg="yellow"), type=click.Path(path_type=Path, file_okay=False))
        if destination_path is None:
            destination_path = click.prompt(click.style("  New Connic project path", fg="yellow"), type=click.Path(path_type=Path, file_okay=False))

        if source_path is None or destination_path is None:
            click.echo("Error: Source path and destination path are required.", err=True)
            sys.exit(1)

        source_root = source_path.expanduser().resolve()
        destination_root = destination_path.expanduser().resolve()

        if not source_root.exists():
            click.echo(f"Error: Source path '{source_root}' does not exist.", err=True)
            sys.exit(1)
        if not source_root.is_dir():
            click.echo(f"Error: Source path '{source_root}' is not a directory.", err=True)
            sys.exit(1)
        if source_root == destination_root:
            click.echo("Error: Source path and destination path must be different.", err=True)
            sys.exit(1)
        if _is_same_or_nested_path(destination_root, source_root):
            click.echo("Error: Destination path cannot be inside the source project.", err=True)
            sys.exit(1)
        if destination_root.exists() and any(destination_root.iterdir()):
            click.echo(f"Error: Destination path '{destination_root}' already exists and is not empty.", err=True)
            sys.exit(1)

        click.echo("  Scanning source project...")
        framework, detection_notes, agents, module_infos = _build_migration_candidates(source_root)

        if not agents:
            click.echo(f"  Framework: {framework}")
            for note in detection_notes:
                click.echo(f"    - {note}", err=True)
            click.echo("  ✗ No migratable agents were found.", err=True)
            click.echo("  Add the project manually and use the migration guides for framework-specific help.", err=True)
            sys.exit(1)

        tool_count = len({tool.ref or (str(tool.source_file), tool.function_name) for agent in agents for tool in agent.tool_candidates})
        click.echo(f"  Framework: {framework}")
        click.echo(f"  Agents found: {len(agents)}")
        click.echo(f"  Tools found: {tool_count}")
        for note in detection_notes:
            click.echo(f"    - {note}")
        click.echo()

        report_notes = _generate_migrated_project(
            source_root,
            destination_root,
            framework,
            detection_notes,
            agents,
            module_infos,
            write_essential_files,
        )

        click.echo(f"  Generated Connic project in {destination_root}")
        click.echo("  Running connic lint on the migrated project...")
        lint_ok = run_lint(quiet=True, project_root=str(destination_root))
        click.echo()

        if lint_ok:
            click.secho("  ✓ Migration complete", fg="green", bold=True)
        else:
            click.secho("  ⚠ Migration completed with lint issues", fg="yellow", bold=True)
        click.echo(f"    Project: {destination_root}")
        click.echo(f"    Report:  {destination_root / 'MIGRATION_REPORT.md'}")
        if report_notes:
            click.echo("    Follow-up items were recorded in the migration report.")
        click.echo()
