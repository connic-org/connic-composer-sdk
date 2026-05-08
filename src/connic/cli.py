import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import click
import httpx

from .loader import ProjectLoader
from .migrate import register_migrate_command
from .update_check import print_update_hint

DEFAULT_API_URL = os.environ.get("CONNIC_API_URL", "https://api.connic.co/v1")
DEFAULT_BASE_URL = os.environ.get("CONNIC_BASE_URL", "https://connic.co")
TEMPLATES_REPO = "connic-org/connic-awesome-agents"
TEMPLATES_ZIP_URL = f"https://github.com/{TEMPLATES_REPO}/archive/refs/heads/main.zip"


# =============================================================================
# Output style helpers
# =============================================================================

def _h1(title: str) -> None:
    """Top-of-command banner. Same shape across every CLI subcommand."""
    click.echo()
    click.secho(f"  Connic {title}", fg="cyan", bold=True)
    click.echo("  " + "─" * 30)
    click.echo()


def _step(msg: str) -> None:
    """Announce that a step is starting."""
    click.echo(f"  → {msg}")


def _ok(msg: str) -> None:
    """Sub-detail under a step: success."""
    click.secho(f"    ✓ {msg}", fg="green")


def _err(msg: str) -> None:
    """Sub-detail under a step: failure. Always to stderr."""
    click.secho(f"    ✗ {msg}", fg="red", err=True)


def _warn(msg: str) -> None:
    """Sub-detail under a step: warning."""
    click.secho(f"    ! {msg}", fg="yellow")


def _info(msg: str) -> None:
    """Sub-detail under a step: neutral info, no glyph."""
    click.echo(f"    {msg}")


def _done(msg: str = "Done.") -> None:
    """Final line of a successful command."""
    click.echo()
    click.secho(f"  ✓ {msg}", fg="green", bold=True)
    click.echo()


def _fail_and_exit(msg: str, code: int = 1) -> None:
    """Print an error in the standard format and exit."""
    _err(msg)
    sys.exit(code)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(s: str) -> int:
    """Length of ``s`` ignoring ANSI color escapes."""
    return len(_ANSI_RE.sub("", s))


def _table(headers: list[str], rows: list[list[str]], indent: int = 4) -> None:
    """Render a unicode-box table that respects ANSI-colored cells."""
    if not rows:
        return
    pad = " " * indent
    widths = [_visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _visible_len(cell))

    def cell(text: str, w: int) -> str:
        return text + " " * (w - _visible_len(text))

    top = "┬".join("─" * (w + 2) for w in widths)
    mid = "┼".join("─" * (w + 2) for w in widths)
    bot = "┴".join("─" * (w + 2) for w in widths)
    click.echo(f"{pad}┌{top}┐")
    click.echo(f"{pad}│ " + " │ ".join(cell(h, w) for h, w in zip(headers, widths)) + " │")
    click.echo(f"{pad}├{mid}┤")
    for row in rows:
        click.echo(f"{pad}│ " + " │ ".join(cell(c, w) for c, w in zip(row, widths)) + " │")
    click.echo(f"{pad}└{bot}┘")


# =============================================================================
# File Validation Constants and Helpers
# =============================================================================

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    ".py",      # Python scripts
    ".yaml",    # YAML config
    ".yml",     # YAML config (alt extension)
    ".txt",     # Text files
    ".json",    # JSON data/schemas
    ".csv",     # CSV data files
    ".md",      # Markdown (prompts, templates)
    ".toml",    # TOML configuration
    ".jsonl",   # JSON Lines data
}

# Maximum total upload size: 25MB (covers tests/files/ fixtures).
# Code/config (everything outside tests/files/) is held to a 1MB sub-cap.
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 26,214,400 bytes
MAX_CODE_SIZE = 1024 * 1024  # legacy 1MB cap, applied to non-fixture content

# Paths under tests/files/ are treated as opaque fixture content: any
# extension, binary OK, larger size budget. tests/builders/ accepts only .py.
TEST_FILES_PREFIX = "tests/files"
TEST_BUILDERS_PREFIX = "tests/builders"


def _is_under(path: Path, prefix: str) -> bool:
    """True if `path` is at or beneath the slash-joined `prefix`."""
    parts = path.parts
    prefix_parts = tuple(prefix.split("/"))
    return parts[: len(prefix_parts)] == prefix_parts


def _validate_project_files() -> tuple[bool, str, list[Path]]:
    """
    Validate all project files before packaging.

    Performs basic validation (extension and size checks).
    Full content validation is done server-side.

    Returns:
        Tuple of (is_valid, error_message, list_of_valid_files).
    """
    valid_files = []
    total_size = 0
    code_size = 0

    dirs_to_check = ["agents", "tools", "middleware", "schemas", "guardrails", "hooks", "tests"]

    for dirname in dirs_to_check:
        dirpath = Path(dirname)
        if not dirpath.exists():
            continue

        for filepath in dirpath.rglob("*"):
            if not filepath.is_file():
                continue

            # Skip hidden files and __pycache__
            if any(part.startswith(".") for part in filepath.parts):
                continue
            if "__pycache__" in str(filepath) or filepath.suffix == ".pyc":
                continue

            try:
                content = filepath.read_bytes()
            except IOError as e:
                return False, f"Could not read {filepath}: {e}", []

            total_size += len(content)
            if total_size > MAX_UPLOAD_SIZE:
                return False, f"Total file size exceeds {MAX_UPLOAD_SIZE:,} byte limit", []

            # tests/files/<...>: opaque test fixture, any extension/binary.
            if _is_under(filepath, TEST_FILES_PREFIX):
                valid_files.append(filepath)
                continue

            # tests/builders/<...>: must be Python source.
            if _is_under(filepath, TEST_BUILDERS_PREFIX):
                if filepath.suffix != ".py":
                    return False, (
                        f"{filepath}: only .py files are allowed under "
                        f"{TEST_BUILDERS_PREFIX}/"
                    ), []
                code_size += len(content)
                if code_size > MAX_CODE_SIZE:
                    return False, (
                        f"Code/config size exceeds {MAX_CODE_SIZE:,} byte limit. "
                        "Move large fixtures into tests/files/."
                    ), []
                valid_files.append(filepath)
                continue

            ext = filepath.suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return False, f"{filepath}: File type '{ext}' not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", []

            code_size += len(content)
            if code_size > MAX_CODE_SIZE:
                return False, (
                    f"Code/config size exceeds {MAX_CODE_SIZE:,} byte limit. "
                    "Move large fixtures into tests/files/."
                ), []

            valid_files.append(filepath)

    # Check requirements.txt
    req_file = Path("requirements.txt")
    if req_file.exists():
        try:
            content = req_file.read_bytes()
            total_size += len(content)
            if total_size > MAX_UPLOAD_SIZE:
                return False, f"Total file size exceeds {MAX_UPLOAD_SIZE:,} byte limit", []
            code_size += len(content)
            if code_size > MAX_CODE_SIZE:
                return False, (
                    f"Code/config size exceeds {MAX_CODE_SIZE:,} byte limit. "
                    "Move large fixtures into tests/files/."
                ), []
            valid_files.append(req_file)
        except IOError as e:
            return False, f"Could not read requirements.txt: {e}", []

    return True, "", valid_files


@click.group()
@click.version_option(version="0.1.18", prog_name="connic")
def main():
    """Connic Composer SDK - Build agents with code."""
    print_update_hint()


def _write_essential_files(base_path: Path, quiet: bool = False):
    """Write files that are always created during project init."""

    # .gitignore
    gitignore = base_path / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text('''# Connic
.connic

# Python
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Environment
.env
.env.local

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
''')

    # requirements.txt
    requirements = base_path / "requirements.txt"
    if not requirements.exists():
        requirements.write_text('''# Add your tool dependencies below
# httpx>=0.25.0  # For async HTTP requests
# pandas>=2.0.0  # For data processing
''')

    # README.md
    readme = base_path / "README.md"
    if not readme.exists():
        readme.write_text('''# Connic Agent Project

This project contains AI agents built with the Connic Composer SDK.

## Structure

```
├── agents/       # Agent YAML configurations (subfolders supported)
├── tools/        # Python tool modules
├── middleware/    # Optional middleware for agents
├── schemas/      # Output schemas for structured responses
└── requirements.txt
```

## Getting Started

1. Create your first agent anywhere under `agents/`:
   ```yaml
   version: "1.0"
   name: my-agent
   type: llm
   model: gemini/gemini-2.5-flash
   description: "My first agent"
   system_prompt: |
     You are a helpful assistant.
   ```

2. Optionally add tools in `tools/` and reference them in your agent config.

3. Validate your project:
   ```bash
   connic lint
   ```

4. Connect your repository to Connic and push to deploy, or run `connic deploy`.

## Documentation

See the [Connic Composer docs]({base_url}/docs/v1/composer/overview) for:
- [Agent Configuration]({base_url}/docs/v1/composer/agent-configuration)
- [Writing Tools]({base_url}/docs/v1/composer/write-tools)
- [Middleware]({base_url}/docs/v1/composer/middleware)
'''.format(base_url=DEFAULT_BASE_URL))

    # Output (skip when quiet=True, e.g. when init used templates)
    if not quiet:
        _step("Created essential files:")
        _info(".gitignore")
        _info("requirements.txt")
        _info("README.md")
        _step("Next steps:")
        _info("1. Create your first agent anywhere under agents/")
        _info("2. Run `connic lint` to validate your project")
        _info("3. Push to your connected repository to deploy")
        _done(f"Initialized Connic project in {base_path.resolve()}")


def _get_local_templates_path() -> Path | None:
    """Find local connic-awesome-agents directory for development."""
    cwd = Path.cwd()
    for candidate in [cwd / "connic-awesome-agents", cwd.parent / "connic-awesome-agents"]:
        if candidate.exists():
            return candidate
    return None


def _fetch_templates_from_github() -> Path | None:
    """Download repo zip and extract. Returns path to extracted dir or None."""
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            r = client.get(TEMPLATES_ZIP_URL)
            r.raise_for_status()
            zip_bytes = r.content
    except Exception as e:
        click.echo(f"Could not fetch templates from GitHub: {e}", err=True)
        return None
    try:
        tmp = tempfile.mkdtemp()
        zip_path = Path(tmp) / "templates.zip"
        zip_path.write_bytes(zip_bytes)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        # Extract creates connic-awesome-agents-main/
        extracted = Path(tmp) / "connic-awesome-agents-main"
        if not extracted.exists():
            return None
        return extracted
    except Exception as e:
        click.echo(f"Could not extract templates: {e}", err=True)
        return None


def _merge_template_into_project(
    template_src: Path,
    base_path: Path,
    requirements_lines: list[str],
    template_id: str,
) -> str | None:
    """Copy template folder contents into project, merge requirements.
    
    Returns the template README content if it exists, None otherwise.
    """
    for subdir in ["agents", "tools", "middleware", "schemas", "guardrails", "hooks"]:
        src_dir = template_src / subdir
        dst_dir = base_path / subdir
        if src_dir.exists():
            dst_dir.mkdir(exist_ok=True)
            for f in src_dir.rglob("*"):
                if not f.is_file() or f.name.startswith("_"):
                    continue
                relative_path = f.relative_to(src_dir)
                destination = dst_dir / template_id / relative_path if subdir == "agents" else dst_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(f.read_bytes())
    req_file = template_src / "requirements.txt"
    if req_file.exists():
        requirements_lines.extend(req_file.read_text().strip().splitlines())
    readme_file = template_src / "README.md"
    if readme_file.exists():
        return readme_file.read_text().strip()
    return None


def _write_merged_requirements(base_path: Path, lines: list[str]) -> None:
    """Write merged, deduplicated requirements.txt."""
    seen = set()
    unique = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique.append(line)
        else:
            pkg = stripped.split("#")[0].strip().split("==")[0].split(">=")[0].split("<=")[0]
            if pkg and pkg not in seen:
                seen.add(pkg)
                unique.append(line)
    (base_path / "requirements.txt").write_text("\n".join(unique) + "\n")


def _append_template_readmes(base_path: Path, template_readmes: list[str]) -> None:
    """Append template README content to the project README."""
    readme = base_path / "README.md"
    if not readme.exists():
        return
    existing = readme.read_text()

    sections = []
    for content in template_readmes:
        lines = content.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        sections.append(f"## {heading}\n\n{body}")

    if sections:
        separator = "\n\n---\n\n"
        readme.write_text(existing.rstrip() + separator + separator.join(sections) + "\n")


@main.command()
@click.argument("name", required=False, default=".")
@click.option(
    "--templates",
    "-t",
    default=None,
    help="Comma-separated template names (e.g., invoice,support). Browse at connic.co/agents",
)
def init(name: str, templates: str | None):
    """Initialize a new Connic project.

    Creates the project structure. Use --templates to add agent templates from
    connic-awesome-agents. Without --templates, creates a clean scaffold only.

    Examples:
        connic init                        # Clean project, no examples
        connic init my-project             # Clean project in new directory
        connic init . --templates=invoice  # Project with invoice template
        connic init app --templates=invoice,support
    """
    _h1("Init")

    base_path = Path(name)

    if name != ".":
        if base_path.exists():
            _fail_and_exit(f"Directory '{name}' already exists.")
        base_path.mkdir(parents=True)
        _step(f"Created directory: {name}")

    _step("Creating project structure...")
    (base_path / "agents").mkdir(exist_ok=True)
    (base_path / "tools").mkdir(exist_ok=True)
    (base_path / "middleware").mkdir(exist_ok=True)
    (base_path / "schemas").mkdir(exist_ok=True)
    _ok("agents/, tools/, middleware/, schemas/")

    if templates:
        # Fetch and merge templates
        template_ids = [t.strip().lower() for t in templates.split(",") if t.strip()]
        if not template_ids:
            _fail_and_exit("No valid template names provided.")

        _step("Fetching templates from connic-awesome-agents...")
        extracted = _fetch_templates_from_github()
        if not extracted:
            local_path = _get_local_templates_path()
            if local_path:
                extracted = local_path
                _warn("GitHub unavailable; using local connic-awesome-agents")
            else:
                _fail_and_exit("Could not fetch templates. Try again or use a local connic-awesome-agents folder.")
        else:
            _ok("Fetched")

        _step("Merging templates...")
        requirements_lines = []
        template_readmes: list[str] = []
        for tid in template_ids:
            template_dir = extracted / tid
            if not template_dir.is_dir():
                _fail_and_exit(f"Template '{tid}' not found in connic-awesome-agents")
            readme_content = _merge_template_into_project(template_dir, base_path, requirements_lines, tid)
            if readme_content:
                template_readmes.append(readme_content)
            _ok(f"Added template: {tid}")

        if requirements_lines:
            _write_merged_requirements(base_path, requirements_lines)
        _write_essential_files(base_path, quiet=True)

        if template_readmes:
            _append_template_readmes(base_path, template_readmes)

        _step("Next steps:")
        _info("1. Run `connic lint` to validate your project")
        _info("2. Run `connic test` to test against Connic cloud")
        _info("3. Run `connic deploy` when ready")
        _done(f"Initialized with templates: {', '.join(template_ids)}")
        return

    # No templates: create clean project only (no example files)
    _write_essential_files(base_path)


def _run_lint(verbose: bool = False, quiet: bool = False, project_root: str = ".") -> bool:
    """Run project validation. Returns True on success, False on failure.

    Args:
        verbose: Show extra detail per agent.
        quiet:   Suppress per-agent output (used by deploy for a compact pre-flight check).
    """
    errors: list[str] = []

    try:
        loader = ProjectLoader(project_root)
    except Exception as e:
        _err(str(e))
        return False

    # Discover tools
    if not quiet:
        _step("Discovering tools...")
    try:
        tools = loader.discover_tools()
        errors.extend(loader._load_errors)
        loader._load_errors.clear()
        total_tools = sum(len(funcs) for funcs in tools.values())
        if not quiet:
            if tools:
                _ok(f"{total_tools} tool(s) in {len(tools)} module(s)")
                for module, functions in sorted(tools.items()):
                    if verbose:
                        _info(f"{module}:")
                        for func in functions:
                            _info(f"  - {func}")
                    else:
                        _info(f"{module}: {', '.join(functions)}")
            else:
                _info("No tools found in tools/ directory")
    except FileNotFoundError:
        if not quiet:
            _info("No tools/ directory found")
        tools = {}

    # Discover middlewares
    if not quiet:
        _step("Discovering middlewares...")
    try:
        middlewares = loader.discover_middlewares()
        loader._load_errors.clear()  # middleware errors are non-fatal
        if not quiet:
            if middlewares:
                _ok(f"middlewares for {len(middlewares)} agent(s)")
                for agent_name, hooks in sorted(middlewares.items()):
                    _info(f"{agent_name}: {', '.join(hooks)}")
            else:
                _info("No middlewares found in middleware/ directory")
    except FileNotFoundError:
        if not quiet:
            _info("No middleware/ directory found")
        middlewares = {}

    # Discover tool hooks
    if not quiet:
        _step("Discovering tool hooks...")
    try:
        tool_hooks = loader.discover_tool_hooks()
        loader._load_errors.clear()  # hook errors are non-fatal
        if not quiet:
            if tool_hooks:
                _ok(f"hooks for {len(tool_hooks)} agent(s)")
                for agent_name, available in sorted(tool_hooks.items()):
                    _info(f"{agent_name}: {', '.join(available)}")
            else:
                _info("No hooks found in hooks/ directory")
    except FileNotFoundError:
        if not quiet:
            _info("No hooks/ directory found")
        tool_hooks = {}

    # Load agents
    if not quiet:
        _step("Loading agents...")
    try:
        agents = loader.load_agents()
        errors.extend(loader._load_errors)
        loader._load_errors.clear()
        api_spec_warnings = list(loader._api_spec_warnings)
        loader._api_spec_warnings.clear()
    except FileNotFoundError as e:
        _err(str(e))
        _info("Run `connic init` to create a sample project.")
        return False

    if not agents and not errors:
        _err("No agents found in agents/")
        _info("Run `connic init` to create a sample project.")
        return False

    # Validate each agent
    loaded_agent_names = {a.config.name for a in agents}
    for agent in agents:
        config = agent.config
        agent_type = config.type.value if hasattr(config.type, 'value') else str(config.type)

        if agent_type == "sequential":
            for ref in config.agents:
                if ref not in loaded_agent_names:
                    location = f" ({config.source_path})" if config.source_path else ""
                    errors.append(f"Sequential agent '{config.name}'{location} references unknown agent '{ref}'")

    if quiet:
        if errors:
            for err in errors:
                _err(err)
            return False
        if api_spec_warnings:
            _info(f"{len(api_spec_warnings)} API spec tool ref(s) skipped (validated at deploy time)")
        agent_names = [a.config.name for a in agents]
        _ok(f"Lint passed: {len(agents)} agent(s) validated ({', '.join(agent_names)})")
        return True

    # Verbose output: print per-agent boxes
    _ok(f"{len(agents)} agent(s) loaded")
    click.echo()

    for agent in agents:
        config = agent.config
        agent_type = config.type.value if hasattr(config.type, 'value') else str(config.type)
        type_label = {"llm": "🧠 LLM", "sequential": "🔗 Sequential", "tool": "🔧 Tool"}.get(agent_type, agent_type)

        click.echo(f"  ┌─ {config.name} [{type_label}]")
        if config.source_path:
            click.echo(f"  │  Path: {config.source_path}")
        click.echo(f"  │  Description: {config.description}")

        if agent_type == "llm":
            click.echo(f"  │  Model: {config.model}")
            if config.fallback_model:
                click.echo(f"  │  Fallback Model: {config.fallback_model}")
            if verbose:
                click.echo(f"  │  Temperature: {config.temperature}")
        elif agent_type == "sequential":
            click.echo(f"  │  Chain: {' → '.join(config.agents)}")
        elif agent_type == "tool":
            click.echo(f"  │  Tool: {config.tool_name}")

        if verbose:
            click.echo(f"  │  Max Concurrent Runs: {config.max_concurrent_runs}")
            if config.retry_options:
                retry_info = f"  │  Retry: {config.retry_options.attempts} attempts, max {config.retry_options.max_delay}s delay"
                if config.retry_options.rerun_middleware:
                    retry_info += " (rerun middleware)"
                click.echo(retry_info)
            if config.timeout:
                click.echo(f"  │  Timeout: {config.timeout}s")

        if agent_type == "llm":
            if agent.tools:
                tool_names = [t.ref or t.name for t in agent.tools]
                click.echo(f"  │  Tools: {', '.join(tool_names)}")
            else:
                click.echo("  │  Tools: (none)")

            if config.mcp_servers:
                for mcp_server in config.mcp_servers:
                    bridge_suffix = f" via bridge {mcp_server.bridge}" if mcp_server.bridge else ""
                    click.echo(
                        f"  │  MCP Server: {mcp_server.name} ({mcp_server.url}){bridge_suffix}"
                    )

        if agent_type == "sequential":
            for ref in config.agents:
                if ref not in loaded_agent_names:
                    click.echo(f"  │  ✗ Unknown agent: '{ref}'")

        click.echo("  └─")
        click.echo()

    if api_spec_warnings:
        _step(f"{len(api_spec_warnings)} API spec tool ref(s) skipped (validated at deploy time):")
        for ref in api_spec_warnings:
            _info(f"- {ref}")

    if errors:
        _step(f"Validation failed with {len(errors)} error(s):")
        for err in errors:
            _err(err)
        return False

    _done("Project validation complete. To deploy, push to your repository or run `connic deploy`.")
    return True


register_migrate_command(main, _write_essential_files, _run_lint)

@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def lint(verbose: bool):
    """Validate agent configurations and tools.

    Loads all agents and tools, validates configurations,
    and displays a summary of the project.
    """
    _h1("Lint")
    if not _run_lint(verbose=verbose):
        sys.exit(1)


@main.command()
def tools():
    """List all available tools in the project."""
    _h1("Tools")
    try:
        loader = ProjectLoader(".")
        discovered = loader.discover_tools()
    except FileNotFoundError:
        _fail_and_exit("No tools/ directory found.")

    if not discovered:
        _step("No tools found in tools/ directory.")
        _info("Create Python files in tools/ with typed functions.")
        sys.exit(0)

    _step("Available tools:")
    for module, functions in sorted(discovered.items()):
        _info(f"{module.replace('.', '/')}.py:")
        for func_name in functions:
            try:
                tool = loader._resolve_tools(f"{module}.{func_name}")[0]
                desc = tool.description.split('\n')[0][:60]
                if len(tool.description.split('\n')[0]) > 60:
                    desc += "..."
                _info(f"  - {func_name}: {desc}")
            except Exception:
                _info(f"  - {func_name}")

    _step("Reference in agent YAML:")
    _info("Use the exact module path under tools/, e.g. <module>.<function> or <directory>.<module>.<function>")


# =============================================================================
# Test Command - Cloud Dev Mode with Hot Reload
# =============================================================================


# Backend `phase` strings → one-line user-facing labels. Unknown phases are
# shown verbatim so the CLI is forward-compatible with new backend additions.
_TEST_PHASE_LABELS = {
    "validating": "Validating uploaded files on backend...",
    "building": "Building tests...",
    "starting_container": "Starting test runner container...",
    "running_tests": "Test runner is executing your test cases...",
    "done": "Test runner finished.",
    "error": "Backend reported an error.",
}


def _package_project_for_tests(*, quiet: bool = False) -> tuple[bytes, list[Path], int]:
    """Validate project files and build the gzip tarball used by `/test-runs`.

    Returns ``(tar_data, valid_files, n_test_files)``. Raises ``ValueError``
    on validation or size errors. Whether ``n_test_files == 0`` is fatal is
    up to the caller.
    """
    import io
    import tarfile

    if not quiet:
        _step("Validating project files...")
    is_valid, err, valid_files = _validate_project_files()
    if not is_valid:
        raise ValueError(f"File validation failed: {err}")
    test_files = [f for f in valid_files if f.parts and f.parts[0] == "tests"]
    if not quiet:
        _ok(f"{len(valid_files)} files, {len(test_files)} test file(s)")
        _step("Packaging upload...")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in valid_files:
            tar.add(f, arcname=str(f))
    tar_data = buf.getvalue()
    if len(tar_data) > MAX_UPLOAD_SIZE:
        raise ValueError(f"Package size ({len(tar_data):,} bytes) exceeds {MAX_UPLOAD_SIZE:,} byte limit")
    if not quiet:
        _ok(f"{len(tar_data):,} bytes")
    return tar_data, valid_files, len(test_files)


def _kickoff_test_run(client: "httpx.Client", project_id: str, env_id: str, tar_data: bytes) -> str:
    """POST a tarball as a test run. Returns the new ``test_run_id``.

    Raises ``RuntimeError`` with a user-facing message on any non-2xx response.
    """
    import base64

    resp = client.post(
        f"/projects/{project_id}/test-runs",
        json={
            "files_data": base64.b64encode(tar_data).decode("utf-8"),
            "environment_id": env_id,
        },
    )
    if resp.status_code == 400:
        raise RuntimeError(f"Test request rejected: {resp.json().get('detail', resp.text)}")
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"Failed to start test run: {resp.text}")
    return resp.json()["id"]


def _poll_test_run(client: "httpx.Client", poll_url: str, *, quiet: bool = False) -> dict:
    """Poll a test run until it reaches a terminal status. Returns the result dict.

    Prints phase transitions and the "running N cases" announcement unless
    ``quiet``. Tolerates a handful of consecutive transient network errors
    before giving up (raises ``RuntimeError``).
    """
    import time

    last_phase: str | None = None
    announced_running = False
    consecutive_poll_errors = 0
    while True:
        try:
            resp = client.get(poll_url)
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            consecutive_poll_errors += 1
            if consecutive_poll_errors >= 5:
                raise RuntimeError(f"Lost contact with backend ({e!s})")
            time.sleep(5)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to poll test run: {resp.text}")
        consecutive_poll_errors = 0
        result = resp.json()
        phase = result.get("phase")
        if not quiet:
            if phase and phase != last_phase:
                _step(_TEST_PHASE_LABELS.get(phase, phase))
                last_phase = phase
            if not announced_running and phase == "running_tests" and result.get("total_cases"):
                _info(f"Running {result['total_cases']} test case(s)...")
                announced_running = True
        if result["status"] in ("passed", "failed", "error"):
            return result
        time.sleep(2)


def _render_test_cases(cases: list[dict]) -> None:
    """Render the per-case result table and the failure-detail block (if any)."""
    if not cases:
        return
    click.echo()
    rows = []
    for c in cases:
        cell = (
            click.style(" PASS ", fg="green", bold=True)
            if c["passed"]
            else click.style(" FAIL ", fg="red", bold=True)
        )
        rows.append([
            cell,
            f"{c['agent_name']}::{c['test_name']}",
            f"{c['successes']}/{c['runs']}",
            f"{c['success_threshold']}%",
        ])
    _table(["Result", "Test", "Runs", "Threshold"], rows)

    failed = [c for c in cases if not c["passed"] and c.get("failure_reason")]
    if failed:
        click.echo()
        _step("Failure details:")
        for c in failed:
            _err(f"{c['agent_name']}::{c['test_name']}: {c['failure_reason']}")


def _render_dashboard_link(project_id: str, deployment_id: str, env_id: str) -> None:
    """Print the dashboard link to the deployment that backed a test run."""
    click.echo()
    _step("View detailed results in the dashboard:")
    _info(f"{DEFAULT_BASE_URL}/projects/{project_id}/deployments/{deployment_id}?env={env_id}")


def _run_tests_in_dev_session(client: "httpx.Client", project_id: str, env_id: str) -> None:
    """Run ./tests against the dev session's env. Never calls sys.exit."""
    click.echo()
    try:
        tar_data, _, n_tests = _package_project_for_tests()
    except ValueError as e:
        _err(str(e))
        return
    if n_tests == 0:
        _warn("No tests/ directory found — nothing to run.")
        return

    _step("Submitting tests to backend (target: dev session env)...")
    try:
        test_run_id = _kickoff_test_run(client, project_id, env_id, tar_data)
    except (RuntimeError, httpx.RequestError) as e:
        _err(str(e))
        return
    _ok(f"Test run id: {test_run_id}")

    try:
        result = _poll_test_run(client, f"/projects/{project_id}/test-runs/{test_run_id}")
    except RuntimeError as e:
        _err(f"{e}; giving up on this run.")
        return

    if result["status"] == "error":
        _err(f"Test run errored: {result.get('error') or 'unknown error'}")
        return

    cases = result.get("cases", [])
    _render_test_cases(cases)
    if result.get("deployment_id"):
        _render_dashboard_link(project_id, result["deployment_id"], env_id)

    click.echo()
    passed_n = sum(1 for c in cases if c["passed"])
    summary = f"{passed_n}/{len(cases)} cases passed."
    if result["status"] == "passed":
        click.secho(f"  ✓ {summary}", fg="green", bold=True)
    else:
        click.secho(f"  ✗ {summary}", fg="red", bold=True)
    click.echo()


@main.command()
@click.argument("name", required=False, default=None)
@click.option("--api-url", envvar="CONNIC_API_URL", default=DEFAULT_API_URL, help="Connic API URL")
@click.option("--api-key", envvar="CONNIC_API_KEY", default=None, help="Connic API key")
@click.option("--project-id", envvar="CONNIC_PROJECT_ID", default=None, help="Connic project ID")
def dev(name: str, api_url: str, api_key: str, project_id: str):
    """
    Start a dev session with hot-reload against Connic cloud.

    Creates an isolated test environment and syncs your local files
    for rapid development. Changes are reflected in 2-5 seconds.

    \b
    Examples:
        connic dev               # Ephemeral test env (auto-deleted on exit)
        connic dev my-feature    # Named test env (persists after exit)

    Environment variables:
        CONNIC_API_URL      - API URL (default: https://api.connic.co/v1)
        CONNIC_API_KEY      - Your API key
        CONNIC_PROJECT_ID   - Your project ID
    """
    import hashlib
    import io
    import queue
    import signal
    import tarfile
    import threading
    import time

    import httpx
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    # Raw-mode keypress reading is Unix-only; on Windows we fall back to Ctrl+C.
    try:
        import select as _select_mod
        import termios as _termios_mod
        import tty as _tty_mod
        _KEYS_SUPPORTED = True
    except ImportError:
        _select_mod = None
        _termios_mod = None
        _tty_mod = None
        _KEYS_SUPPORTED = False
    
    # Validate required config
    # Try to read from .connic file
    connic_file = Path(".connic")
    if connic_file.exists():
        try:
            import json
            config = json.loads(connic_file.read_text())
            api_key = api_key or config.get("api_key")
            project_id = project_id or config.get("project_id")
        except Exception:
            pass
    
    _h1("Dev")

    if not api_key:
        _err("API key required. Set CONNIC_API_KEY or use --api-key")
        _info("Create one in the dashboard: Project Settings → CLI → Create Key")
        _info("Or run: connic login")
        sys.exit(1)

    if not project_id:
        _err("Project ID required. Set CONNIC_PROJECT_ID or use --project-id")
        _info("Find your Project ID in the dashboard: Project Settings → CLI")
        _info("Or run: connic login")
        sys.exit(1)

    # Validate local project
    _step("Validating project files...")
    try:
        loader = ProjectLoader(".")
        agents = loader.load_agents()
        if not agents:
            _fail_and_exit("No agents found. Run `connic init` first.")
        agent_summaries = [
            f"{a.config.name} ({a.config.source_path})" if a.config.source_path else a.config.name
            for a in agents
        ]
        _ok(f"{len(agents)} agent(s): {', '.join(agent_summaries)}")
    except Exception as e:
        _fail_and_exit(f"Error loading project: {e}")

    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    client = httpx.Client(base_url=api_url, headers=headers, timeout=120.0)
    
    session_id = None
    cleaned_up = False
    server_terminated = False  # True if session was already cleaned up by server
    
    def cleanup():
        """Clean up dev session on exit."""
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True

        # Skip cleanup if server already terminated the session
        if session_id and not server_terminated:
            click.echo()
            _step("Cleaning up dev session...")
            try:
                resp = client.delete(f"/test-sessions/{session_id}")
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("environment_deleted"):
                        _ok("Ephemeral environment deleted.")
                    else:
                        _ok("Session ended (named environment preserved).")
                elif resp.status_code != 404:
                    _warn(f"Cleanup returned {resp.status_code}")
            except Exception as e:
                _warn(f"Cleanup failed: {e}")
        client.close()
    
    # Register cleanup handler
    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create test session
        _step("Creating dev session...")
        body = {}
        if name:
            body["name"] = name
            _info(f"Using named environment: {name}")
        else:
            _info("Creating ephemeral environment (will be deleted on exit)")

        resp = client.post(f"/projects/{project_id}/test-sessions", json=body)

        if resp.status_code == 409:
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text
            _err(detail)
            _info("To stop an existing session, press Ctrl+C in the terminal where it's running.")
            sys.exit(1)
        elif resp.status_code != 200:
            _fail_and_exit(f"Error creating dev session: {resp.text}")

        session_data = resp.json()
        session_id = session_data["id"]
        env_id = session_data["environment_id"]
        env_name = session_data["environment_name"]

        _ok(f"Session id: {session_id}")
        _ok(f"Environment: {env_name}")

        # Poll for container to be ready
        _step("Starting dev runner container (this can take a minute on first build)...")
        max_wait = 600  # 10 minutes max
        poll_interval = 3
        waited = 0
        container_ready = False

        consecutive_errors = 0
        while waited < max_wait:
            try:
                status_resp = client.get(f"/test-sessions/{session_id}")
                if status_resp.status_code == 200:
                    consecutive_errors = 0
                    status_data = status_resp.json()
                    container_status = status_data.get("container_status", "starting")

                    if container_status == "running":
                        _ok("Container: running")
                        container_ready = True
                        break
                    elif container_status == "failed":
                        _err("Container failed to start")
                        cleanup()
                        sys.exit(1)
                else:
                    consecutive_errors += 1

            except Exception:
                consecutive_errors += 1

            time.sleep(poll_interval)
            waited += poll_interval

        if not container_ready:
            _err("Container did not start within 10 minutes")
            _info("Check backend logs for details")
            cleanup()
            sys.exit(1)
        
        # Create and upload tarball of agent files
        def create_tarball() -> tuple[bytes, str]:
            """Create a tarball of agent files and return (content, hash)."""
            # Validate files first
            is_valid, error, valid_files = _validate_project_files()
            if not is_valid:
                raise ValueError(f"File validation failed: {error}")
            
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                # Add only validated files
                for filepath in valid_files:
                    arcname = str(filepath)
                    tar.add(filepath, arcname=arcname)
            
            content = buffer.getvalue()
            
            # Final size check on compressed tarball
            if len(content) > MAX_UPLOAD_SIZE:
                raise ValueError(f"Package size ({len(content):,} bytes) exceeds {MAX_UPLOAD_SIZE:,} byte limit")
            
            content_hash = hashlib.sha256(content).hexdigest()
            return content, content_hash
        
        def upload_files() -> tuple[str | None, int, str | None]:
            """Upload current files to the test session.
            
            Returns:
                Tuple of (files_hash, size_bytes, error_message).
                If error_message is set, files_hash will be None.
            """
            try:
                content, content_hash = create_tarball()
            except ValueError as e:
                # Validation error - return error message instead of crashing
                return None, 0, str(e)
            
            upload_resp = client.post(
                f"/test-sessions/{session_id}/files",
                files={"file": ("files.tar.gz", content, "application/gzip")},
                timeout=60.0,
            )
            
            if upload_resp.status_code == 200:
                result = upload_resp.json()
                return result.get("files_hash"), result.get("size_bytes"), None
            elif upload_resp.status_code == 400:
                # Check if this is a "session not active" error
                try:
                    detail = upload_resp.json().get("detail", "")
                    if "not active" in detail.lower():
                        return None, 0, "SESSION_ENDED"
                except Exception:
                    pass
                return None, 0, f"Upload failed: {upload_resp.text}"
            elif upload_resp.status_code == 404:
                return None, 0, "SESSION_ENDED"
            else:
                return None, 0, f"Upload failed: {upload_resp.text}"
        
        # Initial upload
        _step("Uploading initial files...")
        current_hash, size, error = upload_files()
        if error == "SESSION_ENDED":
            _err("Session ended unexpectedly")
            cleanup()
            sys.exit(1)
        elif error:
            _warn(error)
            _info("Fix the issue and save to retry...")
            current_hash = None  # Will retry on file change
        elif current_hash:
            _ok(f"Uploaded {size} bytes (hash: {current_hash[:16]}...)")

        keys_active = _KEYS_SUPPORTED and sys.stdin.isatty()

        _step("Watching for file changes...")
        dashboard_url = f"{DEFAULT_BASE_URL}/projects/{project_id}/agents?env={env_id}"
        _info(f"View and trigger your agents: {dashboard_url}")
        if keys_active:
            _info("Keys: [r] refresh  [t] run tests  [q] quit  (or Ctrl+C)")
        else:
            _info("Press Ctrl+C to stop.")
        click.echo()

        last_upload_time = time.time()
        pending_upload = False
        next_upload_label: str | None = None
        DEBOUNCE_SECONDS = 1.0  # Wait for changes to settle

        # Keypress reader: daemon thread pushes intents onto a queue drained by the main loop.
        key_queue: "queue.Queue[str]" = queue.Queue()
        key_stop = threading.Event()
        key_old_termios = None
        key_thread = None

        def _key_reader_loop():
            assert _select_mod is not None
            while not key_stop.is_set():
                try:
                    r, _, _ = _select_mod.select([sys.stdin], [], [], 0.2)
                except (OSError, ValueError):
                    return
                if not r:
                    continue
                try:
                    ch = sys.stdin.read(1)
                except (OSError, ValueError):
                    return
                if not ch:
                    return
                low = ch.lower()
                if low == "r":
                    key_queue.put("refresh")
                elif low == "t":
                    key_queue.put("test")
                elif low == "q":
                    key_queue.put("quit")

        if keys_active:
            try:
                key_old_termios = _termios_mod.tcgetattr(sys.stdin.fileno())
                _tty_mod.setcbreak(sys.stdin.fileno())
                key_thread = threading.Thread(target=_key_reader_loop, daemon=True)
                key_thread.start()
            except OSError:
                keys_active = False
                key_old_termios = None
        
        class FileChangeHandler(FileSystemEventHandler):
            def on_any_event(self, event):
                nonlocal pending_upload, last_upload_time
                
                # Ignore directories and hidden files
                if event.is_directory:
                    return
                
                src_path = Path(event.src_path)
                
                # Ignore hidden files
                if any(part.startswith(".") for part in src_path.parts):
                    return
                
                # Ignore __pycache__ and .pyc files
                if "__pycache__" in str(src_path) or src_path.suffix == ".pyc":
                    return
                
                # Check if file is in a watched directory or is requirements.txt
                watched_dirs = ["agents", "tools", "middleware", "schemas", "guardrails", "hooks", "tests"]
                is_watched = any(d in src_path.parts for d in watched_dirs)
                is_requirements = src_path.name == "requirements.txt"
                
                if not is_watched and not is_requirements:
                    return
                
                click.echo(f"  [{time.strftime('%H:%M:%S')}] → Detected change: {src_path.name}")
                pending_upload = True
                last_upload_time = time.time()
        
        observer = Observer()
        handler = FileChangeHandler()
        
        # Watch the project directories
        for dirname in ["agents", "tools", "middleware", "schemas", "guardrails", "hooks", "tests"]:
            dirpath = Path(dirname)
            if dirpath.exists():
                observer.schedule(handler, str(dirpath), recursive=True)
        
        # Watch requirements.txt
        observer.schedule(handler, ".", recursive=False)
        
        observer.start()
        
        # Track last status check time
        last_status_check = time.time()
        STATUS_CHECK_INTERVAL = 30  # Check session status every 30 seconds
        
        try:
            while True:
                time.sleep(0.5)

                if keys_active:
                    try:
                        intent = key_queue.get_nowait()
                    except queue.Empty:
                        intent = None
                    if intent == "quit":
                        click.echo()
                        click.echo(f"  [{time.strftime('%H:%M:%S')}] → Quit requested")
                        break
                    elif intent == "refresh":
                        if not pending_upload:
                            next_upload_label = "Manual refresh — uploading"
                        pending_upload = True
                        last_upload_time = time.time() - DEBOUNCE_SECONDS
                    elif intent == "test":
                        _run_tests_in_dev_session(client, project_id, env_id)
                        # Drop any keypresses queued while tests were running.
                        while True:
                            try:
                                key_queue.get_nowait()
                            except queue.Empty:
                                break
                        last_status_check = time.time()
                        click.echo(f"  [{time.strftime('%H:%M:%S')}] → Back to watching. Keys: [r] refresh  [t] run tests  [q] quit")

                # Periodically check if session is still active
                if (time.time() - last_status_check) >= STATUS_CHECK_INTERVAL:
                    last_status_check = time.time()
                    try:
                        status_resp = client.get(f"/test-sessions/{session_id}/status")
                        if status_resp.status_code == 404:
                            click.echo()
                            click.secho(f"  [{time.strftime('%H:%M:%S')}] ! Session ended (deleted by server)", fg="yellow")
                            click.echo("    Session was cleaned up due to inactivity or manual deletion.")
                            server_terminated = True
                            break
                        elif status_resp.status_code == 200:
                            status_data = status_resp.json()
                            if status_data.get("status") != "active":
                                click.echo()
                                status = status_data.get('status')
                                click.secho(f"  [{time.strftime('%H:%M:%S')}] ! Session ended (status: {status})", fg="yellow")
                                click.echo("    Session was stopped due to inactivity timeout.")
                                server_terminated = True
                                break
                    except httpx.RequestError:
                        # Network error - don't break, just skip this check
                        pass

                # Check if we need to upload (with debounce)
                if pending_upload and (time.time() - last_upload_time) >= DEBOUNCE_SECONDS:
                    pending_upload = False
                    label = next_upload_label or "Files changed, uploading..."
                    next_upload_label = None
                    click.echo(f"  [{time.strftime('%H:%M:%S')}] → {label}")

                    new_hash, size, error = upload_files()
                    if error == "SESSION_ENDED":
                        click.echo()
                        click.secho(f"  [{time.strftime('%H:%M:%S')}] ! Session ended", fg="yellow")
                        click.echo("    Session was stopped due to inactivity timeout.")
                        server_terminated = True
                        break
                    elif error:
                        click.secho(f"  [{time.strftime('%H:%M:%S')}]     ! {error}", fg="yellow", err=True)
                        click.echo(f"  [{time.strftime('%H:%M:%S')}]     Fix the issue and save to retry...")
                    elif new_hash and new_hash != current_hash:
                        current_hash = new_hash
                        click.secho(f"  [{time.strftime('%H:%M:%S')}]     ✓ Uploaded {size} bytes", fg="green")
                    elif new_hash == current_hash:
                        click.echo(f"  [{time.strftime('%H:%M:%S')}]     No content changes detected")
                    
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join()
            key_stop.set()
            if key_thread is not None:
                key_thread.join(timeout=1.0)
            if key_old_termios is not None and _termios_mod is not None:
                _termios_mod.tcsetattr(
                    sys.stdin.fileno(),
                    _termios_mod.TCSADRAIN,
                    key_old_termios,
                )
    
    except Exception as e:
        _err(str(e))
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


@main.command()
@click.option("--env", help="Environment ID to run tests against (defaults to env's test_environment_id, falling back to itself).")
@click.option("--filter", "filter_name", help="Run only tests whose name matches this string.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON for CI.")
@click.option("--api-url", envvar="CONNIC_API_URL", default=DEFAULT_API_URL, help="Connic API URL")
@click.option("--api-key", envvar="CONNIC_API_KEY", default=None, help="Connic API key")
@click.option("--project-id", envvar="CONNIC_PROJECT_ID", default=None, help="Connic project ID")
def test(env: str | None, filter_name: str | None, as_json: bool, api_url: str, api_key: str | None, project_id: str | None):
    """
    Run the test suite from ./tests against a Connic environment.

    Discovers `tests/*.yaml` files (one per agent, mirroring `middleware/`),
    invokes each agent N times in the chosen environment, and asserts on
    output and tool-call traces. Exits non-zero if any test fails.

    \b
    Examples:
        connic test                       # Run all tests against the default env
        connic test --env <env-id>        # Run against a specific env
        connic test --filter login        # Run only tests with "login" in the name
        connic test --json                # Machine-readable output for CI
    """
    import json

    import httpx

    # Load credentials from .connic if available.
    connic_file = Path(".connic")
    if connic_file.exists():
        try:
            cfg = json.loads(connic_file.read_text())
            api_key = api_key or cfg.get("api_key")
            project_id = project_id or cfg.get("project_id")
        except Exception:
            pass

    if not api_key or not project_id:
        if not as_json:
            _h1("Test")
        _fail_and_exit("API key and project ID required. Run `connic login`.")

    if not as_json:
        _h1("Test")

    try:
        tar_data, _, n_tests = _package_project_for_tests(quiet=as_json)
    except ValueError as e:
        _fail_and_exit(str(e))
    if n_tests == 0:
        _fail_and_exit("No tests/ directory found in project — nothing to run.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Long read timeout: image builds can take several minutes.
    test_timeout = httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=30.0)
    with httpx.Client(base_url=api_url, headers=headers, timeout=test_timeout) as client:
        # Resolve target env. If no --env given, use the default standard env's
        # test_environment_id, falling back to the env itself.
        if env is None:
            if not as_json:
                _step("Resolving target environment...")
            envs_resp = client.get(f"/projects/{project_id}/environments/")
            if envs_resp.status_code != 200:
                _fail_and_exit(f"Failed to list environments: {envs_resp.text}")
            envs = envs_resp.json()
            standard = [e for e in envs if e.get("env_type") != "test"]
            default_env = next((e for e in standard if e.get("is_default")), None) or (standard[0] if standard else None)
            if default_env is None:
                _fail_and_exit("No environments found. Create one in the dashboard first.")
            env = default_env.get("test_environment_id") or default_env["id"]
            target_label = default_env.get("name") if env == default_env["id"] else f"{default_env.get('name')}.test_environment_id"
            if not as_json:
                _ok(f"{target_label} ({env})")
        else:
            if not as_json:
                _step(f"Target environment: {env}")

        if not as_json:
            _step("Submitting to backend...")
        try:
            test_run_id = _kickoff_test_run(client, project_id, env, tar_data)
        except RuntimeError as e:
            _fail_and_exit(str(e))
        if not as_json:
            _ok(f"Test run id: {test_run_id}")

        try:
            result = _poll_test_run(
                client,
                f"/projects/{project_id}/test-runs/{test_run_id}",
                quiet=as_json,
            )
        except RuntimeError as e:
            _fail_and_exit(f"{e}; giving up.")

    cases = result.get("cases", [])
    if filter_name:
        cases = [c for c in cases if filter_name in c["test_name"]]

    if result["status"] == "error":
        _fail_and_exit(f"Test run errored: {result.get('error') or 'unknown error'}", code=2)

    if as_json:
        click.echo(json.dumps({"status": result["status"], "cases": cases}, indent=2))
    else:
        _render_test_cases(cases)
        passed_n = sum(1 for c in cases if c["passed"])
        if result.get("deployment_id"):
            _render_dashboard_link(project_id, result["deployment_id"], env)
        _done(f"{passed_n}/{len(cases)} cases passed.")

    sys.exit(0 if result["status"] == "passed" else 1)


@main.command()
@click.option("--token", envvar="CONNIC_TOKEN", help="Login token (project_id:api_key) from the dashboard")
@click.option("--api-key", envvar="CONNIC_API_KEY", help="Connic API key")
@click.option("--project-id", envvar="CONNIC_PROJECT_ID", help="Connic project ID")
@click.option("--base-url", envvar="CONNIC_BASE_URL", default=DEFAULT_BASE_URL, help="Connic dashboard URL")
def login(token: str | None, api_key: str | None, project_id: str | None, base_url: str):
    """
    Save Connic credentials for the current project.

    Creates a .connic file with your API key and project ID.
    Run without options for interactive mode - opens the dashboard
    to create an API key and gives you a single token to paste.

    \b
    Example:
        connic login
        connic login --token <project_id>:<api_key>
        connic login --api-key cnc_xxx --project-id <uuid>
    """
    import json
    import webbrowser

    _h1("Login")

    if token:
        project_id, api_key = _parse_login_token(token)

    if not api_key or not project_id:
        login_url = f"{base_url}/projects?to=/settings/cli?add=1"

        _step("Opening the Connic dashboard to create an API key...")
        _info("If the browser doesn't open, visit:")
        click.secho(f"    {login_url}", fg="cyan", underline=True)

        try:
            webbrowser.open(login_url)
        except Exception:
            pass

        _info("After creating a key, copy the login token and paste it below.")
        click.echo()
        raw_token = click.prompt(click.style("  Login token", fg="yellow"), hide_input=True)
        project_id, api_key = _parse_login_token(raw_token.strip())

    config = {
        "api_key": api_key,
        "project_id": project_id,
    }

    connic_file = Path(".connic")
    connic_file.write_text(json.dumps(config, indent=2))

    _step("Credentials saved to .connic")
    _info(f"API key:  {api_key[:12]}...")
    _info(f"Project:  {project_id}")
    _warn("Remember to add .connic to your .gitignore!")

    _done("Logged in.")


def _parse_login_token(token: str) -> tuple[str, str]:
    """Parse a login token in the format project_id:api_key."""
    if ":" not in token:
        _err("Invalid token format. Expected project_id:api_key")
        raise SystemExit(1)
    project_id, api_key = token.split(":", 1)
    if not project_id or not api_key:
        _err("Invalid token format. Expected project_id:api_key")
        raise SystemExit(1)
    return project_id, api_key


# =============================================================================
# Deploy Command - Upload and deploy to Connic cloud
# =============================================================================

@main.command()
@click.option("--env", help="Target environment ID (get from Project Settings → Environments)")
@click.option("--api-url", envvar="CONNIC_API_URL", default=DEFAULT_API_URL, help="Connic API URL")
@click.option("--api-key", envvar="CONNIC_API_KEY", default=None, help="Connic API key")
@click.option("--project-id", envvar="CONNIC_PROJECT_ID", default=None, help="Connic project ID")
@click.option("--skip-tests", is_flag=True, help="Skip the test phase even if tests/ exists.")
def deploy(env: str | None, api_url: str, api_key: str | None, project_id: str | None, skip_tests: bool):
    """
    Deploy local agents to Connic cloud.
    
    Packages your local agent files and uploads them for deployment.
    Only works for projects without a connected git repository.
    
    Uses credentials from 'connic login' or environment variables.
    
    \b
    Examples:
        connic deploy                           # Deploy to default environment
        connic deploy --env <environment-id>    # Deploy to specific environment
    
    \b
    Get your Environment ID from:
        Project Settings → Environments → Copy ID button
    
    \b
    Environment variables:
        CONNIC_API_URL      - API URL (default: https://api.connic.co/v1)
        CONNIC_API_KEY      - Your API key
        CONNIC_PROJECT_ID   - Your project ID
        CONNIC_BASE_URL     - Base URL (default: https://connic.co)
    """
    import base64
    import hashlib
    import io
    import json
    import tarfile

    import httpx
    
    # Load config from .connic file
    connic_file = Path(".connic")
    if connic_file.exists():
        try:
            config = json.loads(connic_file.read_text())
            api_key = api_key or config.get("api_key")
            project_id = project_id or config.get("project_id")
        except Exception:
            pass
    
    _h1("Deploy")

    # Validate required config
    if not api_key or not project_id:
        missing = "API key" if not api_key else "Project ID"
        _err(f"{missing} required.")
        _info("Run `connic login` to save your credentials interactively, or set:")
        _info("  CONNIC_API_KEY    — Project Settings → CLI → Create Key")
        _info("  CONNIC_PROJECT_ID — Project Settings → CLI")
        sys.exit(1)

    # Lint before deploying
    _step("Validating project files...")
    if not _run_lint(quiet=True):
        _fail_and_exit("Lint failed. Fix the errors above before deploying.")
    _ok("Lint passed")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Check project status and resolve environment.
    _step("Checking project and environment...")
    try:
        with httpx.Client(base_url=api_url, headers=headers, timeout=30.0) as client:
            resp = client.get(f"/projects/{project_id}")
            if resp.status_code == 401:
                _fail_and_exit("Invalid API key")
            elif resp.status_code == 404:
                _fail_and_exit("Project not found")
            elif resp.status_code != 200:
                _fail_and_exit(f"Failed to get project: {resp.text}")

            project = resp.json()

            if project.get("git_provider"):
                _err("This project has a git repository connected.")
                _info("CLI deploy only works for projects without git.")
                _info("Use git push to deploy, or disconnect git in project settings.")
                sys.exit(1)

            _ok(f"Project: {project['name']}")

            resp = client.get(f"/projects/{project_id}/environments/")
            if resp.status_code != 200:
                _fail_and_exit(f"Failed to get environments: {resp.text}")

            environments = resp.json()
            standard_envs = [e for e in environments if e.get("env_type") != "test"]
            if not standard_envs:
                _fail_and_exit("No environments found. Create one in the dashboard first.")

            target_env = None
            if env:
                target_env = next((e for e in standard_envs if e["id"] == env), None)
                if not target_env:
                    _err(f"Environment with ID '{env}' not found")
                    _info("Available environments:")
                    for e in standard_envs:
                        default_marker = " (default)" if e.get("is_default") else ""
                        _info(f"  {e['name']}: {e['id']}{default_marker}")
                    _info("Copy the ID from Project Settings → Environments")
                    sys.exit(1)
            else:
                target_env = next((e for e in standard_envs if e.get("is_default")), None) or standard_envs[0]

            _ok(f"Environment: {target_env['name']}")

    except httpx.ConnectError:
        _fail_and_exit("Failed to connect to Connic API")

    # Package files into tarball
    _step("Packaging upload...")
    try:
        is_valid, error, valid_files = _validate_project_files()
        if not is_valid:
            _fail_and_exit(f"File validation failed: {error}")

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            for f in valid_files:
                tar.add(f, arcname=str(f))

        tar_data = tar_buffer.getvalue()

        if len(tar_data) > MAX_UPLOAD_SIZE:
            _fail_and_exit(f"Package size ({len(tar_data):,} bytes) exceeds 1MB limit")

        files_b64 = base64.b64encode(tar_data).decode('utf-8')
        files_hash = hashlib.sha256(tar_data).hexdigest()[:12]

        _ok(f"{len(valid_files)} files, {len(tar_data):,} bytes")

    except Exception as e:
        _fail_and_exit(f"Failed to package files: {e}")

    # Upload and create deployment
    _step("Submitting to backend...")
    try:
        with httpx.Client(base_url=api_url, headers=headers, timeout=120.0) as client:
            resp = client.post(
                f"/projects/{project_id}/deploy/upload",
                params={"environment_id": target_env["id"]},
                json={
                    "files_data": files_b64,
                    "files_hash": files_hash,
                    "skip_tests": skip_tests,
                },
            )

            if resp.status_code == 400:
                error = resp.json().get("detail", resp.text)
                _fail_and_exit(error)
            elif resp.status_code != 200:
                _fail_and_exit(f"Failed to create deployment: {resp.text}")

            deployment = resp.json()
            deployment_id = deployment["id"]
            queued = resp.headers.get("x-deployment-queued", "").lower() == "true"

            _ok(f"Deployment id: {deployment_id}")
            if queued:
                _info("Another deployment is currently building; yours will start when a slot is available.")
            if skip_tests:
                _info("--skip-tests was set; the test phase will be skipped server-side.")

            _step("Track your deployment:")
            _info(f"{DEFAULT_BASE_URL}/projects/{project_id}/deployments/{deployment_id}")

            _done("Deployment created.")

    except Exception as e:
        _fail_and_exit(f"Failed to upload: {e}")


if __name__ == "__main__":
    main()
