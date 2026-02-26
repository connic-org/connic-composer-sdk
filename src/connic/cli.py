import os
import sys
import tempfile
import zipfile
from pathlib import Path
import click
import httpx

from .loader import ProjectLoader

DEFAULT_API_URL = os.environ.get("CONNIC_API_URL", "https://api.connic.co/v1")
DEFAULT_BASE_URL = os.environ.get("CONNIC_BASE_URL", "https://connic.co")
TEMPLATES_REPO = "connic-org/connic-awesome-agents"
TEMPLATES_ZIP_URL = f"https://github.com/{TEMPLATES_REPO}/archive/refs/heads/main.zip"

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

# Maximum total upload size: 1MB
MAX_UPLOAD_SIZE = 1024 * 1024  # 1,048,576 bytes


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
    
    dirs_to_check = ["agents", "tools", "middleware", "schemas"]
    
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
            
            # Check extension
            ext = filepath.suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                return False, f"{filepath}: File type '{ext}' not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", []
            
            try:
                content = filepath.read_bytes()
            except IOError as e:
                return False, f"Could not read {filepath}: {e}", []
            
            total_size += len(content)
            if total_size > MAX_UPLOAD_SIZE:
                return False, f"Total file size exceeds 1MB limit ({total_size:,} bytes)", []
            
            valid_files.append(filepath)
    
    # Check requirements.txt
    req_file = Path("requirements.txt")
    if req_file.exists():
        try:
            content = req_file.read_bytes()
            total_size += len(content)
            if total_size > MAX_UPLOAD_SIZE:
                return False, f"Total file size exceeds 1MB limit ({total_size:,} bytes)", []
            valid_files.append(req_file)
        except IOError as e:
            return False, f"Could not read requirements.txt: {e}", []
    
    return True, "", valid_files


@click.group()
@click.version_option(version="0.1.3", prog_name="connic")
def main():
    """Connic Composer SDK - Build agents with code."""
    pass


def _write_essential_files(base_path: Path, include_examples: bool = False, quiet: bool = False):
    """Write files that are always created regardless of example inclusion."""

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
        if include_examples:
            readme.write_text('''# Connic Agent Project

This project contains AI agents built with the Connic Composer SDK.

## Structure

```
├── agents/                    # Agent YAML configurations
│   ├── assistant.yaml         # LLM agent with tools
│   ├── invoice-processor.yaml # LLM agent with retry options
│   ├── tax-calculator.yaml    # Tool agent (direct tool execution)
│   ├── document-pipeline.yaml # Sequential agent (chains agents)
│   ├── orchestrator.yaml      # Orchestrator agent (triggers other agents)
│   ├── knowledge-agent.yaml   # Knowledge agent (RAG with query/store/delete)
│   └── mcp-docs.yaml          # MCP agent (external tools via MCP)
├── tools/                     # Python tool modules
│   └── calculator.py
├── middleware/                # Optional middleware for agents
│   └── assistant.py           # Runs before/after assistant agent
└── requirements.txt           # Python dependencies
```

## Agent Types

Connic supports three types of agents:

### LLM Agents (type: llm)
Standard AI agents that use a language model to process requests.
```yaml
type: llm
model: gemini/gemini-2.5-flash  # Provider prefix required
system_prompt: "You are a helpful assistant..."
tools:
  - calculator.add
```

### Sequential Agents (type: sequential)
Chain multiple agents together - each agent's output becomes the next agent's input.
```yaml
type: sequential
agents:
  - assistant
  - invoice-processor
```

### Tool Agents (type: tool)
Execute a tool directly without LLM reasoning. Perfect for deterministic operations.
```yaml
type: tool
tool_name: calculator.calculate_tax
```

### Orchestrator Pattern (using trigger_agent)
LLM agents can dynamically trigger other agents using the `trigger_agent` predefined tool.
```yaml
type: llm
model: gemini/gemini-2.5-flash
system_prompt: "You can trigger other agents..."
tools:
  - trigger_agent  # Predefined tool
```

### Knowledge Agents (using RAG tools)
Agents can access a persistent knowledge base using semantic search.
```yaml
type: llm
model: gemini/gemini-2.5-flash
system_prompt: "You can search and store knowledge..."
tools:
  - query_knowledge   # Semantic search
  - store_knowledge   # Add to knowledge base
  - delete_knowledge  # Remove from knowledge base
```

### MCP Agents (using external MCP servers)
Agents can connect to external MCP (Model Context Protocol) servers for additional tools.
```yaml
type: llm
model: gemini/gemini-2.5-flash
system_prompt: "You can fetch documentation..."
mcp_servers:
  - name: context7
    url: https://mcp.context7.com/mcp
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Validate your project:
   ```bash
   connic lint
   ```

3. Connect your repository to Connic and push to deploy.

## Middleware

Middleware files are auto-discovered by agent name. Create a file in `middleware/`
with the same name as your agent (e.g., `middleware/assistant.py` for the `assistant` agent).

Middleware can define `before` and `after` functions:
- `before(payload, context)` - Runs before the agent, can modify the payload
- `after(response, context)` - Runs after the agent, can modify the response

## Documentation

See the [Connic Composer docs]({base_url}/docs/v1/composer/overview) for:
- [Agent Configuration]({base_url}/docs/v1/composer/agent-configuration)
- [Writing Tools]({base_url}/docs/v1/composer/write-tools)
- [Middleware]({base_url}/docs/v1/composer/middleware)
'''.format(base_url=DEFAULT_BASE_URL))
        else:
            readme.write_text('''# Connic Agent Project

This project contains AI agents built with the Connic Composer SDK.

## Structure

```
├── agents/       # Agent YAML configurations
├── tools/        # Python tool modules
├── middleware/    # Optional middleware for agents
├── schemas/      # Output schemas for structured responses
└── requirements.txt
```

## Getting Started

1. Create your first agent in `agents/`:
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

4. Connect your repository to Connic and push to deploy.

## Documentation

See the [Connic Composer docs]({base_url}/docs/v1/composer/overview) for:
- [Agent Configuration]({base_url}/docs/v1/composer/agent-configuration)
- [Writing Tools]({base_url}/docs/v1/composer/write-tools)
- [Middleware]({base_url}/docs/v1/composer/middleware)
'''.format(base_url=DEFAULT_BASE_URL))

    # Output (skip when quiet=True, e.g. when init used templates)
    if not quiet:
        click.echo(f"\n> Initialized Connic project in {base_path.resolve()}\n")
        click.echo("Created files:")
        if include_examples:
            click.echo("  agents/assistant.yaml          (LLM agent)")
            click.echo("  agents/invoice-processor.yaml  (LLM agent with retry)")
            click.echo("  agents/tax-calculator.yaml     (Tool agent)")
            click.echo("  agents/document-pipeline.yaml  (Sequential agent)")
            click.echo("  agents/orchestrator.yaml       (Orchestrator with trigger_agent)")
            click.echo("  agents/knowledge-agent.yaml    (Knowledge agent with RAG)")
            click.echo("  agents/mcp-docs.yaml           (MCP agent with Context7)")
            click.echo("  tools/calculator.py")
            click.echo("  middleware/assistant.py")
        click.echo("  .gitignore")
        click.echo("  requirements.txt")
        click.echo("  README.md")
        click.echo("\nNext steps:")
        if include_examples:
            click.echo("  1. Run 'connic lint' to validate your project")
            click.echo("  2. Edit the agent configs and tools as needed")
            click.echo("  3. Push to your connected repository to deploy")
        else:
            click.echo("  1. Create your first agent in agents/")
            click.echo("  2. Run 'connic lint' to validate your project")
            click.echo("  3. Push to your connected repository to deploy")


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
) -> str | None:
    """Copy template folder contents into project, merge requirements.
    
    Returns the template README content if it exists, None otherwise.
    """
    for subdir in ["agents", "tools", "middleware", "schemas"]:
        src_dir = template_src / subdir
        dst_dir = base_path / subdir
        if src_dir.exists():
            dst_dir.mkdir(exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file() and not f.name.startswith("_"):
                    (dst_dir / f.name).write_bytes(f.read_bytes())
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
    base_path = Path(name)

    if name != ".":
        if base_path.exists():
            click.echo(f"Error: Directory '{name}' already exists.", err=True)
            sys.exit(1)
        base_path.mkdir(parents=True)
        click.echo(f"Created directory: {name}")

    # Create directories
    (base_path / "agents").mkdir(exist_ok=True)
    (base_path / "tools").mkdir(exist_ok=True)
    (base_path / "middleware").mkdir(exist_ok=True)
    (base_path / "schemas").mkdir(exist_ok=True)

    if templates:
        # Fetch and merge templates
        template_ids = [t.strip().lower() for t in templates.split(",") if t.strip()]
        if not template_ids:
            click.echo("Error: No valid template names provided.", err=True)
            sys.exit(1)

        extracted = _fetch_templates_from_github()
        if not extracted:
            local_path = _get_local_templates_path()
            if local_path:
                extracted = local_path
                click.echo("Using local connic-awesome-agents (GitHub unavailable)")
            else:
                click.echo("Error: Could not fetch templates. Try again or use a local connic-awesome-agents folder.", err=True)
                sys.exit(1)
        else:
            click.echo("Fetched templates from connic-awesome-agents")

        requirements_lines = []
        template_readmes: list[str] = []
        for tid in template_ids:
            template_dir = extracted / tid
            if not template_dir.is_dir():
                click.echo(f"Error: Template '{tid}' not found in connic-awesome-agents", err=True)
                sys.exit(1)
            readme_content = _merge_template_into_project(template_dir, base_path, requirements_lines)
            if readme_content:
                template_readmes.append(readme_content)
            click.echo(f"  Added template: {tid}")

        if requirements_lines:
            _write_merged_requirements(base_path, requirements_lines)
        _write_essential_files(base_path, include_examples=False, quiet=True)

        if template_readmes:
            _append_template_readmes(base_path, template_readmes)

        click.echo(f"\n> Initialized with templates: {', '.join(template_ids)}")
        click.echo(f"Added agents, tools, middleware, schemas from: {', '.join(template_ids)}")
        click.echo("Next steps:")
        click.echo("  1. Run 'connic lint' to validate your project")
        click.echo("  2. Run 'connic test' to test against Connic cloud")
        click.echo("  3. Run 'connic deploy' when ready")
        return

    # No templates: create clean project only (no example files)
    _write_essential_files(base_path, include_examples=False)


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def lint(verbose: bool):
    """Validate agent configurations and tools.

    Loads all agents and tools, validates configurations,
    and displays a summary of the project.
    """
    click.echo("Connic Composer SDK - Validation\n")
    
    try:
        loader = ProjectLoader(".")
    except Exception as e:
        click.echo(f"Error initializing project: {e}", err=True)
        sys.exit(1)
    
    # Discover tools
    click.echo("Discovering tools...")
    try:
        tools = loader.discover_tools()
        total_tools = sum(len(funcs) for funcs in tools.values())
        
        if tools:
            click.echo(f"  Found {total_tools} tools in {len(tools)} modules:")
            for module, functions in sorted(tools.items()):
                if verbose:
                    click.echo(f"    {module}:")
                    for func in functions:
                        click.echo(f"      - {func}")
                else:
                    click.echo(f"    {module}: {', '.join(functions)}")
        else:
            click.echo("  No tools found in tools/ directory")
    except FileNotFoundError:
        click.echo("  No tools/ directory found")
        tools = {}
    
    click.echo()
    
    # Discover middlewares
    click.echo("Discovering middlewares...")
    try:
        middlewares = loader.discover_middlewares()
        if middlewares:
            click.echo(f"  Found middlewares for {len(middlewares)} agents:")
            for agent_name, hooks in sorted(middlewares.items()):
                click.echo(f"    {agent_name}: {', '.join(hooks)}")
        else:
            click.echo("  No middlewares found in middleware/ directory")
    except FileNotFoundError:
        click.echo("  No middleware/ directory found")
        middlewares = {}
    
    click.echo()
    
    # Load agents
    click.echo("Loading agents...")
    try:
        agents = loader.load_agents()
    except FileNotFoundError as e:
        click.echo(f"  {e}", err=True)
        click.echo("\nRun 'connic init' to create a sample project.")
        sys.exit(1)
    
    if not agents:
        click.echo("  No agents found in agents/ directory")
        click.echo("\nRun 'connic init' to create a sample project.")
        sys.exit(1)
    
    click.echo(f"  Found {len(agents)} agents:\n")
    
    for agent in agents:
        config = agent.config
        agent_type = config.type.value if hasattr(config.type, 'value') else str(config.type)
        type_label = {"llm": "🧠 LLM", "sequential": "🔗 Sequential", "tool": "🔧 Tool"}.get(agent_type, agent_type)
        
        click.echo(f"  ┌─ {config.name} [{type_label}]")
        click.echo(f"  │  Description: {config.description}")
        
        # Type-specific info
        if agent_type == "llm":
            click.echo(f"  │  Model: {config.model}")
            if verbose:
                click.echo(f"  │  Temperature: {config.temperature}")
        elif agent_type == "sequential":
            click.echo(f"  │  Chain: {' → '.join(config.agents)}")
        elif agent_type == "tool":
            click.echo(f"  │  Tool: {config.tool_name}")
        
        if verbose:
            click.echo(f"  │  Max Concurrent Runs: {config.max_concurrent_runs}")
            if config.retry_options:
                click.echo(f"  │  Retry: {config.retry_options.attempts} attempts, max {config.retry_options.max_delay}s delay")
            if config.timeout:
                click.echo(f"  │  Timeout: {config.timeout}s")
        
        # Tools (for LLM agents)
        if agent_type == "llm":
            if agent.tools:
                tool_names = [t.name for t in agent.tools]
                click.echo(f"  │  Tools: {', '.join(tool_names)}")
            else:
                click.echo("  │  Tools: (none)")
            
            # Show MCP servers if configured
            if config.mcp_servers:
                for mcp_server in config.mcp_servers:
                    click.echo(f"  │  MCP Server: {mcp_server.name} ({mcp_server.url})")
            
            # Show missing tools as warnings
            loaded_tool_names = {t.name for t in agent.tools}
            for tool_entry in config.tools:
                if isinstance(tool_entry, dict):
                    tool_ref = list(tool_entry.keys())[0]
                else:
                    tool_ref = str(tool_entry)
                parts = tool_ref.split(".")
                func_name = parts[-1]
                if func_name not in loaded_tool_names:
                    click.echo(f"  │  ⚠ Missing tool: {tool_ref}")
        
        click.echo("  └─")
        click.echo()
    
    click.echo("✓ Project validation complete")
    click.echo("\nTo deploy, push to your connected repository.")


@main.command(hidden=True)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def dev(verbose: bool):
    """Alias for 'connic lint' (deprecated)."""
    ctx = click.get_current_context()
    return ctx.invoke(lint, verbose=verbose)


@main.command()
def tools():
    """List all available tools in the project."""
    try:
        loader = ProjectLoader(".")
        discovered = loader.discover_tools()
    except FileNotFoundError:
        click.echo("No tools/ directory found.", err=True)
        sys.exit(1)
    
    if not discovered:
        click.echo("No tools found in tools/ directory.")
        click.echo("Create Python files in tools/ with typed functions.")
        sys.exit(0)
    
    click.echo("Available tools:\n")
    
    for module, functions in sorted(discovered.items()):
        click.echo(f"  {module}.py:")
        for func_name in functions:
            # Load the tool to get description
            try:
                tool = loader._resolve_tool(f"{module}.{func_name}")
                # Get first line of description
                desc = tool.description.split('\n')[0][:60]
                if len(tool.description.split('\n')[0]) > 60:
                    desc += "..."
                click.echo(f"    - {func_name}: {desc}")
            except Exception:
                click.echo(f"    - {func_name}")
        click.echo()
    
    click.echo("Use in agent YAML as: <module>.<function>")


# =============================================================================
# Test Command - Cloud Dev Mode with Hot Reload
# =============================================================================

@main.command()
@click.argument("name", required=False, default=None)
@click.option("--api-url", envvar="CONNIC_API_URL", default=DEFAULT_API_URL, help="Connic API URL")
@click.option("--api-key", envvar="CONNIC_API_KEY", default=None, help="Connic API key")
@click.option("--project-id", envvar="CONNIC_PROJECT_ID", default=None, help="Connic project ID")
def test(name: str, api_url: str, api_key: str, project_id: str):
    """
    Start a test session with hot-reload against Connic cloud.
    
    Creates an isolated test environment and syncs your local files
    for rapid development. Changes are reflected in 2-5 seconds.
    
    \b
    Examples:
        connic test              # Ephemeral test env (auto-deleted on exit)
        connic test my-feature   # Named test env (persists after exit)
    
    Environment variables:
        CONNIC_API_URL      - API URL (default: https://api.connic.co/v1)
        CONNIC_API_KEY      - Your API key
        CONNIC_PROJECT_ID   - Your project ID
    """
    import hashlib
    import io
    import signal
    import tarfile
    import time

    import httpx
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    
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
    
    if not api_key:
        click.echo("Error: API key required. Set CONNIC_API_KEY or use --api-key", err=True)
        click.echo("\nCreate an API key in the dashboard: Project Settings → CLI → Create Key")
        click.echo("\nOr run: connic login")
        sys.exit(1)
    
    if not project_id:
        click.echo("Error: Project ID required. Set CONNIC_PROJECT_ID or use --project-id", err=True)
        click.echo("\nFind your Project ID in the dashboard: Project Settings → CLI")
        click.echo("\nOr run: connic login")
        sys.exit(1)
    
    # Validate local project
    click.echo("Connic Test Mode - Hot Reload Development\n")
    click.echo("Validating local project...")
    
    try:
        loader = ProjectLoader(".")
        agents = loader.load_agents()
        if not agents:
            click.echo("Error: No agents found. Run 'connic init' first.", err=True)
            sys.exit(1)
        click.echo(f"  Found {len(agents)} agents: {[a.config.name for a in agents]}")
    except Exception as e:
        click.echo(f"Error loading project: {e}", err=True)
        sys.exit(1)
    
    # Create HTTP client with auth
    # Use longer timeout for session creation (image building can take time)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    client = httpx.Client(base_url=api_url, headers=headers, timeout=120.0)
    
    click.echo(f"  API Key: {api_key[:12]}•••")
    
    session_id = None
    cleaned_up = False
    server_terminated = False  # True if session was already cleaned up by server
    
    def cleanup():
        """Clean up test session on exit."""
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        
        # Skip cleanup if server already terminated the session
        if session_id and not server_terminated:
            click.echo("\n\nCleaning up test session...")
            try:
                resp = client.delete(f"/test-sessions/{session_id}")
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("environment_deleted"):
                        click.echo("  Ephemeral environment deleted.")
                    else:
                        click.echo("  Session ended (named environment preserved).")
                elif resp.status_code != 404:
                    click.echo(f"  Warning: Cleanup returned {resp.status_code}")
            except Exception as e:
                click.echo(f"  Warning: Cleanup failed: {e}")
        client.close()
    
    # Register cleanup handler
    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create test session
        click.echo("\nCreating test session...")
        body = {}
        if name:
            body["name"] = name
            click.echo(f"  Using named environment: {name}")
        else:
            click.echo("  Creating ephemeral environment (will be deleted on exit)")
        
        resp = client.post(f"/projects/{project_id}/test-sessions", json=body)
        
        if resp.status_code == 409:
            # Already an active session for this environment
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text
            click.echo(f"\nError: {detail}", err=True)
            click.echo("\nTo stop an existing session, press Ctrl+C in the terminal where it's running.", err=True)
            sys.exit(1)
        elif resp.status_code != 200:
            click.echo(f"Error creating test session: {resp.text}", err=True)
            sys.exit(1)
        
        session_data = resp.json()
        session_id = session_data["id"]
        env_id = session_data["environment_id"]
        env_name = session_data["environment_name"]
        
        click.echo(f"  Session ID: {session_id}")
        click.echo(f"  Environment: {env_name}")
        
        # Poll for container to be ready
        click.echo("\n  Waiting for container to start (this can take a few seconds)", nl=False)
        max_wait = 300  # 5 minutes max
        poll_interval = 3
        waited = 0
        container_ready = False
        
        while waited < max_wait:
            try:
                status_resp = client.get(f"/test-sessions/{session_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    container_status = status_data.get("container_status", "starting")
                    
                    if container_status == "running":
                        click.echo(" ✓")
                        click.echo("  Container: running")
                        container_ready = True
                        break
                    elif container_status == "failed":
                        click.echo(" ✗")
                        click.echo("  Container failed to start", err=True)
                        cleanup()
                        sys.exit(1)
                    else:
                        click.echo(".", nl=False)
            except Exception:
                click.echo(".", nl=False)
            
            time.sleep(poll_interval)
            waited += poll_interval
        
        if not container_ready:
            click.echo(" timeout")
            click.echo("  Container did not start within 5 minutes", err=True)
            click.echo("  Check backend logs for details", err=True)
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
                raise ValueError(f"Package size ({len(content):,} bytes) exceeds 1MB limit")
            
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
            
            # Upload as multipart form - client already has auth headers
            # We need to use httpx directly without Content-Type header for multipart
            upload_resp = httpx.post(
                f"{api_url}/test-sessions/{session_id}/files",
                files={"file": ("files.tar.gz", content, "application/gzip")},
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
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
        click.echo("\nUploading initial files...")
        current_hash, size, error = upload_files()
        if error == "SESSION_ENDED":
            click.secho("  ✗ Session ended unexpectedly", fg="red", err=True)
            cleanup()
            sys.exit(1)
        elif error:
            click.secho(f"  ⚠ {error}", fg="yellow", err=True)
            click.echo("  Fix the issue and save to retry...\n")
            current_hash = None  # Will retry on file change
        elif current_hash:
            click.echo(f"  Uploaded {size} bytes (hash: {current_hash[:16]}...)")
        
        # Set up file watcher
        click.echo("\nWatching for file changes...")
        click.echo("  Press Ctrl+C to stop\n")
        
        # Display link to test environment
        dashboard_url = f"{DEFAULT_BASE_URL}/projects/{project_id}/agents?env={env_id}"
        click.echo("─" * 60)
        click.secho("  View and trigger your agents: ", fg="cyan", nl=False)
        click.echo(dashboard_url)
        click.echo("─" * 60)
        click.echo()
        
        last_upload_time = time.time()
        pending_upload = False
        DEBOUNCE_SECONDS = 1.0  # Wait for changes to settle
        
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
                
                # Check if file is in watched directories (agents, tools, middleware)
                # or is requirements.txt
                watched_dirs = ["agents", "tools", "middleware", "schemas"]
                is_watched = any(d in src_path.parts for d in watched_dirs)
                is_requirements = src_path.name == "requirements.txt"
                
                if not is_watched and not is_requirements:
                    return
                
                click.echo(f"[{time.strftime('%H:%M:%S')}] Detected change: {src_path.name}")
                pending_upload = True
                last_upload_time = time.time()
        
        observer = Observer()
        handler = FileChangeHandler()
        
        # Watch the project directories
        for dirname in ["agents", "tools", "middleware", "schemas"]:
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
                
                # Periodically check if session is still active
                if (time.time() - last_status_check) >= STATUS_CHECK_INTERVAL:
                    last_status_check = time.time()
                    try:
                        status_resp = client.get(f"/test-sessions/{session_id}/status")
                        if status_resp.status_code == 404:
                            click.echo()
                            click.secho(f"[{time.strftime('%H:%M:%S')}] Session ended (deleted by server)", fg="yellow")
                            click.echo("Session was cleaned up due to inactivity or manual deletion.")
                            server_terminated = True
                            break
                        elif status_resp.status_code == 200:
                            status_data = status_resp.json()
                            if status_data.get("status") != "active":
                                click.echo()
                                status = status_data.get('status')
                                click.secho(f"[{time.strftime('%H:%M:%S')}] Session ended (status: {status})", fg="yellow")
                                click.echo("Session was stopped due to inactivity timeout.")
                                server_terminated = True
                                break
                    except httpx.RequestError:
                        # Network error - don't break, just skip this check
                        pass
                
                # Check if we need to upload (with debounce)
                if pending_upload and (time.time() - last_upload_time) >= DEBOUNCE_SECONDS:
                    pending_upload = False
                    click.echo(f"[{time.strftime('%H:%M:%S')}] Files changed, uploading...")
                    
                    new_hash, size, error = upload_files()
                    if error == "SESSION_ENDED":
                        click.echo()
                        click.secho(f"[{time.strftime('%H:%M:%S')}] Session ended", fg="yellow")
                        click.echo("Session was stopped due to inactivity timeout.")
                        server_terminated = True
                        break
                    elif error:
                        click.secho(f"[{time.strftime('%H:%M:%S')}] ⚠ {error}", fg="yellow", err=True)
                        click.echo(f"[{time.strftime('%H:%M:%S')}] Fix the issue and save to retry...")
                    elif new_hash and new_hash != current_hash:
                        current_hash = new_hash
                        click.echo(f"[{time.strftime('%H:%M:%S')}] Uploaded {size} bytes")
                    elif new_hash == current_hash:
                        click.echo(f"[{time.strftime('%H:%M:%S')}] No content changes detected")
                    
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join()
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


@main.command()
@click.option("--api-key", envvar="CONNIC_API_KEY", help="Connic API key")
@click.option("--project-id", envvar="CONNIC_PROJECT_ID", help="Connic project ID")
def login(api_key: str | None, project_id: str | None):
    """
    Save Connic credentials for the current project.
    
    Creates a .connic file with your API key and project ID.
    Run without options for interactive mode.
    
    \b
    Example:
        connic login
        connic login --api-key cnc_xxx --project-id <uuid>
    """
    import json
    
    click.echo()
    click.secho("  Connic CLI Login", fg="cyan", bold=True)
    click.echo("  " + "─" * 30)
    click.echo()
    
    # Interactive prompts for missing values
    if not api_key:
        click.echo("  Create an API key in the Connic dashboard under")
        click.echo("  Project Settings → CLI → Create Key")
        click.echo()
        api_key = click.prompt(click.style("  API Key", fg="yellow"), hide_input=True)
    
    if not project_id:
        click.echo()
        click.echo("  The Project ID is shown in Project Settings → CLI")
        click.echo()
        project_id = click.prompt(click.style("  Project ID", fg="yellow"))
    
    config = {
        "api_key": api_key,
        "project_id": project_id,
    }
    
    connic_file = Path(".connic")
    connic_file.write_text(json.dumps(config, indent=2))
    
    click.echo()
    click.secho("  ✓ Credentials saved to .connic", fg="green", bold=True)
    click.echo()
    click.echo(f"    API Key:  {api_key[:12]}•••")
    click.echo(f"    Project:  {project_id}")
    click.echo()
    click.secho("  ⚠️  Remember to add .connic to your .gitignore!", fg="yellow")
    click.echo()


# =============================================================================
# Deploy Command - Upload and deploy to Connic cloud
# =============================================================================

@main.command()
@click.option("--env", help="Target environment ID (get from Project Settings → Environments)")
@click.option("--api-url", envvar="CONNIC_API_URL", default=DEFAULT_API_URL, help="Connic API URL")
@click.option("--api-key", envvar="CONNIC_API_KEY", default=None, help="Connic API key")
@click.option("--project-id", envvar="CONNIC_PROJECT_ID", default=None, help="Connic project ID")
def deploy(env: str | None, api_url: str, api_key: str | None, project_id: str | None):
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
    
    # Validate required config
    if not api_key:
        click.echo("Error: API key required. Set CONNIC_API_KEY or use --api-key", err=True)
        click.echo("\nCreate an API key in the dashboard: Project Settings → CLI → Create Key")
        click.echo("\nOr run: connic login")
        sys.exit(1)
    
    if not project_id:
        click.echo("Error: Project ID required. Set CONNIC_PROJECT_ID or use --project-id", err=True)
        click.echo("\nFind your Project ID in the dashboard: Project Settings → CLI")
        click.echo("\nOr run: connic login")
        sys.exit(1)
    
    click.echo()
    click.secho("  Connic Deploy", fg="cyan", bold=True)
    click.echo("  " + "─" * 30)
    click.echo()
    
    # Validate local project
    click.echo("  📦 Validating local project...")
    try:
        loader = ProjectLoader(".")
        agents = loader.load_agents()
        if not agents:
            click.echo("  ✗ No agents found. Run 'connic init' first.", err=True)
            sys.exit(1)
        click.echo(f"     Found {len(agents)} agent(s): {[a.config.name for a in agents]}")
    except Exception as e:
        click.echo(f"  ✗ Error loading project: {e}", err=True)
        sys.exit(1)
    
    # Create HTTP client
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Check project status and get environments
    click.echo("  🔍 Checking project...")
    try:
        with httpx.Client(base_url=api_url, headers=headers, timeout=30.0) as client:
            # Get project info
            resp = client.get(f"/projects/{project_id}")
            if resp.status_code == 401:
                click.echo("  ✗ Invalid API key", err=True)
                sys.exit(1)
            elif resp.status_code == 404:
                click.echo("  ✗ Project not found", err=True)
                sys.exit(1)
            elif resp.status_code != 200:
                click.echo(f"  ✗ Failed to get project: {resp.text}", err=True)
                sys.exit(1)
            
            project = resp.json()
            
            # Check if project has git connected
            if project.get("git_provider"):
                click.echo()
                click.secho("  ✗ This project has a git repository connected.", fg="red", bold=True)
                click.echo()
                click.echo("     CLI deploy only works for projects without git.")
                click.echo("     Use git push to deploy, or disconnect git in project settings.")
                click.echo()
                sys.exit(1)
            
            click.echo(f"     Project: {project['name']}")
            
            # Get environments
            resp = client.get(f"/projects/{project_id}/environments/")
            if resp.status_code != 200:
                click.echo(f"  ✗ Failed to get environments: {resp.text}", err=True)
                sys.exit(1)
            
            environments = resp.json()
            standard_envs = [e for e in environments if e.get("env_type") != "test"]
            
            if not standard_envs:
                click.echo("  ✗ No environments found. Create one in the dashboard first.", err=True)
                sys.exit(1)
            
            # Select target environment
            target_env = None
            if env:
                # Find by ID
                target_env = next((e for e in standard_envs if e["id"] == env), None)
                if not target_env:
                    click.echo(f"  ✗ Environment with ID '{env}' not found", err=True)
                    click.echo()
                    click.echo("     Available environments:")
                    for e in standard_envs:
                        default_marker = " (default)" if e.get("is_default") else ""
                        click.echo(f"       {e['name']}: {e['id']}{default_marker}")
                    click.echo()
                    click.echo("     Copy the ID from Project Settings → Environments")
                    sys.exit(1)
            else:
                # Use default environment
                target_env = next((e for e in standard_envs if e.get("is_default")), None)
                if not target_env:
                    target_env = standard_envs[0]
            
            click.echo(f"     Environment: {target_env['name']}")
            
    except httpx.ConnectError:
        click.echo("  ✗ Failed to connect to Connic API", err=True)
        sys.exit(1)
    
    # Package files into tarball
    click.echo("  📤 Packaging files...")
    
    try:
        # Validate files first
        is_valid, error, valid_files = _validate_project_files()
        if not is_valid:
            click.echo(f"  ✗ File validation failed: {error}", err=True)
            sys.exit(1)
        
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            # Add only validated files
            for f in valid_files:
                tar.add(f, arcname=str(f))
        
        tar_data = tar_buffer.getvalue()
        
        # Check final package size
        if len(tar_data) > MAX_UPLOAD_SIZE:
            click.echo(f"  ✗ Package size ({len(tar_data):,} bytes) exceeds 1MB limit", err=True)
            sys.exit(1)
        
        files_b64 = base64.b64encode(tar_data).decode('utf-8')
        files_hash = hashlib.sha256(tar_data).hexdigest()[:12]
        
        click.echo(f"     Package size: {len(tar_data) / 1024:.1f} KB")
        
    except Exception as e:
        click.echo(f"  ✗ Failed to package files: {e}", err=True)
        sys.exit(1)
    
    # Upload and create deployment
    click.echo("  🚀 Deploying...")
    
    try:
        with httpx.Client(base_url=api_url, headers=headers, timeout=120.0) as client:
            resp = client.post(
                f"/projects/{project_id}/deploy/upload",
                params={"environment_id": target_env["id"]},
                json={
                    "files_data": files_b64,
                    "files_hash": files_hash,
                },
            )
            
            if resp.status_code == 400:
                error = resp.json().get("detail", resp.text)
                click.echo(f"  ✗ {error}", err=True)
                sys.exit(1)
            elif resp.status_code != 200:
                click.echo(f"  ✗ Failed to create deployment: {resp.text}", err=True)
                sys.exit(1)
            
            deployment = resp.json()
            deployment_id = deployment["id"]
            
            click.echo()
            click.secho("  ✓ Deployment triggered!", fg="green", bold=True)
            click.echo()
            click.echo(f"     Deployment ID: {deployment_id[:8]}...")
            click.echo()
            click.echo("     Check deployment status in your project dashboard:")
            click.echo(f"     {DEFAULT_BASE_URL}/projects/{project_id}/deployments")
            click.echo()
            
    except Exception as e:
        click.echo(f"  ✗ Failed to upload: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
