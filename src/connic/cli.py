import os
import sys
from pathlib import Path

import click

from .loader import ProjectLoader

DEFAULT_API_URL = os.environ.get("CONNIC_API_URL", "https://api.connic.co/v1")
DEFAULT_BASE_URL = os.environ.get("CONNIC_BASE_URL", "https://connic.co")

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
@click.version_option(version="0.1.0", prog_name="connic")
def main():
    """Connic Composer SDK - Build agents with code."""
    pass


@main.command()
@click.argument("name", required=False, default=".")
def init(name: str):
    """Initialize a new Connic project.
    
    Creates the project structure with sample agent and tool files.
    
    Examples:
        connic init              # Initialize in current directory
        connic init my-project   # Create new directory
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
    
    # 1. Sample Tool - Calculator
    tools_file = base_path / "tools" / "calculator.py"
    if not tools_file.exists():
        tools_file.write_text('''"""Calculator tools for mathematical operations."""


def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: The first number
        b: The second number
    
    Returns:
        The sum of a and b
    """
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    
    Args:
        a: The first number
        b: The second number
    
    Returns:
        The product of a and b
    """
    return a * b


def calculate_tax(amount: float, rate: float = 0.19) -> float:
    """Calculate tax for a given amount.
    
    Args:
        amount: The base amount to calculate tax on
        rate: The tax rate as a decimal (default: 0.19 for 19%)
    
    Returns:
        The calculated tax amount
    """
    return amount * rate
''')
    
    # 2. Sample LLM Agent - Assistant
    agent_file = base_path / "agents" / "assistant.yaml"
    if not agent_file.exists():
        agent_file.write_text('''version: "1.0"

# LLM agents use AI models to process requests
# type: llm is the default, so it can be omitted

name: assistant
type: llm
model: gemini/gemini-2.5-flash
description: "A helpful general-purpose assistant with calculator capabilities"
system_prompt: |
  You are a helpful assistant with access to calculator tools.
  
  When users ask mathematical questions, use the available tools:
  - Use 'add' for addition
  - Use 'multiply' for multiplication  
  - Use 'calculate_tax' for tax calculations
  
  Always show your work and explain the calculations.

tools:
  - calculator.add
  - calculator.multiply
  - calculator.calculate_tax
''')

    # 3. Sample Agent - Invoice Processor (more complex LLM example)
    invoice_agent = base_path / "agents" / "invoice-processor.yaml"
    if not invoice_agent.exists():
        invoice_agent.write_text('''version: "1.0"

name: invoice-processor
type: llm
model: gemini/gemini-2.5-pro
description: "Extracts data from invoices and validates totals"
system_prompt: |
  You are an expert accountant specializing in invoice processing.
  
  Your responsibilities:
  1. Extract all relevant fields from invoices (vendor, date, line items, totals)
  2. Use the calculator tools to verify mathematical accuracy
  3. Flag any discrepancies between line items and totals
  4. Format extracted data in a structured JSON format
  
  Always double-check calculations before confirming totals are correct.

max_concurrent_runs: 1
temperature: 0.3
retry_options:
  attempts: 5
  max_delay: 60

# Timeout in seconds (min: 5). Capped by your subscription's limit.
timeout: 30

tools:
  - calculator.add
  - calculator.multiply
  - calculator.calculate_tax

# Output schema for structured JSON response (see schemas/invoice.json)
output_schema: invoice
''')

    # 3b. Sample Output Schema - Invoice
    schema_file = base_path / "schemas" / "invoice.json"
    if not schema_file.exists():
        schema_file.write_text('''{
  "type": "object",
  "description": "Extracted invoice data",
  "properties": {
    "vendor": {
      "type": "string",
      "description": "Vendor or company name"
    },
    "date": {
      "type": "string",
      "description": "Invoice date (YYYY-MM-DD)"
    },
    "total": {
      "type": "number",
      "description": "Total amount"
    },
    "currency": {
      "type": "string",
      "description": "Currency code (e.g., USD, EUR)"
    }
  },
  "required": ["vendor", "total"]
}
''')

    # 4. Sample TOOL Agent - Tax Calculator (direct tool execution)
    tax_agent = base_path / "agents" / "tax-calculator.yaml"
    if not tax_agent.exists():
        tax_agent.write_text('''version: "1.0"

# Tool agents execute a single tool directly without LLM
# Perfect for deterministic operations that don't need AI reasoning

name: tax-calculator
type: tool
description: "Calculates tax directly using the calculator tool"
tool_name: calculator.calculate_tax
''')

    # 5. Sample SEQUENTIAL Agent - Document Pipeline (chains agents)
    pipeline_agent = base_path / "agents" / "document-pipeline.yaml"
    if not pipeline_agent.exists():
        pipeline_agent.write_text('''version: "1.0"

# Sequential agents chain multiple agents together
# Each agent in the chain receives the previous agent's output as input

name: document-pipeline
type: sequential
description: "Processes documents through extraction and validation"

# Agents execute in order: assistant -> invoice-processor
# The output of 'assistant' becomes the input to 'invoice-processor'
agents:
  - assistant
  - invoice-processor
''')

    # 6. Sample Orchestrator Agent - Uses trigger_agent predefined tool
    orchestrator_agent = base_path / "agents" / "orchestrator.yaml"
    if not orchestrator_agent.exists():
        orchestrator_agent.write_text('''version: "1.0"

# Orchestrator agents dynamically trigger other agents using the trigger_agent tool
# Unlike sequential agents, orchestrators can decide at runtime which agents to call

name: orchestrator
type: llm
model: gemini/gemini-2.5-flash
description: "Orchestrates multiple specialized agents based on the task"
system_prompt: |
  You are an orchestrator that coordinates specialized agents to complete tasks.
  
  Available agents you can trigger:
  - assistant: General-purpose assistant with calculator tools
  - invoice-processor: Extracts and validates invoice data
  - tax-calculator: Calculates tax amounts directly
  
  Use the trigger_agent tool to delegate work. You can:
  - Wait for results: trigger_agent(agent_name="assistant", payload={"message": "..."})
  - Fire-and-forget: trigger_agent(agent_name="...", payload=..., wait_for_response=False)
  
  Analyze the user's request and decide which agent(s) to use.
  You can trigger multiple agents and combine their results.

tools:
  - trigger_agent  # Predefined tool - triggers other agents in the project
''')

    # 7. Sample Knowledge Agent - Uses RAG predefined tools
    knowledge_agent = base_path / "agents" / "knowledge-agent.yaml"
    if not knowledge_agent.exists():
        knowledge_agent.write_text('''version: "1.0"

# Knowledge Agent - Uses RAG (Retrieval-Augmented Generation) to query and store knowledge
# This agent can search, remember, and manage information in the project's knowledge base

name: knowledge-agent
type: llm
model: gemini/gemini-2.5-flash
description: "An agent with persistent memory that can store and retrieve knowledge"
system_prompt: |
  You are a knowledge-aware assistant with access to a persistent knowledge base.
  
  You can:
  - Search the knowledge base for relevant information using query_knowledge
  - Store new information for future retrieval using store_knowledge
  - Delete outdated information using delete_knowledge
  
  ## How the Knowledge Base Works
  
  The knowledge base uses semantic search (vector embeddings), so:
  - Queries find information by meaning, not just exact keywords
  - Long content is automatically chunked for better retrieval
  - Use namespaces to organize information (e.g., "user_preferences", "meeting_notes")
  - Entry IDs are unique within a namespace
  
  ## Best Practices
  
  1. **Before answering questions**, search the knowledge base first to see if relevant
     information exists. Use specific, descriptive queries for best results.
  
  2. **When storing knowledge**, use descriptive entry_ids (e.g., "company-refund-policy")
     and appropriate namespaces to make future retrieval easier.
  
  3. **Be proactive** - if a user shares important information, offer to save it
     to the knowledge base for future reference.
  
  4. **Cite your sources** - when using information from the knowledge base, mention
     the entry_id and any relevant context.

temperature: 0.7

tools:
  - query_knowledge   # Search the knowledge base using semantic search
  - store_knowledge   # Store new information for future retrieval
  - delete_knowledge  # Remove outdated entries from the knowledge base
''')

    # 8. Sample MCP Agent - Uses Context7 public MCP server
    mcp_agent = base_path / "agents" / "mcp-docs.yaml"
    if not mcp_agent.exists():
        mcp_agent.write_text('''version: "1.0"

# MCP Agent Example - Connects to Context7's public MCP server
# This demonstrates how to use external MCP servers for additional tools

name: mcp-docs
type: llm
model: gemini/gemini-2.5-flash
description: "An agent that can fetch library documentation using Context7 MCP"
system_prompt: |
  You are a helpful coding assistant with access to up-to-date library documentation.
  
  You have access to MCP tools that can:
  - Resolve library names to their documentation IDs
  - Fetch current documentation for popular libraries and frameworks
  
  When a user asks about a library or framework:
  1. Use the available tools to fetch the latest documentation
  2. Provide accurate, up-to-date information
  3. Include code examples when relevant
  
  Always base your answers on the documentation you retrieve.

temperature: 0.7
max_concurrent_runs: 1

# MCP Server Configuration - Context7 public documentation server (no auth required)
mcp_servers:
  - name: context7
    url: https://mcp.context7.com/mcp
''')

    # 9. Sample Middleware - Assistant
    middleware_file = base_path / "middleware" / "assistant.py"
    if not middleware_file.exists():
        middleware_file.write_text('''"""Middleware for the assistant agent.

Middleware functions run before and after agent execution.
The file name must match the agent name (assistant.py for assistant agent).

This example:
- Before: Adds current date/time context so the agent knows today's date
- After: Wraps the response in a structured JSON format with metadata
"""
import json
from datetime import datetime
from typing import Any, Dict


async def before(content: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Called before the agent runs. Adds current timestamp context.
    """
    # Add current date/time context so the agent knows "today's date"
    now = datetime.now()
    time_context = f"[Current time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}]"
    
    # Insert context at the beginning of the message
    content["parts"].insert(0, {"text": time_context})
    
    return content


async def after(response: str, context: Dict[str, Any]) -> str:
    """
    Called after the agent completes. Wraps response in structured JSON.
    """
    # Wrap the response in a structured format with metadata
    structured = {
        "response": response,
        "metadata": {
            "run_id": context.get("run_id"),
            "agent": context.get("agent_name"),
            "model": context.get("model"),
            "duration_ms": context.get("duration_ms"),
            "tokens": context.get("token_usage", {}),
        }
    }
    
    return json.dumps(structured, indent=2)
''')

    # 10. .gitignore
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

    # 11. requirements.txt
    requirements = base_path / "requirements.txt"
    if not requirements.exists():
        requirements.write_text('''# Add your tool dependencies below
# httpx>=0.25.0  # For async HTTP requests
# pandas>=2.0.0  # For data processing
''')

    # 12. README
    readme = base_path / "README.md"
    if not readme.exists():
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
   connic dev
   ```

3. Connect your repository to Connic and push to deploy.

## Middleware

Middleware files are auto-discovered by agent name. Create a file in `middleware/`
with the same name as your agent (e.g., `middleware/assistant.py` for the `assistant` agent).

Middleware can define `before` and `after` functions:
- `before(payload, context)` - Runs before the agent, can modify the payload
- `after(response, context)` - Runs after the agent, can modify the response

## Documentation

See the [Connic Composer docs]({base_url}/docs/v1/composer) for:
- [Agent Configuration]({base_url}/docs/v1/composer/agent-configuration)
- [Writing Tools]({base_url}/docs/v1/composer/write-tools)
- [Middleware]({base_url}/docs/v1/composer/middleware)
'''.format(base_url=DEFAULT_BASE_URL))

    click.echo(f"\n✓ Initialized Connic project in {base_path.resolve()}\n")
    click.echo("Created files:")
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
    click.echo("  1. Run 'connic dev' to validate your project")
    click.echo("  2. Edit the agent configs and tools as needed")
    click.echo("  3. Push to your connected repository to deploy")


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def dev(verbose: bool):
    """Validate and preview agents locally.
    
    Loads all agents and tools, validates configurations,
    and displays a summary of the project.
    """
    click.echo("Connic Composer SDK - Development Mode\n")
    
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
            for tool_ref in config.tools:
                # Extract function name from tool reference
                parts = tool_ref.split(".")
                func_name = parts[-1]
                if func_name not in loaded_tool_names:
                    click.echo(f"  │  ⚠ Missing tool: {tool_ref}")
        
        click.echo("  └─")
        click.echo()
    
    click.echo("✓ Project validation complete")
    click.echo("\nTo deploy, push to your connected repository.")


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
        env_name = session_data["environment_name"]
        
        click.echo(f"  Session ID: {session_id}")
        click.echo(f"  Environment: {env_name}")
        
        # Poll for container to be ready
        click.echo("\n  Waiting for container to start", nl=False)
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
