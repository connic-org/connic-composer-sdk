# Connic Composer SDK

<div align="center">

**Build production-ready AI agents with code.**

Define agents in YAML. Extend them with Python. Deploy anywhere.

[![PyPI version](https://img.shields.io/pypi/v/connic.svg)](https://pypi.org/project/connic/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://connic.co/docs/v1/composer) • [Quick Start](#quick-start) • [Examples](#examples) • [CLI Reference](#cli-reference)

</div>

---

## What is Connic Composer?

Connic Composer is a Python SDK for building **enterprise-grade AI agents** on the Connic platform:

- 🚀 **Deploy with git push** - Automatic builds and deployments
- 🔧 **Extend with Python** - Write custom tools as simple Python functions
- 🔗 **Connect to anything** - Webhooks, Kafka, WebSockets, MCP servers
- 🧠 **Multi-agent workflows** - Chain agents, trigger dynamically, or run in parallel
- 📚 **Built-in RAG** - Persistent knowledge base with semantic search
- ⚡ **Hot reload** - Test locally with instant feedback (`connic test`)

## Installation

```bash
pip install connic
```

## Quick Start

### 1. Initialize a New Project

```bash
connic init my-agents
cd my-agents
```

This creates a project structure:

```
my-agents/
├── agents/          # Agent YAML configurations
├── tools/           # Custom Python tools
├── middleware/      # Optional request/response hooks
├── schemas/         # JSON schemas for structured output
└── requirements.txt
```

### 2. Define Your First Agent

Create `agents/assistant.yaml`:

```yaml
version: "1.0"

name: assistant
model: gemini/gemini-2.5-flash
description: "A helpful assistant with calculator tools"

system_prompt: |
  You are a helpful assistant with access to calculator tools.
  Use them when users ask mathematical questions.

tools:
  - calculator.add
  - calculator.multiply
```

### 3. Create Custom Tools

Write Python functions in `tools/calculator.py`:

```python
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
```

> **💡 Tool Requirements:**
> - Type hints on all parameters (required for LLM function calling)
> - Docstring with Args section (used for tool descriptions)
> - Return type hint

### 4. Test Locally

```bash
# Validate your project
connic dev

# Test with hot reload (requires Connic account)
connic login
connic test
```

### 5. Deploy

**Option 1: Via Git (Recommended)**
```bash
# Connect your repository in the Connic dashboard
git push  # Automatically builds and deploys
```

**Option 2: Direct Upload**
```bash
connic deploy
```

---

## Agent Types

Connic supports **three types of agents** for different use cases:

### 🧠 LLM Agents (Default)

Standard AI agents powered by language models.

```yaml
version: "1.0"
type: llm  # Optional, llm is the default

name: customer-support
model: gemini/gemini-2.5-pro
description: "Customer support agent"

system_prompt: |
  You are a customer support agent. Be helpful and professional.

tools:
  - database.lookup_order
  - email.send_confirmation

max_concurrent_runs: 10
temperature: 0.7
timeout: 30  # seconds

retry_options:
  attempts: 5
  max_delay: 60
```

### 🔗 Sequential Agents

Chain multiple agents together - each agent's output becomes the next agent's input.

```yaml
version: "1.0"
type: sequential

name: document-pipeline
description: "Extract data, validate, and store"

agents:
  - extractor      # Runs first
  - validator      # Receives extractor's output
  - storage        # Receives validator's output
```

**Use cases:** Multi-step workflows, data pipelines, progressive refinement

### 🔧 Tool Agents

Execute a tool directly without LLM reasoning. Perfect for deterministic operations.

```yaml
version: "1.0"
type: tool

name: tax-calculator
description: "Calculate tax amount"
tool_name: calculator.calculate_tax
```

**Use cases:** Fast calculations, API calls, data transformations

---

## Advanced Features

### 🎯 Orchestrator Pattern

Dynamically trigger other agents using the `trigger_agent` predefined tool:

```yaml
version: "1.0"

name: orchestrator
model: gemini/gemini-2.5-flash
description: "Routes tasks to specialized agents"

system_prompt: |
  You coordinate specialized agents:
  - invoice-processor: Extract data from invoices
  - email-agent: Send emails
  - database-agent: Query/update database
  
  Analyze the user's request and trigger the appropriate agents.
  You can call multiple agents and combine their results.

tools:
  - trigger_agent  # Predefined tool
```

**In Python tools:**

```python
from connic.tools import trigger_agent

async def process_invoice(invoice_url: str) -> dict:
    """Process an invoice and send confirmation."""
    
    # Trigger invoice processor
    result = await trigger_agent(
        agent_name="invoice-processor",
        payload={"url": invoice_url},
        wait_for_response=True
    )
    
    # Send confirmation email
    await trigger_agent(
        agent_name="email-agent",
        payload={"to": result["customer_email"]},
        wait_for_response=False  # Fire and forget
    )
    
    return result
```

### 📚 Knowledge & RAG

Agents can store and retrieve information using built-in RAG:

```yaml
version: "1.0"

name: knowledge-agent
model: gemini/gemini-2.5-flash
description: "Agent with persistent memory"

system_prompt: |
  You have access to a persistent knowledge base.
  
  Before answering, search the knowledge base for relevant info.
  When users share important info, offer to save it for later.

tools:
  - query_knowledge    # Semantic search
  - store_knowledge    # Add knowledge
  - delete_knowledge   # Remove knowledge
```

**In Python tools:**

```python
from connic.tools import query_knowledge, store_knowledge

async def answer_with_context(question: str) -> str:
    """Answer using knowledge base."""
    
    # Search for relevant knowledge
    results = await query_knowledge(
        query=question,
        namespace="policies",
        min_score=0.7,
        max_results=3
    )
    
    # Use results to answer...
    context = "\n".join([r["content"] for r in results["results"]])
    return f"Based on our policies: {context}"

async def save_preference(user_id: str, preference: str):
    """Save user preference."""
    await store_knowledge(
        content=preference,
        entry_id=f"user-{user_id}-pref",
        namespace="preferences"
    )
```

### 🔌 MCP Server Integration

Connect to external MCP (Model Context Protocol) servers for additional tools:

```yaml
version: "1.0"

name: docs-assistant
model: gemini/gemini-2.5-flash
description: "Fetches library documentation"

system_prompt: |
  You help with coding questions by fetching up-to-date documentation.
  Use the MCP tools to get the latest docs.

mcp_servers:
  - name: context7
    url: https://mcp.context7.com/mcp
    # Optional: filter specific tools
    # tools:
    #   - user-context7-resolve-library-id
    #   - user-context7-get-library-docs
  
  - name: internal-tools
    url: https://mcp.internal.example.com
    headers:
      Authorization: "Bearer ${MCP_API_KEY}"  # Env var substitution
```

### 📋 Structured Output

Define JSON schemas for type-safe responses:

**Create `schemas/invoice.json`:**

```json
{
  "type": "object",
  "description": "Extracted invoice data",
  "properties": {
    "vendor": {
      "type": "string",
      "description": "Vendor name"
    },
    "invoice_number": {
      "type": "string"
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
      "enum": ["USD", "EUR", "GBP"]
    },
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": {"type": "string"},
          "quantity": {"type": "number"},
          "unit_price": {"type": "number"}
        },
        "required": ["description", "quantity", "unit_price"]
      }
    }
  },
  "required": ["vendor", "total", "currency"]
}
```

**Reference in agent:**

```yaml
version: "1.0"

name: invoice-processor
model: gemini/gemini-2.5-pro
description: "Extracts structured data from invoices"

system_prompt: |
  Extract all relevant data from the invoice and return as structured JSON.

output_schema: invoice  # References schemas/invoice.json
```

### 🎛️ Middleware

Intercept and modify requests/responses with middleware:

**Create `middleware/assistant.py`:**

```python
from connic import StopProcessing

async def before(content: dict, context: dict) -> dict:
    """
    Runs before the agent processes the request.
    
    Args:
        content: Dict with "role" and "parts" keys.
                 Each part is {"text": "..."} or {"data": bytes, "mime_type": "..."}.
        context: Metadata (run_id, agent_name, connector_id, timestamp, etc.)
    
    Returns:
        Modified content dict
    """
    # Example 1: Attach a PDF document
    pdf_bytes = open("knowledge_base.pdf", "rb").read()
    content["parts"].append({"data": pdf_bytes, "mime_type": "application/pdf"})
    
    # Example 2: Add system context
    user_tier = context.get("user_tier", "free")
    content["parts"].insert(0, {"text": f"[User tier: {user_tier}]"})
    
    # Example 3: Authentication check
    if not context.get("api_key"):
        raise StopProcessing("Authentication required")
    
    return content

async def after(response: str, context: dict) -> str:
    """
    Runs after the agent completes.
    
    Args:
        response: The agent's response text
        context: Metadata (run_id, token_usage, duration_ms, etc.)
    
    Returns:
        Modified response
    """
    # Log usage
    tokens = context.get("token_usage", {})
    print(f"Run {context['run_id']}: {tokens.get('total', 0)} tokens")
    
    # Add footer
    return f"{response}\n\n---\nProcessed by {context['agent_name']}"
```

> **💡 Middleware files are auto-discovered by agent name.**
> `middleware/assistant.py` applies to the `assistant` agent.

---

## Examples

### E-commerce Order Processing

```yaml
# agents/order-processor.yaml
version: "1.0"

name: order-processor
model: gemini/gemini-2.5-flash
description: "Processes customer orders"

system_prompt: |
  You process customer orders. For each order:
  1. Validate inventory using check_inventory
  2. Calculate total with calculate_total
  3. Charge payment with process_payment
  4. Send confirmation with send_email

tools:
  - inventory.check_inventory
  - billing.calculate_total
  - billing.process_payment
  - notifications.send_email

max_concurrent_runs: 50
timeout: 60

retry_options:
  attempts: 3
  max_delay: 30
```

### Multi-Agent Document Processing

```yaml
# agents/document-pipeline.yaml
version: "1.0"
type: sequential

name: document-pipeline
description: "Extract, classify, and store documents"

agents:
  - ocr-agent         # Extract text from images/PDFs
  - classifier-agent  # Classify document type
  - extractor-agent   # Extract structured data
  - validator-agent   # Validate extracted data
  - storage-agent     # Store in database
```

### Smart Email Assistant

```yaml
# agents/email-assistant.yaml
version: "1.0"

name: email-assistant
model: gemini/gemini-2.5-pro
description: "Intelligent email management"

system_prompt: |
  You manage emails intelligently:
  - Search past emails for context
  - Draft professional responses
  - Remember user preferences
  - Categorize and prioritize

tools:
  - email.search
  - email.send
  - email.draft
  - query_knowledge   # Search email history
  - store_knowledge   # Remember preferences

temperature: 0.7
max_concurrent_runs: 20
```

---

## CLI Reference

### `connic init`

Initialize a new agent project.

```bash
connic init                # Initialize in current directory
connic init my-project     # Create new directory
```

### `connic dev`

Validate and preview agents locally.

```bash
connic dev           # Validate project
connic dev --verbose # Show detailed output
```

**Checks:**
- ✅ YAML syntax and schema validation
- ✅ Tool references and type hints
- ✅ Middleware discovery
- ✅ Output schema validation
- ✅ File size and count limits

### `connic test`

Start a hot-reload test session against Connic cloud.

```bash
connic login                  # Save credentials (one-time)
connic test                   # Ephemeral environment
connic test my-feature        # Named environment (persists)
```

**Features:**
- ⚡ 2-5 second hot reload on file changes
- 🔗 Real webhook URL for testing
- 📦 Full runtime environment
- 🧹 Auto-cleanup (ephemeral mode)

**Environment variables:**
```bash
CONNIC_API_KEY=cnc_xxx
CONNIC_PROJECT_ID=<uuid>
```

### `connic deploy`

Deploy agents directly (for projects without git).

```bash
connic deploy                    # Deploy to default environment
connic deploy --env <env-id>     # Deploy to specific environment
```

**Requirements:**
- Project must not have git connected
- API key and project ID configured (`connic login`)

### `connic tools`

List all available tools in the project.

```bash
connic tools
```

### `connic login`

Save credentials for the current project.

```bash
connic login                                    # Interactive
connic login --api-key cnc_xxx --project-id ... # Non-interactive
```

Creates `.connic` file (add to `.gitignore`).

---

## Configuration Reference

### Agent Configuration Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | ✅ | Schema version (currently `"1.0"`) |
| `name` | string | ✅ | Unique agent identifier (lowercase, hyphens) |
| `description` | string | ✅ | Human-readable description |
| `type` | string | | Agent type: `llm` (default), `sequential`, or `tool` |
| `model` | string | LLM only | AI model (e.g., `gemini/gemini-2.5-flash`) |
| `system_prompt` | string | LLM only | Instructions for the agent |
| `tools` | string[] | | List of tools (`module.function`) |
| `agents` | string[] | Sequential only | List of agent names to chain |
| `tool_name` | string | Tool only | Tool to execute (`module.function`) |
| `temperature` | number | | Randomness 0-2 (default: 1.0) |
| `max_concurrent_runs` | integer | | Max simultaneous runs (default: 1) |
| `timeout` | integer | | Execution timeout in seconds (min: 5) |
| `retry_options` | object | | Retry configuration |
| `output_schema` | string | | Schema name for structured output |
| `mcp_servers` | array | | MCP server connections (max: 50) |

### Retry Options

```yaml
retry_options:
  attempts: 5      # Max attempts (1-10, default: 3)
  max_delay: 60    # Max delay between retries in seconds (1-300, default: 30)
```

### Supported Models

| Model | Description | Best For |
|-------|-------------|----------|
| `gemini/gemini-2.5-pro` | Most capable | Complex reasoning, long context |
| `gemini/gemini-2.5-flash` | Fast and efficient | General purpose, high volume |
| `gemini/gemini-2.5-flash-lite` | Lightweight | Simple tasks, cost optimization |

> **Note:** Model availability depends on your subscription tier.

---

## Python API

Use the SDK programmatically:

```python
from connic import ProjectLoader, Agent

# Load all agents
loader = ProjectLoader("./my-project")
agents = loader.load_agents()

print(f"Found {len(agents)} agents")
for agent in agents:
    print(f"  - {agent.config.name}: {agent.config.description}")

# Load specific agent
assistant = loader.load_agent("assistant")

# Get tools schema for LLM
tools_schema = assistant.get_tools_schema()

# Execute a tool directly
calculator_tool = assistant.get_tool("add")
result = calculator_tool.execute_sync(a=5, b=3)
print(f"5 + 3 = {result}")

# Execute async tool
result = await calculator_tool.execute(a=10, b=20)
```

### Discover Tools

```python
# Discover all available tools
tools = loader.discover_tools()
for module, functions in tools.items():
    print(f"{module}: {', '.join(functions)}")

# Discover middleware
middlewares = loader.discover_middlewares()
for agent_name, hooks in middlewares.items():
    print(f"{agent_name}: {', '.join(hooks)}")
```

---

## Best Practices

### 🎯 Agent Design

- **Keep agents focused** - One responsibility per agent
- **Use sequential agents** for multi-step workflows
- **Use tool agents** for deterministic operations
- **Use orchestrator pattern** for dynamic routing

### 🔧 Tool Development

- **Always use type hints** - Required for LLM function calling
- **Write detailed docstrings** - The LLM reads them
- **Handle errors gracefully** - Return error messages, don't raise
- **Keep tools atomic** - Each tool does one thing well
- **Use async for I/O** - Network calls, database queries, file I/O

**Good:**

```python
async def fetch_user(user_id: str) -> dict:
    """Fetch user information from the database.
    
    Args:
        user_id: The unique user identifier
    
    Returns:
        Dictionary with user data, or error dict if not found
    """
    try:
        async with db.get_connection() as conn:
            user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
            if not user:
                return {"error": "User not found"}
            return dict(user)
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}
```

**Bad:**

```python
def get_user(id):  # ❌ No type hints, no docstring, no error handling
    user = db.query(f"SELECT * FROM users WHERE id = {id}")  # ❌ SQL injection
    return user  # ❌ What if user is None?
```

### 📊 Observability

- Use middleware to log token usage
- Add context to error messages
- Store important metrics in knowledge base

### 🔒 Security

- **Never commit `.connic`** - Add to `.gitignore`
- **Use environment variables** for secrets in MCP headers
- **Validate input** in tools before processing
- **Sanitize output** before returning sensitive data

---

## Limits & Quotas

### File Limits

- **Max total upload size:** 1MB (compressed)
- **Allowed file types:** `.py`, `.yaml`, `.yml`, `.json`, `.txt`, `.csv`, `.md`, `.toml`, `.jsonl`
- Hidden files and `__pycache__` are automatically excluded

### Agent Limits

- **MCP servers per agent:** 50
- **Total tools per project:** 100
- **Max concurrent runs:** Tier-based (Free: 1, Pro: 10, Ultimate: 100)
- **Max retry attempts:** 10
- **Max retry delay:** 300 seconds
- **Min timeout:** 5 seconds

### Naming Conventions

- **Agent names:** Lowercase letters, numbers, hyphens (e.g., `my-agent-123`)
- **Tool references:** `module.function` format (e.g., `calculator.add`)
- **Schema references:** Name without `.json` extension (e.g., `output_schema: invoice`)

---

## Troubleshooting

### Common Issues

**❌ "Tool 'calculator.add' not found"**
- Ensure `tools/calculator.py` exists
- Check function name matches exactly
- Run `connic tools` to see available tools

**❌ "Invalid agent name"**
- Use only lowercase letters, numbers, and hyphens
- Must start and end with letter or number
- Examples: `my-agent`, `agent1`, `invoice-processor`

**❌ "Output schema validation failed"**
- Check `schemas/<name>.json` exists
- Ensure valid JSON Schema format
- Must have `type` field

**❌ "Module 'tools.calculator' has no function 'add'"**
- Check function is not private (doesn't start with `_`)
- Ensure function is defined at module level
- Verify spelling matches exactly

**❌ "Package size exceeds 1MB limit"**
- Remove large dependencies from `requirements.txt`
- Use `.gitignore` to exclude large files
- Consider moving large files to external storage

### Getting Help

- 📖 [Documentation](https://connic.co/docs/v1/composer)
- 🐛 [Issue Tracker](https://github.com/connic-org/connic-composer-sdk/issues)
- 📧 Email: [support@connic.co](mailto:support@connic.co)

---

## Project Structure

A typical Connic project:

```
my-agent-project/
├── agents/                    # Agent configurations
│   ├── assistant.yaml         # LLM agent
│   ├── invoice-processor.yaml # LLM agent with output schema
│   ├── tax-calculator.yaml    # Tool agent
│   ├── document-pipeline.yaml # Sequential agent
│   ├── orchestrator.yaml      # Orchestrator with trigger_agent
│   ├── knowledge-agent.yaml   # Knowledge agent with RAG
│   └── mcp-docs.yaml          # MCP agent
│
├── tools/                     # Custom Python tools
│   ├── calculator.py          # Math operations
│   ├── database.py            # Database queries
│   └── api.py                 # External API calls
│
├── middleware/                # Optional hooks
│   ├── assistant.py           # Middleware for assistant agent
│   └── invoice-processor.py   # Middleware for invoice-processor
│
├── schemas/                   # JSON schemas for structured output
│   ├── invoice.json           # Invoice schema
│   └── order.json             # Order schema
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── .connic                    # Local credentials (DO NOT COMMIT)
└── README.md                  # Project documentation
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## What's Next?

- 🔗 Read the [full documentation](https://connic.co/docs/v1/composer)
- 🚀 [Create your first agent](https://connic.co/dashboard)

---

<div align="center">

**Built with ❤️ by the Connic team**

[Website](https://connic.co) • [Documentation](https://connic.co/docs) • [Dashboard](https://connic.co/dashboard)

</div>
