# Connic Composer SDK

Build Connic agents in code.

Define agents in YAML, extend them with Python, test them with hot reload against Connic cloud, and deploy them to Connic-managed environments.

[PyPI version](https://pypi.org/project/connic-composer-sdk/)
[Python 3.10+](https://www.python.org/downloads/)
[License: MIT](https://opensource.org/licenses/MIT)

[Documentation](https://connic.co/docs/v1/composer/overview) • [Quickstart](https://connic.co/docs/v1/quickstart) • [Agent Templates](https://connic.co/agents) • [Dashboard](https://connic.co/projects)

## What It Is

`connic-composer-sdk` is the Python SDK and CLI for Connic's code-first agent workflow.

Use it to:

- define agents in YAML
- write custom tools as plain Python functions
- add middleware, schemas, sessions, retries, concurrency, and guardrails
- connect MCP servers and use Connic's predefined tools
- validate projects locally before uploading anything
- run hot-reload test sessions in isolated cloud environments
- deploy to Connic from the CLI or through a connected Git repository

## Installation

```bash
pip install connic-composer-sdk
```

Requires Python 3.10 or newer.

## Quick Start

```bash
# Create a new project
connic init my-agents
cd my-agents

# Authenticate this project with Connic
connic login

# Validate the project locally
connic lint

# Start a hot-reload test session in Connic cloud
connic test
```

The default scaffold is intentionally minimal. If you want a starter project with working examples, use templates:

```bash
connic init my-agents --templates=invoice,customer-support
```

Browse available templates at [connic.co/agents](https://connic.co/agents).

## Example Project

```text
my-agents/
├── agents/
│   └── support-assistant.yaml
├── tools/
│   └── billing.py
├── middleware/
│   └── support-assistant.py
├── schemas/
└── requirements.txt
```

### `agents/support-assistant.yaml`

```yaml
version: "1.0"

name: support-assistant
type: llm
model: gemini/gemini-2.5-pro
description: "Customer support agent with billing and knowledge access"
system_prompt: |
  You are a concise support agent.
  Use tools when they help produce a more accurate answer.

tools:
  - billing.lookup_invoice
  - query_knowledge

session:
  key: input.user_id
  ttl: 86400

guardrails:
  input:
    - type: prompt_injection
      mode: block
  output:
    - type: system_prompt_leakage
      mode: block
```

### `tools/billing.py`

```python
def lookup_invoice(invoice_id: str) -> dict:
    """Look up invoice status by invoice ID.

    Args:
        invoice_id: The invoice identifier

    Returns:
        Invoice details for the requested invoice
    """
    return {
        "invoice_id": invoice_id,
        "status": "paid",
        "amount": 199.0,
        "currency": "USD",
    }
```

Run:

```bash
connic lint
connic tools
connic test
```

## Core Concepts

### Agent Types

See [Agent Configuration](https://connic.co/docs/v1/composer/agent-configuration) for the full YAML reference.

- `llm`: an LLM-driven agent with prompts, tools, MCP servers, schemas, and guardrails
- `tool`: a direct wrapper around a Python tool
- `sequential`: a pipeline that executes multiple agents in order

### Tools

Custom tools are plain Python functions discovered from `tools/`, including nested modules. Type hints and docstrings are used to generate tool schemas automatically. See [Writing Tools](https://connic.co/docs/v1/composer/write-tools) for details.

The SDK also exposes predefined Connic tools such as the ones documented in [Predefined Tools](https://connic.co/docs/v1/composer/predefined-tools):

- `trigger_agent`
- `trigger_agent_at`
- `query_knowledge`
- `store_knowledge`
- `delete_knowledge`
- `kb_list_namespaces`
- `web_search`
- `web_read_page`
- `db_find`
- `db_insert`
- `db_update`
- `db_delete`
- `db_count`
- `db_list_collections`

### Middleware and Runtime Controls

Per-agent middleware lets you modify inputs, enrich context, attach files, stop execution early, and transform outputs. See [Middleware](https://connic.co/docs/v1/composer/middleware).

The YAML model also supports:

- retries
- timeouts
- max iteration limits
- key-based concurrency control
- persistent sessions
- output schemas
- MCP server connections
- input and output guardrails

Related docs:

- [MCP](https://connic.co/docs/v1/composer/mcp)
- [Guardrails](https://connic.co/docs/v1/composer/guardrails)
- [Variables](https://connic.co/docs/v1/composer/variables)
- [Knowledge Tools](https://connic.co/docs/v1/composer/knowledge-tools)
- [Database Tools](https://connic.co/docs/v1/composer/database-tools)

## CLI Commands


| Command                              | Description                                                     |
| ------------------------------------ | --------------------------------------------------------------- |
| `connic init [name]`                 | Create a new project scaffold                                   |
| `connic init [name] --templates=...` | Create a project from one or more starter templates             |
| `connic login`                       | Save project credentials in `.connic`                           |
| `connic lint`                        | Validate agents, tools, middleware, and schemas locally         |
| `connic tools`                       | List discovered tools and signatures                            |
| `connic test [name]`                 | Start an isolated cloud test environment with hot reload        |
| `connic deploy`                      | Deploy from the CLI to a Connic environment                     |
| `connic migrate`                     | Migrate a LangChain or Google ADK project into a Connic project |


Run `connic <command> --help` for flags and examples.

## Development Workflow

### Local Validation

`connic lint` loads your project locally and catches issues like:

- invalid YAML
- missing required agent fields
- unresolved tool references
- duplicate agent names
- schema and middleware loading problems

### Hot-Reload Testing

`connic test` creates an isolated development environment in Connic cloud, uploads your local files, and re-syncs changes in a few seconds while you iterate.

This is the main development loop when you need real connectors, predefined tools, and environment-scoped services.

### Deployment

Deployment targets Connic-managed environments.

- If your Connic project is connected to a Git repository, pushing to the configured branch is the primary deployment flow.
- `connic deploy` is available for CLI-driven deployments and for projects that do not use a connected repository.

## Documentation


| Topic               | Link                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| Overview            | [connic.co/docs/v1/composer/overview](https://connic.co/docs/v1/composer/overview)                       |
| Quickstart          | [connic.co/docs/v1/quickstart](https://connic.co/docs/v1/quickstart)                                     |
| Agent Configuration | [connic.co/docs/v1/composer/agent-configuration](https://connic.co/docs/v1/composer/agent-configuration) |
| Writing Tools       | [connic.co/docs/v1/composer/write-tools](https://connic.co/docs/v1/composer/write-tools)                 |
| Middleware          | [connic.co/docs/v1/composer/middleware](https://connic.co/docs/v1/composer/middleware)                   |
| Predefined Tools    | [connic.co/docs/v1/composer/predefined-tools](https://connic.co/docs/v1/composer/predefined-tools)       |
| MCP                 | [connic.co/docs/v1/composer/mcp](https://connic.co/docs/v1/composer/mcp)                                 |
| Testing             | [connic.co/docs/v1/composer/testing](https://connic.co/docs/v1/composer/testing)                         |
| Variables           | [connic.co/docs/v1/composer/variables](https://connic.co/docs/v1/composer/variables)                     |
| Guardrails          | [connic.co/docs/v1/composer/guardrails](https://connic.co/docs/v1/composer/guardrails)                   |
| Knowledge Tools     | [connic.co/docs/v1/composer/knowledge-tools](https://connic.co/docs/v1/composer/knowledge-tools)         |
| Database Tools      | [connic.co/docs/v1/composer/database-tools](https://connic.co/docs/v1/composer/database-tools)           |


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Support

- Issues: [github.com/connic-org/connic-composer-sdk/issues](https://github.com/connic-org/connic-composer-sdk/issues)
- Email: [support@connic.co](mailto:support@connic.co)

## License

MIT. See [LICENSE](LICENSE).