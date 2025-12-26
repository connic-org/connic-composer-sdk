# Connic Composer SDK

<div align="center">

**Build production-ready AI agents with code.**

Define agents in YAML. Extend them with Python. Deploy anywhere.

[![PyPI version](https://img.shields.io/pypi/v/connic-composer-sdk.svg)](https://pypi.org/project/connic-composer-sdk/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Documentation](https://connic.co/docs/v1/composer/overview) • [Quick Start](https://connic.co/docs/v1/quickstart) • [Dashboard](https://connic.co/projects)

</div>

---

## Installation

```bash
pip install connic-composer-sdk
```

## Quick Start

```bash
# Initialize a new project
connic init my-agents
cd my-agents

# Validate your project
connic dev

# Test with hot reload
connic login
connic test

# Deploy
connic deploy
```

## Documentation

For detailed guides and configuration options, see the full documentation:

| Topic | Link |
|-------|------|
| **Overview** | [Getting Started](https://connic.co/docs/v1/composer/overview) |
| **Agent Configuration** | [Agent YAML Reference](https://connic.co/docs/v1/composer/agent-configuration) |
| **Custom Tools** | [Writing Python Tools](https://connic.co/docs/v1/composer/tools) |
| **Testing** | [Local Development & Testing](https://connic.co/docs/v1/composer/testing) |
| **Deployment** | [Deploying Agents](https://connic.co/docs/v1/composer/deployment) |
| **Knowledge & RAG** | [Built-in RAG](https://connic.co/docs/v1/composer/knowledge) |
| **Middleware** | [Request/Response Hooks](https://connic.co/docs/v1/composer/middleware) |

## CLI Commands

| Command | Description |
|---------|-------------|
| `connic init [name]` | Initialize a new project |
| `connic dev` | Validate project locally |
| `connic test [env]` | Start hot-reload test session |
| `connic deploy` | Deploy to production |
| `connic tools` | List available tools |
| `connic login` | Save credentials |

## Getting Help

- 📖 [Full Documentation](https://connic.co/docs/v1/composer/overview)
- 🐛 [Issue Tracker](https://github.com/connic-org/connic-composer-sdk/issues)
- 📧 [support@connic.co](mailto:support@connic.co)

## License

MIT License - see [LICENSE](LICENSE) file.

---

<div align="center">

**Built with ❤️ by the Connic team**

[Website](https://connic.co) • [Documentation](https://connic.co/docs) • [Projects](https://connic.co/projects)

</div>
