# LangChain Migration Validation

This directory contains modern LangChain and LangGraph migration trials for `connic migrate`.

`test_official_langchain_fixture_migrates_loads_and_lints` regenerates the three
official fixtures in temporary directories and is the automated source of truth.
The sibling generated project directories are manual inspection snapshots, not
golden test outputs.

## Sources used

Local fixtures based on current official LangChain docs:

- `tests/langchain/sources/official-overview-agent`
- `tests/langchain/sources/official-rag-agent`
- `tests/langchain/sources/official-supervisor-agent`

Cloned modern upstream repositories/examples:

- `tests/langchain/sources/langgraph-swarm-py/examples/customer_support`
- `tests/langchain/sources/langgraph-swarm-py/examples/research`
- `tests/langchain/sources/new-langgraph-project`

## Migration results

### Passing migrations

#### `official-overview-agent`

- output: `tests/langchain/official-overview-agent`
- validation: `connic lint` passes
- preserved:
  - single `create_agent(...)`
  - static system prompt
  - plain Python tool
- good baseline for current LangChain `create_agent` support

#### `official-rag-agent`

- output: `tests/langchain/official-rag-agent`
- validation: `connic lint` passes
- preserved:
  - `create_agent(...)`
  - retrieval tool pattern
  - local support class used by the tool
- useful regression for tool extraction plus local helper/class extraction

#### `customer-support`

- output: `tests/langchain/customer-support`
- source: `langgraph-swarm-py/examples/customer_support`
- validation: `connic lint` passes
- preserved:
  - both `create_agent(...)` assistants
  - flight/hotel tool functions
  - same-file data/constants used by the tools
- limitations:
  - swarm handoff tools are not migrated
  - dynamic prompt factory was not converted; migrated agents use placeholder prompts

#### `research-swarm`

- output: `tests/langchain/research-swarm`
- source: `langgraph-swarm-py/examples/research`
- validation: `connic lint` passes
- preserved:
  - planner/researcher agents
  - imported tool from neighboring `utils.py`
  - model extraction from `init_chat_model(...)`
- limitations:
  - handoff tools are not migrated
  - imported prompt construction remains too dynamic, so placeholder prompts were used

#### `official-supervisor-agent`

- output: `tests/langchain/official-supervisor-agent`
- validation: `connic lint` passes without `OPENAI_API_KEY`
- preserved:
  - calendar, email, and supervisor agents
  - low-level calendar and email tool functions
  - simple sub-agent wrappers rewritten to Connic `trigger_agent` delegation
- useful regression for credential-free supervisor migration and wrapper isolation

### Unsupported modern example

#### `new-langgraph-project`

- source: `tests/langchain/sources/new-langgraph-project`
- migration result: no output generated
- current behavior: `connic migrate` reports `No migratable agents were found.`
- reason:
  - this project is a low-level LangGraph `StateGraph` template, not a LangChain `create_agent(...)` codebase
- takeaway:
  - current migration coverage is LangChain-agent oriented, not generic LangGraph graph compilation

## Improvements made while validating

During this validation pass the migrator was improved to better handle modern LangChain examples:

- support for `create_react_agent(...)` detection in addition to `create_agent(...)`
- better resolution of imported string constants and imported helper functions for prompts/models
- support for neighboring-module imports like `from utils import fetch_doc`
- extraction of local helper classes needed by migrated tool modules

## Suggested next targets

1. add explicit reporting for unsupported LangGraph `StateGraph` templates
2. consider first-class migration handling for handoff tools / swarm patterns
