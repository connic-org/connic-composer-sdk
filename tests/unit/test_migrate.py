from pathlib import Path
from textwrap import dedent

import click
import pytest
import yaml
from click.testing import CliRunner

from connic import migrate
from connic.cli import _run_lint
from connic.loader import ProjectLoader


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n")


def no_scaffold(destination, quiet=False):
    (destination / ".gitignore").write_text(".connic\n")


def make_migrate_cli(run_lint=lambda **kwargs: True):
    group = click.Group()
    migrate.register_migrate_command(group, no_scaffold, run_lint)
    return group


@pytest.mark.parametrize(
    ("fixture_name", "expected_agent_names"),
    [
        pytest.param("official-overview-agent", {"agent"}, id="overview"),
        pytest.param("official-rag-agent", {"agent"}, id="rag"),
        pytest.param(
            "official-supervisor-agent",
            {"calendar-agent", "email-agent", "supervisor-agent"},
            id="supervisor",
        ),
    ],
)
def test_official_langchain_fixture_migrates_loads_and_lints(
    tmp_path,
    monkeypatch,
    fixture_name,
    expected_agent_names,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = Path(__file__).parents[1] / "langchain" / "sources" / fixture_name
    destination = tmp_path / fixture_name

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    loader = ProjectLoader(str(destination))
    migrated_agents = loader.load_agents()

    assert framework == "langchain"
    assert loader._load_errors == []
    assert {agent.config.name for agent in migrated_agents} == expected_agent_names
    assert _run_lint(quiet=True, project_root=str(destination)) is True


def test_langchain_migration_generates_agent_yaml_and_extracted_tools(tmp_path):
    source = tmp_path / "langchain-app"
    destination = tmp_path / "connic-app"
    write(
        source / "prompts.py",
        '''
        SUPPORT_PROMPT = "Answer support tickets with invoice context."
        ''',
    )
    write(
        source / "support_tools.py",
        '''
        TAX_RATE = 0.19


        class InvoiceFormatter:
            def format(self, amount):
                return {"amount": amount, "tax": amount * TAX_RATE}


        def normalize_invoice_id(invoice_id: str) -> str:
            return invoice_id.strip().upper()


        def lookup_invoice(invoice_id: str) -> dict:
            formatter = InvoiceFormatter()
            return formatter.format(len(normalize_invoice_id(invoice_id)))


        def unused_internal_helper():
            return "not migrated"
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        from prompts import SUPPORT_PROMPT
        from support_tools import lookup_invoice


        support_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[lookup_invoice, "query_knowledge"],
            system_prompt=SUPPORT_PROMPT,
            name="Support Agent",
        )
        ''',
    )
    (source / "requirements.txt").write_text("langchain>=0.3\n")

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    report_notes = migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "support-agent.yaml").read_text())
    tools_py = (destination / "tools" / "support_tools.py").read_text()

    assert framework == "langchain"
    assert agent_yaml["name"] == "support-agent"
    assert agent_yaml["model"] == "openai/gpt-4o-mini"
    assert agent_yaml["system_prompt"] == "Answer support tickets with invoice context."
    assert agent_yaml["tools"] == ["support_tools.lookup_invoice", "retrieval_query"]
    assert "def lookup_invoice" in tools_py
    assert "def normalize_invoice_id" in tools_py
    assert "class InvoiceFormatter" in tools_py
    assert "unused_internal_helper" not in tools_py
    assert (destination / "requirements.txt").read_text() == "langchain>=0.3\n"
    assert any("Custom guardrails are not migrated automatically" in note for note in report_notes)


def test_langchain_migration_resolves_chat_model_assignment_keyword(tmp_path):
    source = tmp_path / "langchain-chat-model"
    destination = tmp_path / "connic-app"
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI


        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


        def lookup_invoice(invoice_id: str) -> dict:
            return {"invoice_id": invoice_id, "status": "paid"}


        support_agent = create_agent(
            model=llm,
            tools=[lookup_invoice],
            prompt="Resolve invoice questions with billing context.",
            name="Billing Support",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "billing-support.yaml").read_text())

    assert framework == "langchain"
    assert agent_yaml["model"] == "openai/gpt-4.1-mini"
    assert agent_yaml["system_prompt"] == "Resolve invoice questions with billing context."
    assert agent_yaml["tools"] == ["agent.lookup_invoice"]


def test_langchain_migration_reports_wildcard_imported_tools_without_crashing(tmp_path):
    source = tmp_path / "langchain-wildcard-import"
    destination = tmp_path / "connic-app"
    write(
        source / "tools.py",
        '''
        def lookup_invoice(invoice_id: str) -> dict:
            return {"invoice_id": invoice_id, "status": "paid"}
        ''',
    )
    agent_source = dedent(
        '''
        from langchain.agents import create_agent
        from tools import *


        support_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[lookup_invoice],
            prompt="Resolve invoice questions with billing context.",
            name="Billing Support",
        )
        '''
    ).strip().encode() + b"\n# copied from legacy editor: \xff\n"
    (source / "agent.py").write_bytes(agent_source)

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "billing-support.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "langchain"
    assert agent_yaml["model"] == "openai/gpt-4o-mini"
    assert "tools" not in agent_yaml
    assert "Could not resolve tool reference 'lookup_invoice' in agent.py." in report


def test_langchain_migration_reports_external_and_string_tool_refs_for_manual_mapping(tmp_path):
    source = tmp_path / "langchain-external-tools"
    destination = tmp_path / "connic-app"
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        import billing_tools


        support_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[billing_tools.lookup_invoice, "internal_crm.search_account"],
            system_prompt="Answer billing questions with invoice and CRM context.",
            name="Billing Support",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "billing-support.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "langchain"
    assert agent_yaml["model"] == "openai/gpt-4o-mini"
    assert agent_yaml["system_prompt"] == "Answer billing questions with invoice and CRM context."
    assert "tools" not in agent_yaml
    assert "Could not resolve tool attribute 'billing_tools.lookup_invoice'." in report
    assert "Could not migrate string-based tool reference 'internal_crm.search_account'." in report


def test_migrate_command_rejects_destination_inside_source_project(tmp_path):
    source = tmp_path / "langchain-app"
    source.mkdir()
    destination = source / "connic-app"

    result = CliRunner().invoke(
        make_migrate_cli(),
        ["migrate", "--source", str(source), "--dest", str(destination)],
    )

    assert result.exit_code != 0
    assert "Destination path cannot be inside the source project" in result.output


def test_migrate_command_rejects_invalid_source_paths(tmp_path):
    missing_source = tmp_path / "missing"
    destination = tmp_path / "connic-app"
    missing_result = CliRunner().invoke(
        make_migrate_cli(),
        ["migrate", "--source", str(missing_source), "--dest", str(destination)],
    )

    file_source = tmp_path / "agent.py"
    file_source.write_text("# not a project directory\n", encoding="utf-8")
    file_result = CliRunner().invoke(
        make_migrate_cli(),
        ["migrate", "--source", str(file_source), "--dest", str(destination)],
    )

    same_path = tmp_path / "same"
    same_path.mkdir()
    same_result = CliRunner().invoke(
        make_migrate_cli(),
        ["migrate", "--source", str(same_path), "--dest", str(same_path)],
    )

    assert missing_result.exit_code != 0
    assert "does not exist" in missing_result.output
    assert file_result.exit_code != 0
    assert "is a file" in file_result.output
    assert same_result.exit_code != 0
    assert "Source path and destination path must be different" in same_result.output


def test_migrate_command_rejects_non_empty_destination(tmp_path):
    source = tmp_path / "langchain-app"
    destination = tmp_path / "connic-app"
    source.mkdir()
    destination.mkdir()
    (destination / "existing.txt").write_text("keep me\n", encoding="utf-8")

    result = CliRunner().invoke(
        make_migrate_cli(),
        ["migrate", "--source", str(source), "--dest", str(destination)],
    )

    assert result.exit_code != 0
    assert "already exists and is not empty" in result.output
    assert (destination / "existing.txt").read_text(encoding="utf-8") == "keep me\n"


def test_migrate_command_generates_project_and_reports_lint_issues(tmp_path):
    source = tmp_path / "langchain-app"
    destination = tmp_path / "connic-app"
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent


        def check_order_status(order_id: str) -> dict:
            return {"order_id": order_id, "status": "shipped"}


        support_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[check_order_status],
            system_prompt="Answer support questions with order status context.",
            name="Order Support",
        )
        ''',
    )

    lint_calls = []

    def failing_lint(**kwargs):
        lint_calls.append(kwargs)
        return False

    result = CliRunner().invoke(
        make_migrate_cli(failing_lint),
        ["migrate", "--source", str(source), "--dest", str(destination)],
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "order-support.yaml").read_text())

    assert result.exit_code == 0
    assert "Framework: langchain" in result.output
    assert "Agents found: 1" in result.output
    assert "Tools found: 1" in result.output
    assert "Migration completed with lint issues" in result.output
    assert lint_calls == [{"quiet": True, "project_root": str(destination.resolve())}]
    assert agent_yaml["model"] == "openai/gpt-4o-mini"
    assert agent_yaml["tools"] == ["agent.check_order_status"]
    assert "def check_order_status" in (destination / "tools" / "agent.py").read_text()
    assert "Follow-up Items" in (destination / "MIGRATION_REPORT.md").read_text()


def test_migrate_command_prompts_for_paths_and_completes_successfully(tmp_path):
    source = tmp_path / "langchain-app"
    destination = tmp_path / "connic-app"
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent


        def lookup_account(account_id: str) -> dict:
            return {"account_id": account_id, "tier": "enterprise"}


        account_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[lookup_account],
            system_prompt="Answer account questions with CRM context.",
            name="Account Support",
        )
        ''',
    )

    lint_calls = []

    def passing_lint(**kwargs):
        lint_calls.append(kwargs)
        return True

    result = CliRunner().invoke(
        make_migrate_cli(passing_lint),
        ["migrate"],
        input=f"{source}\n{destination}\n",
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "account-support.yaml").read_text())

    assert result.exit_code == 0
    assert "Existing project path" in result.output
    assert "New Connic project path" in result.output
    assert "Migration complete." in result.output
    assert lint_calls == [{"quiet": True, "project_root": str(destination.resolve())}]
    assert agent_yaml["model"] == "openai/gpt-4o-mini"
    assert agent_yaml["tools"] == ["agent.lookup_account"]


def test_adk_migration_preserves_agent_chain_and_function_tools(tmp_path):
    source = tmp_path / "adk-app"
    destination = tmp_path / "connic-app"
    write(
        source / "tools" / "catalog.py",
        '''
        def search_catalog(query: str, limit: int = 5) -> dict:
            return {"query": query, "limit": limit}
        ''',
    )
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import Agent, SequentialAgent
        from google.adk.tools import FunctionTool
        from tools.catalog import search_catalog


        catalog_agent = Agent(
            name="Catalog Agent",
            model="gemini-2.0-flash",
            instruction="Find products for the shopper.",
            tools=[FunctionTool(func=search_catalog), "web_search"],
        )

        root_agent = SequentialAgent(
            name="Shopping Flow",
            sub_agents=[catalog_agent],
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    catalog_yaml = yaml.safe_load((destination / "agents" / "catalog-agent.yaml").read_text())
    flow_yaml = yaml.safe_load((destination / "agents" / "shopping-flow.yaml").read_text())

    assert framework == "adk"
    assert catalog_yaml["type"] == "llm"
    assert catalog_yaml["model"] == "gemini/gemini-2.0-flash"
    assert catalog_yaml["tools"] == ["tools.catalog.search_catalog", "web_search"]
    assert flow_yaml == {
        "version": "1.0",
        "name": "shopping-flow",
        "type": "sequential",
        "description": "Migrated from ADK agent 'Shopping Flow'.",
        "agents": ["catalog-agent"],
    }
    assert "def search_catalog" in (destination / "tools" / "tools" / "catalog.py").read_text()


def test_adk_migration_matches_docs_for_llm_agent_with_wrapped_tools_and_google_search(tmp_path):
    source = tmp_path / "adk-docs-example"
    destination = tmp_path / "connic-app"
    write(
        source / "tools.py",
        '''
        def search_docs(query: str) -> dict:
            return {"query": query, "matches": ["billing guide"]}


        def summarize(text: str) -> str:
            return text[:80]
        ''',
    )
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool
        from tools import search_docs, summarize


        support_agent = LlmAgent(
            name="support-agent",
            model="gemini-2.5-flash",
            instruction="""You are a technical support agent.
        Help users resolve issues by searching documentation
        and summarizing the results.""",
            tools=[FunctionTool(search_docs), summarize, "google_search"],
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "support-agent.yaml").read_text())
    migrated_tools = (destination / "tools" / "tools.py").read_text()

    assert framework == "adk"
    assert agent_yaml == {
        "version": "1.0",
        "name": "support-agent",
        "type": "llm",
        "description": "Migrated from ADK agent 'support-agent'.",
        "model": "gemini/gemini-2.5-flash",
        "system_prompt": (
            "You are a technical support agent.\n"
            "Help users resolve issues by searching documentation\n"
            "and summarizing the results."
        ),
        "tools": ["tools.search_docs", "tools.summarize", "web_search"],
    }
    assert "def search_docs" in migrated_tools
    assert "def summarize" in migrated_tools


def test_adk_migration_resolves_reusable_static_tool_list(tmp_path):
    source = tmp_path / "adk-shared-tools"
    destination = tmp_path / "connic-app"
    write(
        source / "tools.py",
        '''
        def search_docs(query: str) -> dict:
            return {"query": query, "matches": ["refund policy"]}
        ''',
    )
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import LlmAgent
        from google.adk.tools import FunctionTool, google_search
        from tools import search_docs


        SUPPORT_TOOLS = [
            FunctionTool(func=search_docs),
            google_search,
        ]


        support_agent = LlmAgent(
            name="support-agent",
            model="gemini-2.5-flash",
            instruction="Answer support questions using current product documentation.",
            tools=SUPPORT_TOOLS,
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    report_notes = migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "support-agent.yaml").read_text())
    loader = ProjectLoader(str(destination))
    migrated_agents = loader.load_agents()
    migrated_tools = {tool.name: tool for tool in migrated_agents[0].tools}

    assert framework == "adk"
    assert agent_yaml["tools"] == ["tools.search_docs", "web_search"]
    assert not any("dynamic tool expression" in note for note in report_notes)
    assert _run_lint(quiet=True, project_root=str(destination)) is True
    assert loader._load_errors == []
    assert migrated_tools["search_docs"].func("returns") == {
        "query": "returns",
        "matches": ["refund policy"],
    }


def test_adk_agent_with_sub_agents_list_variable_becomes_sequential_workflow(tmp_path):
    source = tmp_path / "adk-variable-workflow"
    destination = tmp_path / "connic-app"
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import Agent


        intake_agent = Agent(
            name="Intake Agent",
            model="gemini-2.5-flash",
            instruction="Collect the customer request.",
        )

        resolution_agent = Agent(
            name="Resolution Agent",
            model="gemini-2.5-flash",
            instruction="Resolve the request.",
        )

        SUPPORT_WORKFLOW = [intake_agent, resolution_agent]

        support_flow = Agent(
            name="Support Flow",
            description="Run intake before resolution.",
            sub_agents=SUPPORT_WORKFLOW,
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    workflow_yaml = yaml.safe_load((destination / "agents" / "support-flow.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "adk"
    assert workflow_yaml == {
        "version": "1.0",
        "name": "support-flow",
        "type": "sequential",
        "description": "Run intake before resolution.",
        "agents": ["intake-agent", "resolution-agent"],
    }
    assert "ADK sub_agents were migrated as a sequential Connic workflow." in report


def test_adk_migration_maps_agent_tool_and_reports_remaining_wrappers(tmp_path):
    source = tmp_path / "adk-wrappers"
    destination = tmp_path / "connic-app"
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import Agent
        from google.adk.tools import AgentTool, MCPToolset


        escalation_agent = Agent(
            name="Escalation Specialist",
            model="gemini-2.0-flash",
            instruction="Handle escalations.",
        )

        def build_dynamic_tool():
            return lambda payload: payload


        triage_agent = Agent(
            name="Triage Agent",
            model="gemini-2.0-flash",
            instruction="Triage support requests.",
            tools=[
                AgentTool(agent=escalation_agent),
                MCPToolset(connection_params={"url": "https://mcp.example"}),
                build_dynamic_tool(),
            ],
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    triage_yaml = yaml.safe_load((destination / "agents" / "triage-agent.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "adk"
    assert triage_yaml["tools"] == ["trigger_agent"]
    assert (
        "Mapped ADK AgentTool wrapper to Connic trigger_agent. "
        "Review calls to make sure they pass agent_name='escalation-specialist'."
    ) in report
    assert "Skipped ADK MCPToolset because it requires manual MCP server mapping in Connic." in report
    assert "Could not safely migrate dynamic tool expression 'build_dynamic_tool()'." in report


def test_adk_agent_tool_with_dynamic_target_still_maps_to_trigger_agent_with_review_note(tmp_path):
    source = tmp_path / "adk-dynamic-agent-tool"
    destination = tmp_path / "connic-app"
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import Agent
        from google.adk.tools import AgentTool


        def get_escalation_agent():
            return Agent(
                name="Escalation Specialist",
                model="gemini-2.0-flash",
                instruction="Handle escalations.",
            )


        triage_agent = Agent(
            name="Triage Agent",
            model="gemini-2.0-flash",
            instruction="Triage support requests.",
            tools=[
                AgentTool(get_escalation_agent()),
            ],
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    triage_yaml = yaml.safe_load((destination / "agents" / "triage-agent.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "adk"
    assert triage_yaml["tools"] == ["trigger_agent"]
    assert (
        "Mapped ADK AgentTool wrapper to Connic trigger_agent. "
        "Review calls to make sure they pass the intended agent_name."
    ) in report


def test_adk_llm_agent_with_sub_agents_gets_trigger_agent(tmp_path):
    source = tmp_path / "adk-router"
    destination = tmp_path / "connic-app"
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import Agent


        escalation_agent = Agent(
            name="Escalation Specialist",
            model="gemini-2.0-flash",
            instruction="Handle escalations.",
        )

        billing_agent = Agent(
            name="Billing Specialist",
            model="gemini-2.0-flash",
            instruction="Handle billing questions.",
        )

        root_agent = Agent(
            name="Support Router",
            model="gemini-2.0-flash",
            instruction="Route support requests to the right specialist.",
            sub_agents=[escalation_agent, billing_agent],
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    router_yaml = yaml.safe_load((destination / "agents" / "support-router.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "adk"
    assert router_yaml["type"] == "llm"
    assert router_yaml["tools"] == ["trigger_agent"]
    assert "Agent refs: escalation-specialist, billing-specialist" in report
    assert (
        "Mapped ADK sub_agents to Connic trigger_agent for delegation. "
        "Review the listed agent refs and update prompts as needed."
    ) in report


def test_adk_python_import_refs_distinguish_same_named_agents(tmp_path):
    source = tmp_path / "adk-module-router"
    destination = tmp_path / "connic-app"
    write(
        source / "billing" / "agents.py",
        '''
        from google.adk.agents import Agent


        support_agent = Agent(
            name="Support Agent",
            model="gemini-2.5-flash",
            instruction="Resolve billing questions.",
        )
        ''',
    )
    write(
        source / "technical" / "agents.py",
        '''
        from google.adk.agents import Agent


        support_agent = Agent(
            name="Support Agent",
            model="gemini-2.5-flash",
            instruction="Resolve technical questions.",
        )
        ''',
    )
    write(
        source / "root_agent.py",
        '''
        import billing.agents as billing_agents
        from google.adk.agents import Agent
        from technical.agents import support_agent as technical_support


        root_agent = Agent(
            name="Support Router",
            model="gemini-2.5-flash",
            instruction="Delegate each request to the right support specialist.",
            sub_agents=[billing_agents.support_agent, technical_support],
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    router = yaml.safe_load((destination / "agents" / "support-router.yaml").read_text())
    billing = yaml.safe_load((destination / "agents" / "support-agent.yaml").read_text())
    technical = yaml.safe_load((destination / "agents" / "support-agent-2.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "adk"
    assert router["tools"] == ["trigger_agent"]
    assert billing["system_prompt"] == "Resolve billing questions."
    assert technical["system_prompt"] == "Resolve technical questions."
    assert "Agent refs: support-agent, support-agent-2" in report


def test_adk_parallel_agent_is_migrated_as_reviewable_sequential_workflow(tmp_path):
    source = tmp_path / "adk-parallel"
    destination = tmp_path / "connic-app"
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import Agent, ParallelAgent


        inventory_agent = Agent(
            name="Inventory Check",
            model="gemini-2.0-flash",
            instruction="Check whether the product is available.",
        )

        pricing_agent = Agent(
            name="Pricing Check",
            model="gemini-2.0-flash",
            instruction="Check whether the price is still valid.",
        )

        checkout_agent = ParallelAgent(
            name="Checkout Review",
            sub_agents=[inventory_agent, pricing_agent],
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    workflow_yaml = yaml.safe_load((destination / "agents" / "checkout-review.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "adk"
    assert workflow_yaml == {
        "version": "1.0",
        "name": "checkout-review",
        "type": "sequential",
        "description": "Migrated from ADK agent 'Checkout Review'.",
        "agents": ["inventory-check", "pricing-check"],
    }
    assert "Original ADK agent used ParallelAgent; migrated as a sequential Connic agent for review." in report


def test_adk_migration_combines_python_and_yaml_agents(tmp_path):
    source = tmp_path / "mixed-adk-project"
    destination = tmp_path / "connic-app"
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import Agent


        root_agent = Agent(
            name="Python Agent",
            model="gemini-2.5-flash",
            instruction="Route requests to the right specialist.",
        )
        ''',
    )
    write(
        source / "support_agent.yaml",
        '''
        name: YAML Support
        model: claude-3-5-sonnet
        instruction: Resolve escalated support requests.
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    python_agent = yaml.safe_load((destination / "agents" / "python-agent.yaml").read_text())
    yaml_agent = yaml.safe_load((destination / "agents" / "yaml-support.yaml").read_text())

    assert framework == "adk"
    assert [agent.name for agent in agents] == ["python-agent", "yaml-support"]
    assert python_agent["system_prompt"] == "Route requests to the right specialist."
    assert yaml_agent["model"] == "anthropic/claude-3-5-sonnet"
    assert yaml_agent["system_prompt"] == "Resolve escalated support requests."


def test_adk_yaml_migration_reports_unresolved_tool_references(tmp_path):
    source = tmp_path / "adk-yaml"
    write(
        source / "root_agent.yaml",
        '''
        name: Policy Bot
        model: claude-3-5-sonnet
        instruction: Use approved support policy.
        tools:
          - query_knowledge
          - private_tool_name
        ''',
    )

    framework, detection_notes, agents, _module_infos = migrate._build_migration_candidates(source)

    assert framework == "adk"
    assert agents[0].name == "policy-bot"
    assert agents[0].model == "anthropic/claude-3-5-sonnet"
    assert [tool.ref for tool in agents[0].tool_candidates] == ["retrieval_query"]
    assert agents[0].notes == ["Could not migrate YAML tool reference 'private_tool_name'."]


def test_adk_yaml_llm_migration_links_canonical_child_configs(tmp_path):
    source = tmp_path / "adk-yaml-workflow"
    destination = tmp_path / "connic-app"
    write(
        source / "root_agent.py",
        '''
        from google.adk.agents import Agent


        triage_agent = Agent(
            name="Triage Agent",
            model="gemini-2.5-flash",
            instruction="Handle Python-defined triage requests.",
        )
        ''',
    )
    write(
        source / "root_agent.yaml",
        '''
        agent_class: LlmAgent
        name: Support Flow
        model: gemini-2.5-flash
        description: Route support tickets before escalation.
        instruction: Delegate each request to the right support specialist.
        sub_agents:
          - config_path: triage_agent.yaml
          - config_path: escalation_agent.yaml
        ''',
    )
    write(
        source / "triage_agent.yaml",
        '''
        name: Triage Agent
        model: gemini-2.5-flash
        instruction: Classify the ticket and decide whether escalation is needed.
        tools:
          - name: google_search
          - name: query_knowledge
        ''',
    )
    write(
        source / "escalation_agent.yaml",
        '''
        name: Escalation Agent
        model: claude-3-5-sonnet
        system_instruction: Resolve high-priority customer escalations.
        ''',
    )
    write(
        source / "settings.yaml",
        '''
        project: customer-support
        environment: staging
        ''',
    )
    write(
        source / "draft_agent.yaml",
        '''
        name: Draft Agent
        ''',
    )
    write(
        source / "dependencies.yaml",
        '''
        - google-adk
        - langchain
        ''',
    )
    (source / "broken_agent.yaml").write_text("name: [unterminated\n")

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    workflow_yaml = yaml.safe_load((destination / "agents" / "support-flow.yaml").read_text())
    python_triage_yaml = yaml.safe_load((destination / "agents" / "triage-agent.yaml").read_text())
    triage_yaml = yaml.safe_load((destination / "agents" / "triage-agent-2.yaml").read_text())
    escalation_yaml = yaml.safe_load((destination / "agents" / "escalation-agent.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "adk"
    assert workflow_yaml == {
        "version": "1.0",
        "name": "support-flow",
        "type": "llm",
        "description": "Route support tickets before escalation.",
        "model": "gemini/gemini-2.5-flash",
        "system_prompt": "Delegate each request to the right support specialist.",
        "tools": ["trigger_agent"],
    }
    assert python_triage_yaml["system_prompt"] == "Handle Python-defined triage requests."
    assert triage_yaml["model"] == "gemini/gemini-2.5-flash"
    assert triage_yaml["system_prompt"] == "Classify the ticket and decide whether escalation is needed."
    assert triage_yaml["tools"] == ["web_search", "retrieval_query"]
    assert escalation_yaml["model"] == "anthropic/claude-3-5-sonnet"
    assert escalation_yaml["system_prompt"] == "Resolve high-priority customer escalations."
    assert "- `support-flow` (llm)" in report
    assert "Agent refs: triage-agent-2, escalation-agent" in report


def test_adk_yaml_config_paths_distinguish_nested_same_stem_agents(tmp_path):
    source = tmp_path / "nested-adk-workflow"
    destination = tmp_path / "connic-app"
    write(
        source / "root_agent.yaml",
        '''
        agent_class: SequentialAgent
        name: Support Workflow
        sub_agents:
          - config_path: billing/support.yaml
          - config_path: technical/support.yaml
        ''',
    )
    write(
        source / "billing" / "support.yaml",
        '''
        name: Billing Support
        model: gemini-2.5-flash
        instruction: Resolve billing questions.
        ''',
    )
    write(
        source / "technical" / "support.yaml",
        '''
        name: Technical Support
        model: gemini-2.5-flash
        instruction: Resolve technical questions.
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    workflow = yaml.safe_load((destination / "agents" / "support-workflow.yaml").read_text())

    assert framework == "adk"
    assert workflow == {
        "version": "1.0",
        "name": "support-workflow",
        "type": "sequential",
        "description": "Migrated from ADK YAML agent 'Support Workflow'.",
        "agents": ["billing-support", "technical-support"],
    }


def test_adk_yaml_code_refs_distinguish_same_named_python_agents(tmp_path):
    source = tmp_path / "mixed-adk-workflow"
    destination = tmp_path / "connic-app"
    write(
        source / "billing" / "agents.py",
        '''
        from google.adk.agents import Agent


        support_agent = Agent(
            name="Support Agent",
            model="gemini-2.5-flash",
            instruction="Resolve billing questions.",
        )
        ''',
    )
    write(
        source / "technical" / "agents.py",
        '''
        from google.adk.agents import Agent


        support_agent = Agent(
            name="Support Agent",
            model="gemini-2.5-flash",
            instruction="Resolve technical questions.",
        )
        ''',
    )
    write(
        source / "root_agent.yaml",
        '''
        agent_class: LlmAgent
        name: Support Router
        model: gemini-2.5-flash
        instruction: Delegate each request to the right support specialist.
        sub_agents:
          - code: billing.agents.support_agent
          - code: technical.agents.support_agent
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    router = yaml.safe_load((destination / "agents" / "support-router.yaml").read_text())
    billing = yaml.safe_load((destination / "agents" / "support-agent.yaml").read_text())
    technical = yaml.safe_load((destination / "agents" / "support-agent-2.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "adk"
    assert router == {
        "version": "1.0",
        "name": "support-router",
        "type": "llm",
        "description": "Migrated from ADK YAML agent 'Support Router'.",
        "model": "gemini/gemini-2.5-flash",
        "system_prompt": "Delegate each request to the right support specialist.",
        "tools": ["trigger_agent"],
    }
    assert billing["system_prompt"] == "Resolve billing questions."
    assert technical["system_prompt"] == "Resolve technical questions."
    assert "Agent refs: support-agent, support-agent-2" in report


def test_adk_yaml_workflow_with_missing_agent_refs_becomes_review_placeholder(tmp_path):
    source = tmp_path / "adk-yaml-workflow"
    write(
        source / "root_agent.yaml",
        '''
        name: Escalation Flow
        sub_agents:
          - triage_agent
          - escalation_agent
        ''',
    )

    framework, _detection_notes, agents, _module_infos = migrate._build_migration_candidates(source)

    assert framework == "adk"
    assert len(agents) == 1
    assert agents[0].name == "escalation-flow"
    assert agents[0].agent_type == "llm"
    assert agents[0].agent_ref_keys == []
    assert "Removed unresolved workflow reference 'triage-agent'." in agents[0].notes
    assert "Removed unresolved workflow reference 'escalation-agent'." in agents[0].notes
    assert "Converted unsupported workflow to an LLM placeholder because no valid Connic agent chain was recovered." in agents[0].notes
    assert f"No model could be extracted. Defaulted to {migrate.MIGRATION_DEFAULT_MODEL}." in agents[0].notes
    assert "No static system prompt could be extracted. Added a placeholder prompt." in agents[0].notes


def test_migrate_command_reports_custom_adk_wrappers_when_no_explicit_agents_are_found(tmp_path):
    source = tmp_path / "custom-adk"
    destination = tmp_path / "connic-app"
    write(
        source / "custom_agents.py",
        '''
        from google.adk.agents import LlmAgent


        class TenantAwareAgent(LlmAgent):
            pass


        def register_agent(registry):
            registry.add(TenantAwareAgent(name="Tenant Support"))
        ''',
    )

    result = CliRunner().invoke(
        make_migrate_cli(),
        ["migrate", "--source", str(source), "--dest", str(destination)],
    )

    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert result.exit_code == 0
    assert "Framework: adk" in result.output
    assert "Detected dynamically registered custom ADK agent wrappers (TenantAwareAgent) in custom_agents.py" in result.output
    assert "No migratable agents were found. Generating an empty scaffold for manual migration." in result.output
    assert list((destination / "agents").iterdir()) == []
    assert (destination / "tools").is_dir()
    assert (destination / "middleware").is_dir()
    assert (destination / "schemas").is_dir()
    assert "- Generated agents: 0" in report
    assert "Detected dynamically registered custom ADK agent wrappers (TenantAwareAgent) in custom_agents.py" in report


def test_migrate_command_reports_custom_adk_wrappers_with_qualified_base_class(tmp_path):
    source = tmp_path / "qualified-custom-adk"
    destination = tmp_path / "connic-app"
    write(
        source / "custom_agents.py",
        '''
        import google.adk.agents


        class TenantAwareAgent(google.adk.agents.Agent):
            pass


        def register_agent(registry):
            agent = TenantAwareAgent(name="Tenant Support")
            registry.add(agent)
        ''',
    )

    result = CliRunner().invoke(
        make_migrate_cli(),
        ["migrate", "--source", str(source), "--dest", str(destination)],
    )

    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert result.exit_code == 0
    assert "Framework: adk" in result.output
    assert "Detected dynamically registered custom ADK agent wrappers (TenantAwareAgent) in custom_agents.py" in result.output
    assert "No migratable agents were found. Generating an empty scaffold for manual migration." in result.output
    assert "- Generated agents: 0" in report
    assert "Detected dynamically registered custom ADK agent wrappers (TenantAwareAgent) in custom_agents.py" in report


def test_langchain_migration_resolves_imported_prompt_model_and_tool_dependencies(tmp_path):
    source = tmp_path / "langchain-cross-file"
    destination = tmp_path / "connic-app"
    write(
        source / "prompts.py",
        '''
        CUSTOMER_TONE = "enterprise support"
        PROMPT_TEMPLATE = "You are an {tone} agent."


        def support_prompt():
            return PROMPT_TEMPLATE.format(tone=CUSTOMER_TONE)
        ''',
    )
    write(
        source / "order_ids.py",
        '''
        def normalize_order_id(order_id: str) -> str:
            return order_id.strip().upper()
        ''',
    )
    write(
        source / "tools" / "symbols.py",
        '''
        DEFAULT_CURRENCY = "EUR"
        ''',
    )
    write(
        source / "tools" / "rates.py",
        '''
        DEFAULT_TAX_RATE = 0.19
        ''',
    )
    write(
        source / "tools" / "currency.py",
        '''
        import tools.symbols
        import tools.rates


        CURRENCY = tools.symbols.DEFAULT_CURRENCY
        TAX_RATE = tools.rates.DEFAULT_TAX_RATE
        ''',
    )
    write(
        source / "tools" / "formatters.py",
        '''
        from tools.currency import CURRENCY, TAX_RATE


        def format_total(amount: float) -> dict:
            return {
                "amount": amount,
                "currency": CURRENCY,
                "tax": amount * TAX_RATE,
            }
        ''',
    )
    write(
        source / "tools" / "orders.py",
        '''
        import decimal
        import order_ids
        import tools.formatters as fmt


        def lookup_order(order_id: str) -> dict:
            normalized_order_id = order_ids.normalize_order_id(order_id)
            amount = decimal.Decimal(len(normalized_order_id)) * decimal.Decimal("10")
            return {
                "order_id": normalized_order_id,
                "total": fmt.format_total(float(amount)),
            }
        ''',
    )
    write(
        source / "agent.py",
        '''
        from os import getenv
        from langchain.agents import create_agent
        import prompts
        from tools.orders import lookup_order


        order_agent = create_agent(
            model=getenv("ORDER_MODEL", "gpt-4.1-mini"),
            tools=[lookup_order],
            prompt=prompts.support_prompt(),
            name="Order Support",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    report_notes = migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "order-support.yaml").read_text())
    order_tool = (destination / "tools" / "tools" / "orders.py").read_text()
    formatter_dependency = destination / "tools" / "tools" / "formatters.py"
    currency_dependency = destination / "tools" / "tools" / "currency.py"
    symbols_dependency = destination / "tools" / "tools" / "symbols.py"
    rates_dependency = destination / "tools" / "tools" / "rates.py"
    order_ids_dependency = destination / "tools" / "order_ids.py"

    assert framework == "langchain"
    assert agent_yaml["model"] == "openai/gpt-4.1-mini"
    assert agent_yaml["description"] == "You are an enterprise support agent."
    assert agent_yaml["system_prompt"] == "You are an enterprise support agent."
    assert agent_yaml["tools"] == ["tools.orders.lookup_order"]
    assert "import decimal" in order_tool
    assert "import tools.order_ids as order_ids" in order_tool
    assert "import tools.tools.formatters as fmt" in order_tool
    assert "def lookup_order" in order_tool
    assert formatter_dependency.exists()
    assert "from tools.tools.currency import CURRENCY" in formatter_dependency.read_text()
    assert "def format_total" in formatter_dependency.read_text()
    assert "import tools.tools.symbols" in currency_dependency.read_text()
    assert "import tools.tools.rates" in currency_dependency.read_text()
    assert "from tools import tools" in currency_dependency.read_text()
    assert symbols_dependency.read_text() == 'DEFAULT_CURRENCY = "EUR"\n'
    assert rates_dependency.read_text() == "DEFAULT_TAX_RATE = 0.19\n"
    assert "def normalize_order_id" in order_ids_dependency.read_text()
    assert any("No source requirements.txt found" in note for note in report_notes)

    loader = ProjectLoader(str(destination))
    migrated_agents = loader.load_agents()

    assert loader._load_errors == []
    assert len(migrated_agents) == 1
    assert migrated_agents[0].tools[0].func(" order-42 ") == {
        "order_id": "ORDER-42",
        "total": {"amount": 80.0, "currency": "EUR", "tax": 15.2},
    }


def test_langchain_migration_resolves_imported_module_alias_tool_and_config(tmp_path):
    source = tmp_path / "langchain-module-alias"
    destination = tmp_path / "connic-app"
    write(
        source / "settings.py",
        '''
        MODEL_NAME = "gpt-4.1-mini"
        BASE_PROMPT = "Use catalog search to answer shopper questions."


        def support_prompt():
            return BASE_PROMPT
        ''',
    )
    write(
        source / "support" / "catalog.py",
        '''
        def normalize_query(query: str) -> str:
            return query.strip().lower()


        def search_catalog(query: str) -> dict:
            return {"query": normalize_query(query), "matches": []}
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_react_agent
        import settings
        import support.catalog as catalog_tools


        shopper_agent = create_react_agent(
            model=settings.MODEL_NAME,
            tools=[catalog_tools.search_catalog],
            prompt=settings.support_prompt(),
            name="Shopper Support",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    report_notes = migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "shopper-support.yaml").read_text())
    migrated_tool = (destination / "tools" / "support" / "catalog.py").read_text()

    assert framework == "langchain"
    assert agent_yaml["model"] == "openai/gpt-4.1-mini"
    assert agent_yaml["system_prompt"] == "Use catalog search to answer shopper questions."
    assert agent_yaml["tools"] == ["support.catalog.search_catalog"]
    assert "def search_catalog" in migrated_tool
    assert "def normalize_query" in migrated_tool
    assert any("No source requirements.txt found" in note for note in report_notes)


def test_langchain_migration_preserves_package_tool_with_relative_dependency(tmp_path):
    source = tmp_path / "langchain-package"
    destination = tmp_path / "connic-app"
    write(
        source / "support" / "__init__.py",
        '''
        # Support package.
        ''',
    )
    write(
        source / "support" / "formatters.py",
        '''
        def format_status(order_id: str, status: str) -> dict:
            return {"order_id": order_id, "status": status}
        ''',
    )
    write(
        source / "support" / "tools.py",
        '''
        from .formatters import format_status


        def lookup_order_status(order_id: str) -> dict:
            return format_status(order_id, "processing")
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        from support.tools import lookup_order_status


        support_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[lookup_order_status],
            system_prompt="Use order tooling to answer customer support requests.",
            name="Support Agent",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    report_notes = migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "support-agent.yaml").read_text())
    migrated_tool = (destination / "tools" / "support" / "tools.py").read_text()
    migrated_dependency = destination / "tools" / "support" / "formatters.py"

    assert framework == "langchain"
    assert agent_yaml["tools"] == ["support.tools.lookup_order_status"]
    assert "from .formatters import format_status" in migrated_tool
    assert "def lookup_order_status" in migrated_tool
    assert migrated_dependency.exists()
    assert "def format_status" in migrated_dependency.read_text()
    assert any("Relative import 'from .formatters import format_status' may need manual review." in note for note in report_notes)


def test_langchain_migration_preserves_tool_defined_in_package_init(tmp_path):
    source = tmp_path / "langchain-package-init"
    destination = tmp_path / "connic-app"
    write(
        source / "support" / "__init__.py",
        '''
        POLICY_STATUS = "approved"


        def lookup_policy(policy_id: str) -> dict:
            return {"policy_id": policy_id, "status": POLICY_STATUS}
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        from support import lookup_policy


        policy_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[lookup_policy],
            system_prompt="Answer policy questions with approved internal policy data.",
            name="Policy Support",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    report_notes = migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "policy-support.yaml").read_text())
    migrated_tool = (destination / "tools" / "support" / "module.py").read_text()

    assert framework == "langchain"
    assert agent_yaml["tools"] == ["support.module.lookup_policy"]
    assert "POLICY_STATUS = \"approved\"" in migrated_tool
    assert "def lookup_policy" in migrated_tool
    assert not (destination / "tools" / "support" / "__init__.py").exists()
    assert any("No source requirements.txt found" in note for note in report_notes)


def test_langchain_migration_resolves_imported_prompt_callable_with_imported_format_values(tmp_path):
    source = tmp_path / "langchain-imported-prompt"
    destination = tmp_path / "connic-app"
    write(
        source / "settings.py",
        '''
        CUSTOMER_SEGMENT = "enterprise"
        REGION_COUNT = 3
        ''',
    )
    write(
        source / "prompts.py",
        '''
        from settings import CUSTOMER_SEGMENT, REGION_COUNT

        PROMPT_TEMPLATE = "Support {segment} customers across {regions} regions."


        def build_prompt():
            return PROMPT_TEMPLATE.format(segment=CUSTOMER_SEGMENT, regions=REGION_COUNT)
        ''',
    )
    write(
        source / "tools.py",
        '''
        def find_account(account_id: str) -> dict:
            return {"account_id": account_id}
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        from prompts import build_prompt
        from tools import find_account


        account_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[find_account],
            prompt=build_prompt(),
            name="Account Support",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "account-support.yaml").read_text())

    assert framework == "langchain"
    assert agent_yaml["description"] == "Support enterprise customers across 3 regions."
    assert agent_yaml["system_prompt"] == "Support enterprise customers across 3 regions."
    assert agent_yaml["tools"] == ["tools.find_account"]


def test_langchain_migration_resolves_factory_model_and_module_prompt_callable(tmp_path):
    source = tmp_path / "langchain-factories"
    destination = tmp_path / "connic-app"
    write(
        source / "prompts.py",
        '''
        def support_prompt():
            return "Resolve account issues with policy and billing context."
        ''',
    )
    write(
        source / "tools.py",
        '''
        def find_customer(email: str) -> dict:
            return {"email": email, "status": "active"}
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        import prompts as prompt_library
        from tools import find_customer


        def default_model():
            return "gpt-4.1-mini"


        account_agent = create_agent(
            model=default_model(),
            tools=[find_customer],
            prompt=prompt_library.support_prompt(),
            name="Account Resolution",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "account-resolution.yaml").read_text())

    assert framework == "langchain"
    assert agent_yaml["model"] == "openai/gpt-4.1-mini"
    assert agent_yaml["description"] == "Resolve account issues with policy and billing context."
    assert agent_yaml["system_prompt"] == "Resolve account issues with policy and billing context."
    assert agent_yaml["tools"] == ["tools.find_customer"]


def test_langchain_migration_reports_dynamic_prompt_that_cannot_be_statically_resolved(tmp_path):
    source = tmp_path / "langchain-dynamic-prompt"
    destination = tmp_path / "connic-app"
    write(
        source / "tools.py",
        '''
        def lookup_account(account_id: str) -> dict:
            return {"account_id": account_id, "tier": "enterprise"}
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        from tools import lookup_account


        tenant_name = load_tenant_name_from_runtime_config()

        account_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[lookup_account],
            system_prompt=f"Resolve account issues for {tenant_name}.",
            name="Account Runtime Support",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "account-runtime-support.yaml").read_text())
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "langchain"
    assert agent_yaml["description"] == "Migrated from LangChain agent 'Account Runtime Support'."
    assert agent_yaml["system_prompt"] == migrate.MIGRATION_DEFAULT_PROMPT
    assert agent_yaml["tools"] == ["tools.lookup_account"]
    assert "No static system prompt could be extracted. Added a placeholder prompt." in report


def test_langgraph_react_agent_migration_extracts_decorated_tool_function(tmp_path):
    source = tmp_path / "langgraph-react"
    destination = tmp_path / "connic-app"
    write(
        source / "agent.py",
        '''
        from langchain.chat_models import init_chat_model
        import langchain_core.tools as lc_tools
        from langgraph.prebuilt import create_react_agent


        @lc_tools.tool
        def fetch_weather(city: str) -> str:
            """Return a weather summary for a city."""
            return f"Weather for {city}: clear"


        research_agent = create_react_agent(
            init_chat_model("gpt-4.1-mini"),
            [fetch_weather],
            prompt="Answer travel planning questions with weather context.",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    agent_yaml = yaml.safe_load((destination / "agents" / "research-agent.yaml").read_text())
    tool_module = (destination / "tools" / "agent.py").read_text()

    assert framework == "langchain"
    assert detection_notes == ["Detected LangChain/LangGraph/LangSmith imports."]
    assert agent_yaml["model"] == "openai/gpt-4.1-mini"
    assert agent_yaml["system_prompt"] == "Answer travel planning questions with weather context."
    assert agent_yaml["tools"] == ["agent.fetch_weather"]
    assert "@lc_tools.tool" not in tool_module
    assert "import langchain_core.tools as lc_tools" not in tool_module
    assert "def fetch_weather" in tool_module


def test_langchain_migration_preserves_decorated_tool_dependencies(tmp_path):
    source = tmp_path / "langchain-billing"
    destination = tmp_path / "connic-app"
    write(
        source / "invoice_tools.py",
        '''
        from dataclasses import dataclass
        from functools import lru_cache
        from langchain_core.tools import tool as langchain_tool


        @dataclass(frozen=True)
        class Invoice:
            invoice_id: str
            status: str = "paid"


        @lru_cache(maxsize=32)
        def normalize_invoice_id(invoice_id: str) -> str:
            return invoice_id.strip().lower()


        @langchain_tool
        def lookup_invoice(invoice_id: str) -> dict:
            invoice = Invoice(normalize_invoice_id(invoice_id))
            return {"invoice_id": invoice.invoice_id, "status": invoice.status}
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        from invoice_tools import lookup_invoice


        billing_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[lookup_invoice],
            system_prompt="Resolve billing questions with invoice context.",
            name="Billing Support",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    tool_module = (destination / "tools" / "invoice_tools.py").read_text()
    loader = ProjectLoader(str(destination))
    migrated_agents = loader.load_agents()

    assert "from dataclasses import dataclass" in tool_module
    assert "@dataclass(frozen=True)" in tool_module
    assert "from functools import lru_cache" in tool_module
    assert "@lru_cache(maxsize=32)" in tool_module
    assert "@langchain_tool" not in tool_module
    assert "from langchain_core.tools import tool as langchain_tool" not in tool_module
    assert loader._load_errors == []
    assert migrated_agents[0].tools[0].func(" inv-123 ") == {
        "invoice_id": "inv-123",
        "status": "paid",
    }


def test_langchain_migration_preserves_imports_for_annotated_tool_dependencies(tmp_path):
    source = tmp_path / "langchain-pricing"
    destination = tmp_path / "connic-app"
    write(
        source / "pricing_tools.py",
        '''
        from dataclasses import dataclass
        from typing import Final

        from langchain.tools import tool


        DEFAULT_CURRENCY: Final[str] = "EUR"
        DEFAULT_TAX_RATE: Final[float] = 0.19


        @dataclass(frozen=True)
        class PricingSettings:
            currency: str
            tax_rate: float


        SETTINGS: PricingSettings = PricingSettings(
            currency=DEFAULT_CURRENCY,
            tax_rate=DEFAULT_TAX_RATE,
        )


        @tool
        def quote_order(amount: float) -> dict:
            """Quote an order total using the configured currency and tax rate."""
            return {
                "amount": amount,
                "currency": SETTINGS.currency,
                "tax": amount * SETTINGS.tax_rate,
            }
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        from pricing_tools import quote_order


        pricing_agent = create_agent(
            model="openai:gpt-5.2",
            tools=[quote_order],
            system_prompt="Quote customer orders with the configured pricing policy.",
            name="Pricing Agent",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    tool_module = (destination / "tools" / "pricing_tools.py").read_text()
    loader = ProjectLoader(str(destination))
    migrated_agents = loader.load_agents()

    assert "from typing import Final" in tool_module
    assert 'DEFAULT_CURRENCY: Final[str] = "EUR"' in tool_module
    assert "SETTINGS: PricingSettings = PricingSettings(" in tool_module
    assert loader._load_errors == []
    assert migrated_agents[0].tools[0].func(100.0) == {
        "amount": 100.0,
        "currency": "EUR",
        "tax": 19.0,
    }


def test_langchain_migration_preserves_future_annotations_for_forward_references(tmp_path):
    source = tmp_path / "langchain-orders"
    destination = tmp_path / "connic-app"
    write(
        source / "order_tools.py",
        '''
        from __future__ import annotations

        from langchain.tools import tool


        @tool
        def describe_order(order: Order) -> dict:
            """Return the customer-facing identifier for an order."""
            return {"order_id": order.order_id}


        class Order:
            def __init__(self, order_id: str):
                self.order_id = order_id
        ''',
    )
    write(
        source / "agent.py",
        '''
        from langchain.agents import create_agent
        from order_tools import describe_order


        order_agent = create_agent(
            model="openai:gpt-5.2",
            tools=[describe_order],
            system_prompt="Answer order questions using the order tools.",
            name="Order Agent",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    tool_module = (destination / "tools" / "order_tools.py").read_text()
    loader = ProjectLoader(str(destination))
    migrated_agents = loader.load_agents()

    assert tool_module.startswith("from __future__ import annotations\n")
    assert loader._load_errors == []
    tool = migrated_agents[0].tools[0]
    order_type = tool.func.__globals__["Order"]
    assert tool.func(order_type("order-42")) == {"order_id": "order-42"}


def test_langchain_supervisor_agent_wrappers_migrate_to_connic_delegation(tmp_path, monkeypatch):
    source = tmp_path / "official-supervisor-agent"
    destination = tmp_path / "connic-app"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    write(
        source / "assistant.py",
        '''
        from __future__ import annotations

        from langchain.agents import create_agent
        from langchain.tools import tool


        @tool
        def create_calendar_event(title: str, start_time: str, end_time: str) -> str:
            """Create a calendar event."""
            return f"Event created: {title} from {start_time} to {end_time}"


        @tool
        def get_available_time_slots(date: str, duration_minutes: int) -> list[str]:
            """Check calendar availability for a given date."""
            return ["09:00", "14:00", "16:00"]


        @tool
        def send_email(to: str, subject: str, body: str) -> str:
            """Send an email notification."""
            return f"Email sent to {to} with subject '{subject}'"


        calendar_agent = create_agent(
            model="openai:gpt-5.2",
            tools=[create_calendar_event, get_available_time_slots],
            system_prompt=(
                "You are a calendar scheduling assistant. "
                "Parse scheduling requests and use the calendar tools to fulfill them."
            ),
        )


        email_agent = create_agent(
            model="openai:gpt-5.2",
            tools=[send_email],
            system_prompt=(
                "You are an email assistant. Compose professional emails and send them when needed."
            ),
        )


        @tool
        def schedule_event(request: str) -> str:
            """Schedule calendar events using natural language."""
            result = calendar_agent.invoke(
                {"messages": [{"role": "user", "content": request}]}
            )
            return result["messages"][-1].text


        @tool
        def manage_email(request: str) -> str:
            """Send emails using natural language."""
            result = email_agent.invoke(
                {"messages": [{"role": "user", "content": request}]}
            )
            return result["messages"][-1].text


        supervisor_agent = create_agent(
            model="openai:gpt-5.2",
            tools=[schedule_event, manage_email],
            system_prompt=(
                "You are a helpful personal assistant. "
                "Break down user requests into scheduling and email actions as needed."
            ),
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    calendar_agent = yaml.safe_load((destination / "agents" / "calendar-agent.yaml").read_text())
    email_agent = yaml.safe_load((destination / "agents" / "email-agent.yaml").read_text())
    supervisor_agent = yaml.safe_load((destination / "agents" / "supervisor-agent.yaml").read_text())
    tool_module = (destination / "tools" / "assistant.py").read_text()
    report = (destination / "MIGRATION_REPORT.md").read_text()

    assert framework == "langchain"
    assert calendar_agent["tools"] == [
        "assistant.create_calendar_event",
        "assistant.get_available_time_slots",
    ]
    assert email_agent["tools"] == ["assistant.send_email"]
    assert supervisor_agent["tools"] == [
        "assistant.schedule_event",
        "assistant.manage_email",
    ]
    assert "async def schedule_event" in tool_module
    assert "async def manage_email" in tool_module
    assert tool_module.index("from __future__ import annotations") < tool_module.index(
        "from connic.tools import trigger_agent"
    )
    assert "langchain" not in tool_module
    assert "create_agent" not in tool_module
    assert "calendar_agent =" not in tool_module
    assert "email_agent =" not in tool_module
    assert ".invoke(" not in tool_module
    assert "Rewrote LangChain sub-agent tool wrapper 'schedule_event'" in report
    assert "Rewrote LangChain sub-agent tool wrapper 'manage_email'" in report
    assert _run_lint(quiet=True, project_root=str(destination)) is True

    calls = []

    async def fake_trigger_agent(agent_name, payload, **_kwargs):
        calls.append((agent_name, payload))
        return {"response": f"{agent_name}: {payload}"}

    monkeypatch.setattr("connic.tools.trigger_agent", fake_trigger_agent)
    loader = ProjectLoader(str(destination))
    migrated_agents = {agent.config.name: agent for agent in loader.load_agents()}
    supervisor_tools = {tool.name: tool for tool in migrated_agents["supervisor-agent"].tools}

    assert loader._load_errors == []
    assert set(migrated_agents) == {"calendar-agent", "email-agent", "supervisor-agent"}
    assert set(supervisor_tools) == {"schedule_event", "manage_email"}
    assert all(tool.is_async for tool in supervisor_tools.values())
    assert supervisor_tools["schedule_event"].description == "Schedule calendar events using natural language."
    assert supervisor_tools["manage_email"].description == "Send emails using natural language."
    assert supervisor_tools["schedule_event"].execute_sync(request="Book Tuesday at 14:00") == (
        "calendar-agent: Book Tuesday at 14:00"
    )
    assert supervisor_tools["manage_email"].execute_sync(request="Email the attendee") == (
        "email-agent: Email the attendee"
    )
    assert calls == [
        ("calendar-agent", "Book Tuesday at 14:00"),
        ("email-agent", "Email the attendee"),
    ]


def test_langchain_async_subagent_wrapper_preserves_direct_payload(tmp_path, monkeypatch):
    source = tmp_path / "langchain-async-supervisor"
    destination = tmp_path / "connic-app"
    write(
        source / "assistant.py",
        '''
        from langchain.agents import create_agent


        calendar_agent = create_agent(
            model="openai:gpt-5.2",
            tools=[],
            system_prompt="Schedule calendar events.",
        )


        async def delegate_to_calendar(request: str) -> str:
            response: dict = await calendar_agent.ainvoke(request)
            return response["messages"][-1].content


        supervisor_agent = create_agent(
            model="openai:gpt-5.2",
            tools=[delegate_to_calendar],
            system_prompt="Delegate scheduling requests.",
        )
        ''',
    )

    framework, detection_notes, agents, module_infos = migrate._build_migration_candidates(source)
    migrate._generate_migrated_project(
        source,
        destination,
        framework,
        detection_notes,
        agents,
        module_infos,
        no_scaffold,
    )

    calls = []

    async def fake_trigger_agent(agent_name, payload, **_kwargs):
        calls.append((agent_name, payload))
        return {"response": "scheduled"}

    monkeypatch.setattr("connic.tools.trigger_agent", fake_trigger_agent)
    loader = ProjectLoader(str(destination))
    supervisor = loader.load_agent("supervisor-agent")
    tool = supervisor.tools[0]

    assert loader._load_errors == []
    assert tool.name == "delegate_to_calendar"
    assert tool.is_async is True
    assert tool.execute_sync(request="Book Friday at 09:00") == "scheduled"
    assert calls == [("calendar-agent", "Book Friday at 09:00")]


@pytest.mark.parametrize(
    "wrapper_source",
    [
        '''
        def audited(func):
            return func

        @audited
        def delegate(request: str) -> str:
            result = child_agent.invoke(request)
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            request = request.strip()
            result = child_agent.invoke(request)
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            result, metadata = child_agent.invoke(request)
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.stream(request)
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.invoke(request)
            return result
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.invoke(request)
            return result.text
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.invoke(request)
            return result["messages"][0].text
        ''',
        '''
        def delegate(request: str) -> str:
            result = external_agent.invoke(request)
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.invoke(request, config={})
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.invoke({"input": request})
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.invoke({"messages": {"role": "user", "content": request}})
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.invoke({"messages": [request]})
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.invoke({"messages": [{"role": "assistant", "content": request}]})
            return result["messages"][-1].text
        ''',
        '''
        def delegate(request: str) -> str:
            result = child_agent.invoke({"messages": [{"role": "user", "content": f"Task: {request}"}]})
            return result["messages"][-1].text
        ''',
    ],
)
def test_langchain_subagent_wrapper_rewrite_rejects_shapes_that_change_delegation_semantics(
    tmp_path,
    wrapper_source,
):
    source_file = tmp_path / "assistant.py"
    write(
        source_file,
        '''
        from langchain.agents import create_agent


        child_agent = create_agent(
            model="openai:gpt-5.2",
            tools=[],
            system_prompt="Handle delegated requests.",
        )
        '''
        + wrapper_source,
    )

    module_info = migrate._parse_module_info(source_file)

    assert module_info is not None
    assert migrate._match_langchain_agent_wrapper(module_info.functions["delegate"], module_info) is None


def test_collect_python_files_skips_venv_and_pycache(tmp_path):
    (tmp_path / "app").mkdir(parents=True, exist_ok=True)
    (tmp_path / "app" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".venv" / "lib" / "site.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".venv" / "lib" / "site.py").write_text("# venv\n", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__" / "x.cpython-312.pyc").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pkg" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")

    files = migrate._collect_python_files(tmp_path)

    assert files == [tmp_path / "app" / "main.py"]


def test_parse_module_info_returns_none_on_syntax_error(tmp_path):
    bad = tmp_path / "broken.py"
    bad.write_text("def foo(\n", encoding="utf-8")
    assert migrate._parse_module_info(bad) is None


def test_normalize_model_name_covers_provider_prefixes():
    assert migrate._normalize_model_name("  openai/gpt-4o-mini  ") == "openai/gpt-4o-mini"
    assert migrate._normalize_model_name(None) is None
    assert migrate._normalize_model_name("   ") is None
    assert migrate._normalize_model_name("openai:gpt-4o") == "openai/gpt-4o"
    assert migrate._normalize_model_name("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert migrate._normalize_model_name("claude-3-5-sonnet") == "anthropic/claude-3-5-sonnet"
    assert migrate._normalize_model_name("gemini-2.0-flash") == "gemini/gemini-2.0-flash"
    assert migrate._normalize_model_name("custom-raw-id") == "custom-raw-id"
    assert migrate._normalize_model_name("openai") == "openai"
    assert migrate._normalize_model_name("anthropic") == "anthropic"
    assert migrate._normalize_model_name("openrouter/mistral/medium") == "openrouter/mistral/medium"


def test_sanitize_agent_name_fixes_invalid_edges():
    assert migrate._sanitize_agent_name("Support Bot!", "fallback") == "support-bot"
    assert migrate._sanitize_agent_name("!!!", "fb") == "fb"
    assert migrate._sanitize_agent_name(None, "default-name") == "default-name"


def test_detect_framework_unknown_for_plain_python(tmp_path):
    (tmp_path / "script.py").write_text("# no frameworks\n", encoding="utf-8")
    py_files = migrate._collect_python_files(tmp_path)
    yaml_files = migrate._collect_yaml_files(tmp_path)
    fw, notes = migrate._detect_framework(tmp_path, py_files, yaml_files)
    assert fw == "unknown"
    assert any("Could not confidently detect" in n for n in notes)


def test_detect_framework_prefers_langchain_when_both_signals(tmp_path):
    (tmp_path / "mixed.py").write_text(
        "import langchain\nfrom google.adk.agents import Agent\n",
        encoding="utf-8",
    )
    py_files = migrate._collect_python_files(tmp_path)
    fw, notes = migrate._detect_framework(tmp_path, py_files, [])
    assert fw == "langchain"
    assert any("both ADK and LangChain" in n for n in notes)


def test_dedupe_agent_names_adds_suffix_when_colliding(tmp_path):
    a = migrate.AgentCandidate(
        source_id="1",
        framework="lc",
        source_file=None,
        name="Same Name",
        agent_type="llm",
    )
    b = migrate.AgentCandidate(
        source_id="2",
        framework="lc",
        source_file=None,
        name="Same Name",
        agent_type="llm",
    )
    migrate._dedupe_agent_names([a, b])
    assert a.name != b.name
    assert a.name == "same-name"
    assert b.name == "same-name-2"


def test_is_hidden_or_skipped_path_outside_root_uses_parts_fallback(tmp_path):
    outside = tmp_path / "outside" / "x.py"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("y = 2\n", encoding="utf-8")
    assert migrate._is_hidden_or_skipped(outside, tmp_path / "nested" / "root") is False


def test_build_module_name_variants_strips_src_and_init(tmp_path):
    root = tmp_path / "repo"
    mod = root / "src" / "billing" / "calc" / "__init__.py"
    mod.parent.mkdir(parents=True, exist_ok=True)
    mod.write_text("# package\n", encoding="utf-8")
    variants = migrate._build_module_name_variants(root, mod)
    assert "src.billing.calc" in variants
    assert "billing.calc" in variants


def test_collect_yaml_files_skips_dependency_vendor_trees(tmp_path):
    tracked = tmp_path / "config" / "app.yaml"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text("a: 1\n", encoding="utf-8")
    vendor = tmp_path / "node_modules" / "some-lib" / "config.yml"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    vendor.write_text("lib: true\n", encoding="utf-8")

    files = migrate._collect_yaml_files(tmp_path)

    assert files == [tracked]
