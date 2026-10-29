from textwrap import dedent

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from connic import cli
from connic.app_schema import AppFile, is_app_source
from connic.connector_schema import ConnectorFile
from connic.loader import ProjectLoader


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n")


def write_agent(project):
    write_file(
        project / "agents" / "assistant.yaml",
        """
        version: "1.0"
        name: assistant
        description: Answers questions
        model: openai/gpt-4o
        system_prompt: Help the user.
        """,
    )


def write_connector(project, key="support-chat", apps_only=True, environments=None):
    lines = ["type: websocket", f"apps_only: {str(apps_only).lower()}", "agents: [assistant]"]
    if environments:
        lines.append("environments:")
        lines.extend(f"  {environment}: {{}}" for environment in environments)
    write_file(project / "connectors" / f"{key}.yaml", "\n".join(lines))


def write_app_source(project, key="portal", lockfile="package-lock.json"):
    write_file(project / "apps" / key / "package.json", '{"name": "portal", "scripts": {"build": "vite build"}}')
    if lockfile:
        write_file(project / "apps" / key / lockfile, "{}")


def app(**overrides):
    data = {
        "key": "portal",
        "build": {"command": "npm run build", "output": "dist"},
        "auth": {"mode": "public"},
        **overrides,
    }
    return AppFile.model_validate(data)


def load_apps(project, **kwargs):
    loader = ProjectLoader(str(project), validation_only=True)
    agents = loader.load_agents()
    connectors = loader.load_connectors(agents)
    return loader.load_apps(connectors, **kwargs), loader._load_errors


def test_app_defaults_and_environment_overrides_replace_whole_values():
    parsed = app(
        name="Support portal",
        auth={
            "mode": "oidc",
            "issuer": "https://login.example.com",
            "client_id": "${PORTAL_CLIENT_ID}",
            "client_secret": "${PORTAL_CLIENT_SECRET}",
            "allowed_domains": ["example.com"],
        },
        connectors=["support-chat"],
        environments={"Staging": {"auth": {"mode": "connic", "groups": ["Admin"]}, "connectors": []}, "Production": None},
    )

    assert parsed.build.node == "24"
    staging = parsed.for_environment("Staging")
    assert staging["auth"].mode == "connic"
    assert staging["auth"].groups == ["Admin"]
    assert staging["connectors"] == []
    production = parsed.for_environment("Production")
    assert production["name"] == "Support portal"
    assert production["auth"].mode == "oidc"
    assert production["connectors"] == ["support-chat"]
    assert parsed.for_environment("Development") is None


def test_auth_is_optional_and_can_be_set_per_environment():
    parsed = AppFile.model_validate({
        "key": "portal",
        "build": {"command": "npm run build", "output": "dist"},
        "environments": {"Staging": {"auth": {"mode": "connic"}}, "Production": {}},
    })

    assert parsed.auth is None
    assert parsed.for_environment("Staging")["auth"].mode == "connic"
    assert parsed.for_environment("Production")["auth"] is None


def test_null_override_removes_domain():
    parsed = app(domain="portal.example.com", environments={"Staging": {"domain": None}})

    assert parsed.for_environment("Staging")["domain"] is None


@pytest.mark.parametrize(
    "auth, message",
    [
        ({"mode": "public", "issuer": "https://login.example.com"}, "auth.issuer doesn't apply to public apps"),
        ({"mode": "connic", "client_id": "abc"}, "auth.client_id doesn't apply to connic apps"),
        ({"mode": "oidc", "client_id": "abc", "client_secret": "${SECRET}"}, "auth.issuer is required for oidc apps"),
        (
            {"mode": "oidc", "issuer": "https://login.example.com", "client_id": "abc", "client_secret": "plain"},
            "auth.client_secret is a secret",
        ),
        (
            {"mode": "oidc", "issuer": "http://login.example.com", "client_id": "abc", "client_secret": "${SECRET}"},
            "auth.issuer must be an https:// URL",
        ),
        (
            {
                "mode": "oidc", "issuer": "https://login.example.com", "client_id": "abc", "client_secret": "${SECRET}",
                "allowed_domains": ["user@example.com"],
            },
            "auth.allowed_domains must list email domains",
        ),
        ({"mode": "oidc", "issuer": "https://login.example.com", "client_id": "${1x}", "client_secret": "${S}"}, "not a variable reference"),
    ],
)
def test_auth_settings_are_checked_per_mode(auth, message):
    with pytest.raises(ValidationError, match=message):
        app(auth=auth)


@pytest.mark.parametrize(
    "build, message",
    [
        ({"command": "npm run build", "output": "../dist"}, "build.output must be a folder inside the app"),
        ({"command": "npm run build", "output": "/dist"}, "build.output must be a folder inside the app"),
        ({"command": "npm run build", "output": "."}, "build.output must be a folder inside the app"),
        ({"command": "npm run build\nrm -rf /", "output": "dist"}, "build commands must be a single line"),
        ({"command": "npm run build -- ${MODE}", "output": "dist"}, "build commands can't use"),
        ({"command": "npm run build", "output": "dist", "node": "18"}, "node"),
    ],
)
def test_build_settings_are_checked(build, message):
    with pytest.raises(ValidationError, match=message):
        app(build=build)


def test_node_version_accepts_numbers_and_output_is_normalized():
    parsed = app(build={"command": "npm run build", "output": "./build/client/", "node": 22})

    assert parsed.build.node == "22"
    assert parsed.build.output == "build/client"


def test_connectors_must_be_unique_keys():
    with pytest.raises(ValidationError, match="connector 'chat' is listed twice"):
        app(connectors=["chat", "chat"])


def test_lookups_and_access_produce_the_spec_shape_with_bools_as_rules():
    parsed = app(
        lookups={"member": {"collection": "members", "where": "doc.email == user.email"}},
        access={
            "database": {
                "tickets": {
                    "read": "doc.author_id == user.id or member.role == 'agent'",
                    "insert": True,
                    "update": "doc.author_id == user.id and new.status in ['open', 'closed']",
                    "delete": False,
                    "set": {"author_id": "user.id", "team": "member.team"},
                },
            },
            "files": {"/reports/2026/": {"read": "'support' in user.groups", "write": "new.size < 1000000"}},
            "runs": {"read": "run.context.end_user.id == user.id", "cancel": "run.status == 'running'"},
            "sessions": {"read": "session.state.owner == user.id", "delete": False},
            "approvals": {"read": True, "decide": "approval.params.amount < 100 and 'finance' in user.groups"},
            "retrieval": {"help": {"search": True}, "docs.internal": {"search": "member.role == 'agent'"}},
        },
    )

    spec = parsed.for_environment("Production")
    assert {name: lookup.model_dump() for name, lookup in spec["lookups"].items()} == {
        "member": {"collection": "members", "where": "doc.email == user.email"},
    }
    assert spec["access"].model_dump(exclude_none=True) == {
        "database": {
            "tickets": {
                "read": "doc.author_id == user.id or member.role == 'agent'",
                "insert": "true",
                "update": "doc.author_id == user.id and new.status in ['open', 'closed']",
                "delete": "false",
                "set": {"author_id": "user.id", "team": "member.team"},
            },
        },
        "files": {"reports/2026": {"read": "'support' in user.groups", "write": "new.size < 1000000"}},
        "runs": {"read": "run.context.end_user.id == user.id", "cancel": "run.status == 'running'"},
        "sessions": {"read": "session.state.owner == user.id", "delete": "false"},
        "approvals": {"read": "true", "decide": "approval.params.amount < 100 and 'finance' in user.groups"},
        "retrieval": {"help": {"search": "true"}, "docs.internal": {"search": "member.role == 'agent'"}},
    }


def test_app_without_lookups_or_access_has_none_in_its_spec():
    spec = app().for_environment("Production")

    assert spec["lookups"] is None
    assert spec["access"] is None


@pytest.mark.parametrize(
    "access, message",
    [
        ({"database": {"tickets": {"read": "docs.author_id == user.id"}}}, "access.database.tickets.read: unknown name 'docs'"),
        (
            {"database": {"tickets": {"insert": "doc.author_id == user.id"}}},
            "access.database.tickets.insert: 'doc' isn't available here; use new, user",
        ),
        ({"database": {"tickets": {"read": "new.status == 'open'"}}}, "access.database.tickets.read: 'new' isn't available here"),
        ({"database": {"tickets": {"delete": "new.id"}}}, "access.database.tickets.delete: 'new' isn't available"),
        ({"database": {"tickets": {"set": {"author_id": "doc.id"}}}}, "access.database.tickets.set.author_id: 'doc' isn't available here; use user"),
        ({"files": {"reports": {"read": "doc.path"}}}, "access.files.reports.read: 'doc' isn't available here; use file, user"),
        ({"files": {"reports": {"write": "file.size < 10"}}}, "access.files.reports.write: 'file' isn't available here; use new, user"),
        ({"runs": {"read": "approval.id"}}, "access.runs.read: 'approval' isn't available here; use run, user"),
        ({"sessions": {"delete": "run.id"}}, "access.sessions.delete: 'run' isn't available here; use session, user"),
        ({"approvals": {"decide": "session.id"}}, "access.approvals.decide: 'session' isn't available here; use approval, user"),
        ({"retrieval": {"help": {"search": "doc.public"}}}, "access.retrieval.help.search: 'doc' isn't available here; use user"),
        ({"runs": {"read": "context.user_id == user.id"}}, "access.runs.read: unknown name 'context'"),
        ({"database": {"tickets": {"read": "member.role == 'agent'"}}}, "access.database.tickets.read: unknown name 'member'"),
    ],
)
def test_rules_may_only_use_the_names_of_their_resource_and_operation(access, message):
    with pytest.raises(ValidationError, match=message):
        app(access=access)


@pytest.mark.parametrize(
    "rule, message",
    [
        ("doc.author_id ==", "access.database.tickets.read: invalid syntax"),
        ("(doc.author_id == user.id", r"access.database.tickets.read: '\(' was never closed"),
        ("len(doc.tags) > 1", "access.database.tickets.read: unsupported expression element Call"),
        ("doc.count + 1 > 2", "access.database.tickets.read: unsupported expression element BinOp"),
        ("doc.count > -1", "access.database.tickets.read: unsupported expression element USub"),
        ("[t for t in doc.tags]", "access.database.tickets.read: unsupported expression element ListComp"),
        ("doc.__class__ == user", "access.database.tickets.read: private attribute '__class__' isn't allowed"),
        (" or ".join(["doc.a == 1"] * 60), "access.database.tickets.read: rule is too complex"),
        ("doc.a == '" + "x" * 2000 + "'", "access.database.tickets.read: rules are limited to 2000 characters"),
        ("doc.owner is user.id", "access.database.tickets.read: use is and is not only with null, true or false"),
        ("doc.status is not 'open'", "access.database.tickets.read: use is and is not only with null, true or false"),
        ("1 < doc.n is 2", "access.database.tickets.read: use is and is not only with null, true or false"),
    ],
)
def test_rules_follow_connics_condition_grammar(rule, message):
    with pytest.raises(ValidationError, match=message):
        app(access={"database": {"tickets": {"read": rule}}})


def test_rules_accept_literals_paths_and_membership():
    parsed = app(access={"database": {"tickets": {
        "read": "doc.status in ('open', 'pending') and not doc.archived and doc['priority'] >= 2 or doc.owner is null",
        "update": "new.tags[0:2] != [] and new.meta == {'a': 1} and doc.flag == True",
        "delete": "doc.done is true and doc.archived is not False and null is doc.owner",
    }}})

    assert parsed.access.database["tickets"].update.startswith("new.tags")


@pytest.mark.parametrize("value", [1, ["doc.id"], {"read": "true"}])
def test_rules_must_be_expressions_or_bools(value):
    with pytest.raises(ValidationError, match="valid string"):
        app(access={"database": {"tickets": {"read": value}}})


@pytest.mark.parametrize(
    "access, message",
    [
        ({"database": {"tickets": {"write": "true"}}}, "database.tickets.write"),
        ({"files": {"reports": {"insert": "true"}}}, "files.reports.insert"),
        ({"runs": {"delete": "true"}}, "runs.delete"),
        ({"sessions": {"cancel": "true"}}, "sessions.cancel"),
        ({"approvals": {"update": "true"}}, "approvals.update"),
        ({"retrieval": {"help": {"read": "true"}}}, "retrieval.help.read"),
        ({"search": {}}, "search"),
    ],
)
def test_unknown_resources_and_operations_are_rejected(access, message):
    with pytest.raises(ValidationError, match=message):
        app(access=access)


def test_files_folders_are_normalized_like_agent_file_access():
    parsed = app(access={"files": {"/reports/": {"read": True}, "exports/2026/q1": {"read": True}}})

    assert list(parsed.access.files) == ["reports", "exports/2026/q1"]


@pytest.mark.parametrize("folder", ["/", "", "reports/../secrets", "reports//2026", "./reports"])
def test_invalid_files_folders_are_rejected(folder):
    with pytest.raises(ValidationError, match="Invalid files folder"):
        app(access={"files": {folder: {"read": True}}})


@pytest.mark.parametrize(
    "lookups, message",
    [
        ({"doc": {"collection": "members", "where": "true"}}, "lookups: 'doc' is reserved"),
        ({"user": {"collection": "members", "where": "true"}}, "lookups: 'user' is reserved"),
        ({"null": {"collection": "members", "where": "true"}}, "lookups: 'null' is reserved"),
        ({"team-member": {"collection": "members", "where": "true"}}, "lookups: 'team-member' must be a name"),
        ({"class": {"collection": "members", "where": "true"}}, "lookups: 'class' must be a name"),
        ({"member": {"collection": "members", "where": "new.email == user.email"}}, "lookups.member.where: 'new' isn't available here; use doc, user"),
        (
            {"team": {"collection": "teams", "where": "true"}, "member": {"collection": "members", "where": "doc.team == team.id"}},
            "lookups.member.where: 'team' isn't available here; use doc, user",
        ),
        ({"member": {"collection": "members", "where": "doc.email == email"}}, "lookups.member.where: unknown name 'email'"),
        ({"member": {"collection": "members"}}, "where"),
        ({"member": {"collection": "members", "where": "true", "limit": 1}}, "limit"),
    ],
)
def test_lookup_names_and_where_rules_are_checked(lookups, message):
    with pytest.raises(ValidationError, match=message):
        app(lookups=lookups)


def test_lookups_and_access_are_overridable_per_environment():
    parsed = app(
        lookups={"member": {"collection": "members", "where": "doc.email == user.email"}},
        access={"database": {"tickets": {"read": "member.role == 'agent'"}}},
        environments={
            "Staging": {"access": {"database": {"tickets": {"read": True, "insert": True}}}},
            "Production": {},
            "Demo": {"access": None, "lookups": {}},
        },
    )

    staging = parsed.for_environment("Staging")
    assert staging["lookups"]["member"].collection == "members"
    assert staging["access"].database["tickets"].insert == "true"
    production = parsed.for_environment("Production")
    assert production["access"].database["tickets"].read == "member.role == 'agent'"
    demo = parsed.for_environment("Demo")
    assert (demo["lookups"], demo["access"]) == ({}, None)


def test_rules_are_checked_against_the_lookups_of_each_environment():
    with pytest.raises(ValidationError, match="environment 'Staging': access.database.tickets.read: unknown name 'member'"):
        app(
            access={"database": {"tickets": {"read": "member.role == 'agent'"}}},
            environments={
                "Production": {"lookups": {"member": {"collection": "members", "where": "doc.email == user.email"}}},
                "Staging": {},
            },
        )


def test_apps_only_is_limited_to_sync_http_and_websocket_connectors():
    data = {"key": "events", "type": "kafka", "mode": "inbound", "connection": "kafka", "config": {"topic": "t"},
            "agents": ["assistant"], "apps_only": True}
    with pytest.raises(ValidationError, match="apps_only applies to sync http and websocket connectors only"):
        ConnectorFile.model_validate(data)

    connector = ConnectorFile.model_validate({"key": "chat", "type": "webhook", "mode": "sync", "agents": ["assistant"], "apps_only": True})
    assert connector.for_environment("Production")["apps_only"] is True


def test_loader_reads_apps_and_checks_connectors(tmp_path):
    write_agent(tmp_path)
    write_connector(tmp_path)
    write_connector(tmp_path, key="open-chat", apps_only=False)
    write_app_source(tmp_path)
    write_file(
        tmp_path / "apps" / "portal.yaml",
        """
        build:
          command: npm run build
          output: dist
        auth:
          mode: public
        connectors:
          - support-chat
          - open-chat
          - missing
        """,
    )

    apps, errors = load_apps(tmp_path)

    assert list(apps) == ["portal"]
    assert errors == [
        "apps/portal.yaml: connector 'open-chat' needs apps_only: true in connectors/open-chat.yaml",
        "apps/portal.yaml: connector 'missing' isn't in connectors/",
    ]


def test_loader_reports_missing_sources_lockfiles_and_stray_folders(tmp_path):
    write_agent(tmp_path)
    write_app_source(tmp_path, key="portal", lockfile=None)
    write_file(tmp_path / "apps" / "portal.yaml", "build: {command: npm run build, output: dist}\nauth: {mode: public}")
    write_file(tmp_path / "apps" / "admin.yaml", "build: {command: npm run build, output: dist}")
    write_file(tmp_path / "apps" / "shared" / "theme.css", "body {}")
    write_file(tmp_path / "apps" / "notes.md", "hello")
    write_file(tmp_path / "apps" / "broken.yml", "build: {}")

    apps, errors = load_apps(tmp_path)

    assert set(apps) == {"portal", "admin"}
    assert errors == [
        "apps/admin.yaml: the app's source folder apps/admin/ doesn't exist",
        "apps/broken.yml: app files must use the .yaml extension",
        "apps/notes.md: only apps/<key>.yaml files and apps/<key>/ folders belong in apps/",
        "apps/portal.yaml: apps/portal/ needs a lockfile (package-lock.json, pnpm-lock.yaml, yarn.lock) so builds are reproducible",
        "apps/shared/ has no apps/shared.yaml; add one or move the folder out of apps/",
    ]


def test_loader_skips_source_checks_for_uploads_without_app_sources(tmp_path):
    write_agent(tmp_path)
    write_file(tmp_path / "apps" / "portal.yaml", "build: {command: npm run build, output: dist}")

    apps, errors = load_apps(tmp_path, check_sources=False)

    assert list(apps) == ["portal"]
    assert errors == []


def test_loader_reports_invalid_rules_with_the_file(tmp_path):
    write_agent(tmp_path)
    write_app_source(tmp_path)
    write_file(
        tmp_path / "apps" / "portal.yaml",
        """
        build: {command: npm run build, output: dist}
        access:
          database:
            tickets:
              read: docs.author_id == user.id
        """,
    )

    apps, errors = load_apps(tmp_path)

    assert apps == {}
    assert errors[0].startswith("Failed to load app: apps/portal.yaml: ")
    assert "access.database.tickets.read: unknown name 'docs'" in errors[0]


def test_loader_matches_app_and_connector_environments(tmp_path):
    write_agent(tmp_path)
    write_connector(tmp_path, environments=["Staging"])
    write_app_source(tmp_path)
    write_file(
        tmp_path / "apps" / "portal.yaml",
        """
        build: {command: npm run build, output: dist}
        auth: {mode: public}
        connectors: [support-chat]
        """,
    )
    write_app_source(tmp_path, key="admin")
    write_file(
        tmp_path / "apps" / "admin.yaml",
        """
        build: {command: npm run build, output: dist}
        auth: {mode: connic}
        connectors: [support-chat]
        environments:
          Staging: {}
          Production: {}
        """,
    )

    _, errors = load_apps(tmp_path)

    assert errors == [
        "apps/admin.yaml (environment 'Production'): connector 'support-chat' isn't deployed to this environment",
        "apps/portal.yaml: connector 'support-chat' is only deployed to Staging; list the same environments in this file",
    ]


def test_lint_lists_apps_with_their_sign_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_update_hint", lambda: None)
    write_agent(tmp_path)
    write_app_source(tmp_path)
    write_file(
        tmp_path / "apps" / "portal.yaml",
        """
        build: {command: npm run build, output: dist}
        environments:
          Staging: {auth: {mode: connic}}
          Production: {}
        """,
    )

    result = CliRunner().invoke(cli.main, ["lint"])

    assert result.exit_code == 0, result.output
    assert "1 app(s) loaded" in result.output
    assert "portal: connic/dashboard sign-in → apps/portal/dist [Staging, Production]" in result.output


def test_project_without_apps_directory_declares_no_apps(tmp_path):
    write_agent(tmp_path)

    assert load_apps(tmp_path) == ({}, [])


@pytest.mark.parametrize(
    "parts, included",
    [
        (("portal.yaml",), True),
        (("portal", "src", "App.tsx"), True),
        (("portal", "node_modules", "react", "index.js"), False),
        (("portal", "dist", "index.html"), False),
        (("portal", "src", "build", "helper.ts"), True),
        (("portal", ".env"), False),
    ],
)
def test_app_source_skips_dependencies_build_output_and_hidden_files(parts, included):
    assert is_app_source(parts) is included


def test_deploy_upload_includes_app_sources_and_dev_uploads_only_app_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_agent(tmp_path)
    write_app_source(tmp_path)
    write_file(tmp_path / "apps" / "portal.yaml", "build: {command: npm run build, output: dist}\nauth: {mode: public}")
    write_file(tmp_path / "apps" / "notes.md", "hello")
    write_file(tmp_path / "apps" / "portal" / "src" / "main.tsx", "export {}")
    (tmp_path / "apps" / "portal" / "public").mkdir(parents=True)
    (tmp_path / "apps" / "portal" / "public" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    write_file(tmp_path / "apps" / "portal" / "node_modules" / "react" / "index.js", "module.exports = {}")
    write_file(tmp_path / "apps" / "portal" / "dist" / "index.html", "<html></html>")

    ok, error, files = cli._validate_project_files(include_apps=True)
    assert (ok, error) == (True, "")
    assert sorted(path.as_posix() for path in files if path.parts[0] == "apps") == [
        "apps/notes.md",
        "apps/portal.yaml",
        "apps/portal/package-lock.json",
        "apps/portal/package.json",
        "apps/portal/public/logo.png",
        "apps/portal/src/main.tsx",
    ]

    _, _, dev_files = cli._validate_project_files(include_apps=True, app_sources=False)
    assert [path.as_posix() for path in dev_files if path.parts[0] == "apps"] == ["apps/portal.yaml"]

    _, _, test_files = cli._validate_project_files()
    assert not any(path.parts[0] == "apps" for path in test_files)


def test_app_upload_rejects_unknown_file_types_and_oversized_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_agent(tmp_path)
    write_app_source(tmp_path)
    write_file(tmp_path / "apps" / "portal" / "run.sh", "echo hi")

    ok, error, _ = cli._validate_project_files(include_apps=True)
    assert not ok
    assert error == "apps/portal/run.sh: File type '.sh' not allowed in apps/"

    (tmp_path / "apps" / "portal" / "run.sh").unlink()
    monkeypatch.setattr(cli, "MAX_APPS_SIZE", 10)
    ok, error, _ = cli._validate_project_files(include_apps=True)
    assert not ok
    assert error == "App sources exceed the 10 byte limit"
