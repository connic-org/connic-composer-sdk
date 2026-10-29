"""Pydantic models for apps declared in code.

Each ``apps/<key>.yaml`` file declares one app: a static frontend that connic
builds from ``apps/<key>/`` on every deploy and serves on its own URL. The
filename stem is the app's key and identity.

Example::

    name: Support portal              # optional, defaults to the key
    build:
      command: npm run build
      output: dist                    # folder with index.html, relative to apps/<key>/
    auth:                             # optional, omit to manage sign-in in the dashboard
      mode: oidc                      # public, oidc or connic
      issuer: https://login.example.com
      client_id: ${PORTAL_CLIENT_ID}
      client_secret: ${PORTAL_CLIENT_SECRET}
      allowed_domains: [example.com]
    connectors:                       # apps_only connectors the app may call
      - support-chat
    lookups:                          # documents resolved once per request
      member:
        collection: members
        where: doc.email == user.email
    access:                           # what the browser may read and change; the rest is denied
      database:
        tickets:
          read: doc.author_id == user.id or member.role == "agent"
          insert: true
          set:
            author_id: user.id
      files:
        reports/:
          read: "'support' in user.groups"
      runs:
        read: run.context.end_user.id == user.id
      retrieval:                      # Retrieval namespaces visitors may search
        help:
          search: true
    environments:                     # omit to deploy to every environment
      Staging:
        auth:
          mode: connic
      Production: {}

``${VAR}`` references are resolved at deploy time from the target
environment's variables, in ``domain`` and the ``issuer``, ``client_id`` and
``client_secret`` of ``auth`` only.
"""

from __future__ import annotations

import ast
import keyword
import re
from pathlib import PurePosixPath
from typing import AbstractSet, Annotated, Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from .connector_schema import CONNECTOR_KEY_PATTERN, _check_variables
from .core import FilesAccessConfig

APP_KEY_PATTERN = CONNECTOR_KEY_PATTERN
NODE_VERSIONS = ("22", "24")
LOCKFILES = ("package-lock.json", "pnpm-lock.yaml", "yarn.lock")
# Connector types an app can call through its /_connic/c/<key> path.
APP_CONNECTOR_TYPES = {"http": "sync", "websocket": "sync"}

# App sources are uploaded without their dependencies and build output.
APP_TEXT_EXTENSIONS = frozenset({
    ".js", ".mjs", ".cjs", ".jsx", ".ts", ".mts", ".cts", ".tsx", ".vue", ".svelte", ".astro",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".json", ".jsonc", ".md", ".mdx", ".txt",
    ".yaml", ".yml", ".toml", ".svg", ".xml", ".webmanifest", ".graphql", ".gql", ".lock",
})
APP_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".ico", ".woff", ".woff2", ".ttf", ".otf",
    ".mp3", ".mp4", ".webm", ".pdf",
})
APP_SKIPPED_FOLDERS = frozenset({"node_modules"})
# Build output folders directly inside apps/<key>/.
APP_BUILD_FOLDERS = frozenset({"dist", "build", "out", "coverage"})
MAX_APPS_SIZE = 20 * 1024 * 1024


def is_app_source(relative_parts: Tuple[str, ...]) -> bool:
    """Whether a path inside apps/ is app source that belongs in the upload."""
    if any(part.startswith(".") or part in APP_SKIPPED_FOLDERS for part in relative_parts):
        return False
    return not (len(relative_parts) > 2 and relative_parts[1] in APP_BUILD_FOLDERS)

_OUTPUT_PATTERN = re.compile(r"^\.?/?[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*/?$")
_OIDC_FIELDS = frozenset({"issuer", "client_id", "client_secret", "allowed_domains", "allowed_groups", "groups_claim", "scopes"})
_CONNIC_FIELDS = frozenset({"groups"})

_LITERAL_NAMES = frozenset({"true", "false", "null"})
_RESERVED_LOOKUP_NAMES = frozenset({"user", "doc", "new", "file", "run", "session", "approval"}) | _LITERAL_NAMES
# Record names each operation's rule may use, on top of user and the lookups.
_RULE_NAMES = {
    "database": {"read": {"doc"}, "insert": {"new"}, "update": {"doc", "new"}, "delete": {"doc"}},
    "files": {"read": {"file"}, "write": {"new"}, "update": {"file", "new"}, "delete": {"file"}},
    "runs": {"read": {"run"}, "cancel": {"run"}},
    "sessions": {"read": {"session"}, "delete": {"session"}},
    "approvals": {"read": {"approval"}, "decide": {"approval"}},
    "retrieval": {"search": set()},
}
# connic's condition grammar (shared/expression_filter.py): comparisons, and/or/not, paths and literals.
_RULE_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Compare, ast.Name, ast.Load,
    ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Subscript, ast.Attribute, ast.Slice,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
)
_MAX_RULE_LENGTH = 2000
_MAX_RULE_NODES = 200
_MAX_RULE_DEPTH = 30


def _single_line(value: Optional[str], label: str) -> Optional[str]:
    if value is not None and ("\n" in value or "\r" in value):
        raise ValueError(f"{label} must be a single line")
    return value


def _is_variable(value: str) -> bool:
    return value.startswith("${") and value.endswith("}") and value.count("${") == 1


def _rule_text(value: Any) -> Any:
    return ("true" if value else "false") if isinstance(value, bool) else value


# A rule is an expression; YAML true and false become "true" and "false".
Rule = Annotated[str, BeforeValidator(_rule_text)]


def _depth(node: ast.AST) -> int:
    return 1 + max((_depth(child) for child in ast.iter_child_nodes(node)), default=0)


def _is_literal_singleton(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None or node.value is True or node.value is False
    return isinstance(node, ast.Name) and node.id in _LITERAL_NAMES


def _check_rule(
    rule: str, path: str, allowed: AbstractSet[str], known: AbstractSet[str] = _RESERVED_LOOKUP_NAMES,
) -> None:
    """Check a rule against connic's condition grammar and the names it may use."""
    if len(rule) > _MAX_RULE_LENGTH:
        raise ValueError(f"{path}: rules are limited to {_MAX_RULE_LENGTH} characters")
    try:
        tree = ast.parse(rule.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{path}: {exc.msg}") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_RULE_NODES or _depth(tree) > _MAX_RULE_DEPTH:
        raise ValueError(f"{path}: rule is too complex")
    for node in nodes:
        if not isinstance(node, _RULE_NODES):
            raise ValueError(f"{path}: unsupported expression element {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"{path}: private attribute '{node.attr}' isn't allowed")
        if isinstance(node, ast.Name) and node.id not in allowed and node.id not in _LITERAL_NAMES:
            if node.id in known:
                raise ValueError(f"{path}: '{node.id}' isn't available here; use {', '.join(sorted(allowed))}")
            raise ValueError(f"{path}: unknown name '{node.id}'")
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for index, op in enumerate(node.ops):
                if isinstance(op, (ast.Is, ast.IsNot)) and not any(map(_is_literal_singleton, operands[index:index + 2])):
                    raise ValueError(f"{path}: use is and is not only with null, true or false")


class AppBuild(BaseModel):
    """How connic builds the app."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1, max_length=500)
    output: str = Field(min_length=1, max_length=200, description="Build output folder, relative to apps/<key>/.")
    install: Optional[str] = Field(
        default=None, min_length=1, max_length=500, description="Defaults to the install command for the lockfile.",
    )
    node: Literal["22", "24"] = "24"

    @field_validator("node", mode="before")
    @classmethod
    def _node_as_text(cls, v: Any) -> Any:
        return str(v) if type(v) is int else v

    @field_validator("command", "install")
    @classmethod
    def _check_command(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and "${" in v:
            raise ValueError("build commands can't use ${...}; read settings from /_connic/config.json at runtime")
        return _single_line(v, "build commands")

    @field_validator("output")
    @classmethod
    def _check_output(cls, v: str) -> str:
        path = PurePosixPath(v)
        if (
            not _OUTPUT_PATTERN.fullmatch(v) or path.is_absolute() or ".." in path.parts
            or path.parts in ((), (".",))
        ):
            raise ValueError("build.output must be a folder inside the app, such as dist")
        return path.as_posix()


class AppAuth(BaseModel):
    """Who can open the app."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["public", "oidc", "connic"]
    issuer: Optional[str] = Field(default=None, min_length=1)
    client_id: Optional[str] = Field(default=None, min_length=1)
    client_secret: Optional[str] = Field(default=None, min_length=1)
    allowed_domains: List[str] = Field(default_factory=list)
    allowed_groups: List[str] = Field(default_factory=list)
    groups_claim: str = Field(default="groups", min_length=1)
    scopes: List[str] = Field(default_factory=list)
    groups: List[str] = Field(default_factory=list, description="Permission groups that may open a connic app.")

    @model_validator(mode="after")
    def _check_mode(self) -> "AppAuth":
        allowed = {"oidc": _OIDC_FIELDS, "connic": _CONNIC_FIELDS}.get(self.mode, frozenset())
        extra = sorted(self.model_fields_set - allowed - {"mode"})
        if extra:
            raise ValueError(f"auth.{extra[0]} doesn't apply to {self.mode} apps")
        if self.mode != "oidc":
            return self
        for field in ("issuer", "client_id", "client_secret"):
            if getattr(self, field) is None:
                raise ValueError(f"auth.{field} is required for oidc apps")
            _check_variables(getattr(self, field), f"auth.{field}")
        if not _is_variable(self.client_secret):
            raise ValueError("auth.client_secret is a secret; reference a variable such as ${APP_CLIENT_SECRET} instead")
        if "${" not in self.issuer and not self.issuer.startswith("https://"):
            raise ValueError("auth.issuer must be an https:// URL")
        for domain in self.allowed_domains:
            if not domain or "@" in domain or "${" in domain:
                raise ValueError(f"auth.allowed_domains must list email domains such as example.com; '{domain}' isn't one")
        return self


class AppLookup(BaseModel):
    """A database document resolved once per request, available to rules under the lookup's name."""

    model_config = ConfigDict(extra="forbid")

    collection: str = Field(min_length=1)
    where: Rule = Field(description="Matches the document; may use doc and user. The first match wins.")


class AppCollectionAccess(BaseModel):
    """Rules for one database collection."""

    model_config = ConfigDict(extra="forbid")

    read: Optional[Rule] = None
    insert: Optional[Rule] = None
    update: Optional[Rule] = None
    delete: Optional[Rule] = None
    set: Optional[Dict[str, Rule]] = Field(
        default=None, description="Fields connic writes on insert and keeps unchanged on update; may use user and lookups.",
    )


class AppFolderAccess(BaseModel):
    """Rules for one files folder and everything below it."""

    model_config = ConfigDict(extra="forbid")

    read: Optional[Rule] = None
    write: Optional[Rule] = None
    update: Optional[Rule] = None
    delete: Optional[Rule] = None


class AppRunAccess(BaseModel):
    """Rules for agent runs."""

    model_config = ConfigDict(extra="forbid")

    read: Optional[Rule] = None
    cancel: Optional[Rule] = None


class AppSessionAccess(BaseModel):
    """Rules for agent sessions."""

    model_config = ConfigDict(extra="forbid")

    read: Optional[Rule] = None
    delete: Optional[Rule] = None


class AppApprovalAccess(BaseModel):
    """Rules for approvals."""

    model_config = ConfigDict(extra="forbid")

    read: Optional[Rule] = None
    decide: Optional[Rule] = None


class AppRetrievalAccess(BaseModel):
    """Rules for one Retrieval namespace and the namespaces below it."""

    model_config = ConfigDict(extra="forbid")

    search: Optional[Rule] = None


class AppAccess(BaseModel):
    """What the browser may read and change. A missing resource or operation is denied."""

    model_config = ConfigDict(extra="forbid")

    database: Optional[Dict[str, AppCollectionAccess]] = None
    files: Optional[Dict[str, AppFolderAccess]] = Field(
        default=None, description="Folders cover everything below them; the most specific folder wins.",
    )
    runs: Optional[AppRunAccess] = None
    sessions: Optional[AppSessionAccess] = None
    approvals: Optional[AppApprovalAccess] = None
    retrieval: Optional[Dict[str, AppRetrievalAccess]] = Field(
        default=None, description="Namespaces visitors may search; each covers the namespaces below it.",
    )

    @field_validator("files")
    @classmethod
    def _normalize_folders(cls, folders: Optional[Dict[str, AppFolderAccess]]) -> Optional[Dict[str, AppFolderAccess]]:
        return FilesAccessConfig._normalize_folders(folders)

    def check_rules(self, lookups: AbstractSet[str]) -> None:
        """Check every rule's syntax and names; ``lookups`` are the lookup names declared next to it."""
        allowed = {"user", *lookups}
        scopes = [
            *((f"access.database.{name}", "database", rules) for name, rules in (self.database or {}).items()),
            *((f"access.files.{folder}", "files", rules) for folder, rules in (self.files or {}).items()),
            *((f"access.{resource}", resource, getattr(self, resource)) for resource in ("runs", "sessions", "approvals")),
            *((f"access.retrieval.{namespace}", "retrieval", rules) for namespace, rules in (self.retrieval or {}).items()),
        ]
        for path, resource, rules in scopes:
            if rules is None:
                continue
            for operation, names in _RULE_NAMES[resource].items():
                rule = getattr(rules, operation)
                if rule is not None:
                    _check_rule(rule, f"{path}.{operation}", allowed | names)
        for name, rules in (self.database or {}).items():
            for field, rule in (rules.set or {}).items():
                _check_rule(rule, f"access.database.{name}.set.{field}", allowed)


class AppOverride(BaseModel):
    """Per-environment changes to an app. Each field replaces the value from the top of the file."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    auth: Optional[AppAuth] = None
    connectors: Optional[List[str]] = None
    domain: Optional[str] = Field(default=None, min_length=1)
    content_security_policy: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    lookups: Optional[Dict[str, AppLookup]] = None
    access: Optional[AppAccess] = None


class AppFile(BaseModel):
    """Parsed contents of one ``apps/<key>.yaml`` file."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="Taken from the filename stem.")
    name: Optional[str] = Field(default=None, min_length=1, max_length=200, description="Defaults to the key.")
    build: AppBuild
    auth: Optional[AppAuth] = Field(default=None, description="Omit to manage sign-in in the dashboard.")
    connectors: List[str] = Field(default_factory=list)
    domain: Optional[str] = Field(default=None, min_length=1)
    content_security_policy: Optional[str] = Field(
        default=None, min_length=1, max_length=4000, description="Replaces connic's default Content-Security-Policy.",
    )
    lookups: Optional[Dict[str, AppLookup]] = Field(
        default=None, description="Documents resolved once per request, available to access rules by name.",
    )
    access: Optional[AppAccess] = Field(
        default=None, description="Data the browser may read and change. Everything else is denied.",
    )
    environments: Optional[Dict[str, Optional[AppOverride]]] = Field(
        default=None,
        description="Environment names the app is deployed to, with optional overrides. Omit for all.",
    )

    @field_validator("key")
    @classmethod
    def _validate_key(cls, v: str) -> str:
        if not APP_KEY_PATTERN.fullmatch(v):
            raise ValueError(
                f"file name '{v}' must start with a letter or digit and use only letters, digits, '-' or '_' "
                "(100 characters max)"
            )
        return v

    @model_validator(mode="after")
    def _validate_app(self) -> "AppFile":
        if self.environments is not None:
            if not self.environments:
                raise ValueError("environments must list at least one environment; omit it to deploy everywhere")
            for environment in list(self.environments):
                if "${" in environment:
                    raise ValueError(f"environment '{environment}' can't use ${{...}}")
                if self.environments[environment] is None:
                    self.environments[environment] = AppOverride()

        for environment, spec in self.specs():
            where = f"environment '{environment}': " if environment else ""
            try:
                self._validate_spec(spec)
            except ValueError as exc:
                raise ValueError(f"{where}{exc}") from None
        return self

    @staticmethod
    def _validate_spec(spec: Dict[str, Any]) -> None:
        if "${" in spec["name"]:
            raise ValueError("name can't use ${...}")
        _check_variables(spec["domain"], "domain")
        _single_line(spec["content_security_policy"], "content_security_policy")
        seen = set()
        for key in spec["connectors"]:
            if not CONNECTOR_KEY_PATTERN.fullmatch(key):
                raise ValueError(f"connectors: '{key}' isn't a connector key")
            if key in seen:
                raise ValueError(f"connector '{key}' is listed twice")
            seen.add(key)

        lookups = spec["lookups"] or {}
        for name, lookup in lookups.items():
            if not (name.isascii() and name.isidentifier()) or keyword.iskeyword(name):
                raise ValueError(f"lookups: '{name}' must be a name such as member")
            if name in _RESERVED_LOOKUP_NAMES:
                raise ValueError(f"lookups: '{name}' is reserved; choose another name")
            _check_rule(lookup.where, f"lookups.{name}.where", {"doc", "user"}, _RESERVED_LOOKUP_NAMES | set(lookups))
        if spec["access"] is not None:
            spec["access"].check_rules(set(lookups))

    def specs(self) -> List[Tuple[Optional[str], Dict[str, Any]]]:
        """Every environment variant of the app. The environment is None when it deploys everywhere."""
        if self.environments is None:
            return [(None, self._merge(AppOverride()))]
        return [(environment, self._merge(override)) for environment, override in self.environments.items()]

    def for_environment(self, environment: str) -> Optional[Dict[str, Any]]:
        """The app as deployed to ``environment``, or None when it isn't declared there."""
        if self.environments is None:
            return self._merge(AppOverride())
        override = self.environments.get(environment)
        return self._merge(override) if override is not None else None

    def _merge(self, override: AppOverride) -> Dict[str, Any]:
        # An override set to null removes the value from the top of the file.
        def pick(field: str) -> Any:
            return getattr(override, field) if field in override.model_fields_set else getattr(self, field)

        return {
            "key": self.key,
            "name": override.name or self.name or self.key,
            "build": self.build,
            "auth": override.auth or self.auth,
            "connectors": override.connectors if override.connectors is not None else self.connectors,
            "domain": pick("domain"),
            "content_security_policy": pick("content_security_policy"),
            "lookups": pick("lookups"),
            "access": pick("access"),
        }
