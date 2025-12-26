import asyncio
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml

from .core import Agent, AgentConfig, AgentType, McpServerConfig, Middleware, RetryOptions, Tool

# List of predefined tool names - SDK knows names only, not implementations
# Actual implementations are in the runner (backend/app/templates/predefined_tools/)
PREDEFINED_TOOL_NAMES = {
    "trigger_agent",
    "query_knowledge",   # Query the knowledge base using semantic search
    "store_knowledge",   # Store new knowledge in the knowledge base
    "delete_knowledge",  # Delete knowledge from the knowledge base
    "web_search",        # Search the web for real-time information (costs 2x runs)
}


class ProjectLoader:
    """
    Loads a Connic project from disk, discovering agents, tools, and middlewares.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.agents_dir = self.project_root / "agents"
        self.tools_dir = self.project_root / "tools"
        self.middleware_dir = self.project_root / "middleware"
        self.schemas_dir = self.project_root / "schemas"
        
        # Cache for loaded modules to avoid reloading
        self._loaded_modules: Dict[str, Any] = {}
        # Cache for loaded middlewares
        self._loaded_middlewares: Dict[str, Optional[Middleware]] = {}
        # Cache for loaded schemas
        self._loaded_schemas: Dict[str, Dict[str, Any]] = {}
        
        # Ensure project root is in sys.path for imports
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

    def load_agents(self) -> List[Agent]:
        """
        Scans the agents directory and loads all defined agents.
        
        Returns:
            List of loaded Agent objects
        
        Raises:
            FileNotFoundError: If agents directory doesn't exist
        """
        if not self.agents_dir.exists():
            raise FileNotFoundError(f"Agents directory not found at {self.agents_dir}")

        agents = []
        errors = []
        
        for agent_file in sorted(self.agents_dir.glob("*.yaml")):
            try:
                agent = self._load_single_agent(agent_file)
                agents.append(agent)
            except Exception as e:
                errors.append(f"{agent_file.name}: {e}")
        
        if errors:
            for error in errors:
                print(f"Warning: {error}")
        
        return agents

    def load_agent(self, name: str) -> Agent:
        """
        Load a specific agent by name.
        
        Args:
            name: The agent name (filename without .yaml extension)
            
        Returns:
            The loaded Agent object
        """
        agent_file = self.agents_dir / f"{name}.yaml"
        if not agent_file.exists():
            raise FileNotFoundError(f"Agent '{name}' not found at {agent_file}")
        return self._load_single_agent(agent_file)

    def _load_single_agent(self, agent_file: Path) -> Agent:
        """Load a single agent from a YAML file."""
        with open(agent_file, "r") as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError(f"Empty configuration in {agent_file.name}")
        
        # Handle retry_options as nested object
        if "retry_options" in config_data and config_data["retry_options"]:
            config_data["retry_options"] = RetryOptions(**config_data["retry_options"])
        
        # Handle mcp_servers as list of nested objects
        if "mcp_servers" in config_data and config_data["mcp_servers"]:
            config_data["mcp_servers"] = [
                McpServerConfig(**server) if isinstance(server, dict) else server
                for server in config_data["mcp_servers"]
            ]
        
        # Convert type string to enum if present
        if "type" in config_data and isinstance(config_data["type"], str):
            config_data["type"] = AgentType(config_data["type"])
        
        # Parse config
        config = AgentConfig(**config_data)
        
        # Load output schema if specified (LLM agents only)
        if config.output_schema:
            if config.type == AgentType.LLM:
                try:
                    schema_dict = self._load_schema(config.output_schema)
                    config.output_schema_dict = schema_dict
                except Exception as e:
                    print(f"Warning: Could not load output schema '{config.output_schema}' for agent '{config.name}': {e}")
            else:
                print(f"Warning: output_schema is only supported for LLM agents. Ignoring for '{config.name}' (type: {config.type.value})")
        
        # Resolve tools based on agent type
        tools = []
        
        if config.type == AgentType.LLM:
            # LLM agents: resolve all tools from the tools list
            for tool_ref in config.tools:
                try:
                    tool = self._resolve_tool(tool_ref)
                    tools.append(tool)
                except Exception as e:
                    print(f"Warning: Could not resolve tool '{tool_ref}' for agent '{config.name}': {e}")
        
        elif config.type == AgentType.TOOL:
            # Tool agents: resolve the single tool_name
            if config.tool_name:
                try:
                    tool = self._resolve_tool(config.tool_name)
                    tools.append(tool)
                except Exception as e:
                    print(f"Warning: Could not resolve tool '{config.tool_name}' for tool agent '{config.name}': {e}")
        
        # Sequential agents don't need tools - they orchestrate other agents

        return Agent(config=config, tools=tools)

    def _resolve_tool(self, tool_ref: str) -> Tool:
        """
        Resolves a tool reference to a Tool object.
        
        For predefined tools (e.g., "trigger_agent"), creates a marker that
        the runner will inject the actual implementation for.
        
        For user-defined tools, format: "module.function" (e.g., "calculator.add")
        
        Args:
            tool_ref: The tool reference string
            
        Returns:
            A Tool object wrapping the function (or a marker for predefined tools)
        """
        # Check if it's a predefined tool (name only, no implementation)
        if tool_ref in PREDEFINED_TOOL_NAMES:
            return self._create_predefined_tool_marker(tool_ref)
        
        # User-defined tool: expect "module.function" format
        parts = tool_ref.split(".")
        
        if len(parts) != 2:
            raise ValueError(
                f"Invalid tool reference '{tool_ref}'. "
                "Use 'module.function' format (e.g., 'calculator.add') "
                f"or a predefined tool name ({', '.join(PREDEFINED_TOOL_NAMES)})."
            )
        
        module_name = parts[0]
        function_name = parts[1]

        module = self._load_tool_module(module_name)
        
        func = getattr(module, function_name, None)
        if func is None:
            raise ValueError(f"Function '{function_name}' not found in module '{module_name}'")
        
        if not callable(func):
            raise ValueError(f"'{function_name}' in module '{module_name}' is not callable")

        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        # Get description from docstring
        description = inspect.getdoc(func) or "No description provided."
        
        # Generate JSON schema for parameters
        schema = self._generate_schema(func)
        
        return Tool(
            name=function_name,
            description=description,
            func=func,
            is_async=is_async,
            parameters=schema
        )
    
    def _create_predefined_tool_marker(self, tool_name: str) -> Tool:
        """
        Create a marker for a predefined tool.
        The actual implementation will be injected by the runner at build time.
        """
        return Tool(
            name=tool_name,
            description="",  # Will be set by runner
            func=None,       # Will be injected by runner
            is_async=True,   # Will be set by runner
            parameters={},   # Will be set by runner
            is_predefined=True,  # Marker for runner to inject implementation
        )

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Load a JSON Schema file from the schemas/ directory.
        
        Args:
            schema_name: Name of the schema file (without .json extension)
            
        Returns:
            The parsed JSON Schema as a dictionary
            
        Raises:
            FileNotFoundError: If schema file doesn't exist
            json.JSONDecodeError: If schema file is not valid JSON
        """
        # Check cache first
        if schema_name in self._loaded_schemas:
            return self._loaded_schemas[schema_name]
        
        schema_file = self.schemas_dir / f"{schema_name}.json"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema '{schema_name}' not found at {schema_file}")
        
        with open(schema_file, "r") as f:
            schema_dict = json.load(f)
        
        # Validate it's a valid JSON Schema structure
        if not isinstance(schema_dict, dict):
            raise ValueError(f"Schema '{schema_name}' must be a JSON object")
        
        if "type" not in schema_dict:
            raise ValueError(f"Schema '{schema_name}' must have a 'type' field")
        
        # Cache the loaded schema
        self._loaded_schemas[schema_name] = schema_dict
        return schema_dict

    def discover_schemas(self) -> List[str]:
        """
        Discover all available schemas in the schemas/ directory.
        
        Returns:
            List of schema names (without .json extension)
        """
        if not self.schemas_dir.exists():
            return []
        
        schemas = []
        for schema_file in self.schemas_dir.glob("*.json"):
            if not schema_file.name.startswith("_"):
                schemas.append(schema_file.stem)
        
        return sorted(schemas)

    def _generate_schema(self, func) -> Dict[str, Any]:
        """
        Generate a JSON Schema for a function's parameters.
        
        Uses type hints and docstring to build a schema suitable for LLM function calling.
        """
        sig = inspect.signature(func)
        
        # Try to get type hints (handles forward references)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        
        # Parse docstring for parameter descriptions
        param_docs = self._parse_docstring_params(func.__doc__ or "")
        
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            # Get type from hints or annotation
            annotation = hints.get(name, param.annotation)
            
            # Convert Python type to JSON Schema type
            prop_schema = self._type_to_schema(annotation)
            
            # Add description from docstring if available
            if name in param_docs:
                prop_schema["description"] = param_docs[name]
            
            properties[name] = prop_schema
            
            # Parameter is required if it has no default value
            if param.default == inspect.Parameter.empty:
                required.append(name)
                
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _type_to_schema(self, annotation) -> Dict[str, Any]:
        """Convert a Python type annotation to JSON Schema."""
        if annotation == inspect.Parameter.empty:
            return {"type": "string"}
        
        # Handle None type
        if annotation is type(None):
            return {"type": "null"}
        
        # Basic type mapping
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
            type(None): {"type": "null"},
        }
        
        # Check for basic types first
        if annotation in type_map:
            return type_map[annotation].copy()
        
        # Handle generic types (List[X], Dict[K, V], Optional[X], etc.)
        origin = get_origin(annotation)
        args = get_args(annotation)
        
        if origin is Union:
            # Handle Optional[X] which is Union[X, None]
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                # This is Optional[X]
                return self._type_to_schema(non_none_args[0])
            # General Union - just use first type for simplicity
            if non_none_args:
                return self._type_to_schema(non_none_args[0])
            return {"type": "string"}
        
        if origin is list:
            schema = {"type": "array"}
            if args:
                schema["items"] = self._type_to_schema(args[0])
            return schema
        
        if origin is dict:
            schema = {"type": "object"}
            if len(args) >= 2:
                schema["additionalProperties"] = self._type_to_schema(args[1])
            return schema
        
        # Default to string for unknown types
        return {"type": "string"}
    
    def _parse_docstring_params(self, docstring: str) -> Dict[str, str]:
        """
        Parse parameter descriptions from a docstring.
        
        Supports Google-style docstrings:
            Args:
                param_name: Description of the parameter
        """
        params = {}
        
        if not docstring:
            return params
        
        lines = docstring.split('\n')
        in_args_section = False
        current_param = None
        current_desc = []
        
        for line in lines:
            stripped = line.strip()
            
            # Check for Args: section start
            if stripped.lower() in ('args:', 'arguments:', 'parameters:'):
                in_args_section = True
                continue
            
            # Check for section end (Returns:, Raises:, etc.)
            if stripped.lower().endswith(':') and not stripped.startswith(' '):
                if in_args_section and current_param:
                    params[current_param] = ' '.join(current_desc).strip()
                in_args_section = False
                current_param = None
                current_desc = []
                continue
            
            if in_args_section and stripped:
                # Check if this is a new parameter (contains ':')
                if ':' in stripped and not stripped.startswith(' '):
                    # Save previous parameter
                    if current_param:
                        params[current_param] = ' '.join(current_desc).strip()
                    
                    # Parse new parameter
                    parts = stripped.split(':', 1)
                    # Handle "param_name (type)" format
                    param_part = parts[0].strip()
                    if '(' in param_part:
                        param_part = param_part.split('(')[0].strip()
                    current_param = param_part
                    current_desc = [parts[1].strip()] if len(parts) > 1 else []
                elif current_param:
                    # Continuation of previous parameter description
                    current_desc.append(stripped)
        
        # Don't forget the last parameter
        if current_param:
            params[current_param] = ' '.join(current_desc).strip()
        
        return params

    def _load_tool_module(self, module_name: str):
        """
        Dynamically loads a Python module from the tools/ directory.
        
        Args:
            module_name: Name of the module (filename without .py)
            
        Returns:
            The loaded module object
        """
        if module_name in self._loaded_modules:
            return self._loaded_modules[module_name]

        file_path = self.tools_dir / f"{module_name}.py"
        if not file_path.exists():
            raise FileNotFoundError(f"Tool module '{module_name}' not found at {file_path}")

        spec = importlib.util.spec_from_file_location(f"tools.{module_name}", file_path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load module spec for {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules so imports inside the module work
        sys.modules[f"tools.{module_name}"] = module
        
        spec.loader.exec_module(module)
        self._loaded_modules[module_name] = module
        return module

    def discover_tools(self) -> Dict[str, List[str]]:
        """
        Discover all available tools in the project.
        
        Returns:
            Dict mapping module names to lists of function names
        """
        if not self.tools_dir.exists():
            return {}
        
        tools = {}
        for tool_file in self.tools_dir.glob("*.py"):
            if tool_file.name.startswith("_"):
                continue
            
            module_name = tool_file.stem
            try:
                module = self._load_tool_module(module_name)
                functions = []
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    # Only include public functions defined in this module
                    if not name.startswith("_") and obj.__module__ == f"tools.{module_name}":
                        functions.append(name)
                if functions:
                    tools[module_name] = functions
            except Exception as e:
                print(f"Warning: Could not discover tools in {module_name}: {e}")
        
        return tools

    def load_middleware(self, agent_name: str) -> Optional[Middleware]:
        """
        Load middleware for a specific agent by name.
        
        Middleware files are automatically discovered from the middleware/ directory.
        The file must be named after the agent (e.g., middleware/assistant.py for agent 'assistant').
        
        Args:
            agent_name: Name of the agent (matches YAML filename without extension)
            
        Returns:
            Middleware object with before/after functions, or None if no middleware exists
        """
        # Check cache first
        if agent_name in self._loaded_middlewares:
            return self._loaded_middlewares[agent_name]
        
        middleware_file = self.middleware_dir / f"{agent_name}.py"
        if not middleware_file.exists():
            self._loaded_middlewares[agent_name] = None
            return None
        
        try:
            # Load the middleware module
            spec = importlib.util.spec_from_file_location(
                f"middleware.{agent_name}", middleware_file
            )
            if not spec or not spec.loader:
                self._loaded_middlewares[agent_name] = None
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"middleware.{agent_name}"] = module
            spec.loader.exec_module(module)
            
            # Extract before and after functions
            before_func = getattr(module, 'before', None)
            after_func = getattr(module, 'after', None)
            
            # Validate they're callable if present
            if before_func is not None and not callable(before_func):
                before_func = None
            if after_func is not None and not callable(after_func):
                after_func = None
            
            # If neither exists, return None
            if before_func is None and after_func is None:
                self._loaded_middlewares[agent_name] = None
                return None
            
            middleware = Middleware(before=before_func, after=after_func)
            self._loaded_middlewares[agent_name] = middleware
            return middleware
            
        except Exception as e:
            print(f"Warning: Failed to load middleware for {agent_name}: {e}")
            self._loaded_middlewares[agent_name] = None
            return None

    def discover_middlewares(self) -> Dict[str, List[str]]:
        """
        Discover all middleware files and their available hooks.
        
        Returns:
            Dict mapping agent names to lists of available hooks ('before', 'after')
        """
        if not self.middleware_dir.exists():
            return {}
        
        middlewares = {}
        for mw_file in self.middleware_dir.glob("*.py"):
            if mw_file.name.startswith("_"):
                continue
            
            agent_name = mw_file.stem
            try:
                mw = self.load_middleware(agent_name)
                if mw:
                    hooks = []
                    if mw.before is not None:
                        hooks.append("before")
                    if mw.after is not None:
                        hooks.append("after")
                    if hooks:
                        middlewares[agent_name] = hooks
            except Exception as e:
                print(f"Warning: Could not discover middleware for {agent_name}: {e}")
        
        return middlewares
