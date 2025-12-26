"""
Connic Composer SDK - Build agents with code.

This SDK allows you to define AI agents using YAML configuration files
and extend them with custom Python tools.

Basic usage:
    from connic import Agent, Tool, ProjectLoader
    
    loader = ProjectLoader("./my-project")
    agents = loader.load_agents()

Middleware:
    The 'before' middleware receives a dict representing the user message,
    allowing you to attach documents, images, or modify the content.
    
    from connic import StopProcessing
    
    async def before(content: dict, context: dict) -> dict:
        # content = {"role": "user", "parts": [...]}
        # Each part is either {"text": "..."} or {"data": bytes, "mime_type": "..."}
        pdf = open("doc.pdf", "rb").read()
        content["parts"].append({"data": pdf, "mime_type": "application/pdf"})
        return content
    
    async def after(response: str, context: dict) -> str:
        return response
"""

# Predefined tools module - for use in custom tools
from . import tools
from .core import Agent, AgentConfig, Middleware, RetryOptions, StopProcessing, Tool
from .loader import ProjectLoader

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig", 
    "Tool",
    "RetryOptions",
    "Middleware",
    "StopProcessing",
    "ProjectLoader",
    "tools",
    "__version__",
]
