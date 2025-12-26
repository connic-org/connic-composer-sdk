"""
Predefined tools for use in custom tools.

These are placeholder functions that get auto-injected with real implementations
when running via `connic test` or after deployment.

Usage in custom tools:
    from connic.tools import trigger_agent, query_knowledge

    async def my_custom_tool(input: str) -> dict:
        # Call another agent
        result = await trigger_agent(
            agent_name="summarizer",
            payload={"text": input}
        )
        return {"summary": result["response"]}
"""
from typing import Any, Dict, Optional


async def trigger_agent(
    agent_name: str,
    payload: Any,
    wait_for_response: bool = True,
    timeout_seconds: int = 60
) -> dict:
    """
    Trigger another agent within the same project/environment.
    
    Args:
        agent_name: Name of the agent to trigger
        payload: Data to send to the agent (dict, list, or string)
        wait_for_response: If True, wait for the agent to complete and return its response
        timeout_seconds: Maximum time to wait for response (only if wait_for_response=True)
    
    Returns:
        dict with 'run_id' and optionally 'response', 'status', 'error' if wait_for_response=True
    
    Example:
        result = await trigger_agent(
            agent_name="summarizer",
            payload={"text": "Long document to summarize..."},
            wait_for_response=True
        )
        summary = result["response"]
    """
    raise RuntimeError(
        "trigger_agent will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def query_knowledge(
    query: str,
    namespace: Optional[str] = None,
    min_score: float = 0.7,
    max_results: int = 3
) -> Dict[str, Any]:
    """
    Query the knowledge base for relevant information using semantic search.
    
    This tool searches the environment's knowledge base and returns the most
    relevant text chunks based on semantic similarity to your query.
    
    Args:
        query: The search query - describe what information you're looking for.
               Be specific and descriptive for better results.
        namespace: Optional namespace to filter results. Use this to search
                   only within a specific category of knowledge (e.g., "policies",
                   "products", "faq"). If not provided, searches all namespaces.
        min_score: Minimum similarity score threshold (default: 0.7).
                   Only results with score >= min_score are returned.
                   Range is 0.0 to 1.0 where 1.0 is a perfect match.
        max_results: Maximum number of results to return (default: 3).
    
    Returns:
        A dictionary containing:
        - results: List of matching chunks, each with:
            - content: The text content of the chunk
            - entry_id: The ID of the source entry
            - namespace: The namespace (if any)
            - score: Similarity score (higher is better, max 1.0)
    
    Example:
        result = await query_knowledge("What is the refund policy?")
        for chunk in result["results"]:
            print(f"[{chunk['score']:.2f}] {chunk['content'][:100]}...")
    """
    raise RuntimeError(
        "query_knowledge will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def store_knowledge(
    content: str,
    entry_id: Optional[str] = None,
    namespace: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Store new knowledge in the knowledge base for future retrieval.
    
    Use this tool to save important information, facts, or content that
    should be remembered and searchable later. The content is automatically
    chunked (if long) and embedded for semantic search.
    
    Args:
        content: The text content to store. This can be any length - long
                 content is automatically split into searchable chunks.
        entry_id: Optional custom identifier for this entry. If not provided,
                  a random UUID is generated. Use this to update existing entries
                  by providing the same entry_id within the same namespace.
        namespace: Optional namespace for organization. Use namespaces to
                   categorize knowledge (e.g., "user_preferences", "meeting_notes").
        metadata: Optional dictionary of additional metadata to store.
    
    Returns:
        A dictionary containing:
        - entry_id: Identifier for this knowledge entry
        - chunk_count: Number of chunks the content was split into
        - success: True if stored successfully
    
    Example:
        result = await store_knowledge(
            content="The user prefers dark mode.",
            entry_id="user-prefs",
            namespace="preferences"
        )
    """
    raise RuntimeError(
        "store_knowledge will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def delete_knowledge(
    entry_id: str,
    namespace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a specific knowledge entry from the knowledge base.
    
    Args:
        entry_id: The identifier of the knowledge entry to delete.
        namespace: Optional namespace to scope the deletion.
    
    Returns:
        A dictionary containing:
        - ok: True if deletion was successful
        - deleted_chunks: Number of chunks deleted
    
    Example:
        result = await delete_knowledge(
            entry_id="outdated-info",
            namespace="products"
        )
    """
    raise RuntimeError(
        "delete_knowledge will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def web_search(
    query: str,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Search the web for real-time information.
    
    This is a managed service - no configuration required.
    Note: Each call to web_search adds 1 additional billable run.
    (e.g., a run with 2 searches counts as 3 runs: 1 base + 2 searches)
    
    Args:
        query: The search query
        max_results: Number of results to return (default: 5, max: 10)
    
    Returns:
        A dictionary containing:
        - answer: AI-generated summary of search findings
        - results: List of search results, each with:
            - title: Page title
            - url: Page URL
            - content: Snippet of page content
            - score: Relevance score
    
    Example:
        result = await web_search("latest news on AI regulations")
        print(result["answer"])
        for r in result["results"]:
            print(f"- {r['title']}: {r['url']}")
    """
    raise RuntimeError(
        "web_search will be auto-injected when running via connic CLI or after deployment. "
        "Run 'connic test' to test your agents with predefined tools."
    )


# All available predefined tools
__all__ = [
    "trigger_agent",
    "query_knowledge", 
    "store_knowledge",
    "delete_knowledge",
    "web_search",
]

