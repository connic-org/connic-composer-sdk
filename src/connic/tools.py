"""
Predefined tools for use in custom tools.

These are placeholder functions that get auto-injected with real implementations
when running via `connic test` or after deployment.

Usage in custom tools:
    from connic.tools import trigger_agent, query_knowledge, db_find

    async def my_custom_tool(input: str) -> dict:
        # Call another agent
        result = await trigger_agent(
            agent_name="summarizer",
            payload={"text": input}
        )
        return {"summary": result["response"]}
"""
from typing import Any, Dict, List, Optional, Union


async def trigger_agent(
    agent_name: str,
    payload: str,
    wait_for_response: bool = True,
    timeout_seconds: int = 60
) -> dict:
    """
    Trigger another agent within the same project/environment.

    Args:
        agent_name: Name of the agent to trigger
        payload: The data to send to the agent, as a string (JSON or plain
                 text), e.g. '{"text": "Long document to summarize..."}'.
        wait_for_response: If True, wait for the agent to complete and return its response
        timeout_seconds: Maximum time to wait for response (only if wait_for_response=True)

    Returns:
        dict with 'run_id' and optionally 'response', 'status', 'error' if wait_for_response=True

    Example:
        result = await trigger_agent(
            agent_name="summarizer",
            payload='{"text": "Long document to summarize..."}',
            wait_for_response=True
        )
        summary = result["response"]
    """
    raise RuntimeError(
        "trigger_agent will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def trigger_agent_at(
    agent_name: str,
    payload: str,
    delay: Optional[Dict[str, float]] = None,
    unix_timestamp: Optional[float] = None,
) -> dict:
    """
    Schedule another agent to be triggered at a specific time in the future.

    Exactly one of ``delay`` or ``unix_timestamp`` must be provided.

    Args:
        agent_name: Name of the agent to trigger
        payload: The data to send to the agent, as a string (JSON or plain
                 text), e.g. '{"report_type": "daily"}'.
        delay: Relative time offset as a dict with keys d (days), h (hours),
               m (minutes), s (seconds). At least one key is required.
               Example: {"h": 2, "m": 30} triggers in 2 hours and 30 minutes.
        unix_timestamp: Absolute Unix timestamp (seconds since epoch) for
                        when to trigger the agent.

    Returns:
        dict with 'run_id', 'scheduled_at' (ISO 8601 UTC), and 'status'

    Example:
        result = await trigger_agent_at(
            agent_name="report-generator",
            payload='{"report_type": "daily"}',
            delay={"h": 2, "m": 30}
        )
    """
    raise RuntimeError(
        "trigger_agent_at will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def query_knowledge(
    query: str,
    namespace: Optional[str] = None,
    min_score: float = 0.7,
    max_results: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None,
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
        metadata_filter: Optional MongoDB-style filter applied to entry
                   metadata. Shorthand `{"field": value}` is equality; use
                   operators for anything else.

                   Supported operators (same set as db_find):
                     $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists,
                     $regex, $contains, $elemMatch, $and, $or, $nor, $not.
                   Dot-notation works for nested keys.

                   Examples:
                     {"product_id": "X"}
                     {"status": {"$in": ["active", "pending"]}}
                     {"run_id": {"$ne": current_run_id}}

    Returns:
        A dictionary containing:
        - results: List of matching chunks, each with:
            - content: The text content of the chunk
            - entry_id: The ID of the source entry
            - namespace: The namespace (if any)
            - score: Similarity score (higher is better, max 1.0)
            - metadata: The entry's metadata dict

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
    entry_id: Optional[str] = None,
    namespace: Optional[str] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Delete knowledge from the knowledge base.

    Two deletion modes are supported:

    1. By entry_id (optionally scoped to a namespace) — deletes one entry.
    2. By namespace + optional metadata_filter — bulk-deletes every entry
       within the namespace (and its sub-namespaces) matching the filter.
       Without a filter, the entire namespace subtree is deleted.

    Args:
        entry_id: ID of a single entry to delete. Omit when bulk-deleting by
                  namespace/metadata_filter.
        namespace: Namespace to scope the deletion. Required when using
                   metadata_filter; optional with entry_id. Sub-namespaces are
                   always included.
        metadata_filter: MongoDB-style filter applied to entry metadata. Only
                   entries matching the filter are deleted. Requires
                   `namespace`.

                   Supported operators (same set as db_find):
                     $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists,
                     $regex, $contains, $elemMatch, $and, $or, $nor, $not.
                   Dot-notation works for nested keys.

    Returns:
        A dictionary containing:
        - ok: True if deletion was successful
        - deleted_chunks: Number of underlying chunks deleted

    Example:
        # Delete one entry
        await delete_knowledge(entry_id="outdated-info", namespace="products")

        # Orphan cleanup: keep current run's writes, delete the rest in scope
        await delete_knowledge(
            namespace="confluence",
            metadata_filter={
                "root_page_id": page_id,
                "run_id": {"$ne": current_run_id},
            },
        )

        # Wipe an entire namespace subtree
        await delete_knowledge(namespace="meetings")
    """
    raise RuntimeError(
        "delete_knowledge will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def kb_list_namespaces(
    parent: Optional[str] = None,
    depth: int = 1,
) -> Dict[str, Any]:
    """
    List knowledge base namespaces to discover how content is organized.

    Namespaces are hierarchical, separated by dots (e.g. "policies.hr.leave").
    Use this tool to explore the namespace hierarchy before searching.
    Call without arguments to see top-level namespaces, then drill down
    into specific namespaces to see their sub-namespaces.

    Args:
        parent: Optional parent namespace to list children of.
                e.g. "policies" returns "policies.hr", "policies.legal", etc.
                If not provided, lists top-level namespaces.
        depth: How many levels deep to list (default: 1, max: 10).
               1 = direct children only, 0 = all descendants.

    Returns:
        When parent is not provided:
            A list of namespace objects, each with:
            - name: Full namespace path
            - entry_count: Number of entries directly in this namespace
            - total_entry_count: Entries in this namespace and all sub-namespaces
            - has_children: Whether sub-namespaces exist

        When parent is provided:
            A dictionary with:
            - parent: Info about the parent namespace
            - namespaces: List of child namespace objects

    Example:
        # List top-level namespaces
        result = await kb_list_namespaces()

        # Drill into a specific namespace
        result = await kb_list_namespaces(parent="policies")
    """
    raise RuntimeError(
        "kb_list_namespaces will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def web_search(
    query: str,
    max_results: int = 5,
    country: Optional[str] = None,
    include_news: bool = False,
) -> Dict[str, Any]:
    """
    Search the web for real-time information.
    
    This is a managed service - no configuration required.
    Note: Each call to web_search adds 1 additional billable run.
    (e.g., a run with 2 searches counts as 3 runs: 1 base + 2 searches)
    
    Args:
        query: The search query
        max_results: Number of results to return (default: 5, max: 10)
        country: ISO 3166-1 alpha-2 country code for geo-targeted results
                 (e.g. "DE", "US", "FR"). If not set, defaults to global results.
        include_news: Also search recent news articles (default: False).
                      News results are merged into the results list.
    
    Returns:
        A dictionary containing:
        - results: List of search results, each with:
            - title: Page title
            - url: Page URL
            - content: Snippet of page content
    
    Example:
        result = await web_search("latest news on AI regulations")
        for r in result["results"]:
            print(f"- {r['title']}: {r['url']}")
    """
    raise RuntimeError(
        "web_search will be auto-injected when running via connic CLI or after deployment. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def web_read_page(
    url: str,
    follow_redirects: bool = True,
) -> Dict[str, Any]:
    """
    Fetch a web page and return its content as markdown.

    This is a managed service - no configuration required.
    Note: Each call to web_read_page adds 1 additional billable run.
    (e.g., a run with 2 scrapes counts as 3 runs: 1 base + 2 scrapes)

    Args:
        url: The URL of the page to fetch
        follow_redirects: When True (default), follow redirects and read the
            final page. Set to False to return an error with the redirect
            target in the `redirect_url` field instead.

    Returns:
        A dictionary containing:
        - markdown: The page content as markdown
        - url: The URL that was fetched
        - error: Present when the fetch failed or the URL redirected
        - redirect_url: When the URL redirected, the target URL

    Example:
        result = await web_read_page("https://example.com/article")
        print(result["markdown"])
    """
    raise RuntimeError(
        "web_read_page will be auto-injected when running via connic CLI or after deployment. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def db_find(
    collection: str,
    filter: Optional[Dict[str, Any]] = None,
    sort: Optional[Dict[str, int]] = None,
    limit: int = 100,
    skip: int = 0,
    fields: Optional[List[str]] = None,
    distinct: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query documents from a database collection using JSON filters.

    Args:
        collection: Name of the collection to query.
        filter: JSON filter dict. Supports $eq, $ne, $gt, $gte, $lt, $lte,
                $in, $nin, $and, $or, $not, $exists, $contains, $regex.
        sort: Sort dict. 1 = ascending, -1 = descending. e.g. {"created_at": -1}
        limit: Max documents to return (default 100, max 1000).
        skip: Documents to skip for pagination.
        fields: Field paths to include. Omit for full documents.
        distinct: If set, return unique values for this field.

    Returns:
        {"documents": [...], "count": N}  or  {"values": [...], "count": N} when distinct is set

    Example:
        result = await db_find("orders", filter={"status": "active"}, sort={"created_at": -1})
    """
    raise RuntimeError(
        "db_find will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def db_insert(
    collection: str,
    documents: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Insert one or more documents into a database collection.

    Collections are created automatically if they don't exist yet.

    Args:
        collection: Collection name (lowercase, letters/digits/underscores).
        documents: A single document dict or a list of document dicts.

    Returns:
        {"inserted": [...], "inserted_count": N}

    Example:
        result = await db_insert("orders", {"product": "Widget", "qty": 5, "status": "pending"})
    """
    raise RuntimeError(
        "db_insert will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def db_update(
    collection: str,
    filter: Dict[str, Any],
    update: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update documents matching a filter in a database collection.

    The update dict is merged into each matching document (partial update).

    Args:
        collection: Collection name.
        filter: JSON filter dict to select documents to update.
        update: Fields to set or overwrite in matching documents.

    Returns:
        {"updated_ids": [...], "updated_count": N}

    Example:
        result = await db_update("orders", {"order_id": "ORD-001"}, {"status": "shipped"})
    """
    raise RuntimeError(
        "db_update will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def db_upsert(
    collection: str,
    filter: Dict[str, Any],
    update: Dict[str, Any],
    insert_only: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Update the first document matching ``filter``, or insert a new one if no
    document matches.

    The inserted document is built from (later wins): top-level equality keys
    of ``filter``, then ``update``, then ``insert_only``. Operator constructs
    in ``filter`` (``{"$gt": ...}``, ``$and``/``$or``, dot-notation keys) are
    NOT auto-extracted — put anything else that should land on a newly inserted
    document in ``insert_only``.

    Args:
        collection: Collection name. Auto-created on first write.
        filter: Non-empty JSON filter identifying the target row.
        update: Fields applied on both branches (partial merge; ``None`` removes
                a field on the update path). Cannot be empty.
        insert_only: Optional fields written ONLY when a new document is
                     inserted; ignored on the update path.

    Returns:
        {"upserted_id": "<uuid>", "operation": "inserted" | "updated"}

    Example:
        # Natural-key upsert
        await db_upsert(
            "orders",
            filter={"order_id": "ORD-001"},
            update={"status": "shipped"},
            insert_only={"source": "etl"},
        )
    """
    raise RuntimeError(
        "db_upsert will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def db_delete(
    collection: str,
    filter: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Delete documents matching a filter from a database collection.

    A non-empty filter is required to prevent accidental full-collection deletion.

    Args:
        collection: Collection name.
        filter: JSON filter dict. Must not be empty.

    Returns:
        {"deleted_ids": [...], "deleted_count": N}

    Example:
        result = await db_delete("orders", {"status": "cancelled", "order_id": "ORD-001"})
    """
    raise RuntimeError(
        "db_delete will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def db_count(
    collection: str,
    filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Count documents in a database collection.

    Args:
        collection: Collection name.
        filter: Optional JSON filter dict. Counts all documents if omitted.

    Returns:
        {"count": N}

    Example:
        result = await db_count("orders", {"status": "active"})
        print(f"Active orders: {result['count']}")
    """
    raise RuntimeError(
        "db_count will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


async def db_list_collections() -> Dict[str, Any]:
    """
    List all database collections in the current environment.

    Returns:
        {"collections": [{"name": ..., "document_count": ..., "size_bytes": ...}], "total": N}

    Example:
        result = await db_list_collections()
        for col in result["collections"]:
            print(f"{col['name']}: {col['document_count']} documents")
    """
    raise RuntimeError(
        "db_list_collections will be auto-injected when testing using the connic CLI or deploying. "
        "Run 'connic test' to test your agents with predefined tools."
    )


# All available predefined tools
__all__ = [
    "trigger_agent",
    "trigger_agent_at",
    "query_knowledge",
    "store_knowledge",
    "delete_knowledge",
    "kb_list_namespaces",
    "web_search",
    "web_read_page",
    # Database tools
    "db_find",
    "db_insert",
    "db_update",
    "db_upsert",
    "db_delete",
    "db_count",
    "db_list_collections",
]
