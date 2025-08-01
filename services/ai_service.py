import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from config import supabase, claude_client, claude_msg
from embeddings import get_embedding

# Simple in-memory cache for search results
_search_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_MINUTES = 15
MAX_CACHE_SIZE = 100


def _get_cache_key(query: str, limit: int, match_threshold: float) -> str:
    """Generate cache key for search parameters"""
    return f"{query.lower().strip()}:{limit}:{match_threshold}"


def _is_cache_valid(timestamp: datetime) -> bool:
    """Check if cached result is still valid"""
    return datetime.now() - timestamp < timedelta(minutes=CACHE_TTL_MINUTES)


def _clean_cache():
    """Remove expired entries and limit cache size"""
    global _search_cache
    current_time = datetime.now()

    # Remove expired entries
    _search_cache = {
        k: v for k, v in _search_cache.items() if _is_cache_valid(v["timestamp"])
    }

    # Limit cache size (remove oldest entries)
    if len(_search_cache) > MAX_CACHE_SIZE:
        sorted_items = sorted(_search_cache.items(), key=lambda x: x[1]["timestamp"])
        _search_cache = dict(sorted_items[-MAX_CACHE_SIZE:])


async def search_vectorized_sources(
    query: str,
    limit: int = 5,
    match_threshold: float = -0.7,
    use_approximate: bool = True,
    timeout_seconds: int = 15,  # Increased timeout
) -> List[Dict[str, Any]]:
    """
    Search through vectorized sources using pgvector with approximate nearest neighbor.

    Features:
    - Fast HNSW approximate nearest neighbor search for performance
    - Timeout handling with graceful fallback
    - Result caching to avoid repeated searches
    - Fallback to keyword search if vector search fails

    Args:
        query: Search query string
        limit: Maximum number of results to return
        match_threshold: Similarity threshold (lower = more strict)
        use_approximate: Use approximate search (HNSW/IVFFlat) vs exact search
        timeout_seconds: Maximum time to wait for vector search (increased for IVFFlat)
    """
    if not supabase:
        print("[WARNING] supabase client not available")
        return []

    # Check cache first
    cache_key = _get_cache_key(query, limit, match_threshold)
    if cache_key in _search_cache and _is_cache_valid(
        _search_cache[cache_key]["timestamp"]
    ):
        print(f"[PASS] returning cached results for query: {query}")
        return _search_cache[cache_key]["results"]

    try:
        print(f"[INFO] searching vectorized sources for: {query}")
        _clean_cache()  # Clean cache periodically

        # Generate query embedding
        query_embedding, token_count = get_embedding(query)
        if not query_embedding:
            print("[ERROR] failed to generate query embedding")
            return await _fallback_keyword_search(query, limit)

        print(f"[INFO] generated embedding with {token_count} tokens")

        # Try pgvector approximate nearest neighbor search with timeout
        vector_results = await _try_vector_search_with_timeout(
            query_embedding, limit, match_threshold, use_approximate, timeout_seconds
        )

        if vector_results:
            # Cache successful results
            _search_cache[cache_key] = {
                "results": vector_results,
                "timestamp": datetime.now(),
            }
            return vector_results

        # Fallback to keyword search if vector search failed
        print("[INFO] vector search failed, falling back to keyword search")
        return await _fallback_keyword_search(query, limit)

    except Exception as e:
        print(f"[ERROR] search error: {e}")
        return await _fallback_keyword_search(query, limit)


async def _try_vector_search_with_timeout(
    query_embedding: List[float],
    limit: int,
    match_threshold: float,
    use_approximate: bool,
    timeout_seconds: int,
) -> Optional[List[Dict[str, Any]]]:
    """Try vector search with timeout handling"""
    try:
        # Use asyncio.wait_for for timeout handling
        if use_approximate:
            # Use pgvector's native operators for approximate search
            # This assumes vectorized_sources has an embedding column of type vector
            search_coro = _pgvector_approximate_search(
                query_embedding, limit, match_threshold
            )
        else:
            # Fall back to exact search (your original RPC)
            search_coro = _exact_vector_search(query_embedding, limit, match_threshold)

        result = await asyncio.wait_for(search_coro, timeout=timeout_seconds)
        return result

    except asyncio.TimeoutError:
        print(f"[ERROR] vector search timed out after {timeout_seconds} seconds")
        return None
    except Exception as e:
        print(f"[ERROR] vector search error: {e}")
        return None


async def _pgvector_approximate_search(
    query_embedding: List[float], limit: int, match_threshold: float
) -> List[Dict[str, Any]]:
    """
    Use pgvector's native operators for approximate nearest neighbor search.
    This requires that vectorized_sources table has:
    1. An 'embedding' column of type 'vector'
    2. An HNSW or IVFFlat index on the embedding column for performance
    """
    try:
        # Use RPC function to avoid URI length limits with large embedding vectors
        # Parameter order: match_count, match_threshold, query_embedding
        result = supabase.rpc(
            "search_vectorized_sources_hnsw",
            {
                "match_count": limit,
                "match_threshold": match_threshold,
                "query_embedding": query_embedding,
            },
        ).execute()

        if result.data:
            print(
                f"[PASS] pgvector approximate search found {len(result.data)} results"
            )

            # NOTE: Show details of each result
            print("[DEBUG] pgvector search results:")
            for i, row in enumerate(result.data, 1):
                # Extract key fields for debugging
                title = row.get("source_name", "N/A")
                content_preview = (
                    (row.get("content", "") or "")[:100] + "..."
                    if row.get("content")
                    else "N/A"
                )
                similarity = row.get("similarity", "N/A")
                url = row.get("url", "N/A")

                print(f"[DEBUG]   {i}. Title: {title}")
                print(f"[DEBUG]      Similarity: {similarity}")
                print(f"[DEBUG]      URL: {url}")
                print(f"[DEBUG]      Content: {content_preview}")
                print(f"[DEBUG]      ---")

            return result.data
        else:
            print("[INFO] no pgvector search results found")
            return []

    except Exception as e:
        print(f"[ERROR] pgvector search failed: {e}")
        print(
            "[INFO] falling back to exact search - run migration steps 1-3 to enable fast search"
        )
        # Fall back to exact search if HNSW RPC doesn't exist
        return await _exact_vector_search(query_embedding, limit, match_threshold)


async def _exact_vector_search(
    query_embedding: List[float], limit: int, match_threshold: float
) -> List[Dict[str, Any]]:
    """Fallback to exact vector search using RPC function"""
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": limit,
        },
    ).execute()

    if result.data:
        print(f"[PASS] exact vector search found {len(result.data)} results")
        return result.data
    else:
        print("[INFO] no exact vector search results found")
        return []


async def _fallback_keyword_search(query: str, limit: int) -> List[Dict[str, Any]]:
    """Fallback to simple keyword search when vector search fails"""
    try:
        print(f"[INFO] performing keyword fallback search for: {query}")

        # Try content search first
        response = (
            supabase.table("vectorized_sources")
            .select("id, source_name, content, url, metadata, created_at")
            .ilike("content", f"%{query}%")
            .limit(limit)
            .execute()
        )

        if response.data:
            print(f"[PASS] keyword search found {len(response.data)} results")
            return response.data

        # Try source_name search as backup
        response = (
            supabase.table("vectorized_sources")
            .select("id, source_name, content, url, metadata, created_at")
            .ilike("source_name", f"%{query}%")
            .limit(limit)
            .execute()
        )

        if response.data:
            print(f"[PASS] source name search found {len(response.data)} results")
            return response.data

        print("[INFO] no keyword search results found")
        return []

    except Exception as e:
        print(f"[ERROR] keyword search failed: {e}")
        return []


async def generate_ai_reply(
    note_content: str,
    pdf_url: str = None,
    scratchpad_context: str = None,
    use_tools: bool = True,
) -> str:
    """
    Generate AI reply using Claude with optional tool calling capabilities.

    Args:
        note_content: The user's note content
        pdf_url: Optional PDF URL for context
        scratchpad_context: Other notes on the paper
        use_tools: Whether to enable tool calling (default: True)
    """
    if not claude_client:
        return "ai reply functionality requires anthropic api key"

    try:
        from services.tool_service import get_available_tools, execute_tool

        # Build message content with text and optional PDF
        content_list = []

        if pdf_url:
            content_list.append(
                {"type": "document", "source": {"type": "url", "url": pdf_url}}
            )

        # Build context-aware prompt
        context_section = ""
        if scratchpad_context:
            context_section = f"""Here are the user's other notes on this paper:

{scratchpad_context}

"""

        prompt = f"""You are an AI assistant helping a researcher understand a paper. {context_section}The user has written the following new note:

"{note_content}"

You have access to a vectorized knowledge base of research papers and sources. Use the search_knowledge_base tool when the user's note would benefit from additional context, references, or specific information.

Provide a helpful, thoughtful response that is concise but informative. If you use the knowledge base, integrate the findings naturally into your response."""

        content_list.append({"type": "text", "text": prompt})

        # Log the text passed to Claude
        print(f"[INFO] text passed to claude: {prompt}")

        # Prepare Claude message with tool support
        message = claude_msg("user", content_list)

        # Get available tools if tool calling is enabled
        tools = get_available_tools() if use_tools else None

        if tools:
            print(f"[INFO] claude has access to {len(tools)} tools")

        # Make initial Claude request
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[message],
            tools=tools if tools else None,
        )

        # Handle tool use if Claude requests it
        if response.content and any(
            block.type == "tool_use" for block in response.content
        ):
            print("[INFO] claude is using tools...")

            # Execute tools and collect results
            tool_results = []
            messages = [message]  # Start with original message

            # Add Claude's response (which contains tool use)
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool request
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    print(f"[INFO] executing tool: {tool_name}")

                    # Execute the tool
                    tool_result = await execute_tool(tool_name, tool_input)

                    # Add tool result to messages
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(tool_result.get("result", tool_result)),
                        }
                    )

            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

                # Get Claude's final response with tool results
                final_response = claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    messages=messages,
                    tools=tools,
                )

                # Extract final text response
                final_text = ""
                for block in final_response.content:
                    if block.type == "text":
                        final_text += block.text

                return (
                    final_text.strip()
                    if final_text
                    else "Could not generate response with tool results"
                )

        # Handle regular response (no tools used)
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        return response_text.strip() if response_text else "Could not generate response"

    except Exception as e:
        print(f"[ERROR] ai reply generation error: {e}")
        return f"ai reply generation failed: {str(e)}"
