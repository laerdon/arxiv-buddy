"""
Tool Service for Claude Integration
Handles tool definitions, registration, and execution for Claude's tool calling capabilities.
"""

from typing import Dict, Any, List, Callable, Optional
import json
from services.ai_service import search_vectorized_sources

# Tool registry to store available tools
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_tool(name: str, schema: Dict[str, Any], handler: Callable):
    """Register a tool with its schema and handler function"""
    TOOL_REGISTRY[name] = {"schema": schema, "handler": handler}
    print(f"[INFO] registered tool: {name}")


async def execute_tool(name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with given parameters"""
    if name not in TOOL_REGISTRY:
        return {"error": f"Tool '{name}' not found"}

    try:
        handler = TOOL_REGISTRY[name]["handler"]
        result = await handler(**parameters)
        return {"success": True, "result": result}
    except Exception as e:
        print(f"[ERROR] tool execution failed for {name}: {e}")
        return {"error": str(e)}


def get_available_tools() -> List[Dict[str, Any]]:
    """Get list of available tools for Claude"""
    return [tool_data["schema"] for tool_data in TOOL_REGISTRY.values()]


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

# Vector Search Tool Schema for Claude
VECTOR_SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": """Search through the vectorized research knowledge base when the user's question would benefit from additional context, references, or specific information that might not be in the current conversation. 

Use this tool when:
- User asks about specific topics, concepts, or research areas
- User requests examples, references, or detailed explanations  
- User mentions wanting to learn more about something
- The question would benefit from academic or research context
- You need to verify or supplement information

Do NOT use for:
- Simple conversational responses
- Questions you can answer well without additional context
- General writing/editing tasks
- Mathematical calculations""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query - be specific and use key terms that would appear in academic/research content",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (1-10, default: 5)",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
            "search_context": {
                "type": "string",
                "description": "Brief context about why you're searching (helps with result interpretation)",
                "default": "general_research",
            },
        },
        "required": ["query"],
    },
}

# =============================================================================
# TOOL HANDLERS
# =============================================================================


async def handle_vector_search(
    query: str, limit: int = 5, search_context: str = "general_research"
) -> Dict[str, Any]:
    """
    Handle vector search tool calls from Claude
    Returns formatted results that Claude can easily use
    """
    print(
        f"[INFO] claude requested vector search: '{query}' (context: {search_context})"
    )

    try:
        # Execute the search
        search_results = await search_vectorized_sources(
            query=query, limit=limit, use_approximate=True, timeout_seconds=10
        )

        if not search_results:
            return {
                "status": "no_results",
                "message": "No relevant results found in the knowledge base",
                "query": query,
                "results": [],
            }

        # Format results for Claude
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            formatted_result = {
                "rank": i,
                "title": result.get("source_name", "Unknown Source"),
                "content": (
                    result.get("content", "")[:500] + "..."
                    if len(result.get("content", "")) > 500
                    else result.get("content", "")
                ),
                "url": result.get("url"),
                "similarity_score": result.get("similarity", "N/A"),
                "metadata": result.get("metadata"),
            }
            formatted_results.append(formatted_result)

        return {
            "status": "success",
            "message": f"Found {len(search_results)} relevant sources",
            "query": query,
            "search_context": search_context,
            "results": formatted_results,
            "total_results": len(search_results),
        }

    except Exception as e:
        print(f"[ERROR] vector search tool failed: {e}")
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "query": query,
            "results": [],
        }


# =============================================================================
# TOOL REGISTRATION
# =============================================================================


def initialize_tools():
    """Initialize and register all available tools"""
    register_tool("search_knowledge_base", VECTOR_SEARCH_TOOL, handle_vector_search)
    print(f"[PASS] initialized {len(TOOL_REGISTRY)} tools for claude")


# Auto-initialize when module is imported
initialize_tools()
