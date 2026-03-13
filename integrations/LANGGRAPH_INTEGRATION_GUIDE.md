# LangGraph Integration Guide

OpenStinger v0.7 natively supports **LangGraph** (and any other orchestrator) via the Model Context Protocol (MCP). No custom Python adapters or LangChain wrappers are required.

Because OpenStinger exposes its graph memory via MCP, you simply point LangGraph's `mcp-client` at OpenStinger's SSE endpoint and pass the resulting tools into your `ToolNode`.

## 1. Start OpenStinger

Start the OpenStinger MCP server in network mode (SSE transport):

```bash
openstinger-mcp --config .openstinger/config.yaml --transport sse --port 8766
```

## 2. Connect LangGraph

In your LangGraph application, use the official `mcp` and `langchain-mcp` packages to connect to the SSE endpoint.

```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters import get_mcp_tools
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

async def main():
    # 1. Connect to OpenStinger MCP Server
    async with sse_client("http://localhost:8766/sse") as (read_ctx, write_ctx):
        async with ClientSession(read_ctx, write_ctx) as session:
            await session.initialize()
            
            # 2. Extract OpenStinger Tools
            # This automatically gives your LangGraph agent:
            # - memory_add, memory_search, memory_query
            # - vault_note_add, vault_notes_list
            tools = await get_mcp_tools(session)
            
            # 3. Create your Tool Node
            tool_node = ToolNode(tools)
            
            # ... define your LLM node, your state, and build your graph ...
            
            # Example graph layout:
            # graph.add_node("agent", llm_node)
            # graph.add_node("tools", tool_node)
            # graph.add_conditional_edges("agent", should_continue, ["tools", END])
            # graph.add_edge("tools", "agent")
```

## 3. The "Naked Agent" Advantage

With OpenStinger v0.7, you **do not** need to hardcode a massive system prompt into your LangGraph LLM defining its identity and rules.

If you have a `system_prompt.txt` or `agent.yaml` in your workspace, OpenStinger's background `AgentProfileIngester` will automatically parse it, extract your identity, and anchor it in the FalkorDB Vault.

When your LangGraph agent boots up, its first tool call to `vault_notes_list` or `memory_search` will return its complete identity perfectly intact.
