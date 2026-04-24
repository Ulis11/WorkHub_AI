"""
LangGraph WorkHub agent graph.

Responsibilities:
  - Connect to the MCP server via MultiServerMCPClient
  - Load MCP tools as LangChain-compatible tools
  - Build and compile the StateGraph

Graph topology:
                    ┌──────────────┐
         START ───► │  call_model  │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │      should_continue    │
              └────┬──────────┬─────────┘
              tool │          │ no tools
              calls│          │
                   ▼          ▼
            ┌──────────┐    END
            │call_tools│
            └────┬─────┘
                 │
                 └──────────► call_model  (loop)
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import AgentState, make_call_model_node, should_continue

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")


def build_graph(tools: list):
    """
    Compile the agent StateGraph for the given list of LangChain tools.

    Steps:
    1. Bind the tools to the LLM so it knows when to call them.
    2. Wrap the raw tool list in LangGraph's ToolNode for automatic execution.
    3. Wire call_model → should_continue → (call_tools | END) → call_model.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    llm_with_tools = llm.bind_tools(tools)

    call_model = make_call_model_node(llm_with_tools)
    tool_node = ToolNode(tools)  # handles tool dispatch + ToolMessage creation automatically

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", tool_node)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "call_tools": "call_tools",
            "__end__": END,
        },
    )
    graph.add_edge("call_tools", "call_model")

    return graph.compile()


@asynccontextmanager
async def create_agent() -> AsyncIterator:
    """
    Async context manager that:
      1. Opens a connection to the MCP server.
      2. Loads available tools.
      3. Builds + yields the compiled graph.

    Usage:
        async with create_agent() as agent:
            result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    """
    client = MultiServerMCPClient(
        {
            "workhub": {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
            }
        }
    )
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} MCP tool(s): {[t.name for t in tools]}\n")
    yield build_graph(tools)
