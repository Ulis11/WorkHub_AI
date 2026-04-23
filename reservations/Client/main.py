"""
Entry point for the LangGraph weather agent.

Usage:
    1. Start the MCP server:  python mcp_server.py
    2. Set your API key:      $env:GEMINI_API_KEY = "your-key"  (or use .env)
    3. Run this agent:        python main.py
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from Graph.graph import create_agent


async def run(query: str) -> None:
    async with create_agent() as agent:
        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        final_message = ai_messages[-1]
        content = final_message.content
        if isinstance(content, list):
            content = "\n".join(block["text"] for block in content if block.get("type") == "text")
        print("\nFinal answer:")
        print(content)


if __name__ == "__main__":
    query = input("Ask about US weather: ").strip()
    if not query:
        query = "What are the current weather alerts in CA?"
    asyncio.run(run(query))
