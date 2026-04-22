"""
Gemini MCP Client
Connects to the running MCP weather server and uses Gemini as the LLM,
routing tool calls back through the MCP session.

Usage:
    1. Start the MCP server:  python mcp_server.py
    2. Set your API key:      $env:GEMINI_API_KEY = "your-key"
    3. Run this client:       python gemini_client.py
"""

import asyncio
import os

from google import genai
from google.genai import types as genai_types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv()

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


# Fields not supported by Gemini's Schema proto
_UNSUPPORTED_SCHEMA_FIELDS = {"title", "default", "examples", "$schema", "$id", "additionalProperties"}


def _sanitize_schema(schema: dict) -> dict:
    """Recursively remove JSON Schema fields that Gemini's SDK does not support."""
    if not isinstance(schema, dict):
        return schema
    return {
        k: (_sanitize_schema(v) if isinstance(v, dict) else v)
        for k, v in schema.items()
        if k not in _UNSUPPORTED_SCHEMA_FIELDS
    }


def mcp_tool_to_gemini(tool) -> dict:
    """Convert an MCP tool definition to a Gemini function declaration."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": _sanitize_schema(dict(tool.inputSchema)),
    }


async def run_agent(user_query: str) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools exposed by the MCP server
            tools_response = await session.list_tools()
            gemini_tools = [mcp_tool_to_gemini(t) for t in tools_response.tools]
            print(f"Loaded {len(gemini_tools)} tool(s): {[t['name'] for t in gemini_tools]}\n")

            tool_config = genai_types.Tool(
                function_declarations=[
                    genai_types.FunctionDeclaration(**t) for t in gemini_tools
                ]
            )

            messages: list[genai_types.ContentUnion] = [user_query]

            # Agentic loop — keep going until Gemini stops requesting tool calls
            while True:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=messages,
                    config=genai_types.GenerateContentConfig(tools=[tool_config]),
                )

                fn_calls = [
                    part.function_call
                    for part in response.candidates[0].content.parts
                    if part.function_call is not None
                ]

                if not fn_calls:
                    print("\nFinal answer:")
                    print(response.text)
                    break

                # Append model turn to history
                messages.append(response.candidates[0].content)

                # Execute each tool call via the MCP session
                fn_response_parts = []
                for fn_call in fn_calls:
                    args = dict(fn_call.args)
                    print(f"  -> Calling MCP tool '{fn_call.name}' with args {args}")
                    result = await session.call_tool(fn_call.name, args)
                    tool_output = result.content[0].text if result.content else ""
                    fn_response_parts.append(
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=fn_call.name,
                                response={"result": tool_output},
                            )
                        )
                    )

                # Append tool results as a user turn and continue
                messages.append(genai_types.Content(role="user", parts=fn_response_parts))


if __name__ == "__main__":
    query = input("Ask about US weather: ").strip()
    if not query:
        query = "What is the weather forecast for San Francisco, CA?"
    asyncio.run(run_agent(query))
