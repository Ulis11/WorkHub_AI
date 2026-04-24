"""
FastAPI integration for the LangGraph reservation agent.

Exposes:
  POST /suggest         { "query": "..." }  →  { "result": "..." }
  POST /suggest/stream  { "query": "..." }  →  text/plain chunked stream

"""

from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from Graph.graph import create_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Keep the MCP connection and compiled graph alive for the app lifetime."""
    async with create_agent() as agent:
        app.state.agent = agent
        yield


app = FastAPI(lifespan=lifespan)


class SuggestRequest(BaseModel):
    query: str
    user_id: int
    today: date | None = None  # user's local date; falls back to server date if omitted


@app.post("/suggest")
async def suggest(request: SuggestRequest):
    today = request.today or date.today()
    context = SystemMessage(
        content=f"The current user ID is {request.user_id}. Today's date is {today.isoformat()}."
    )
    result = await app.state.agent.ainvoke(
        {"messages": [context, HumanMessage(content=request.query)]}
    )
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    final_message = ai_messages[-1]
    content = final_message.content
    if isinstance(content, list):
        content = "\n".join(
            block["text"] for block in content if block.get("type") == "text"
        )
    return {"result": content}


@app.post("/suggest/stream")
async def suggest_stream(request: SuggestRequest):
    """
    Streams the final LLM response token-by-token as plain text.
    Tool-call intermediate steps are silent; only the final answer is streamed.
    """
    today = request.today or date.today()
    context = SystemMessage(
        content=f"The current user ID is {request.user_id}. Today's date is {today.isoformat()}."
    )

    async def token_generator():
        async for event in app.state.agent.astream_events(
            {"messages": [context, HumanMessage(content=request.query)]},
            version="v2",
        ):
            kind = event["event"]

            # Notify when a tool call is dispatched
            if kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                yield f"[Calling {tool_name}...]\n"
                continue

            if kind != "on_chat_model_stream":
                continue

            chunk_content = event["data"]["chunk"].content
            if not chunk_content:
                continue
            # Gemini may return a list of content blocks
            if isinstance(chunk_content, list):
                chunk_content = "".join(
                    block.get("text", "")
                    for block in chunk_content
                    if block.get("type") == "text"
                )
            if chunk_content:
                yield chunk_content

    return StreamingResponse(token_generator(), media_type="text/plain")
