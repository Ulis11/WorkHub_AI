"""
FastAPI integration for the LangGraph reservation agent.

Exposes:
  POST /suggest         { "query": "..." }  →  { "result": "..." }
  POST /suggest/stream  { "query": "..." }  →  text/plain chunked stream

"""

import os
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from Graph.graph import create_agent


class SpeedInterval(BaseModel):
    startPolylinePointIndex: int
    endPolylinePointIndex: int
    speed: str   # "NORMAL" | "SLOW" | "TRAFFIC_JAM")


class RouteTravelAdvisory(BaseModel):
    speedReadingIntervals: list[SpeedInterval] | None = None


class RouteObject(BaseModel):
    """
    Matches the actual Routes API response shape forwarded by the frontend.
    All fields are optional so unknown extra fields are silently ignored.
    """
    distanceMeters: int | None = None
    durationMillis: int | None = None        # travel time with current traffic, in ms
    staticDurationMillis: int | None = None  # baseline without traffic, in ms
    travelAdvisory: RouteTravelAdvisory | None = None

    model_config = {"extra": "ignore"}       # drop path, speedPaths, etc.


def _derive_traffic_condition(intervals: list[SpeedInterval]) -> str:
    """Summarise speed intervals into a single traffic condition label."""
    if not intervals:
        return "unknown"
    jam_count = sum(1 for i in intervals if i.speed == "TRAFFIC_JAM")
    slow_count = sum(1 for i in intervals if i.speed == "SLOW")
    total = len(intervals)
    jam_ratio = jam_count / total
    if jam_ratio >= 0.4:
        return "heavy"
    elif jam_ratio >= 0.2 or slow_count / total >= 0.3:
        return "moderate"
    return "light"


def _make_bbox(lat: float, lng: float, delta: float = 0.015) -> str:
    """Return a TomTom bbox string: minLon,minLat,maxLon,maxLat (no spaces)."""
    return f"{lng - delta:.6f},{lat - delta:.6f},{lng + delta:.6f},{lat + delta:.6f}"


def _build_commute_bbox_message(origin: "Location | None", destination: "Location | None") -> str:
    """
    Pre-compute bbox strings for tomtom-traffic-incidents so the LLM never
    has to do floating-point arithmetic (which produces regex-breaking strings).
    """
    lines = [
        "Pre-computed bboxes for tomtom-traffic-incidents (minLon,minLat,maxLon,maxLat):",
        "Call the tool with the bboxes array exactly as shown below — copy the strings verbatim.",
        "",
        "bboxes: [",
    ]
    bboxes = []
    if origin is not None:
        bboxes.append(f'  {{"name": "Origin", "bbox": "{_make_bbox(origin.lat, origin.lng)}"}}')
    if origin is not None and destination is not None:
        mid_lat = (origin.lat + destination.lat) / 2
        mid_lng = (origin.lng + destination.lng) / 2
        bboxes.append(f'  {{"name": "Midpoint", "bbox": "{_make_bbox(mid_lat, mid_lng)}"}}')
    if destination is not None:
        bboxes.append(f'  {{"name": "Destination", "bbox": "{_make_bbox(destination.lat, destination.lng)}"}}')
    lines.append(",\n".join(bboxes))
    lines.append("]")
    return "\n".join(lines)


def _build_traffic_message(route: RouteObject) -> str:
    """Format a RouteObject into a concise system message for the LLM."""
    duration_min = round(route.durationMillis / 60000) if route.durationMillis is not None else None
    static_min = round(route.staticDurationMillis / 60000) if route.staticDurationMillis is not None else None
    delay_min = (duration_min - static_min) if duration_min is not None and static_min is not None else None

    intervals = (
        route.travelAdvisory.speedReadingIntervals
        if route.travelAdvisory and route.travelAdvisory.speedReadingIntervals
        else []
    )
    condition = _derive_traffic_condition(intervals)

    lines = ["Current commute data (Google Routes API):"]
    if duration_min is not None:
        lines.append(f"- Travel time with current traffic: {duration_min} min")
    if static_min is not None:
        lines.append(f"- Travel time without traffic (baseline): {static_min} min")
    if delay_min is not None:
        if delay_min > 0:
            lines.append(f"- Current traffic delay: +{delay_min} min")
        else:
            lines.append("- No traffic delay detected.")
    if route.distanceMeters is not None:
        lines.append(f"- Distance: {round(route.distanceMeters / 1000, 1)} km")
    lines.append(f"- Overall traffic condition: {condition}")
    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Keep the MCP connection and compiled graph alive for the app lifetime."""
    async with create_agent() as agent:
        app.state.agent = agent
        yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://coruscating-naiad-2204fb.netlify.app",
        "http://localhost:5173",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?" if os.getenv("DEV") else None,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


class Location(BaseModel):
    lat: float
    lng: float


class SuggestRequest(BaseModel):
    query: str
    user_id: int
    today: date | None = None      # user's local date; falls back to server date if omitted
    route: RouteObject | None = None   # Google Routes API response for the user's commute
    origin: Location | None = None     # user's current location (start of commute)
    destination: Location | None = None  # office location (end of commute)


def _build_messages(request: SuggestRequest):
    """Assemble the ordered message list for both /suggest endpoints."""
    today = request.today or date.today()
    messages = [
        SystemMessage(
            content=f"The current user ID is {request.user_id}. Today's date is {today.isoformat()}."
        )
    ]
    if request.route is not None:
        messages.append(SystemMessage(content=_build_traffic_message(request.route)))
    if request.origin is not None or request.destination is not None:
        messages.append(SystemMessage(content=_build_commute_bbox_message(request.origin, request.destination)))
    messages.append(HumanMessage(content=request.query))
    return messages


@app.post("/suggest")
async def suggest(request: SuggestRequest):
    result = await app.state.agent.ainvoke(
        {"messages": _build_messages(request)}
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
    async def token_generator():
        async for event in app.state.agent.astream_events(
            {"messages": _build_messages(request)},
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
            # Some models return a list of content blocks
            if isinstance(chunk_content, list):
                chunk_content = "".join(
                    block.get("text", "")
                    for block in chunk_content
                    if block.get("type") == "text"
                )
            if chunk_content:
                yield chunk_content

    return StreamingResponse(token_generator(), media_type="text/plain")