# WorkHub AI — Reservation Suggestion Backend

An AI-powered backend that generates personalized workspace reservation suggestions for WorkHub users. It uses a LangGraph agent backed by Google Gemini, with tools served over MCP (Model Context Protocol) to query the WorkHub REST API.

---

## Architecture

```
Frontend / Postman
      │
      │  POST /suggest  { query, user_id, today }
      ▼
┌─────────────────────┐
│  FastAPI  (port 8001)│   Client/api.py
│  /suggest            │
│  /suggest/stream     │
└────────┬────────────┘
         │  LangGraph agent (ainvoke / astream_events)
         ▼
┌─────────────────────┐
│   LangGraph Graph   │   Graph/graph.py + Graph/nodes.py
│   call_model ──────►│──► Gemini (gemini-3.1-flash-lite-preview)
│       │  ▲          │
│   call_tools        │
└────────┬────────────┘
         │  MCP tools via langchain-mcp-adapters
         ▼
┌─────────────────────┐
│  FastMCP Server     │   Server/mcp_server.py  (port 8000)
│  get_user_preferences│
│  get_availability   │
│  get_reservation_   │
│    history          │
└────────┬────────────┘
         │  HTTP
         ▼
  WorkHub REST API  (port 5500)
```

---

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- A Google Gemini API key
- The WorkHub REST API running on `http://localhost:5500`

---

## Setup

**1. Install dependencies**
```bash
cd reservations
uv sync
```

**2. Create a `.env` file** in the `reservations/` folder:
```env
GEMINI_API_KEY=your_key_here

# Optional overrides (these are the defaults)
MCP_SERVER_URL=http://127.0.0.1:8000/mcp
GEMINI_MODEL=gemini-3.1-flash-lite-preview
```

**3. Create the system prompt file** at `reservations/prompts/system_prompt.txt`:

```
You are a WorkHub reservation assistant.
Always call get_user_preferences and get_reservation_history before forming any suggestion.
Call get_availability for each day you are considering.

Your response must follow this EXACT format — no more, no less:

**<Short title>**
<One sentence explanation.>

...repeated 4 times...

Rules:
- Always output exactly 4 items in the format above.
- Each title must be 2-5 words. Each explanation must be one sentence, max 15 words.
- Do NOT greet the user, ask questions, offer to create a reservation, or add any text outside the 4 items.
- Base every item on real data from the tools — never invent availability or preferences.
```

> `prompts/` is listed in `.gitignore` and will not be committed. This keeps the agent's instruction strategy out of version control.

---

## Running

Both processes must be running at the same time, started from the `reservations/` directory.

**Terminal 1 — MCP server**
```bash
uv run Server/mcp_server.py
```

**Terminal 2 — FastAPI app**
```bash
uv run uvicorn Client.api:app --reload --port 8001
```

---

## API

### `POST /suggest`

Returns a single AI-generated suggestion block.

**Request body**
```json
{
  "query": "Suggest a reservation for me next week",
  "user_id": 12,
  "today": "2026-04-24"
}
```

- `query` — natural language instruction
- `user_id` — numeric WorkHub user ID; used by all MCP tools
- `today` *(optional)* — user's local date in `YYYY-MM-DD`; falls back to server date if omitted

**Response**
```json
{
  "result": "**Best Day This Week**\nTuesday has your preferred Zona Silenciosa desk available all morning.\n\n..."
}
```

The `result` string always contains exactly **4 items** in this format:
```
**<Title>**
<One sentence explanation.>
```

---

### `POST /suggest/stream`

Same request body as `/suggest`. Returns the response as a plain-text chunked stream (`text/plain`). Intermediate tool call notifications are included in the stream as `[Calling <tool_name>...]` lines.

---

## MCP Tools

Defined in `Server/mcp_server.py`, served over Streamable HTTP at `http://127.0.0.1:8000/mcp`.

| Tool | Endpoint hit | Purpose |
|---|---|---|
| `get_user_preferences(id)` | `GET /api/preferencias/inferidas/{id}` | Fetches inferred preferences (zone, space type, days, arrival time) |
| `get_availability(date)` | `GET /api/reservas/disponibilidad?date=` | Lists all available spaces for a given date |
| `get_reservation_history(user_id)` | `GET /api/preferencias/historial/{id}` | Last 10 confirmed reservations for behavioral pattern analysis |

All tool responses are sanitized against indirect prompt injection before being returned to the LLM.

---

## Security

- **Indirect prompt injection guard** — all WorkHub API responses are recursively scanned for instruction-like patterns (`ignore previous instructions`, `SYSTEM:`, `act as`, etc.) and redacted before reaching the LLM.
- **User ID scoping** — `user_id` is injected via a trusted `SystemMessage` from the FastAPI layer, not derived from the user query, so the LLM cannot be manipulated into fetching another user's data.
- **Date injection** — today's date is also injected via `SystemMessage` so the LLM resolves relative dates (`tomorrow`, `next week`) correctly regardless of training cutoff.

---

## Project Structure

```
reservations/
├── Client/
│   ├── api.py          # FastAPI app — /suggest and /suggest/stream endpoints
│   └── main.py         # CLI entry point for direct agent queries
├── Graph/
│   ├── graph.py        # LangGraph StateGraph definition and MCP client setup
│   └── nodes.py        # Agent nodes, system prompt loader, routing logic
├── Server/
│   └── mcp_server.py   # FastMCP server — WorkHub MCP tools + injection sanitizer
├── prompts/
│   └── system_prompt.txt  # NOT committed — contains the agent's instruction strategy
├── pyproject.toml
└── .env                # NOT committed — contains GEMINI_API_KEY
```
