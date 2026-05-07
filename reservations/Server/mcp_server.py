import json
import re
import os
from dotenv import load_dotenv
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

load_dotenv()
# Initialize FastMCP server
mcp = FastMCP("workhub", json_response=True)

# Constants
WORKHUB_API_BASE = os.getenv("WORKHUB_API_BASE")

# Patterns that resemble LLM instruction syntax — stripped from API responses
# to prevent indirect prompt injection via user-controlled data fields.
_INJECTION_PATTERN = re.compile(
    r"(ignore\s+(previous|all|prior)|"
    r"system\s*:|"
    r"you\s+are\s+(now|a|an)|"
    r"forget\s+(all|previous|prior|your)|"
    r"disregard\s+(all|previous|prior)|"
    r"new\s+instruction|"
    r"act\s+as|"
    r"jailbreak|"
    r"<\s*(system|instruction|prompt)\s*>)",
    re.IGNORECASE,
)

def _sanitize(value: Any) -> Any:
    """
    Recursively walk API response data and redact any string values that
    contain instruction-like patterns (indirect prompt injection guard).
    """
    if isinstance(value, str):
        if _INJECTION_PATTERN.search(value):
            return "[redacted]"
        return value
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value

# Helper functions/ querying and formatting WorkHub API data
async def make_workhub_request(endpoint: str) -> dict[str, Any] | None:
    """Make a request to the WorkHub API with proper error handling."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{WORKHUB_API_BASE}/{endpoint}", timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

# MCP tool definitions
@mcp.tool()
async def get_user_preferences(id: int) -> Any:
    """
    Retrieves the workspace preferences for a given user.
    
    WHEN TO CALL:
    - Always call this as the first tool before forming any suggestion.
    - This is the foundation of every suggestion — never skip it.
    - Call it once per request, result stays valid for the entire session.
    
    WHAT IT RETURNS:
    - preferred_zone: the zone the user prefers to sit in (e.g. 'Zona Silenciosa')
    - preferred_space_type: type of space they prefer (e.g. 'Escritorio', 'Pod')
    - preferred_days: list of days they prefer to come in (e.g. ['lunes', 'miercoles'])
    - usual_arrival_time: what time they usually arrive (e.g. '09:00')
    - usual_leave_time: what time they usually leave (e.g. '18:00')
    - transport_mode: how they commute ('carro', 'transporte_publico', 'bicicleta', 'a pie')
    - nombre: user's first name — use this to personalize the response greeting
    - apellido: user's last name
    
    NOTE: If preferences have not been explicitly set by the user, values may be
    inferred from their reservation history. Treat inferred values as soft preferences.
    
    RELATIONSHIP TO OTHER TOOLS:
    - Use preferred_zone and preferred_space_type to filter results from
      check_availability — only show spaces that match what the user likes.
    - Use preferred_days as the baseline set of days to verify with
      check_availability before suggesting them.
    - Use nombre to greet the user by name in the final suggestion.

    Args:
        id: User ID
    """
    data = await make_workhub_request(f"preferencias/inferidas/{id}")
    if not data:
        return "Unable to fetch user preferences."

    return json.dumps(_sanitize(data))

@mcp.tool()
async def get_availability(date: str) -> Any:
    """
    Returns all available and active spaces for a given date,
    excluding any spaces that already have a confirmed reservation.
    
    WHEN TO CALL:
    - Call this for every day you are considering suggesting to the user.
    - Never suggest a day without first verifying availability — a day with
      no spaces matching the user's preferences is not a good suggestion.
    - Call multiple times if checking several days at once
      (e.g. checking all 5 days of next week).
    - Always call get_user_preferences and get_reservation_history first
      so you can filter results meaningfully.
    
    PARAMETERS:
    - date: the date to check in YYYY-MM-DD format (e.g. '2026-04-28')
    
    WHAT IT RETURNS:
    A list of available spaces, each containing:
    - id_espacio: space identifier
    - codigo_espacio: space code
    - nombre_espacio: human readable space name
    - estado_actual: current space status (should be DISPONIBLE)
    - nombre_tipo: space type (e.g. 'Escritorio', 'Pod', 'Sala de reuniones')
    - nombre_zona: zone name (e.g. 'Zona Silenciosa', 'Zona Colaborativa')
    - edificio: building name
    
    WHAT TO DO WITH RESULTS:
    - Filter by preferred_zone from get_user_preferences to find spaces
      the user will actually enjoy.
    - Filter by preferred_space_type from get_user_preferences or from
      patterns found in get_reservation_history.
    - If no spaces match the user's preferences on a given day, flag this
      in the suggestion — do not recommend that day as a top pick.
    - If the list is empty, that day is fully booked — do not suggest it.
    
    RELATIONSHIP TO OTHER TOOLS:
    - Always call get_user_preferences before this so you know which
      zone and space type to prioritize in the results.
    - Always call get_reservation_history before this so you know which
      specific spaces and zones the user gravitates toward historically.
    - These three tools together form the complete reasoning chain —
      preferences and history define what good looks like,
      availability confirms whether it is actually possible.

    Args:
        date: Date in YYYY-MM-DD format
    """
    data = await make_workhub_request(f"reservas/disponibilidad?date={date}")
    if not data:
        return "Unable to fetch availability data."

    return json.dumps(_sanitize(data))

@mcp.tool()
async def get_reservation_history(user_id: int) -> Any:
    """
    Retrieves the last 10 confirmed reservations for a given user,
    including full space, zone and space type details for each.
    
    WHEN TO CALL:
    - Always call this alongside get_user_preferences before forming any suggestion.
    - Use this to understand the user's real behavior, not just their stated preferences.
    - Especially useful when the user asks for suggestions based on their habits
      (e.g. 'what do I usually do?', 'suggest something like last week').
    
    WHAT IT RETURNS:
    A list of reservations, each containing:
    - fecha_reserva: date of the reservation
    - hora_inicio / hora_fin: time block reserved
    - estado_reserva: current status (ACTIVO, CHECKED_IN, CHECKED_OUT, etc.)
    - tipo_reserva: type of reservation made
    - check_in / check_out: actual arrival and departure timestamps if checked in
    - nombre_espacio: name of the space reserved
    - codigo_espacio: space code identifier
    - nombre_tipo: type of space (e.g. 'Escritorio', 'Sala de reuniones')
    - nombre_zona: zone where the space is located
    - edificio: building name
    
    WHAT TO LOOK FOR:
    - Recurring days of the week → user's natural pattern
    - Most booked zone and space type → real preference vs stated preference
    - Frequent check_in times → when they actually arrive
    - No show patterns (EXPIRADO with no check_in) → days they tend to cancel
    - Gaps in history → days they consistently avoid
    
    RELATIONSHIP TO OTHER TOOLS:
    - Patterns found here should reinforce or override values from
      get_user_preferences when they conflict — real behavior beats stated preference.
    - Cross reference the most booked zone and space type with check_availability
      results to find spaces the user will actually want on suggested days.
    - If history shows recurring no shows on a specific day, avoid suggesting
      that day even if check_availability shows it as free.

    Args:
        user_id: User ID
    """
    data = await make_workhub_request(f"preferencias/historial/{user_id}")
    if not data:
        return "Unable to fetch reservation history."

    return json.dumps(_sanitize(data))

# Main entry point / server startup
def main():
    # Render (and most PaaS) assigns a PORT env var; fall back to 8000 locally.
    port = int(os.getenv("PORT", 8000))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()