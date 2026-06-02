"""
Agent nodes for the LangGraph weather agent.

Each node is a plain async function that receives the current AgentState
and returns a dict with the updated state keys.

Node factories are used so nodes can close over the LLM and tools
without relying on global state.
"""

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import MessagesState

# Re-export MessagesState as AgentState for clarity in graph.py
AgentState = MessagesState

SYSTEM_PROMPT = """You are a WorkHub reservation assistant.
Always call get_user_preferences and get_reservation_history before forming any suggestion.
Call get_availability for each day you are considering.

Your response must follow this EXACT valid JSON format — no more, no less:

{
  "suggestions": [
    {
      "box_title": "<type of suggestion 1>",
      "items": [
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"}
      ]
    },
    {
      "box_title": "<type of suggestion 2>",
      "items": [
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"}
      ]
    },
    {
      "box_title": "<type of suggestion 3>",
      "items": [
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"}
      ]
    }
  ],
  "traffic_suggestions": [
    {
      "box_title": "<traffic-related suggestion type>",
      "items": [
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"},
        {"item_title": "<short title>", "item_explanation": "<one sentence explanation>"}
      ]
    }
  ]
}

TRAFFIC DATA:
- If a "Current commute data" system message is present in the conversation, the Google Routes
  API has provided live route information for the user's current commute.
- "Travel time with current traffic" (durationMillis converted to minutes) is the actual
  travel time right now under real traffic conditions.
- "Travel time without traffic" (staticDurationMillis converted to minutes) is the baseline
  with no congestion at all.
- The difference between them is the current delay — use it to judge severity.
- "Overall traffic condition" is pre-computed from speed intervals along the route:
    light    → mostly free-flowing, minimal jams
    moderate → some congestion, noticeable slowdowns
    heavy    → significant jams covering a large portion of the route
- All users commute by car — traffic data is always relevant when present.
- When traffic data IS present, output exactly 1 box in traffic_suggestions
  with exactly 4 items. Suggest optimal departure/arrival times, flag heavy delays,
  or recommend remote work if the condition is "heavy" and delay exceeds 15 min.
- When traffic data is NOT present, return: "traffic_suggestions": []

Rules:
- Always output exactly 3 boxes in "suggestions", each with exactly 4 items.
- Always output the "traffic_suggestions" key — use an empty array when not applicable.
- Each title must be 2-5 words. Each explanation must be one sentence, max 15 words.
- Do NOT greet the user, ask questions, offer to create a reservation, or add any text outside the JSON.
- Base every item on real data from the tools — never invent availability or preferences of a user.
"""

def make_call_model_node(llm_with_tools):
    """
    Returns a `call_model` node bound to the given LLM+tools.

    The LLM already has tools bound to it, so it will automatically
    emit tool_calls when it decides to use a tool.
    """
    async def call_model(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        if isinstance(response, AIMessage) and response.tool_calls:
            for tc in response.tool_calls:
                print(f"[tool call] {tc['name']}({tc['args']})")
        return {"messages": [response]}

    return call_model


def should_continue(state: AgentState) -> str:
    """
    Conditional edge router.

    Returns 'call_tools' if the last model message contains tool calls,
    otherwise returns '__end__' to finish the run.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tools"
    return "__end__"
