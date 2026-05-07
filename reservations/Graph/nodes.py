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

Your response must follow this EXACT format — no more, no less:

**<Short title>**
<One sentence explanation.>

**<Short title>**
<One sentence explanation.>

**<Short title>**
<One sentence explanation.>

**<Short title>**
<One sentence explanation.>

Rules:
- Always output exactly 4 items in the format above.
- Each title must be 2-5 words. Each explanation must be one sentence, max 15 words.
- Do NOT greet the user, ask questions, offer to create a reservation, or add any text outside the 4 items.
- Base every item on real data from the tools — never invent availability or preferences.
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
