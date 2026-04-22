"""
Agent nodes for the LangGraph weather agent.

Each node is a plain async function that receives the current AgentState
and returns a dict with the updated state keys.

Node factories are used so nodes can close over the LLM and tools
without relying on global state.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState

# Re-export MessagesState as AgentState for clarity in graph.py
AgentState = MessagesState


def make_call_model_node(llm_with_tools):
    """
    Returns a `call_model` node bound to the given LLM+tools.

    The LLM already has tools bound to it, so it will automatically
    emit tool_calls when it decides to use a tool.
    """
    async def call_model(state: AgentState) -> dict:
        response = await llm_with_tools.ainvoke(state["messages"])
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
