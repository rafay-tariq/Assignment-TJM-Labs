"""The LangGraph conversation graph.

Graph shape::

              ┌──────────┐
    START ──▶ │ identify │ (only on call start)
      │       └────┬─────┘
      │            ▼
      │        ┌───────┐
      │        │ greet │ ──▶ END
      │        └───────┘
      │
      └──▶ ┌───────┐            (subsequent turns)
           │ agent │ ─┬─▶ END
           └───────┘  │
              ▲       ▼
           ┌───────────┐
           │   tools   │   (LLM path only)
           └───────────┘

On the first invocation ``stage == "start"`` routes through identification and
greeting. Every later turn routes straight to the agent, which either calls the
LLM (with follow-up tools bound) or the deterministic fallback.
"""

from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import settings
from .fallback import fallback_agent_node, fallback_greeting
from .llm import get_llm, llm_available
from .logging_config import get_logger
from .models import Pharmacy
from .pharmacy_api import identify_pharmacy_by_phone
from .prompts import GREETING_INSTRUCTION, build_system_content
from .state import AgentState
from .tools import FOLLOW_UP_TOOLS

logger = get_logger("pharmacy_agent.graph")


# --------------------------------------------------------------------------- #
# Nodes                                                                        #
# --------------------------------------------------------------------------- #
def identify_node(state: AgentState) -> Dict[str, Any]:
    """Look up the caller by phone and stash grounded context in state."""
    phone = state.get("caller_phone", "")
    pharmacy: Pharmacy = identify_pharmacy_by_phone(phone)
    threshold = settings.high_rx_volume_threshold

    if pharmacy is not None:
        return {
            "pharmacy": pharmacy.to_context(threshold),
            "identified": True,
            "threshold": threshold,
        }

    return {
        "pharmacy": None,
        "identified": False,
        "threshold": threshold,
    }


def greet_node(state: AgentState) -> Dict[str, Any]:
    """Produce the opening line and flip the conversation into chat mode."""
    pharmacy = state.get("pharmacy")
    threshold = state.get("threshold", settings.high_rx_volume_threshold)

    if llm_available():
        llm = get_llm()
        # The greeting instruction is ephemeral: we don't persist it, only the
        # resulting greeting, so the trigger never pollutes the transcript.
        response = llm.invoke(
            [
                SystemMessage(content=build_system_content(pharmacy, threshold)),
                HumanMessage(content=GREETING_INSTRUCTION),
            ]
        )
        greeting = response.content
    else:
        greeting = fallback_greeting(pharmacy, threshold)

    logger.info("Call greeting delivered (recognised=%s)", bool(pharmacy))
    return {"messages": [AIMessage(content=greeting)], "stage": "chat"}


def llm_agent_node(state: AgentState) -> Dict[str, Any]:
    """One agent turn driven by Claude, with follow-up tools bound."""
    llm = get_llm().bind_tools(FOLLOW_UP_TOOLS)
    system_content = build_system_content(
        state.get("pharmacy"),
        state.get("threshold", settings.high_rx_volume_threshold),
    )
    conversation = [SystemMessage(content=system_content)] + list(state["messages"])
    response = llm.invoke(conversation)
    return {"messages": [response]}


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #
def route_entry(state: AgentState) -> str:
    """Entry router: identify+greet on call start, else go straight to agent."""
    return "identify" if state.get("stage") == "start" else "agent"


# --------------------------------------------------------------------------- #
# Graph assembly                                                               #
# --------------------------------------------------------------------------- #
def build_graph():
    """Compile the conversation graph, choosing LLM vs deterministic nodes."""
    use_llm = llm_available()
    logger.info("Building conversation graph (llm=%s)", use_llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("identify", identify_node)
    workflow.add_node("greet", greet_node)

    if use_llm:
        from langgraph.prebuilt import ToolNode, tools_condition

        workflow.add_node("agent", llm_agent_node)
        workflow.add_node("tools", ToolNode(FOLLOW_UP_TOOLS))
    else:
        workflow.add_node("agent", fallback_agent_node)

    workflow.add_conditional_edges(
        START, route_entry, {"identify": "identify", "agent": "agent"}
    )
    workflow.add_edge("identify", "greet")
    workflow.add_edge("greet", END)

    if use_llm:
        workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_edge("agent", END)

    # An in-memory checkpointer lets us drive the graph one turn at a time while
    # preserving conversation state across invocations (keyed by thread_id).
    return workflow.compile(checkpointer=MemorySaver())
