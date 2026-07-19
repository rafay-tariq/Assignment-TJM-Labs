"""LangGraph conversation state definition."""

from typing import Any, Dict, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    """State threaded through the conversation graph.

    ``messages`` uses the ``add_messages`` reducer so each node can return only
    the new messages it produced and LangGraph appends them.
    """

    # Conversation transcript (Human/AI/Tool messages).
    messages: Annotated[List[Any], add_messages]

    # Simulated caller ID for this call.
    caller_phone: str

    # "start" -> run identification + greeting; "chat" -> normal turns.
    stage: str

    # Serialised pharmacy context if recognised (see Pharmacy.to_context), else None.
    pharmacy: Optional[Dict[str, Any]]
    identified: bool

    # High-Rx-volume threshold captured into state for prompt/fallback use.
    threshold: int

    # Info gathered conversationally from unrecognised callers (fallback path).
    collected_name: Optional[str]
    collected_rx_volume: Optional[int]
