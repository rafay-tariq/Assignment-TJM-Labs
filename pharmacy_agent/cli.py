"""Text-based REPL that simulates an inbound phone call.

Run with ``python main.py``. The caller's phone number is a mock (configurable
via ``--phone`` or ``MOCK_CALLER_PHONE``); on "call start" the agent identifies
the pharmacy, greets, and then converses turn by turn.
"""

import argparse
from typing import Any

from langchain_core.messages import HumanMessage

from .config import settings
from .graph import build_graph
from .llm import llm_available
from .logging_config import get_logger

logger = get_logger("pharmacy_agent.cli")

_EXIT_WORDS = {"quit", "exit", "bye", "goodbye", "hang up", "q"}
_THREAD_ID = "call-session-1"


def _message_text(message: Any) -> str:
    """Coerce a message's content (str or Anthropic block list) into text."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(p for p in parts if p).strip()
    return str(content)


def _print_agent(state: dict) -> None:
    """Print the agent's most recent message from a graph result."""
    messages = state.get("messages", [])
    if not messages:
        return
    text = _message_text(messages[-1])
    if text:
        print(f"\nAgent: {text}")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inbound Pharmacy Sales Agent (text simulation)."
    )
    parser.add_argument(
        "--phone",
        default=settings.mock_caller_phone,
        help="Mock caller phone number to identify the pharmacy.",
    )
    return parser.parse_args(argv)


def run_cli(argv=None) -> None:
    """Entry point for the interactive simulation."""
    args = _parse_args(argv)
    graph = build_graph()
    config = {"configurable": {"thread_id": _THREAD_ID}}

    mode = "Claude LLM" if llm_available() else "deterministic fallback (no API key)"
    print("=" * 68)
    print("  TJM Labs - Inbound Pharmacy Sales Agent (simulation)")
    print(f"  Engine: {mode}")
    print(f"  Incoming call from: {args.phone}")
    print("  Type your reply and press Enter. Type 'quit' to hang up.")
    print("=" * 68)

    logger.info("Call started from %s", args.phone)

    # Call start: identify + greet.
    result = graph.invoke(
        {"caller_phone": args.phone, "stage": "start", "messages": []},
        config,
    )
    _print_agent(result)

    # Conversation loop.
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nCall ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in _EXIT_WORDS:
            print("\nAgent: Thanks for calling TJM Labs. Have a great day!")
            print("\nCall ended.")
            logger.info("Call ended by caller")
            break

        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        _print_agent(result)
