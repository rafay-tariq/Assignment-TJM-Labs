"""Deterministic, no-LLM conversation logic.

This is the graceful fallback used when no Anthropic API key is configured. It
is intentionally simple: rule/keyword based, with fully templated replies. That
templating is also its safety guarantee — because every response is canned, the
fallback cannot hallucinate facts.
"""

import re
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage

from .logging_config import get_logger
from .prompts import TJM_LABS_VALUE_PROP
from .tools import _schedule_callback, _send_email_followup

logger = get_logger("pharmacy_agent.fallback")

_GREETING_WORDS = {"hi", "hello", "hey", "yo", "hiya", "greetings"}
_EMAIL_INTENT = ("email", "e-mail", "send me", "send info", "brochure", "details")
_CALLBACK_INTENT = ("call me", "callback", "call back", "schedule", "phone me", "ring me")
_PITCH_INTENT = ("tell me more", "what do you", "services", "pricing", "how do you", "help", "offer")


def _last_user_text(state: Dict[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        # HumanMessage has type == "human"; be tolerant of shapes.
        if getattr(message, "type", None) == "human":
            return str(message.content)
    return ""


def _extract_rx_volume(text: str) -> Optional[int]:
    """Pull the first integer (allowing commas) out of free text."""
    match = re.search(r"(\d[\d,]{0,6})", text)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _looks_like_greeting_only(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return bool(words) and all(w in _GREETING_WORDS for w in words)


def _contains_any(text: str, needles) -> bool:
    return any(n in text for n in needles)


def _pitch(name: Optional[str], is_high_volume: bool, threshold: int) -> str:
    who = f"{name}, " if name else ""
    if is_high_volume:
        return (
            f"{who}since you're running a high prescription volume, TJM Labs is a "
            f"strong fit. We help high-Rx pharmacies with bulk fulfillment, "
            f"volume-based pricing that improves margins, fill-trend analytics, and "
            f"a dedicated account manager. Would you like me to email you the "
            f"details or schedule a callback with a specialist?"
        )
    return (
        f"{who}TJM Labs helps pharmacies scale as their prescription volume grows — "
        f"faster fulfillment, better pricing as you grow, and analytics on your "
        f"fill trends. Would you like me to email an overview or set up a callback?"
    )


def fallback_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Produce one agent turn without an LLM.

    Returns a partial state update (new message plus any collected fields).
    """
    text = _last_user_text(state)
    lowered = text.lower()
    pharmacy = state.get("pharmacy")
    threshold = state.get("threshold", 100)

    name = pharmacy.get("name") if pharmacy else state.get("collected_name")
    phone = (pharmacy.get("phone") if pharmacy else state.get("caller_phone")) or ""
    email = pharmacy.get("email") if pharmacy else None
    is_high_volume = (
        pharmacy.get("is_high_volume")
        if pharmacy
        else (state.get("collected_rx_volume") or 0) >= threshold
    )

    updates: Dict[str, Any] = {}

    # 1) Explicit follow-up intents take priority.
    if _contains_any(lowered, _CALLBACK_INTENT):
        confirmation = _schedule_callback(name or "your pharmacy", phone, "your earliest convenience")
        reply = f"{confirmation} Is there anything else I can help with?"
        return {**updates, "messages": [AIMessage(reply)]}

    if _contains_any(lowered, _EMAIL_INTENT):
        confirmation = _send_email_followup(
            email or "your pharmacy's email",
            name or "your pharmacy",
            "Overview of how TJM Labs supports high-Rx-volume pharmacies.",
        )
        reply = f"{confirmation} Anything else I can do for you?"
        return {**updates, "messages": [AIMessage(reply)]}

    # 2) Collect info from unrecognised callers.
    if not pharmacy:
        if not name:
            if _looks_like_greeting_only(text) or not text.strip():
                reply = "Hi there! I don't have your number on file — may I have your pharmacy's name?"
            else:
                candidate = text.strip().rstrip(".")
                updates["collected_name"] = candidate
                reply = (
                    f"Thanks — great to meet you, {candidate}. Roughly how many "
                    f"prescriptions do you fill per month?"
                )
            return {**updates, "messages": [AIMessage(reply)]}

        if state.get("collected_rx_volume") is None:
            volume = _extract_rx_volume(text)
            if volume is None:
                reply = (
                    f"No problem, {name}. Just a rough monthly prescription count "
                    f"is fine — even a ballpark number helps."
                )
                return {**updates, "messages": [AIMessage(reply)]}
            updates["collected_rx_volume"] = volume
            high = volume >= threshold
            reply = f"Got it — about {volume} a month. " + _pitch(name, high, threshold)
            return {**updates, "messages": [AIMessage(reply)]}

    # 3) Pitch / general help.
    if _contains_any(lowered, _PITCH_INTENT) or _looks_like_greeting_only(text):
        return {**updates, "messages": [AIMessage(_pitch(name, bool(is_high_volume), threshold))]}

    # 4) Anything else: safe, non-hallucinated fallback that stays in scope.
    logger.info("Fallback: out-of-scope or unrecognised input, returning safe reply")
    reply = (
        "I want to make sure I only share accurate information, so I can't speak "
        "to that. What I can help with is how TJM Labs supports pharmacies — "
        "would you like an overview by email, or a callback from our team?"
    )
    return {**updates, "messages": [AIMessage(reply)]}


def fallback_greeting(pharmacy: Optional[Dict[str, Any]], threshold: int) -> str:
    """Template opening line used when no LLM is available."""
    if pharmacy:
        name = pharmacy.get("name", "there")
        location = pharmacy.get("location")
        volume = pharmacy.get("total_rx_volume")
        parts = [f"Hi {name}, thanks for calling TJM Labs!"]
        if location:
            parts.append(f"Great to hear from you over in {location}.")
        if volume is not None:
            parts.append(f"I see you're filling around {volume} prescriptions a month.")
        parts.append("How can I help you today?")
        return " ".join(parts)
    return (
        "Thanks for calling TJM Labs! I don't see your number on file yet — "
        "may I have your pharmacy's name to get started?"
    )
