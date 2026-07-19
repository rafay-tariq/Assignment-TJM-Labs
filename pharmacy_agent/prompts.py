"""Prompt construction for the sales agent.

The system prompt is deliberately split so it can be cached efficiently:

  1. A STATIC section (role, product facts, rules, tool guidance) that is
     identical on every call. It is placed FIRST and carries an Anthropic
     ``cache_control`` breakpoint, so Claude caches this prefix and reuses it
     across calls — cutting latency and token cost.
  2. A DYNAMIC ``<caller_context>`` section (who is calling, their pharmacy
     data) that changes per call. It is placed LAST, after the cache breakpoint,
     so only this small piece is re-processed each time.

Sections are wrapped in XML tags because Claude follows XML-delimited structure
more reliably than free-form text, and it lets us point the model at the
verified ``<caller_context>`` as its single source of truth (anti-hallucination).
"""

from typing import Any, Dict, List, Optional

# Static description of what TJM Labs offers. In a real system this would come
# from a maintained knowledge base / CMS rather than a constant.
TJM_LABS_VALUE_PROP = """\
TJM Labs partners with pharmacies to help them serve high prescription volumes
profitably and reliably. For high-Rx-volume pharmacies we offer:
  - Streamlined bulk prescription fulfillment and faster turnaround
  - Volume-based pricing that improves margins as Rx counts grow
  - Analytics on prescribing and fill trends to spot growth opportunities
  - A dedicated account manager for high-volume partners
"""

# --------------------------------------------------------------------------- #
# STATIC prefix (cacheable). No per-call data may appear here.                 #
# --------------------------------------------------------------------------- #
STATIC_SYSTEM_PROMPT = f"""\
<role>
You are an inbound sales agent for TJM Labs. Pharmacies call the number listed
on the TJM Labs website. Your goals, in order:
  1. Make the caller feel recognised and understood.
  2. Understand why they are calling.
  3. Explain how TJM Labs supports pharmacies, especially high-Rx-volume ones.
  4. Offer a concrete next step (email follow-up or a scheduled callback) and
     use the provided tools to arrange it when the caller agrees.
</role>

<about_tjm_labs>
{TJM_LABS_VALUE_PROP}</about_tjm_labs>

<rules>
- Be warm, concise, and professional. This is a phone conversation: keep replies
  to a few sentences.
- NEVER invent facts about the caller, their pharmacy, pricing, or TJM Labs
  services. Only state what is given to you or what the caller tells you.
- If asked something outside pharmacy sales / TJM Labs (e.g. medical advice,
  unrelated topics), politely decline and steer back. Do not guess.
- If you are unsure or lack the information, say so and offer to have someone
  follow up rather than making something up.
</rules>

<tools>
You can arrange follow-ups with two tools: `send_email_followup` and
`schedule_callback`. Only call a tool after the caller has agreed to it.
</tools>

<caller_context_instructions>
The <caller_context> section that follows is the ONLY verified information you
have about this caller. Treat it as ground truth; never contradict it or state
anything beyond it.
</caller_context_instructions>"""


def _recognised_block(pharmacy: Dict[str, Any], threshold: int) -> str:
    """Body of the caller context for a recognised caller."""
    lines = [
        "CALLER STATUS: RECOGNISED pharmacy (identified by phone number).",
        f"  - Name: {pharmacy.get('name')}",
    ]
    if pharmacy.get("location"):
        lines.append(f"  - Location: {pharmacy['location']}")
    if pharmacy.get("email"):
        lines.append(f"  - Email on file: {pharmacy['email']}")
    lines.append(f"  - Monthly Rx volume on file: {pharmacy.get('total_rx_volume')}")
    top = pharmacy.get("top_drugs") or []
    if top:
        drugs = ", ".join(f"{d['drug']} ({d['count']})" for d in top)
        lines.append(f"  - Top prescriptions: {drugs}")
    if pharmacy.get("is_high_volume"):
        lines.append(
            f"  - This IS a high-volume pharmacy (>= {threshold}/mo). Emphasise "
            "the high-volume benefits above."
        )
    else:
        lines.append(
            f"  - This is below the high-volume threshold ({threshold}/mo). Be "
            "helpful; mention how TJM Labs helps pharmacies scale as they grow."
        )
    lines.append(
        "Greet them BY NAME and reference what you know (location and Rx volume) "
        "naturally. Do not read the data back like a database record."
    )
    return "\n".join(lines)


_UNRECOGNISED_BLOCK = """\
CALLER STATUS: NOT RECOGNISED (their phone number is not on file).
  - Do NOT guess who they are.
  - Early in the conversation, collect: (1) the pharmacy's name and (2) their
    approximate monthly prescription (Rx) volume. Ask conversationally, one
    thing at a time.
  - Once you know the pharmacy name, use it naturally through the conversation.
  - When they share their Rx volume, tailor your pitch: at or above
    {threshold}/month, emphasise TJM Labs' high-volume benefits."""


def build_caller_context(pharmacy: Optional[Dict[str, Any]], threshold: int) -> str:
    """Build the DYNAMIC, per-call caller-context block (XML-wrapped)."""
    if pharmacy:
        inner = _recognised_block(pharmacy, threshold)
    else:
        inner = _UNRECOGNISED_BLOCK.format(threshold=threshold)
    return f"<caller_context>\n{inner}\n</caller_context>"


def build_system_prompt(pharmacy: Optional[Dict[str, Any]], threshold: int) -> str:
    """Full system prompt as a single string (static prefix + dynamic context).

    Used where a plain string is convenient (e.g. logging/debugging). The LLM
    path uses :func:`build_system_content` instead to get prompt caching.
    """
    return f"{STATIC_SYSTEM_PROMPT}\n\n{build_caller_context(pharmacy, threshold)}"


def build_system_content(
    pharmacy: Optional[Dict[str, Any]],
    threshold: int,
) -> List[Dict[str, Any]]:
    """System prompt as Anthropic content blocks for prompt caching.

    The static block comes first and is marked with ``cache_control`` so Claude
    caches everything up to that breakpoint and reuses it across calls; the
    dynamic caller context is a separate block appended after it.
    """
    return [
        {
            "type": "text",
            "text": STATIC_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": build_caller_context(pharmacy, threshold),
        },
    ]


# Instruction used to trigger the opening line once the call connects.
GREETING_INSTRUCTION = (
    "[CALL CONNECTED] Deliver a brief, warm opening greeting now, following all "
    "the rules above. If the caller is recognised, greet them by name."
)
