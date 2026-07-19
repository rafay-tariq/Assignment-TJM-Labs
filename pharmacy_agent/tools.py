"""Mocked follow-up tools exposed to the agent.

These simulate side effects (sending an email, booking a callback). They do
not actually integrate with any provider; they log the action and return a
human-readable confirmation. Each tool is a LangChain ``@tool`` so the LLM can
call it directly, and the underlying implementation is also importable for the
deterministic fallback path and for tests.
"""

from langchain_core.tools import tool

from .logging_config import get_logger

logger = get_logger("pharmacy_agent.tools")


def _send_email_followup(to_email: str, pharmacy_name: str, summary: str) -> str:
    """Implementation for the email follow-up mock (see :func:`send_email_followup`)."""
    logger.info(
        "[MOCK EMAIL] to=%s | pharmacy=%s | summary=%s",
        to_email or "<unknown>",
        pharmacy_name or "<unknown>",
        summary,
    )
    print(
        f"\n[MOCK EMAIL SENT]\n"
        f"    To:       {to_email or 'unknown@pharmacy'}\n"
        f"    Pharmacy: {pharmacy_name or 'Unknown'}\n"
        f"    Summary:  {summary}\n"
    )
    return (
        f"Email follow-up queued to {to_email or 'the pharmacy'} "
        f"for {pharmacy_name or 'the pharmacy'}."
    )


def _schedule_callback(pharmacy_name: str, phone: str, preferred_time: str) -> str:
    """Implementation for the callback mock (see :func:`schedule_callback`)."""
    logger.info(
        "[MOCK CALLBACK] pharmacy=%s | phone=%s | preferred_time=%s",
        pharmacy_name or "<unknown>",
        phone or "<unknown>",
        preferred_time or "<unspecified>",
    )
    print(
        f"\n[MOCK CALLBACK SCHEDULED]\n"
        f"    Pharmacy: {pharmacy_name or 'Unknown'}\n"
        f"    Phone:    {phone or 'unknown'}\n"
        f"    When:     {preferred_time or 'to be confirmed'}\n"
    )
    return (
        f"Callback scheduled for {pharmacy_name or 'the pharmacy'} "
        f"at {preferred_time or 'a time our team will confirm'}."
    )


@tool
def send_email_followup(to_email: str, pharmacy_name: str, summary: str) -> str:
    """Send a follow-up email to a pharmacy.

    Args:
        to_email: Destination email address for the pharmacy.
        pharmacy_name: Name of the pharmacy being followed up with.
        summary: Short summary of what to include in the follow-up.
    """
    return _send_email_followup(to_email, pharmacy_name, summary)


@tool
def schedule_callback(pharmacy_name: str, phone: str, preferred_time: str) -> str:
    """Schedule a callback from a TJM Labs sales representative.

    Args:
        pharmacy_name: Name of the pharmacy to call back.
        phone: Phone number to reach the pharmacy on.
        preferred_time: The caller's preferred day/time for the callback.
    """
    return _schedule_callback(pharmacy_name, phone, preferred_time)


# Convenience list for binding to the LLM.
FOLLOW_UP_TOOLS = [send_email_followup, schedule_callback]
