"""Tests for the mocked follow-up tools."""

from pharmacy_agent.tools import (
    _schedule_callback,
    _send_email_followup,
    schedule_callback,
    send_email_followup,
)


def test_send_email_followup_returns_confirmation(capsys):
    result = _send_email_followup("a@b.com", "HealthFirst", "summary text")
    assert "HealthFirst" in result
    assert "a@b.com" in result
    # The mock also prints a visible confirmation block.
    assert "MOCK EMAIL SENT" in capsys.readouterr().out


def test_schedule_callback_returns_confirmation(capsys):
    result = _schedule_callback("HealthFirst", "+1-555-123-4567", "tomorrow 2pm")
    assert "HealthFirst" in result
    assert "tomorrow 2pm" in result
    assert "MOCK CALLBACK SCHEDULED" in capsys.readouterr().out


def test_tools_are_exposed_as_langchain_tools():
    # Names are what the LLM sees when the tools are bound.
    assert send_email_followup.name == "send_email_followup"
    assert schedule_callback.name == "schedule_callback"
