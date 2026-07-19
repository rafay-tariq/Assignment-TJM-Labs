"""Tests for the cache-friendly, XML-structured system prompt."""

from pharmacy_agent.prompts import (
    STATIC_SYSTEM_PROMPT,
    build_system_content,
    build_system_prompt,
)

PHARMACY = {
    "name": "HealthFirst Pharmacy",
    "location": "New York, NY",
    "email": "contact@healthfirst.com",
    "total_rx_volume": 120,
    "is_high_volume": True,
    "top_drugs": [{"drug": "Lisinopril", "count": 120}],
}


def test_static_prefix_has_no_caller_data():
    # The cached prefix must be identical on every call, so it cannot contain
    # anything caller-specific.
    assert "HealthFirst" not in STATIC_SYSTEM_PROMPT
    assert "<role>" in STATIC_SYSTEM_PROMPT
    assert "<rules>" in STATIC_SYSTEM_PROMPT


def test_content_blocks_are_static_then_dynamic():
    blocks = build_system_content(PHARMACY, 100)
    assert len(blocks) == 2
    # Block 0: static, carries the cache breakpoint.
    assert blocks[0]["text"] == STATIC_SYSTEM_PROMPT
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    # Block 1: dynamic caller context, no cache breakpoint.
    assert "cache_control" not in blocks[1]
    assert "<caller_context>" in blocks[1]["text"]
    assert "HealthFirst Pharmacy" in blocks[1]["text"]


def test_system_prompt_string_orders_static_before_dynamic():
    prompt = build_system_prompt(PHARMACY, 100)
    assert prompt.index("<role>") < prompt.index("<caller_context>")


def test_unrecognised_context_asks_for_details():
    blocks = build_system_content(None, 100)
    dynamic = blocks[1]["text"]
    assert "NOT RECOGNISED" in dynamic
    assert "100/month" in dynamic
