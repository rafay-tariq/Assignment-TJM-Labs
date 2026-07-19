"""End-to-end tests of the conversation graph on the deterministic path.

These force the no-LLM fallback so the graph runs without network or API keys,
and verify the two headline behaviours: recognised greeting-by-name and
unrecognised info collection.
"""

import pytest
from langchain_core.messages import HumanMessage

from pharmacy_agent import graph as graph_module
from pharmacy_agent.models import Pharmacy

RECOGNISED = Pharmacy.from_api(
    {
        "id": 1,
        "name": "HealthFirst Pharmacy",
        "phone": "+1-555-123-4567",
        "city": "New York",
        "state": "NY",
        "prescriptions": [{"drug": "Lisinopril", "count": 120}],
    }
)


@pytest.fixture
def force_fallback(monkeypatch):
    """Run the graph without an LLM."""
    monkeypatch.setattr(graph_module, "llm_available", lambda: False)


def _last_text(state):
    return str(state["messages"][-1].content)


def test_recognised_caller_is_greeted_by_name(force_fallback, monkeypatch):
    monkeypatch.setattr(graph_module, "identify_pharmacy_by_phone", lambda phone: RECOGNISED)
    graph = graph_module.build_graph()
    config = {"configurable": {"thread_id": "t-recognised"}}

    result = graph.invoke(
        {"caller_phone": "+1-555-123-4567", "stage": "start", "messages": []}, config
    )
    greeting = _last_text(result)
    assert "HealthFirst Pharmacy" in greeting
    assert "New York, NY" in greeting


def test_unrecognised_caller_collects_name(force_fallback, monkeypatch):
    monkeypatch.setattr(graph_module, "identify_pharmacy_by_phone", lambda phone: None)
    graph = graph_module.build_graph()
    config = {"configurable": {"thread_id": "t-unrecognised"}}

    start = graph.invoke(
        {"caller_phone": "+1-555-000-0000", "stage": "start", "messages": []}, config
    )
    assert "name" in _last_text(start).lower()

    # Caller provides the pharmacy name; agent should acknowledge and ask for Rx volume.
    turn = graph.invoke({"messages": [HumanMessage(content="Downtown Drugs")]}, config)
    reply = _last_text(turn)
    assert "Downtown Drugs" in reply
    assert "prescription" in reply.lower()


def test_callback_intent_triggers_mock_tool(force_fallback, monkeypatch, capsys):
    monkeypatch.setattr(graph_module, "identify_pharmacy_by_phone", lambda phone: RECOGNISED)
    graph = graph_module.build_graph()
    config = {"configurable": {"thread_id": "t-callback"}}

    graph.invoke(
        {"caller_phone": "+1-555-123-4567", "stage": "start", "messages": []}, config
    )
    graph.invoke({"messages": [HumanMessage(content="Please call me back")]}, config)
    assert "MOCK CALLBACK SCHEDULED" in capsys.readouterr().out
