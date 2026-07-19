"""Tests for phone normalisation and caller identification."""

from pharmacy_agent import pharmacy_api
from pharmacy_agent.models import Pharmacy

DIRECTORY = [
    Pharmacy.from_api(
        {"id": 1, "name": "HealthFirst Pharmacy", "phone": "+1-555-123-4567"}
    ),
    Pharmacy.from_api({"id": 2, "name": "QuickMeds Rx", "phone": "+1-555-987-6543"}),
]


def test_normalize_phone_strips_formatting_and_country_code():
    assert pharmacy_api._normalize_phone("+1-555-123-4567") == "5551234567"
    assert pharmacy_api._normalize_phone("(555) 123 4567") == "5551234567"
    assert pharmacy_api._normalize_phone("15551234567") == "5551234567"
    assert pharmacy_api._normalize_phone(None) == ""


def test_identify_matches_across_formats(monkeypatch):
    monkeypatch.setattr(pharmacy_api, "fetch_pharmacies", lambda: DIRECTORY)
    match = pharmacy_api.identify_pharmacy_by_phone("(555) 123-4567")
    assert match is not None
    assert match.name == "HealthFirst Pharmacy"


def test_identify_returns_none_for_unknown_number(monkeypatch):
    monkeypatch.setattr(pharmacy_api, "fetch_pharmacies", lambda: DIRECTORY)
    assert pharmacy_api.identify_pharmacy_by_phone("+1-555-000-0000") is None


def test_identify_degrades_gracefully_on_api_error(monkeypatch):
    def boom():
        raise pharmacy_api.PharmacyAPIError("down")

    monkeypatch.setattr(pharmacy_api, "fetch_pharmacies", boom)
    # An outage should look like an unrecognised caller, not crash the call.
    assert pharmacy_api.identify_pharmacy_by_phone("+1-555-123-4567") is None
