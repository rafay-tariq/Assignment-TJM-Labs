"""Tests for domain models and derived Rx-volume logic."""

from pharmacy_agent.models import Pharmacy

RAW = {
    "id": 1,
    "name": "HealthFirst Pharmacy",
    "phone": "+1-555-123-4567",
    "email": "contact@healthfirst.com",
    "city": "New York",
    "state": "NY",
    "prescriptions": [
        {"drug": "Lisinopril", "count": 42},
        {"drug": "Atorvastatin", "count": 38},
        {"drug": "Metformin", "count": 20},
    ],
}


def test_from_api_parses_fields():
    pharmacy = Pharmacy.from_api(RAW)
    assert pharmacy.name == "HealthFirst Pharmacy"
    assert pharmacy.location == "New York, NY"
    assert len(pharmacy.prescriptions) == 3


def test_total_rx_volume_is_sum_of_counts():
    pharmacy = Pharmacy.from_api(RAW)
    assert pharmacy.total_rx_volume == 100


def test_is_high_volume_threshold_boundary():
    pharmacy = Pharmacy.from_api(RAW)  # volume == 100
    assert pharmacy.is_high_volume(100) is True
    assert pharmacy.is_high_volume(101) is False


def test_handles_null_and_missing_fields():
    pharmacy = Pharmacy.from_api({"id": 2, "name": "QuickMeds", "phone": "x", "email": None})
    assert pharmacy.email is None
    assert pharmacy.location is None
    assert pharmacy.total_rx_volume == 0


def test_to_context_is_json_friendly():
    context = Pharmacy.from_api(RAW).to_context(100)
    assert context["name"] == "HealthFirst Pharmacy"
    assert context["total_rx_volume"] == 100
    assert context["is_high_volume"] is True
    assert context["top_drugs"][0] == {"drug": "Lisinopril", "count": 42}
