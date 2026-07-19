"""Domain models for pharmacies and their prescription data.

These wrap the raw JSON returned by the directory API in typed objects and
centralise business logic such as deriving total Rx volume.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Prescription:
    """A single drug line item with its monthly fill count."""

    drug: str
    count: int

    @classmethod
    def from_api(cls, raw: Dict[str, Any]) -> "Prescription":
        return cls(drug=str(raw.get("drug", "")), count=int(raw.get("count", 0) or 0))


@dataclass(frozen=True)
class Pharmacy:
    """A pharmacy record from the directory API.

    ``prescriptions`` is a list of per-drug fill counts; the API does not
    expose a single "Rx volume" number, so we derive it (see
    :attr:`total_rx_volume`).
    """

    id: str
    name: str
    phone: str
    email: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    prescriptions: List[Prescription] = field(default_factory=list)

    @classmethod
    def from_api(cls, raw: Dict[str, Any]) -> "Pharmacy":
        """Build a :class:`Pharmacy` from a raw API record, tolerant of nulls."""
        prescriptions = [
            Prescription.from_api(p) for p in (raw.get("prescriptions") or [])
        ]
        return cls(
            id=str(raw.get("id", "")),
            name=str(raw.get("name", "")).strip(),
            phone=str(raw.get("phone", "")).strip(),
            email=(raw.get("email") or None),
            city=(raw.get("city") or None),
            state=(raw.get("state") or None),
            prescriptions=prescriptions,
        )

    @property
    def total_rx_volume(self) -> int:
        """Total monthly prescription volume across all drugs."""
        return sum(p.count for p in self.prescriptions)

    @property
    def location(self) -> Optional[str]:
        """Human-readable "City, ST" location, or None if unknown."""
        parts = [p for p in (self.city, self.state) if p]
        return ", ".join(parts) if parts else None

    def is_high_volume(self, threshold: int) -> bool:
        """Whether this pharmacy meets the high-Rx-volume threshold."""
        return self.total_rx_volume >= threshold

    def top_drugs(self, limit: int = 3) -> List[Prescription]:
        """Highest-volume prescriptions, most first."""
        return sorted(self.prescriptions, key=lambda p: p.count, reverse=True)[:limit]

    def to_context(self, threshold: int) -> Dict[str, Any]:
        """Serialisable summary used to ground prompts and greetings.

        Kept JSON-friendly so it can live inside LangGraph state and survive
        checkpointing.
        """
        return {
            "id": self.id,
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
            "location": self.location,
            "total_rx_volume": self.total_rx_volume,
            "is_high_volume": self.is_high_volume(threshold),
            "top_drugs": [
                {"drug": p.drug, "count": p.count} for p in self.top_drugs()
            ],
        }
