"""Client for the pharmacy directory API.

Responsible for fetching pharmacy records and identifying a caller by phone
number. Phone matching is done on normalised digits so that formatting
differences (spaces, dashes, leading ``+1``) do not cause misses.
"""

import re
from typing import List, Optional

import requests

from .config import settings
from .logging_config import get_logger
from .models import Pharmacy

logger = get_logger("pharmacy_agent.pharmacy_api")


class PharmacyAPIError(RuntimeError):
    """Raised when the directory API cannot be reached or returns bad data."""


def _normalize_phone(phone: Optional[str]) -> str:
    """Reduce a phone number to comparable digits.

    Strips all non-digit characters and a leading US country code so that
    ``+1-555-123-4567`` and ``(555) 123 4567`` compare equal.
    """
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def fetch_pharmacies() -> List[Pharmacy]:
    """Fetch and parse all pharmacies from the directory API.

    Raises:
        PharmacyAPIError: on network failure or an unexpected payload shape.
    """
    url = settings.pharmacy_api_url
    logger.info("Fetching pharmacy directory from %s", url)
    try:
        response = requests.get(url, timeout=settings.api_timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.error("Failed to reach pharmacy API: %s", exc)
        raise PharmacyAPIError(f"Could not reach pharmacy API: {exc}") from exc
    except ValueError as exc:  # JSON decode error
        logger.error("Pharmacy API returned invalid JSON: %s", exc)
        raise PharmacyAPIError("Pharmacy API returned invalid JSON") from exc

    if not isinstance(payload, list):
        raise PharmacyAPIError("Expected a list of pharmacies from the API")

    pharmacies = [Pharmacy.from_api(item) for item in payload]
    logger.info("Loaded %d pharmacies from directory", len(pharmacies))
    return pharmacies


def identify_pharmacy_by_phone(phone: str) -> Optional[Pharmacy]:
    """Return the pharmacy whose phone matches ``phone``, or None.

    Network/parse errors are swallowed and logged so that a directory outage
    degrades gracefully into the "unrecognised caller" flow rather than
    crashing the call.
    """
    target = _normalize_phone(phone)
    if not target:
        logger.warning("No caller phone provided; treating as unrecognised")
        return None

    try:
        pharmacies = fetch_pharmacies()
    except PharmacyAPIError:
        logger.warning("Directory unavailable; treating caller as unrecognised")
        return None

    for pharmacy in pharmacies:
        if _normalize_phone(pharmacy.phone) == target:
            logger.info(
                "Identified caller %s as '%s' (id=%s)",
                phone,
                pharmacy.name,
                pharmacy.id,
            )
            return pharmacy

    logger.info("Caller %s did not match any pharmacy on file", phone)
    return None
