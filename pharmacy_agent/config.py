"""Central configuration for the pharmacy sales agent.

All tunables are read from environment variables (optionally loaded from a
local ``.env`` file) so the behaviour can be changed without touching code.
"""

import os
from dataclasses import dataclass

try:  # Loading a .env file is a convenience, not a hard dependency.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional at runtime.
    pass


def _get_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` if unset/invalid."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "claude-sonnet-5")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    llm_max_tokens: int = _get_int("LLM_MAX_TOKENS", 1024)

    # --- Pharmacy directory API -------------------------------------------
    pharmacy_api_url: str = os.getenv(
        "PHARMACY_API_URL",
        "https://67e14fb758cc6bf785254550.mockapi.io/pharmacies",
    )
    api_timeout_seconds: int = _get_int("API_TIMEOUT_SECONDS", 10)

    # --- Business rules ----------------------------------------------------
    # A pharmacy at or above this monthly Rx count is treated as "high volume",
    # which is the segment TJM Labs specialises in supporting.
    high_rx_volume_threshold: int = _get_int("HIGH_RX_VOLUME_THRESHOLD", 100)

    # --- Simulation --------------------------------------------------------
    # Stand-in for the caller ID that a real telephony provider would supply.
    mock_caller_phone: str = os.getenv("MOCK_CALLER_PHONE", "+1-555-123-4567")

    # --- Logging -----------------------------------------------------------
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def llm_available(self) -> bool:
        """True when an Anthropic API key is configured."""
        return bool(self.anthropic_api_key.strip())


# A single shared settings instance for the whole process.
settings = Settings()
