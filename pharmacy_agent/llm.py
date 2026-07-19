"""LLM access layer.

Wraps the Anthropic chat model behind a small, lazily-initialised accessor so
the rest of the code can ask "is an LLM available?" and get a configured client
without worrying about credentials or imports. When no API key is present the
agent falls back to a deterministic implementation (see ``graph.py``).
"""

from functools import lru_cache
from typing import Optional

from .config import settings
from .logging_config import get_logger

logger = get_logger("pharmacy_agent.llm")


def llm_available() -> bool:
    """Whether a real LLM can be used (API key present and package importable)."""
    if not settings.llm_available:
        return False
    try:
        import langchain_anthropic  # noqa: F401
    except ImportError:
        logger.warning(
            "ANTHROPIC_API_KEY is set but langchain-anthropic is not installed; "
            "using deterministic fallback."
        )
        return False
    return True


@lru_cache(maxsize=1)
def get_llm():
    """Return a configured ChatAnthropic client, or None if unavailable.

    The result is cached so we build the client only once per process.
    """
    if not llm_available():
        return None

    from langchain_anthropic import ChatAnthropic

    logger.info("Initialising Claude model '%s'", settings.model_name)
    return ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
