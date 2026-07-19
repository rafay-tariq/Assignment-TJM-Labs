"""Logging setup for the agent.

A single :func:`configure_logging` call wires up a consistent, timestamped
format. Modules obtain their own logger via :func:`get_logger`.
"""

import logging
from typing import Optional

from .config import settings

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once for the process.

    Safe to call multiple times; only the first call takes effect.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        level=getattr(logging, (level or settings.log_level).upper(), logging.INFO),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )
    # The Anthropic/httpx clients are chatty at INFO; keep them at WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring logging is configured."""
    configure_logging()
    return logging.getLogger(name)
