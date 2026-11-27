"""
utils/logging.py
────────────────
Structured logger with Rich support for output formatting and phase timing.
"""

import logging
import time
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme

    _theme = Theme({
        "agent": "bold cyan",
        "timing": "bold yellow",
        "success": "bold green",
        "error": "bold red",
    })
    console = Console(theme=_theme)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Creates logger with Rich handler (or standard StreamHandler fallback)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        if HAS_RICH and console is not None:
            handler = RichHandler(
                console=console,
                show_path=False,
                show_time=True,
                rich_tracebacks=True,
                markup=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextmanager
def LogTimer(label: str, logger: logging.Logger | None = None):
    """Context manager for timing execution phases."""
    _log = logger or get_logger("timer")
    _log.info(f"> {label}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _log.info(f"* {label} completed in {elapsed:.2f}s")
