"""
Structured logging setup for the project.

Call ``setup_logging()`` once at script entry to configure console + file handlers.
Then use ``logging.getLogger(__name__)`` in each module.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_CONFIGURED = False

CONSOLE_FORMAT = "%(levelname)-8s %(name)s — %(message)s"
FILE_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
) -> None:
    """Configure root logger with console and (optionally) file handlers.

    Safe to call multiple times; only the first call takes effect.

    Args:
        level: logging level name (DEBUG, INFO, WARNING, …).
        log_dir: directory for log files. If ``None``, defaults to
                 ``experiments/logs``. Set to an empty ``Path("")`` to
                 disable file logging.
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)

    # File handler
    if log_dir is None:
        log_dir = Path("experiments/logs")

    if str(log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"{timestamp}.log")
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for name in ("ultralytics", "matplotlib", "PIL", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
