"""Application logging configuration with bounded on-disk retention."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "stimtrace"


def configure_app_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    resolved = Path(log_path).resolve()
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename).resolve() == resolved
        for handler in logger.handlers
    ):
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            resolved,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


def get_logger(component: str) -> logging.Logger:
    logger = logging.getLogger(f"{LOGGER_NAME}.{component}")
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger
