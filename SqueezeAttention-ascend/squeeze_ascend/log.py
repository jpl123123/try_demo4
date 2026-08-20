"""Logging for squeeze-ascend: independent logger with a distinct prefix."""

import logging
import sys

from squeeze_ascend.envs import get_config

_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}


def get_logger(name: str = "squeeze-ascend") -> logging.Logger:
    cfg = get_config()
    logger = logging.getLogger(name)
    level = _LEVELS.get(cfg.log_level, logging.INFO)
    if logger.level != level or not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
            logger.addHandler(handler)
    return logger
