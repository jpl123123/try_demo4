"""Centralized environment-variable configuration for squeeze-ascend.

Gate env vars are matched case-insensitively: ``export squeeze=1``,
``export SQUEEZE_ASCEND=1`` and ``export squeeze_ascend=1`` all enable the
patch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_FALSE_VALUES = {"", "0", "false", "off", "no", "n", "none"}


def _getenv(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default


def _bool(*names: str, default: bool) -> bool:
    value = _getenv(*names)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_VALUES


def _int(*names: str, default: int) -> int:
    value = _getenv(*names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float(*names: str, default: float) -> float:
    value = _getenv(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _str(*names: str, default: str) -> str:
    value = _getenv(*names)
    return value if value is not None else default


def _layer_range(value: Optional[str]) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("", "all", "none"):
        return None
    try:
        if "-" in value:
            lo, hi = value.split("-", 1)
            return (int(lo), int(hi))
        return (int(value), int(value))
    except ValueError:
        return None


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    # ---- 2D budgets -------------------------------------------------------
    ini_size: float = 0.21        # base per-layer KV budget (fraction of prompt)
    start_size: int = 4           # sink tokens always kept (StreamingLLM)
    kv_class3: float = 0.0        # 0.0 => uniform mode; else class-3 budget
    clusters: int = 3             # KMeans clusters for layer importance
    # ---- trigger ----------------------------------------------------------
    mid_prefill: bool = True
    mid_prefill_budget: int = 65536
    layers: Optional[tuple[int, int]] = None
    # ---- observability -----------------------------------------------------
    log_level: str = "info"
    step_log: bool = True
    dry_run: bool = False
    # ---- strategy ------------------------------------------------------------
    policy: str = "auto"  # auto | primary | defer (vs kvpress-ascend)
    block_size_override: Optional[int] = None


def load_config() -> Config:
    ini = min(0.99, max(0.01, _float("SQUEEZE_ASCEND_INI_SIZE", "squeeze_ini_size", default=0.21)))
    class3 = _float("SQUEEZE_ASCEND_KV_CLASS3", "squeeze_kv_class3", default=0.0)
    if class3 <= 0.0:
        class3 = ini  # uniform mode
    class3 = min(0.99, max(0.01, class3))
    return Config(
        enabled=True,
        ini_size=ini,
        start_size=max(0, _int("SQUEEZE_ASCEND_START_SIZE", "squeeze_start_size", default=4)),
        kv_class3=class3,
        clusters=max(2, min(8, _int("SQUEEZE_ASCEND_CLUSTERS", default=3))),
        mid_prefill=_bool("SQUEEZE_ASCEND_MID_PREFILL", "squeeze_mid_prefill", default=True),
        mid_prefill_budget=max(
            16, _int("SQUEEZE_ASCEND_MID_PREFILL_BUDGET", "squeeze_mid_budget", default=65536)
        ),
        layers=_layer_range(_getenv("SQUEEZE_ASCEND_LAYERS", "squeeze_layers")),
        log_level=_str("SQUEEZE_ASCEND_LOG", "squeeze_log", default="info").lower(),
        step_log=_bool("SQUEEZE_ASCEND_STEP_LOG", "squeeze_step_log", default=True),
        dry_run=_bool("SQUEEZE_ASCEND_DRY_RUN", "squeeze_dry_run", default=False),
        policy=_str("SQUEEZE_ASCEND_POLICY", "squeeze_policy", default="auto").lower(),
        block_size_override=_int("SQUEEZE_ASCEND_BLOCK_SIZE", default=0) or None,
    )


_CFG: Optional[Config] = None


def get_config() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def reset_config() -> None:
    global _CFG
    _CFG = None
