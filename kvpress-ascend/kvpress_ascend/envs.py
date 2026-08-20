"""Centralized environment-variable configuration for kvpress-ascend.

Every knob of the adapter lives here.  Gate env vars are matched
case-insensitively, so ``export kvpress=1``, ``export KVPRESS_ASCEND=1`` and
``export kvpress_ascend=1`` are all valid ways to enable the patch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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


def _json_list(*names: str, default: Optional[list]) -> Optional[list]:
    value = _getenv(*names)
    if value is None:
        return default
    try:
        parsed = json.loads(value)
    except ValueError:
        return default
    if not isinstance(parsed, list):
        return default
    return parsed


def _layer_range(value: Optional[str]) -> Optional[tuple[int, int]]:
    """Parse '0-31' / 'all' / '' into an inclusive (lo, hi) or None."""
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
    """All runtime knobs, resolved once at install time."""

    # ---- gate ----------------------------------------------------------
    enabled: bool = False
    # ---- press ---------------------------------------------------------
    press: str = "snapkv"                    # snapkv | streaming | random | per_layer
    compression_ratio: float = 0.5           # fraction of KV tokens to remove (0..1)
    window_size: int = 64                    # SnapKV observation window (tokens)
    n_sink: int = 4                          # StreamingLLM sink tokens
    block_score_mode: str = "mean"           # mean | max (token -> block aggregation)
    force_keep_window: bool = True           # SnapKV: never drop the observation window
    force_keep_sink: bool = True             # StreamingLLM: never drop sink tokens
    per_layer_ratios: Optional[list] = None  # per-layer compression ratios (per_layer)
    layers: Optional[tuple[int, int]] = None  # inclusive layer range, None = all
    # ---- trigger -------------------------------------------------------
    mid_prefill: bool = True                 # compress mid-prefill at budget anchors
    mid_prefill_budget: int = 65536          # tokens between mid-prefill anchors
    mid_prefill_refresh: bool = False        # re-anchor on a fixed grid vs on demand
    decode_reanchor: bool = True             # re-compress during decode when tail grows
    decode_reanchor_window: int = 8192       # max new tokens between decode re-anchors
    # ---- observability ---------------------------------------------------
    log_level: str = "info"                  # debug | info | warning
    step_log: bool = True                    # per-step heartbeat line
    dry_run: bool = False                    # score + log only, no view application
    # ---- strategy --------------------------------------------------------
    policy: str = "auto"                     # auto | primary | defer (vs squeeze-ascend)
    # ---- internals (mostly for tests) ------------------------------------
    block_size_override: Optional[int] = None  # tests only


def load_config() -> Config:
    return Config(
        enabled=True,
        press=_str("KVPRESS_ASCEND_PRESS", "kvpress_press", default="snapkv").lower(),
        compression_ratio=min(
            0.99,
            max(0.0, _float("KVPRESS_ASCEND_RATIO", "kvpress_ratio", default=0.5)),
        ),
        window_size=max(1, _int("KVPRESS_ASCEND_WINDOW", "kvpress_window", default=64)),
        n_sink=max(0, _int("KVPRESS_ASCEND_N_SINK", "kvpress_n_sink", default=4)),
        block_score_mode=_str("KVPRESS_ASCEND_BLOCK_MODE", default="mean").lower(),
        force_keep_window=_bool("KVPRESS_ASCEND_FORCE_WINDOW", default=True),
        force_keep_sink=_bool("KVPRESS_ASCEND_FORCE_SINK", default=True),
        per_layer_ratios=_json_list(
            "KVPRESS_ASCEND_PER_LAYER_RATIOS", "kvpress_per_layer_ratios", default=None
        ),
        layers=_layer_range(_getenv("KVPRESS_ASCEND_LAYERS", "kvpress_layers")),
        mid_prefill=_bool("KVPRESS_ASCEND_MID_PREFILL", "kvpress_mid_prefill", default=True),
        mid_prefill_budget=max(
            16, _int("KVPRESS_ASCEND_MID_PREFILL_BUDGET", "kvpress_mid_budget", default=65536)
        ),
        mid_prefill_refresh=_bool(
            "KVPRESS_ASCEND_MID_PREFILL_REFRESH", "kvpress_mid_refresh", default=False
        ),
        decode_reanchor=_bool("KVPRESS_ASCEND_DECODE_REANCHOR", default=True),
        decode_reanchor_window=max(
            16, _int("KVPRESS_ASCEND_DECODE_REANCHOR_WINDOW", "kvpress_reanchor_window", default=8192)
        ),
        log_level=_str("KVPRESS_ASCEND_LOG", "kvpress_log", default="info").lower(),
        step_log=_bool("KVPRESS_ASCEND_STEP_LOG", "kvpress_step_log", default=True),
        dry_run=_bool("KVPRESS_ASCEND_DRY_RUN", "kvpress_dry_run", default=False),
        policy=_str("KVPRESS_ASCEND_POLICY", "kvpress_policy", default="auto").lower(),
        block_size_override=_int("KVPRESS_ASCEND_BLOCK_SIZE", default=0) or None,
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


@dataclass
class _EnvDoc:
    name: str
    default: str
    doc: str


ENV_DOC = [
    _EnvDoc("kvpress / KVPRESS_ASCEND", "unset", "Master gate. Any truthy value enables the patch."),
    _EnvDoc("KVPRESS_ASCEND_PRESS", "snapkv", "Press: snapkv | streaming | random | per_layer"),
    _EnvDoc("KVPRESS_ASCEND_RATIO", "0.5", "Compression ratio (fraction of KV tokens removed)."),
    _EnvDoc("KVPRESS_ASCEND_WINDOW", "64", "SnapKV observation window in tokens."),
    _EnvDoc("KVPRESS_ASCEND_N_SINK", "4", "StreamingLLM sink token count."),
    _EnvDoc("KVPRESS_ASCEND_BLOCK_MODE", "mean", "Token->block score aggregation: mean | max."),
    _EnvDoc("KVPRESS_ASCEND_PER_LAYER_RATIOS", "-", "JSON list of per-layer ratios (per_layer press)."),
    _EnvDoc("KVPRESS_ASCEND_LAYERS", "all", "Compressed layer range, e.g. 0-31."),
    _EnvDoc("KVPRESS_ASCEND_MID_PREFILL", "1", "Compress mid-prefill at budget anchors."),
    _EnvDoc("KVPRESS_ASCEND_MID_PREFILL_BUDGET", "65536", "Tokens between mid-prefill anchors."),
    _EnvDoc("KVPRESS_ASCEND_DECODE_REANCHOR", "1", "Re-compress during decode when tail grows."),
    _EnvDoc("KVPRESS_ASCEND_DECODE_REANCHOR_WINDOW", "8192", "Max new tokens between decode re-anchors."),
    _EnvDoc("KVPRESS_ASCEND_STEP_LOG", "1", "Per-step heartbeat line."),
    _EnvDoc("KVPRESS_ASCEND_DRY_RUN", "0", "Score and log only; never apply views."),
    _EnvDoc("KVPRESS_ASCEND_POLICY", "auto", "auto | primary | defer (vs squeeze-ascend)."),
    _EnvDoc("KVPRESS_ASCEND_LOG", "info", "debug | info | warning"),
]
