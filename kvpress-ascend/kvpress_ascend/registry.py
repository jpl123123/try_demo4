"""Seam registry: probe marks, counters and the per-step heartbeat.

The heartbeat is the proof that the patch really entered the core code:
every inference step prints one line with the seam probes
(``seams=installed/total hit=N FAIL=...``) plus core parameters and counters.
"""

from __future__ import annotations

import threading
from collections import defaultdict

from kvpress_ascend.envs import get_config
from kvpress_ascend.log import get_logger

logger = get_logger()

# Seams of this adapter (each maps to a wrapped vllm-ascend entry point).
SEAMS = (
    "backend_forward",       # S1: AscendAttentionBackendImpl.forward (TND query capture)
    "backend_forward_c8",    # S1b: AscendC8AttentionBackendImpl.forward
    "build_attn_metadata",   # S4: _build_attention_metadata (view rows + seq_lens)
    "execute_model",         # S5: execute_model (context + compression pass + heartbeat)
)

_installed: set = set()
_counters: defaultdict = defaultdict(int)
_lock = threading.Lock()
_hit = 0
_step = 0
_deferred_reason: str | None = None
_deferred_logged = False

# Core parameters shown in every heartbeat (refreshed by pass_engine).
_heartbeat_params: dict = {}


def mark_installed(name: str) -> None:
    if name in SEAMS:
        with _lock:
            _installed.add(name)


def mark_hit(name: str) -> None:
    global _hit
    if name in SEAMS:
        with _lock:
            _hit += 1


def bump(name: str, value: int = 1) -> None:
    with _lock:
        _counters[name] += value


def set_deferred(reason: str) -> None:
    global _deferred_reason
    _deferred_reason = reason


def is_deferred() -> bool:
    return _deferred_reason is not None


def set_heartbeat_params(**kwargs) -> None:
    _heartbeat_params.update(kwargs)


def get_counters() -> dict:
    with _lock:
        return dict(_counters)


def get_hit() -> int:
    with _lock:
        return _hit


def get_installed() -> set:
    with _lock:
        return set(_installed)


def reset() -> None:
    global _hit, _step, _deferred_reason, _deferred_logged
    with _lock:
        _installed.clear()
        _counters.clear()
        _hit = 0
        _step = 0
        _deferred_reason = None
        _deferred_logged = False
        _heartbeat_params.clear()


def heartbeat(step: int | None = None) -> None:
    """Emit one heartbeat line for this inference step (env-gated)."""
    cfg = get_config()
    if not cfg.step_log:
        return
    global _step
    if step is not None:
        _step = step
    with _lock:
        installed = sorted(_installed)
        counters = dict(_counters)
    if _deferred_reason is not None:
        logger.info(
            "DEFERRED step=%d reason=%s (no monkeypatches installed; passive observer only)",
            _step,
            _deferred_reason,
        )
        return
    fail = [s for s in SEAMS if s not in installed]
    params = _heartbeat_params
    core = (
        "core=%s ratio=%.3f window=%d sink=%d mode=%s"
        % (
            params.get("press", "-"),
            params.get("ratio", 0.0),
            params.get("window", 0),
            params.get("sink", 0),
            params.get("block_mode", "-"),
        )
    )
    per_req = (
        "prefill=%d decode=%d viewed=%d compressed=%d mid=%d reanchor=%d"
        % (
            counters.get("steps_prefill", 0),
            counters.get("steps_decode", 0),
            counters.get("reqs_viewed", 0),
            counters.get("reqs_compressed", 0),
            counters.get("anchors_mid", 0),
            counters.get("anchors_decode", 0),
        )
    )
    skipped = " ".join(
        "%s=%d" % (k, v)
        for k, v in sorted(counters.items())
        if k.startswith("skipped") and v > 0
    )
    if skipped:
        skipped = " skipped: " + skipped
    logger.info(
        "step=%d reqs=%d seams=%d/%d hit=%d FAIL=%s %s %s%s",
        _step,
        params.get("num_reqs", 0),
        len(installed),
        len(SEAMS),
        get_hit(),
        ",".join(fail) if fail else "-",
        core,
        per_req,
        skipped,
    )


def summary() -> str:
    """One-line install summary (printed once at activation)."""
    cfg = get_config()
    if _deferred_reason is not None:
        return "kvpress-ascend deferred: %s" % _deferred_reason
    installed = sorted(_installed)
    fail = [s for s in SEAMS if s not in installed]
    status = "OK" if not fail else "FAILED"
    return (
        "kvpress-ascend installed seams=%d/%d (%s) press=%s ratio=%.3f window=%d "
        "mid_prefill=%s decode_reanchor=%s dry_run=%s | %s"
        % (
            len(installed),
            len(SEAMS),
            ",".join(installed) or "-",
            cfg.press,
            cfg.compression_ratio,
            cfg.window_size,
            cfg.mid_prefill,
            cfg.decode_reanchor,
            cfg.dry_run,
            status,
        )
    )
