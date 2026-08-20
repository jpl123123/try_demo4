"""Seam registry: probe marks, counters and the per-step heartbeat.

The heartbeat is the proof that the patch really entered the core code:
every inference step prints one line with the seam probes
(``seams=installed/total hit=N FAIL=...``) plus core parameters and counters.
"""

from __future__ import annotations

import threading
from collections import defaultdict

from squeeze_ascend.envs import get_config
from squeeze_ascend.log import get_logger

logger = get_logger()

# Seams of this adapter.
SEAMS = (
    "layer_hook",            # S6: decoder layer forward wrap (cos-sim capture)
    "build_attn_metadata",   # S4: _build_attention_metadata (window views)
    "execute_model",         # S5: execute_model (context + clustering + heartbeat)
)

# Seams installed eagerly at engine.install(); layer_hook is installed lazily
# at the first inference step (the model only exists after load_model).
EAGER_SEAMS = ("build_attn_metadata", "execute_model")

_installed: set = set()
_counters: defaultdict = defaultdict(int)
_lock = threading.Lock()
_hit = 0
_step = 0
_deferred_reason: str | None = None

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
    global _hit, _step, _deferred_reason
    with _lock:
        _installed.clear()
        _counters.clear()
        _hit = 0
        _step = 0
        _deferred_reason = None
        _heartbeat_params.clear()


def heartbeat(step: int | None = None) -> None:
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
    core = "core=%s ini=%.3f start=%d class3=%s clusters=%d" % (
        params.get("mode", "-"),
        params.get("ini_size", 0.0),
        params.get("start_size", 0),
        params.get("kv_class3", "-"),
        params.get("clusters", 0),
    )
    per_req = "prefill=%d decode=%d viewed=%d clustered=%d anchored=%d" % (
        counters.get("steps_prefill", 0),
        counters.get("steps_decode", 0),
        counters.get("reqs_viewed", 0),
        counters.get("reqs_clustered", 0),
        counters.get("anchors_mid", 0),
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
    cfg = get_config()
    if _deferred_reason is not None:
        return "squeeze-ascend deferred: %s" % _deferred_reason
    installed = sorted(_installed)
    fail = [s for s in EAGER_SEAMS if s not in installed]
    status = "OK" if not fail else "FAILED"
    mode = "squeeze" if cfg.kv_class3 != cfg.ini_size else "uniform"
    return (
        "squeeze-ascend installed seams=%d/%d (%s) mode=%s ini_size=%.3f "
        "start_size=%d kv_class3=%.3f dry_run=%s | %s"
        % (
            len(installed),
            len(SEAMS),
            ",".join(installed) or "-",
            mode,
            cfg.ini_size,
            cfg.start_size,
            cfg.kv_class3,
            cfg.dry_run,
            status,
        )
    )
