"""Per-step capture context and long-lived runner state (squeeze-ascend).

Lifecycle mirrors kvpress-ascend:
  * S5 (execute_model wrapper) creates a StepContext for the step and runs
    the anchor/clustering pass in its finally block.
  * S6 (decoder layer forward wrapper) reads the current context and the
    runner's per-step query lengths to accumulate cos-sim stats.
  * S4 (attention-metadata wrapper) reads the runner state to apply the
    per-layer window views.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from squeeze_ascend.log import get_logger

logger = get_logger()

_CURRENT_CTX: "StepContext | None" = None


def current_context() -> "StepContext | None":
    return _CURRENT_CTX


@dataclass
class WindowLayout:
    """Per-(request, layer) streaming window (sink + recent)."""

    window: int  # window size in tokens (layer budget)
    start_size: int  # sink tokens


@dataclass
class RowMarker:
    """Buffer row sync state for one (request, layer) row (window views)."""

    first: int = -1
    sink_synced: bool = False
    recent_first: int = 0  # first block of the recent part in the true row
    recent_last: int = 0  # exclusive last block of the recent part


@dataclass
class ReqState:
    req_id: str
    prompt_len: int = 0
    compression_done: bool = False
    last_anchor_len: int = 0
    next_anchor: int = 0
    layouts: dict = field(default_factory=dict)  # layer_name -> WindowLayout
    last_seen: int = -1


@dataclass
class RunnerState:
    cfg: object = None
    runner: object = None
    target_layers: tuple = ()
    block_size: int = 16
    req: dict = field(default_factory=dict)  # req_id -> ReqState
    stats: dict = field(default_factory=dict)  # req_id -> layer -> [sum, count]
    buffers: dict = field(default_factory=dict)  # layer_name -> torch.Tensor
    buf_markers: dict = field(default_factory=dict)  # (req_id, layer) -> RowMarker
    hook_installer: object = None
    ctx: "StepContext | None" = None
    step_no: int = 0


class StepContext:
    def __init__(self, runner, rs: RunnerState, scheduler_output):
        self.runner = runner
        self.rs = rs
        self.scheduler_output = scheduler_output
        self.req_ids = list(getattr(runner.input_batch, "req_ids", ()) or ())
        self.num_computed = np.asarray(
            getattr(runner.input_batch, "num_computed_tokens_cpu", np.zeros(len(self.req_ids), dtype=np.int64))
        ).astype(np.int64).reshape(-1)
        self.num_prompt = np.asarray(
            getattr(runner.input_batch, "num_prompt_tokens", np.zeros(len(self.req_ids), dtype=np.int64))
        ).astype(np.int64).reshape(-1)
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        if isinstance(scheduled, dict):
            self.num_scheduled = np.array(
                [int(scheduled.get(rid, 0)) for rid in self.req_ids], dtype=np.int64
            )
        else:
            arr = np.asarray(scheduled, dtype=np.int64).reshape(-1)
            if arr.size < len(self.req_ids):
                arr = np.pad(arr, (0, len(self.req_ids) - arr.size))
            self.num_scheduled = arr[: len(self.req_ids)]

    def req_state(self, req_id: str, prompt_len: int = 0) -> ReqState:
        rs_req = self.rs.req.get(req_id)
        if rs_req is None:
            rs_req = ReqState(req_id=req_id, prompt_len=int(prompt_len))
            rs_req.next_anchor = self.rs.cfg.mid_prefill_budget
            self.rs.req[req_id] = rs_req
        if prompt_len and rs_req.prompt_len != int(prompt_len):
            rs_req.prompt_len = int(prompt_len)
        return rs_req

    def cleanup_finished(self) -> None:
        finished = getattr(self.scheduler_output, "finished_req_ids", ()) or ()
        for rid in finished:
            self.rs.req.pop(rid, None)
            self.rs.stats.pop(rid, None)
        for key in [k for k in self.rs.buf_markers if k[0] not in self.rs.req]:
            self.rs.buf_markers.pop(key, None)

    @classmethod
    def begin(cls, runner, rs: RunnerState, scheduler_output) -> "StepContext":
        global _CURRENT_CTX
        ctx = cls(runner, rs, scheduler_output)
        rs.ctx = ctx
        rs.runner = runner
        rs.step_no += 1
        _CURRENT_CTX = ctx
        return ctx

    @classmethod
    def end(cls) -> None:
        global _CURRENT_CTX
        _CURRENT_CTX = None


def ensure_runner_state(runner) -> RunnerState:
    rs = getattr(runner, "_squeeze_ascend_rs", None)
    if rs is None:
        from squeeze_ascend.envs import get_config
        from squeeze_ascend.stats import LayerHookInstaller

        rs = RunnerState(cfg=get_config(), runner=runner)
        runner._squeeze_ascend_rs = rs
        _resolve_target_layers(runner, rs)
        rs.hook_installer = LayerHookInstaller(rs)
    return rs


def _resolve_target_layers(runner, rs: RunnerState) -> None:
    cfg = rs.cfg
    candidates = []
    block_size = None
    try:
        groups = runner.kv_cache_config.kv_cache_groups
        static = runner.compilation_config.static_forward_context
        for group in groups:
            spec = getattr(group, "kv_cache_spec", None)
            spec_bs = getattr(spec, "block_size", None)
            layer_names = list(getattr(group, "layer_names", ()) or ())
            if spec_bs is None:
                continue
            for name in layer_names:
                low = name.lower()
                if "mtp" in low or "draft" in low or "encoder" in low:
                    continue
                if name not in static:
                    continue
                candidates.append(name)
            if block_size is None and candidates:
                block_size = int(spec_bs)
    except Exception:
        logger.debug("could not resolve kv cache groups", exc_info=True)

    if cfg.layers is not None and candidates:
        lo, hi = cfg.layers

        def _idx(name: str):
            try:
                return int(name.split("layers.")[1].split(".")[0])
            except Exception:
                return -1

        candidates = [n for n in candidates if lo <= _idx(n) <= hi]

    rs.target_layers = tuple(candidates)
    rs.block_size = block_size or rs.block_size
    if rs.target_layers:
        logger.info(
            "squeeze target layers=%d block_size=%d first=%s",
            len(rs.target_layers),
            rs.block_size,
            rs.target_layers[0],
        )


def parse_layer_idx(layer_name: str) -> int:
    try:
        return int(layer_name.split("layers.")[1].split(".")[0])
    except Exception:
        return -1
