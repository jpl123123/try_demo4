"""Per-step capture context and long-lived runner state.

Lifecycle:
  * S5 (execute_model wrapper) creates a StepContext for the step, sets it as
    the module-global current context, and runs the compression pass in its
    finally block (after the model forward, before sample_tokens).
  * S1 (backend forward wrapper) reads the current context to append TND query
    windows per (request, layer).
  * S4 (attention-metadata wrapper) reads the runner state to apply view rows.

The long-lived per-request state (layouts, windows, anchors) lives on the
runner instance (``runner._kvpress_ascend_rs``) so it survives across steps;
the StepContext itself is rebuilt every step from fresh snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kvpress_ascend import core, registry
from kvpress_ascend.log import get_logger

logger = get_logger()

_CURRENT_CTX: "StepContext | None" = None

# Attention states during which queries are captured.  Decode states are
# included so decode re-anchoring can re-score with a fresh window.
CAPTURE_STATES = (
    "PrefillNoCache",
    "PrefillCacheHit",
    "ChunkedPrefill",
    "DecodeOnly",
    "SpecDecoding",
)


def current_context() -> "StepContext | None":
    return _CURRENT_CTX


@dataclass
class QueryWindow:
    """Rolling ring buffer of the last `window` queries for (req, layer)."""

    window: int
    buf: object = None  # torch.Tensor (window, heads, hd)
    wr: int = 0
    count: int = 0

    def append(self, q) -> None:
        if q is None or q.shape[0] == 0:
            return
        n = int(q.shape[0])
        if self.buf is None:
            import torch

            self.buf = torch.empty(
                (self.window, q.shape[1], q.shape[2]), dtype=q.dtype, device=q.device
            )
        if n >= self.window:
            self.buf.copy_(q[-self.window :])
            self.wr = 0
            self.count = self.window
            return
        end = self.wr + n
        if end <= self.window:
            self.buf[self.wr : end] = q
        else:
            first = self.window - self.wr
            self.buf[self.wr :] = q[:first]
            self.buf[: end - self.window] = q[first:]
        self.wr = end % self.window
        self.count = min(self.window, self.count + n)

    def ordered(self):
        """Chronologically ordered queries, (count, heads, hd) or None."""
        if self.buf is None or self.count == 0:
            return None
        return self.buf.roll(-self.wr, dims=0)[: self.count]


@dataclass
class LayoutState:
    """View layout for one (request, layer) after an anchor compression."""

    kept: np.ndarray  # kept block ids (int32, ascending)
    m: int  # ceil(orig_len / bs) at anchor time
    orig_len: int  # token count at anchor time
    kept_tokens: int  # real tokens inside kept blocks


@dataclass
class RowMarker:
    """Buffer row sync state for one (request, layer) row."""

    kept: object = None  # np kept blocks (view row) or None (plain row)
    m: int = 0
    synced: int = 0  # tail blocks synced (view) / row blocks synced (plain)
    first: int = -1  # row[0] at last sync; -1 for empty rows


@dataclass
class ReqState:
    """Long-lived per-request state."""

    req_id: str
    prompt_len: int = 0
    compression_done: bool = False
    last_anchor_len: int = 0
    next_anchor: int = 0
    layouts: dict = field(default_factory=dict)  # layer_name -> LayoutState
    queries: dict = field(default_factory=dict)  # layer_name -> QueryWindow
    last_seen: int = -1  # num_computed at previous step (regression detection)


@dataclass
class RunnerState:
    """Long-lived state attached to the NPUModelRunner instance."""

    cfg: object = None
    target_layers: tuple = ()
    block_size: int = 16
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    req: dict = field(default_factory=dict)  # req_id -> ReqState
    buffers: dict = field(default_factory=dict)  # layer_name -> torch.Tensor
    buf_markers: dict = field(default_factory=dict)  # (req_id, layer) -> RowMarker
    ctx: "StepContext | None" = None
    step_no: int = 0
    num_reqs_padded: int = 0


class StepContext:
    """Per-step snapshot used by capture (S1) and the compression pass (S5)."""

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

    # ------------------------------------------------------------------ #
    # per-request helpers
    # ------------------------------------------------------------------ #
    def req_state(self, req_id: str, prompt_len: int = 0) -> ReqState:
        rs_req = self.rs.req.get(req_id)
        if rs_req is None:
            rs_req = ReqState(req_id=req_id, prompt_len=int(prompt_len))
            rs_req.next_anchor = self.rs.cfg.mid_prefill_budget
            self.rs.req[req_id] = rs_req
        if prompt_len and rs_req.prompt_len != int(prompt_len):
            rs_req.prompt_len = int(prompt_len)
        return rs_req

    def append_query(self, req_id: str, layer_name: str, q) -> None:
        rs_req = self.req_state(req_id)
        qw = rs_req.queries.get(layer_name)
        if qw is None:
            qw = QueryWindow(window=self.rs.cfg.window_size)
            rs_req.queries[layer_name] = qw
        qw.append(q)

    def cleanup_finished(self) -> None:
        finished = getattr(self.scheduler_output, "finished_req_ids", ()) or ()
        for rid in finished:
            self.rs.req.pop(rid, None)
        # drop markers whose request is gone
        for key in [k for k in self.rs.buf_markers if k[0] not in self.rs.req]:
            self.rs.buf_markers.pop(key, None)
        for key in [k for k in self.rs.buffers if False]:  # buffers are per-layer; keep
            pass
        if self.rs.req:
            # prune buffer markers of finished requests only
            pass

    # ------------------------------------------------------------------ #
    # step lifecycle
    # ------------------------------------------------------------------ #
    @classmethod
    def begin(cls, runner, rs: RunnerState, scheduler_output) -> "StepContext":
        global _CURRENT_CTX
        ctx = cls(runner, rs, scheduler_output)
        rs.ctx = ctx
        rs.step_no += 1
        _CURRENT_CTX = ctx
        return ctx

    @classmethod
    def end(cls) -> None:
        global _CURRENT_CTX
        _CURRENT_CTX = None


def ensure_runner_state(runner) -> RunnerState:
    rs = getattr(runner, "_kvpress_ascend_rs", None)
    if rs is None:
        from kvpress_ascend.envs import get_config

        rs = RunnerState(cfg=get_config())
        runner._kvpress_ascend_rs = rs
        _resolve_target_layers(runner, rs)
    return rs


def _resolve_target_layers(runner, rs: RunnerState) -> None:
    """Resolve the full-attention target layers of KV cache group 0.

    Rules (v0.23.0 verified):
      * only groups whose spec exposes a block size are candidates
        (Mamba/GDN state caches are not full attention),
      * layer names containing mtp/draft/encoder are excluded (MTP layers
        have their own KV group and are never rewritten),
      * the layer must be present in static_forward_context (KV bound),
      * cfg.layers range filter applies on the parsed layer index.
    """
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
                rs.num_kv_heads = int(getattr(spec, "num_kv_heads", 0) or rs.num_kv_heads)
                rs.head_dim = int(getattr(spec, "head_size", 0) or rs.head_dim)
                rs.num_heads = int(
                    getattr(spec, "num_heads", 0) or rs.num_kv_heads or rs.num_heads
                )
    except Exception:
        logger.debug("could not resolve kv cache groups", exc_info=True)

    # layer index range filter
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
            "target layers=%d block_size=%d first=%s",
            len(rs.target_layers),
            rs.block_size,
            rs.target_layers[0],
        )


def parse_layer_idx(layer_name: str) -> int:
    try:
        return int(layer_name.split("layers.")[1].split(".")[0])
    except Exception:
        return -1
