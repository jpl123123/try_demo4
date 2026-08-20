"""S5 seam: per-step compression pass engine.

Wraps ``NPUModelRunner.execute_model``.  In the finally block (after the model
forward wrote KV, before sample_tokens updates num_computed) it:

  1. detects compression events per request:
       - COMPLETE: before + scheduled >= prompt (G1), plus the one-step-late
         complement (G20: last_before < prompt <= before),
       - MID: progressive anchors while a long prompt is still prefilling
         (G17: KV must not be exhausted before completion),
       - DECODE: re-anchor when the visible tail grows beyond the configured
         window (bounded view rows),
  2. runs the press scoring + block selection for every target layer,
  3. stores the per-(request, layer) layout state consumed by S4,
  4. emits the COMPRESS log (core params per request) and the step heartbeat
     proving the patch is live.
"""

from __future__ import annotations

import numpy as np

from kvpress_ascend import core, registry
from kvpress_ascend.log import get_logger
from kvpress_ascend.presses import ScoreRequest, score_layer
from kvpress_ascend.runtime.context import (
    LayoutState,
    StepContext,
    ensure_runner_state,
    parse_layer_idx,
)

logger = get_logger()

_MIN_COMPRESS_TOKENS = 32  # do not compress tiny requests

_ALLOWED_PHASES = ("complete", "mid", "decode")


class _SkipLayer(Exception):
    def __init__(self, kind: str, msg: str = ""):
        super().__init__(msg)
        self.kind = kind


def _kv_tensors(runner, layer_name: str):
    """Return (key_cache, value_cache) or None for a layer."""
    try:
        static = runner.compilation_config.static_forward_context
        module = static.get(layer_name)
        if module is None:
            return None
        kv_cache = getattr(module, "kv_cache", None)
        if kv_cache is None:
            return None
        if isinstance(kv_cache, (list, tuple)):
            if len(kv_cache) < 2 or kv_cache[0] is None or kv_cache[1] is None:
                return None  # G21: (None, None) tuple is a real shape
            return kv_cache[0], kv_cache[1]
        if getattr(kv_cache, "dim", lambda: 0)() > 0 and kv_cache.shape[0] == 2:
            return kv_cache[0], kv_cache[1]
        return None
    except Exception:
        return None


def _is_quantized_cache(key_cache) -> bool:
    dtype = getattr(key_cache, "dtype", None)
    name = str(dtype)
    return "int" in name or "uint" in name


def _visible_blocks_for(ctx, rs_req, layer_name: str, m: int, valid: int) -> np.ndarray:
    old = rs_req.layouts.get(layer_name)
    if old is None:
        visible = np.arange(0, min(m, valid), dtype=np.int32)
    else:
        visible = core.visible_blocks(old.kept, old.m, valid)
        visible = visible[visible < valid]
    return np.asarray(visible, dtype=np.int32)


def _gather_keys(key_cache, slots, kv_heads: int) -> object:
    import torch

    flat = key_cache.reshape(-1, kv_heads, key_cache.shape[-1])
    return flat[torch.from_numpy(np.asarray(slots, dtype=np.int64))]


def _compress_layer(ctx, rs, rs_req, req_id: str, layer_name: str,
                    row: np.ndarray, valid: int, m: int, true_len: int,
                    cfg) -> None:
    """Score + select + store layout for one (request, layer)."""
    layer_idx = parse_layer_idx(layer_name)
    ratio = cfg.compression_ratio
    if ratio <= 0.0:
        raise _SkipLayer("ratio_zero")
    if true_len < _MIN_COMPRESS_TOKENS:
        raise _SkipLayer("short")
    n_kept = max(1, int(round(true_len * (1.0 - ratio))))
    if n_kept >= true_len:
        raise _SkipLayer("ratio_zero")

    bs = rs.block_size
    visible = _visible_blocks_for(ctx, rs_req, layer_name, m, valid)
    if visible.size == 0:
        raise _SkipLayer("no_blocks")

    counts = np.array(
        [min(bs, max(0, true_len - int(b) * bs)) for b in visible], dtype=np.int64
    )
    total_visible = int(counts.sum())
    if total_visible < _MIN_COMPRESS_TOKENS:
        raise _SkipLayer("short")

    window_q = None
    keys = None
    if cfg.press == "snapkv":
        qw = rs_req.queries.get(layer_name)
        window_q = qw.ordered() if qw is not None else None
        if window_q is None or window_q.shape[0] == 0:
            raise _SkipLayer("no_q")
        kv = _kv_tensors(ctx.runner, layer_name)
        if kv is None:
            raise _SkipLayer("no_kv")
        key_cache = kv[0]
        if _is_quantized_cache(key_cache):
            raise _SkipLayer("c8")
        # ground truth: the TP-split local cache tensor shapes
        kv_heads = int(key_cache.shape[2]) if key_cache.dim() == 4 else rs.num_kv_heads
        slots = core.slots_for_len(visible, total_visible, bs)
        keys = _gather_keys(key_cache, slots, kv_heads)

    req = ScoreRequest(
        layer_name=layer_name,
        layer_idx=layer_idx,
        window_q=window_q,
        keys=keys,
        seq_len=total_visible,
        head_dim=int(keys.shape[-1]) if keys is not None else rs.head_dim,
        num_heads=int(window_q.shape[1]) if window_q is not None else rs.num_heads,
        num_kv_heads=int(keys.shape[1]) if keys is not None else rs.num_kv_heads,
        block_size=bs,
    )
    tok, forced = score_layer(req, ratio, cfg)
    if tok.shape[0] != total_visible:
        raise _SkipLayer("bad_scores", "score length mismatch")
    forced = forced[forced < valid] if forced.size else forced
    block_scores = core.aggregate_token_scores(
        tok, total_visible, bs, cfg.block_score_mode
    )
    kept = core.select_kept_blocks(
        block_scores, visible, n_kept, true_len, bs, forced
    )
    if kept.size == 0:
        raise _SkipLayer("no_kept")

    kept_tokens = core.token_count_in_blocks(kept, true_len, bs)
    rs_req.layouts[layer_name] = LayoutState(
        kept=kept.astype(np.int32),
        m=m,
        orig_len=true_len,
        kept_tokens=kept_tokens,
    )
    registry.bump("layer_compressed")
    if logger.isEnabledFor(10):  # DEBUG
        logger.debug(
            "layer=%s idx=%d orig=%d kept_blocks=%d kept_tokens=%d view_len=%d",
            layer_name, layer_idx, true_len, kept.size, kept_tokens,
            core.view_len(kept, true_len, bs, true_len),
        )


def _compress_request(ctx, rs, i: int, req_id: str, phase: str, true_len: int) -> int:
    """Compress one request at an anchor. Returns the number of layers done."""
    cfg = rs.cfg
    rs_req = ctx.req_state(req_id)
    runner = ctx.runner
    try:
        table = runner.input_batch.block_table[0]
        row_idx = int(runner.input_batch.req_id_to_index.get(req_id, i))
        valid = int(table.num_blocks_per_row[row_idx])
        row = np.asarray(table.block_table.np[row_idx], dtype=np.int32)
    except Exception:
        registry.bump("skipped_bad_row")
        return 0
    if valid == 0:
        registry.bump("skipped_no_blocks")
        return 0
    bs = rs.block_size
    m = core.blocks_for_len(true_len, bs)

    layers_done = 0
    for layer_name in rs.target_layers:
        try:
            _compress_layer(ctx, rs, rs_req, req_id, layer_name, row, valid, m,
                            true_len, cfg)
            layers_done += 1
        except _SkipLayer as e:
            registry.bump("skipped_%s" % e.kind)
            if logger.isEnabledFor(10):
                logger.debug("skip layer %s (%s)", layer_name, e)
        except Exception:
            registry.bump("skipped_error")
            logger.debug("layer compression failed: %s", layer_name, exc_info=True)
    return layers_done


def _finish_compress(ctx, rs, req_id: str, phase: str, true_len: int,
                     layers_done: int) -> None:
    cfg = rs.cfg
    rs_req = ctx.req_state(req_id)
    if layers_done == 0:
        registry.bump("skipped_all_layers")
        return
    if phase == "complete":
        rs_req.compression_done = True
        registry.bump("reqs_compressed")
    elif phase == "mid":
        registry.bump("anchors_mid")
    else:
        registry.bump("anchors_decode")
    rs_req.last_anchor_len = true_len
    if not rs_req.compression_done:
        rs_req.next_anchor = true_len + cfg.mid_prefill_budget
    registry.set_heartbeat_params(
        press=cfg.press,
        ratio=cfg.compression_ratio,
        window=cfg.window_size,
        sink=cfg.n_sink,
        block_mode=cfg.block_score_mode,
    )
    logger.info(
        "COMPRESS req=%s phase=%s press=%s ratio=%.3f orig=%d n_kept=%d "
        "layers=%d/%d dry_run=%s",
        req_id,
        phase,
        cfg.press,
        cfg.compression_ratio,
        true_len,
        max(1, int(round(true_len * (1.0 - cfg.compression_ratio)))),
        layers_done,
        len(rs.target_layers),
        cfg.dry_run,
    )


def run_compression_pass(ctx: StepContext) -> None:
    """Anchor detection + compression + cleanup + heartbeat (S5 finally)."""
    rs = ctx.rs
    cfg = rs.cfg
    n_prefill = 0
    n_decode = 0
    req_ids = ctx.req_ids
    for i, req_id in enumerate(req_ids):
        before = int(ctx.num_computed[i]) if i < len(ctx.num_computed) else 0
        sched = int(ctx.num_scheduled[i]) if i < len(ctx.num_scheduled) else 0
        prompt = int(ctx.num_prompt[i]) if i < len(ctx.num_prompt) else 0
        true_len = before + sched
        rs_req = ctx.req_state(req_id, prompt_len=prompt)
        if prompt <= 0:
            continue
        if before + sched < prompt:
            n_prefill += 1
        else:
            n_decode += 1

        phase = None
        if not rs_req.compression_done:
            if before + sched >= prompt:
                phase = "complete"
            else:
                # G20 complement: scheduler may have finished the prompt in the
                # previous step while num_computed was only updated later.
                if 0 < prompt <= before and rs_req.last_seen < prompt:
                    phase = "complete"
        if phase is None and not rs_req.compression_done and cfg.mid_prefill:
            if before + sched < prompt and true_len >= rs_req.next_anchor:
                phase = "mid"
        if phase is None and rs_req.compression_done and cfg.decode_reanchor:
            if true_len - rs_req.last_anchor_len >= cfg.decode_reanchor_window:
                phase = "decode"

        if phase is not None and phase in _ALLOWED_PHASES:
            layers_done = _compress_request(ctx, rs, i, req_id, phase, true_len)
            _finish_compress(ctx, rs, req_id, phase, true_len, layers_done)
        rs_req.last_seen = before

    ctx.cleanup_finished()
    n_viewed = sum(
        1 for r in rs.req.values() if r.layouts
    )
    registry.bump("steps_prefill" if n_prefill else "steps_decode")
    registry.set_heartbeat_params(
        num_reqs=len(req_ids),
        press=cfg.press,
        ratio=cfg.compression_ratio,
        window=cfg.window_size,
        sink=cfg.n_sink,
        block_mode=cfg.block_score_mode,
        reqs_viewed=n_viewed,
    )
    registry.heartbeat(rs.step_no)


def make_execute_model_wrapper(orig):
    """Wrap NPUModelRunner.execute_model (fail-soft)."""
    import functools

    @functools.wraps(orig)
    def wrapped(self, scheduler_output, intermediate_tensors=None):
        rs = ensure_runner_state(self)
        ctx = StepContext.begin(self, rs, scheduler_output)
        try:
            return orig(self, scheduler_output, intermediate_tensors)
        finally:
            try:
                run_compression_pass(ctx)
            except Exception:
                registry.bump("skipped_error")
                logger.debug("compression pass failed", exc_info=True)
            StepContext.end()

    return wrapped
