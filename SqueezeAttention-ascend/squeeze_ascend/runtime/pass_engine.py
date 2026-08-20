"""S5 seam: per-step anchor/clustering pass (squeeze-ascend).

Wraps ``NPUModelRunner.execute_model``.  In the finally block:

  1. lazily installs the decoder-layer cos-sim hooks (needs runner.model),
  2. detects anchors per request:
       - MID: progressive uniform windows while a long prompt is prefilling
         (bounded KV during prefill; the window slides automatically),
       - COMPLETE: 1D KMeans clustering of the per-layer cos-sim means ->
         per-layer budgets (total conserved) -> per-layer windows,
  3. stores the per-(request, layer) WindowLayout consumed by S4,
  4. emits the CLUSTER log (core parameters) and the step heartbeat.
"""

from __future__ import annotations

import numpy as np

from squeeze_ascend import registry
from squeeze_ascend.log import get_logger
from squeeze_ascend.runtime.context import StepContext, WindowLayout
from squeeze_ascend.stats import (
    budgets_to_windows,
    compute_layer_budgets,
    kmeans_1d,
)

logger = get_logger()

_MIN_PROMPT = 64  # do not compress tiny requests


def _layer_means(rs, req_id: str) -> np.ndarray:
    """Per-layer mean cos-sim for a request, aligned with target layers."""
    stats = rs.stats.get(req_id, {})
    means = []
    for layer_name in rs.target_layers:
        s = stats.get(layer_name)
        if s is not None and s[1] > 0:
            means.append(s[0] / s[1])
        else:
            means.append(np.nan)
    return np.asarray(means, dtype=np.float64)


def _apply_budgets(ctx, rs, req_id: str, phase: str, true_len: int, prompt: int) -> None:
    cfg = rs.cfg
    rs_req = ctx.req_state(req_id, prompt_len=prompt)
    if prompt < _MIN_PROMPT:
        registry.bump("skipped_short")
        return
    if cfg.dry_run:
        registry.bump("dry_run")
        return

    if phase == "mid":
        # provisional uniform windows proportional to the current length
        window = max(cfg.start_size, int(round(cfg.ini_size * true_len)))
        for layer_name in rs.target_layers:
            rs_req.layouts[layer_name] = WindowLayout(window=window, start_size=cfg.start_size)
        rs_req.last_anchor_len = true_len
        rs_req.next_anchor = true_len + cfg.mid_prefill_budget
        registry.bump("anchors_mid")
        logger.info(
            "ANCHOR req=%s phase=mid ini_size=%.3f window=%d layers=%d dry_run=%s",
            req_id, cfg.ini_size, window, len(rs.target_layers), cfg.dry_run,
        )
        return

    # phase == "complete": cluster and allocate per-layer budgets
    mode = "uniform" if abs(cfg.kv_class3 - cfg.ini_size) < 1e-9 else "squeeze"
    means = _layer_means(rs, req_id)
    if np.isnan(means).all():
        registry.bump("skipped_no_stats")
        budget_ratios = np.full(len(rs.target_layers), cfg.ini_size)
        mode = "uniform(fallback)"
    else:
        means = np.where(np.isnan(means), np.nanmean(means), means)
        budget_ratios = compute_layer_budgets(means, cfg.ini_size, cfg.kv_class3, cfg.clusters)
        if mode == "squeeze":
            labels = kmeans_1d(means, k=cfg.clusters)
            centers = np.array(
                [means[labels == c].mean() if (labels == c).any() else -np.inf
                 for c in range(cfg.clusters)]
            )
            class3 = int(np.argsort(centers)[-1])
            registry.bump("reqs_clustered")
            logger.info(
                "CLUSTER req=%s mode=%s layers=%d class3_layers=%d ini_size=%.3f "
                "kv_class3=%.3f budgets_min=%.3f budgets_max=%.3f dry_run=%s",
                req_id, mode, len(rs.target_layers), int((labels == class3).sum()),
                cfg.ini_size, cfg.kv_class3,
                float(budget_ratios.min()), float(budget_ratios.max()), cfg.dry_run,
            )
    windows = budgets_to_windows(budget_ratios, prompt, cfg.start_size)
    for j, layer_name in enumerate(rs.target_layers):
        rs_req.layouts[layer_name] = WindowLayout(window=int(windows[j]), start_size=cfg.start_size)
    rs_req.compression_done = True
    rs_req.last_anchor_len = true_len
    registry.bump("reqs_compressed")
    logger.info(
        "COMPRESS req=%s phase=complete mode=%s ini_size=%.3f start=%d prompt=%d "
        "windows_min=%d windows_max=%d layers=%d dry_run=%s",
        req_id, mode, cfg.ini_size, cfg.start_size, prompt,
        int(windows.min()), int(windows.max()), len(rs.target_layers), cfg.dry_run,
    )


def run_compression_pass(ctx: StepContext) -> None:
    rs = ctx.rs
    cfg = rs.cfg
    try:
        rs.hook_installer.ensure(ctx.runner)
    except Exception:
        registry.bump("skipped_error")
        logger.debug("layer hook install failed", exc_info=True)

    n_prefill = 0
    n_decode = 0
    for i, req_id in enumerate(ctx.req_ids):
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
            elif 0 < prompt <= before and rs_req.last_seen < prompt:
                phase = "complete"  # G20 complement (num_computed updates late)
        if phase is None and not rs_req.compression_done and cfg.mid_prefill:
            if before + sched < prompt and true_len >= rs_req.next_anchor:
                phase = "mid"
        if phase is not None:
            _apply_budgets(ctx, rs, req_id, phase, true_len, prompt)
        rs_req.last_seen = before

    ctx.cleanup_finished()
    n_viewed = sum(1 for r in rs.req.values() if r.layouts)
    mode = "uniform" if abs(cfg.kv_class3 - cfg.ini_size) < 1e-9 else "squeeze"
    registry.bump("steps_prefill" if n_prefill else "steps_decode")
    registry.set_heartbeat_params(
        num_reqs=len(ctx.req_ids),
        mode=mode,
        ini_size=cfg.ini_size,
        start_size=cfg.start_size,
        kv_class3=cfg.kv_class3,
        clusters=cfg.clusters,
        reqs_viewed=n_viewed,
    )
    registry.heartbeat(rs.step_no)


def make_execute_model_wrapper(orig):
    import functools

    @functools.wraps(orig)
    def wrapped(self, scheduler_output, intermediate_tensors=None):
        from squeeze_ascend.runtime.context import StepContext, ensure_runner_state

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
