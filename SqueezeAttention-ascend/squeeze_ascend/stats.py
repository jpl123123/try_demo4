"""SqueezeAttention core: layer-importance capture and 2D budget allocation.

Mechanism (ported from SqueezeAttention, see utils_hh/modify_llama_drop.py):

  1. During prefill, each decoder layer's forward is wrapped: the cosine
     similarity between the layer input (residual) and the layer output is
     computed per token and accumulated per (request, layer).
  2. At prompt completion the per-layer means are clustered with 1D KMeans
     into `clusters` classes; the class with the highest centroid gets
     `kv_class3` budget, the remaining layers share the leftover total budget
     (`num_layers * ini_size - n_class3 * kv_class3`), preserving the total
     KV budget exactly as upstream does.
  3. Each layer then gets a streaming window of `budget_ratio * prompt_len`
     tokens (sink + recent), expressed as window views (see core.window_view).

Adaptation notes (documented in RISK_REGISTER.md):
  * upstream computes cos_sim(residual, residual + attn_output) right after
    the attention residual add; in vllm-ascend the post-attention residual
    sum is fused inside the layer, so the wrapped layer hook measures
    cos_sim(layer_input, layer_output) (after the full layer incl. MLP).
    Both signals rank layers by "how much this layer changes the hidden
    state", so the clustering result is expected to be stable.
  * KMeans is reimplemented in pure numpy (no sklearn dependency).
"""

from __future__ import annotations

import functools
import inspect

import numpy as np

from squeeze_ascend import registry
from squeeze_ascend.log import get_logger
from squeeze_ascend.runtime.context import current_context

logger = get_logger()

PREFILL_STATES = ("PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill")


def kmeans_1d(values: np.ndarray, k: int = 3, iters: int = 200, seed: int = 0) -> np.ndarray:
    """1D Lloyd KMeans. Returns cluster labels (0..k-1), deterministic.

    `values` is (n,) float64.  Centroids are initialized on a sorted grid.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return np.zeros(n, dtype=np.int64)
    centers = np.linspace(vmin, vmax, k)
    # deterministic jitter to break ties
    centers += (rng.random(k) - 0.5) * (vmax - vmin) / max(k, 2) * 0.1
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # assign: distance to centers
        dists = np.abs(values[:, None] - centers[None, :])  # (n, k)
        new_labels = np.argmin(dists, axis=1)
        new_centers = np.array(
            [values[new_labels == c].mean() if (new_labels == c).any() else centers[c]
             for c in range(k)]
        )
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    return labels


def compute_layer_budgets(
    layer_means: np.ndarray,
    ini_size: float,
    kv_class3: float,
    clusters: int,
) -> np.ndarray:
    """Per-layer budget ratios (fraction of prompt length), total conserved.

    Returns (n_layers,) float64.  When kv_class3 == ini_size (uniform mode)
    every layer gets the same ratio.
    """
    layer_means = np.asarray(layer_means, dtype=np.float64).reshape(-1)
    n = layer_means.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if abs(kv_class3 - ini_size) < 1e-9 or n <= clusters:
        return np.full(n, ini_size, dtype=np.float64)
    if np.ptp(layer_means) < 1e-12:
        # indistinguishable layers: no meaningful classes -> uniform budget
        return np.full(n, ini_size, dtype=np.float64)
    labels = kmeans_1d(layer_means, k=clusters)
    centers = np.array(
        [layer_means[labels == c].mean() if (labels == c).any() else -np.inf for c in range(clusters)]
    )
    # class3 = the cluster with the highest centroid
    order = np.argsort(centers)
    class3 = int(order[-1])
    n3 = int((labels == class3).sum())
    n_rest = n - n3
    budget = np.full(n, ini_size, dtype=np.float64)
    if n3 == 0 or n_rest == 0:
        # degenerate assignment (e.g. all points in one class): uniform
        return budget
    a = max(0.01, (n * ini_size - n3 * kv_class3) / n_rest)
    budget[labels != class3] = a
    budget[labels == class3] = kv_class3
    return budget


def budgets_to_windows(budget_ratios: np.ndarray, prompt_len: int, start_size: int) -> np.ndarray:
    """Per-layer window sizes in tokens: ratio * prompt_len (upstream rule)."""
    windows = np.maximum(start_size, np.round(budget_ratios * prompt_len)).astype(np.int64)
    windows = np.minimum(windows, prompt_len)
    return windows


class LayerHookInstaller:
    """Wraps decoder-layer forwards to accumulate per-request cos-sim stats."""

    def __init__(self, rs):
        self.rs = rs
        self._installed = False
        self._wraps = []  # (module, orig_forward)

    def ensure(self, runner) -> None:
        if self._installed:
            return
        model = getattr(runner, "model", None)
        if model is None:
            return
        ok = 0
        for layer_name in self.rs.target_layers:
            try:
                module = _resolve_layer_module(model, layer_name)
                if module is None:
                    continue
                orig = module.forward
                if getattr(orig, "_squeeze_ascend_patched", False):
                    ok += 1
                    continue
                residual_pos = _find_residual_pos(orig)
                rs_ref = self.rs

                @functools.wraps(orig)
                def wrapped(*args, _rs=rs_ref, _layer=layer_name, _orig=orig,
                            _pos=residual_pos, **kwargs):
                    _stash_layer_input(_rs, _layer, args, kwargs, _pos)
                    out = _orig(*args, **kwargs)
                    try:
                        cos = compute_cos_sim_from_holder(_rs, _layer, out)
                        accumulate_cos_stats(_rs, _layer, cos)
                    except Exception:
                        registry.bump("skipped_error")
                        logger.debug("cos-sim capture failed: %s", _layer, exc_info=True)
                    return out

                wrapped._squeeze_ascend_patched = True  # type: ignore[attr-defined]
                module.forward = wrapped  # type: ignore[assignment]
                self._wraps.append((module, orig))
                ok += 1
            except Exception:
                logger.debug("layer hook install failed: %s", layer_name, exc_info=True)
        if ok:
            registry.mark_installed("layer_hook")
            logger.info("squeeze layer hooks installed: %d layers", ok)
        self._installed = True

    def restore(self) -> None:
        for module, orig in self._wraps:
            try:
                module.forward = orig
            except Exception:
                pass
        self._wraps = []
        self._installed = False


def _resolve_layer_module(model, layer_name: str):
    """Resolve the decoder layer module from a vllm layer_name.

    layer_name looks like ``model.layers.3.self_attn.attn``; the layer module
    is at the path up to (and including) ``layers.<idx>`` under the model.
    """
    head = layer_name.split(".self_attn")[0]  # e.g. model.layers.3
    module = model
    for part in head.split("."):
        if part.isdigit():
            try:
                module = module[int(part)]
            except (IndexError, TypeError, KeyError):
                return None
        elif hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module


def _find_residual_pos(forward) -> int:
    """Position of the residual parameter in the forward signature, or -1."""
    try:
        sig = inspect.signature(forward)
        for i, (name, p) in enumerate(sig.parameters.items()):
            if name in ("residual", "residual_in"):
                return i
    except (TypeError, ValueError):
        pass
    return -1


def _stash_layer_input(rs, layer_name, args, kwargs, residual_pos):
    """Stash the layer input (residual) for the post-forward cos-sim."""
    ctx = current_context()
    if ctx is None:
        return
    # guards: only real prefill inference on the target model
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    except Exception:
        _EXTRA_CTX = None
    if _EXTRA_CTX is not None:
        if getattr(_EXTRA_CTX, "capturing", False):
            return
        if getattr(_EXTRA_CTX, "is_draft_model", False):
            return
        if getattr(_EXTRA_CTX, "in_profile_run", False):
            return
    runner = ctx.runner
    state = getattr(runner, "attn_state", None)
    state_name = getattr(state, "name", state)  # G8: Enum name, not .value
    if state_name not in PREFILL_STATES:
        return
    inp = None
    if residual_pos >= 0 and residual_pos < len(args):
        inp = args[residual_pos]
    if inp is None:
        inp = kwargs.get("residual")
    if inp is None:
        inp = kwargs.get("residual_in")
    if inp is None:
        # first layer: residual is None, the layer input IS the residual
        inp = kwargs.get("hidden_states")
    if inp is None and args:
        inp = args[0]
    if inp is None:
        return
    try:
        holder = getattr(runner, "_squeeze_ascend_holder", None)
        if holder is None:
            holder = {}
            runner._squeeze_ascend_holder = holder
        holder[layer_name] = inp
    except Exception:
        pass


def compute_cos_sim_from_holder(rs, layer_name, out) -> object:
    """Compute cos-sim(layer input, layer output) after the layer forward.

    Returns a per-token (T,) CPU float tensor or None.  The runner's holder
    is consumed here so we never measure stale inputs.
    """
    runner = getattr(rs, "runner", None)
    if runner is None:
        return None
    holder = getattr(runner, "_squeeze_ascend_holder", None)
    if not holder:
        return None
    inp = holder.pop(layer_name, None)
    if inp is None:
        return None
    out_t = out[0] if isinstance(out, (tuple, list)) else out
    if inp is None or out_t is None:
        return None
    if inp.shape != out_t.shape:
        return None
    import torch
    from torch.nn import functional as F

    cos = F.cosine_similarity(inp.float(), out_t.float(), dim=-1)  # (T,)
    return cos.detach().cpu()


def accumulate_cos_stats(rs, layer_name, cos_t) -> None:
    """Accumulate per-request mean cos-sim for (req, layer)."""
    if cos_t is None or cos_t.numel() == 0:
        return
    ctx = current_context()
    if ctx is None:
        return
    q_lens = getattr(ctx.runner, "actual_seq_lengths_q", None)
    if not q_lens:
        return
    req_ids = ctx.req_ids
    n = min(len(req_ids), len(q_lens))
    if n == 0:
        return
    starts = np.cumsum([0] + [int(x) for x in q_lens[:n]])
    if starts[-1] > cos_t.shape[0]:
        return
    cos_np = cos_t.numpy().astype(np.float64)
    for i in range(n):
        lo, hi = int(starts[i]), int(starts[i + 1])
        if hi <= lo:
            continue
        seg = cos_np[lo:hi]
        stats = rs.stats.setdefault(req_ids[i], {}).setdefault(layer_name, [0.0, 0])
        stats[0] += float(seg.sum())
        stats[1] += int(seg.size)
    registry.mark_hit("layer_hook")
