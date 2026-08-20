"""Portable press implementations (kvpress score functions, block form).

Each press turns captured data into per-token scores + forced block sets.
The heavy QK^T math runs with torch (GPU at runtime / CPU in simulation);
selection happens later in ``core.select_kept_blocks`` (numpy).

Porting notes (mechanism conversion, kvpress -> block view):
  * kvpress keeps the top-(1-ratio) *tokens* per layer; here the top blocks
    are kept and forced sets (sink / observation window / last partial block)
    mirror kvpress's "max score + 1" paddings.
  * SnapKV scoring uses the post-RoPE query window captured from the Ascend
    backend (TND) against post-RoPE cached keys - the real attention logits,
    up to the same scale factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kvpress_ascend import core


@dataclass
class ScoreRequest:
    """Everything a press needs for one (request, layer) scoring event."""

    layer_name: str
    layer_idx: int
    window_q: object  # torch.Tensor (count, heads, hd) or None
    keys: object  # torch.Tensor (n, kv_heads, hd) or None
    seq_len: int  # number of visible tokens being scored
    head_dim: int
    num_heads: int  # TP-split query heads
    num_kv_heads: int  # TP-split kv heads
    block_size: int


def _pool1d_scores(scores, kernel: int = 5):
    """avg_pool1d with same padding (kvpress SnapKV smoothing)."""
    import torch
    from torch.nn import functional as F

    if kernel <= 1 or scores.shape[0] < kernel:
        return scores
    x = scores.view(1, 1, -1).float()
    pad = kernel // 2
    out = F.avg_pool1d(x, kernel_size=kernel, padding=pad, stride=1).view(-1)
    return out[: scores.shape[0]]


def score_snapkv(req: ScoreRequest, window_size: int, force_keep_window: bool):
    """SnapKV: importance from attention of the recent query window."""
    import torch

    q = req.window_q
    k = req.keys
    if q is None or k is None or q.shape[0] == 0 or req.seq_len == 0:
        raise ValueError("snapkv requires captured queries and cached keys")
    n = req.seq_len
    q = q.float()
    k = k.float()
    w, heads, hd = q.shape
    kv_heads = k.shape[1]
    groups = heads // kv_heads if kv_heads else 1
    scale = math.sqrt(float(hd))
    q = q.view(w, kv_heads, groups, hd)
    attn = torch.einsum("wghd,ohd->wgo", q, k) / scale  # (w, groups, n)
    attn = attn.mean(dim=1)  # mean over kv groups (kvpress semantics)
    attn = torch.softmax(attn.float(), dim=-1)
    scores = attn.mean(dim=0)  # (n,) mean over the observation window
    scores = _pool1d_scores(scores)
    tok = scores.detach().cpu().numpy().astype(np.float64)
    tok = np.maximum(tok, 0.0)
    forced = (
        core.window_block_forced(n, window_size, req.block_size)
        if force_keep_window
        else np.zeros(0, dtype=np.int32)
    )
    return tok, forced


def score_streaming(seq_len: int, n_kept: int, n_sink: int, force_keep_sink: bool, bs: int):
    tok = core.streaming_token_scores(seq_len, n_sink, n_kept)
    forced = (
        core.sink_block_forced(n_sink, bs)
        if force_keep_sink
        else np.zeros(0, dtype=np.int32)
    )
    return tok, forced


def score_random(seq_len: int, rng=None):
    rng = rng or np.random.default_rng()
    return rng.random(seq_len).astype(np.float64), np.zeros(0, dtype=np.int32)


def score_layer(
    req: ScoreRequest,
    ratio: float,
    cfg,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the configured press. Returns (token_scores, forced_blocks)."""
    n_kept = max(0, int(round(req.seq_len * (1.0 - ratio))))
    press = cfg.press
    if press == "snapkv":
        tok, forced = score_snapkv(req, cfg.window_size, cfg.force_keep_window)
    elif press == "streaming":
        tok, forced = score_streaming(
            req.seq_len, n_kept, cfg.n_sink, cfg.force_keep_sink, req.block_size
        )
    elif press == "random":
        tok, forced = score_random(req.seq_len)
    elif press == "per_layer":
        ratios = cfg.per_layer_ratios
        if not ratios or req.layer_idx >= len(ratios):
            raise ValueError(
                "per_layer press requires KVPRESS_ASCEND_PER_LAYER_RATIOS "
                "covering every layer"
            )
        return score_layer(req, float(ratios[req.layer_idx]), cfg)
    else:
        raise ValueError("unknown press: %s" % press)
    return tok, forced
