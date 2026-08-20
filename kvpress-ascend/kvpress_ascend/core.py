"""kvpress-ascend core: device-independent KV layout math (pure numpy).

Implements the "view rewrite" layout used to express kvpress compression on
vllm-ascend block-based KV cache without touching physical cache content
(prefix-cache safe).  All functions are pure and CPU-testable (L0).

Layout formulas (verified against vllm-ascend v0.23.0):

    m        = ceil(orig_len / bs)                 # prefill block count
    kept     = selected kept block ids (ascending)
    view_row = [kept blocks] + [true row m .. valid)
    view_len = sum_b min(bs, orig_len - b*bs)      # tokens in kept blocks
               + (true_len - orig_len)             # + all new tokens

Invariants:
    * if orig_len % bs != 0 the last block m-1 is forced into `kept`
      (new decode tokens land in its padding slots; without it the newest
      token is invisible),
    * k = number of kept blocks satisfies the slack invariant
      k*bs - n_kept >= m*bs - orig_len (compression never outruns the
      scheduler's block-table growth),
    * slots used for scoring/reading are always `repeat(row, bs)[:n] * bs +
      arange(n) % bs` for n real tokens (never m*bs: tail-block padding would
      pollute top-k).
"""

from __future__ import annotations

import numpy as np


def blocks_for_len(seq_len: int, bs: int) -> int:
    """Number of blocks needed for seq_len tokens."""
    return (seq_len + bs - 1) // bs if seq_len > 0 else 0


def block_token_counts(seq_len: int, bs: int) -> np.ndarray:
    """Per-block token counts (last block may be partial). Length m."""
    m = blocks_for_len(seq_len, bs)
    counts = np.full(m, bs, dtype=np.int64)
    if m and seq_len % bs:
        counts[-1] = seq_len % bs
    return counts


def token_count_in_blocks(block_ids, seq_len: int, bs: int) -> int:
    """Sum of real token counts inside the given blocks (padding-aware)."""
    total = 0
    for b in block_ids:
        b = int(b)
        if b * bs >= seq_len:
            break
        total += min(bs, seq_len - b * bs)
    return total


def n_blocks_to_cover(n_kept: int, seq_len: int, bs: int) -> int:
    """Smallest k such that k chosen blocks can hold n_kept tokens.

    When the last block is partial (seq_len % bs != 0), the partial block is
    mandatory, so k = 1 + ceil(max(n_kept - partial, 0) / bs).  This is
    exactly the slack invariant: k*bs - n_kept >= m*bs - seq_len.
    """
    if n_kept <= 0:
        return 0
    partial = seq_len % bs
    if partial == 0:
        return blocks_for_len(n_kept, bs)
    return 1 + max(0, (n_kept - partial + bs - 1) // bs)


def select_kept_blocks(
    block_scores: np.ndarray,
    candidates: np.ndarray,
    n_kept: int,
    seq_len: int,
    bs: int,
    forced_blocks=(),
) -> np.ndarray:
    """Select kept block ids from per-block scores over candidate blocks.

    Parameters
    ----------
    block_scores : (V,) float block scores aligned with `candidates`
                   (higher = more important).
    candidates   : (V,) candidate (visible) block ids, ascending.
    n_kept       : number of tokens to keep.
    seq_len      : current true sequence length (for the slack invariant).
    forced_blocks: iterable of block ids that are never dropped.
                   (Sink/window forces are passed by the caller.)

    Returns np.int32 array of kept block ids, ascending.  Guarantees:
      * forced blocks are always present (they win over any score),
      * the partial last block (m-1) is forced when seq_len % bs != 0
        (new decode tokens land in its padding slots),
      * the selection covers at least n_kept tokens (slack invariant).
    """
    candidates = np.asarray(candidates, dtype=np.int32).reshape(-1)
    V = candidates.shape[0]
    if V == 0 or n_kept <= 0:
        return np.zeros(0, dtype=np.int32)
    assert block_scores.shape[0] == V, (block_scores.shape, V)
    m = blocks_for_len(seq_len, bs)
    if m == 0:
        return np.zeros(0, dtype=np.int32)

    forced = set(int(b) for b in forced_blocks)
    forced &= set(int(b) for b in candidates)
    if seq_len % bs != 0 and (m - 1) in candidates:
        forced.add(m - 1)  # rule 1: last partial block must stay visible

    k = n_blocks_to_cover(n_kept, seq_len, bs)
    k = max(k, len(forced))
    k = min(k, V)

    if len(forced) >= V:
        kept = np.array(sorted(forced), dtype=np.int32)
    else:
        rest = [j for j in range(V) if int(candidates[j]) not in forced]
        scores_rest = block_scores[rest]
        n_pick = min(len(rest), max(0, k - len(forced)))
        if n_pick > 0:
            # numpy top-k via argpartition (stable enough; ties are fine)
            idx = np.argpartition(scores_rest, -n_pick)[-n_pick:]
            picked = [int(candidates[rest[j]]) for j in idx.tolist()]
        else:
            picked = []
        kept = np.array(sorted(forced | set(picked)), dtype=np.int32)

    # Safety: never return more blocks than candidates.
    return kept[:V]


def aggregate_token_scores(
    token_scores: np.ndarray,
    seq_len: int,
    bs: int,
    mode: str = "mean",
) -> np.ndarray:
    """Aggregate per-token scores (seq_len,) into per-block scores (m,).

    Padding-aware: the partial last block averages only over its real tokens.
    """
    m = blocks_for_len(seq_len, bs)
    if m == 0:
        return np.zeros(0, dtype=np.float64)
    counts = block_token_counts(seq_len, bs)
    out = np.zeros(m, dtype=np.float64)
    for b in range(m):
        lo = b * bs
        hi = lo + int(counts[b])
        seg = token_scores[lo:hi]
        if seg.size == 0:
            out[b] = 0.0
        elif mode == "max":
            out[b] = float(np.max(seg))
        else:
            out[b] = float(np.mean(seg))
    return out


def view_row_np(kept, true_row: np.ndarray, m: int, valid: int, width: int) -> np.ndarray:
    """Build one view row: [kept] + [true_row[m:valid]] padded to `width`.

    `true_row` is the request's real block table row (CPU np) and `valid` its
    current number of blocks.  FIA reads the row as a block sequence, so the
    tail segment starts exactly at the kept segment.
    """
    kept = np.asarray(kept, dtype=np.int32).reshape(-1)
    tail = np.asarray(true_row, dtype=np.int32)[m:valid]
    row = np.concatenate([kept, tail]) if kept.size or tail.size else np.zeros(0, dtype=np.int32)
    if row.size > width:
        # Should not happen (width covers worst case); truncate defensively.
        row = row[:width]
    out = np.zeros(width, dtype=np.int32)
    out[: row.size] = row
    return out


def view_len(kept, orig_len: int, bs: int, true_len: int) -> int:
    """Number of tokens the view exposes: kept tokens + all new tokens.

    At anchor time (true_len == orig_len) this is exactly the number of kept
    tokens - the compression effect.  New tokens (true_len > orig_len) are
    always fully visible (they land in the forced last partial block or the
    tail blocks, both part of the view).
    """
    kept_tokens = token_count_in_blocks(kept, orig_len, bs)
    return kept_tokens + max(0, true_len - orig_len)


def visible_blocks(kept, m: int, valid: int) -> np.ndarray:
    """Block ids currently visible through the view: kept + true tail."""
    kept = np.asarray(kept, dtype=np.int32).reshape(-1)
    if valid > m:
        return np.concatenate([kept, np.arange(m, valid, dtype=np.int32)])
    return kept


def slots_for_len(true_row: np.ndarray, n: int, bs: int) -> np.ndarray:
    """Physical cache slots for the first n real tokens of `true_row`.

    K3 rule: repeat(row, bs)[:n] - never m*bs (tail padding would pollute).
    """
    row = np.asarray(true_row, dtype=np.int64).reshape(-1)
    reps = np.repeat(row, bs)[:n]
    ar = np.arange(n, dtype=np.int64) % bs
    return reps * bs + ar


def streaming_token_scores(seq_len: int, n_sink: int, n_kept: int) -> np.ndarray:
    """StreamingLLM-style per-token scores: sink + recent kept, middle pruned.

    Returns (seq_len,) float64: 1.0 for sink/recent tokens, 0.0 for the middle
    region that is eligible for pruning.
    """
    scores = np.ones(seq_len, dtype=np.float64)
    if seq_len <= n_sink:
        return scores
    n_pruned = max(0, seq_len - n_kept)
    lo = min(seq_len, n_sink)
    hi = min(seq_len, n_sink + n_pruned)
    if lo < hi:
        scores[lo:hi] = 0.0
    return scores


def window_block_forced(seq_len: int, window: int, bs: int) -> np.ndarray:
    """Blocks covering the last `window` tokens (SnapKV observation window)."""
    if seq_len <= 0 or window <= 0:
        return np.zeros(0, dtype=np.int32)
    first = max(0, seq_len - window)
    first_block = first // bs
    last_block = blocks_for_len(seq_len, bs) - 1
    return np.arange(first_block, last_block + 1, dtype=np.int32)


def sink_block_forced(n_sink: int, bs: int) -> np.ndarray:
    """Blocks covering the first n_sink tokens."""
    if n_sink <= 0:
        return np.zeros(0, dtype=np.int32)
    return np.arange(0, blocks_for_len(n_sink, bs), dtype=np.int32)


def window_view_layout(
    true_len: int,
    bs: int,
    window: int,
    start_size: int,
) -> tuple[np.ndarray, int, int]:
    """SqueezeAttention-style window layout (used by the squeeze package).

    Returns (view_blocks, view_len, recent_first):
      * sink blocks [0, ceil(start_size/bs)) + recent blocks
        [recent_first, ceil(true_len/bs)),
      * recent_first clamped so the recent region never overlaps the sink
        region (duplicate-block guard),
      * view_len = true_len - (recent_first - sink_blocks) * bs (capped).
    """
    m = blocks_for_len(true_len, bs)
    sink_blocks = blocks_for_len(start_size, bs)
    recent = max(0, window - start_size)
    recent_first = max(sink_blocks, (true_len - recent) // bs) if recent > 0 else m
    recent_first = min(recent_first, m)
    view_blocks = np.concatenate(
        [
            np.arange(0, sink_blocks, dtype=np.int32),
            np.arange(recent_first, m, dtype=np.int32),
        ]
    ) if (sink_blocks or m > recent_first) else np.zeros(0, dtype=np.int32)
    recent_part = max(0, true_len - recent_first * bs)
    view_len = min(true_len, sink_blocks * bs + recent_part)
    return view_blocks, view_len, recent_first
