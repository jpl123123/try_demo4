"""squeeze-ascend core: window-view layout math (pure numpy).

The SqueezeAttention token strategy is a streaming window per layer:
keep the first `start_size` sink tokens plus the last `window - start_size`
tokens, expressed as a window VIEW row ([sink blocks] + [recent blocks]).
"""

from __future__ import annotations

import numpy as np


def blocks_for_len(seq_len: int, bs: int) -> int:
    return (seq_len + bs - 1) // bs if seq_len > 0 else 0


def window_view_layout(
    true_len: int,
    bs: int,
    window: int,
    start_size: int,
) -> tuple[np.ndarray, int, int]:
    """SqueezeAttention-style window layout.

    Returns (view_blocks, view_len, recent_first):
      * sink blocks [0, ceil(start_size/bs)) + recent blocks
        [recent_first, ceil(true_len/bs)),
      * recent_first clamped so the recent region never overlaps the sink
        region (duplicate-block guard),
      * view_len = sink_blocks*bs + max(0, true_len - recent_first*bs),
        capped at true_len (never reads padding).
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
