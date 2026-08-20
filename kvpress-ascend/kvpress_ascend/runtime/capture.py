"""S1 seam: TND query capture inside the Ascend attention backend forward.

Wraps ``AscendAttentionBackendImpl.forward`` (and the C8 variant) so that
during prefill (and decode, for re-anchoring) the per-request query segments
are appended to the rolling windows of the current StepContext.

Guards (v0.23.0 verified):
  * ``_EXTRA_CTX.capturing``      - graph capture uses fake metadata
  * ``_EXTRA_CTX.is_draft_model`` - MTP/eagle draft forwards run in
    sample_tokens with their own KV groups
  * ``_EXTRA_CTX.in_profile_run`` - warmup/profiling steps
  * ubatch/PCP are skipped (list metadata / use_cp)
"""

from __future__ import annotations

import numpy as np

from kvpress_ascend import registry
from kvpress_ascend.log import get_logger
from kvpress_ascend.runtime.context import CAPTURE_STATES, current_context

logger = get_logger()

# Optional: patch this seam off entirely (for tests / troubleshooting).
DISABLED = False


def _ascend_ctx_ok() -> bool:
    try:
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    except Exception:
        return True  # not on NPU (simulation)
    if getattr(_EXTRA_CTX, "capturing", False):
        return False
    if getattr(_EXTRA_CTX, "is_draft_model", False):
        return False
    if getattr(_EXTRA_CTX, "in_profile_run", False):
        return False
    return True


def maybe_capture_query(layer, query, attn_metadata) -> int:
    """Best-effort capture; never raises into the attention path.

    Returns the number of query segments appended.  Every early return bumps a
    ``cap_*`` diagnostic counter so a real-machine "no_q everywhere" can be
    attributed to the exact guard (heartbeat prints cap=appended/skipped).
    ``mark_hit`` fires ONLY when at least one segment was appended - the hit
    counter is a capture-success signal, not a "forward was called" signal.
    """
    if DISABLED:
        return 0
    ctx = current_context()
    if ctx is None or query is None or attn_metadata is None:
        registry.bump("cap_ctx_none")
        return 0
    if query.dim() != 3:  # (T, heads, hd)
        registry.bump("cap_not_3d")
        return 0
    if not _ascend_ctx_ok():
        registry.bump("cap_guarded")  # capturing / draft / profile
        return 0
    layer_name = getattr(layer, "layer_name", None)
    if layer_name is None or layer_name not in ctx.rs.target_layers:
        registry.bump("cap_layer_miss")
        return 0
    state = getattr(attn_metadata, "attn_state", None)
    state_name = getattr(state, "name", state)  # G8: Enum name, not .value
    if state_name not in CAPTURE_STATES:
        registry.bump("cap_state_miss")
        return 0
    q_lens = getattr(attn_metadata, "actual_seq_lengths_q", None)
    if not q_lens:
        registry.bump("cap_no_qlens")
        return 0
    req_ids = ctx.req_ids
    n = min(len(req_ids), len(q_lens))
    if n == 0:
        registry.bump("cap_no_reqs")
        return 0
    starts = np.cumsum([0] + [int(x) for x in q_lens[:n]])
    if starts[-1] > query.shape[0]:
        registry.bump("cap_len_mismatch")  # padding mismatch; skip defensively
        return 0
    appended = 0
    for i in range(n):
        lo, hi = int(starts[i]), int(starts[i + 1])
        if hi <= lo:
            continue
        q_i = query[lo:hi]
        if q_i.shape[0] > 0:
            ctx.append_query(req_ids[i], layer_name, q_i)
            appended += 1
    if appended > 0:
        registry.mark_hit("backend_forward")
        registry.bump("cap_appended", appended)
    else:
        registry.bump("cap_zero_len")
    return appended


def make_backend_forward_wrapper(orig):
    """Wrap AscendAttentionBackendImpl.forward with fail-soft capture."""
    import functools

    @functools.wraps(orig)
    def wrapped(self, layer, query, key, value, kv_cache, attn_metadata, output=None, *args, **kwargs):
        try:
            maybe_capture_query(layer, query, attn_metadata)
        except Exception:
            registry.bump("skipped_error")
            logger.debug("query capture failed", exc_info=True)
        return orig(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)

    return wrapped
