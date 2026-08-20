"""S4 seam: per-layer window-view application (squeeze-ascend).

After ``_build_attention_metadata`` returns, layers whose requests have a
window layout get shallow-copied metadata whose ``block_tables`` is a
per-layer persistent buffer holding the window view rows
(``[sink blocks] + [recent blocks]``) and whose seq_lens carry the window
view lengths (``core.window_view_layout``).

Sync strategy: the sink part is written once; the recent part slides forward
as the true length grows.  Append-only steps copy only the newly entered tail
blocks; steps where the window front advances (every ~block_size tokens) do a
full CPU assembly of that row (rare, amortized small).  Row changes from
add_row/move/swap/preemption are detected via the first-block signature and
force a full resync.
"""

from __future__ import annotations

import copy

import numpy as np

from squeeze_ascend import registry
from squeeze_ascend.log import get_logger
from squeeze_ascend.runtime.context import RowMarker

logger = get_logger()


def _layer_buffer(rs, layer_name: str, prototype) -> object:
    buf = rs.buffers.get(layer_name)
    rows = int(prototype.shape[0])
    width = int(prototype.shape[1])
    if buf is None or buf.shape[0] < rows or buf.shape[1] != width:
        import torch

        buf = torch.zeros((max(rows, buf.shape[0] if buf is not None else 0), width),
                          dtype=torch.int32, device=prototype.device)
        rs.buffers[layer_name] = buf
        for key in [k for k in rs.buf_markers if k[1] == layer_name]:
            rs.buf_markers.pop(key, None)
    return buf


def _torch_row(arr, ref):
    import torch

    return torch.from_numpy(np.asarray(arr, dtype=np.int32)).to(ref.device)


def _window_view_params(true_len: int, bs: int, window: int, start_size: int):
    """Returns (sink_blocks, recent_first, recent_last, view_len) or None."""
    if true_len <= window:
        return None  # no rewrite: full row is already the window
    m = (true_len + bs - 1) // bs
    sink_blocks = (start_size + bs - 1) // bs if start_size > 0 else 0
    recent = max(0, window - start_size)
    recent_first = max(sink_blocks, (true_len - recent) // bs) if recent > 0 else m
    recent_first = min(recent_first, m)
    recent_last = m
    view_len = min(true_len, sink_blocks * bs + max(0, true_len - recent_first * bs))
    return sink_blocks, recent_first, recent_last, view_len


def _sync_window_row(buf, i: int, row: np.ndarray, valid: int, marker: RowMarker,
                     sink_blocks: int, recent_first: int, recent_last: int) -> None:
    first = int(row[0]) if valid > 0 else -1
    need_full = (
        marker.first != first
        or not marker.sink_synced
        or recent_last > valid
        or recent_first < marker.recent_first
    )
    if need_full:
        # full assembly (also covers shift-back / row changes)
        n_recent = max(0, recent_last - recent_first)
        parts = []
        if sink_blocks:
            parts.append(row[0:sink_blocks])
        if n_recent:
            parts.append(row[recent_first:recent_last])
        full = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
        full = full[: buf.shape[1]]
        buf[i, : full.shape[0]] = _torch_row(full, buf)
        marker.first = first
        marker.sink_synced = True
        marker.recent_first = recent_first
        marker.recent_last = recent_last
        return
    # append-only: recent_first unchanged
    if recent_first == marker.recent_first:
        if recent_last > marker.recent_last:
            base = sink_blocks + (marker.recent_last - marker.recent_first)
            n_new = recent_last - marker.recent_last
            buf[i, base : base + n_new] = _torch_row(
                row[marker.recent_last:recent_last], buf
            )
            marker.recent_last = recent_last
        return
    # recent_first advanced (window slid forward): shift + append via assembly
    if recent_first > marker.recent_first:
        n_recent = max(0, recent_last - recent_first)
        parts = []
        if sink_blocks:
            parts.append(row[0:sink_blocks])
        if n_recent:
            parts.append(row[recent_first:recent_last])
        full = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
        full = full[: buf.shape[1]]
        buf[i, : full.shape[0]] = _torch_row(full, buf)
        marker.recent_first = recent_first
        marker.recent_last = recent_last


def _sync_plain_row(buf, i: int, row: np.ndarray, valid: int, marker: RowMarker) -> None:
    first = int(row[0]) if valid > 0 else -1
    if marker.first != first or marker.recent_last > valid:
        if valid > 0:
            buf[i, :valid] = _torch_row(row[:valid], buf)
        marker.first = first
        marker.recent_last = valid
        return
    if valid > marker.recent_last:
        buf[i, marker.recent_last : valid] = _torch_row(
            row[marker.recent_last : valid], buf
        )
        marker.recent_last = valid


def apply_views(runner, attn_metadata, spec_decode_common, rs) -> None:
    if rs.cfg is None or not rs.target_layers:
        return
    if isinstance(attn_metadata, (list, tuple)):
        registry.bump("skipped_ubatch")
        return
    if getattr(runner, "use_cp", False):
        registry.bump("skipped_cp")
        return
    try:
        table = runner.input_batch.block_table[0]
    except Exception:
        registry.bump("skipped_bad_row")
        return
    req_ids = list(getattr(runner.input_batch, "req_ids", ()) or ())
    if not req_ids:
        return
    bs = rs.block_size
    n_reqs = len(req_ids)
    row_ids = getattr(runner.input_batch, "req_id_to_index", {}) or {}

    for layer_name in rs.target_layers:
        meta = attn_metadata.get(layer_name)
        if meta is None or getattr(meta, "block_tables", None) is None:
            continue
        seqs = getattr(meta, "seq_lens", None)
        if seqs is None:
            continue
        try:
            seq_list = seqs.tolist()
        except Exception:
            continue
        n_seq = len(seq_list)
        viewed = []
        for i in range(min(n_reqs, n_seq)):
            rid = req_ids[i]
            rs_req = rs.req.get(rid)
            layout = (rs_req.layouts.get(layer_name) if rs_req is not None else None)
            if layout is not None:
                viewed.append((i, rid, layout))
        if not viewed:
            continue
        registry.mark_hit("build_attn_metadata")
        if rs.cfg.dry_run:
            registry.bump("dry_run")
            continue

        buf = _layer_buffer(rs, layer_name, meta.block_tables)
        n_fia = len(getattr(meta, "actual_seq_lengths_q", ()) or ())
        new_seq = list(seq_list)
        while len(new_seq) < n_fia:
            new_seq.append(1)

        for i, rid, layout in viewed:
            row_idx = int(row_ids.get(rid, i))
            valid = int(table.num_blocks_per_row[row_idx])
            row = np.asarray(table.block_table.np[row_idx], dtype=np.int32)
            true_len = int(seq_list[i]) if i < len(seq_list) else 0
            marker = rs.buf_markers.get((rid, layer_name))
            if marker is None:
                marker = RowMarker()
                rs.buf_markers[(rid, layer_name)] = marker
            params = _window_view_params(true_len, bs, layout.window, layout.start_size)
            if params is None:
                _sync_plain_row(buf, i, row, valid, marker)
            else:
                sink_blocks, rf, rl, view_len = params
                _sync_window_row(buf, i, row, valid, marker, sink_blocks, rf, rl)
                new_seq[i] = view_len

        # plain rows of this layer still need syncing into the buffer
        for i in range(min(n_reqs, n_seq)):
            if any(v[0] == i for v in viewed):
                continue
            rid = req_ids[i]
            row_idx = int(row_ids.get(rid, i))
            valid = int(table.num_blocks_per_row[row_idx])
            row = np.asarray(table.block_table.np[row_idx], dtype=np.int32)
            marker = rs.buf_markers.get((rid, layer_name))
            if marker is None:
                marker = RowMarker()
                rs.buf_markers[(rid, layer_name)] = marker
            _sync_plain_row(buf, i, row, valid, marker)

        import torch

        new_meta = copy.copy(meta)
        new_meta.block_tables = buf
        seq_t = torch.tensor(new_seq, dtype=torch.int64)
        new_meta.seq_lens = seq_t
        new_meta.seq_lens_cpu = seq_t
        new_meta.seq_lens_list = new_seq
        attn_metadata[layer_name] = new_meta
        registry.bump("viewed_layers")


def make_build_attn_metadata_wrapper(orig):
    import functools

    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            if kwargs.get("for_cudagraph_capture", False) or (
                len(args) > 8 and bool(args[8])
            ):
                return out
            rs = getattr(self, "_squeeze_ascend_rs", None)
            if rs is None or rs.ctx is None:
                return out
            if isinstance(out, tuple):
                apply_views(self, out[0], out[1] if len(out) > 1 else None, rs)
        except Exception:
            registry.bump("skipped_error")
            logger.debug("view application failed", exc_info=True)
        return out

    return wrapped
