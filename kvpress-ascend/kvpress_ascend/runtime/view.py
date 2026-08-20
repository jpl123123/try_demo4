"""S4 seam: per-layer view-row application on attention metadata.

After ``_build_attention_metadata`` returns, layers that have a compressed
layout for any request in the batch get a shallow-copied metadata whose
``block_tables`` is a per-layer persistent buffer holding the view rows
(``[kept] + [true row m..]``) and whose ``seq_lens``/``seq_lens_cpu``/
``seq_lens_list`` carry the per-layer view lengths.

Cost control: block-table rows are append-only in vllm (except add_row /
move/swap, detected via the first-block signature), so each step only copies
the newly appended tail blocks per (request, layer) into the persistent
buffers.

Prefix-cache safety: physical KV cache content is never modified - only the
per-step metadata that attention reads (the scheduler's hash table is keyed
on the original rows, which stay untouched).
"""

from __future__ import annotations

import copy

import numpy as np

from kvpress_ascend import core, registry
from kvpress_ascend.log import get_logger
from kvpress_ascend.runtime.context import RowMarker, ensure_runner_state

logger = get_logger()


def _layer_buffer(rs, layer_name: str, prototype) -> object:
    """Persistent per-layer view buffer, grown on demand."""
    buf = rs.buffers.get(layer_name)
    rows = int(prototype.shape[0])
    width = int(prototype.shape[1])
    if buf is None or buf.shape[0] < rows or buf.shape[1] != width:
        import torch

        buf = torch.zeros((max(rows, buf.shape[0] if buf is not None else 0), width),
                          dtype=torch.int32, device=prototype.device)
        rs.buffers[layer_name] = buf
        # Reset markers for this layer: full re-sync on next step.
        for key in [k for k in rs.buf_markers if k[1] == layer_name]:
            rs.buf_markers.pop(key, None)
    return buf


def _sync_row(buf, i: int, row: np.ndarray, valid: int, marker: RowMarker,
              kept, m: int) -> None:
    """Sync one buffer row from the true row (append-only aware)."""
    first = int(row[0]) if valid > 0 else -1
    view = kept is not None
    expect_m = m if view else 0
    expect_synced = (valid - m) if view else valid
    need_full = (
        marker.first != first
        or marker.synced > max(expect_synced, 0)
        or (view and (marker.m != m))
        or (view and marker.kept is None)
        or (not view and marker.kept is not None)
    )
    if need_full:
        if view:
            klen = int(len(kept))
            if klen:
                # kept holds LOGICAL block indices; the buffer row must hold
                # the PHYSICAL block ids the true row maps them to.
                buf[i, :klen] = torch_from_np(row[kept], buf)
            n_tail = max(0, valid - m)
            if n_tail:
                buf[i, klen : klen + n_tail] = torch_from_np(row[m : m + n_tail], buf)
            marker.synced = n_tail
        else:
            if valid > 0:
                buf[i, :valid] = torch_from_np(row[:valid], buf)
            marker.synced = valid
        marker.first = first
        marker.m = m
        marker.kept = kept.copy() if kept is not None else None
        return
    # incremental append-only sync
    if view:
        base = int(len(kept)) + marker.synced
        n_new = max(0, valid - (m + marker.synced))
        if n_new:
            buf[i, base : base + n_new] = torch_from_np(
                row[m + marker.synced : m + marker.synced + n_new], buf
            )
        marker.synced += n_new
    else:
        base = marker.synced
        n_new = max(0, valid - marker.synced)
        if n_new:
            buf[i, base : base + n_new] = torch_from_np(row[base : base + n_new], buf)
        marker.synced += n_new


def torch_from_np(arr, ref):
    import torch

    return torch.from_numpy(np.asarray(arr, dtype=np.int32)).to(ref.device)


def apply_views(runner, attn_metadata, spec_decode_common, rs) -> None:
    """Rewrite per-layer metadata for layers with active views (fail-soft)."""
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

        # sync rows + compute per-request view lengths
        row_ids = getattr(runner.input_batch, "req_id_to_index", {}) or {}
        for i, rid, layout in viewed:
            row_idx = int(row_ids.get(rid, i))
            valid = int(table.num_blocks_per_row[row_idx])
            row = np.asarray(table.block_table.np[row_idx], dtype=np.int32)
            marker = rs.buf_markers.get((rid, layer_name))
            if marker is None:
                marker = RowMarker()
                rs.buf_markers[(rid, layer_name)] = marker
            _sync_row(buf, i, row, valid, marker, layout.kept, layout.m)
            true_len = int(seq_list[i]) if i < len(seq_list) else 0
            if true_len > 0:
                new_seq[i] = core.view_len(layout.kept, layout.orig_len, bs, true_len)

        # plain (unviewed) rows still need syncing into this layer's buffer
        for i in range(min(n_reqs, n_seq)):
            if i in [v[0] for v in viewed]:
                continue
            rid = req_ids[i]
            row_idx = int(row_ids.get(rid, i))
            valid = int(table.num_blocks_per_row[row_idx])
            row = np.asarray(table.block_table.np[row_idx], dtype=np.int32)
            marker = rs.buf_markers.get((rid, layer_name))
            if marker is None:
                marker = RowMarker()
                rs.buf_markers[(rid, layer_name)] = marker
            _sync_row(buf, i, row, valid, marker, None, 0)

        # replace this layer's metadata with the view copy
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
    """Wrap NPUModelRunner._build_attention_metadata (fail-soft)."""
    import functools

    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            if kwargs.get("for_cudagraph_capture", False) or (
                len(args) > 8 and bool(args[8])
            ):
                return out
            rs = getattr(self, "_kvpress_ascend_rs", None)
            if rs is None or rs.ctx is None:
                return out
            if isinstance(out, tuple):
                apply_views(self, out[0], out[1] if len(out) > 1 else None, rs)
        except Exception:
            registry.bump("skipped_error")
            logger.debug("view application failed", exc_info=True)
        return out

    return wrapped
