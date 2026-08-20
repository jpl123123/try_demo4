"""L0: pure layout logic (no torch)."""

import numpy as np

from kvpress_ascend import core

BS = 16


def test_view_len_anchor_and_growth():
    kept = np.array([0, 3], dtype=np.int32)
    assert core.view_len(kept, 100, BS, 100) == 32  # compression effect
    assert core.view_len(kept, 100, BS, 150) == 82  # all new tokens visible
    assert core.view_len(kept, 100, BS, 90) == 32  # defensive shrink


def test_slack_invariant():
    for orig, n_kept in [(100, 50), (100, 62), (100, 4), (96, 48), (96, 64),
                         (17, 9), (32, 31), (262144, 131072)]:
        k = core.n_blocks_to_cover(n_kept, orig, BS)
        assert k * BS - n_kept >= core.blocks_for_len(orig, BS) * BS - orig, (orig, n_kept, k)


def test_select_kept_blocks_rules():
    scores = np.arange(7, dtype=np.float64)
    kept = core.select_kept_blocks(scores, np.arange(7, dtype=np.int32), 50, 100, BS, [0])
    assert 0 in kept and 6 in kept  # forced + last partial
    assert core.token_count_in_blocks(kept, 100, BS) >= 50
    # candidates subset (re-anchor)
    vis = np.array([0, 2, 5, 6], dtype=np.int32)
    k2 = core.select_kept_blocks(np.array([1.0, 9.0, 3.0, 4.0]), vis, 60, 100, BS, [6])
    assert 6 in k2 and 2 in k2 and set(k2).issubset(set(vis.tolist()))


def test_streaming_scores():
    tok = core.streaming_token_scores(100, 4, 50)
    assert tok[0] == 1.0 and tok[3] == 1.0
    assert tok[4:54].sum() == 0 and tok[54] == 1.0


def test_slots_k3_never_m_times_bs():
    row = np.arange(7, dtype=np.int64)
    slots = core.slots_for_len(row, 100, BS)
    assert slots.max() < 7 * BS and len(slots) == 100
    assert set(slots[:16].tolist()) == {int(row[0]) * BS + j for j in range(16)}


def test_aggregate_padding_aware():
    tok = np.arange(100, dtype=np.float64)
    blk = core.aggregate_token_scores(tok, 100, BS, "mean")
    assert blk.shape == (7,)
    assert abs(blk[-1] - np.mean(tok[96:100])) < 1e-9
    assert abs(blk[0] - np.mean(tok[0:16])) < 1e-9


def test_window_view_layout_no_overlap():
    vb, vl, rf = core.window_view_layout(200, BS, 32, 4)
    assert rf >= core.blocks_for_len(4, BS)
    assert len(set(vb.tolist())) == len(vb)  # no duplicate blocks (K8)
    assert vl == 16 + (200 - rf * 16)
    # small sequence: no compression
    vb2, vl2, rf2 = core.window_view_layout(20, BS, 32, 4)
    assert vl2 == 20
    # window smaller than start: sink only
    vb3, vl3, _ = core.window_view_layout(100, BS, 2, 4)
    assert vl3 == 16


def test_window_view_layout_extremes():
    # start_size 0
    vb, vl, rf = core.window_view_layout(100, BS, 32, 0)
    assert 0 not in vb
    assert vl == 100 - rf * BS
    # true_len multiple of bs
    vb, vl, rf = core.window_view_layout(96, BS, 32, 4)
    assert vl == 16 + (96 - rf * 16)


def test_view_row_np():
    row = np.array([10, 11, 12, 13, 14], dtype=np.int32)
    kept = np.array([0, 2], dtype=np.int32)
    v = core.view_row_np(kept, row, 3, 5, 8)
    assert v.tolist() == [0, 2, 13, 14, 0, 0, 0, 0]
    v2 = core.view_row_np(kept, row, 3, 3, 8)
    assert v2.tolist() == [0, 2, 0, 0, 0, 0, 0, 0]
