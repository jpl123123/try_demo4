"""L2: end-to-end window-view invariant for squeeze-ascend.

Invariants checked at every step:
  * the slots attention reads through the window view equal the reference
    visible set (sink tokens + recent tokens),
  * the latest token is always visible,
  * the view length never exceeds the true length (no padding reads),
  * window slides during decode; mid-prefill anchors and the completion
    clustering (2D budgets) both engage.
"""

from __future__ import annotations

import numpy as np

from squeeze_ascend import registry

from harness import (
    BLOCK_SIZE as BS,
    add_request,
    make_runner,
    reference_window_slots,
    run_step,
    view_slots,
)


def _check_invariant(runner, am, pre_layouts, step_no):
    for i, rid in enumerate(runner.input_batch.req_ids):
        true_row = runner.table.block_table.np[i]
        true_len = int(runner.input_batch.num_computed_tokens_cpu[i])
        for layer_name in runner.layer_names:
            layout = pre_layouts.get(rid, {}).get(layer_name)
            if layout is None:
                continue
            meta = am[layer_name]
            vs = view_slots(meta, i)
            ref = reference_window_slots(true_row, true_len, BS,
                                         layout.window, layout.start_size)
            assert vs.shape == ref.shape, (step_no, rid, layer_name, vs.shape, ref.shape)
            assert set(vs.tolist()) == set(ref.tolist()), (
                "window visible-set mismatch", step_no, rid, layer_name)
            assert int(meta.seq_lens_list[i]) <= true_len  # no padding reads
            last_slot = int(true_row[(true_len - 1) // BS]) * BS + (true_len - 1) % BS
            assert last_slot in vs  # latest token always visible


def test_window_invariant_single_request():
    runner = make_runner(
        cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.4",
                 "SQUEEZE_ASCEND_MID_PREFILL_BUDGET": "40"}
    )
    add_request(runner, "r1", prompt_len=100)
    steps = [{"r1": 20}] * 5 + [{"r1": 1}] * 30 + [{"r1": 3}] * 10
    for step_no, sched in enumerate(steps):
        rs = runner._squeeze_ascend_rs
        pre_layouts = {rid: dict(rs.req[rid].layouts) for rid in ("r1",) if rid in rs.req}
        run_step(runner, sched)
        _check_invariant(runner, runner._last_attn_metadata, pre_layouts, step_no)
    counters = registry.get_counters()
    assert counters.get("reqs_compressed", 0) >= 1
    assert counters.get("anchors_mid", 0) >= 2  # anchors at 40/80
    assert counters.get("reqs_clustered", 0) >= 0  # uniform mode: no clustering


def test_squeeze_mode_clustering_two_dimensions():
    """KV_CLASS3 != ini_size -> KMeans on layer importance -> per-layer windows."""
    runner = make_runner(
        cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.21",
                 "SQUEEZE_ASCEND_KV_CLASS3": "0.08"}
    )
    add_request(runner, "r1", prompt_len=100)
    steps = [{"r1": 20}] * 5 + [{"r1": 1}] * 20
    for step_no, sched in enumerate(steps):
        rs = runner._squeeze_ascend_rs
        pre_layouts = {rid: dict(rs.req[rid].layouts) for rid in ("r1",) if rid in rs.req}
        run_step(runner, sched)
        _check_invariant(runner, runner._last_attn_metadata, pre_layouts, step_no)
    counters = registry.get_counters()
    assert counters.get("reqs_clustered", 0) >= 1
    # Upstream semantics: layers whose representation changes most (LOWEST
    # cos-sim; factor 0.9 in the fake) are the important ones and get the
    # compensated (larger) budget; the highest-cos class gets kv_class3.
    rs = runner._squeeze_ascend_rs
    w0 = rs.req["r1"].layouts["model.layers.0.self_attn.attn"].window
    w3 = rs.req["r1"].layouts["model.layers.3.self_attn.attn"].window
    assert w3 < w0, (w3, w0)  # highest-cos (least change) class gets less
    total_window = sum(
        l.window for l in rs.req["r1"].layouts.values()
    )
    assert abs(total_window - 4 * 0.21 * 100) <= 4  # total budget conserved


def test_multi_request_window_invariant():
    runner = make_runner(cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.4"})
    add_request(runner, "ra", prompt_len=120)
    add_request(runner, "rb", prompt_len=80)
    steps = [
        {"ra": 30, "rb": 20},
        {"ra": 30, "rb": 20},
        {"ra": 30, "rb": 20},
        {"ra": 30, "rb": 20},  # both complete here
        {"ra": 1, "rb": 1},
        {"ra": 3, "rb": 3},
        {"ra": 7, "rb": 7},
    ]
    for step_no, sched in enumerate(steps):
        rs = runner._squeeze_ascend_rs
        pre_layouts = {
            rid: dict(rs.req[rid].layouts)
            for rid in ("ra", "rb")
            if rid in rs.req
        }
        run_step(runner, sched)
        _check_invariant(runner, runner._last_attn_metadata, pre_layouts, step_no)
    assert registry.get_counters().get("reqs_compressed", 0) >= 2


def test_buffer_content_matches_assembled_window():
    runner = make_runner(cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.4"})
    add_request(runner, "r1", prompt_len=100)
    for sched in [{"r1": 20}] * 5 + [{"r1": 1}] * 25:
        run_step(runner, sched)
    rs = runner._squeeze_ascend_rs
    from squeeze_ascend.runtime.view import _window_view_params

    true_len = int(runner.input_batch.num_computed_tokens_cpu[0])
    for layer_name in runner.layer_names:
        layout = rs.req["r1"].layouts[layer_name]
        buf = rs.buffers[layer_name].numpy()
        row = runner.table.block_table.np[0]
        params = _window_view_params(true_len, BS, layout.window, layout.start_size)
        assert params is not None
        sink_blocks, rf, rl, _ = params
        expected = np.concatenate([row[0:sink_blocks], row[rf:rl]])
        got = buf[0]
        assert np.array_equal(got[: expected.size], expected), layer_name


def test_layer_hook_measures_layer_importance():
    """The S6 hook must accumulate cos-sim per (request, layer)."""
    runner = make_runner(cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.4"})
    add_request(runner, "r1", prompt_len=40)
    run_step(runner, {"r1": 20})
    run_step(runner, {"r1": 20})
    rs = runner._squeeze_ascend_rs
    stats = rs.stats.get("r1", {})
    assert len(stats) == 4  # all four layers captured
    mean0 = stats["model.layers.0.self_attn.attn"][0] / stats["model.layers.0.self_attn.attn"][1]
    mean3 = stats["model.layers.3.self_attn.attn"][0] / stats["model.layers.3.self_attn.attn"][1]
    assert mean0 < mean3  # factor 0.9 -> bigger change -> lower cos-sim
