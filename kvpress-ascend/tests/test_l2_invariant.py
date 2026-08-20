"""L2: end-to-end visible-set invariant across chunked prefill + decode.

The invariant (framework section 4.5):

    slots the attention would read through the view metadata
        ==  reference visible set (kept tokens + ALL new tokens)

is checked at every step, across block boundaries, for multiple requests,
with mid-prefill anchors and decode re-anchoring enabled.
"""

from __future__ import annotations

import numpy as np

from kvpress_ascend import registry

from harness import (
    BLOCK_SIZE as BS,
    NUM_LAYERS,
    add_request,
    make_runner,
    reference_slots,
    run_step,
    view_slots,
)


def _layer_layouts(runner, req_id):
    rs = runner._kvpress_ascend_rs
    return rs.req[req_id].layouts


def test_single_request_invariant():
    runner = make_runner(
        cfg_env={
            "kvpress": "1",
            "KVPRESS_ASCEND_PRESS": "streaming",
            "KVPRESS_ASCEND_MID_PREFILL_BUDGET": "40",   # anchors at 40/80
            "KVPRESS_ASCEND_DECODE_REANCHOR_WINDOW": "8",  # decode re-anchors
        }
    )
    add_request(runner, "r1", prompt_len=100)
    table = runner.table
    true_row = table.block_table.np[0]

    # chunked prefill: 20 tokens per step -> completes at step 5
    steps = [{"r1": 20}] * 5 + [{"r1": 1}] * 30 + [{"r1": 3}] * 10
    for step_no, sched in enumerate(steps):
        # Snapshot the layouts ACTIVE at metadata-build time (before this
        # step's anchor pass runs - S4 precedes S5 by design).  The metadata
        # built this step reflects the pre-step layouts.
        rs = runner._kvpress_ascend_rs
        pre_layouts = dict(rs.req["r1"].layouts) if "r1" in rs.req else {}
        run_step(runner, sched)
        am = runner._last_attn_metadata
        for layer_name in runner.layer_names:
            layout = pre_layouts.get(layer_name)
            if layout is None:
                continue  # not compressed at metadata-build time: full row
            m = am[layer_name]
            # true length at metadata-build time == num_computed after this
            # step (seq_lens_list holds the VIEW length once views apply)
            true_len = int(runner.input_batch.num_computed_tokens_cpu[0])
            vs = view_slots(m, 0)
            ref = reference_slots(layout.kept, true_row, layout.orig_len, true_len, BS)
            # Attention reads keys as a SET (each slot exactly once), so the
            # order inside the view row is irrelevant; what must match is the
            # slot set and the read length (no unwritten padding is ever read).
            assert vs.shape == ref.shape, (step_no, layer_name, vs.shape, ref.shape)
            assert set(vs.tolist()) == set(ref.tolist()), (
                "visible-set mismatch",
                step_no, layer_name, sorted(vs.tolist()), sorted(ref.tolist()),
            )
            # latest token always visible
            assert int(true_row[(true_len - 1) // BS]) * BS + (true_len - 1) % BS in vs
    counters = registry.get_counters()
    assert counters.get("reqs_compressed", 0) >= 1
    assert counters.get("anchors_mid", 0) >= 2  # anchors at 40 and 80
    assert counters.get("anchors_decode", 0) >= 1
    # view rows physically match the assembled view
    rs = runner._kvpress_ascend_rs
    assert rs.req["r1"].layouts  # layouts exist


def test_multi_request_invariant():
    runner = make_runner(cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "random"})
    add_request(runner, "ra", prompt_len=80)
    add_request(runner, "rb", prompt_len=40)
    table = runner.table

    steps = [
        {"ra": 20, "rb": 20},
        {"ra": 20, "rb": 20},
        {"ra": 20, "rb": 1},  # ra still prefilling, rb decoding
        {"ra": 20, "rb": 1},  # ra completes
        {"ra": 1, "rb": 1},  # both decode
        {"ra": 3, "rb": 3},
        {"ra": 7, "rb": 7},
    ]
    for step_no, sched in enumerate(steps):
        rs = runner._kvpress_ascend_rs
        pre_layouts = {
            rid: dict(rs.req[rid].layouts) for rid in ("ra", "rb") if rid in rs.req
        }
        run_step(runner, sched)
        am = runner._last_attn_metadata
        for i, rid in enumerate(["ra", "rb"]):
            true_row = table.block_table.np[i]
            for layer_name in runner.layer_names:
                layout = pre_layouts.get(rid, {}).get(layer_name)
                if layout is None:
                    continue
                true_len = int(runner.input_batch.num_computed_tokens_cpu[i])
                vs = view_slots(am[layer_name], i)
                ref = reference_slots(layout.kept, true_row, layout.orig_len, true_len, BS)
                assert vs.shape == ref.shape, (step_no, rid, layer_name)
                assert set(vs.tolist()) == set(ref.tolist()), (step_no, rid, layer_name)
    counters = registry.get_counters()
    assert counters.get("reqs_compressed", 0) >= 2


def test_isolated_buffer_content():
    """Buffer rows must equal the assembled view rows at every step."""
    runner = make_runner(cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "streaming"})
    add_request(runner, "r1", prompt_len=100)
    table = runner.table
    for sched in [{"r1": 20}] * 5 + [{"r1": 1}] * 25:
        run_step(runner, sched)
    rs = runner._kvpress_ascend_rs
    for layer_name in runner.layer_names:
        layout = rs.req["r1"].layouts[layer_name]
        buf = rs.buffers[layer_name].numpy()
        m = layout.m
        valid = int(table.num_blocks_per_row[0])
        row = table.block_table.np[0]
        width = buf.shape[1]
        expected = np.concatenate([layout.kept, row[m:valid]])
        expected = np.pad(expected, (0, max(0, width - expected.size)))[:width]
        got = buf[0]
        assert np.array_equal(got[: expected.size], expected[: expected.size]), layer_name
        assert got[expected.size :].sum() == 0, layer_name


def test_add_row_detection_resync():
    """Simulate a prefix-cache add_row: row content changes -> full resync."""
    runner = make_runner(cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "streaming"})
    add_request(runner, "r1", prompt_len=100)
    for sched in [{"r1": 20}] * 5:
        run_step(runner, sched)
    # emulate add_row: insert 1 block at the front of the row (shift)
    row = runner.table.block_table.np[0]
    runner.table.block_table.np[0, 1:] = row[:-1]
    runner.table.block_table.np[0, 0] = 999  # fresh cached block
    run_step(runner, {"r1": 1})
    rs = runner._kvpress_ascend_rs
    for layer_name in runner.layer_names:
        marker = rs.buf_markers[("r1", layer_name)]
        assert marker.first == 999  # resynced to the new first block
