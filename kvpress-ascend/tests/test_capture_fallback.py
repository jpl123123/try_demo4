"""Capture diagnostics + snapkv-no-window fallback tests.

These cover the real-machine failure mode observed in NPU logs:
  seams=4/4 ... viewed=0 compressed=120 skipped: skipped_all_layers=N skipped_no_q=M
i.e. the scoring pass could never find a captured query window.  The fixes:
  1. capture now reports per-branch cap_* diagnostic counters, and the hit
     counter only counts successful appends;
  2. snapkv degrades to positional streaming scoring when the window is
     missing (KVPRESS_ASCEND_PRESS_FALLBACK=streaming, default), so
     compression keeps working even if Q-capture is blocked on the machine.
"""

from __future__ import annotations

from harness import BLOCK_SIZE as BS, add_request, make_runner, run_step

from kvpress_ascend import registry
from kvpress_ascend.runtime import capture as cap_mod


def test_capture_appends_and_hit_means_appended():
    runner = make_runner(cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "snapkv"})
    add_request(runner, "r1", prompt_len=100)
    run_step(runner, {"r1": 20})
    run_step(runner, {"r1": 20})
    counters = registry.get_counters()
    assert counters.get("cap_appended", 0) >= 2  # per-step per-layer segments
    assert registry.get_hit() >= 2  # hit only fires when appended
    assert counters.get("cap_ctx_none", 0) == 0
    assert counters.get("cap_guarded", 0) == 0


def test_capture_guard_reports_reason():
    runner = make_runner(cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "snapkv"})
    add_request(runner, "r1", prompt_len=100)
    # block capture through the ascend-context guard (as if graph capture /
    # draft forward were active on the machine)
    orig_ok = cap_mod._ascend_ctx_ok
    cap_mod._ascend_ctx_ok = lambda: False
    try:
        run_step(runner, {"r1": 20})
    finally:
        cap_mod._ascend_ctx_ok = orig_ok
    counters = registry.get_counters()
    assert counters.get("cap_guarded", 0) >= 1
    assert counters.get("cap_appended", 0) == 0


def test_capture_len_mismatch_reports_reason():
    runner = make_runner(cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "snapkv"})
    add_request(runner, "r1", prompt_len=100)
    # corrupt the per-request lengths so the TND split is impossible
    orig_meta = runner._build_attention_metadata

    def broken_meta(*args, **kwargs):
        out = orig_meta(*args, **kwargs)
        am = out[0]
        for meta in am.values():
            meta.actual_seq_lengths_q = [99999]
        return out

    runner._build_attention_metadata = broken_meta  # inner, below the wrapper
    run_step(runner, {"r1": 20})
    counters = registry.get_counters()
    assert counters.get("cap_len_mismatch", 0) >= 1


def test_snapkv_falls_back_to_streaming_without_window():
    """Real-machine no_q scenario: capture fully blocked -> compression still
    happens via the positional streaming fallback."""
    runner = make_runner(
        cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "snapkv",
                 "KVPRESS_ASCEND_PRESS_FALLBACK": "streaming"}
    )
    add_request(runner, "r1", prompt_len=100)
    orig_ok = cap_mod._ascend_ctx_ok
    cap_mod._ascend_ctx_ok = lambda: False  # simulate blocked capture
    try:
        for sched in [{"r1": 20}] * 5 + [{"r1": 1}] * 10:
            run_step(runner, sched)
    finally:
        cap_mod._ascend_ctx_ok = orig_ok
    counters = registry.get_counters()
    assert counters.get("fallback_streaming", 0) >= 4  # every layer fell back
    assert counters.get("reqs_compressed", 0) >= 1  # completion still worked
    rs = runner._kvpress_ascend_rs
    assert rs.req["r1"].layouts  # views installed
    assert rs.buffers  # view rows materialized
    # invariant: view slots == reference visible set
    from harness import reference_slots, view_slots

    from kvpress_ascend import core

    true_row = runner.table.block_table.np[0]
    am = runner._last_attn_metadata
    for layer_name in runner.layer_names:
        layout = rs.req["r1"].layouts[layer_name]
        true_len = int(runner.input_batch.num_computed_tokens_cpu[0])
        vs = view_slots(am[layer_name], 0)
        ref = reference_slots(layout.kept, true_row, layout.orig_len, true_len, BS)
        assert vs.shape == ref.shape
        assert set(vs.tolist()) == set(ref.tolist())


def test_no_fallback_when_disabled():
    runner = make_runner(
        cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "snapkv",
                 "KVPRESS_ASCEND_PRESS_FALLBACK": "none"}
    )
    add_request(runner, "r1", prompt_len=100)
    orig_ok = cap_mod._ascend_ctx_ok
    cap_mod._ascend_ctx_ok = lambda: False
    try:
        for sched in [{"r1": 20}] * 5:
            run_step(runner, sched)
    finally:
        cap_mod._ascend_ctx_ok = orig_ok
    counters = registry.get_counters()
    assert counters.get("fallback_streaming", 0) == 0
    assert counters.get("skipped_no_q", 0) >= 4
    assert counters.get("reqs_compressed", 0) == 0
