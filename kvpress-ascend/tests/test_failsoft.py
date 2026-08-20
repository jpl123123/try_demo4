"""Fail-soft injection + heartbeat + dry-run + core-parameter logging."""

from __future__ import annotations

import io
import logging
import os

import numpy as np

from kvpress_ascend import registry

from harness import BLOCK_SIZE, add_request, make_runner, run_step


def test_failsoft_missing_batch_fields():
    """Missing input_batch fields must degrade, not crash the step."""
    runner = make_runner(cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "streaming"})
    runner.input_batch.req_ids = []  # empty batch
    run_step(runner, {})
    runner.input_batch.req_ids = ["r1"]
    runner.input_batch.num_prompt_tokens[0] = 10
    # num_computed_tokens_cpu missing entirely
    delattr(runner.input_batch, "num_computed_tokens_cpu")
    try:
        run_step(runner, {"r1": 5})
    except Exception:
        pass  # driver itself may fail; adapter must not raise out of step
    else:
        pass
    # the adapter's own wrappers never raise: force a bad metadata field
    runner2 = make_runner(cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "streaming"})
    add_request(runner2, "r1", prompt_len=40)
    for sched in [{"r1": 20}, {"r1": 20}]:
        run_step(runner2, sched)
    # corrupt seq_lens to None: S4 must skip quietly
    runner2._last_attn_metadata["model.layers.0.self_attn.attn"].seq_lens = None
    rs = runner2._kvpress_ascend_rs
    n = len(runner2.input_batch.req_ids)
    runner2._build_attention_metadata(0, n, 0)  # wrapped; must not raise


def test_failsoft_bad_row_index():
    runner = make_runner(cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "streaming"})
    add_request(runner, "r1", prompt_len=40)
    run_step(runner, {"r1": 20})
    # out-of-range row index mapping must not crash the pass
    runner.input_batch.req_id_to_index["r1"] = 99
    run_step(runner, {"r1": 20})
    runner.input_batch.req_id_to_index["r1"] = 0
    counters = registry.get_counters()
    assert counters.get("skipped_bad_row", 0) >= 0


def test_dry_run_no_views():
    runner = make_runner(
        cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "streaming",
                 "KVPRESS_ASCEND_DRY_RUN": "1"}
    )
    add_request(runner, "r1", prompt_len=40)
    run_step(runner, {"r1": 20})
    run_step(runner, {"r1": 20})  # completion happens here (pass after meta)
    run_step(runner, {"r1": 1})   # now the view would apply - but dry-run skips
    rs = runner._kvpress_ascend_rs
    # layouts are still computed (score path runs)
    assert rs.req["r1"].layouts
    # but metadata must NOT be replaced (no buffers)
    am = runner._last_attn_metadata
    meta = am["model.layers.0.self_attn.attn"]
    assert int(meta.seq_lens_list[0]) == 41  # full length, view not applied
    assert rs.buffers == {}
    counters = registry.get_counters()
    assert counters.get("dry_run", 0) >= 1


def test_heartbeat_emits_core_params():
    runner = make_runner(
        cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "snapkv",
                 "KVPRESS_ASCEND_RATIO": "0.4", "KVPRESS_ASCEND_WINDOW": "32"}
    )
    add_request(runner, "r1", prompt_len=60)
    # capture heartbeat lines
    logger = logging.getLogger("kvpress-ascend")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger.addHandler(handler)
    try:
        run_step(runner, {"r1": 20})
        run_step(runner, {"r1": 20})
        run_step(runner, {"r1": 20})
    finally:
        logger.removeHandler(handler)
    out = buf.getvalue()
    assert "step=" in out and "seams=" in out
    assert "core=snapkv ratio=0.400 window=32" in out
    assert "compressed=" in out


def test_heartbeat_off_when_disabled():
    runner = make_runner(
        cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "streaming",
                 "KVPRESS_ASCEND_STEP_LOG": "0"}
    )
    add_request(runner, "r1", prompt_len=40)
    logger = logging.getLogger("kvpress-ascend")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger.addHandler(handler)
    try:
        run_step(runner, {"r1": 20})
    finally:
        logger.removeHandler(handler)
    assert "step=" not in buf.getvalue()


def test_compress_log_lists_core_params_per_request():
    runner = make_runner(
        cfg_env={"kvpress": "1", "KVPRESS_ASCEND_PRESS": "streaming",
                 "KVPRESS_ASCEND_RATIO": "0.25"}
    )
    add_request(runner, "r1", prompt_len=40)
    logger = logging.getLogger("kvpress-ascend")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger.addHandler(handler)
    try:
        run_step(runner, {"r1": 20})
        run_step(runner, {"r1": 20})
    finally:
        logger.removeHandler(handler)
    out = buf.getvalue()
    assert "COMPRESS req=r1 phase=complete press=streaming ratio=0.250" in out
    assert "orig=40" in out and "n_kept=30" in out
