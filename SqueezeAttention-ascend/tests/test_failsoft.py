"""Fail-soft injection + heartbeat + dry-run + core-parameter logging
(squeeze-ascend)."""

from __future__ import annotations

import io
import logging

from squeeze_ascend import registry

from harness import add_request, make_runner, run_step


def test_dry_run_skips_views():
    runner = make_runner(
        cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.4",
                 "SQUEEZE_ASCEND_DRY_RUN": "1"}
    )
    add_request(runner, "r1", prompt_len=100)
    run_step(runner, {"r1": 20})
    run_step(runner, {"r1": 20})
    run_step(runner, {"r1": 20})
    run_step(runner, {"r1": 20})
    run_step(runner, {"r1": 20})  # completes here (pass after meta)
    run_step(runner, {"r1": 1})   # view would apply - dry-run skips
    rs = runner._squeeze_ascend_rs
    # dry-run: budgets are never applied (no layouts, no buffers)
    assert rs.req["r1"].layouts == {}
    assert rs.buffers == {}
    am = runner._last_attn_metadata
    meta = am["model.layers.0.self_attn.attn"]
    assert int(meta.seq_lens_list[0]) == 101  # full length
    assert registry.get_counters().get("dry_run", 0) >= 1


def test_heartbeat_emits_core_params():
    runner = make_runner(
        cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.3",
                 "SQUEEZE_ASCEND_KV_CLASS3": "0.1", "SQUEEZE_ASCEND_START_SIZE": "8"}
    )
    add_request(runner, "r1", prompt_len=100)
    logger = logging.getLogger("squeeze-ascend")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger.addHandler(handler)
    try:
        run_step(runner, {"r1": 20})
    finally:
        logger.removeHandler(handler)
    out = buf.getvalue()
    assert "step=" in out and "seams=" in out
    assert "core=squeeze ini=0.300 start=8" in out


def test_compress_log_lists_core_params_per_request():
    runner = make_runner(
        cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.4",
                 "SQUEEZE_ASCEND_KV_CLASS3": "0.4"}  # uniform mode
    )
    add_request(runner, "r1", prompt_len=100)
    logger = logging.getLogger("squeeze-ascend")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger.addHandler(handler)
    try:
        for _ in range(5):
            run_step(runner, {"r1": 20})
    finally:
        logger.removeHandler(handler)
    out = buf.getvalue()
    assert "COMPRESS req=r1 phase=complete mode=uniform ini_size=0.400" in out
    assert "prompt=100" in out and "windows_max=40" in out


def test_cluster_log_lists_class_composition():
    runner = make_runner(
        cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.21",
                 "SQUEEZE_ASCEND_KV_CLASS3": "0.08"}
    )
    add_request(runner, "r1", prompt_len=100)
    logger = logging.getLogger("squeeze-ascend")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger.addHandler(handler)
    try:
        for _ in range(5):
            run_step(runner, {"r1": 20})
    finally:
        logger.removeHandler(handler)
    out = buf.getvalue()
    assert "CLUSTER req=r1 mode=squeeze" in out
    assert "class3_layers=" in out and "budgets_min=" in out


def test_failsoft_corrupt_metadata():
    """Corrupted metadata fields must be skipped quietly, not crash."""
    runner = make_runner(cfg_env={"squeeze": "1", "SQUEEZE_ASCEND_INI_SIZE": "0.4"})
    add_request(runner, "r1", prompt_len=100)
    for _ in range(5):
        run_step(runner, {"r1": 20})
    runner._last_attn_metadata["model.layers.0.self_attn.attn"].seq_lens = None
    n = len(runner.input_batch.req_ids)
    runner._build_attention_metadata(0, n, 0)  # wrapped; must not raise
    rs = runner._squeeze_ascend_rs
    assert rs.req["r1"].layouts  # state untouched
