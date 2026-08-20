"""Compose mode: kvpress-ascend + squeeze-ascend run TOGETHER.

Division of labor:
  * squeeze-ascend: S6 cos-sim capture + clustering pass -> per-layer budgets
    (its window-view S4 application is deferred in compose mode);
  * kvpress-ascend: S1 scoring capture (or fallback), S5 compression pass
    consuming squeeze's per-layer budgets (n_kept = budget), S4 view rows.

The two execute_model wrappers nest: the one installed LAST is outermost, so
its finally (compression pass) runs after the inner one.  Here squeeze's
harness installs its wrappers first, then kvpress's wrappers -> kvpress's pass
sees the budgets computed by squeeze's pass in the SAME step.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KVP = os.path.dirname(ROOT)  # try_4/
for p in (os.path.join(KVP, "kvpress-ascend"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from squeeze_ascend import registry as sreg

from harness import BLOCK_SIZE as BS, add_request, make_runner, run_step


# kvpress wrappers are installed on a DEDICATED runner subclass so the shared
# FakeRunner class of the other squeeze tests stays untouched (test isolation).
ComposeRunner = type("ComposeRunner", (__import__("harness").FakeRunner,), {})


def _install_kvpress_wrappers():
    """Install kvpress's real wrappers on the compose runner class."""
    from kvpress_ascend.runtime.context import ensure_runner_state
    from kvpress_ascend.runtime.pass_engine import make_execute_model_wrapper
    from kvpress_ascend.runtime.view import make_build_attn_metadata_wrapper

    if not getattr(ComposeRunner.execute_model, "_kvpress_compose_patched", False):
        ComposeRunner.execute_model = make_execute_model_wrapper(ComposeRunner.execute_model)  # type: ignore[assignment]
        ComposeRunner.execute_model._kvpress_compose_patched = True  # type: ignore[attr-defined]
    if not getattr(ComposeRunner._build_attention_metadata, "_kvpress_compose_patched", False):
        ComposeRunner._build_attention_metadata = make_build_attn_metadata_wrapper(  # type: ignore[assignment]
            ComposeRunner._build_attention_metadata
        )
        ComposeRunner._build_attention_metadata._kvpress_compose_patched = True  # type: ignore[attr-defined]
    return ensure_runner_state


def _kvpress_counters():
    from kvpress_ascend import registry as kreg

    return kreg.get_counters()


_COMPOSE_ENV = {
    "squeeze": "1",
    "kvpress": "1",
    "SQUEEZE_ASCEND_POLICY": "compose",
    "KVPRESS_ASCEND_POLICY": "compose",
    "KVPRESS_ASCEND_PRESS": "streaming",  # no Q/keys needed
    "SQUEEZE_ASCEND_INI_SIZE": "0.21",
    "SQUEEZE_ASCEND_KV_CLASS3": "0.08",   # squeeze mode (2D budgets)
    "SQUEEZE_ASCEND_START_SIZE": "4",
}


def test_compose_budgets_drive_kvpress_compression():
    runner = make_runner(cfg_env=dict(_COMPOSE_ENV), runner_cls=ComposeRunner)
    ensure_kv_rs = _install_kvpress_wrappers()
    ensure_kv_rs(runner)  # create kvpress runner state up front
    add_request(runner, "r1", prompt_len=160)
    # 40-token prefill chunks: completion at step 4
    for sched in [{"r1": 40}] * 4 + [{"r1": 1}] * 20:
        run_step(runner, sched)

    kc = _kvpress_counters()
    sc = sreg.get_counters()
    assert sc.get("reqs_clustered", 0) >= 1, "squeeze clustering ran"
    assert kc.get("reqs_compressed", 0) >= 1, "kvpress compressed at completion"
    assert kc.get("compose_budget_used", 0) >= 4, "kvpress used squeeze budgets"
    assert kc.get("compose_wait_budget", 0) >= 0
    assert sc.get("compose_deferred_views", 0) >= 1, "squeeze S4 deferred"

    rs_kv = runner._kvpress_ascend_rs
    rs_sq = runner._squeeze_ascend_rs
    assert rs_kv.req["r1"].layouts, "kvpress layouts installed"
    assert rs_kv.buffers, "kvpress view rows materialized"
    assert rs_sq.buffers == {}, "squeeze must NOT materialize window views"

    # per-layer kept tokens must track squeeze's per-layer budget (within the
    # block-granularity slack: kept_tokens >= budget and <= budget + bs)
    for layer_name in runner.layer_names:
        kv_layout = rs_kv.req["r1"].layouts[layer_name]
        sq_layout = rs_sq.req["r1"].layouts[layer_name]
        budget = int(sq_layout.window)
        kept = kv_layout.kept_tokens
        assert budget <= kept <= budget + BS, (layer_name, budget, kept)


def test_compose_invariant_holds():
    runner = make_runner(cfg_env=dict(_COMPOSE_ENV), runner_cls=ComposeRunner)
    ensure_kv_rs = _install_kvpress_wrappers()
    ensure_kv_rs(runner)
    add_request(runner, "r1", prompt_len=160)
    true_row = runner.table.block_table.np[0]
    for step_no, sched in enumerate([{"r1": 40}] * 4 + [{"r1": 1}] * 20):
        rs_kv = runner._kvpress_ascend_rs
        pre = dict(rs_kv.req["r1"].layouts) if "r1" in rs_kv.req else {}
        run_step(runner, sched)
        am = runner._last_attn_metadata
        for layer_name in runner.layer_names:
            layout = pre.get(layer_name)
            if layout is None:
                continue
            true_len = int(runner.input_batch.num_computed_tokens_cpu[0])
            vs = _view_slots(am[layer_name], 0)
            ref = _reference_slots(layout.kept, true_row, layout.orig_len, true_len)
            assert vs.shape == ref.shape, (step_no, layer_name)
            assert set(vs.tolist()) == set(ref.tolist()), (step_no, layer_name)


def _make_sq_stub(budget_by_layer):
    """Minimal squeeze runner-state stub for the defer-decision unit test."""
    import types

    if budget_by_layer is None:
        return types.SimpleNamespace(req={})
    layouts = {
        name: types.SimpleNamespace(window=budget) for name, budget in budget_by_layer.items()
    }
    return types.SimpleNamespace(req={"r1": types.SimpleNamespace(layouts=layouts)})


def test_compose_completion_defer_decision():
    """Real-machine install order: kvpress innermost -> budgets arrive one
    step later -> completion defers (bounded), then proceeds."""
    import types

    from kvpress_ascend.envs import get_config
    from kvpress_ascend.runtime.pass_engine import _compose_completion_deferred

    cfg = get_config()  # KVPRESS_ASCEND_POLICY=compose is set by make_runner

    # squeeze not active -> never defer
    runner = types.SimpleNamespace(_squeeze_ascend_rs=None)
    req = types.SimpleNamespace(compose_defer_count=0)
    assert not _compose_completion_deferred(cfg, runner, "r1", req)

    # squeeze active, budgets not ready -> defer (up to 2 times)
    runner2 = types.SimpleNamespace(_squeeze_ascend_rs=_make_sq_stub(None))
    req2 = types.SimpleNamespace(compose_defer_count=0)
    assert _compose_completion_deferred(cfg, runner2, "r1", req2)
    req2.compose_defer_count = 2
    assert not _compose_completion_deferred(cfg, runner2, "r1", req2)  # bounded

    # budgets ready -> never defer
    runner3 = types.SimpleNamespace(
        _squeeze_ascend_rs=_make_sq_stub({"model.layers.0.self_attn.attn": 40})
    )
    req3 = types.SimpleNamespace(compose_defer_count=0)
    assert not _compose_completion_deferred(cfg, runner3, "r1", req3)


def _view_slots(meta, i):
    import numpy as np

    row = meta.block_tables[i].numpy()
    vlen = int(meta.seq_lens_list[i])
    return (np.repeat(row, BS)[:vlen] * BS + np.arange(vlen) % BS).astype(np.int64)


def _reference_slots(kept, true_row, orig_len, true_len):
    import numpy as np

    slots = []
    for b in sorted(int(x) for x in kept):
        n = min(BS, max(0, orig_len - b * BS))
        slots.extend([int(true_row[b]) * BS + j for j in range(n)])
    for p in range(orig_len, true_len):
        slots.append(int(true_row[p // BS]) * BS + p % BS)
    return np.array(slots, dtype=np.int64)
