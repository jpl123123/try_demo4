"""Self-check CLI: run the offline simulated debugging flow for squeeze-ascend.

Usage:
    python -m squeeze_ascend.simulate          # quick L2 scenario + invariant
    python -m squeeze_ascend.simulate --suite  # run the full test suite

Runs without NPU hardware (CPU torch): proves the layer-hook seam, the 2D
budget clustering and the window-view invariant across prefill + decode.
"""

from __future__ import annotations

import os
import sys


def _quiet_env() -> None:
    os.environ.setdefault("squeeze", "1")
    os.environ["SQUEEZE_ASCEND_STEP_LOG"] = "0"


def run_scenario() -> None:
    from squeeze_ascend import registry

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from tests.harness import (
        BLOCK_SIZE as BS,
        add_request,
        make_runner,
        reference_window_slots,
        run_step,
        view_slots,
    )

    runner = make_runner(
        cfg_env={
            "squeeze": "1",
            "SQUEEZE_ASCEND_INI_SIZE": "0.21",
            "SQUEEZE_ASCEND_KV_CLASS3": "0.08",  # squeeze mode (2D budgets)
            "SQUEEZE_ASCEND_START_SIZE": "4",
            "SQUEEZE_ASCEND_MID_PREFILL_BUDGET": "64",
            "SQUEEZE_ASCEND_STEP_LOG": "0",
        }
    )
    add_request(runner, "sim", prompt_len=160)
    steps = [{"sim": 40}] * 4 + [{"sim": 1}] * 40
    checked = 0
    for step_no, sched in enumerate(steps):
        rs = runner._squeeze_ascend_rs
        pre = dict(rs.req["sim"].layouts) if "sim" in rs.req else {}
        run_step(runner, sched)
        am = runner._last_attn_metadata
        for layer_name in runner.layer_names:
            layout = pre.get(layer_name)
            if layout is None:
                continue
            true_len = int(runner.input_batch.num_computed_tokens_cpu[0])
            true_row = runner.table.block_table.np[0]
            vs = view_slots(am[layer_name], 0)
            ref = reference_window_slots(true_row, true_len, BS,
                                         layout.window, layout.start_size)
            assert vs.shape == ref.shape, (step_no, layer_name)
            assert set(vs.tolist()) == set(ref.tolist()), (step_no, layer_name)
            checked += 1
    counters = registry.get_counters()
    print("[squeeze-ascend] SIMULATE OK")
    print("  steps=%d invariant_checks=%d" % (len(steps), checked))
    print("  reqs_compressed=%d clustered=%d mid_anchors=%d"
          % (counters.get("reqs_compressed", 0),
             counters.get("reqs_clustered", 0),
             counters.get("anchors_mid", 0)))
    print("  counters=%s" % dict(counters))
    if not (counters.get("reqs_compressed", 0) >= 1
            and counters.get("reqs_clustered", 0) >= 1):
        raise SystemExit("SIMULATE FAILED: expected clustering/compression missing")


def main(argv) -> int:
    _quiet_env()
    if "--suite" in argv:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(here, "tests"))
        import run_tests

        return run_tests.main([sys.argv[0]] + [a for a in argv if a != "--suite"])
    run_scenario()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
