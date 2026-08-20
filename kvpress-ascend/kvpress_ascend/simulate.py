"""Self-check CLI: run the offline simulated debugging flow for kvpress-ascend.

Usage:
    python -m kvpress_ascend.simulate          # quick L2 scenario + invariant
    python -m kvpress_ascend.simulate --suite  # run the full test suite

Runs without NPU hardware (CPU torch), proving the seams enter the core code
and the visible-set invariant holds across chunked prefill + decode.
"""

from __future__ import annotations

import os
import sys


def _quiet_env() -> None:
    os.environ.setdefault("kvpress", "1")
    os.environ["KVPRESS_ASCEND_STEP_LOG"] = "0"


def run_scenario() -> None:
    """L2 scenario: mid anchors + completion + decode re-anchor."""
    from kvpress_ascend import registry

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from tests.harness import add_request, make_runner, run_step

    runner = make_runner(
        cfg_env={
            "kvpress": "1",
            "KVPRESS_ASCEND_PRESS": "snapkv",
            "KVPRESS_ASCEND_RATIO": "0.5",
            "KVPRESS_ASCEND_MID_PREFILL_BUDGET": "64",
            "KVPRESS_ASCEND_DECODE_REANCHOR_WINDOW": "16",
            "KVPRESS_ASCEND_STEP_LOG": "0",
        }
    )
    add_request(runner, "sim", prompt_len=160)
    steps = [{"sim": 40}] * 4 + [{"sim": 1}] * 40
    checked = 0
    for step_no, sched in enumerate(steps):
        rs = runner._kvpress_ascend_rs
        pre = dict(rs.req["sim"].layouts) if "sim" in rs.req else {}
        run_step(runner, sched)
        am = runner._last_attn_metadata
        for layer_name in runner.layer_names:
            layout = pre.get(layer_name)
            if layout is None:
                continue
            true_len = int(runner.input_batch.num_computed_tokens_cpu[0])
            true_row = runner.table.block_table.np[0]
            vs = _view_slots(am[layer_name], 0)
            ref = _reference(layout.kept, true_row, layout.orig_len, true_len)
            assert vs.shape == ref.shape, (step_no, layer_name)
            assert set(vs.tolist()) == set(ref.tolist()), (step_no, layer_name)
            checked += 1
    counters = registry.get_counters()
    print("[kvpress-ascend] SIMULATE OK")
    print("  steps=%d invariant_checks=%d" % (len(steps), checked))
    print("  reqs_compressed=%d mid_anchors=%d decode_reanchors=%d"
          % (counters.get("reqs_compressed", 0),
             counters.get("anchors_mid", 0),
             counters.get("anchors_decode", 0)))
    print("  seams_declared=%d counters=%s" % (len(registry.SEAMS), dict(counters)))
    if not (counters.get("reqs_compressed", 0) >= 1
            and counters.get("anchors_mid", 0) >= 1
            and counters.get("anchors_decode", 0) >= 1):
        raise SystemExit("SIMULATE FAILED: expected compression events missing")


def _view_slots(meta, i):
    import numpy as np

    row = meta.block_tables[i].numpy()
    vlen = int(meta.seq_lens_list[i])
    bs = 16
    return (np.repeat(row, bs)[:vlen] * bs + np.arange(vlen) % bs).astype(np.int64)


def _reference(kept, row, orig_len, true_len):
    import numpy as np

    from kvpress_ascend import core

    slots = []
    for b in sorted(int(x) for x in kept):
        n = min(16, max(0, orig_len - b * 16))
        slots.extend([int(row[b]) * 16 + j for j in range(n)])
    for p in range(orig_len, true_len):
        slots.append(int(row[p // 16]) * 16 + p % 16)
    return np.array(slots, dtype=np.int64)


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
