"""Tiny self-contained test runner (no pytest dependency).

Usage:  python tests/run_tests.py [pattern]
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def discover():
    return sorted(
        n for n in os.listdir(HERE) if n.startswith("test_") and n.endswith(".py")
    )


def run_file(path: str):
    mod_name = "t_" + os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = []
    for name in sorted(dir(mod)):
        fn = getattr(mod, name)
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                results.append((name, True, "", time.time() - t0))
            except Exception as exc:  # noqa: BLE001
                results.append((name, False, traceback.format_exc(), time.time() - t0))
    return results


def main(argv):
    pattern = argv[1] if len(argv) > 1 else ""
    files = [f for f in discover() if pattern in f]
    total_pass = total_fail = 0
    failed = []
    for f in files:
        for name, ok, tb, dt in run_file(os.path.join(HERE, f)):
            if ok:
                total_pass += 1
                print("PASS %-28s %6.0fms" % (f + "::" + name, dt * 1000))
            else:
                total_fail += 1
                failed.append((f, name, tb))
                print("FAIL %-28s %6.0fms" % (f + "::" + name, dt * 1000))
    print("-" * 60)
    print("passed=%d failed=%d" % (total_pass, total_fail))
    for f, name, tb in failed:
        print("=" * 60)
        print("FAILURE: %s::%s" % (f, name))
        print(tb)
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
