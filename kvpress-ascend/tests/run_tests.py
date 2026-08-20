"""Tiny self-contained test runner (no pytest dependency).

Usage:  python tests/run_tests.py [pattern]
Discovers tests/test_*.py, runs every `test_*` function, reports pass/fail
with a short traceback.  Exit code = number of failures.
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
    tests = []
    for name in sorted(os.listdir(HERE)):
        if name.startswith("test_") and name.endswith(".py"):
            tests.append(name)
    return tests


def run_file(path: str):
    mod_name = "t_" + os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    funcs = [
        (n, getattr(mod, n))
        for n in sorted(dir(mod))
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
    results = []
    for name, fn in funcs:
        t0 = time.time()
        try:
            fn()
            results.append((name, True, "", time.time() - t0))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            results.append((name, False, tb, time.time() - t0))
    return results


def main(argv):
    pattern = argv[1] if len(argv) > 1 else ""
    files = [f for f in discover() if pattern in f]
    total_pass = total_fail = 0
    failed = []
    for f in files:
        results = run_file(os.path.join(HERE, f))
        for name, ok, tb, dt in results:
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
