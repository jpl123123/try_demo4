"""SqueezeAttention-ascend: 2D KV-cache management adapted to vllm-ascend
v0.23.0.

Env-gated monkeypatch adapter.  Loading this module (via the
``squeeze_ascend.pth`` file in site-packages) does **nothing** unless the
gate env var is set:

    export squeeze=1            # or squeeze_ascend / SQUEEZE_ASCEND
    vllm serve ...

When the gate is off, this module never imports torch/vllm/vllm_ascend.
When it is on, ``apply()`` installs the runtime monkeypatches (fail-soft).
"""

import os

__version__ = "0.1.0"

_APPLIED = False
_GATE_NAMES = ("squeeze", "squeeze_ascend", "SQUEEZE_ASCEND")
_FALSE_VALUES = {"", "0", "false", "off", "no", "n", "none"}


def gate_enabled() -> bool:
    for name in _GATE_NAMES:
        value = os.environ.get(name)
        if value is not None and value.strip().lower() not in _FALSE_VALUES:
            return True
    return False


def apply() -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    if not gate_enabled():
        return False
    try:
        from squeeze_ascend.engine import install

        ok = install()
        _APPLIED = ok
        return ok
    except Exception:  # pragma: no cover - defensive; never break the server
        import traceback

        traceback.print_exc()
        return False


if gate_enabled():
    apply()
