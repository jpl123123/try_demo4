"""L1: seam installation and contracts against a fake vllm_ascend.

These tests exercise the real adapter wrappers and engine.install() against
field-faithful stand-ins of the vllm-ascend classes, without the NPU stack.
"""

from __future__ import annotations

import os
import sys
import types

from kvpress_ascend import registry


def _make_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _inject_fake_vllm_ascend():
    """Install fake vllm_ascend module hierarchy into sys.modules."""
    base = _make_module("vllm_ascend")
    attn_pkg = _make_module("vllm_ascend.attention")
    attn = _make_module("vllm_ascend.attention.attention_v1")
    worker_pkg = _make_module("vllm_ascend.worker")
    mr = _make_module("vllm_ascend.worker.model_runner_v1")

    class FakeBackend:
        calls = 0

        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, output_scale=None, output_block_scale=None):
            FakeBackend.calls += 1
            return output

    attn.AscendAttentionBackendImpl = FakeBackend
    attn.AscendC8AttentionBackendImpl = FakeBackend

    class FakeRunnerCls:
        def __init__(self):
            self.input_batch = types.SimpleNamespace(
                req_ids=[],
                num_computed_tokens_cpu=__import__("numpy").zeros(0, dtype="int64"),
                num_prompt_tokens=__import__("numpy").zeros(0, dtype="int64"),
                req_id_to_index={},
                block_table=types.SimpleNamespace(),
            )
            self.use_cp = False

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return None

        def _build_attention_metadata(self, *args, **kwargs):
            return {}, None

    mr.NPUModelRunner = FakeRunnerCls
    base.__path__ = []
    attn_pkg.__path__ = []
    worker_pkg.__path__ = []


def _cleanup_fake_vllm_ascend():
    for name in list(sys.modules):
        if name.startswith("vllm_ascend"):
            sys.modules.pop(name, None)


def _inject_fake_squeeze(policy: str = "primary"):
    pkg = _make_module("squeeze_ascend")
    envs_mod = _make_module("squeeze_ascend.envs")
    _cfg = types.SimpleNamespace(policy=policy)
    envs_mod.get_config = lambda: _cfg
    pkg._APPLIED = False
    pkg.envs = envs_mod


def _cleanup_fake_squeeze():
    sys.modules.pop("squeeze_ascend", None)
    sys.modules.pop("squeeze_ascend.envs", None)


def _reset_engine():
    """Reset engine/registry state between tests (module globals persist)."""
    from kvpress_ascend import engine

    try:
        engine.uninstall()
    except Exception:
        pass
    engine._INSTALLED = False
    registry.reset()


def test_engine_install_all_seams():
    os.environ["kvpress"] = "1"
    os.environ.pop("KVPRESS_ASCEND_POLICY", None)
    os.environ.pop("SQUEEZE_ASCEND_POLICY", None)
    from kvpress_ascend.envs import reset_config

    reset_config()
    _reset_engine()
    _inject_fake_vllm_ascend()
    try:
        from kvpress_ascend import engine

        assert engine.install() is True
        installed = registry.get_installed()
        assert installed == set(registry.SEAMS), installed
        assert not registry.is_deferred()
        summary = registry.summary()
        assert "seams=4/4" in summary and "OK" in summary
        # seams are callable: execute a step through the wrapped runner
        mr_mod = sys.modules["vllm_ascend.worker.model_runner_v1"]
        inst = mr_mod.NPUModelRunner()
        inst.execute_model(None)
        inst._build_attention_metadata(0, 0, 0)
        # backend forward capture runs without vllm attn context
        attn_mod = sys.modules["vllm_ascend.attention.attention_v1"]
        b = attn_mod.AscendAttentionBackendImpl()
        import torch

        out = torch.zeros(2, 4)
        b.forward(None, torch.zeros(2, 4, 8), None, None, None, None, out)
        engine.uninstall()
        assert registry.get_installed() == set()
    finally:
        _cleanup_fake_vllm_ascend()


def test_engine_install_idempotent_and_uninstall_restores():
    os.environ["kvpress"] = "1"
    os.environ.pop("KVPRESS_ASCEND_POLICY", None)
    from kvpress_ascend.envs import reset_config

    reset_config()
    _reset_engine()
    _inject_fake_vllm_ascend()
    try:
        from kvpress_ascend import engine

        engine.install()
        engine.install()  # idempotent
        mr_mod = sys.modules["vllm_ascend.worker.model_runner_v1"]
        wrapped = mr_mod.NPUModelRunner.execute_model
        assert getattr(wrapped, "_kvpress_ascend_patched", False)
        engine.uninstall()
        restored = mr_mod.NPUModelRunner.execute_model
        assert not getattr(restored, "_kvpress_ascend_patched", False)
        assert restored.__name__ == "execute_model"
    finally:
        _cleanup_fake_vllm_ascend()


def test_engine_defers_when_squeeze_primary():
    os.environ["kvpress"] = "1"
    os.environ["SQUEEZE_ASCEND_POLICY"] = "primary"
    from kvpress_ascend.envs import reset_config

    reset_config()
    _reset_engine()
    _inject_fake_vllm_ascend()
    _inject_fake_squeeze(policy="primary")
    try:
        from kvpress_ascend import engine

        engine.install()
        assert registry.is_deferred()
        assert "primary" in registry.summary()
        assert registry.get_installed() == set()
    finally:
        os.environ.pop("SQUEEZE_ASCEND_POLICY", None)
        _cleanup_fake_squeeze()
        _cleanup_fake_vllm_ascend()


def test_engine_defers_when_squeeze_already_applied():
    os.environ["kvpress"] = "1"
    from kvpress_ascend.envs import reset_config

    reset_config()
    _reset_engine()
    _inject_fake_vllm_ascend()
    _inject_fake_squeeze(policy="auto")
    sys.modules["squeeze_ascend"]._APPLIED = True
    try:
        from kvpress_ascend import engine

        engine.install()
        assert registry.is_deferred()
        assert "first-installed wins" in registry.summary()
    finally:
        _cleanup_fake_squeeze()
        _cleanup_fake_vllm_ascend()


def test_gate_off_zero_imports():
    """Fresh interpreter: gate off -> import must not load torch/vllm."""
    import subprocess

    env = {
        k: v
        for k, v in os.environ.items()
        if k != "PYTHONPATH" and k.lower() not in ("kvpress", "kvpress_ascend")
        and not k.lower().startswith("kvpress_ascend_")
        and not k.lower().startswith("squeeze_ascend_")
        and k.lower() not in ("squeeze", "squeeze_ascend")
    }
    code = (
        "import sys\n"
        "import kvpress_ascend\n"
        "assert not kvpress_ascend.gate_enabled()\n"
        "assert 'torch' not in sys.modules, 'torch imported with gate off'\n"
        "assert 'vllm' not in sys.modules and 'vllm_ascend' not in sys.modules\n"
        "print('OK')\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env=env,
        cwd=root,
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_failsoft_missing_vllm_ascend():
    os.environ["kvpress"] = "1"
    os.environ.pop("SQUEEZE_ASCEND_POLICY", None)
    from kvpress_ascend.envs import reset_config

    reset_config()
    registry.reset()
    for m in list(sys.modules):
        if m.startswith("vllm_ascend") or m.startswith("squeeze_ascend"):
            sys.modules.pop(m)
    from kvpress_ascend import engine

    assert engine.install() is True  # fail-soft: no exception
    assert registry.is_deferred()
    assert registry.get_installed() == set()
