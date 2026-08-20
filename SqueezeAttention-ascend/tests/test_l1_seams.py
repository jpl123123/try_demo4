"""L1: seam installation and contracts for squeeze-ascend (fake vllm_ascend)."""

from __future__ import annotations

import os
import sys
import types

from squeeze_ascend import registry


def _make_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _inject_fake_vllm_ascend():
    base = _make_module("vllm_ascend")
    worker_pkg = _make_module("vllm_ascend.worker")
    mr = _make_module("vllm_ascend.worker.model_runner_v1")

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
    worker_pkg.__path__ = []


def _cleanup_fake_vllm_ascend():
    for name in list(sys.modules):
        if name.startswith("vllm_ascend"):
            sys.modules.pop(name, None)


def _inject_fake_kvpress(policy: str = "primary"):
    pkg = _make_module("kvpress_ascend")
    envs_mod = _make_module("kvpress_ascend.envs")
    _cfg = types.SimpleNamespace(policy=policy)
    envs_mod.get_config = lambda: _cfg
    pkg._APPLIED = False
    pkg.envs = envs_mod


def _cleanup_fake_kvpress():
    sys.modules.pop("kvpress_ascend", None)
    sys.modules.pop("kvpress_ascend.envs", None)


def _reset_engine():
    from squeeze_ascend import engine

    try:
        engine.uninstall()
    except Exception:
        pass
    engine._INSTALLED = False
    registry.reset()


def test_engine_install_all_seams():
    os.environ["squeeze"] = "1"
    os.environ.pop("SQUEEZE_ASCEND_POLICY", None)
    os.environ.pop("KVPRESS_ASCEND_POLICY", None)
    from squeeze_ascend.envs import reset_config

    reset_config()
    _reset_engine()
    _inject_fake_vllm_ascend()
    try:
        from squeeze_ascend import engine

        assert engine.install() is True
        installed = registry.get_installed()
        # layer_hook is installed lazily at the first inference step (the
        # model only exists after load_model); the two runner seams are eager.
        assert {"execute_model", "build_attn_metadata"} <= installed, installed
        assert not registry.is_deferred()
        assert "OK" in registry.summary()
        mr_mod = sys.modules["vllm_ascend.worker.model_runner_v1"]
        inst = mr_mod.NPUModelRunner()
        inst.execute_model(None)
        inst._build_attention_metadata(0, 0, 0)
        engine.uninstall()
        assert registry.get_installed() == set()
    finally:
        _cleanup_fake_vllm_ascend()


def test_engine_defers_when_kvpress_primary():
    os.environ["squeeze"] = "1"
    os.environ["KVPRESS_ASCEND_POLICY"] = "primary"
    from squeeze_ascend.envs import reset_config

    reset_config()
    _reset_engine()
    _inject_fake_vllm_ascend()
    _inject_fake_kvpress(policy="primary")
    try:
        from squeeze_ascend import engine

        engine.install()
        assert registry.is_deferred()
        assert "primary" in registry.summary()
        assert registry.get_installed() == set()
    finally:
        os.environ.pop("KVPRESS_ASCEND_POLICY", None)
        _cleanup_fake_kvpress()
        _cleanup_fake_vllm_ascend()


def test_engine_defers_when_kvpress_already_applied():
    os.environ["squeeze"] = "1"
    from squeeze_ascend.envs import reset_config

    reset_config()
    _reset_engine()
    _inject_fake_vllm_ascend()
    _inject_fake_kvpress(policy="auto")
    sys.modules["kvpress_ascend"]._APPLIED = True
    try:
        from squeeze_ascend import engine

        engine.install()
        assert registry.is_deferred()
        assert "first-installed wins" in registry.summary()
    finally:
        _cleanup_fake_kvpress()
        _cleanup_fake_vllm_ascend()


def test_failsoft_missing_vllm_ascend():
    os.environ["squeeze"] = "1"
    os.environ.pop("KVPRESS_ASCEND_POLICY", None)
    from squeeze_ascend.envs import reset_config

    reset_config()
    _reset_engine()
    for m in list(sys.modules):
        if m.startswith("vllm_ascend") or m.startswith("kvpress_ascend"):
            sys.modules.pop(m)
    from squeeze_ascend import engine

    assert engine.install() is True  # fail-soft
    assert registry.is_deferred()
    assert registry.get_installed() == set()


def test_gate_off_zero_imports():
    import subprocess

    env = {
        k: v
        for k, v in os.environ.items()
        if k != "PYTHONPATH" and k.lower() not in ("squeeze", "squeeze_ascend")
        and not k.lower().startswith("squeeze_ascend_")
        and not k.lower().startswith("kvpress_ascend_")
        and k.lower() not in ("kvpress", "kvpress_ascend")
    }
    code = (
        "import sys\n"
        "import squeeze_ascend\n"
        "assert not squeeze_ascend.gate_enabled()\n"
        "assert 'torch' not in sys.modules, 'torch imported with gate off'\n"
        "assert 'vllm' not in sys.modules and 'vllm_ascend' not in sys.modules\n"
        "print('OK')\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=env, cwd=root,
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
