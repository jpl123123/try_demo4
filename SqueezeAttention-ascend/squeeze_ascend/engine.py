"""squeeze-ascend monkeypatch installation.

Fail-soft contract: every installation step is guarded; any error only logs
and the server keeps running unoptimized.  Import-order defuse (G7) first.

Strategy vs kvpress-ascend: both adapters rewrite the same seams.  When both
are enabled the first-installed adapter wins and the other defers (passive
observer with DEFERRED heartbeat).  Policies:
    SQUEEZE_ASCEND_POLICY=primary   -> this adapter always installs
    SQUEEZE_ASCEND_POLICY=defer     -> this adapter always defers
    KVPRESS_ASCEND_POLICY=primary   -> kvpress always installs -> we defer
    default (auto)                  -> first-installed wins
"""

from __future__ import annotations

import importlib.util
import sys
import os

from squeeze_ascend import registry
from squeeze_ascend.envs import get_config
from squeeze_ascend.log import get_logger
from squeeze_ascend.runtime.pass_engine import make_execute_model_wrapper
from squeeze_ascend.runtime.view import make_build_attn_metadata_wrapper

logger = get_logger()

_INSTALLED = False


def _other_package_active() -> bool:
    """Is kvpress-ascend already applied in THIS process?

    Deliberately does not import the other package (both .pth files run at
    interpreter startup; eager cross-imports would make the ownership order
    racy). Ownership is decided by the KV_ASCEND_OWNER marker.
    """
    mod = sys.modules.get("kvpress_ascend")
    return bool(mod is not None and getattr(mod, "_APPLIED", False))


def _strategy_decision() -> tuple[bool, str]:
    cfg = get_config()
    my_policy = cfg.policy
    other_active = _other_package_active()
    owner = os.environ.get("KV_ASCEND_OWNER", "")
    if my_policy == "defer":
        return False, "SQUEEZE_ASCEND_POLICY=defer"
    if my_policy == "primary":
        return True, ""
    if os.environ.get("KVPRESS_ASCEND_POLICY", "").lower() == "primary":
        return False, "KVPRESS_ASCEND_POLICY=primary (other adapter owns the seams)"
    if owner and owner != "squeeze":
        return False, "owner=%s installed first (auto policy: first-installed wins)" % owner
    if other_active:
        return False, "kvpress-ascend installed first (auto policy: first-installed wins)"
    return True, ""


def _defuse_import_order() -> None:
    try:
        import vllm_ascend.ops.fused_moe.fused_moe  # noqa: F401
    except Exception:
        logger.debug("import defuse skipped (vllm_ascend ops not importable here)")


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    cfg = get_config()
    if not cfg.enabled:
        registry.set_deferred("gate not enabled")
        return True

    install_ok, reason = _strategy_decision()
    if not install_ok:
        registry.set_deferred(reason)
        logger.info("DEFERRED: %s", reason)
        return True

    _defuse_import_order()

    try:
        import vllm_ascend.worker.model_runner_v1 as mr_mod
    except Exception:
        logger.warning("vllm_ascend not importable - squeeze-ascend inactive", exc_info=True)
        registry.set_deferred("vllm_ascend not importable")
        return True

    failed = []
    try:
        runner_cls = mr_mod.NPUModelRunner
        orig = runner_cls.execute_model
        if not getattr(orig, "_squeeze_ascend_patched", False):
            runner_cls.execute_model = make_execute_model_wrapper(orig)  # type: ignore[assignment]
            runner_cls.execute_model._squeeze_ascend_patched = True  # type: ignore[attr-defined]
        registry.mark_installed("execute_model")
    except Exception:
        failed.append("execute_model")

    try:
        runner_cls = mr_mod.NPUModelRunner
        orig = runner_cls._build_attention_metadata
        if not getattr(orig, "_squeeze_ascend_patched", False):
            runner_cls._build_attention_metadata = make_build_attn_metadata_wrapper(orig)  # type: ignore[assignment]
            runner_cls._build_attention_metadata._squeeze_ascend_patched = True  # type: ignore[attr-defined]
        registry.mark_installed("build_attn_metadata")
    except Exception:
        failed.append("build_attn_metadata")

    # S6 (layer hooks) is installed lazily at the first inference step
    # (the model is only loaded after install time).
    os.environ.setdefault("KV_ASCEND_OWNER", "squeeze")
    if failed:
        logger.error("squeeze-ascend installed with FAILED seams: %s", ",".join(failed))
    logger.info("%s", registry.summary())
    _INSTALLED = True
    return True


def uninstall() -> None:
    global _INSTALLED
    try:
        import vllm_ascend.worker.model_runner_v1 as mr_mod
    except Exception:
        _INSTALLED = False
        return
    runner_cls = mr_mod.NPUModelRunner
    for method in ("execute_model", "_build_attention_metadata"):
        wrapped = getattr(runner_cls, method, None)
        if wrapped is not None and getattr(wrapped, "_squeeze_ascend_patched", False):
            setattr(runner_cls, method, wrapped.__wrapped__)
    registry.reset()
    _INSTALLED = False
