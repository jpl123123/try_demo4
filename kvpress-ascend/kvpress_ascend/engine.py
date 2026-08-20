"""kvpress-ascend monkeypatch installation.

Fail-soft contract: every step of the installation is guarded; any error only
logs and the server keeps running unoptimized.  Import-order defuse (G7):
the vllm-ascend ops package has a fragile import chain, so the activation
path first imports a safe entry point.

Strategy vs squeeze-ascend: both adapters rewrite the same seams.  When both
are enabled the first-installed adapter wins and the other defers (passive
observer with DEFERRED heartbeat).  Policies:
    KVPRESS_ASCEND_POLICY=primary   -> this adapter always installs
    KVPRESS_ASCEND_POLICY=defer     -> this adapter always defers
    SQUEEZE_ASCEND_POLICY=primary   -> squeeze always installs -> we defer
    default (auto)                  -> first-installed wins
"""

from __future__ import annotations

import importlib.util
import sys
import os

from kvpress_ascend import registry
from kvpress_ascend.envs import get_config
from kvpress_ascend.log import get_logger
from kvpress_ascend.runtime.capture import make_backend_forward_wrapper
from kvpress_ascend.runtime.pass_engine import make_execute_model_wrapper
from kvpress_ascend.runtime.view import make_build_attn_metadata_wrapper

logger = get_logger()

_INSTALLED = False


def _other_package_active() -> bool:
    """Is squeeze-ascend already applied in THIS process?

    Deliberately does not import the other package (both .pth files run at
    interpreter startup; eager cross-imports would make the ownership order
    racy). Ownership is decided by the KV_ASCEND_OWNER marker.
    """
    mod = sys.modules.get("squeeze_ascend")
    return bool(mod is not None and getattr(mod, "_APPLIED", False))


def _strategy_decision() -> tuple[bool, str]:
    """Return (install, reason).

    Ownership is decided by a shared process-global marker (env var) so that
    the simultaneous .pth imports at interpreter startup are deterministic:
    the first adapter that installs sets ``KV_ASCEND_OWNER``; the other defers.
    """
    cfg = get_config()
    my_policy = cfg.policy
    other_active = _other_package_active()
    owner = os.environ.get("KV_ASCEND_OWNER", "")
    if my_policy == "defer":
        return False, "KVPRESS_ASCEND_POLICY=defer"
    if my_policy == "primary":
        return True, ""
    if os.environ.get("SQUEEZE_ASCEND_POLICY", "").lower() == "primary":
        return False, "SQUEEZE_ASCEND_POLICY=primary (other adapter owns the seams)"
    if owner and owner != "kvpress":
        return False, "owner=%s installed first (auto policy: first-installed wins)" % owner
    if other_active:
        return False, "squeeze-ascend installed first (auto policy: first-installed wins)"
    return True, ""


def _defuse_import_order() -> None:
    """G7: pre-import the fragile vllm-ascend ops entry point safely."""
    try:
        import vllm_ascend.ops.fused_moe.fused_moe  # noqa: F401
    except Exception:
        logger.debug("import defuse skipped (vllm_ascend ops not importable here)")


def install() -> bool:
    """Install all seams. Returns True on success (or clean deferral)."""
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
        import vllm_ascend.attention.attention_v1 as attn_mod
        import vllm_ascend.worker.model_runner_v1 as mr_mod
    except Exception:
        logger.warning("vllm_ascend not importable - kvpress-ascend inactive", exc_info=True)
        registry.set_deferred("vllm_ascend not importable")
        return True

    failed = []

    # ---- S1: backend forward capture (both impl variants) ---------------
    for cls_name, seam in (
        ("AscendAttentionBackendImpl", "backend_forward"),
        ("AscendC8AttentionBackendImpl", "backend_forward_c8"),
    ):
        try:
            cls = getattr(attn_mod, cls_name, None)
            if cls is None:
                continue
            orig = cls.forward
            if getattr(orig, "_kvpress_ascend_patched", False):
                registry.mark_installed(seam)
                continue
            cls.forward = make_backend_forward_wrapper(orig)
            cls.forward._kvpress_ascend_patched = True  # type: ignore[attr-defined]
            registry.mark_installed(seam)
        except Exception:
            failed.append(cls_name)

    # ---- S5: execute_model (context + pass + heartbeat) -----------------
    try:
        runner_cls = mr_mod.NPUModelRunner
        orig = runner_cls.execute_model
        if not getattr(orig, "_kvpress_ascend_patched", False):
            runner_cls.execute_model = make_execute_model_wrapper(orig)  # type: ignore[assignment]
            runner_cls.execute_model._kvpress_ascend_patched = True  # type: ignore[attr-defined]
        registry.mark_installed("execute_model")
    except Exception:
        failed.append("execute_model")

    # ---- S4: attention metadata (view rows) ------------------------------
    try:
        runner_cls = mr_mod.NPUModelRunner
        orig = runner_cls._build_attention_metadata
        if not getattr(orig, "_kvpress_ascend_patched", False):
            runner_cls._build_attention_metadata = make_build_attn_metadata_wrapper(orig)  # type: ignore[assignment]
            runner_cls._build_attention_metadata._kvpress_ascend_patched = True  # type: ignore[attr-defined]
        registry.mark_installed("build_attn_metadata")
    except Exception:
        failed.append("build_attn_metadata")

    os.environ.setdefault("KV_ASCEND_OWNER", "kvpress")
    if failed:
        logger.error(
            "kvpress-ascend installed with FAILED seams: %s",
            ",".join(failed),
        )
    logger.info("%s", registry.summary())
    _INSTALLED = True
    return True


def uninstall() -> None:
    """Restore original methods (used by tests)."""
    global _INSTALLED
    try:
        import vllm_ascend.attention.attention_v1 as attn_mod
        import vllm_ascend.worker.model_runner_v1 as mr_mod
    except Exception:
        _INSTALLED = False
        return
    for cls_name in ("AscendAttentionBackendImpl", "AscendC8AttentionBackendImpl"):
        cls = getattr(attn_mod, cls_name, None)
        if cls is None:
            continue
        wrapped = getattr(cls, "forward", None)
        if wrapped is not None and getattr(wrapped, "_kvpress_ascend_patched", False):
            cls.forward = wrapped.__wrapped__ if hasattr(wrapped, "__wrapped__") else cls.forward
    runner_cls = mr_mod.NPUModelRunner
    for method in ("execute_model", "_build_attention_metadata"):
        wrapped = getattr(runner_cls, method, None)
        if wrapped is not None and getattr(wrapped, "_kvpress_ascend_patched", False):
            setattr(runner_cls, method, wrapped.__wrapped__)
    registry.reset()
    _INSTALLED = False
