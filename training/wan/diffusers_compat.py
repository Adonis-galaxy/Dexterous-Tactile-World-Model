from __future__ import annotations

import importlib.util
import warnings


_PATCHED = False
_ORIGINAL_FIND_SPEC = None
# NOTE(cross-server fix): "deepspeed" removed from this set. Hiding it from importlib.find_spec
# makes accelerate's is_deepspeed_available() return False, which in accelerate>=1.x drops
# DummyScheduler from accelerate.utils (guarded by `if is_deepspeed_available():`) -> train_dwm_wan.py
# import crashes. deepspeed is REQUIRED here (accelerate launches via deepspeed), so it must stay visible.
# The compat shim only ever disabled torchao/bitsandbytes/xformers in diffusers' import_utils anyway.
_HIDDEN_OPTIONAL_PACKAGES = {"torchao", "bitsandbytes", "xformers"}


def disable_problematic_optional_backends() -> None:
    """Disable unsupported optional diffusers integrations in the validated release env."""

    global _PATCHED, _ORIGINAL_FIND_SPEC
    if _PATCHED:
        return

    if _ORIGINAL_FIND_SPEC is None:
        _ORIGINAL_FIND_SPEC = importlib.util.find_spec

        def _patched_find_spec(name, package=None):
            top_level = name.split(".", 1)[0]
            if top_level in _HIDDEN_OPTIONAL_PACKAGES:
                return None
            return _ORIGINAL_FIND_SPEC(name, package)

        importlib.util.find_spec = _patched_find_spec

    try:
        import diffusers.utils.import_utils as import_utils
    except Exception:
        return

    disabled_any = False

    if getattr(import_utils, "_torchao_available", False):
        import_utils._torchao_available = False
        disabled_any = True

    if getattr(import_utils, "_bitsandbytes_available", False):
        import_utils._bitsandbytes_available = False
        disabled_any = True

    if getattr(import_utils, "_xformers_available", False):
        import_utils._xformers_available = False
        disabled_any = True

    if disabled_any:
        warnings.warn(
            "Disabled unsupported optional acceleration backends for this process to keep the validated "
            "WAN release path stable.",
            RuntimeWarning,
            stacklevel=2,
        )

    _PATCHED = True


__all__ = ["disable_problematic_optional_backends"]
