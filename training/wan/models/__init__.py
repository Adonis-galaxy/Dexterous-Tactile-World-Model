from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AutoTokenizer": ("transformers", "AutoTokenizer"),
    "T5Tokenizer": ("transformers", "T5Tokenizer"),
    "CLIPModel": (".wan_image_encoder", "CLIPModel"),
    "WanT5EncoderModel": (".wan_text_encoder", "WanT5EncoderModel"),
    "Wan2_2Transformer3DModel": (".wan_transformer3d", "Wan2_2Transformer3DModel"),
    "WanSelfAttention": (".wan_transformer3d", "WanSelfAttention"),
    "WanTransformer3DModel": (".wan_transformer3d", "WanTransformer3DModel"),
    "WanTransformer3DModelWithConcat": (
        ".wan_transformer3d_with_conditions",
        "WanTransformer3DModelWithConcat",
    ),
    "AutoencoderKLWan": (".wan_vae", "AutoencoderKLWan"),
    "AutoencoderKLWan_": (".wan_vae", "AutoencoderKLWan_"),
    "AutoencoderKLWan3_8": (".wan_vae3_8", "AutoencoderKLWan3_8"),
    "AutoencoderKLWan2_2_": (".wan_vae3_8", "AutoencoderKLWan2_2_"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__) if module_name.startswith(".") else import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
