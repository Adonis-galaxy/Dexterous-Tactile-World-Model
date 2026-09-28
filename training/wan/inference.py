#!/usr/bin/env python3
"""
WAN-based inference script for DTWM (Wan2.2 hand-concat inpainting pipeline with spatially
grounded tactile injection and teacher-forced autoregressive chunk prediction). LoRA checkpoints.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"   # DTWM: the `dtwm` package lives in <repo>/src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.wan.diffusers_compat import disable_problematic_optional_backends

disable_problematic_optional_backends()

import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from transformers import AutoTokenizer

from training.dataset_layout_paths import build_output_stem
from training.wan.config_loader import load_experiment_config
from training.wan.models import WanTransformer3DModelWithConcat
from training.wan.pipeline.pipeline_wan2_2_fun_inpaint_hand_concat import (
    Wan2_2FunInpaintHandConcatPipeline,
)
from training.wan.utils.utils import (
    filter_kwargs,
    save_videos_grid,
)
from training.wan.utils.lora_utils import merge_lora
from training.wan.models import AutoencoderKLWan, AutoencoderKLWan3_8, WanT5EncoderModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_distributed():
    """Setup distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def load_experiment_config_wrapper(config_path: str, overrides=None) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    return load_experiment_config(config_path, overrides=overrides or [])


def find_experiment_config(checkpoint_path: str) -> Optional[str]:
    """Find experiment config YAML near a checkpoint directory or packaged release folder."""
    checkpoint_dir = Path(checkpoint_path)
    search_dir = checkpoint_dir.parent if checkpoint_dir.name.startswith("checkpoint-") else checkpoint_dir
    
    yaml_files = list(search_dir.glob("*.yaml")) + list(search_dir.glob("*.yml"))
    
    if yaml_files:
        return str(yaml_files[0])
    
    return None


def _normalize_bucket_values(raw_value: Any) -> List[int]:
    """Normalize config bucket values into a sorted positive-int list."""
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        values = raw_value
    else:
        values = [raw_value]

    normalized = []
    for value in values:
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0:
            normalized.append(parsed)
    return sorted(set(normalized))


def _resize_and_pad_video(
    video: Union[np.ndarray, torch.Tensor],
    target_h: int,
    target_w: int,
    pad_value: float = 0.0,
    short_side_mode: bool = True,
    center_pad: bool = True,
    layout_spec: str = "FHWC",
):
    """Resize video preserving aspect ratio, then pad to exact target size."""
    if layout_spec == "FHWC":
        tensor = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
    elif layout_spec == "BCFHW":
        b, c, f, h, w = video.shape
        tensor = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w).contiguous()
    else:
        raise ValueError(f"Unsupported layout_spec for resize/pad: {layout_spec}")

    _, _, src_h, src_w = tensor.shape
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid video shape for resize/pad: H={src_h}, W={src_w}")

    if short_side_mode == "stretch":
        # plain stretch to target (matches the TRAINING dataloader's torchvision resize geometry)
        new_h, new_w = target_h, target_w
    elif short_side_mode:
        src_short = min(src_h, src_w)
        tgt_short = min(target_h, target_w)
        scale = float(tgt_short) / float(src_short)
        new_h = max(1, int(round(src_h * scale)))
        new_w = max(1, int(round(src_w * scale)))
        if new_h > target_h or new_w > target_w:
            scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
            new_h = max(1, int(round(src_h * scale)))
            new_w = max(1, int(round(src_w * scale)))
    else:
        scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
        new_h = max(1, int(round(src_h * scale)))
        new_w = max(1, int(round(src_w * scale)))

    resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_h = max(0, target_h - new_h)
    pad_w = max(0, target_w - new_w)
    if center_pad:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    else:
        pad_top, pad_left = 0, 0
        pad_bottom, pad_right = pad_h, pad_w

    padded = F.pad(
        resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=float(pad_value),
    )

    if layout_spec == "FHWC":
        return padded.permute(0, 2, 3, 1).cpu().numpy().astype(video.dtype, copy=False)

    b, c, f, _, _ = video.shape
    return padded.reshape(b, f, c, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous().to(video.dtype)


# Frame count for the tactile path (tokenizer frame embeddings, pressure sequence). Set from
# --num_frames in main (default 49, as in the paper).
_DTWM_L = 49


def _clip_or_pad_video_fhwc(video_np: np.ndarray, target_num_frames: int) -> np.ndarray:
    """Deterministically trim or tail-pad an FHWC video to a fixed length."""
    if video_np.shape[0] <= 0:
        raise ValueError("Video has no frames")
    if target_num_frames <= 0:
        raise ValueError(f"Invalid target_num_frames={target_num_frames}")

    if video_np.shape[0] > target_num_frames:
        return video_np[:target_num_frames]
    if video_np.shape[0] < target_num_frames:
        pad = np.repeat(video_np[-1:], target_num_frames - video_np.shape[0], axis=0)
        return np.concatenate([video_np, pad], axis=0)
    return video_np


def _clip_or_pad_video_bcfhw(video: torch.Tensor, target_num_frames: int) -> torch.Tensor:
    """Deterministically trim or tail-pad a BCFHW video to a fixed length."""
    if video.shape[2] <= 0:
        raise ValueError("Video has no frames")
    if target_num_frames <= 0:
        raise ValueError(f"Invalid target_num_frames={target_num_frames}")

    if video.shape[2] > target_num_frames:
        return video[:, :, :target_num_frames].contiguous()
    if video.shape[2] < target_num_frames:
        pad = video[:, :, -1:].repeat(1, 1, target_num_frames - video.shape[2], 1, 1)
        return torch.cat([video, pad], dim=2).contiguous()
    return video


def _resolve_inference_num_frames(config: Optional[Dict[str, Any]]) -> Optional[int]:
    """Resolve target num_frames using validation/training-style config fallbacks."""
    if config is None or not isinstance(config, dict):
        return None

    data_config = config.get("data", {})
    custom_settings = config.get("training", {}).get("custom_settings", {})
    raw_target_num_frames = data_config.get("max_num_frames", custom_settings.get("max_num_frames"))
    if raw_target_num_frames is not None:
        try:
            target_num_frames = int(raw_target_num_frames)
            if target_num_frames > 0:
                return target_num_frames
        except Exception:
            pass

    frame_buckets = _normalize_bucket_values(data_config.get("frame_buckets"))
    if len(frame_buckets) == 1:
        return frame_buckets[0]
    return None


def _resolve_inference_spatial_size(
    source_h: int,
    source_w: int,
    config: Optional[Dict[str, Any]],
) -> Optional[tuple[int, int]]:
    """Resolve inference target size using validation-style config fallbacks."""
    if config is None:
        return None

    data_config = config.get("data", {}) if isinstance(config, dict) else {}
    pipeline_type = str(config.get("pipeline", {}).get("type", "")) if isinstance(config, dict) else ""
    min_divisor = 32 if "2.2" in pipeline_type else 8

    validation_runtime_adapt = bool(data_config.get("validation_runtime_adapt", False))
    validation_inference_size = data_config.get("validation_inference_size")
    if (
        validation_runtime_adapt
        and isinstance(validation_inference_size, (list, tuple))
        and len(validation_inference_size) == 2
    ):
        try:
            target_h = int(validation_inference_size[0])
            target_w = int(validation_inference_size[1])
            if target_h > 0 and target_w > 0:
                return target_h, target_w
        except Exception:
            pass

    height_buckets = _normalize_bucket_values(data_config.get("height_buckets"))
    width_buckets = _normalize_bucket_values(data_config.get("width_buckets"))
    if len(height_buckets) == 1 and len(width_buckets) == 1:
        return height_buckets[0], width_buckets[0]

    if height_buckets and width_buckets and (source_h % min_divisor != 0 or source_w % min_divisor != 0):
        return min(
            ((height, width) for height in height_buckets for width in width_buckets),
            key=lambda size: abs(size[0] - source_h) + abs(size[1] - source_w),
        )

    if source_h % min_divisor != 0 or source_w % min_divisor != 0:
        target_h = max(min_divisor, int(round(source_h / min_divisor)) * min_divisor)
        target_w = max(min_divisor, int(round(source_w / min_divisor)) * min_divisor)
        return target_h, target_w

    return None


def build_pipeline(args: argparse.Namespace, config: Dict[str, Any]) -> Wan2_2FunInpaintHandConcatPipeline:
    """Build WAN pipeline from config and checkpoint."""
    
    pipeline_config = config.get("pipeline", {})
    transformer_config = config.get("transformer", {})
    training_config = config.get("training", {})
    
    # Get base model path (--base_model_path takes precedence over the config value)
    base_model_path = args.base_model_path or pipeline_config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError("base_model_name_or_path must be provided in config or via --base_model_path")
    base_model_path = os.path.expanduser(base_model_path)
    
    logger.info(f"🔧 Building WAN pipeline from base model: {base_model_path}")
    
    # Determine weight dtype
    weight_dtype = torch.bfloat16
    
    # Get training mode
    training_mode = training_config.get("mode", "sft")
    logger.info(f"🔧 Training mode: {training_mode}")
    
    # Load components
    logger.info("📦 Loading model components...")

    text_encoder_kwargs = config.get("text_encoder_kwargs", {})
    vae_kwargs = config.get("vae_kwargs", {})

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(base_model_path, text_encoder_kwargs.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=text_encoder_kwargs,
        torch_dtype=weight_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(base_model_path, text_encoder_kwargs.get("tokenizer_subpath", "tokenizer")),
    )
    # WAN 2.2: high-compression 48-ch VAE (AutoencoderKLWan3_8), selected via vae_kwargs (mirrors training)
    _VAECls = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[vae_kwargs.get("vae_type", "AutoencoderKLWan")]
    vae = _VAECls.from_pretrained(
        os.path.join(base_model_path, vae_kwargs.get("vae_subpath", "vae")),
        additional_kwargs=vae_kwargs,
    ).eval()
    
    # Transformer
    transformer_additional_kwargs = config.get("transformer_additional_kwargs", {})
    transformer_subpath = transformer_additional_kwargs.get("transformer_subpath", "transformer")

    transformer_class_spec = transformer_config.get("class", "WanTransformer3DModelWithConcat")
    if transformer_class_spec not in ("WanTransformer3DModelWithConcat", WanTransformer3DModelWithConcat):
        raise ValueError(
            f"DTWM release supports transformer.class=WanTransformer3DModelWithConcat only, got {transformer_class_spec!r}")
    TransformerCls = WanTransformer3DModelWithConcat

    # Model FPS (RoPE scaling). This is distinct from the FPS used when saving mp4 files.
    model_fps = int(getattr(args, "model_fps", 16))
    logger.info(f"🎞️ model_fps (transformer/RoPE): {model_fps}")

    # Extra kwargs for transformer init
    transformer_init_kwargs = {
        **transformer_additional_kwargs,
        "fps": model_fps,
    }
    if "2.2" in pipeline_config.get("type", ""):
        transformer_init_kwargs["is_wan2_2"] = True   # mirrors train_dwm_wan.py transformer loading

    if TransformerCls is WanTransformer3DModelWithConcat:
        condition_channels = transformer_config.get("condition_channels", 16)
        transformer_init_kwargs["condition_channels"] = condition_channels

    if training_mode != "lora":
        raise ValueError(f"DTWM release supports LoRA checkpoints (training.mode=lora) only, got {training_mode!r}")
    # For LoRA, load base transformer
    transformer3d = TransformerCls.from_pretrained(
        os.path.join(base_model_path, transformer_subpath),
        transformer_additional_kwargs=transformer_init_kwargs,
        torch_dtype=weight_dtype,
    )
    
    # Scheduler
    scheduler_kwargs = filter_kwargs(FlowMatchEulerDiscreteScheduler, 
                                    config.get("scheduler_kwargs", {}))   
    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_kwargs)
    
    # Create pipeline (match training script behavior)
    pipeline_type = pipeline_config.get("type", "wan2.1_fun_inp_hand_concat")
    if pipeline_type == "wan2.2_fun_inp_hand_concat":
        logger.info("🔧 Using WAN 2.2 hand concat pipeline for inference (no CLIP; single 5B transformer)")
        pipeline = Wan2_2FunInpaintHandConcatPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer3d,
            transformer_2=None,
            scheduler=scheduler,
        )
    else:
        raise ValueError(f"Unsupported WAN pipeline type for inference: {pipeline_type}")
    pipeline = pipeline.to(device, dtype=weight_dtype)
    
    # Load LoRA weights if in LoRA mode
    # ===== DTWM: attach TactileTokenizer + spatial injector (weights restore from non_lora_weights =====
    # below). Layers MUST match training ([0,6,12,18,24]) so param names align.
    if getattr(args, "tactile_spatial", False):
        from dtwm.models.tactile_tokens import TactileTokenizer, TactileSpatialInjector
        _dim = int(transformer3d.patch_embedding.weight.shape[0])  # robust dim (wan2.2 dict_mapping may hide config.dim)
        if getattr(transformer3d, "tactile_tokenizer", None) is None:
            transformer3d.tactile_tokenizer = TactileTokenizer(dim=_dim, n_frames=_DTWM_L).to(device=device, dtype=torch.float32)
        _layers = [int(x) for x in (getattr(args, "tactile_spatial_layers", None) or [0, 6, 12, 18, 24])]
        # Inference default sigma is 3.0 (the paper's evaluation used this default while training used
        # tactile_spatial_sigma=1.8); pass --tactile_spatial_sigma to change it.
        transformer3d.tactile_injector = TactileSpatialInjector(
            dim=_dim, layers=_layers, sigma=float(getattr(args, "tactile_spatial_sigma", 3.0)),
            forced=bool(getattr(args, "tactile_forced", False)),
        ).to(device=device, dtype=torch.float32)
        transformer3d._dtwm_tac_spatial = None
        logger.info(f"🖐️  DTWM eval: TactileSpatialInjector(dim={_dim}, layers={_layers}) attached")

    # DTWM causal-AR: enable BLOCK-CAUSAL temporal attention for teacher-forced chunk prediction.
    _ar_chunk = int(getattr(args, "ar_chunk", 0) or 0)
    if _ar_chunk > 0:
        transformer3d._dtwm_bc = {"chunk": _ar_chunk}
        logger.info(f"🎬 DTWM causal-AR eval: BLOCK-CAUSAL attention ON (ar_chunk={_ar_chunk} latent frames/chunk)")

    if training_mode == "lora" and args.checkpoint_path:
        logger.info("🔧 Loading LoRA weights via merge_lora...")
        lora_path = os.path.join(args.checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
        if os.path.exists(lora_path):
            merge_lora(
                pipeline,
                lora_path,
                multiplier=1.0,
                device=device,
                dtype=weight_dtype,
            )
            logger.info("✅ LoRA weights loaded via merge_lora")
        else:
            # A missing LoRA file used to be a warning, which silently evaluated the bare base model.
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        
        # Load non-LoRA trainable weights (e.g., patch_embedding)
        non_lora_path = os.path.join(args.checkpoint_path, "non_lora_weights.safetensors")
        if os.path.exists(non_lora_path):
            non_lora_state_dict = load_file(non_lora_path)
            transformer_state_dict = transformer3d.state_dict()
            loaded_keys = []
            for name, param_data in non_lora_state_dict.items():
                if name in transformer_state_dict:
                    transformer_state_dict[name].copy_(param_data.to(transformer_state_dict[name].dtype))
                    loaded_keys.append(name)
            transformer3d.load_state_dict(transformer_state_dict)
            logger.info(f"✅ Loaded non-LoRA weights: {len(loaded_keys)} parameters")
            # GUARD: the loop above SILENTLY SKIPS any checkpoint key the eval-time model has no slot
            # for, which would turn a train/eval configuration mismatch into a plausible-looking
            # result instead of a crash. Non-persistent buffers legitimately absent from the eval
            # module are exempt.
            _exempt = ("tactile_injector.scale", "tactile_tokenizer.cp_region_",
                       "tactile_tokenizer.cp_taxel_")
            _dropped = [n for n in non_lora_state_dict
                        if n.startswith(("tactile_injector.", "tactile_tokenizer."))
                        and n not in transformer_state_dict
                        and not n.startswith(_exempt)]
            # the reverse mismatch, and the dangerous one: a parameter the model expects that the
            # checkpoint does not carry stays at its random init and is used as if trained.
            _missing = [n for n in transformer_state_dict
                        if n.startswith(("tactile_injector.", "tactile_tokenizer."))
                        and n not in non_lora_state_dict]
            # One case is not a mismatch: a checkpoint that carries NO tactile tensors at all (tactile
            # pathway attached but never trained). Its residual is identically zero because the
            # injector's output projection is zero-initialised, so leaving the tensors at their init
            # reproduces training exactly. A PARTIAL absence stays fatal, and a total absence is
            # checked against the projection actually being zero rather than assumed.
            if _missing:
                _all_absent = not any(
                    n.startswith(("tactile_injector.", "tactile_tokenizer."))
                    for n in non_lora_state_dict)
                _proj_zero = all(
                    float(p.abs().max()) == 0.0
                    for n, p in transformer3d.named_parameters()
                    if ".proj." in n and n.startswith("tactile_injector."))
                if _all_absent and _proj_zero:
                    logger.warning(
                        f"⚠️  {len(_missing)} tactile tensors absent from the checkpoint and NONE are "
                        f"present: this is a frozen-injector arm. proj is zero, so the residual is "
                        f"identically zero and the injector is inert. Proceeding.")
                else:
                    raise RuntimeError(
                        f"train/eval mismatch: {len(_missing)} tactile tensors the model expects are "
                        f"absent from the checkpoint (e.g. {_missing[:3]}) while "
                        f"{'some' if not _all_absent else 'no'} others are present"
                        f"{'' if _proj_zero else ' and proj is NOT zero'}; "
                        f"check --tactile_spatial_layers.")

            if _dropped:
                raise RuntimeError(
                    f"train/eval mismatch: {len(_dropped)} tactile tensors in the checkpoint have no "
                    f"slot in the eval model and would be silently discarded, e.g. {_dropped[:4]}. "
                    f"Check --tactile_spatial_layers and that the checkpoint was trained with the "
                    f"tactile configuration of this release.")
            # Inference-time pathway switch (--tactile_gate_scale): multiply the injector's fixed unit
            # scale after loading. 0.0 disables the tactile pathway on the SAME checkpoint (the paper's
            # "pathway disabled at inference" row); the default 1.0 changes nothing.
            _gs = float(getattr(args, "tactile_gate_scale", 1.0))
            if getattr(args, "tactile_spatial", False) and _gs != 1.0:
                inj = getattr(transformer3d, "tactile_injector", None)
                if inj is not None:
                    with torch.no_grad():
                        inj.scale.mul_(_gs)
                    logger.info(f"🧪 tactile pathway scaled by {_gs} "
                                f"({'PATHWAY DISABLED (pure-vision counterfactual)' if _gs == 0.0 else 'scaled'})")
        elif getattr(args, "tactile_spatial", False):
            raise FileNotFoundError(
                f"non_lora_weights.safetensors not found in {args.checkpoint_path}; the tactile pathway "
                f"(and the extended patch embedding) would stay at initialisation.")

    logger.info("✅ Pipeline built successfully")
    return pipeline


def load_video_data(
    video_path: str,
    data_root: str,
    config: Optional[Dict[str, Any]] = None,
    prompt_subdir: Optional[str] = None,
    ar_prefix: Optional[int] = None,
    ar_action_horizon: int = 0,
) -> Dict[str, Any]:
    """Load one dataset clip: GT video, prompt and the reference / hand conditions, all derived from
    `<data_root>/<video_path>` (siblings of the clip's `videos/` directory)."""
    if not video_path or not data_root:
        raise ValueError("load_video_data needs video_path and data_root")
    
    data = {}
    data_config = config.get("data", {}) if config else {}
    video_path_obj = Path(video_path) if video_path else None
    full_video_path = Path(data_root) / video_path if video_path and data_root else None
    static_video_source_mode = str(data_config.get("static_video_source_mode", "directory")).strip().lower()
    if static_video_source_mode not in {"directory", "copy_first_frame"}:
        raise ValueError(
            f"Unsupported static_video_source_mode: {static_video_source_mode}. "
            "Use one of: directory, copy_first_frame."
        )

    gt_video = None
    if full_video_path is not None and full_video_path.exists():
        gt_video = iio.imread(str(full_video_path)).astype(np.float32) / 255.0
    
    # Load prompt (auto-derived: <action_dir>/<prompt_subdir>/<stem>.txt)
    video_name = video_path_obj.stem
    # Use provided prompt_subdir if available, otherwise use config, otherwise default
    if prompt_subdir is None:
        prompt_subdir = data_config.get("prompt_subdir", "prompts")
    prompt_path = full_video_path.parent.parent / prompt_subdir / f"{video_name}.txt"
    if prompt_path.exists():
        with open(prompt_path, "r") as f:
            data["prompt"] = f.read().strip()
    else:
        logger.warning(f"⚠️  Prompt not found: {prompt_path}")
        data["prompt"] = ""
    
    # Load static video (auto-derived; replaced by the GT prefix when --ar_prefix is set)
    if static_video_source_mode == "copy_first_frame":
        if gt_video is None:
            raise ValueError(
                f"Ground-truth video not found for copy_first_frame mode: {full_video_path}"
            )
        if gt_video.shape[0] <= 0:
            raise ValueError(f"Ground-truth video has no frames: {full_video_path}")
        static_video = np.repeat(gt_video[:1], gt_video.shape[0], axis=0).copy()
        data["static_video"] = torch.from_numpy(static_video).permute(3, 0, 1, 2).unsqueeze(0)
    else:
        static_video_subdir = data_config.get("static_video_subdir", "videos_static")
        static_video_path = full_video_path.parent.parent / static_video_subdir / video_path_obj.name
        if static_video_path.exists():
            static_video = iio.imread(str(static_video_path)).astype(np.float32) / 255.0
            data["static_video"] = torch.from_numpy(static_video).permute(3, 0, 1, 2).unsqueeze(0)
        else:
            raise ValueError(f"Static video not found: {static_video_path}")
    
    # Load hand video (optional, auto-derived from data.hand_video_subdir)
    hand_video_subdir = data_config.get("hand_video_subdir", "videos_hands")
    if hand_video_subdir:
        hand_video_path = full_video_path.parent.parent / hand_video_subdir / video_path_obj.name
        if hand_video_path.exists():
            hand_video = iio.imread(str(hand_video_path)).astype(np.float32) / 255.0
            data["hand_video"] = torch.from_numpy(hand_video).permute(3, 0, 1, 2).unsqueeze(0)
        else:
            logger.warning(f"⚠️  Hand video not found: {hand_video_path}. Using None.")
            data["hand_video"] = None
    else:
        data["hand_video"] = None

    # ===== DTWM AR: prefix-conditioned inference (teacher-forced chunk prediction). =====
    # Rebuild the conditions at the RAW stage so they flow through the same resize/pad path below:
    #   reference (static slot) := [GT frames 0..K | zeros]                        (matches AR training)
    #   hand condition          := [observed skeleton 0..K | future stream K+1..K+H | black]
    # The future stream (data.hand_video_future_subdir) must be configured: it is what blanks the hand
    # condition after K. With H = --ar_action_horizon = 0 (the paper) nothing of it is copied.
    if ar_prefix is not None:
        if gt_video is None:
            raise ValueError("--ar_prefix requires the GT video (video_path + data_root)")
        _K = int(ar_prefix)
        _gt_t = torch.from_numpy(gt_video).permute(3, 0, 1, 2).unsqueeze(0)   # [1,c,f,h,w]
        _ref = _gt_t.clone()
        _ref[:, :, _K + 1:] = 0.0        # InP-native unknown future
        data["static_video"] = _ref
        _fut_subdir = data_config.get("hand_video_future_subdir")
        if data.get("hand_video") is not None and _fut_subdir and full_video_path is not None:
            _fpath = full_video_path.parent.parent / _fut_subdir / video_path_obj.name
            if not _fpath.exists():
                raise ValueError(f"AR future hand video not found: {_fpath}")
            _fv = iio.imread(str(_fpath)).astype(np.float32) / 255.0
            _fv_t = torch.from_numpy(_fv).permute(3, 0, 1, 2).unsqueeze(0)
            _hv = data["hand_video"].clone()
            _hv[:, :, _K + 1:] = 0
            _end = min(_K + 1 + int(ar_action_horizon), _hv.shape[2])
            _hv[:, :, _K + 1:_end] = _fv_t[:, :, _K + 1:_end]
            data["hand_video"] = _hv
        logger.info(f"🎬 AR inference: prefix K={_K} (latents 0..{_K // 4} observed), "
                    f"action horizon={ar_action_horizon}")

    # Load GT video if available (for comparison)
    if gt_video is not None:
        data["gt_video"] = gt_video
    else:
        data["gt_video"] = None

    temporal_lengths = []
    if data["gt_video"] is not None:
        temporal_lengths.append(int(data["gt_video"].shape[0]))
    for key in ("static_video", "hand_video"):
        if data.get(key) is not None:
            temporal_lengths.append(int(data[key].shape[2]))

    if temporal_lengths:
        common_len = min(temporal_lengths)
        if common_len <= 0:
            raise ValueError("Inference sample has non-positive temporal length")
        if data["gt_video"] is not None and data["gt_video"].shape[0] != common_len:
            data["gt_video"] = data["gt_video"][:common_len]
        for key in ("static_video", "hand_video"):
            if data.get(key) is not None and data[key].shape[2] != common_len:
                data[key] = data[key][:, :, :common_len].contiguous()

    target_num_frames = _resolve_inference_num_frames(config)
    if target_num_frames is not None:
        if data["gt_video"] is not None:
            data["gt_video"] = _clip_or_pad_video_fhwc(data["gt_video"], target_num_frames)
        for key in ("static_video", "hand_video"):
            if data.get(key) is not None:
                data[key] = _clip_or_pad_video_bcfhw(data[key], target_num_frames)

    reference_h = None
    reference_w = None
    if data["gt_video"] is not None:
        reference_h = int(data["gt_video"].shape[1])
        reference_w = int(data["gt_video"].shape[2])
    elif data["static_video"] is not None:
        reference_h = int(data["static_video"].shape[-2])
        reference_w = int(data["static_video"].shape[-1])

    target_size = None
    if reference_h is not None and reference_w is not None:
        target_size = _resolve_inference_spatial_size(reference_h, reference_w, config)

    if target_size is not None and (reference_h, reference_w) != target_size:
        target_h, target_w = target_size
        validation_resize_mode = data_config.get("validation_resize_mode", "short_side_pad")
        validation_pad_value = float(data_config.get("validation_pad_value", 0.0))
        validation_center_pad = bool(data_config.get("validation_center_pad", True))
        short_side_mode = "stretch" if validation_resize_mode == "stretch" else (validation_resize_mode == "short_side_pad")
        logger.info(
            "📐 Resizing inference sample from %dx%d to %dx%d",
            reference_h,
            reference_w,
            target_h,
            target_w,
        )

        if data["gt_video"] is not None:
            data["gt_video"] = _resize_and_pad_video(
                data["gt_video"],
                target_h=target_h,
                target_w=target_w,
                pad_value=validation_pad_value,
                short_side_mode=short_side_mode,
                center_pad=validation_center_pad,
                layout_spec="FHWC",
            )
        for key in ("static_video", "hand_video"):
            if data.get(key) is not None:
                data[key] = _resize_and_pad_video(
                    data[key],
                    target_h=target_h,
                    target_w=target_w,
                    pad_value=validation_pad_value,
                    short_side_mode=short_side_mode,
                    center_pad=validation_center_pad,
                    layout_spec="BCFHW",
                )
    
    return data


def run_inference(
    pipeline: Wan2_2FunInpaintHandConcatPipeline,
    prompt: str,
    static_video: torch.Tensor,
    hand_video: Optional[torch.Tensor],
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: Optional[int] = None,
    negative_prompt: str = "bad detailed",
    conditioning_mode: str = "dwm",
    device: Optional[torch.device] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    ar_prefix: Optional[int] = None,
    wan22_shift: float = 12.0,
    pin_prefix_latents: int = 0,
    clean_history: bool = False,
) -> np.ndarray:
    """Run inference with WAN pipeline."""
    
    # Use provided device or infer from static_video
    if device is None:
        device = static_video.device if isinstance(static_video, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # `is not None`, not `if seed`: --seed 0 is a seed (an earlier `if seed` left seed 0 unseeded).
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    
    # Mask video (dwm conditioning): an all-zero mask keeps the whole reference; with ar_prefix
    # (prefix inpainting) frames 0..K are kept and K+1.. are generated.
    if conditioning_mode != "dwm":
        raise ValueError(f"Unsupported conditioning_mode for inference: {conditioning_mode}")
    mask_video = torch.zeros(1, 1, num_frames, height, width, device=device, dtype=static_video.dtype)
    if ar_prefix is not None:
        mask_video[:, :, int(ar_prefix) + 1:, :, :] = 255

    # Move videos to device
    static_video = static_video.to(device, dtype=static_video.dtype)
    if hand_video is not None:
        hand_video = hand_video.to(device, dtype=hand_video.dtype)
    
    start_time = time.time()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=static_video.dtype):
            # WAN 2.2: no CLIP/disparity args; shift must match training's scheduler (5B-InP: 12.0)
            call_kwargs = {}
            if attention_kwargs is not None:
                call_kwargs["attention_kwargs"] = attention_kwargs
            output = pipeline(
                prompt=prompt,
                num_frames=num_frames,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                static_video=static_video,
                mask_video=mask_video,
                hand_video=hand_video,
                shift=wan22_shift,
                pin_prefix_latents=pin_prefix_latents,
                clean_history=clean_history,
                **call_kwargs,
            )
    
    generation_time = time.time() - start_time
    logger.info(f"✅ Generated video in {generation_time:.2f}s")
    
    # Convert output.videos to [1, c, f, h, w] format for save_videos_grid
    video = output.videos
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    
    # Handle different input shapes
    # Based on error: [3, 49, 480, 720] = [c, f, h, w]
    if video.ndim == 4:
        # Check if [c, f, h, w] (channel first) or [f, h, w, c] (channel last)
        if video.shape[0] in [3, 4]:  # [c, f, h, w] - channel first
            video = video.unsqueeze(0)  # [1, c, f, h, w]
        elif video.shape[-1] in [3, 4]:  # [f, h, w, c] - channel last
            video = video.permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
        else:
            # Try to infer: if first dim is small (3-4), assume [c, f, h, w]
            if video.shape[0] < 10:
                video = video.unsqueeze(0)  # [1, c, f, h, w]
            else:
                # Assume [f, h, w, c]
                video = video.permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
    elif video.ndim == 5:
        # Already [b, c, f, h, w] or [b, f, h, w, c]
        if video.shape[-1] in [3, 4]:  # [b, f, h, w, c]
            video = video.permute(0, 4, 1, 2, 3)  # [b, c, f, h, w]
        # Otherwise assume already [b, c, f, h, w]
    else:
        raise ValueError(f"Unexpected video shape: {video.shape}, expected 4D or 5D tensor")
    
    return video.cpu(), generation_time


def get_output_filename(video_path, suffix=""):
    """Generate output filename from video path."""
    video_path_obj = Path(video_path)
    try:
        base_name = build_output_stem(video_path_obj)
    except Exception:
        base_name = "_".join(list(video_path_obj.parts[-4:-2]) + [video_path_obj.stem]) \
            if len(video_path_obj.parts) >= 4 else video_path_obj.stem

    # Add suffix to base_name if provided
    if suffix:
        base_name = f"{base_name}{suffix}"
    
    return f"{base_name}_generated.mp4"


def check_output_exists(video_path, output_dir, args, suffix=""):
    """Check if output files already exist in subfolder or unified output_dir.
    
    For distributed modes, checks both:
    1. Subfolder (rank_{rank}) - current output location
    2. Unified output_dir (args.output_dir) - already moved location
    """
    if not args.skip_existing:
        return False
    filename = get_output_filename(video_path, suffix)
    output_dir = Path(output_dir)
    unified_output_dir = Path(args.output_dir)
    
    # Check in subfolder (current output location)
    subfolder_file = output_dir / filename
    if subfolder_file.exists():
        logger.info(f"⏭️  Skipping {video_path}: output exists in {subfolder_file}")
        return True
    
    # Check in unified output_dir (if different from subfolder)
    # This handles the case where files have already been moved
    if unified_output_dir != output_dir:
        unified_file = unified_output_dir / filename
        if unified_file.exists():
            logger.info(f"⏭️  Skipping {video_path}: output exists in unified dir {unified_file}")
            return True
    
    return False


def get_partition_bounds(total_items: int, partition_idx: int, num_partitions: int) -> tuple[int, int]:
    """Return the inclusive-exclusive slice for a partition."""
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be positive, got {num_partitions}")
    if partition_idx < 0 or partition_idx >= num_partitions:
        raise ValueError(
            f"partition_idx must satisfy 0 <= partition_idx < num_partitions, "
            f"got partition_idx={partition_idx}, num_partitions={num_partitions}"
        )

    items_per_partition = total_items // num_partitions
    remainder = total_items % num_partitions
    start_idx = partition_idx * items_per_partition + min(partition_idx, remainder)
    end_idx = start_idx + items_per_partition + (1 if partition_idx < remainder else 0)
    return start_idx, end_idx


def prepare_video_paths(video_paths: List[str], args: argparse.Namespace) -> List[str]:
    """Apply optional dataset shuffle and truncation before worker splitting."""
    prepared_paths = list(video_paths)

    if args.shuffle_dataset and len(prepared_paths) > 1:
        shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.seed
        order = np.random.default_rng(shuffle_seed).permutation(len(prepared_paths))
        prepared_paths = [prepared_paths[idx] for idx in order]
        logger.info(
            "🔀 Shuffled dataset entries with seed=%s (%d videos)",
            shuffle_seed,
            len(prepared_paths),
        )

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(f"--max_samples must be positive, got {args.max_samples}")
        original_count = len(prepared_paths)
        prepared_paths = prepared_paths[: args.max_samples]
        logger.info(
            "✂️  Applied max_samples=%d: %d -> %d videos",
            args.max_samples,
            original_count,
            len(prepared_paths),
        )

    return prepared_paths


def get_batch_summary_path(
    output_dir: Path,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Path:
    """Choose a summary filename that avoids collisions across split modes."""
    if args.split_id is not None and world_size == 1:
        return output_dir / f"batch_summary_split_{args.split_id}.json"
    if world_size > 1:
        return output_dir / f"batch_summary_rank_{rank}.json"
    return output_dir / "batch_summary.json"


def _ensure_video_bcfhw(video: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert FHWC numpy or BCFHW/CFHW/FHWC tensor video to BCFHW torch."""
    if isinstance(video, np.ndarray):
        if video.ndim != 4:
            raise ValueError(f"Expected FHWC numpy video, got shape {video.shape}")
        return torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)

    if not isinstance(video, torch.Tensor):
        raise TypeError(f"Unsupported video type: {type(video)}")

    if video.ndim == 5:
        return video
    if video.ndim != 4:
        raise ValueError(f"Expected 4D/5D tensor video, got shape {tuple(video.shape)}")

    if video.shape[0] in (3, 4):
        return video.unsqueeze(0)
    if video.shape[-1] in (3, 4):
        return video.permute(3, 0, 1, 2).unsqueeze(0)

    raise ValueError(f"Cannot infer tensor video layout from shape {tuple(video.shape)}")


def _prepare_comparison_parts(
    parts: List[torch.Tensor],
    target_h: int,
    target_w: int,
) -> List[torch.Tensor]:
    """Resize/pad comparison parts to a shared spatial size and trim to min frames."""
    normalized_parts = []
    for part in parts:
        if part.shape[3] != target_h or part.shape[4] != target_w:
            part = _resize_and_pad_video(
                part,
                target_h=target_h,
                target_w=target_w,
                layout_spec="BCFHW",
            )
        normalized_parts.append(part)

    min_frames = min(part.shape[2] for part in normalized_parts)
    return [part[:, :, :min_frames, :, :] for part in normalized_parts]


def save_outputs(
    generated_video: torch.Tensor,
    output_dir: Path,
    base_name: str,
    gt_video: Optional[np.ndarray] = None,
    static_video: Optional[torch.Tensor] = None,
    hand_video: Optional[torch.Tensor] = None,
    generation_time: float = 0.0,
    args: Optional[argparse.Namespace] = None,
    suffix: str = "",
):
    """Save inference outputs using save_videos_grid for consistency."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add suffix to base_name
    if suffix:
        base_name = f"{base_name}{suffix}"
    
    # Save generated video
    # generated_video is [1, c, f, h, w] format
    generated_path = output_dir / f"{base_name}_generated.mp4"
    save_fps = getattr(args, "save_fps", None)
    if save_fps is None:
        # Backward compatibility: fall back to legacy args.fps if present
        save_fps = getattr(args, "fps", 8) if args else 8
    save_videos_grid(generated_video, str(generated_path), fps=int(save_fps))
    logger.info(f"✅ Saved generated video: {generated_path}")
    
    # Save comparison video if GT available
    if gt_video is not None:
        # Convert GT video to torch tensor [1, c, f, h, w]
        gt_tensor = torch.from_numpy(gt_video).permute(3, 0, 1, 2).unsqueeze(0)  # [1, c, f, h, w]
        
        # Ensure same number of frames
        min_frames = min(generated_video.shape[2], gt_tensor.shape[2])
        generated_video_trimmed = generated_video[:, :, :min_frames, :, :]
        gt_tensor_trimmed = gt_tensor[:, :, :min_frames, :, :]
        
        # Concatenate along width: generated | gt
        comparison_video = torch.cat([generated_video_trimmed, gt_tensor_trimmed], dim=4)  # [1, c, f, h, 2*w]
        comparison_path = output_dir / f"{base_name}_comparison.mp4"
        save_videos_grid(comparison_video, str(comparison_path), fps=int(save_fps))
        logger.info(f"✅ Saved comparison video: {comparison_path}")

    conditioning_parts = []
    conditioning_labels = []
    comparison_dtype = generated_video.dtype
    for label, conditioning_video in (
        ("static", static_video),
        ("hand", hand_video),
    ):
        if conditioning_video is not None:
            conditioning_parts.append(_ensure_video_bcfhw(conditioning_video).cpu().to(dtype=comparison_dtype))
            conditioning_labels.append(label)

    if conditioning_parts:
        full_parts = conditioning_parts + [generated_video.cpu().to(dtype=comparison_dtype)]
        full_labels = conditioning_labels + ["generated"]
        if gt_video is not None:
            full_parts.append(_ensure_video_bcfhw(gt_video).cpu().to(dtype=comparison_dtype))
            full_labels.append("gt")

        full_parts = _prepare_comparison_parts(
            full_parts,
            target_h=int(generated_video.shape[3]),
            target_w=int(generated_video.shape[4]),
        )
        full_comparison_video = torch.cat(full_parts, dim=4)
        full_comparison_path = output_dir / f"{base_name}_full_comparison.mp4"
        save_videos_grid(full_comparison_video, str(full_comparison_path), fps=int(save_fps))
        logger.info(
            "✅ Saved full comparison video (%s): %s",
            " | ".join(full_labels),
            full_comparison_path,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DTWM inference script")
    
    # Model configuration
    parser.add_argument(
        "--override", type=str, nargs="*", default=None,
        help="Config overrides key=value (same as training), e.g. data.hand_video_subdir=videos_hands_wilor",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint directory (…/checkpoint-<step>) with lora_diffusion_pytorch_model.safetensors "
             "and non_lora_weights.safetensors",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the Wan2.2-Fun-5B-InP base model (overrides pipeline.base_model_name_or_path of the config)",
    )
    parser.add_argument(
        "--tactile_spatial",
        action="store_true",
        help="DTWM: attach TactileTokenizer + TactileSpatialInjector and feed each clip's pressure "
             "(<action_dir>/pressure/<stem>.npy) as a residual at the hand's latent location "
             "(<action_dir>/hand_centroids/<stem>.npy). With --ar_prefix only observed frames get touch.",
    )
    parser.add_argument("--tactile_spatial_layers", type=int, nargs="+", default=None,
                        help="DiT blocks the spatial residual is injected at. MUST match training: the "
                             "injector's parameters are per-block, so a mismatch silently leaves blocks "
                             "randomly initialised. Default [0,6,12,18,24].")
    parser.add_argument("--tactile_constant", action="store_true",
                        help="Vision-only baseline: zero the raw pressure before the tokenizer (placement "
                             "unchanged). Use for checkpoints trained with training.tactile_input_constant=true "
                             "(scripts/train.sh VISION_ONLY=true).")
    parser.add_argument("--tactile_spatial_sigma", type=float, default=3.0,
                        help="Gaussian footprint std (latent cells) of the spatial tactile injection. The "
                             "paper's evaluation used this default (3.0) while training used 1.8.")
    parser.add_argument("--tactile_gate_scale", type=float, default=1.0,
                        help="Inference-time tactile pathway switch: multiply the injector's unit scale by "
                             "this after loading. 1.0 = as trained (default); 0.0 = pathway disabled on the "
                             "same checkpoint.")
    parser.add_argument("--ar_prefix", type=int, default=None,
                        help="DTWM AR: teacher-forced prefix length K (pixel frames, multiple of 4). "
                             "Conditions on GT frames 0..K (+ causal tactile); the model predicts K+1..end.")
    parser.add_argument("--ar_action_horizon", type=int, default=0,
                        help="DTWM AR: number of frames after K that receive the future hand stream "
                             "(data.hand_video_future_subdir). 0 = no future hand input (the paper).")
    parser.add_argument("--tactile_forced", action="store_true",
                        help="Required with --tactile_spatial: the injector uses a fixed unit scale and a "
                             "zero-init projection, as in training (training.tactile_forced=true).")
    parser.add_argument("--pin_observed", action="store_true",
                        help="Deployment-pinning: at every sampling step overwrite observed-prefix latents "
                             "with (1-sigma)*GT+sigma*eps0 so later chunks attend TRUE noised-GT (matches "
                             "training teacher-forcing). Requires --ar_prefix; wan2.2 pipeline only.")
    parser.add_argument("--clean_history", action="store_true",
                        help="Observed prefix stays CLEAN for the whole sampling run + per-token timesteps "
                             "(history t=0). Must match a checkpoint trained with ar_clean_history.")
    parser.add_argument("--ar_chunk", type=int, default=0,
                        help="DTWM AR: latent frames per chunk (>0 enables BLOCK-CAUSAL attention). Must "
                             "match training.ar_bc_chunk (4 in the paper).")
    parser.add_argument(
        "--experiment_config",
        type=str,
        default=None,
        help="Path to experiment config YAML (auto-detected from checkpoint if not provided)",
    )
    
    # Input data
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Text file with one clip path per line, relative to --data_root",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root directory (the clip paths in --dataset_file are relative to it)",
    )
    parser.add_argument(
        "--prompt_subdir",
        type=str,
        default=None,
        help="Subdirectory name for prompts (overrides config value, default: 'prompts')",
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default=None,
        help="Prefix to prepend to all prompts (e.g., 'first-person view, egocentric perspective,'). Default: None",
    )
    parser.add_argument(
        "--shuffle_dataset",
        dest="shuffle_dataset",
        action="store_true",
        help="Shuffle dataset entries before applying max_samples and worker splitting.",
    )
    parser.add_argument(
        "--no_shuffle_dataset",
        dest="shuffle_dataset",
        action="store_false",
        help="Disable dataset shuffling.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Random seed used only for dataset shuffling (defaults to --seed).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on the number of dataset entries processed after optional shuffle.",
    )
    parser.set_defaults(shuffle_dataset=False)
    
    # Inference parameters
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Output width",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=49,
        help="Number of frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="(Deprecated; use --save_fps) FPS for output video",
    )
    parser.add_argument(
        "--save_fps",
        type=int,
        default=None,
        help="FPS for saved output videos (defaults to --fps).",
    )
    parser.add_argument(
        "--model_fps",
        type=int,
        default=None,
        help="FPS assumed by the model (RoPE scaling). Default: pipeline_config.fps if set, else 16.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="bad detailed",
        help="Negative prompt",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {checkpoint_path}/eval)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to append to output filenames",
    )
    
    # Multi-GPU support
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=None,
        help="Optional chunk id for dataset-file inference (0-based).",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="Total number of chunks when using --chunk_id.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes for distributed training",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=None,
        help="Optional explicit split id for single-process sharding (0-based).",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=None,
        help="Total number of splits when using --split_id or SLURM array mode.",
    )
    
    # Skip existing files
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip inference if output files already exist (default: True)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Disable skip_existing (force re-run even if files exist)",
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    if args.chunk_id is None and args.split_id is not None:
        args.chunk_id = args.split_id
    if args.num_chunks is None and args.num_splits is not None:
        args.num_chunks = args.num_splits

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"--max_samples must be positive, got {args.max_samples}")

    if args.chunk_id is not None or args.num_chunks is not None:
        if args.chunk_id is None or args.num_chunks is None:
            raise ValueError("Both --chunk_id and --num_chunks must be provided together.")
        if args.num_chunks <= 0:
            raise ValueError(f"--num_chunks must be positive, got {args.num_chunks}")
        if args.chunk_id < 0 or args.chunk_id >= args.num_chunks:
            raise ValueError(
                f"--chunk_id must satisfy 0 <= chunk_id < num_chunks, "
                f"got chunk_id={args.chunk_id}, num_chunks={args.num_chunks}"
            )

    # Setup distributed training if available
    rank, world_size, local_rank = setup_distributed()

    args.split_id = args.chunk_id
    args.num_splits = args.num_chunks
    if args.chunk_id is not None and rank == 0:
        logger.info(f"📊 Chunk mode enabled: chunk_id={args.chunk_id}, num_chunks={args.num_chunks}")
    
    # Set device based on distributed setup
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load experiment config
    if args.experiment_config is None and args.checkpoint_path:
        config_path = find_experiment_config(args.checkpoint_path)
        if config_path:
            if rank == 0:
                logger.info(f"🔧 Auto-detected config: {config_path}")
            args.experiment_config = config_path
            config = load_experiment_config_wrapper(config_path, getattr(args, "override", None))
        else:
            if rank == 0:
                logger.warning("⚠️  Could not auto-detect config. Some settings may use defaults.")
            config = {}
    else:
        if args.experiment_config:
            config = load_experiment_config_wrapper(args.experiment_config, getattr(args, "override", None))
        else:
            config = {}
    
    # Resolve model_fps and save_fps
    pipeline_fps = None
    try:
        pipeline_fps = config.get("pipeline", {}).get("fps") if isinstance(config, dict) else None
    except Exception:
        pipeline_fps = None

    if args.model_fps is None:
        args.model_fps = int(pipeline_fps) if pipeline_fps is not None else 16
    if args.save_fps is None:
        args.save_fps = int(args.fps)

    if rank == 0:
        logger.info(f"🎞️ model_fps={args.model_fps} (transformer/RoPE), save_fps={args.save_fps} (mp4)")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_path, "eval")
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
    
    # Build pipeline
    # Must precede build_pipeline: that is where the TactileTokenizer is constructed, and it sizes
    # itself from _DTWM_L.
    global _DTWM_L
    _DTWM_L = int(getattr(args, "num_frames", 49) or 49)
    logger.info(f"🖐️  DTWM eval: tactile path sized to _DTWM_L={_DTWM_L} frames")
    pipeline = build_pipeline(args, config)
    
    # Load dataset file (only on rank 0, then split)
    if rank == 0 or world_size == 1:
        # Batch mode: process multiple videos from dataset file
        with open(args.dataset_file, "r") as f:
            video_paths = [line.strip() for line in f if line.strip()]
            
        if rank == 0:
            logger.info(f"📋 Loaded {len(video_paths)} videos from dataset file")
        video_paths = prepare_video_paths(video_paths, args)
        if rank == 0:
            logger.info(f"📋 Effective dataset size after ordering/filtering: {len(video_paths)} videos")
        if args.chunk_id is not None:
            total_videos = len(video_paths)
            start_idx, end_idx = get_partition_bounds(total_videos, args.chunk_id, args.num_chunks)
            video_paths = video_paths[start_idx:end_idx]
            if rank == 0:
                logger.info(
                    "📦 Chunk %d/%d selected %d videos (indices %d-%d of %d)",
                    args.chunk_id,
                    args.num_chunks,
                    len(video_paths),
                    start_idx,
                    end_idx - 1,
                    total_videos,
                )
    else:
        video_paths = []  # Will be received from rank 0

    # Split video_paths across GPUs for multi-GPU inference
    if world_size > 1:
        # Use GPU for broadcasting (nccl backend requires GPU tensors)
        broadcast_device = torch.device(f"cuda:{rank}")
        
        # Broadcast video_paths from rank 0 to all ranks
        if rank == 0:
            # Convert to list of strings for broadcasting
            video_paths_str = json.dumps(video_paths)
            video_paths_tensor = torch.tensor([ord(c) for c in video_paths_str], dtype=torch.int32, device=broadcast_device)
            video_paths_len = torch.tensor([len(video_paths_str)], dtype=torch.int32, device=broadcast_device)
        else:
            video_paths_tensor = torch.zeros(1, dtype=torch.int32, device=broadcast_device)
            video_paths_len = torch.zeros(1, dtype=torch.int32, device=broadcast_device)
        
        # Broadcast length first
        dist.broadcast(video_paths_len, src=0)
        if rank != 0:
            video_paths_tensor = torch.zeros(video_paths_len.item(), dtype=torch.int32, device=broadcast_device)
        
        # Broadcast video_paths
        dist.broadcast(video_paths_tensor, src=0)
        
        if rank != 0:
            video_paths_str = ''.join([chr(c.item()) for c in video_paths_tensor.cpu()])
            video_paths = json.loads(video_paths_str)
        else:
            # Move to CPU for processing
            video_paths_tensor = video_paths_tensor.cpu()
        
        # Split video_paths across ranks
        total_videos = len(video_paths)
        start_idx, end_idx = get_partition_bounds(total_videos, rank, world_size)
        my_video_paths = video_paths[start_idx:end_idx]
        
        logger.info(f"🎯 Rank {rank}/{world_size}: Processing {len(my_video_paths)} videos (indices {start_idx}-{end_idx-1} of {total_videos})")
    else:
        my_video_paths = video_paths
    
    # Create output directory based on mode
    if world_size > 1:
        # Distributed mode: use rank_{rank} directory
        rank_output_dir = output_dir / f"rank_{rank}"
    else:
        # Single GPU mode: use output_dir directly
        rank_output_dir = output_dir
    rank_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process videos
    results = []
    for video_path in my_video_paths:
        try:
            logger.info(f"🎯 Processing: {video_path}")

            # Check if output already exists (skip if exists)
            if check_output_exists(video_path, rank_output_dir, args, args.suffix):
                results.append({
                    "video_path": video_path,
                    "success": True,
                    "skipped": True,
                })
                continue
            
            # Load video data (all paths auto-derived from the dataset entry)
            data = load_video_data(
                video_path=video_path,
                data_root=args.data_root,
                config=config,
                prompt_subdir=args.prompt_subdir,
                ar_prefix=getattr(args, "ar_prefix", None),
                ar_action_horizon=getattr(args, "ar_action_horizon", 0),
            )
            
            # ===== DTWM: per-clip spatial tactile inputs (GT pressure + hand centroids). =====
            if getattr(args, "tactile_spatial", False):
                from dtwm.models.tactile_tokens import spatial_reduce
                _vp = Path(video_path)
                if not _vp.is_absolute() and args.data_root:
                    _vp = Path(args.data_root) / _vp
                _pp = _vp.parent.parent / "pressure" / (_vp.stem + ".npy")
                _cp = _vp.parent.parent / "hand_centroids" / (_vp.stem + ".npy")
                _tk = pipeline.transformer.tactile_tokenizer
                if _pp.is_file() and _cp.is_file():
                    _dev = next(pipeline.transformer.parameters()).device
                    _dt = next(pipeline.transformer.parameters()).dtype
                    _Pnp = np.load(_pp).astype("float32")[:_DTWM_L]
                    if getattr(args, "tactile_constant", False):
                        # Vision-only baseline: zero the raw pressure (mirrors training.tactile_input_constant).
                        _Pnp = np.zeros_like(_Pnp)
                    _P = torch.from_numpy(_Pnp).to(_dev)                                        # (49,2,21,21)
                    _sl, _sr = _tk.gather(_P[None])                                           # (1,49,n)
                    _obs = torch.ones(_sl.shape[1], dtype=torch.bool, device=_dev)
                    _Ef = _tk.frame_embeddings(_sl, _sr, _obs)                            # (1,49,2,dim)
                    _C = torch.from_numpy(np.load(_cp).astype("float32")[:49]).to(_dev)[None]  # (1,49,2,3)
                    _E, _cen, _val = spatial_reduce(_Ef, _C)
                    if getattr(args, "ar_prefix", None) is not None:
                        # AR: CAUSAL touch -- inject only on observed latent frames 0..K//4.
                        _val[:, int(args.ar_prefix) // 4 + 1:] = 0
                    pipeline.transformer._dtwm_tac_spatial = {"E": _E.to(_dt), "centroids": _cen, "valid": _val}
                else:
                    pipeline.transformer._dtwm_tac_spatial = None
                    logger.warning(f"⚠️  no pressure/centroids for {_vp.name}; tactile disabled for this clip")

            # Determine video dimensions
            if data["gt_video"] is not None:
                num_frames = data["gt_video"].shape[0]
                height, width = data["gt_video"].shape[1], data["gt_video"].shape[2]
            else:
                num_frames = args.num_frames
                height, width = args.height, args.width
            
            # Apply prompt prefix if provided
            final_prompt = data["prompt"]
            if args.prompt_prefix:
                final_prompt = f"{args.prompt_prefix} {final_prompt}".strip()
                logger.info(f"📝 Prompt with prefix: {final_prompt[:100]}...")
            
            # Run inference
            conditioning_mode = config.get("training", {}).get("conditioning_mode", "dwm")
            generated_video, gen_time = run_inference(
                pipeline=pipeline,
                prompt=final_prompt,
                static_video=data["static_video"],
                hand_video=data["hand_video"],
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
                conditioning_mode=conditioning_mode,
                device=device,
                ar_prefix=getattr(args, "ar_prefix", None),
                wan22_shift=float(config.get("scheduler_kwargs", {}).get("shift", 12.0)),
                pin_prefix_latents=((int(getattr(args, "ar_prefix", 0) or 0) // 4 + 1)
                                    if ((getattr(args, "pin_observed", False) or getattr(args, "clean_history", False)) and getattr(args, "ar_prefix", None) is not None) else 0),
                clean_history=bool(getattr(args, "clean_history", False)),
            )

            # Save outputs. Unique name <collection>_<sample>_<stem>: clips often share the stem
            # "00000", so the bare stem would collide/overwrite in batch eval.
            _vp = Path(video_path)
            base_name = "_".join(list(_vp.parts[-4:-2]) + [_vp.stem]) if len(_vp.parts) >= 4 else _vp.stem

            save_outputs(
                generated_video=generated_video,
                output_dir=rank_output_dir,
                base_name=base_name,
                gt_video=data["gt_video"],
                static_video=data["static_video"],
                hand_video=data["hand_video"],
                generation_time=gen_time,
                args=args,
                suffix=args.suffix,
            )
            
            results.append({"video_path": video_path, "success": True})
            
        except Exception as e:
            # Several exception types carry an empty str(), which made a whole shard fail silently
            # with a bare "Failed to process" line; the traceback is what identifies the cause.
            import traceback as _tb
            logger.error(f"❌ Failed to process {video_path}: {type(e).__name__}: {e}\n"
                         f"{_tb.format_exc()}")
            results.append({"video_path": video_path, "success": False,
                            "error": f"{type(e).__name__}: {e}"})
    
    # Save batch summary for this rank/split
    summary = {
        "total": len(my_video_paths),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "rank": rank,
        "world_size": world_size,
        "split_id": args.split_id,
        "num_splits": args.num_splits,
        "shuffle_dataset": args.shuffle_dataset,
        "shuffle_seed": args.shuffle_seed if args.shuffle_seed is not None else args.seed,
        "max_samples": args.max_samples,
        "results": results,
    }
    summary_path = get_batch_summary_path(rank_output_dir, args, rank, world_size)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    if rank == 0 or world_size == 1:
        logger.info(f"📄 Batch summary saved: {summary_path}")
    
    # Gather results from all ranks and merge (only on rank 0)
    if world_size > 1:
        # Wait for all ranks to finish
        dist.barrier()
        
        if rank == 0:
            logger.info(f"🔄 Merging results from {world_size} ranks...")
            
            # Collect all results
            all_results = []
            all_summaries = []
            
            for r in range(world_size):
                rank_dir = output_dir / f"rank_{r}"
                if rank_dir.exists():
                    # Load batch summary from each rank
                    summary_file = rank_dir / f"batch_summary_rank_{r}.json"
                    if summary_file.exists():
                        with summary_file.open('r') as f:
                            summary = json.load(f)
                        all_summaries.append(summary)
                        all_results.extend(summary.get("results", []))
            
            # Create merged summary
            merged_summary = {
                "total": sum(s.get("total", 0) for s in all_summaries),
                "successful": sum(s.get("successful", 0) for s in all_summaries),
                "failed": sum(s.get("failed", 0) for s in all_summaries),
                "num_ranks": world_size,
                "rank_summaries": all_summaries,
                "all_results": all_results,
            }
            
            # Save merged summary
            merged_dir = output_dir / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_summary_file = merged_dir / "merged_summary.json"
            with merged_summary_file.open('w') as f:
                json.dump(merged_summary, f, indent=2)
            
            logger.info(f"✅ Merged summary saved: {merged_summary_file}")
            logger.info(f"📊 Total videos: {merged_summary['total']}")
            logger.info(f"✅ Successful: {merged_summary['successful']}")
            logger.info(f"❌ Failed: {merged_summary['failed']}")
            
            # Move results to output_dir
            logger.info(f"📦 Moving results from rank directories to {output_dir}...")
            for r in range(world_size):
                rank_dir = output_dir / f"rank_{r}"
                if rank_dir.exists():
                    for item in rank_dir.iterdir():
                        if item.suffix in ['.mp4', '.json']:
                            dest = output_dir / item.name
                            if dest.exists():
                                # Append rank number if file exists
                                dest = output_dir / f"{item.stem}_rank{r}{item.suffix}"
                            shutil.move(str(item), str(dest))
                    # Remove rank directory if empty
                    try:
                        rank_dir.rmdir()
                    except OSError:
                        pass
            logger.info(f"✅ All results moved to {output_dir}")
        
        # Cleanup distributed training
        dist.destroy_process_group()
    
    if rank == 0 or world_size == 1:
        logger.info("✅ Inference completed")


if __name__ == "__main__":
    main()
