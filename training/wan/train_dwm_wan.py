# Copyright 2024 The HuggingFace Team.
# Modified for WAN-based training with VideoX-Fun style.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
WAN-based training script for DTWM (Dexterous Tactile World Model), derived from the DWM trainer.

Supports:
- LoRA training (VideoX-Fun lora_utils style) of WanTransformer3DModelWithConcat (Wan2.2-Fun-5B-InP)
- Flow Matching loss
- DTWM: prefix-conditioned, block-causal autoregressive chunk prediction with clean history and
  spatially grounded tactile injection (src/dtwm/models/tactile_tokens.py)
"""

import argparse
import logging
import math
import os
import random
import re
import shutil
import sys
import warnings
from datetime import datetime as dt, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

# DTWM: make the repository root (`training.*`) and `src/` (`dtwm.*`) importable from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.wan.diffusers_compat import disable_problematic_optional_backends

disable_problematic_optional_backends()

from safetensors.torch import load_file, save_file

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DummyScheduler,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# WAN models and utilities from training.wan
from training.wan.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    WanT5EncoderModel,
)
from training.wan.models.wan_transformer3d_with_conditions import (
    WanTransformer3DModelWithConcat,
)
from training.wan.utils.lora_utils import create_network
from training.wan.utils.checkpoint_utils import (
    extract_checkpoint_step,
    list_checkpoint_dirs,
    save_training_checkpoint,
)
from training.wan.utils.utils import filter_kwargs
from training.wan.utils.discrete_sampler import DiscreteSampling
from training.wan.config_loader import load_experiment_config

# Dataset
from training.wan.dataset import VideoDatasetWithConditionsAndResizing

# Wandb for logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logger = get_logger(__name__)

PROMPT_MODE_NORMAL = "normal"
PROMPT_MODE_EMPTY = "empty"


def resolve_prompt_mode(training_config: Dict[str, Any]) -> str:
    """Resolve training prompt mode with limited backward compatibility."""
    prompt_mode = str(training_config.get("prompt_mode", "")).strip().lower()
    if prompt_mode:
        if prompt_mode not in {PROMPT_MODE_NORMAL, PROMPT_MODE_EMPTY}:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
        return prompt_mode

    if "prompt_dropout_prob" in training_config:
        legacy_prob = float(training_config["prompt_dropout_prob"])
        if legacy_prob == 0.0:
            warnings.warn(
                "training.prompt_dropout_prob is deprecated. Use training.prompt_mode=normal instead.",
                stacklevel=2,
            )
            return PROMPT_MODE_NORMAL
        if legacy_prob == 1.0:
            warnings.warn(
                "training.prompt_dropout_prob is deprecated. Use training.prompt_mode=empty instead.",
                stacklevel=2,
            )
            return PROMPT_MODE_EMPTY
        raise ValueError(
            "training.prompt_dropout_prob no longer supports fractional values. "
            "Use training.prompt_mode with one of: normal, empty."
        )

    return PROMPT_MODE_NORMAL


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cuda"):
    """Get sigmas for flow matching."""
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
    """Custom MSE loss with optional weighting and threshold."""
    noise_pred = noise_pred.float()
    target = target.float()
    diff = noise_pred - target
    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    if weighting is not None:
        masked_loss = masked_loss * weighting
    final_loss = masked_loss.mean()
    return final_loss


def _resolve_component_path(base_path: str, subpath: Optional[str], default_subpath: str) -> str:
    """
    Resolve model/tokenizer component path.
    - If local `<base_path>/<subpath>` exists, use it.
    - Otherwise treat `subpath` as a direct model id/path (e.g., `google/umt5-xxl`).
    """
    resolved_subpath = subpath or default_subpath
    if os.path.isabs(resolved_subpath):
        return resolved_subpath
    local_candidate = os.path.join(base_path, resolved_subpath)
    if os.path.exists(local_candidate):
        return local_candidate
    return resolved_subpath


def save_model_lora(output_dir, network, weight_dtype, accelerator, transformer3d=None, trainable_parameter_patterns=None):
    """Save LoRA weights and non-LoRA trainable weights."""
    if accelerator.is_main_process:
        # Save LoRA weights
        safetensor_save_path = os.path.join(output_dir, "lora_diffusion_pytorch_model.safetensors")
        accelerator.unwrap_model(network).save_weights(safetensor_save_path, weight_dtype, None)
        logger.info(f"Saved LoRA weights to {safetensor_save_path}")
        
        # Save non-LoRA trainable weights (e.g., patch_embedding)
        if transformer3d is not None and trainable_parameter_patterns:
            non_lora_state_dict = {}
            model = accelerator.unwrap_model(transformer3d)
            for name, param in model.named_parameters():
                for pattern in trainable_parameter_patterns:
                    if re.search(pattern, name):
                        non_lora_state_dict[name] = param.data.clone()
                        break
            # Buffers too: the tactile tokenizer's standardisation statistics (mean/std/cells) are
            # persistent buffers matched by the same patterns, so the checkpoint is self-describing.
            # Buffers here are a few KB.
            for name, buf in model.named_buffers():
                if name in non_lora_state_dict or buf is None:
                    continue
                for pattern in trainable_parameter_patterns:
                    if re.search(pattern, name):
                        non_lora_state_dict[name] = buf.data.clone()
                        break

            if non_lora_state_dict:
                non_lora_path = os.path.join(output_dir, "non_lora_weights.safetensors")
                save_file(non_lora_state_dict, non_lora_path)
                logger.info(f"Saved non-LoRA weights ({len(non_lora_state_dict)} params) to {non_lora_path}")


def save_train_mode_weights(
    save_path,
    args,
    network,
    weight_dtype,
    accelerator,
    transformer3d,
    trainable_parameter_patterns,
):
    """Save evaluation-friendly model weights for the active training mode."""
    os.makedirs(save_path, exist_ok=True)

    save_model_lora(save_path, network, weight_dtype, accelerator, transformer3d, trainable_parameter_patterns)

    logger.info(f"Saved checkpoint to {save_path}")


def main():
    from training.wan.args import add_all_args

    parser = argparse.ArgumentParser(description="DTWM (WAN-based) training script")
    # Core arguments
    parser.add_argument("--experiment_config", type=str, required=True,
                        help="Path to experiment YAML config file")
    parser.add_argument("--override", type=str, nargs="*",
                        help="Override config values (key=value format)")
    parser.add_argument("--mode", type=str, choices=["batch"], default="batch",
                        help="Training mode (the DTWM release keeps only the timestamped batch mode)")
    # # Add all other arguments from args.py
    add_all_args(parser)
    args = parser.parse_args()
    
    # ========== Load Configuration ==========
    print("🚀 Loading experiment configuration...")
    config = load_experiment_config(args.experiment_config, args.override)
    
    # Extract configuration sections
    experiment_config = config.get("experiment", {})
    training_config = config.get("training", {})
    pipeline_config = config.get("pipeline", {})
    transformer_config = config.get("transformer", {})
    data_config = config.get("data", {})
    prompt_mode = resolve_prompt_mode(training_config)
    if prompt_mode != PROMPT_MODE_NORMAL:
        raise ValueError("DTWM release: only training.prompt_mode=normal is supported.")

    # Print training FPS (used for RoPE scaling in WanTransformer3DModel)
    if pipeline_config.get("fps") is not None:
        print(f"🎞️ Training FPS: {int(pipeline_config['fps'])} (from pipeline_config)")
    else:
        print("🎞️ Training FPS: 16 (default)")
    
    # Model-related configs from pipeline config (merged from configs/pipelines/*.yaml)
    text_encoder_kwargs = config.get("text_encoder_kwargs", {})
    vae_kwargs = config.get("vae_kwargs", {})
    scheduler_kwargs = config.get("scheduler_kwargs", {})
    transformer_additional_kwargs = config.get("transformer_additional_kwargs", {})

    # Propagate fixed FPS (used for RoPE scaling) into transformer loading kwargs
    if pipeline_config.get("fps") is not None:
        transformer_additional_kwargs["fps"] = int(pipeline_config["fps"])
    
    # Override args with config values (config values take priority over command line args)
    # Use "is not None" check to ensure config values (including 0, False, empty string) override args
    if training_config.get("learning_rate") is not None:
        args.learning_rate = float(training_config["learning_rate"])
    if training_config.get("batch_size") is not None:
        args.train_batch_size = int(training_config["batch_size"])
    if training_config.get("max_train_steps") is not None:
        args.max_train_steps = int(training_config["max_train_steps"])
    if training_config.get("gradient_accumulation_steps") is not None:
        args.gradient_accumulation_steps = int(training_config["gradient_accumulation_steps"])
    # Note: args.pretrained_model_name_or_path is NOT overridden by pipeline_config
    # to allow specifying SFT checkpoint path via CLI (--pretrained_model_name_or_path)
    # pipeline_config["base_model_name_or_path"] is used for model structure definition
    if args.pretrained_model_name_or_path is None and pipeline_config.get("base_model_name_or_path") is not None:
        args.pretrained_model_name_or_path = pipeline_config["base_model_name_or_path"]
    if data_config.get("data_root") is not None:
        args.train_data_dir = data_config["data_root"]
    if data_config.get("dataset_file") is not None:
        args.train_data_meta = data_config["dataset_file"]
    # Determine transformer type
    transformer_type = transformer_config.get("class", "WanTransformer3DModelWithConcat")
    conditioning_mode = training_config.get("conditioning_mode", "dwm")
    custom_settings = training_config.get("custom_settings", {})
    keep_full_state_checkpoints = int(
        custom_settings.get("keep_full_state_checkpoints", args.keep_full_state_checkpoints)
    )
    if keep_full_state_checkpoints < 0:
        raise ValueError("training.custom_settings.keep_full_state_checkpoints must be >= 0")
    args.keep_full_state_checkpoints = keep_full_state_checkpoints
    # DTWM release: only the configuration used in the paper is implemented.
    if transformer_type != "WanTransformer3DModelWithConcat":
        raise ValueError(
            f"DTWM release supports transformer.class=WanTransformer3DModelWithConcat only, got {transformer_type!r}")
    if conditioning_mode != "dwm":
        raise ValueError(f"DTWM release supports training.conditioning_mode=dwm only, got {conditioning_mode!r}")
    if float(training_config.get("hand_dropout_prob", 0.0) or 0.0) > 0:
        raise ValueError("DTWM release: training.hand_dropout_prob > 0 (upstream DWM option) is not supported.")
    if transformer_config.get("condition_channels") is not None:
        args.condition_channels = int(transformer_config["condition_channels"])
    if training_config.get("mode") is not None:
        args.train_mode = training_config["mode"]
    if training_config.get("lora_rank") is not None:
        args.rank = int(training_config["lora_rank"])
    if training_config.get("lora_alpha") is not None:
        args.network_alpha = int(training_config["lora_alpha"])
    if args.train_mode != "lora":
        raise ValueError(f"DTWM release supports training.mode=lora only, got {args.train_mode!r}")
    
    # ========== Setup Output Directory ==========
    # Check for resume configuration
    resume_from_checkpoint = training_config.get("resume_from_checkpoint") or args.resume_from_checkpoint
    if resume_from_checkpoint:
        print(f"🔄 Resume mode enabled: resume_from_checkpoint = {resume_from_checkpoint}")
    
    # Setup output directory based on mode
    exp_name = experiment_config.get("name", "wan_training")
    exp_date = experiment_config.get("date", "unknown")
    
    # Extract date suffix
    if exp_date != "unknown":
        try:
            parsed_date = dt.strptime(exp_date, "%Y-%m-%d")
            date_suffix = parsed_date.strftime("%y%m%d")
        except:
            date_suffix = "unknown"
    else:
        date_suffix = "unknown"
    
    # Construct output directory based on mode
    # Resolved against the repository root, not the cwd; DTWM_OUTPUT_ROOT overrides the parent directory.
    output_root = os.environ.get("DTWM_OUTPUT_ROOT", str(_REPO_ROOT / "outputs"))
    base_output_dir = os.path.join(output_root, date_suffix, exp_name)
    if args.mode == "batch":
        # Batch mode: add timestamp to output directory (use env var so all processes get same dir)
        timestamp = os.environ.get("BATCH_TIMESTAMP")
        if not timestamp:
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"{base_output_dir}_batch_{timestamp}"
        print(f"🚀 Batch mode: Output directory set to {args.output_dir}")
    
    print(f"📁 Final output directory: {args.output_dir}")
    print(f"   - Base: {base_output_dir}")
    print(f"   - Experiment: {exp_name}")
    print(f"   - Date: {exp_date} -> {date_suffix}")
    print(f"   - Mode: {args.mode}")
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=1800))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=training_config.get("custom_settings", {}).get("mixed_precision", args.mixed_precision),
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )
    
    # ========== Detect DeepSpeed / FSDP ==========
    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"🚀 Using DeepSpeed Zero stage: {zero_stage}")
        args.use_deepspeed = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"🚀 Using FSDP stage: {fsdp_stage}")
        args.use_fsdp = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("🚀 DeepSpeed/FSDP is not enabled.")

    # Create output directory and copy experiment config
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Copy experiment config YAML to output directory
        if args.experiment_config:
            config_filename = os.path.basename(args.experiment_config)
            output_config_path = os.path.join(args.output_dir, config_filename)
            shutil.copy2(args.experiment_config, output_config_path)
            print(f"📋 Copied experiment config to: {output_config_path}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ========== Load Models ==========
    print("🔧 Loading models...")
    
    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
    )

    # Load tokenizer / text encoder / VAE
    tokenizer_path = _resolve_component_path(
        args.pretrained_model_name_or_path,
        text_encoder_kwargs.get('tokenizer_subpath'),
        'tokenizer',
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    text_encoder_path = _resolve_component_path(
        args.pretrained_model_name_or_path,
        text_encoder_kwargs.get('text_encoder_subpath'),
        'text_encoder',
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        text_encoder_path,
        additional_kwargs=text_encoder_kwargs,
        torch_dtype=weight_dtype,
    ).eval()

    vae_path = os.path.join(args.pretrained_model_name_or_path, vae_kwargs.get('vae_subpath', 'vae'))
    Choosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[vae_kwargs.get('vae_type', 'AutoencoderKLWan')]
    vae = Choosen_AutoencoderKL.from_pretrained(
        vae_path,
        additional_kwargs=vae_kwargs,
    ).eval()

    # Check if WAN 2.2 (needed for transformer loading and CLIP encoder)
    is_wan2_2 = "2.2" in pipeline_config.get("type", "")
    
    # WAN 2.2 uses no CLIP image encoder; the Wan2.1 (CLIP) path is not part of this release.
    if not is_wan2_2:
        raise ValueError("DTWM release supports the Wan2.2 pipeline (pipeline.type containing '2.2') only.")

    # Load transformer
    print(f"🔧 Loading WanTransformer3DModelWithConcat with {args.condition_channels} condition channels...")
    transformer3d = WanTransformer3DModelWithConcat.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, transformer_additional_kwargs.get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs={
            "condition_channels": args.condition_channels,
            "is_wan2_2": is_wan2_2,
            **transformer_additional_kwargs,
        },
        torch_dtype=weight_dtype,
    )

    # ========== Setup Training Mode ==========
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    network = None
    trainable_params = None

    # Get trainable_parameter_patterns from config (for non-LoRA trainable params like patch_embedding)
    trainable_parameter_patterns = training_config.get("trainable_parameter_patterns", [])

    # ===== DTWM: spatially grounded tactile injection. =====
    # Tactile is NOT concatenated into the sequence; after each selected block a Gaussian-footprint
    # residual is added at the hand's latent location. The injector's projection is zero-initialised
    # with a fixed unit scale, so training starts exactly from the pure-vision model.
    tactile_spatial = bool(training_config.get("tactile_spatial", False))
    if tactile_spatial:
        from dtwm.models.tactile_tokens import TactileTokenizer, TactileSpatialInjector
        _dim = int(transformer3d.patch_embedding.weight.shape[0])  # robust dim (wan2.2 dict_mapping may hide config.dim)
        _L = int(training_config.get("custom_settings", {}).get("max_num_frames", 49))
        if getattr(transformer3d, "tactile_tokenizer", None) is None:
            transformer3d.tactile_tokenizer = TactileTokenizer(dim=_dim, n_frames=_L)
        _layers = list(training_config.get("tactile_spatial_layers", [0, 6, 12, 18, 24]))
        _sigma = float(training_config.get("tactile_spatial_sigma", 3.0))
        # forced=True (the only variant in this release): fixed unit scale + zero-init proj, so the
        # tactile projection receives gradient from step 0. See TactileSpatialInjector.
        _forced = bool(training_config.get("tactile_forced", True))
        transformer3d.tactile_injector = TactileSpatialInjector(dim=_dim, layers=_layers, sigma=_sigma,
                                                                forced=_forced)
        if _forced:
            print(f"🔓 DTWM: tactile_forced=True -> fixed scale 1.0 + ZERO-INIT proj "
                  f"(residual starts at 0 == pure-vision start)")
        if getattr(args, "lora_skip_name", None) is None:
            args.lora_skip_name = "tactile"
        for _p in ("tactile_tokenizer", "tactile_injector"):
            if _p not in trainable_parameter_patterns:
                trainable_parameter_patterns = list(trainable_parameter_patterns) + [_p]
        print(f"🖐️  DTWM: attached TactileSpatialInjector(dim={_dim}, layers={_layers}, sigma={_sigma}); "
              f"patterns={trainable_parameter_patterns}; lora_skip_name={args.lora_skip_name}")
        if bool(training_config.get("tactile_input_constant", False)):
            print("🧊 DTWM vision-only baseline: training.tactile_input_constant=true -> the raw pressure is "
                  "zeroed before the tokenizer (same architecture and placement, no glove reading)")

    # ===== DTWM AR: prefix-conditioned chunked autoregressive training =====
    # Per batch, sample a prefix K (pixel frames, multiple of 4 so it covers latent frames 0..K/4):
    #   reference video (dwm-mode static slot) := [GT frames 0..K | zeros] (+ light context noise)
    #   hand condition := [observed skeleton 0..K | future stream K+1..K+ar_action_horizon | black]
    #                     (ar_action_horizon=0 in the paper: no future hand input)
    #   tactile := injected only on latent frames <= K/4 (causal touch)
    #   loss := 0 on observed latents, 1.0 on the next ar_bc_chunk latents, ar_loss_weight_far beyond.
    ar_prefix_choices = [int(x) for x in (training_config.get("ar_prefix_choices") or [])]
    ar_action_horizon = int(training_config.get("ar_action_horizon", 4))
    ar_loss_weight_far = float(training_config.get("ar_loss_weight_far", 0.2))
    ar_context_noise = float(training_config.get("ar_context_noise", 0.1))
    ar_hand_loss_weight = float(training_config.get("ar_hand_loss_weight", 0.0))
    # ar_bc_chunk: block-causal chunk size for the RANDOM-PREFIX (deployment/MPC) path. Lets random-prefix
    # teacher-forced training run WITH block-causal attention (the two were mutually exclusive before) and
    # sets the next-chunk loss width. 0 = legacy single-latent bidirectional behaviour.
    ar_bc_chunk = int(training_config.get("ar_bc_chunk", 0))
    # Clean-history AR (diffusion forcing): observed frames get sigma=ar_clean_history_sigma
    # (0.0 = exactly clean) instead of the global sigma, and timesteps become PER-TOKEN with the
    # observed frames pinned to t=0. Attention then sees full-information history at every noise
    # level — isomorphic to deployment, where the observation is clean, never noised.
    ar_clean_history = bool(training_config.get("ar_clean_history", False))
    ar_clean_history_sigma = float(training_config.get("ar_clean_history_sigma", 0.0))
    if ar_clean_history:
        print(f"🧊 DTWM: clean-history AR ON (observed sigma={ar_clean_history_sigma}, per-token timesteps)")
    if ar_prefix_choices:
        assert all(k % 4 == 0 and 4 <= k <= 44 for k in ar_prefix_choices), ar_prefix_choices
        if ar_bc_chunk > 0:
            transformer3d._dtwm_bc = {"chunk": ar_bc_chunk}
            print(f"🎬 DTWM deployment-AR: random-prefix + BLOCK-CAUSAL (chunk={ar_bc_chunk} latent), "
                  f"next-chunk loss width={ar_bc_chunk}")
        print(f"🎬 DTWM AR: prefix_choices={ar_prefix_choices} action_horizon={ar_action_horizon} "
              f"far_weight={ar_loss_weight_far} ctx_noise={ar_context_noise}")

    if args.train_mode == "lora":
        print("🔧 Setting up LoRA training...")
        
        # Create LoRA network using VideoX-Fun style
        network = create_network(
            1.0,
            args.rank,
            args.network_alpha,
            text_encoder,
            transformer3d,
            neuron_dropout=None,
            skip_name=args.lora_skip_name,
        )
        network.apply_to(text_encoder, transformer3d, args.train_text_encoder, True)

        # Enable additional trainable parameters (e.g., patch_embedding for channel expansion)
        non_lora_trainable_params = []
        _tac_named = []                      # tactile pathway, kept separate: see tactile_lr_scale
        if trainable_parameter_patterns:
            print(f"📋 Trainable parameter patterns: {trainable_parameter_patterns}")
            for name, param in transformer3d.named_parameters():
                for pattern in trainable_parameter_patterns:
                    if re.search(pattern, name):
                        param.requires_grad = True
                        if name.startswith(("tactile_injector.", "tactile_tokenizer.")):
                            _tac_named.append(param)
                        else:
                            non_lora_trainable_params.append(param)
                        print(f"   ✓ Enabled: {name}")
                        break
        
        trainable_params = list(filter(lambda p: p.requires_grad, network.parameters()))
        trainable_params_optim = network.prepare_optimizer_params(
            args.learning_rate / 2, args.learning_rate, args.learning_rate
        )
        # Add non-LoRA trainable params to optimizer
        if non_lora_trainable_params:
            trainable_params_optim.append({
                'params': non_lora_trainable_params,
                'lr': args.learning_rate,
            })
            trainable_params.extend(non_lora_trainable_params)
            print(f"📊 Non-LoRA trainable parameters: {sum(p.numel() for p in non_lora_trainable_params):,}")

        # The tactile pathway gets its own optimizer group; training.tactile_lr_scale (default 1.0,
        # the value used in the paper) scales its learning rate.
        if _tac_named:
            _tsc = float(training_config.get("tactile_lr_scale", 1.0))
            trainable_params_optim.append({
                'params': _tac_named,
                'lr': args.learning_rate * _tsc,
            })
            trainable_params.extend(_tac_named)
            print(f"📊 Tactile pathway parameters: {sum(p.numel() for p in _tac_named):,} "
                  f"at lr={args.learning_rate * _tsc:.2e} (scale {_tsc})")
        
        print(f"📊 LoRA trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Enable gradient checkpointing (from args or config)
    gradient_checkpointing = args.gradient_checkpointing or training_config.get("custom_settings", {}).get("gradient_checkpointing", False)
    if gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        print("✅ Gradient checkpointing enabled")

    # Enable TF32
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ========== Setup Optimizer ==========
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    # ========== Setup Learning Rate Scheduler ==========
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    if use_deepspeed_scheduler:
        lr_scheduler = DummyScheduler(
            name=training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            training_config.get("lr_scheduler", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=training_config.get("lr_warmup_steps", 0) * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=training_config.get("lr_num_cycles", 1),
        )

    # ========== Setup Dataset ==========
    print("🔧 Setting up dataset...")
    
    # Use is_wan2_2 already defined above (for width alignment to 32)
    align_width_to_32 = is_wan2_2
    if align_width_to_32:
        print("📐 WAN 2.2 detected: Aligning width to multiples of 32 for compatibility")
    
    # Setup dataset and dataloader
    # Read load_tensors from training.custom_settings (same as cogvideox pattern)
    load_tensors = training_config.get("custom_settings", {}).get("load_tensors", False)
    conditioning_mode = training_config.get("conditioning_mode", "dwm")
    static_video_source_mode = str(data_config.get("static_video_source_mode", "directory")).strip().lower()
    if load_tensors:
        raise ValueError("DTWM training encodes raw videos on the fly; set training.custom_settings.load_tensors=false.")

    dataset_init_kwargs = {
        "data_root": data_config["data_root"],
        "dataset_file": data_config["dataset_file"],
        "max_num_frames": data_config.get("max_num_frames", 49),
        "load_tensors": load_tensors,
        "random_flip": data_config.get("random_flip", False),
        "height_buckets": data_config.get("height_buckets", 480),
        "width_buckets": data_config.get("width_buckets", 720),
        "frame_buckets": data_config.get("frame_buckets", 49),
        "prompt_subdir": data_config.get("prompt_subdir", "prompts"),
        "prompt_embeds_subdir": data_config.get("prompt_embeds_subdir", "prompt_embeds_rewrite_wan"),
        "static_video_subdir": data_config.get("static_video_subdir", "videos_static"),
        "static_video_source_mode": static_video_source_mode,
        "hand_video_subdir": data_config.get("hand_video_subdir", "videos_hands"),
        "hand_video_future_subdir": data_config.get("hand_video_future_subdir"),
        "hand_video_latents_subdir": data_config.get("hand_video_latents_subdir", "hand_video_latents_wan"),
        "static_disparity_subdir": data_config.get("static_disparity_subdir"),
        "static_disparity_latents_subdir": data_config.get("static_disparity_latents_subdir"),
        "hand_disparity_subdir": data_config.get("hand_disparity_subdir"),
        "hand_disparity_latents_subdir": data_config.get("hand_disparity_latents_subdir"),
        "video_latents_subdir": data_config.get("video_latents_subdir", "video_latents_wan"),
        "static_video_latents_subdir": data_config.get("static_video_latents_subdir", "static_video_latents_wan"),
        "image_latents_subdir": data_config.get("image_latents_subdir", "image_latents_wan"),
        "fun_inp_i2v_latents_subdir": data_config.get("fun_inp_i2v_latents_subdir", "fun_inp_i2v_latents_wan"),
        "clip_image_embeds_subdir": data_config.get("clip_image_embeds_subdir", "clip_image_embeds"),
        "exclude_videos_file": data_config.get("exclude_videos_file"),
        "require_static_videos": conditioning_mode == "dwm",
        "align_width_to_32": align_width_to_32,
        "image_to_video": conditioning_mode == "i2v",
    }

    train_dataset = VideoDatasetWithConditionsAndResizing(**dataset_init_kwargs)
    def collate_fn(batch):
        def stack_or_none(key: str):
            if key in batch[0] and batch[0][key] is not None:
                first_tensor = batch[0][key]
                stacked_items = []
                for item in batch:
                    value = item.get(key)
                    if value is None:
                        stacked_items.append(torch.zeros_like(first_tensor))
                    else:
                        stacked_items.append(value)
                return torch.stack(stacked_items)
            return None

        return {
            "videos": torch.stack([item["video"] for item in batch]),
            "prompts": (
                torch.stack([item["prompt"] for item in batch])
                if isinstance(batch[0]["prompt"], torch.Tensor)
                else [item["prompt"] for item in batch]
            ),
            "hand_videos": stack_or_none("hand_videos"),
            "hand_videos_future": stack_or_none("hand_videos_future"),
            "static_videos": stack_or_none("static_videos"),
            "pressure": stack_or_none("pressure"),  # DTWM: (B, L, 2, 21, 21) or None
            "hand_centroids": stack_or_none("hand_centroids"),  # DTWM: (B, L, 2, 3) = (y_norm, x_norm, valid)
        }
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        drop_last=True,  # Drop last incomplete batch to ensure consistent batch size
        collate_fn=collate_fn,
        num_workers=data_config.get("dataloader_num_workers", 0),
        pin_memory=data_config.get("pin_memory", True),
    )

    # ========== Prepare with Accelerator ==========
    use_deepspeed_or_fsdp = (fsdp_stage != 0) or (zero_stage != 0)
    
    if use_deepspeed_or_fsdp:
        # DeepSpeed/FSDP: Attach network to transformer so DeepSpeed can find all params
        transformer3d.network = network
        transformer3d = transformer3d.to(weight_dtype)
        transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer3d, optimizer, train_dataloader, lr_scheduler
        )
        print(f"📦 Prepared transformer3d with attached LoRA network (DeepSpeed/FSDP mode, train_mode={args.train_mode})")
    else:
        # No DeepSpeed/FSDP: Just prepare network
        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            network, optimizer, train_dataloader, lr_scheduler
        )

    # Move models to device (skip transformer3d if already prepared with DeepSpeed/FSDP)
    # VAE and text encoder stay on device: videos and prompts are encoded during training
    vae.to(accelerator.device, dtype=weight_dtype)
    if not use_deepspeed_or_fsdp:
        transformer3d.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    # Recalculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ========== Initialize Trackers (WandB) ==========
    if accelerator.is_main_process:
        # Create run name: {date}_{name}_{mode}_{job_id}
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
        run_name = f"{date_suffix}_{exp_name}_{args.mode}_{slurm_job_id}"
        
        # Prepare tracker config
        tracker_config = {
            "experiment": {
                "name": exp_name,
                "date": exp_date if 'exp_date' in dir() else "unknown",
                "output_dir": args.output_dir,
            },
            "training": {
                "mode": args.train_mode,
                "learning_rate": args.learning_rate,
                "batch_size": args.train_batch_size,
                "max_train_steps": args.max_train_steps,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            },
            "model": {
                "base_model_name_or_path": args.pretrained_model_name_or_path,
                "condition_channels": args.condition_channels,
            },
            "system": {
                "num_gpus": accelerator.num_processes,
                "mixed_precision": args.mixed_precision,
            }
        }
        
        # Setup wandb init kwargs
        wandb_init_kwargs = {
            "name": run_name,
        }
        if args.tracker_entity_name:
            wandb_init_kwargs["entity"] = args.tracker_entity_name

        # Prevent frequent startup timeout in restricted/offline environments.
        wandb_init_timeout = float(custom_settings.get("wandb_init_timeout", 180))
        if HAS_WANDB:
            wandb_init_kwargs["settings"] = wandb.Settings(init_timeout=wandb_init_timeout)
        
        # Batch mode: WandB disabled (as in all DTWM runs)
        wandb_init_kwargs["mode"] = "disabled"
        print(f"🔧 Batch mode: WandB disabled")
        
        try:
            accelerator.init_trackers(
                project_name=args.tracker_project_name,
                config=tracker_config,
                init_kwargs={"wandb": wandb_init_kwargs}
            )
        except Exception as e:
            error_text = f"{type(e).__name__}: {e}"
            is_wandb_init_issue = (
                "wandb" in str(args.report_to).lower()
                and (
                    "timeout" in error_text.lower()
                    or "commerror" in error_text.lower()
                    or "timed out waiting for response" in error_text.lower()
                )
            )
            if not is_wandb_init_issue:
                raise

            logger.warning(
                "WandB init failed (%s). Falling back to disabled WandB tracking for this run.",
                error_text,
            )
            fallback_wandb_kwargs = {k: v for k, v in wandb_init_kwargs.items() if k != "settings"}
            fallback_wandb_kwargs["mode"] = "disabled"
            accelerator.init_trackers(
                project_name=args.tracker_project_name,
                config=tracker_config,
                init_kwargs={"wandb": fallback_wandb_kwargs},
            )
            print("🔧 WandB fallback applied: mode=disabled")
        
        print(f"🔗 Experiment initialized: {run_name}")

    # ========== Training Loop ==========
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    # Resume from checkpoint (supports both config-based and arg-based)
    checkpoint_to_resume = resume_from_checkpoint if 'resume_from_checkpoint' in dir() and resume_from_checkpoint else args.resume_from_checkpoint
    
    if checkpoint_to_resume:
        if checkpoint_to_resume != "latest":
            # Use the provided checkpoint path directly
            checkpoint_path = checkpoint_to_resume
            
            if not os.path.exists(checkpoint_path):
                accelerator.print(f"Checkpoint '{checkpoint_path}' does not exist. Starting a new training run.")
                checkpoint_to_resume = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
                
                # Try to load full accelerator state first, fallback to weight-only loading if it fails
                try:
                    accelerator.load_state(checkpoint_path)
                    accelerator.print(f"Successfully loaded full accelerator state from {checkpoint_path}")
                except Exception as e:
                    accelerator.print(f"Failed to load full accelerator state: {e}")
                    accelerator.print(f"Falling back to weight-only loading...")
                    
                    if args.train_mode == "lora":
                        # Load LoRA weights
                        lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                        if os.path.exists(lora_path):
                            state_dict = load_file(lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                            print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")
                        
                        # Load non-LoRA trainable weights (e.g., patch_embedding)
                        non_lora_path = os.path.join(checkpoint_path, "non_lora_weights.safetensors")
                        if os.path.exists(non_lora_path):
                            non_lora_state_dict = load_file(non_lora_path, device=str(accelerator.device))
                            m, u = accelerator.unwrap_model(transformer3d).load_state_dict(non_lora_state_dict, strict=False)
                            print(f"Loaded non-LoRA weights: missing {len(m)}, unexpected {len(u)}")
                
                # Extract global step from checkpoint name
                checkpoint_name = os.path.basename(checkpoint_path.rstrip('/'))
                if checkpoint_name.startswith("checkpoint-"):
                    global_step = extract_checkpoint_step(checkpoint_name)
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch
                else:
                    accelerator.print(f"Warning: Could not extract global step from checkpoint name '{checkpoint_name}'")
                    initial_global_step = 0
        else:
            # Get the most recent checkpoint from output_dir
            if not os.path.exists(args.output_dir):
                accelerator.print(f"Output directory '{args.output_dir}' does not exist. Starting a new training run.")
                checkpoint_to_resume = None
                initial_global_step = 0
            else:
                dirs = list_checkpoint_dirs(args.output_dir)
                if len(dirs) == 0:
                    accelerator.print("No checkpoint directories found. Starting a new training run.")
                    checkpoint_to_resume = None
                    initial_global_step = 0
                else:
                    latest_checkpoint = dirs[-1]
                    checkpoint_path = os.path.join(args.output_dir, latest_checkpoint)
                    accelerator.print(f"Resuming from latest checkpoint {checkpoint_path}")
                    
                    # Try to load full accelerator state first, fallback to weight-only loading if it fails
                    try:
                        accelerator.load_state(checkpoint_path)
                        accelerator.print(f"Successfully loaded full accelerator state from {checkpoint_path}")
                    except Exception as e:
                        accelerator.print(f"Failed to load full accelerator state: {e}")
                        accelerator.print(f"Falling back to weight-only loading...")
                        
                        if args.train_mode == "lora":
                            lora_path = os.path.join(checkpoint_path, "lora_diffusion_pytorch_model.safetensors")
                            if os.path.exists(lora_path):
                                state_dict = load_file(lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(network).load_state_dict(state_dict, strict=False)
                                print(f"Loaded LoRA: missing {len(m)}, unexpected {len(u)}")
                            
                            # Load non-LoRA trainable weights (e.g., patch_embedding)
                            non_lora_path = os.path.join(checkpoint_path, "non_lora_weights.safetensors")
                            if os.path.exists(non_lora_path):
                                non_lora_state_dict = load_file(non_lora_path, device=str(accelerator.device))
                                m, u = accelerator.unwrap_model(transformer3d).load_state_dict(non_lora_state_dict, strict=False)
                                print(f"Loaded non-LoRA weights: missing {len(m)}, unexpected {len(u)}")
                    
                    global_step = extract_checkpoint_step(latest_checkpoint)
                    initial_global_step = global_step
                    first_epoch = global_step // num_update_steps_per_epoch

    # Get step intervals from config (with args as fallback)
    checkpointing_steps = training_config.get("custom_settings", {}).get("checkpointing_steps", args.checkpointing_steps)
    
    logger.info(f"📊 Step intervals: checkpointing={checkpointing_steps}")
    logger.info(
        "📦 Full-state checkpoint retention: keep_full_state_checkpoints=%s",
        keep_full_state_checkpoints,
    )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Initialize DiscreteSampling for uniform timestep sampling
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # Get patch_size from transformer config (used for seq_len calculation)
    patch_size = accelerator.unwrap_model(transformer3d).config.patch_size

    # VAE mini-batch encoding for memory efficiency
    vae_mini_batch = getattr(args, 'vae_mini_batch', 32)
    
    def _batch_encode_vae(pixel_values):
        """Encode pixel values to latents in mini-batches to save memory."""
        bs = vae_mini_batch
        new_latents = []
        for i in range(0, pixel_values.shape[0], bs):
            pixel_values_bs = pixel_values[i:i + bs]
            latents_bs = vae.encode(pixel_values_bs)[0].sample()
            new_latents.append(latents_bs)
        return torch.cat(new_latents, dim=0)

    requires_hand_videos = transformer_type == "WanTransformer3DModelWithConcat"

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [network]

            with accelerator.accumulate(models_to_accumulate):
                # Get batch data
                videos = batch["videos"].to(accelerator.device, dtype=weight_dtype)
                prompts = batch["prompts"]
                hand_videos = batch.get("hand_videos")
                static_videos = batch.get("static_videos")

                # ===== Encode videos to latents =====
                # Raw videos: [B, F, C, H, W] -> encode to latents
                videos_bcfhw = rearrange(videos, "b f c h w -> b c f h w")
                with torch.no_grad():
                    latents = _batch_encode_vae(videos_bcfhw)

                # ===== DTWM AR line: sample prefix K, rebuild reference + hand conditions =====
                ar_K = ar_m = None
                if ar_prefix_choices and conditioning_mode == "dwm" \
                        and videos_bcfhw is not None:
                    ar_K = random.choice(ar_prefix_choices)
                    ar_m = ar_K // 4                       # last OBSERVED latent-frame index
                    # (1) reference video replaces the static scene: true history 0..K, ZEROS after
                    #     (InP-native "unknown future"; broadcasting frame K instead teaches a copy-the-
                    #     reference shortcut). Light noise augmentation on the observed part.
                    _ref = videos_bcfhw.clone()
                    _ref[:, :, ar_K + 1:, :, :] = 0.0
                    if ar_context_noise > 0:
                        _s = random.uniform(0.0, ar_context_noise)
                        _ref[:, :, :ar_K + 1] = _ref[:, :, :ar_K + 1] + _s * torch.randn_like(_ref[:, :, :ar_K + 1])
                    static_videos = rearrange(_ref, "b c f h w -> b f c h w")
                    # (2) hand condition: observed skeleton 0..K + future stream for ar_action_horizon
                    #     frames + black. The future stream must be present: it is what blanks the
                    #     hand condition after K (with ar_action_horizon=0 nothing of it is copied).
                    hand_videos_future = batch.get("hand_videos_future")
                    if hand_videos is not None and hand_videos_future is not None:
                        _hv = hand_videos.clone()
                        _hv[:, ar_K + 1:] = 0
                        _end = min(ar_K + 1 + ar_action_horizon, _hv.shape[1])
                        _hv[:, ar_K + 1:_end] = hand_videos_future[:, ar_K + 1:_end].to(_hv.dtype)
                        hand_videos = _hv

                # ===== Encode condition videos =====
                use_hand_condition = requires_hand_videos and conditioning_mode == "dwm"
                if use_hand_condition and hand_videos is not None:
                    # Raw hand videos: encode to latents
                    hand_videos = hand_videos.to(accelerator.device, dtype=weight_dtype)
                    hand_videos = rearrange(hand_videos, "b f c h w -> b c f h w")
                    with torch.no_grad():
                        hand_latents = _batch_encode_vae(hand_videos)
                else:
                    hand_latents = None

                hand_condition_tensors = []
                if hand_latents is not None:
                    hand_condition_tensors.append(hand_latents)

                bsz, latent_channels, latent_frames, latent_height, latent_width = latents.size()
                static_latents = None

                if conditioning_mode == "dwm":
                    if static_videos is None:
                        if static_video_source_mode != "copy_first_frame":
                            raise ValueError("conditioning_mode='dwm' requires static_videos in batch")
                        static_videos = videos[:, 0:1, :, :, :].repeat(1, videos.shape[1], 1, 1, 1)

                    static_videos = static_videos.to(accelerator.device, dtype=weight_dtype)
                    static_videos = rearrange(static_videos, "b f c h w -> b c f h w")
                    with torch.no_grad():
                        static_latents = _batch_encode_vae(static_videos)

                    mask_latents = torch.ones(
                        bsz, 4, latent_frames, latent_height, latent_width,
                        device=accelerator.device, dtype=weight_dtype
                    )
                    if ar_m is not None:
                        # AR line: prefix-inpainting semantics — reference is VALID only on the observed
                        # latents 0..ar_m; the future is unknown (mask 0, reference zeroed above).
                        mask_latents[:, :, ar_m + 1:] = 0.0
                    inpaint_latents = torch.cat([mask_latents, static_latents], dim=1)
                else:
                    raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")

                expected_condition_channels = int(getattr(args, "condition_channels", 0) or 0)
                condition_tensors = []
                condition_tensors.extend(hand_condition_tensors)
                condition_latents = torch.cat(condition_tensors, dim=1) if condition_tensors else None

                if expected_condition_channels > 0:
                    if condition_latents is None:
                        raise ValueError(
                            f"transformer.condition_channels={expected_condition_channels} "
                            "but no condition latents were loaded."
                        )
                    actual_condition_channels = int(condition_latents.shape[1])
                    if actual_condition_channels != expected_condition_channels:
                        raise ValueError(
                            "Condition channel mismatch: "
                            f"expected {expected_condition_channels}, got {actual_condition_channels}. "
                            "Check hand/static disparity latent subdir settings."
                        )

                # Encode prompts (prompt_mode=normal).
                with torch.no_grad():
                    if isinstance(prompts, list):
                        prompt_ids = tokenizer(
                            prompts,
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids.to(latents.device)
                        prompt_attention_mask = prompt_ids.attention_mask.to(latents.device)
                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
                        prompt_embeds = [u[:v].to(weight_dtype) for u, v in zip(prompt_embeds, seq_lens)]
                    else:
                        prompt_embeds = prompts.to(dtype=weight_dtype)

                # Sample noise
                bsz, channel, num_frames, height, width = latents.size()
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                # Sample timesteps: uniform DiscreteSampling (supports distributed training)
                indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                indices = indices.long().cpu()

                timesteps = noise_scheduler.timesteps[indices.cpu()].to(device=latents.device)

                # Get sigmas and create noisy latents (Flow Matching)
                sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=accelerator.device)
                # ===== DTWM clean history: per-frame sigma, observed frames stay (near-)clean =====
                timesteps_tok = None
                if ar_clean_history and ar_m is not None:
                    _F = latents.shape[2]
                    _sig_f = sigmas.expand(-1, -1, _F, -1, -1).clone()          # [B,1,F,1,1]
                    _sig_f[:, :, :ar_m + 1] = ar_clean_history_sigma            # history: clean
                    noisy_latents = (1.0 - _sig_f) * latents + _sig_f * noise
                    # per-token timesteps [B, seq_len]; token layout is frame-major (frame = idx//(Hp*Wp)),
                    # matching rope_apply / the block-causal mask.
                    _Hp = latents.shape[3] // patch_size[1]
                    _Wp = latents.shape[4] // patch_size[2]
                    _tpf = _Hp * _Wp
                    _t_f = timesteps.view(-1, 1).expand(-1, _F).clone().to(latents.device)   # [B,F]
                    _t_f[:, :ar_m + 1] = 0
                    timesteps_tok = _t_f.repeat_interleave(_tpf, dim=1)          # [B, F*Hp*Wp]
                else:
                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Flow Matching target
                target = noise - latents

                # Calculate seq_len for transformer (based on patched dimensions)
                seq_len = math.ceil(
                    (height / patch_size[1]) * (width / patch_size[2]) * (num_frames / patch_size[0])
                )

                # ===== DTWM: spatial tactile inputs (per-frame embedding + latent hand centroids). =====
                # No sequence concat, no seq_len change, no tactile loss; causal: observed frames only.
                if tactile_spatial:
                    _um = accelerator.unwrap_model(transformer3d)
                    if batch.get("pressure") is not None and batch.get("hand_centroids") is not None:
                        from dtwm.models.tactile_tokens import spatial_reduce
                        _tk = _um.tactile_tokenizer
                        _P = batch["pressure"].to(accelerator.device, dtype=torch.float32)
                        # VISION-ONLY BASELINE (training.tactile_input_constant=true; the paper's
                        # capacity-matched control). The RAW pressure is zeroed before the tokenizer, which
                        # is exactly what inference.py's --tactile_constant does, so a model trained this way
                        # is evaluated under the identical substitution. The footprint still goes to the
                        # projected hand centroid, so this model keeps every part of the mechanism except
                        # the glove reading.
                        if bool(training_config.get("tactile_input_constant", False)):
                            _P = torch.zeros_like(_P)
                            if global_step < 2 and accelerator.is_main_process:
                                print("[CONST] vision-only baseline: raw pressure zeroed before the "
                                      "tokenizer; footprint placement unchanged", flush=True)
                        _sl, _sr = _tk.gather(_P)
                        _obs = torch.ones(_sl.shape[1], dtype=torch.bool, device=_sl.device)
                        _Ef = _tk.frame_embeddings(_sl, _sr, _obs)           # (B,L,2,dim)
                        _C = batch["hand_centroids"].to(accelerator.device, dtype=torch.float32)  # (B,L,2,3)
                        _E, _cen, _val = spatial_reduce(_Ef, _C)
                        if ar_m is not None:
                            # CAUSAL touch — inject only on observed latent frames 0..ar_m.
                            _val = _val.clone(); _val[:, ar_m + 1:] = 0
                        _um._dtwm_tac_spatial = {"E": _E.to(weight_dtype), "centroids": _cen, "valid": _val}
                    else:
                        _um._dtwm_tac_spatial = None

                # Forward pass
                # x: noisy_latents [B, 16, F, H, W]
                # y: inpaint_latents [B, 17, F, H, W] (mask + static)
                # condition_latents: concatenated extra conditions [B, C_cond, F, H, W]
                #
                # WanTransformer3DModelWithConcat.forward handles conversion to list format
                # and concatenation of condition_latents
                with torch.amp.autocast(device_type='cuda', dtype=weight_dtype):
                    # Concat model: with condition_latents
                    _t_in = timesteps
                    if timesteps_tok is not None:
                        _t_in = timesteps_tok
                        if _t_in.shape[1] != seq_len:      # pad/trim to the transformer's seq_len
                            if _t_in.shape[1] < seq_len:
                                _t_in = torch.cat([_t_in, _t_in[:, -1:].expand(-1, seq_len - _t_in.shape[1])], dim=1)
                            else:
                                _t_in = _t_in[:, :seq_len]
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        t=_t_in,
                        context=prompt_embeds,
                        seq_len=seq_len,
                        y=inpaint_latents,
                        condition_latents=condition_latents,
                    )

                # Compute loss with weighting
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                if ar_m is not None:
                    # random-prefix (deployment/MPC): 0 on observed, 1.0 on the NEXT CHUNK
                    # [ar_m+1 .. ar_m+ar_bc_chunk] (deployment step = predict one chunk from GT history),
                    # far-weight beyond. ar_bc_chunk falls back to 1 for the legacy single-latent target.
                    _nc = max(1, ar_bc_chunk)
                    _w = torch.full((1, 1, latents.shape[2], 1, 1), ar_loss_weight_far,
                                    device=weighting.device, dtype=weighting.dtype)
                    _w[:, :, :ar_m + 1] = 0.0
                    _w[:, :, ar_m + 1:ar_m + 1 + _nc] = 1.0
                    weighting = weighting * _w
                # AR: spatially up-weight the loss on the hand region so the model must render the
                # hands where they are (not hallucinate them).
                if ar_hand_loss_weight > 0 and batch.get("hand_centroids") is not None:
                    from dtwm.models.tactile_tokens import hand_region_weight
                    _hc = batch["hand_centroids"].to(latents.device, dtype=torch.float32)  # (B,L,2,3)
                    _hw = hand_region_weight(_hc, latents.shape[2], latents.shape[3], latents.shape[4],
                                             sigma=float(training_config.get("ar_hand_sigma", 4.0)))
                    weighting = weighting * (1.0 + ar_hand_loss_weight * _hw.to(weighting.dtype))
                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()

                # DTWM: clear the per-forward spatial tactile state.
                if tactile_spatial:
                    accelerator.unwrap_model(transformer3d)._dtwm_tac_spatial = None

                # DTWM: non-finite-loss guard (a single bf16 spike can NaN a run). All ranks agree via
                # MAX all-reduce, then skip this micro-batch TOGETHER (identical collective pattern on
                # every rank -> no DeepSpeed desync/hang).
                _bad = torch.tensor(0.0 if bool(torch.isfinite(loss)) else 1.0, device=loss.device)
                if accelerator.num_processes > 1:
                    torch.distributed.all_reduce(_bad, op=torch.distributed.ReduceOp.MAX)
                if _bad.item() > 0:
                    # Once the weights themselves go non-finite every later loss is non-finite too,
                    # so this branch stops being a spike filter and becomes an infinite no-op loop:
                    # the process stays alive, the log keeps growing, global_step never advances and
                    # no checkpoint is ever written. A run that cannot make progress must die.
                    _run = getattr(args, "_dtwm_consec_nonfinite", 0) + 1
                    args._dtwm_consec_nonfinite = _run
                    logger.warning(f"⚠️  non-finite loss at global_step≈{global_step} — skipping "
                                   f"micro-batch on all ranks (consecutive: {_run})")
                    if _run >= 50:
                        logger.error(f"💀 {_run} consecutive non-finite losses at global_step≈"
                                     f"{global_step}: the weights are corrupt and this run cannot "
                                     f"progress. Aborting so the failure is visible instead of "
                                     f"burning GPUs silently. Resume from the last checkpoint.")
                        raise RuntimeError(
                            f"training diverged: {_run} consecutive non-finite losses "
                            f"at global_step≈{global_step}")
                    del loss
                    continue
                else:
                    args._dtwm_consec_nonfinite = 0

                # Gather losses
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    _clip_targets = list(network.parameters())

                    # NOTE: under DeepSpeed the clipping is done inside the engine and
                    # accelerator.clip_grad_norm_ returns None, so a gradient-norm test here cannot
                    # work -- it was tried and logged "returned None", i.e. it guarded nothing.
                    # max_grad_norm in the yaml is applied by DeepSpeed, not by this call.
                    # The protection that does work is the consecutive-non-finite abort below.
                    _gnorm = accelerator.clip_grad_norm_(_clip_targets, args.max_grad_norm)
                    if not getattr(args, "_dtwm_gnorm_logged", False):
                        args._dtwm_gnorm_logged = True
                        logger.info(f"🛡️  clip_grad_norm_ returned {_gnorm!r} "
                                    f"(None means DeepSpeed owns clipping)")

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Save checkpoint (step interval)
                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        # Remove old checkpoint directories before saving a new one.
                        if args.checkpoints_total_limit is not None:
                            checkpoints = list_checkpoint_dirs(args.output_dir)
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                if accelerator.is_main_process:
                                    for removing_checkpoint in checkpoints[:num_to_remove]:
                                        removing_path = os.path.join(args.output_dir, removing_checkpoint)
                                        logger.info(f"Removing old checkpoint: {removing_path}")
                                        shutil.rmtree(removing_path)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_training_checkpoint(
                            save_path,
                            args.output_dir,
                            accelerator,
                            keep_full_state_checkpoints,
                            lambda path: save_train_mode_weights(
                                path,
                                args,
                                network,
                                weight_dtype,
                                accelerator,
                                transformer3d,
                                trainable_parameter_patterns,
                            ),
                            logger,
                        )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        save_training_checkpoint(
            save_path,
            args.output_dir,
            accelerator,
            keep_full_state_checkpoints,
            lambda path: save_train_mode_weights(
                path,
                args,
                network,
                weight_dtype,
                accelerator,
                transformer3d,
                trainable_parameter_patterns,
            ),
            logger,
        )

    if accelerator.is_main_process:
        logger.info(f"Training completed. Final checkpoint saved to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
