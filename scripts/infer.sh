#!/usr/bin/env bash
# DTWM inference with the paper settings: frames 13..48 are predicted in one pass from the observed
# frames 0..12 (--ar_prefix 12), block-causal chunks of 4 latent frames, clean pinned history, causal
# spatial tactile injection, 30 Euler steps, CFG 6.0, seed 0.
#
#   CKPT=outputs/260723/<run>/checkpoint-4825 BACKBONE=/path/to/Wan2.2-Fun-5B-InP \
#   DATA_ROOT=/path/to/EgoTouch_dwm_full LIST=splits/heldout_3000_49f.txt bash scripts/infer.sh
#
# Sharding: NUM_SPLITS=N SPLIT_ID=i processes the i-th of N contiguous shards (run one process per GPU,
# e.g. CUDA_VISIBLE_DEVICES=i). Outputs: $OUT/<collection>_<sample>_<stem>_generated.mp4 (+ comparison
# mp4s); existing outputs are skipped, so an interrupted shard can simply be restarted.
#
# VISION_ONLY=true      evaluate a vision-only baseline checkpoint (trained with VISION_ONLY=true):
#                       the raw pressure is zeroed before the tokenizer (--tactile_constant).
# GATE_SCALE=0          disable the tactile pathway at inference on the same checkpoint
#                       (--tactile_gate_scale, default 1.0 = as trained).
# TAC_SIGMA             tactile footprint std at inference. Unset = 3.0, the value used for the paper's
#                       evaluation (training used 1.8).
#
# The script changes into the repository root: relative paths are interpreted relative to it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CKPT:?set CKPT to a checkpoint directory (.../checkpoint-<step>)}"
: "${BACKBONE:?set BACKBONE to the Wan2.2-Fun-5B-InP model directory}"
: "${DATA_ROOT:?set DATA_ROOT to the EgoTouch dataset root (EgoTouch_dwm_full)}"
LIST=${LIST:-${REPO}/splits/heldout_3000_49f.txt}
CONFIG=${CONFIG:-${REPO}/training/wan/configs/examples/dwm_wan22_5b_lora.yaml}
OUT=${OUT:-${CKPT}/eval}
NUM_SPLITS=${NUM_SPLITS:-1}
SPLIT_ID=${SPLIT_ID:-0}
VISION_ONLY=${VISION_ONLY:-false}
GATE_SCALE=${GATE_SCALE:-1.0}

EXTRA=()
[[ "$VISION_ONLY" == "true" ]] && EXTRA+=( --tactile_constant )
[[ -n "${TAC_SIGMA:-}" ]] && EXTRA+=( --tactile_spatial_sigma "$TAC_SIGMA" )

cd "$REPO"
export PYTHONPATH="${REPO}:${REPO}/src:${PYTHONPATH:-}"
# Attention backend: the paper runs used PyTorch SDPA (flash-attn / sageattention were not installed).
# Forcing it keeps the numerics independent of which optional attention kernels are installed.
export VIDEOX_ATTENTION_TYPE=${VIDEOX_ATTENTION_TYPE:-TORCH_SDPA}

python training/wan/inference.py \
  --experiment_config "$CONFIG" \
  --base_model_path "$BACKBONE" \
  --checkpoint_path "$CKPT" \
  --data_root "$DATA_ROOT" \
  --dataset_file "$LIST" \
  --num_splits "$NUM_SPLITS" --split_id "$SPLIT_ID" \
  --output_dir "$OUT" \
  --tactile_spatial --tactile_forced \
  --tactile_gate_scale "$GATE_SCALE" \
  --ar_chunk 4 --ar_prefix 12 --ar_action_horizon 0 \
  --pin_observed --clean_history \
  "${EXTRA[@]}" \
  --override data.hand_video_subdir=videos_hands_wilor \
             data.hand_video_future_subdir=videos_hands_palm6dof \
             data.height_buckets=384 data.width_buckets=512 \
             data.prompt_subdir=prompts_rewrite \
             data.validation_resize_mode=stretch \
  --height 384 --width 512 \
  --num_inference_steps 30 \
  --max_samples 100000 \
  --seed 0 \
  --no_shuffle_dataset \
  --skip_existing
