#!/usr/bin/env bash
# DTWM training: block-causal prediction of the frames after a random-length observation on Wan2.2-Fun-5B-InP
# (LoRA) with spatially grounded tactile injection. Defaults reproduce the paper run.
#
# Per training sample a random prefix K in {4,8,...,44} (pixel frames, = latent frames 0..K/4):
#   reference (DWM static slot) = [GT frames 0..K | zeros] + light context noise on the observed part
#   hand condition              = [observed WiLoR skeleton 0..K | black]   (AR_HORIZON=0: no future hand)
#   tactile                     = glove pressure injected ONLY on observed latents (causal touch)
#   attention                   = block-causal over chunks of AR_BC_CHUNK latent frames
#   history                     = observed latents kept clean (sigma=CLEAN_SIG, per-token timestep 0)
#   loss                        = 0 on observed / 1.0 on the next AR_BC_CHUNK latents / AR_FAR_W beyond,
#                                 hand region up-weighted by AR_HAND_W (Gaussian, HAND_SIGMA latent cells)
#
# Usage (from anywhere):
#   BACKBONE=/path/to/Wan2.2-Fun-5B-InP DATA_ROOT=/path/to/EgoTouch_dwm_full bash scripts/train.sh
#   VISION_ONLY=true BACKBONE=... DATA_ROOT=... bash scripts/train.sh      # vision-only baseline
#
# VISION_ONLY=true trains the paper's vision-only baseline: the identical architecture (tokenizer +
# injector, same footprint placement) with the raw pressure zeroed before the tokenizer
# (training.tactile_input_constant=true). Evaluate it with inference --tactile_constant.
#
# Note on batch size: accelerate_configs/deepspeed.yaml sets gradient_accumulation_steps: 1, which
# accelerate does not overwrite, so DeepSpeed takes an optimizer step after every micro-batch.
# GRAD_ACCUM only sets how many micro-batches make one `global_step` (checkpoint/LR-schedule unit).
# With the paper setting NGPU=2 GRAD_ACCUM=4 BATCH=1 an optimizer update therefore sees 2 clips and
# MAX_STEPS=4825 global steps are 19,300 updates (one pass over the 38,600 training clips).
# Keep this configuration to reproduce the paper.
#
# The script changes into the repository root: relative paths are interpreted relative to it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DWM=${DWM:-$REPO}                               # repository root (training/, src/, configs/)
CONFIG="${CONFIG:-${DWM}/training/wan/configs/examples/dwm_wan22_5b_lora.yaml}"
TRAIN="${DWM}/training/wan/train_dwm_wan.py"
DS_CFG="${DS_CFG:-${DWM}/accelerate_configs/deepspeed.yaml}"

: "${BACKBONE:?set BACKBONE to the Wan2.2-Fun-5B-InP model directory}"
: "${DATA_ROOT:?set DATA_ROOT to the EgoTouch dataset root (EgoTouch_dwm_full)}"
TRAIN_TXT=${TRAIN_TXT:-${DWM}/splits/egotouch_train_v2.txt}   # clip list; entries relative to DATA_ROOT

NGPU=${NGPU:-2}
BATCH=${BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-4825}
CKPT_EVERY=${CKPT_EVERY:-1206}
LR=${LR:-3e-5}
SCHED=${SCHED:-constant_with_warmup}           # diffusers get_scheduler name (warmup steps from the config)

HEIGHT=${HEIGHT:-384}                           # Wan2.2 needs H,W % 32 == 0
WIDTH=${WIDTH:-512}
FRAMES=${FRAMES:-49}
PROMPT_SUBDIR=${PROMPT_SUBDIR:-prompts_rewrite}
SUBDIR=${SUBDIR:-videos_hands_wilor}            # observed hand conditioning (rendered WiLoR skeleton)
FUTURE_SUBDIR=${FUTURE_SUBDIR:-videos_hands_palm6dof}  # second hand stream; must exist, see below

SIGMA=${SIGMA:-1.8}                             # tactile footprint std (latent cells)
VISION_ONLY=${VISION_ONLY:-false}               # true = vision-only baseline (pressure zeroed)

AR_CHOICES=${AR_CHOICES:-"[4,8,12,16,20,24,28,32,36,40,44]"}   # prefix lengths K (pixel frames)
AR_BC_CHUNK=${AR_BC_CHUNK:-4}                   # block-causal chunk (latent frames) = next-chunk loss width
AR_FAR_W=${AR_FAR_W:-0.2}                       # loss weight beyond the next chunk
AR_CTX_NOISE=${AR_CTX_NOISE:-0.1}               # max noise std on the observed reference frames
AR_HAND_W=${AR_HAND_W:-5.0}                     # hand-region loss up-weight
HAND_SIGMA=${HAND_SIGMA:-2.4}                   # hand-region Gaussian std (latent cells)
CLEAN_HIST=${CLEAN_HIST:-true}                  # observed latents get sigma=CLEAN_SIG and timestep 0
CLEAN_SIG=${CLEAN_SIG:-0.0}
# The future hand stream is spliced in for AR_HORIZON frames after K. AR_HORIZON is fixed to 0 (no
# future hand input), but the stream must still be configured: splicing it is what blanks the hand
# condition after K. Without FUTURE_SUBDIR the full skeleton (including the future) would be visible.
AR_HORIZON=0

SEED=${SEED:-42}
APPLY_SEED=${APPLY_SEED:-false}                 # true passes --seed SEED; the paper runs did not set a seed
PORT=${PORT:-29500}                             # distinct per concurrent run
RESUME=${RESUME:-}                              # checkpoint dir or "latest"

if [[ -z "${EXP:-}" ]]; then
  if [[ "$VISION_ONLY" == "true" ]]; then EXP="dtwm_vision_only"; else EXP="dtwm_touch"; fi
fi

cd "$DWM"
export PYTHONPATH="${DWM}:${DWM}/src:${PYTHONPATH:-}"
# Attention backend: the paper runs used PyTorch SDPA (flash-attn / sageattention were not installed).
# Forcing it keeps the numerics independent of which optional attention kernels are installed.
export VIDEOX_ATTENTION_TYPE=${VIDEOX_ATTENTION_TYPE:-TORCH_SDPA}
# Transport settings used for the paper runs (2 GPUs on one node without working P2P).
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
# All ranks write into the same output directory: <repo>/outputs/<date>/<EXP>_batch_<BATCH_TIMESTAMP>
# (DTWM_OUTPUT_ROOT overrides <repo>/outputs). Pin BATCH_TIMESTAMP to resume into an existing run dir.
export BATCH_TIMESTAMP=${BATCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
echo "[train] EXP=$EXP BATCH_TIMESTAMP=$BATCH_TIMESTAMP NGPU=$NGPU GRAD_ACCUM=$GRAD_ACCUM VISION_ONLY=$VISION_ONLY"

RESUME_ARG=()
[[ -n "$RESUME" ]] && RESUME_ARG=( --resume_from_checkpoint "$RESUME" )
SEED_ARG=()
[[ "$APPLY_SEED" == "true" ]] && SEED_ARG=( --seed "$SEED" )

accelerate launch --config_file "$DS_CFG" --num_processes "$NGPU" --num_machines 1 \
  --main_process_port "$PORT" \
  "$TRAIN" \
  --experiment_config "$CONFIG" \
  --mode batch \
  "${RESUME_ARG[@]}" \
  "${SEED_ARG[@]}" \
  --override \
    "experiment.name=${EXP}" \
    "pipeline.base_model_name_or_path=${BACKBONE}" \
    "data.data_root=${DATA_ROOT}" \
    "data.dataset_file=[${TRAIN_TXT}]" \
    "data.height_buckets=${HEIGHT}" \
    "data.width_buckets=${WIDTH}" \
    "data.prompt_subdir=${PROMPT_SUBDIR}" \
    "data.frame_buckets=${FRAMES}" \
    "data.max_num_frames=${FRAMES}" \
    "training.custom_settings.max_num_frames=${FRAMES}" \
    "data.hand_video_subdir=${SUBDIR}" \
    "data.hand_video_future_subdir=${FUTURE_SUBDIR}" \
    "training.batch_size=${BATCH}" \
    "training.gradient_accumulation_steps=${GRAD_ACCUM}" \
    "training.learning_rate=${LR}" \
    "training.lr_scheduler=${SCHED}" \
    "training.custom_settings.checkpointing_steps=${CKPT_EVERY}" \
    "training.max_train_steps=${MAX_STEPS}" \
    "training.custom_settings.load_tensors=false" \
    "training.tactile_spatial=true" \
    "training.tactile_spatial_sigma=${SIGMA}" \
    "training.tactile_forced=true" \
    "training.tactile_input_constant=${VISION_ONLY}" \
    "training.ar_prefix_choices=${AR_CHOICES}" \
    "training.ar_bc_chunk=${AR_BC_CHUNK}" \
    "training.ar_clean_history=${CLEAN_HIST}" \
    "training.ar_clean_history_sigma=${CLEAN_SIG}" \
    "training.ar_action_horizon=${AR_HORIZON}" \
    "training.ar_loss_weight_far=${AR_FAR_W}" \
    "training.ar_context_noise=${AR_CTX_NOISE}" \
    "training.ar_hand_loss_weight=${AR_HAND_W}" \
    "training.ar_hand_sigma=${HAND_SIGMA}"
