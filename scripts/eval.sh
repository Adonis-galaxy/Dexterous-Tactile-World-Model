#!/usr/bin/env bash
# DTWM evaluation with the paper settings: all 36 predicted frames after the observed prefix
# (K=12, frames 13..48), checkpoint step 4825.
#
#   OUTPUTS_ROOT=outputs/260723 RUN=<touch run dir> DATA_ROOT=/path/to/EgoTouch_dwm_full bash scripts/eval.sh
#   OUTPUTS_ROOT=... RUN=<touch run> VISION_RUN=<vision-only run> DATA_ROOT=... bash scripts/eval.sh
#
# 1. eval/score_all_metrics.py     per-clip LPIPS / local LPIPS / PSNR / SSIM / copy-last-frame floor
#                                  for RUN (and for VISION_RUN when given)
# 2. eval/score_hand_consistency.py paired hand-behaviour metrics RUN (touch) vs VISION_RUN (vision),
#                                  only when VISION_RUN is given
#
# Both read <OUTPUTS_ROOT>/<run>/checkpoint-<CKPT_STEP>/<SUBDIR>/**/*_generated.mp4, i.e. the files
# written by scripts/infer.sh (SUBDIR=eval is its default output directory inside the checkpoint).
# LPIPS runs on CPU unless CUDA_VISIBLE_DEVICES is set.
#
# The script changes into the repository root: relative paths are interpreted relative to it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OUTPUTS_ROOT:?set OUTPUTS_ROOT to the directory holding the run directories}"
: "${RUN:?set RUN to the (touch) run directory name}"
: "${DATA_ROOT:?set DATA_ROOT to the EgoTouch dataset root (EgoTouch_dwm_full)}"
VISION_RUN=${VISION_RUN:-}
LIST=${LIST:-${REPO}/splits/heldout_3000_49f.txt}
SUBDIR=${SUBDIR:-eval}
CKPT_STEP=${CKPT_STEP:-4825}
RESULTS=${RESULTS:-${REPO}/eval_results}
mkdir -p "$RESULTS"

cd "$REPO"
for R in "$RUN" ${VISION_RUN:+"$VISION_RUN"}; do
  python eval/score_all_metrics.py \
    --runs "$R" --K 12 --horizon 36 --ckpt_step "$CKPT_STEP" \
    --subdir "$SUBDIR" --outputs_root "$OUTPUTS_ROOT" \
    --val_file "$LIST" --gt_root "$DATA_ROOT" \
    --out "$RESULTS/metrics_${R}.json"
done

if [[ -n "$VISION_RUN" ]]; then
  python eval/score_hand_consistency.py \
    --touch_run "$RUN" --vision_run "$VISION_RUN" \
    --subdir "$SUBDIR" --outputs_root "$OUTPUTS_ROOT" \
    --ckpt_step "$CKPT_STEP" --vision_ckpt_step "$CKPT_STEP" \
    --window 13 49 \
    --val_file "$LIST" --gt_root "$DATA_ROOT" \
    --label_a touch --label_b vision \
    --out "$RESULTS/hand_${RUN}_vs_${VISION_RUN}.json"
fi
