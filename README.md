# Dexterous Tactile World Model

**Ziyao Zeng**<sup>1</sup>, **Xiatao Sun**<sup>1</sup>, **Hao Wang**<sup>2</sup>, **Yueyang Pan**<sup>3</sup>, **Zhengxiang Yu**<sup>4</sup>,
**Fengyu Yang**<sup>1</sup>, **Tianyu Liu**<sup>1</sup>, **Zhiwen Fan**<sup>2</sup>, **Daniel Rakita**<sup>1</sup>

<sup>1</sup>Yale University &nbsp; <sup>2</sup>Texas A&M University &nbsp; <sup>3</sup>University of California, Los Angeles &nbsp; <sup>4</sup>University of Washington

[Paper (arXiv)](https://arxiv.org/abs/2609.34286) · [Project page](https://adonis-galaxy.github.io/dtwm-project-page/)

DTWM predicts future egocentric human manipulation videos with whole-hand touch. It conditions a pretrained video
diffusion transformer (Wan2.2-Fun-5B-InP) on each hand's tactile reading, added to the video tokens at that hand's
location through a zero-initialized residual, while a block-causal mask prevents predicted frames from accessing
future information. The model observes 13 frames with their hand poses and tactile readings and predicts the next
36 frames at once.

This repository contains the training, inference and evaluation code of DTWM, together with the two switches used
for the paper's main comparisons: the vision-only baseline (the same model with its tactile input zeroed) and
disabling the tactile pathway at inference.

## Installation

Tested with Python 3.10, CUDA 11.8 and PyTorch 2.6.0.

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Download the backbone [Wan2.2-Fun-5B-InP](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-InP) and pass its
directory as `BACKBONE` below.

## Data

We train and evaluate on the egocentric visual-tactile corpus of
[TouchAnything](https://arxiv.org/abs/2605.13083) (EgoTouch). The clip lists used in the paper are in `splits/`
(paths relative to the dataset root):

| File | Clips | Use |
|---|---:|---|
| `egotouch_train_v2.txt` | 38,600 | training |
| `egotouch_heldout.txt` | 4,704 | clips from recording sessions excluded from training |
| `heldout_3000_49f.txt` | 3,000 | held-out evaluation set |
| `egotouch_val.txt` | 73 | held-out episodes |
| `egotouch_val_indomain.txt` | 200 | training episodes (reserved clips) |

Each clip has 49 frames at 384 x 512. The code expects, under the dataset root, one file per clip in each of
`videos/` (RGB), `videos_static/`, `videos_hands_wilor/` (rendered hand skeleton), `videos_hands_palm6dof/`
(second hand stream), `prompts_rewrite/` (task instruction), `pressure/` (glove readings, `(49, 2, 21, 21)`) and
`hand_centroids/`. The second hand stream must be present even though only the observed frames are used: splicing it
is what blanks the hand condition after the observation.

## Training

The paper setting uses 2 GPUs with 49 GB each and takes about 5.5 to 6.5 hours:

```bash
BACKBONE=/path/to/Wan2.2-Fun-5B-InP DATA_ROOT=/path/to/EgoTouch_dwm_full EXP=dtwm bash scripts/train.sh
```

Checkpoints are written to `outputs/<date>/<EXP>_batch_<timestamp>/checkpoint-<step>/`. All hyperparameters of the
paper are the defaults of `scripts/train.sh` (one epoch over the 38,600 training clips, AdamW, learning rate
3e-5 after a 100-step warmup, LoRA rank 64, tactile footprint sigma 1.8). The seed is not set by default, as in the
paper runs, so repeated trainings differ; `APPLY_SEED=true SEED=<n>` fixes it.

Vision-only baseline: add `VISION_ONLY=true` (the tactile input is zeroed before the embedding, in training and
inference).

## Inference

```bash
CKPT=outputs/<date>/<run>/checkpoint-4825 BACKBONE=/path/to/Wan2.2-Fun-5B-InP \
DATA_ROOT=/path/to/EgoTouch_dwm_full LIST=splits/heldout_3000_49f.txt bash scripts/infer.sh
```

The model observes frames 0 to 12 and predicts frames 13 to 48 in one pass with 30 sampling steps and
classifier-free guidance 6.0 against the negative prompt "bad detailed". As in the paper's evaluation, the tactile
footprint at inference has sigma 3.0 (`TAC_SIGMA` changes it). Use `NUM_SPLITS=N SPLIT_ID=i` to shard over GPUs,
`VISION_ONLY=true` for a vision-only checkpoint, and `GATE_SCALE=0` to disable the tactile pathway at inference.

## Evaluation

```bash
OUTPUTS_ROOT=outputs/<date> RUN=<run> VISION_RUN=<vision-only run> DATA_ROOT=/path/to/EgoTouch_dwm_full bash scripts/eval.sh
```

`eval/score_all_metrics.py` computes per-clip LPIPS over the full frame and over a crop around the hands, together
with PSNR and SSIM, on all 36 predicted frames. `eval/score_hand_consistency.py` computes the hand metrics (overlap,
trajectory and optical flow in the hand region) of a run against a reference run.

## License and acknowledgements

This code is released under the Apache License 2.0 (see `LICENSE` and `NOTICE`). It builds on
[DWM](https://github.com/snuvclab/dwm), which builds on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun),
[Wan](https://github.com/Wan-Video/Wan2.1), [diffusers](https://github.com/huggingface/diffusers) and
[finetrainers](https://github.com/a-r-r-o-w/finetrainers). We thank the authors of TouchAnything for the dataset.

## Citation

```bibtex
@article{zeng2026dtwm,
  title   = {Dexterous Tactile World Model},
  author  = {Zeng, Ziyao and Sun, Xiatao and Wang, Hao and Pan, Yueyang and Yu, Zhengxiang
             and Yang, Fengyu and Liu, Tianyu and Fan, Zhiwen and Rakita, Daniel},
  journal = {arXiv preprint arXiv:2609.34286},
  year    = {2026}
}
```
