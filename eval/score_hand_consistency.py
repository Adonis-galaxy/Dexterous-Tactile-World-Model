"""Hand-action consistency metrics — does the generated hand DO what the GT hand does?

Global LPIPS is dominated by background/texture and is nearly blind to whether the hands are in the
right place doing the right thing — the one thing a manipulation world model must get right. These
metrics target hand behaviour directly. Everything is computed IDENTICALLY on GT and generated video
(same estimator on both), so estimator bias cancels in the paired delta.

METRICS
  1. iou   — glove-silhouette IoU inside a GT-hand-centred crop. Glove = dark pixels (black tactile
             glove on a bright table). The crop is ESSENTIAL: globally, dark pixels are dominated by
             the wearer's body/shadows and all discrimination vanishes (measured: 0.483 vs 0.484).
  2. cen   — |centroid(GT mask) − centroid(gen mask)| in px: is the hand WHERE GT says it is.
  3. area  — gen mask area / GT mask area: catches hands that dissolve instead of move (→1.0 is best).
  4. dtw   — Dynamic Time Warping distance between the two hand-centroid TRAJECTORIES over the window,
             normalised by warp-path length (px). Unlike `cen` (per-frame, alignment-sensitive), DTW
             tolerates timing offsets: a hand that traces the right path 2 frames early scores well.
             Separates "wrong path" from "right path, wrong phase".
  5. flow_epe  — mean endpoint error between GT and generated Farneback optical flow inside the hand
                 crop (px/frame): does the hand region move by the right vector.
  6. flow_cos  — cosine similarity of per-pixel flow vectors where GT flow is non-trivial: pure
                 DIRECTION agreement, insensitive to speed.
  7. flow_mag  — sum|flow_gen| / sum|flow_gt| over the clip (RATIO OF SUMS, not mean of per-frame
                 ratios): is the hand moving as MUCH as GT (<1 = the hand under-commits to the motion).
                 Ratio-of-sums is mandatory: GT contains near-static frame pairs where a per-frame ratio
                 explodes (~240) and a few frames then dominate the average.
                 NOTE flow_epe alone is GAMEABLE: a nearly-static prediction gets a LOW endpoint error
                 simply by not moving (measured: a model whose hands dissolve can "beat" a better one on epe).
                 Always read flow_epe together with flow_mag and the mask metrics.

STATISTICS: the unit is the CLIP (hand-frames averaged within a clip first). Per-frame pooling is
pseudo-replication and inflates significance (measured: turned a null into a spurious +0.0136).
Paired bootstrap CI over clips.

Requires <gt_root>/<action_dir>/hand_centroids/<stem>.npy for every scored clip (crop centres).
Generated videos are looked up as
    <outputs_root>/<run>/checkpoint-<step>/<subdir>/**/<collection>_<sample>_<stem>_generated.mp4

  python eval/score_hand_consistency.py \
      --touch_run RUN_A --vision_run RUN_B --outputs_root OUT \
      --gt_root DATA --val_file LIST --subdir eval --window 13 49 --out hand.json
"""
import argparse, glob, json
import numpy as np
import decord
import cv2
from PIL import Image

GT_ROOT = None   # --gt_root
VAL = None       # --val_file
OUTS = None      # --outputs_root
RNG = np.random.default_rng(0)
FLOW_KW = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def uniq(rel):
    p = rel.split("/")
    return "_".join([p[-4], p[-3], p[-1].replace(".mp4", "")])


def glove_mask(img, thr=70):
    return img.astype(np.float32).mean(-1) < thr


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u > 0 else np.nan


def centroid(m, min_px=10):
    ys, xs = np.where(m)
    return (ys.mean(), xs.mean()) if len(ys) >= min_px else None


def dtw_dist(a, b):
    """DTW between two (T,2) trajectories, normalised by warp-path length -> px."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), len(b)
    if n < 2 or m < 2:
        return np.nan
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        d = np.hypot(a[i - 1, 0] - b[:, 0], a[i - 1, 1] - b[:, 1])
        for j in range(1, m + 1):
            D[i, j] = d[j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def flow_of(prev_rgb, next_rgb):
    p = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    q = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(p, q, None, **FLOW_KW)


def flow_metrics(fg, fp, hand_mask, mag_thr=0.5):
    """fg/fp: (H,W,2) GT / predicted flow inside the hand crop. hand_mask: GT glove pixels.
    Returns (EPE, cosine, magnitude-ratio). The magnitude ratio MUST be restricted to hand pixels:
    over the whole crop the denominator is dominated by static table and the ratio explodes
    (observed 200x) — a metric bug, not a model property."""
    epe = float(np.hypot(*(fg - fp).transpose(2, 0, 1)).mean())
    mg = np.hypot(fg[..., 0], fg[..., 1])
    mp = np.hypot(fp[..., 0], fp[..., 1])
    sel = mg > mag_thr                                   # only where GT actually moves
    if sel.sum() > 20:
        dot = (fg[..., 0] * fp[..., 0] + fg[..., 1] * fp[..., 1])[sel]
        cos = float((dot / (mg[sel] * np.maximum(mp[sel], 1e-6))).clip(-1, 1).mean())
    else:
        cos = np.nan
    # magnitude: return raw SUMS; the clip-level ratio is sum(gen)/sum(gt). A per-frame ratio is
    # unusable here — GT contains near-static frame pairs (measured: adjacent-frame diff as low as
    # 0.018, ~10% of pairs <0.5) where the denominator collapses and single frames produce ratios of
    # ~240, which then dominate any average. Ratio-of-sums is immune to that.
    return epe, cos, float(mp.sum()), float(mg.sum())


def boot_ci(d, n=10000):
    d = np.array([x for x in d if x == x])
    if len(d) < 3:
        return (float(d.mean()) if len(d) else np.nan), None, None
    idx = RNG.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def find(run, subdir, u, step=4825):
    g = glob.glob(f"{OUTS}/{run}/checkpoint-{step}/{subdir}/**/{u}_generated.mp4", recursive=True)
    return g[0] if g else None


METRICS = [("iou", "higher"), ("cen", "lower"), ("area", "toward1"),
           ("dtw", "lower"), ("flow_epe", "lower"), ("flow_cos", "higher"), ("flow_mag", "toward1")]


def main():
    global GT_ROOT, VAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--touch_run", required=True)
    ap.add_argument("--vision_run", required=True)
    ap.add_argument("--subdir", default="eval", help="inference output subdirectory of checkpoint-<step>/")
    ap.add_argument("--window", type=int, nargs=2, default=[13, 49],
                    help="frame range [start, end) scored; 13 49 = the 36 predicted frames after K=12")
    ap.add_argument("--crop", type=float, default=0.22)
    ap.add_argument("--label_a", default="touch")
    ap.add_argument("--label_b", default="vision")
    ap.add_argument("--ckpt_step", type=int, default=4825, help="checkpoint step for the A (touch) arm")
    ap.add_argument("--vision_ckpt_step", type=int, default=0,
                    help="checkpoint step for the B (vision) arm; 0 = same as --ckpt_step. Differing "
                         "steps mean differing LR schedules -- reference only, not a controlled result.")
    ap.add_argument("--outputs_root", required=True, help="directory holding the run directories")
    ap.add_argument("--val_file", required=True, help="clip list (paths relative to --gt_root)")
    ap.add_argument("--gt_root", required=True, help="dataset root holding GT videos/hand_centroids for those clips")
    ap.add_argument("--out", required=True, help="output JSON (per-clip rows)")
    args = ap.parse_args()
    GT_ROOT = args.gt_root; VAL = args.val_file
    global OUTS
    OUTS = args.outputs_root
    W = list(range(args.window[0], args.window[1]))

    rows = []
    for rel in [l.strip() for l in open(VAL) if l.strip()]:
        u = uniq(rel)
        _vstep = args.vision_ckpt_step or args.ckpt_step
        pv = find(args.vision_run, args.subdir, u, _vstep)
        pt = find(args.touch_run, args.subdir, u, args.ckpt_step)
        if not pv or not pt:
            continue
        hc = np.load(f"{GT_ROOT}/{rel}".replace("videos/", "hand_centroids/").replace(".mp4", ".npy"))
        vr, vv, vt = (decord.VideoReader(f"{GT_ROOT}/{rel}"), decord.VideoReader(pv), decord.VideoReader(pt))
        H, Wd = vv[0].shape[:2]
        acc = {k: {**{m: [] for m, _ in METRICS}, "_mag_gen": [], "_mag_gt": []}
               for k in ("vision", "touch")}
        traj = {"gt": {0: [], 1: []}, "vision": {0: [], 1: []}, "touch": {0: [], 1: []}}
        prev = None
        for i in W:
            if i >= len(vr):
                continue
            g = np.array(Image.fromarray(vr[i].asnumpy()).resize((Wd, H), Image.BILINEAR))
            gen = {"vision": vv[i].asnumpy(), "touch": vt[i].asnumpy()}
            cur = {"gt": g, **gen}
            for h in range(hc.shape[1]):
                y, x, val = hc[i, h]
                if val < 0.5:
                    continue
                cy, cx, r = int(y * H), int(x * Wd), int(args.crop * H)
                y0, y1 = max(0, cy - r), min(H, cy + r)
                x0, x1 = max(0, cx - r), min(Wd, cx + r)
                mg = glove_mask(g[y0:y1, x0:x1])
                a = centroid(mg)
                ga = mg.sum()
                if a:
                    traj["gt"][h].append((y0 + a[0], x0 + a[1]))
                # optical flow inside the same crop (needs the previous frame)
                fg = flow_of(prev["gt"][y0:y1, x0:x1], g[y0:y1, x0:x1]) if prev is not None else None
                for k in ("vision", "touch"):
                    mm = glove_mask(gen[k][y0:y1, x0:x1])
                    acc[k]["iou"].append(iou(mg, mm))
                    b = centroid(mm)
                    if b:
                        traj[k][h].append((y0 + b[0], x0 + b[1]))
                    if a and b:
                        acc[k]["cen"].append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
                    if ga > 0:
                        acc[k]["area"].append(float(mm.sum() / ga))
                    if fg is not None:
                        fp = flow_of(prev[k][y0:y1, x0:x1], gen[k][y0:y1, x0:x1])
                        e, c, s_gen, s_gt = flow_metrics(fg, fp, mg)
                        acc[k]["flow_epe"].append(e)
                        acc[k]["flow_cos"].append(c)
                        acc[k]["_mag_gen"].append(s_gen)
                        acc[k]["_mag_gt"].append(s_gt)
            prev = cur
        # DTW per hand on the centroid trajectories, then averaged over hands
        for k in ("vision", "touch"):
            ds = [dtw_dist(traj["gt"][h], traj[k][h]) for h in (0, 1)
                  if len(traj["gt"][h]) > 1 and len(traj[k][h]) > 1]
            acc[k]["dtw"] = [d for d in ds if d == d]
            # clip-level magnitude ratio = sum(gen) / sum(gt)  (see flow_metrics)
            sg, st = float(np.sum(acc[k]["_mag_gen"])), float(np.sum(acc[k]["_mag_gt"]))
            acc[k]["flow_mag"] = [sg / st] if st > 1e-6 else []
        if acc["vision"]["iou"]:
            rows.append({"u": u, **{f"{k}_{m}": (float(np.nanmean(acc[k][m])) if len(acc[k][m]) else np.nan)
                                    for k in ("vision", "touch") for m, _ in METRICS}})
    json.dump(rows, open(args.out, "w"), indent=1, default=lambda o: None)

    A, B = args.label_a, args.label_b
    print(f"\n[hand consistency] N={len(rows)} clips  window={args.window}  subdir={args.subdir}")
    print(f"  A={A} ({args.touch_run[:34]})   B={B} ({args.vision_run[:34]})")
    for m, better in METRICS:
        t = np.array([r[f"touch_{m}"] for r in rows], float)
        v = np.array([r[f"vision_{m}"] for r in rows], float)
        ok = ~(np.isnan(t) | np.isnan(v))
        d, lo, hi = boot_ci(list(t[ok] - v[ok]))
        if lo is None:
            verdict = ""
        elif better == "higher":
            verdict = f"{A} BETTER" if lo > 0 else (f"{B} better" if hi < 0 else "null")
        elif better == "lower":
            verdict = f"{A} BETTER" if hi < 0 else (f"{B} better" if lo > 0 else "null")
        else:
            verdict = f"(|1-x|: {B}={abs(1-np.nanmean(v)):.3f} {A}={abs(1-np.nanmean(t)):.3f})"
        print(f"  {m:9s}: {B}={np.nanmean(v):.4f}  {A}={np.nanmean(t):.4f}  "
              f"delta={d:+.4f} [{lo:+.4f},{hi:+.4f}]  {verdict}")


if __name__ == "__main__":
    main()
