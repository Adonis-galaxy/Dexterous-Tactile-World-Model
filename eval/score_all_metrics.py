"""Every RGB metric for one arm at one anchor, per clip, in a single pass.

Existing scorers were built pairwise (arm vs control) and only carried LPIPS; comparing seven arms that
way needs 21 runs and still has no PSNR/SSIM. This one scores an arm ABSOLUTELY, so any pair can be
differenced afterwards from the per-clip rows.

  lpips        perceptual distance over the generated chunk. The headline metric: it tracks what the
               eye notices, and it is the only one of the three that moves when the hands are wrong.
  lpips_local  the same restricted to the MANIPULATION REGION -- the WiLoR-skeleton bbox padded 25% on
               each side, which is what brings the GRASPED OBJECT into frame, resized to 128^2. So this
               scores hand AND object, not the hand alone (same definition as contact_local_eval.py).
               Global LPIPS is dominated by background/texture and is nearly blind to both -- measured,
               a global dark-mask IoU had 0.483 vs 0.484 discrimination.
  psnr         pixel MSE in dB. Rewards blur: a model that hedges by predicting the average of plausible
               futures scores WELL here while looking obviously wrong. Reported for comparability with
               the video-prediction literature, not as evidence.
  ssim         local luminance/contrast/structure agreement, 11x11 Gaussian, per frame then averaged.
               Same caveat as PSNR but less severe.
  copyK        LPIPS of "freeze the last observed frame" -- the floor any arm must beat to be doing
               anything at all.
  stratum      transition / steady / nocontact, per CLIP, from GT contact events inside the window.
               Tactile's unique information is concentrated in `transition`.
  *_contact / *_free
               the SAME metrics split PER FRAME by whether GT pressure indicates contact on that frame.
               Clip-level strata cannot answer "is the gain on contact frames or not" because a
               transition clip contains both kinds; this can. Needs the glove's pressure-position
               mapping files (pressure_position_mapping_{left,right}.json, --pressure_mapping_dir) to
               separate taxels from bend channels; they are not part of this release, and without them
               frac_contact / *_contact / *_free are skipped (reported as null).

  Generated videos are looked up as
      <outputs_root>/<run>/checkpoint-<ckpt_step>/<subdir>/**/<collection>_<sample>_<stem>_generated.mp4
  (the layout written by training/wan/inference.py), GT as <gt_root>/<clip path from --val_file>.

  python eval/score_all_metrics.py --runs RUN --outputs_root OUT --gt_root DATA --val_file LIST --out rows.json
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
import lpips as lpips_lib
import decord
from PIL import Image

GT_ROOT = None                # set from --gt_root in main()
_MAPS = Path(__file__).resolve().parents[1] / "src/dtwm/resources/pressure_mappings"   # default; --pressure_mapping_dir
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CONTACT_THR, MIN_RUN = 0.5, 3
CONTACT_THR_CELL = 0.1        # a taxel counts as pressed above this
FRAME_CONTACT_FRAC = 0.05     # a frame is "in contact" if >5% of taxels are pressed (the median)
SK_THR, PAD, CROP_RES = 20, 0.25, 128


def taxel_mask():
    """(2,21,21) bool: real contact taxels only (fingertips/thumb/palm), excluding the 12x5 bend cells.
    Returns None (and the per-frame contact metrics are skipped) when the mapping files are absent."""
    import collections, json as _j
    if not all((_MAPS / f"pressure_position_mapping_{hand}.json").is_file() for hand in ("left", "right")):
        print(f"[score] pressure-position mapping files not found in {_MAPS}; skipping the per-frame "
              f"contact split (frac_contact, psnr/ssim_contact, psnr/ssim_free are reported as null).",
              flush=True)
        return None
    try:
        out = np.zeros((2, 21, 21), bool)
        for hi, hand in enumerate(["left", "right"]):
            m = _j.load(open(_MAPS / f"pressure_position_mapping_{hand}.json"))
            cnt = collections.Counter(m.values())
            for k, v in m.items():
                r, c = map(int, k.split(","))
                if not (cnt[v] == 5 and 7 <= r <= 11):
                    out[hi, r, c] = True
        return out
    except Exception as e:
        # Returning None here used to be a silent degradation, and it is not a survivable one:
        # without the mask the contact test falls back to ALL pressure channels, the bend cells
        # (finger flexion, near-never zero) dominate, and ~98.5% of frames score as "in contact".
        # The contact-vs-free contrast then compares two nearly identical sets and reads as a null.
        # Files that exist but cannot be parsed are therefore fatal.
        raise RuntimeError(
            f"taxel mask unavailable: {type(e).__name__}: {e}\n"
            f"  mapping dir: {_MAPS} (exists={_MAPS.exists()})\n"
            f"  Without the mask every contact-stratified metric is meaningless.")


TAXEL_M = None


def uniq(rel):
    p = Path(rel)
    return "_".join(list(p.parts[-4:-2]) + [p.stem])


def to_t(img):
    return (torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)[None] / 127.5 - 1.0).to(DEV)


def lpips_mean(lp, A, B, chunk=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(A), chunk):
            a = torch.cat([to_t(f) for f in A[i:i + chunk]], 0)
            b = torch.cat([to_t(f) for f in B[i:i + chunk]], 0)
            out.append(lp(a, b).flatten().float().cpu())
    return float(torch.cat(out).mean()) if out else None


def read_frames(path, idxs, ref_shape=None):
    vr = decord.VideoReader(str(path))
    n = len(vr)
    out = [vr[min(i, n - 1)].asnumpy() for i in idxs]
    if ref_shape is not None and out and out[0].shape[:2] != tuple(ref_shape):
        out = [np.array(Image.fromarray(f).resize((ref_shape[1], ref_shape[0]), Image.BILINEAR)) for f in out]
    return out


def psnr_ssim(A, B):
    """Mean PSNR (dB) and SSIM over the frame list. SSIM via skimage on greyscale."""
    from skimage.metrics import structural_similarity as sk_ssim
    ps, ss = [], []
    for a, b in zip(A, B):
        a = a.astype(np.float64); b = b.astype(np.float64)
        mse = float(((a - b) ** 2).mean())
        ps.append(10.0 * np.log10(255.0 ** 2 / max(mse, 1e-9)))
        ss.append(float(sk_ssim(a.mean(-1), b.mean(-1), data_range=255.0)))
    return float(np.mean(ps)), float(np.mean(ss))


# Frame count of the clips being scored. It was the literal 49 in six places, so scoring a 29-frame
# arm died with "cannot reshape array of size 9106 into shape (49,newaxis)" -- 9106 = 29 x 314. Set
# from --frames; the default keeps every 49-frame arm byte-identical.
N_FRAMES = 49


def contact_states(rel):
    p = np.load(Path(GT_ROOT) / rel.replace("videos/", "pressure/").replace(".mp4", ".npy")).astype(np.float32)
    p = p[:N_FRAMES]                      # a 49-frame array scored as a 37-frame clip: keep the prefix the model saw
    s = p.reshape(N_FRAMES, 2, -1).sum(-1)
    k = np.ones(3) / 3
    s = np.stack([np.convolve(s[:, h], k, mode="same") for h in (0, 1)], 1)
    c = s > CONTACT_THR
    for h in (0, 1):
        x = c[:, h].astype(int); i = 0
        while i < N_FRAMES:
            j = i
            while j < N_FRAMES and x[j] == x[i]:
                j += 1
            if (j - i) < MIN_RUN and i > 0:
                x[i:j] = x[i - 1]
            i = j
        c[:, h] = x.astype(bool)
    ev = np.zeros(N_FRAMES, bool)
    for h in (0, 1):
        d = np.diff(c[:, h].astype(int))
        ev[np.where(d != 0)[0] + 1] = True
    return c, ev


def hand_bbox(sk):
    m = None
    for f in sk:
        mm = f.max(-1) > SK_THR
        m = mm if m is None else (m | mm)
    if m is None or m.sum() < 8:
        return None
    ys, xs = np.where(m); H, W = m.shape
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py, px = int((y1 - y0) * PAD), int((x1 - x0) * PAD)
    return max(0, y0 - py), min(H, y1 + py + 1), max(0, x0 - px), min(W, x1 + px + 1)


def crop_all(frames, bb):
    y0, y1, x0, x1 = bb
    return [np.array(Image.fromarray(f[y0:y1, x0:x1]).resize((CROP_RES, CROP_RES))) for f in frames]


def main():
    global GT_ROOT, _MAPS   # must precede every use in this scope
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-sep run dir basenames (runs of ONE model)")
    ap.add_argument("--outputs_root", required=True, help="directory holding the run directories")
    ap.add_argument("--ckpt_step", type=int, default=4825)
    ap.add_argument("--K", type=int, default=12, help="last observed frame (inference --ar_prefix)")
    ap.add_argument("--horizon", type=int, default=36, help="number of predicted frames scored after K")
    ap.add_argument("--frames", type=int, default=49, help="clip length")
    ap.add_argument("--subdir", default="eval",
                    help="inference output subdirectory of checkpoint-<step>/ ({K} is formatted)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_file", required=True, help="clip list (paths relative to --gt_root)")
    ap.add_argument("--gt_root", required=True, help="dataset root holding GT videos/pressure for those clips")
    ap.add_argument("--pressure_mapping_dir", default=str(_MAPS),
                    help="directory with pressure_position_mapping_{left,right}.json (optional; needed only "
                         "for the per-frame contact/free split)")
    a = ap.parse_args()
    _MAPS = Path(a.pressure_mapping_dir)
    runs = a.runs.split(",")
    sub = a.subdir.format(K=a.K)
    global TAXEL_M, N_FRAMES
    # Must be set before idxs and before any contact_states() call, both of which read N_FRAMES.
    N_FRAMES = int(a.frames)
    print(f"[score] N_FRAMES={N_FRAMES} horizon={a.horizon} K={a.K}", flush=True)
    TAXEL_M = taxel_mask()
    lp = lpips_lib.LPIPS(net="alex").to(DEV)
    idxs = [i for i in range(a.K + 1, a.K + 1 + a.horizon) if i < N_FRAMES]
    rows = []
    GT_ROOT = a.gt_root
    for rel in [l.strip() for l in open(a.val_file) if l.strip()]:
        u = uniq(rel)
        gens = [g[0] for g in
                (glob.glob(f"{a.outputs_root}/{r}/checkpoint-{a.ckpt_step}/{sub}/**/{u}_generated.mp4",
                           recursive=True) for r in runs) if g]
        if not gens:
            continue
        gt = read_frames(Path(GT_ROOT) / rel, idxs)
        fK = read_frames(Path(GT_ROOT) / rel, [a.K])[0]
        c, ev = contact_states(rel)
        w = slice(a.K + 1, a.K + 1 + a.horizon)
        row = {"u": u, "n_seeds": len(gens),
               "stratum": "transition" if ev[w].any() else ("steady" if c[w].any() else "nocontact"),
               "copyK": lpips_mean(lp, gt, [fK] * len(gt))}
        skp = Path(GT_ROOT) / rel.replace("videos/", "videos_hands_wilor/")
        bb = hand_bbox(read_frames(skp, idxs)) if skp.exists() else None
        # per-frame contact flags, from TAXELS ONLY. Summing the whole grid makes 98.5% of frames look
        # like contact, because the bend channels (finger flexion) are near-never zero -- the same
        # conflation that corrupted the tactile metrics until the sensor forensics were applied.
        _pr = np.load(Path(GT_ROOT) / rel.replace("videos/", "pressure/").replace(".mp4", ".npy")).astype(np.float32)
        _pr = _pr[:N_FRAMES]
        # "in contact" = more than FRAME_CONTACT_FRAC of the taxels are pressed. A SUM threshold does
        # not separate anything here (91% of frames pass even taxel-only); the fraction of pressed
        # taxels has a usable spread, and its median over the scored window is 0.051, so this splits
        # the frames almost exactly in half -- maximum power for the contact-vs-free contrast.
        if TAXEL_M is not None:
            _tx = _pr[:, TAXEL_M].reshape(N_FRAMES, -1)
            _fr = (_tx > CONTACT_THR_CELL).mean(1)
            _cf = np.array([_fr[i] > FRAME_CONTACT_FRAC for i in idxs], bool)
            row["frac_contact"] = float(_cf.mean())
        else:
            _cf = None
            row["frac_contact"] = None
        L, P, S, LH, PC, PF, SC, SF = [], [], [], [], [], [], [], []
        for g in gens:
            gen = read_frames(g, idxs, ref_shape=gt[0].shape[:2])
            if _cf is not None and (_cf.any() or (~_cf).any()):
                for sel, acc_p, acc_s in ((_cf, PC, SC), (~_cf, PF, SF)):
                    if sel.sum() >= 1:
                        p2, s2 = psnr_ssim([gt[i] for i in np.where(sel)[0]],
                                           [gen[i] for i in np.where(sel)[0]])
                        acc_p.append(p2); acc_s.append(s2)
            L.append(lpips_mean(lp, gt, gen))
            p_, s_ = psnr_ssim(gt, gen); P.append(p_); S.append(s_)
            if bb is not None:
                LH.append(lpips_mean(lp, crop_all(gt, bb), crop_all(gen, bb)))
        row["lpips"] = float(np.mean(L)); row["psnr"] = float(np.mean(P)); row["ssim"] = float(np.mean(S))
        row["lpips_local"] = float(np.mean(LH)) if LH else None
        row["psnr_contact"] = float(np.mean(PC)) if PC else None
        row["psnr_free"] = float(np.mean(PF)) if PF else None
        row["ssim_contact"] = float(np.mean(SC)) if SC else None
        row["ssim_free"] = float(np.mean(SF)) if SF else None
        if len(L) == 2:
            row["seed_spread"] = float(abs(L[0] - L[1]))
        rows.append(row)
    json.dump(rows, open(a.out, "w"), indent=1)
    f = lambda k: np.mean([r[k] for r in rows if r.get(k) is not None])
    print(f"N={len(rows)} seeds={rows[0]['n_seeds'] if rows else 0}  "
          f"LPIPS={f('lpips'):.4f}  LPIPS_local={f('lpips_local'):.4f}  "
          f"PSNR={f('psnr'):.2f}  SSIM={f('ssim'):.4f}  copyK={f('copyK'):.4f}")
    if TAXEL_M is not None:
        print(f"  per-frame: PSNR contact={f('psnr_contact'):.2f} free={f('psnr_free'):.2f} | "
              f"SSIM contact={f('ssim_contact'):.4f} free={f('ssim_free'):.4f} | "
              f"contact frames={f('frac_contact'):.3f}")


if __name__ == "__main__":
    main()
