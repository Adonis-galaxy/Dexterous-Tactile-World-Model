"""DTWM tactile pathway: tokenizer + spatially grounded injector.

Per frame and hand, the DEDUPED glove sensor vector (left 138 / right 147 channels) is standardised
with train-set per-channel statistics (resources/tactile_channel_stats.npz), projected to the DiT
width and given learned frame / hand / observed embeddings (TactileTokenizer.frame_embeddings).
The per-frame embeddings are averaged within each Wan latent-frame group (spatial_reduce) and, after
selected DiT blocks, added to the video tokens as a Gaussian footprint centred on the hand's projected
latent location (TactileSpatialInjector.residual). The injector's output projections are
zero-initialised and its scale is a fixed 1.0, so training starts exactly from the pure-vision model
while the projection still receives gradient.

State space : standardized sensors s = (v - mean) / std, clamped to [-8, 8]
Embedding   : in_hand(s) + frame_emb + hand_emb + obs_emb                  -> (B, L, 2, dim)
Injection   : sum_h footprint_h (x) proj_b(LayerNorm_b(E_h))               -> (B, T*H*W, dim)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

STATS_PATH = Path(__file__).resolve().parents[1] / "resources/tactile_channel_stats.npz"


class TactileTokenizer(nn.Module):
    """Owns: standardization stats, input projections and embeddings.

    `head_l` / `head_r` are not used by the spatial-injection model. They are kept because they are
    constructed (and therefore saved) in every released checkpoint; removing them would change the
    checkpoint key set and the order of random parameter initialisation."""

    def __init__(self, dim: int, n_frames: int = 49, stats_path: str = None):
        super().__init__()
        d = np.load(stats_path or STATS_PATH)
        self.n_frames = n_frames
        for h in ("l", "r"):
            self.register_buffer(f"mean_{h}", torch.from_numpy(d[f"mean_{h}"]).float(), persistent=True)
            self.register_buffer(f"std_{h}", torch.from_numpy(d[f"std_{h}"]).float(), persistent=True)
            self.register_buffer(f"cells_{h}", torch.from_numpy(d[f"cells_{h}"]).long(), persistent=True)
        self.n_l, self.n_r = len(d["mean_l"]), len(d["mean_r"])
        self.in_l = nn.Linear(self.n_l, dim)
        self.in_r = nn.Linear(self.n_r, dim)
        self.frame_emb = nn.Embedding(n_frames, dim)
        self.hand_emb = nn.Embedding(2, dim)
        self.obs_emb = nn.Embedding(2, dim)          # 0=generated, 1=observed(clean)
        self.head_l = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512), nn.SiLU(), nn.Linear(512, self.n_l))
        self.head_r = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512), nn.SiLU(), nn.Linear(512, self.n_r))

    # ---------- data plumbing ----------
    def gather(self, pressure: torch.Tensor):
        """(B, L, 2, 21, 21) grid -> standardized sensor states s_l (B,L,n_l), s_r (B,L,n_r).
        Kept in the INPUT dtype (fp32 in practice); clamped so train-dead channels can't explode."""
        cl, cr = self.cells_l, self.cells_r
        vl = pressure[:, :, 0][:, :, cl[:, 0], cl[:, 1]]
        vr = pressure[:, :, 1][:, :, cr[:, 0], cr[:, 1]]
        sl = (vl - self.mean_l.to(vl.dtype)) / self.std_l.to(vl.dtype)
        sr = (vr - self.mean_r.to(vr.dtype)) / self.std_r.to(vr.dtype)
        return sl.clamp(-8.0, 8.0), sr.clamp(-8.0, 8.0)

    def frame_embeddings(self, x_l: torch.Tensor, x_r: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """x_l (B,L,n_l), x_r (B,L,n_r) standardized states, observed (L,) bool
        -> per-frame-per-hand embedding E (B,L,2,dim), used by the spatial injection."""
        _wd = self.in_l.weight.dtype
        x_l, x_r = x_l.to(_wd), x_r.to(_wd)
        L = x_l.shape[1]
        fe = self.frame_emb(torch.arange(L, device=x_l.device))
        oe = self.obs_emb(observed.long())
        tl = self.in_l(x_l) + fe + oe + self.hand_emb.weight[0]
        tr = self.in_r(x_r) + fe + oe + self.hand_emb.weight[1]
        return torch.stack([tl, tr], dim=2)                                  # (B, L, 2, dim)


def latent_frame_groups(n_pixel: int = 49, n_latent: int = 13):
    """Wan-Fun temporal VAE (4x + 1): latent 0 <- pixel 0; latent k <- pixels [4k-3..4k].
    Returns a list of `n_latent` numpy index arrays partitioning range(n_pixel)."""
    import numpy as _np
    groups = [_np.array([0])]
    for k in range(1, n_latent):
        groups.append(_np.arange(4 * k - 3, 4 * k + 1))
    return groups


def spatial_reduce(E_full: torch.Tensor, C: torch.Tensor, n_latent: int = None):
    """Reduce pixel-frame tactile to Wan latent frames for the spatial injection.
    E_full (B,L,2,dim) per-frame tactile emb; C (B,L,2,3) = (y_norm, x_norm, valid).
    -> E (B,T,2,dim), centroids (B,T,2,2)=(y,x), valid (B,T,2), averaged within each latent-frame group."""
    L = E_full.shape[1]
    if n_latent is None:
        n_latent = (L - 1) // 4 + 1
    groups = latent_frame_groups(L, n_latent)
    y, x, v = C[..., 0], C[..., 1], C[..., 2]
    Es, cs, vs = [], [], []
    for g in groups:
        gi = torch.as_tensor(g, device=E_full.device, dtype=torch.long)
        gi = gi[gi < L]
        Es.append(E_full[:, gi].mean(1))                                     # (B,2,dim)
        vg = v[:, gi]                                                        # (B,|g|,2)
        w = vg.sum(1)                                                        # (B,2)
        yc = (y[:, gi] * vg).sum(1) / w.clamp(min=1.0)
        xc = (x[:, gi] * vg).sum(1) / w.clamp(min=1.0)
        cs.append(torch.stack([yc, xc], dim=-1))                            # (B,2,2)
        vs.append((w > 0).float())                                          # (B,2)
    return torch.stack(Es, 1), torch.stack(cs, 1), torch.stack(vs, 1)


def hand_region_weight(C: torch.Tensor, T: int, H: int, W: int, sigma: float = 4.0):
    """Soft hand-region weight map in Wan LATENT space, for spatially up-weighting the diffusion
    loss so the model is FORCED to render the hands at the conditioned location instead of
    hallucinating hand position.

    C (B,L,2,3) = (y_norm, x_norm, valid) pixel-frame hand centroids. Reduces to T latent frames
    (valid-weighted mean per group), then places a UNIT-PEAK Gaussian (sigma latent cells) at each
    hand centroid and takes the max over the two hands.  -> (B,1,T,H,W) in [0,1] (0 where no hand)."""
    B, L, _, _ = C.shape
    dev = C.device
    groups = latent_frame_groups(L, T)
    y, x, v = C[..., 0], C[..., 1], C[..., 2]
    ii = torch.arange(H, device=dev, dtype=torch.float32).view(1, H, 1)
    jj = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, W)
    out = C.new_zeros(B, T, H, W)
    for ti, g in enumerate(groups):
        gi = torch.as_tensor(g, device=dev, dtype=torch.long); gi = gi[gi < L]
        vg = v[:, gi]                                                        # (B,|g|,2)
        w = vg.sum(1)                                                        # (B,2)
        yc = (y[:, gi] * vg).sum(1) / w.clamp(min=1.0)                       # (B,2)
        xc = (x[:, gi] * vg).sum(1) / w.clamp(min=1.0)
        present = (w > 0).float()                                           # (B,2)
        for h in range(2):
            ci = (yc[:, h] * H).view(B, 1, 1)
            cj = (xc[:, h] * W).view(B, 1, 1)
            gm = torch.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2 * sigma ** 2))   # (B,H,W)
            gm = gm * present[:, h].view(B, 1, 1)
            out[:, ti] = torch.maximum(out[:, ti], gm)
    return out.unsqueeze(1)                                                 # (B,1,T,H,W)


class TactileSpatialInjector(nn.Module):
    """Spatially grounded tactile injection into the video tokens.

    After each selected DiT block, video tokens receive an additive residual built from the per-hand
    tactile embedding, placed with a normalised Gaussian footprint at the hand's projected latent
    location. Each block has its own LayerNorm + Linear projection (no competition with the video
    LoRA capacity).

    Zero-init projection, fixed unit scale (the "forced" injector of the paper). A learnable scalar
    gate in front of a randomly initialised projection has a dead state: at gate = 0 the projection
    receives no gradient. Putting the zero on the projection weight instead (as in ControlNet zero
    convolutions or LoRA B = 0) keeps the residual exactly 0 at initialisation, so training starts
    from the pure-vision model, while d(residual)/d(proj_W) = LN(E)^T != 0 lets the projection learn.
    """

    def __init__(self, dim: int, layers, sigma: float = 3.0, forced: bool = True):
        super().__init__()
        if not forced:
            raise ValueError("Only the forced injector (zero-init proj, fixed unit scale) is supported "
                             "in this release; the gated variant was an earlier ablation.")
        self.layers = list(layers)
        self.sigma = float(sigma)
        self.forced = True
        self.proj = nn.ModuleDict({str(b): nn.Linear(dim, dim) for b in self.layers})
        self.norm = nn.ModuleDict({str(b): nn.LayerNorm(dim) for b in self.layers})
        for b in self.layers:
            nn.init.zeros_(self.proj[str(b)].weight)
            nn.init.zeros_(self.proj[str(b)].bias)
        # fixed unit scale (non-persistent buffer, not part of the checkpoint)
        self.register_buffer("scale", torch.ones(1), persistent=False)

    def residual(self, block_idx: int, E: torch.Tensor, centroids: torch.Tensor,
                 valid: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """E (B,T,2,dim) latent-frame tactile emb; centroids (B,T,2,2) NORMALIZED (y,x) in [0,1];
        valid (B,T,2). -> residual (B, T*H*W, dim) added onto the video tokens (frame-major, H, W)."""
        key = str(block_idx)
        wd = self.proj[key].weight.dtype
        B = E.shape[0]
        ii = torch.arange(H, device=E.device, dtype=torch.float32).view(1, 1, H, 1)
        jj = torch.arange(W, device=E.device, dtype=torch.float32).view(1, 1, 1, W)
        contrib = E.new_zeros(B, T, H * W, E.shape[-1])
        for h in range(2):
            ci = (centroids[:, :, h, 0].float() * H).view(B, T, 1, 1)         # normalized y -> latent row
            cj = (centroids[:, :, h, 1].float() * W).view(B, T, 1, 1)         # normalized x -> latent col
            _n = self.norm[key](E[:, :, h, :].to(wd))                         # (B,T,dim)
            g = torch.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2 * self.sigma ** 2))
            g = g.reshape(B, T, H * W) * valid[:, :, h].view(B, T, 1).to(g.dtype)
            g = g / (g.sum(-1, keepdim=True) + 1e-6)                          # normalize footprint per (b,t)
            e = self.proj[key](_n)                                            # (B,T,dim)
            contrib = contrib + g.unsqueeze(-1).to(wd) * e.unsqueeze(2)       # (B,T,HW,dim)
        contrib = contrib.reshape(B, T * H * W, E.shape[-1])
        return self.scale.to(wd) * contrib  # scale fixed at 1.0; proj (zero-init) carries the learning
