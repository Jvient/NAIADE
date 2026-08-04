"""
==========================================================================
  BRICK 1 -- Observability AE-UNet v4 (MC-Dropout + ObsGate + FiLM)
==========================================================================

Why v4 differs from earlier versions
------------------------------------
v1/v2/v3 used a VAE (reparameterisation z = mu + eps*sigma). The v3 training
log showed RMSE_unobs stalling around 0.185 from epoch 40 onwards. Cause: the
noise eps ~ N(0, I) injected at every training forward pass creates an RMSE
floor the model cannot cross, whatever the architectural improvements.

v4 solution: deterministic AE + MC-Dropout for uncertainty
----------------------------------------------------------
- MC-Dropout (Gal & Ghahramani 2016): dropout kept active at inference too
  -> N forward passes -> prediction variance = epistemic uncertainty
  -> same uncertainty quality as a VAE, far better RMSE

Architectural changes
---------------------
1. ObsGate on every skip connection
   A gate sigmoid(conv(mask_downsampled)) modulates the skip features by the
   local observation density. The decoder knows which zone is observed.

2. GroupNorm replaces BatchNorm
   Compatible with batch_size=1 at MC inference (BatchNorm crashes at B=1).

3. Huber loss (delta=0.5) replaces MSE
   Robust to noisy observations and misplaced fronts. Bounded gradients give
   more stable convergence.

4. L_spec and L_ts removed
   In v3, L_spec ~ 0.007 at the end -> negligible contribution. Those terms
   diluted the gradients of the main reconstruction objective.

5. FiLM conditioning and deep supervision retained.

Usage:
  python 01_autoencoder.py --train
  python 01_autoencoder.py --score   --checkpoint outputs/vae_best.pt
  python 01_autoencoder.py --figures --checkpoint outputs/vae_best.pt
"""

import sys, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data.dataset import (SyntheticOceanGenerator, OceanOEDDataset,
                          build_datasets, mesoscale_anomaly,
                          sample_separated_positions)


# =============================================================================
#  BASIC BLOCKS
# =============================================================================

class MCDropout2d(nn.Module):
    """
    Spatial dropout kept ALWAYS active (training and inference).
    That is the point of MC-Dropout: at inference, N passes with dropout on
    give a prediction variance = epistemic uncertainty.
    (Gal & Ghahramani 2016 -- Bayesian deep learning via dropout)
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        # training=True forces dropout even in eval mode
        return F.dropout2d(x, p=self.p, training=True)


class ResDoubleConv(nn.Module):
    """Residual double convolution + spatial MC-Dropout."""
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),   # GroupNorm over BatchNorm
            nn.GELU(),                               # for MC-Dropout (batch size 1)
            MCDropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return self.net(x) + self.skip(x)

    # Note: GroupNorm replaces BatchNorm because MC-Dropout at inference can
    # be called with batch_size=1, which makes BatchNorm crash.


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2),
                                 ResDoubleConv(in_ch, out_ch, dropout_p))
    def forward(self, x): return self.net(x)


class ObsGate(nn.Module):
    """
    Skip gating conditioned on the local observation density.

    For each skip connection (level k) the observation mask is downsampled to
    that level's resolution and a gate sigmoid(conv(mask_ds)) in [0, 1] is
    computed. The gate modulates the skip features:
      - gate ~ 1 in well-observed zones -> the skip passes through strongly
        (the decoder can trust the encoded features)
      - gate ~ 0 in gap zones -> the skip is attenuated
        (the decoder must interpolate from the bottleneck)

    Direct effect on RMSE: the decoder no longer confuses observed zones with
    zones it has to interpolate.
    """
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(1, ch // 4, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, skip_feat, mask_ds):
        return skip_feat * self.gate(mask_ds)


class FiLMUp(nn.Module):
    """Up block with FiLM conditioning (N_obs) + ObsGate on the skip."""
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim, dropout_p=0.1):
        super().__init__()
        self.up      = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.gate    = ObsGate(skip_ch)
        self.conv    = ResDoubleConv(in_ch // 2 + skip_ch, out_ch, dropout_p)
        self.film    = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x, skip, cond, mask_ds):
        x    = self.up(x)
        dy   = skip.shape[2] - x.shape[2]
        dx   = skip.shape[3] - x.shape[3]
        x    = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        skip = self.gate(skip, mask_ds)          # gate by local observation density
        h    = self.conv(torch.cat([skip, x], dim=1))
        gam, bet = self.film(cond).chunk(2, dim=-1)
        gam  = gam.view(-1, h.shape[1], 1, 1)
        bet  = bet.view(-1, h.shape[1], 1, 1)
        return h * (1 + gam) + bet


class CBAM(nn.Module):
    """Channel + spatial attention at the bottleneck."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(1, ch // reduction)), nn.GELU(),
            nn.Linear(max(1, ch // reduction), ch), nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        w = self.channel_att(x).view(x.shape[0], x.shape[1], 1, 1)
        x = x * w
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.spatial_att(torch.cat([avg, mx], dim=1))


# =============================================================================
#  AE-UNet v4 -- MC-Dropout + ObsGate + FiLM
# =============================================================================

class ObservabilityVAE(nn.Module):
    """
    Deterministic AE-UNet with MC-Dropout uncertainty.

    Why drop the VAE reparameterisation
    -----------------------------------
    In the v3 log, RMSE stalls at 0.185 from epoch 40 despite 160 further
    epochs. Cause: the noise eps ~ N(0, I) injected at EVERY training forward
    pass creates an RMSE floor (~0.03-0.05 in absolute terms) the model cannot
    cross.

    MC-Dropout (Gal & Ghahramani 2016)
    ----------------------------------
    - training  : dropout active -> regularisation
    - inference : dropout STILL active -> N passes -> variance = uncertainty
    - mathematically equivalent to approximate variational inference
    - much better RMSE, since no latent noise is injected

    v4 changes
    ----------
    1. MC-Dropout replaces the VAE reparameterisation
    2. ObsGate on every skip: gate conditioned on local observation density
       -> the decoder can tell observed zones from gap zones
    3. GroupNorm replaces BatchNorm (works at batch_size=1 for MC inference)
    4. FiLM conditioning retained (N_obs)
    5. Deep supervision retained
    """

    def __init__(self, in_ch=3, out_ch=2, base_ch=32, latent_ch=64,
                 dropout_p=0.1, cond_dim=32):
        super().__init__()
        bc = base_ch
        dp = dropout_p
        self.latent_ch = latent_ch
        self.cond_dim  = cond_dim
        self.dropout_p = dropout_p

        # -- 4-level encoder ------------------------------------------------
        self.inc   = ResDoubleConv(in_ch, bc,    dp)
        self.down1 = Down(bc,    bc*2,  dp)
        self.down2 = Down(bc*2,  bc*4,  dp)
        self.down3 = Down(bc*4,  bc*8,  dp)
        self.down4 = Down(bc*8,  bc*16, dp)

        # -- FiLM embedding -------------------------------------------------
        self.cond_embed = nn.Sequential(
            nn.Linear(1, cond_dim), nn.GELU(),
            nn.Linear(cond_dim, cond_dim), nn.GELU(),
        )

        # -- Deterministic bottleneck + CBAM --------------------------------
        # No reparameterisation -- the "latent" is a plain feature vector
        self.cbam   = CBAM(bc*16)
        self.to_z   = nn.Conv2d(bc*16, latent_ch, 1)   # deterministic encoding
        self.from_z = nn.Conv2d(latent_ch, bc*16, 1)

        # -- 4-level FiLM decoder + ObsGate ---------------------------------
        self.up1 = FiLMUp(bc*16, bc*8,  bc*8,  cond_dim, dp)
        self.up2 = FiLMUp(bc*8,  bc*4,  bc*4,  cond_dim, dp)
        self.up3 = FiLMUp(bc*4,  bc*2,  bc*2,  cond_dim, dp)
        self.up4 = FiLMUp(bc*2,  bc,    bc,    cond_dim, dp)
        self.head = nn.Conv2d(bc, out_ch, 1)

        # -- Deep supervision heads -----------------------------------------
        self.aux1 = nn.Conv2d(bc*8, out_ch, 1)
        self.aux2 = nn.Conv2d(bc*4, out_ch, 1)
        self.aux3 = nn.Conv2d(bc*2, out_ch, 1)

    def _get_cond(self, x):
        mask = x[:, 2:3]
        return self.cond_embed(mask.mean(dim=[2, 3]))   # (B, cond_dim)

    def _downsample_mask(self, mask, target):
        """Nearest-neighbour downsampling of the mask to a target feature-map size."""
        return F.interpolate(mask, size=target.shape[2:], mode="nearest")

    def encode(self, x):
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        h  = self.down4(s4)
        h  = self.cbam(h)
        z  = self.to_z(h)
        return z, (s1, s2, s3, s4)

    def decode(self, z, skips, cond, mask):
        """Decoding with FiLM + ObsGate on every skip."""
        s1, s2, s3, s4 = skips
        h  = self.from_z(z)
        h  = self.up1(h, s4, cond, self._downsample_mask(mask, s4))
        aux1 = self.aux1(h)
        h  = self.up2(h, s3, cond, self._downsample_mask(mask, s3))
        aux2 = self.aux2(h)
        h  = self.up3(h, s2, cond, self._downsample_mask(mask, s2))
        aux3 = self.aux3(h)
        h  = self.up4(h, s1, cond, self._downsample_mask(mask, s1))
        return self.head(h), [aux1, aux2, aux3]

    def forward(self, x):
        mask      = x[:, 2:3]
        cond      = self._get_cond(x)
        z, skips  = self.encode(x)
        pred, aux = self.decode(z, skips, cond, mask)
        # API compatibility with the rest of the code (score, figures):
        # return mu=z and logvar=zeros (there is no KL term)
        return pred, z, torch.zeros_like(z), aux

    @torch.no_grad()
    def reconstruct_with_uncertainty(self, x, n_samples=50):
        """
        MC-Dropout uncertainty.

        The model is in EVAL mode (normalisation layers frozen) but
        MCDropout2d forces dropout=True, so every pass gives a slightly
        different prediction. The variance is the epistemic uncertainty.

        Advantage over a VAE: no latent noise, so the mean prediction is much
        closer to the ground truth.
        """
        mask = x[:, 2:3]
        cond = self._get_cond(x)
        z, skips = self.encode(x)
        # MC passes: dropout active through MCDropout2d
        samples = [self.decode(z, skips, cond, mask)[0]
                   for _ in range(n_samples)]
        stack = torch.stack(samples)
        return stack.mean(0), stack.std(0), z

    def get_latent(self, x):
        z, _ = self.encode(x)
        return z.flatten(1)


# =============================================================================
#  LOSS v4: Huber + gradient + deep supervision  (spectral/T-S removed)
# =============================================================================

class VAELoss(nn.Module):
    """
    Loss v4 -- focused on RMSE_unobs:

        L = L_recon + lambda_grad * L_grad + sum_k w_k * L_aux_k
        (no KL term, no spectral term)

    Huber loss (delta=0.5) replaces MSE: MSE penalises outliers very heavily
    (noisy observations, a misplaced front). Huber is quadratic for |e| < delta
    and linear beyond, so gradients stay bounded and convergence on fronts is
    more stable.

    L_spec and L_ts removed: in the v3 log, L_spec ended at 0.007, a nearly
    null contribution whose gradients simply polluted the main optimisation.
    Same story for L_ts (physically useful constraint, but weighted too low to
    offset its dilution of the main gradient).
    """
    def __init__(self, w_obs=1.0, w_unobs=4.0, beta_max=0.0,
                 lambda_grad=0.5, lambda_spec=0.0, lambda_ts=0.0,
                 huber_delta=0.5):
        super().__init__()
        self.w_obs       = w_obs
        self.w_unobs     = w_unobs
        self.beta_max    = beta_max          # kept at 0 -- no KL term
        self.lambda_grad = lambda_grad
        self.huber_delta = huber_delta
        self.aux_weights = [0.4, 0.3, 0.2]

    @staticmethod
    def _spatial_grad(f):
        gx = f[..., 1:, :] - f[..., :-1, :]
        gy = f[..., :, 1:] - f[..., :, :-1]
        return gx, gy

    def _huber(self, diff):
        d   = self.huber_delta
        abs_diff = diff.abs()
        return torch.where(abs_diff < d,
                           0.5 * diff**2,
                           d * (abs_diff - 0.5 * d))

    def _recon_loss(self, pred, target, mask):
        err = self._huber(pred - target)
        return (self.w_obs * (err * mask).mean()
                + self.w_unobs * (err * (1 - mask)).mean())

    def forward(self, pred, target, mask, mu, logvar, beta=1.0, aux_preds=None):
        loss_recon = self._recon_loss(pred, target, mask)

        pgx, pgy = self._spatial_grad(pred)
        tgx, tgy = self._spatial_grad(target)
        loss_grad = (self._huber(pgx - tgx).mean()
                     + self._huber(pgy - tgy).mean())

        loss_aux = torch.tensor(0.0, device=pred.device)
        if aux_preds is not None:
            H, W = target.shape[2], target.shape[3]
            for aux, w in zip(aux_preds, self.aux_weights):
                aux_up  = F.interpolate(aux, size=(H, W), mode="bilinear",
                                        align_corners=False)
                mask_ds = F.interpolate(mask, size=(H, W), mode="nearest")
                loss_aux = loss_aux + w * self._recon_loss(aux_up, target, mask_ds)

        # KL = 0 (no reparameterisation)
        kl       = torch.tensor(0.0, device=pred.device)
        loss_spec = torch.tensor(0.0, device=pred.device)

        total = loss_recon + self.lambda_grad * loss_grad + loss_aux
        return total, loss_recon, kl, loss_aux, loss_spec


# =============================================================================
#  TRAINING
# =============================================================================

def train(args):
    print("=" * 62)
    print("  Brick 1 -- AE-UNet v4 training (MC-Dropout + ObsGate)")
    print("=" * 62)

    print("\n[1/4] Nature run generation...")
    # Explicit seed_ocean: without it the training nature run differed from
    # the one used by --figures / --score, and the checkpoint was not
    # reproducible from one run to the next.
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=args.nt, seed=args.seed_ocean)
    print(f"  T: {T.shape}  [{T.min():.1f}, {T.max():.1f}] degC  (seed={args.seed_ocean})")
    print(f"  S: {S.shape}  [{S.min():.2f}, {S.max():.2f}] psu")
    print(f"  sigma(SST)={T.std():.2f} degC   sigma(SSS)={S.std():.3f} psu")
    if args.nt < 365:
        print(f"  [WARNING] nt={args.nt} < 365: incomplete seasonal cycle, "
              f"biased statistics.")


    # augment=True on the training split only: random zonal flip
    # validation is left unaugmented for a stable measurement
    train_ds, val_ds = build_datasets(T, S, split=0.8,
                                      n_obs_min=args.n_obs_min,
                                      n_obs_max=args.n_obs_max,
                                      augment_train=True)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    print(f"\n[2/4] AE-UNet v4 model "
          f"(base_ch={args.base_ch}, latent_ch={args.latent_ch}, "
          f"dropout={args.dropout_p}, cond_dim={args.cond_dim})...")
    model = ObservabilityVAE(base_ch=args.base_ch,
                             latent_ch=args.latent_ch,
                             dropout_p=args.dropout_p,
                             cond_dim=args.cond_dim).to(DEVICE)
    npar = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {npar:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(ep):
        warmup_ep = max(1, args.epochs // 10)
        if ep < warmup_ep:
            return float(ep + 1) / warmup_ep
        progress = (ep - warmup_ep) / max(1, args.epochs - warmup_ep)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = VAELoss(w_obs=1.0, w_unobs=args.w_unobs,
                        beta_max=args.beta_max,
                        lambda_grad=args.lambda_grad,
                        lambda_spec=args.lambda_spec,
                        lambda_ts=args.lambda_ts)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    warmup  = max(1, args.epochs // 3)

    print(f"\n[3/4] Training {args.epochs} epochs | "
          f"Huber delta={args.huber_delta} | lambda_grad={args.lambda_grad} | "
          f"MC-Dropout p={args.dropout_p} | augment=flip | "
          f"n_mc_val={args.n_mc_val}...")
    history = {"train_loss": [], "val_rmse_unobs": [], "kl": [],
               "loss_spec": [], "loss_aux": [], "lr": [],
               "rmse_T_degC": [], "rmse_S_psu": []}
    best_val = np.inf

    for epoch in range(1, args.epochs + 1):
        beta = min(1.0, epoch / warmup)

        model.train()
        ep_loss = ep_kl = ep_spec = ep_aux = 0.0
        for x, y, mask in train_ld:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            pred, mu, logvar, aux_preds = model(x)
            loss, _, kl, l_aux, l_spec = criterion(
                pred, y, mask, mu, logvar, beta, aux_preds)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item(); ep_kl += kl.item()
            ep_spec += l_spec.item() if torch.is_tensor(l_spec) else float(l_spec)
            ep_aux  += l_aux.item()  if torch.is_tensor(l_aux)  else float(l_aux)
        scheduler.step()
        n = len(train_ld)
        ep_loss /= n; ep_kl /= n; ep_spec /= n; ep_aux /= n

        # MC-averaged validation: average n_mc_val passes to get the model's
        # true RMSE rather than the bias of a single dropout draw
        model.eval()
        val_rmses = []
        val_rmse_T, val_rmse_S = [], []
        val_rmse_by_density = {"sparse": [], "medium": [], "dense": []}
        with torch.no_grad():
            for x, y, mask in val_ld:
                x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                # Average over n_mc_val MC-Dropout passes
                preds = torch.stack([model(x)[0] for _ in range(args.n_mc_val)])
                pred_mean = preds.mean(0)
                sq = (pred_mean - y) ** 2
                # RMSE and stratification per sample (not a batch average)
                for b in range(x.shape[0]):
                    n_obs_b = int(mask[b].sum().item())   # observations for this sample
                    rmse_b  = float(torch.sqrt(
                        (sq[b] * (1 - mask[b])).mean()).item())
                    # Per-variable RMSE: the two channels are normalised by
                    # very different standard deviations (2.6 degC against
                    # 0.18 psu), so an aggregate RMSE converts to physical
                    # units for neither of them.
                    w = (1 - mask[b])
                    val_rmse_T.append(float(torch.sqrt(
                        (sq[b, 0:1] * w).mean()).item()))
                    val_rmse_S.append(float(torch.sqrt(
                        (sq[b, 1:2] * w).mean()).item()))
                    val_rmses.append(rmse_b)
                    if n_obs_b < 20:
                        val_rmse_by_density["sparse"].append(rmse_b)
                    elif n_obs_b < 50:
                        val_rmse_by_density["medium"].append(rmse_b)
                    else:
                        val_rmse_by_density["dense"].append(rmse_b)
        val_rmse   = float(np.mean(val_rmses))
        rmse_T_phys = float(np.mean(val_rmse_T)) * train_ds.T_std
        rmse_S_phys = float(np.mean(val_rmse_S)) * train_ds.S_std

        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(ep_loss)
        history["val_rmse_unobs"].append(val_rmse)
        history["rmse_T_degC"].append(rmse_T_phys)
        history["rmse_S_psu"].append(rmse_S_phys)
        history["kl"].append(ep_kl)
        history["loss_spec"].append(ep_spec)
        history["loss_aux"].append(ep_aux)
        history["lr"].append(cur_lr)

        if epoch % 5 == 0 or epoch == 1:
            sp = np.mean(val_rmse_by_density["sparse"])  if val_rmse_by_density["sparse"]  else float("nan")
            me = np.mean(val_rmse_by_density["medium"]) if val_rmse_by_density["medium"] else float("nan")
            de = np.mean(val_rmse_by_density["dense"])  if val_rmse_by_density["dense"]  else float("nan")
            print(f"  ep {epoch:3d}/{args.epochs} | Loss={ep_loss:.4f} | "
                  f"RMSE={val_rmse:.4f} [sp:{sp:.3f} me:{me:.3f} de:{de:.3f}] | "
                  f"{rmse_T_phys:.3f} degC / {rmse_S_phys:.4f} psu | "
                  f"lr={cur_lr:.2e}")

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({
                "model_state": model.state_dict(),
                "args":  vars(args),
                "norm":  {"T_mean": train_ds.T_mean, "T_std": train_ds.T_std,
                          "S_mean": train_ds.S_mean, "S_std": train_ds.S_std},
                "ocean": {"seed_ocean": args.seed_ocean, "nt": args.nt,
                          "obs_noise_T": OBS_NOISE_T, "obs_noise_S": OBS_NOISE_S},
            }, out_dir / "vae_best.pt")

    print(f"\n  Best validation RMSE (unobserved) : {best_val:.4f}")
    print(f"  Final physical RMSE : {rmse_T_phys:.3f} degC | {rmse_S_phys:.4f} psu")

    print("\n[4/4] Saving training curves...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 4), facecolor="#0a1628")
    # KL and spectral loss are 0 by construction in v4: show the physical
    # RMSE values instead, which are the genuinely interpretable numbers.
    data = [("Total loss",            "train_loss",      "#6baed6"),
            ("Val RMSE (unobserved)", "val_rmse_unobs",  "#fc8d59"),
            ("RMSE SST (degC)",       "rmse_T_degC",     "#ff6b6b"),
            ("RMSE SSS (psu)",        "rmse_S_psu",      "#74c476"),
            ("Deep supervision",      "loss_aux",        "#cc99ff")]
    for ax, (lbl, k, col) in zip(axes, data):
        ax.plot(history[k], color=col, lw=1.8)
        ax.set_title(lbl, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
        ax.grid(True, alpha=0.2, color="white")
    fig.tight_layout(pad=2)
    fig.savefig(out_dir / "vae_training_curves.png", dpi=130,
                facecolor="#0a1628", bbox_inches="tight")
    plt.close()
    print(f"  Curves -> {out_dir}/vae_training_curves.png")
    print(f"  Checkpoint -> {out_dir}/vae_best.pt")


# =============================================================================
#  FIGURE 1 -- Evaluation of an existing network
#
#  Goal: given a network of N sensors at fixed positions, show where the
#  network covers the domain well or poorly, and quantify each sensor's
#  contribution.
#
#  Layout (2 rows x 4 columns):
#    Row 1 SST: true field + sensors | reconstruction | MC sigma | gap zones
#    Row 2 SSS: true field + sensors | reconstruction | MC sigma | LOO scores
# =============================================================================

@torch.no_grad()
def plot_network_evaluation(model, T, S, norm, args,
                            positions=None, n_samples=80, n_loo_t=8):
    """
    Sensor-network evaluation figure.

    Parameters
    ----------
    positions : list of (x, y) in pixel coordinates, or None
        If None, a network of N_BUOYS sensors is drawn using args.seed_buoys.
    n_samples  : MC draws for the uncertainty estimate
    n_loo_t    : number of time steps used for the LOO computation
    """
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    sal_cmap = LinearSegmentedColormap.from_list("sal",
        ["#003c30","#01665e","#35978f","#80cdc1","#f5f5f5",
         "#dfc27d","#bf812d","#8c510a","#543005"], N=256)
    BG = "#0a1628"

    model.eval()
    T_n = (T - norm["T_mean"]) / norm["T_std"]
    S_n = (S - norm["S_mean"]) / norm["S_std"]

    # -- Reference network ------------------------------------------------------
    if positions is None:
        seed_b = getattr(args, "seed_buoys", 42)
        rng    = np.random.default_rng(seed_b)
        # Minimum separation: adjacent buoys are forbidden on the RL side, so the
        # AE reference network must obey the same constraint
        positions = sample_separated_positions(NX, NY, N_BUOYS, rng=rng)
        print(f"  Network drawn : {N_BUOYS} buoys (seed_buoys={seed_b}, "
              f"separation >= {MIN_BUOY_SEP_KM:.0f} km)")
    positions = list(positions)
    n_sensors = len(positions)

    # Fixed network mask
    mask_np = np.zeros((NX, NY), dtype=np.float32)
    for (x, y) in positions:
        mask_np[x, y] = 1.0
    obs_pos = np.array(positions)

    # Reference time step for the maps
    t_ref = len(T) // 3
    T_t, S_t = T_n[t_ref], S_n[t_ref]
    T_obs = T_t * mask_np
    S_obs = S_t * mask_np
    x_in = torch.from_numpy(np.stack([T_obs, S_obs, mask_np])[None]).to(DEVICE)

    # Reconstruction + MC uncertainty
    recon_mean, recon_std, _ = model.reconstruct_with_uncertainty(x_in, n_samples=n_samples)
    rm = recon_mean[0].cpu().numpy()
    rs = recon_std[0].cpu().numpy()

    T_true  = T_t * norm["T_std"] + norm["T_mean"]
    S_true  = S_t * norm["S_std"] + norm["S_mean"]
    T_pred  = rm[0] * norm["T_std"] + norm["T_mean"]
    S_pred  = rm[1] * norm["S_std"] + norm["S_mean"]
    T_sigma = rs[0] * norm["T_std"]
    S_sigma = rs[1] * norm["S_std"]

    # -- Gap-zone map -----------------------------------------------------------
    # A zone counts as a "gap" when sigma exceeds a threshold (75th pct).
    # Build a binary map plus a distance-to-nearest-sensor mask.
    from scipy.ndimage import distance_transform_edt

    dist_to_sensor = distance_transform_edt(1 - mask_np)   # distance in pixels
    dist_to_sensor_n = dist_to_sensor / dist_to_sensor.max()   # normalised to [0,1]

    # Combined T+S uncertainty (normalised mean)
    T_sigma_n = T_sigma / (T_sigma.max() + 1e-9)
    S_sigma_n = S_sigma / (S_sigma.max() + 1e-9)
    combined_sigma = 0.5 * (T_sigma_n + S_sigma_n)

    # Coverage score: a gap is high sigma AND far from any sensor
    gap_map = combined_sigma * dist_to_sensor_n   # in [0, 1]
    gap_threshold = np.percentile(gap_map, 80)
    gap_binary = (gap_map > gap_threshold).astype(float)

    # -- 3 proposed buoys, maximising gap coverage ------------------------------
    # Greedy algorithm: at each step place the buoy at the maximum of the
    # residual gap_map, then update the distance to the nearest sensor.
    from scipy.ndimage import distance_transform_edt as _edt
    proposed_positions = []
    gap_residual = gap_map.copy()
    mask_augmented = mask_np.copy()
    for _ in range(3):
        flat_idx = np.argmax(gap_residual)
        px, py   = np.unravel_index(flat_idx, gap_residual.shape)  # px in [0,NX), py in [0,NY)
        proposed_positions.append((int(px), int(py)))
        mask_augmented[px, py] = 1.0
        dist_new = _edt(1 - mask_augmented) / (dist_to_sensor.max() + 1e-9)
        gap_residual = combined_sigma * dist_new
    proposed_arr = np.array(proposed_positions)  # (3, 2) -- (x, y) in pixels

    # -- LOO scores: contribution of each sensor --------------------------------
    t_idx = np.random.choice(len(T), min(n_loo_t, len(T)), replace=False)
    rmse_full = np.mean([
        _compute_rmse_mc(model, T_n[t], S_n[t], positions, norm, n_mc=6)
        for t in t_idx
    ])
    loo_delta = np.zeros(n_sensors)
    for i, pos in enumerate(positions):
        sub = [p for j, p in enumerate(positions) if j != i]
        rmse_i = np.mean([
            _compute_rmse_mc(model, T_n[t], S_n[t], sub, norm, n_mc=6)
            for t in t_idx
        ])
        loo_delta[i] = rmse_i - rmse_full   # > 0: this sensor carries information

    # Normalise the deltas for display
    loo_colors = np.clip(loo_delta / (loo_delta.max() + 1e-9), 0, 1)

    # -- Figure -----------------------------------------------------------------
    fig = plt.figure(figsize=(22, 11), facecolor=BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.35, wspace=0.28,
                            left=0.04, right=0.97, top=0.91, bottom=0.05)

    def cell(ax, data, cmap, vmin, vmax, title, label, pts=None, pts_c=None,
             pts_s=40, pts_cmap="RdYlGn", contour=None):
        im = ax.imshow(data.T, cmap=cmap, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        if pts is not None:
            cc = pts_c if pts_c is not None else "white"
            ax.scatter(pts[:,0], pts[:,1], c=cc, s=pts_s,
                       cmap=pts_cmap, vmin=0, vmax=1,
                       edgecolors="black", linewidths=0.5, zorder=6)
        if contour is not None:
            ax.contour(contour.T, levels=[0.5], colors=["#ff6b6b"],
                       linewidths=1.5, linestyles="--")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.set_title(title, color="white", fontsize=8.5, pad=5, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(label, color="white", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
        return im

    vT = (T_true.min(), T_true.max())
    vS = (S_true.min(), S_true.max())

    # -- Row 1: SST -------------------------------------------------------------
    # [0,0] True SST + sensor positions (coloured by LOO delta)
    ax00 = fig.add_subplot(gs[0, 0])
    cell(ax00, T_true, ocean_cmap, *vT,
         f"True SST + network ({n_sensors} sensors)\n"
         f"colour = LOO contribution (green high, red low)",
         "°C", pts=obs_pos, pts_c=loo_colors, pts_s=55)

    # [0,1] SST reconstruction
    ax01 = fig.add_subplot(gs[0, 1])
    cell(ax01, T_pred, ocean_cmap, *vT,
         f"AE SST reconstruction\n(t={t_ref},  RMSE_unobs={rmse_full:.3f})",
         "°C", pts=obs_pos, pts_c="white", pts_s=12)

    # [0,2] SST uncertainty
    ax02 = fig.add_subplot(gs[0, 2])
    cell(ax02, T_sigma, "YlOrRd", 0, T_sigma.max(),
         f"SST MC uncertainty sigma  (N={n_samples} draws)\n"
         "red = zone poorly constrained by the network",
         "°C", pts=obs_pos, pts_c="cyan", pts_s=12,
         contour=gap_binary)

    # [0,3] Gap map + 3 proposed buoys
    ax03 = fig.add_subplot(gs[0, 3])
    cell(ax03, gap_map, "inferno", 0, gap_map.max(),
         f"Gap zones + 3 proposed buoys\n"
         f"(high sigma x sensor distance) -- {int(gap_binary.sum())} critical px",
         "score", pts=obs_pos, pts_c="cyan", pts_s=15,
         contour=gap_binary)
    # Proposed buoys: numbered yellow stars
    for k, (px, py) in enumerate(proposed_arr):
        ax03.scatter(px, py, marker="*", s=320, c="#ffd93d",
                     edgecolors="black", linewidths=0.8, zorder=8)
        ax03.annotate(f"P{k+1}", (px, py),
                      textcoords="offset points", xytext=(6, 4),
                      fontsize=8, color="#ffd93d", fontweight="bold")

    # -- Row 2: SSS -------------------------------------------------------------
    # [1,0] True SSS
    ax10 = fig.add_subplot(gs[1, 0])
    cell(ax10, S_true, sal_cmap, *vS,
         "True SSS + sensor positions", "psu",
         pts=obs_pos, pts_c="white", pts_s=12)

    # [1,1] SSS reconstruction
    ax11 = fig.add_subplot(gs[1, 1])
    cell(ax11, S_pred, sal_cmap, *vS,
         "AE SSS reconstruction", "psu",
         pts=obs_pos, pts_c="white", pts_s=12)

    # [1,2] SSS uncertainty
    ax12 = fig.add_subplot(gs[1, 2])
    cell(ax12, S_sigma, "YlOrRd", 0, S_sigma.max(),
         "SSS MC uncertainty sigma", "psu",
         pts=obs_pos, pts_c="cyan", pts_s=12,
         contour=gap_binary)

    # [1,3] LOO bar chart: contribution of each sensor
    ax13 = fig.add_subplot(gs[1, 3])
    ax13.set_facecolor("#050d1a")
    for sp in ax13.spines.values(): sp.set_edgecolor("#1a3a5c")

    idx_sort = np.argsort(loo_delta)[::-1]   # sorted by decreasing contribution
    colors_bar = plt.cm.RdYlGn(
        np.clip((loo_delta[idx_sort] - loo_delta.min()) /
                (loo_delta.max() - loo_delta.min() + 1e-9), 0, 1))
    ax13.barh(np.arange(n_sensors), loo_delta[idx_sort], color=colors_bar,
              edgecolor="#1a3a5c", linewidth=0.5)
    ax13.axvline(0, color="white", lw=0.8, alpha=0.5)
    ax13.set_yticks(np.arange(0, n_sensors, max(1, n_sensors//10)))
    ax13.set_yticklabels([f"C{idx_sort[i]}" for i in
                          range(0, n_sensors, max(1, n_sensors//10))],
                         color="white", fontsize=6)
    ax13.set_xlabel("delta RMSE  (LOO - full)", color="white", fontsize=8)
    ax13.set_title("LOO contribution per sensor\n"
                   "green = essential  |  red = redundant",
                   color="white", fontsize=8.5, fontweight="bold", pad=5)
    ax13.tick_params(colors="white", labelsize=6)
    ax13.grid(True, alpha=0.2, color="white", axis="x")

    # Redundancy threshold (delta < 5% of the maximum)
    thr = loo_delta.max() * 0.05
    n_redondant = (loo_delta < thr).sum()
    ax13.axvline(thr, color="#ffd93d", lw=1, linestyle="--", alpha=0.7,
                 label=f"5% threshold  ({n_redondant} redundant)")
    ax13.legend(fontsize=6, labelcolor="white", facecolor="#0a1628", loc="lower right")

    fig.text(0.5, 0.97,
             f"AE-UNet -- network evaluation  ({n_sensors} sensors)  "
             f"|  red contour = gap zones  |  star = proposed buoy",
             ha="center", color="white", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "cyan = existing sensor  |  yellow star = proposed buoy (greedy gap)  "
             "|  sensor colour in row 1 = LOO contribution",
             ha="center", color="#8ab4d4", fontsize=8)

    out = out_dir / "vae_network_evaluation.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Network evaluation figure -> {out}")
    print(f"  Proposed buoys : " +
          "  ".join([f"P{k+1}=({px},{py})" for k,(px,py) in enumerate(proposed_arr)]))
    return loo_delta, gap_map, positions, proposed_arr


# =============================================================================
#  FIGURE 2 - Uncertainty across different network densities
#
#  Same network, sensors are removed progressively to show how uncertainty
#  grows in the areas that were already poorly covered.
# =============================================================================

@torch.no_grad()
def plot_uncertainty_maps(model, T, S, norm, args, n_samples=60):
    """
    Figure 2 - Uncertainty as a function of network density.

    3 columns: Dense (N=40), Medium (N=20), Sparse (N=8)
    For each: SST sigma | SSS sigma | meridional uncertainty profile
    """
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    BG = "#0a1628"

    model.eval()
    T_n = (T - norm["T_mean"]) / norm["T_std"]
    S_n = (S - norm["S_mean"]) / norm["S_std"]
    t   = len(T) // 2

    configs = [("Dense   (N=40)", 40), ("Medium  (N=20)", 20), ("Sparse  (N=8)", 8)]

    # Common colour scale so the three columns are comparable
    unc_max_all = 0.0
    results = []
    for (_, n_obs) in configs:
        mask = np.zeros((NX, NY), dtype=np.float32)
        for (px, py) in sample_separated_positions(
                NX, NY, n_obs, rng=np.random.default_rng(n_obs * 7)):
            mask[px, py] = 1.0
        x_in = torch.from_numpy(
            np.stack([T_n[t]*mask, S_n[t]*mask, mask])[None]).to(DEVICE)
        rm, rs, _ = model.reconstruct_with_uncertainty(x_in, n_samples=n_samples)
        T_s = rs[0, 0].cpu().numpy() * norm["T_std"]
        S_s = rs[0, 1].cpu().numpy() * norm["S_std"]
        unc_max_all = max(unc_max_all, T_s.max(), S_s.max())
        results.append((mask, T_s, S_s, np.argwhere(mask > 0.5)))

    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.28,
                            left=0.05, right=0.97, top=0.91, bottom=0.06)

    for col, ((desc, n_obs), (mask_np, T_s, S_s, obs_pos)) in enumerate(
            zip(configs, results)):

        combined_n = 0.5 * (T_s / (unc_max_all + 1e-9) + S_s / (unc_max_all + 1e-9))

        # Ligne 0 : SST sigma
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(T_s.T, cmap="YlOrRd", origin="lower", aspect="auto",
                       vmin=0, vmax=unc_max_all)
        ax.scatter(obs_pos[:,0], obs_pos[:,1], c="cyan", s=12,
                   edgecolors="black", linewidths=0.3, zorder=5, alpha=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.set_title(f"{desc}\nSST sigma (°C)", color="white", fontsize=9,
                     fontweight="bold", pad=4)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)

        # Ligne 1 : SSS sigma
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(S_s.T, cmap="YlOrRd", origin="lower", aspect="auto",
                       vmin=0, vmax=unc_max_all)
        ax.scatter(obs_pos[:,0], obs_pos[:,1], c="cyan", s=12,
                   edgecolors="black", linewidths=0.3, zorder=5, alpha=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.set_title(f"SSS sigma (psu)", color="white", fontsize=9,
                     fontweight="bold", pad=4)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)

        # Row 2: meridional uncertainty profile (averaged over x)
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        y_ax = np.arange(NY)
        ax.fill_betweenx(y_ax, 0, T_s.mean(axis=0),
                         color="#fc8d59", alpha=0.7, label="SST sigma")
        ax.fill_betweenx(y_ax, 0, S_s.mean(axis=0),
                         color="#6baed6", alpha=0.5, label="SSS sigma")
        # Sensor positions on the profile
        for (_, yp) in obs_pos:
            ax.axhline(yp, color="cyan", lw=0.4, alpha=0.4)
        ax.set_title(f"Meridional profile of sigma (x-average)", color="white",
                     fontsize=8, fontweight="bold", pad=4)
        ax.set_xlabel("sigma moyen", color="white", fontsize=7)
        ax.set_ylabel("y (latitude)", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        ax.legend(fontsize=7, labelcolor="white", facecolor="#0a1628")
        ax.grid(True, alpha=0.2, color="white", axis="x")

    fig.text(0.5, 0.97,
             "AE-UNet - Uncertainty vs Network Density  "
             "(cyan = sensors  |  shared colour scale)",
             ha="center", color="white", fontsize=12, fontweight="bold")

    out = out_dir / "vae_uncertainty_density.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Density/uncertainty figure -> {out}")
    """
    Figure 1 — Reconstruction vs Nature Run.

    4 rows x 4 columns:
      Columns:  Ground truth | Reconstruction (mu) | Error | Uncertainty (sigma)
      Ligne 1 :  SST
      Ligne 2 :  SSS
      Ligne 3 :  SST (diff instant, + dense)
      Ligne 4 :  SSS (diff instant, + dense)
    """
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    sal_cmap = LinearSegmentedColormap.from_list("sal",
        ["#003c30","#01665e","#35978f","#80cdc1","#f5f5f5",
         "#dfc27d","#bf812d","#8c510a","#543005"], N=256)
    unc_cmap = "YlOrRd"
    err_cmap = "RdBu_r"

    model.eval()
    T_n = (T - norm["T_mean"]) / norm["T_std"]
    S_n = (S - norm["S_mean"]) / norm["S_std"]

    # Two snapshots: dense (30 obs) and sparse (8 obs)
    # Time steps chosen relative to the nature run length: indices 50 and 150
    # were hard-coded and crashed as soon as nt < 151.
    nt_ = len(T_n)
    t_a, t_b = min(nt_ - 1, nt_ // 4), min(nt_ - 1, 3 * nt_ // 4)
    scenarios = [
        (f"t={t_a}  N=30 obs.", t_a, 30),
        (f"t={t_b}  N=8 obs.",  t_b, 8),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 18), facecolor="#0a1628")
    col_titles = ["Ground truth", "Reconstruction (mu)", "Error |pred - true|", "Uncertainty (MC sigma)"]

    def cell(ax, data, cmap, vmin=None, vmax=None, title="", label="",
             norm_obj=None):
        kw = dict(cmap=cmap, origin="lower", aspect="auto", interpolation="bilinear")
        if norm_obj is not None:
            kw["norm"] = norm_obj
        elif vmin is not None:
            kw["vmin"] = vmin; kw["vmax"] = vmax
        im = ax.imshow(data.T, **kw)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.set_title(title, color="white", fontsize=8, pad=4)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(label, color="white", fontsize=6)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
        return im

    for row_pair, (desc, t, n_obs) in enumerate(scenarios):
        T_t = T_n[t]; S_t = S_n[t]

        # Random observation mask with n_obs sensors
        mask_np = np.zeros((NX, NY), dtype=np.float32)
        for (px, py) in sample_separated_positions(
                NX, NY, n_obs, rng=np.random.default_rng(t + 100)):
            mask_np[px, py] = 1.0

        T_obs = T_t * mask_np
        S_obs = S_t * mask_np
        x_in  = torch.from_numpy(
            np.stack([T_obs, S_obs, mask_np])[None]).to(DEVICE)

        # Reconstruction MC
        recon_mean, recon_std, _ = model.reconstruct_with_uncertainty(
            x_in, n_samples=n_samples)
        recon_mean = recon_mean[0].cpu().numpy()  # (2, NX, NY)
        recon_std  = recon_std[0].cpu().numpy()

        # Back to physical units for display
        T_true_phys  = T_t * norm["T_std"] + norm["T_mean"]
        S_true_phys  = S_t * norm["S_std"] + norm["S_mean"]
        T_pred_phys  = recon_mean[0] * norm["T_std"] + norm["T_mean"]
        S_pred_phys  = recon_mean[1] * norm["S_std"] + norm["S_mean"]
        T_err_phys   = np.abs(T_pred_phys - T_true_phys)
        S_err_phys   = np.abs(S_pred_phys - S_true_phys)
        T_unc_phys   = recon_std[0] * norm["T_std"]
        S_unc_phys   = recon_std[1] * norm["S_std"]

        vT = (T_true_phys.min(), T_true_phys.max())
        vS = (S_true_phys.min(), S_true_phys.max())

        row_T = row_pair * 2
        row_S = row_pair * 2 + 1

        prefix_T = f"SST — {desc}"
        prefix_S = f"SSS — {desc}"

        # -- SST --
        cell(axes[row_T,0], T_true_phys, ocean_cmap, *vT,
             title=f"SST | Verite  |  {desc}", label="degC")
        # Overlay buoys
        obs_pos = np.argwhere(mask_np > 0.5)
        axes[row_T,0].scatter(obs_pos[:,0], obs_pos[:,1], c="white", s=8,
                              marker="x", linewidths=0.6, zorder=5, alpha=0.8)

        cell(axes[row_T,1], T_pred_phys, ocean_cmap, *vT,
             title=f"SST | Reconstruction  |  {desc}", label="degC")

        err_lim = T_err_phys.max()
        cell(axes[row_T,2], T_err_phys, "hot", 0, err_lim,
             title=f"SST | Error |pred - true|  RMSE={T_err_phys.mean():.3f}", label="degC")

        cell(axes[row_T,3], T_unc_phys, unc_cmap, 0, T_unc_phys.max(),
             title=f"SST | Incertitude sigma MC (N={n_samples})", label="degC")

        # -- SSS --
        cell(axes[row_S,0], S_true_phys, sal_cmap, *vS,
             title=f"SSS | Verite  |  {desc}", label="psu")
        axes[row_S,0].scatter(obs_pos[:,0], obs_pos[:,1], c="white", s=8,
                              marker="x", linewidths=0.6, zorder=5, alpha=0.8)

        cell(axes[row_S,1], S_pred_phys, sal_cmap, *vS,
             title=f"SSS | Reconstruction  |  {desc}", label="psu")

        s_err_lim = S_err_phys.max()
        cell(axes[row_S,2], S_err_phys, "hot", 0, s_err_lim,
             title=f"SSS | Error  RMSE={S_err_phys.mean():.3f}", label="psu")

        cell(axes[row_S,3], S_unc_phys, unc_cmap, 0, S_unc_phys.max(),
             title=f"SSS | Incertitude sigma MC (N={n_samples})", label="psu")

    # Column titles
    for j, ct in enumerate(col_titles):
        axes[0, j].set_xlabel("")
        fig.text(0.12 + j * 0.22, 0.97, ct,
                 ha="center", color="#6baed6", fontsize=10, fontweight="bold")

    fig.text(0.5, 0.995,
             "VAE-UNet — Reconstruction vs Nature Run  (x = position observee)",
             ha="center", color="white", fontsize=13, fontweight="bold")
# =============================================================================
#  HELPERS — RMSE MC
# =============================================================================

@torch.no_grad()
def _compute_rmse_mc(model, T_n_t, S_n_t, positions, norm, n_mc=8):
    """RMSE over unobserved pixels at one time step, averaged over n_mc MC draws."""
    mask = np.zeros((NX, NY), dtype=np.float32)
    T_obs = np.zeros_like(mask); S_obs = np.zeros_like(mask)
    ns_T = OBS_NOISE_T / (norm["T_std"] + 1e-9)
    ns_S = OBS_NOISE_S / (norm["S_std"] + 1e-9)
    for (x, y) in positions:
        mask[x, y] = 1.0
        T_obs[x, y] = T_n_t[x, y] + np.random.normal(0, ns_T)
        S_obs[x, y] = S_n_t[x, y] + np.random.normal(0, ns_S)
    x_in = torch.from_numpy(np.stack([T_obs, S_obs, mask])[None]).to(DEVICE)
    rm, _, _ = model.reconstruct_with_uncertainty(x_in, n_samples=n_mc)
    pred = rm[0].cpu().numpy()
    y_true = np.stack([T_n_t, S_n_t])
    sq = (pred - y_true) ** 2
    return float(np.sqrt((sq * (1 - mask[None])).mean()))


# =============================================================================
#  SCORING — Leave-One-Out


def score(args):
    print("=" * 62)
    print("  Brick 1 - AE scoring")
    print("=" * 62)

    ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = ObservabilityVAE(
        base_ch=ckpt["args"]["base_ch"],
        latent_ch=ckpt["args"]["latent_ch"],
        dropout_p=ckpt["args"].get("dropout_p", 0.1),
        cond_dim=ckpt["args"].get("cond_dim", 32)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Model loaded: {args.checkpoint}")

    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=args.nt, seed=args.seed_ocean)
    norm = ckpt["norm"]

    # The model works on normalised fields: _compute_rmse_mc expects T_n / S_n,
    # not physical fields. (the previous code called a non-existent function
    # _rmse_unobs, on raw fields -> NameError)
    T_n = (T - norm["T_mean"]) / norm["T_std"]
    S_n = (S - norm["S_mean"]) / norm["S_std"]

    rng = np.random.default_rng(args.seed_buoys)
    positions = sample_separated_positions(NX, NY, N_BUOYS, rng=rng)

    # Leave-One-Out
    print("\n  Leave-One-Out...")
    t_idx = rng.choice(len(T), min(10, len(T)), replace=False)
    rmse_full = np.mean([_compute_rmse_mc(model, T_n[t], S_n[t], positions,
                                          norm, n_mc=args.n_mc_val)
                         for t in t_idx])
    print(f"  RMSE, full network : {rmse_full:.4f}")

    loo_scores = {}
    for i, pos in enumerate(positions):
        sub = [p for j, p in enumerate(positions) if j != i]
        rmse_loo = np.mean([_compute_rmse_mc(model, T_n[t], S_n[t], sub,
                                             norm, n_mc=args.n_mc_val)
                            for t in t_idx])
        loo_scores[i] = {"position": list(pos), "delta_rmse": float(rmse_loo - rmse_full)}
        print(f"  Sensor {i:2d} @ {pos} | delta={loo_scores[i]['delta_rmse']:+.4f}")

    out_dir = Path(args.output_dir)
    with open(out_dir / "vae_loo_scores.json", "w") as f:
        json.dump(loo_scores, f, indent=2)
    print(f"  LOO scores -> {out_dir}/vae_loo_scores.json")


# =============================================================================
#  POINT D ENTREE
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="AE-UNet v4 MC-Dropout OED")
    p.add_argument("--train",        action="store_true")
    p.add_argument("--score",        action="store_true")
    p.add_argument("--figures",      action="store_true")
    p.add_argument("--report",       action="store_true",
                   help="Write a .txt report with the key metrics")
    p.add_argument("--seed_ocean",   type=int,   default=42,
                   help="Nature run seed (used by --report)")
    p.add_argument("--seed_buoys",   type=int,   default=7,
                   help="Buoy network seed (used by --report)")
    p.add_argument("--checkpoint",   type=str,   default="outputs/vae_best.pt")
    p.add_argument("--output_dir",   type=str,   default="outputs")
    p.add_argument("--nt",           type=int,   default=NT,
                   help="Nature run length in days")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--base_ch",      type=int,   default=32)
    p.add_argument("--latent_ch",    type=int,   default=64)
    p.add_argument("--cond_dim",     type=int,   default=32)
    p.add_argument("--dropout_p",    type=float, default=0.1,
                   help="MC-Dropout rate p (kept active at inference too)")
    p.add_argument("--w_unobs",      type=float, default=4.0)
    p.add_argument("--lambda_grad",  type=float, default=0.5)
    p.add_argument("--lambda_spec",  type=float, default=0.0)
    p.add_argument("--lambda_ts",    type=float, default=0.0)
    p.add_argument("--huber_delta",  type=float, default=0.5)
    p.add_argument("--beta_max",     type=float, default=0.0)
    p.add_argument("--n_obs_min",    type=int,   default=10)
    p.add_argument("--n_obs_max",    type=int,   default=80)
    p.add_argument("--n_mc_val",     type=int,   default=15)
    p.add_argument("--n_mc",         type=int,   default=60)
    return p.parse_args()


if __name__ == "__main__":
    from datetime import datetime
    args = parse_args()
    if not any([args.train, args.score, args.figures]):
        print("Usage: python 01_autoencoder.py --train [--figures] [--score] [--report]")
        import sys; sys.exit(0)

    if args.train:
        train(args)

    if args.score or args.figures:
        ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        model = ObservabilityVAE(
            base_ch=ckpt["args"]["base_ch"],
            latent_ch=ckpt["args"]["latent_ch"],
            dropout_p=ckpt["args"].get("dropout_p", 0.1),
            cond_dim=ckpt["args"].get("cond_dim", 32)).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        norm = ckpt["norm"]

        print("  Generating the nature run for the figures...")
        gen = SyntheticOceanGenerator()
        T, S = gen.generate_dataset(nt=args.nt, seed=args.seed_ocean)

        if args.figures:
            print("\n  Figure 1: existing network evaluation (gaps + LOO)...")
            plot_network_evaluation(model, T, S, norm, args, n_samples=args.n_mc)
            print("\n  Figure 2: uncertainty vs network density...")
            plot_uncertainty_maps(model, T, S, norm, args, n_samples=args.n_mc)

        if args.score:
            score(args)

    if args.report:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output_dir)
        # Reload metrics from the checkpoint if available
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            saved_args = ckpt.get("args", {})
        except Exception:
            saved_args = {}
        lines = [
            "=" * 68,
            "  Brick 1 - AE-UNet MC-Dropout - Report",
            f"  Generated on : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 68, "",
            "-- REPRODUCIBILITY --------------------------------------------------",
            f"  seed_ocean  : {args.seed_ocean}",
            f"  seed_buoys  : {args.seed_buoys}",
            "",
            "-- HYPERPARAMETERS --------------------------------------------------",
            f"  epochs      : {args.epochs}",
            f"  base_ch     : {args.base_ch}",
            f"  latent_ch   : {args.latent_ch}",
            f"  dropout_p   : {args.dropout_p}",
            f"  w_unobs     : {args.w_unobs}",
            f"  lambda_grad : {args.lambda_grad}",
            f"  huber_delta : {args.huber_delta}",
            f"  n_obs_min   : {args.n_obs_min}  n_obs_max : {args.n_obs_max}",
            "",
            "── FICHIERS PRODUITS ────────────────────────────────────────────────",
        ]
        for f in sorted(out.iterdir()):
            if f.suffix in {".pt", ".png", ".gif"}:
                lines.append(f"  {f.name:<44} {f.stat().st_size//1024:>5} KB")
        lines += ["", "=" * 68]
        rpt = out / f"report_ae_{ts}.txt"
        rpt.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  AE report -> {rpt}")
