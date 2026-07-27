"""
==========================================================================
  BRIQUE 1 — AE-UNet v4 d'Observabilite (MC-Dropout + ObsGate + FiLM)
==========================================================================

Pourquoi v4 est différent des versions précédentes ?
─────────────────────────────────────────────────────
v1/v2/v3 utilisaient un VAE (reparamétrisation z = μ + ε·σ).
Le log de training v3 montrait une stagnation RMSE_unobs ≈ 0.185 dès ep40.
Cause : le bruit ε~N(0,I) injecté à chaque forward de training crée un
plancher de RMSE que le modèle ne peut pas franchir, peu importe les
améliorations architecturales.

Solution v4 : AE déterministe + MC-Dropout pour l'incertitude
───────────────────────────────────────────────────────────────
- MC-Dropout (Gal & Ghahramani 2016) : dropout actif aussi à l'inférence
  → N passes forward → variance des prédictions = incertitude épistémique
  → même qualité d'incertitude qu'un VAE, bien meilleur RMSE

Nouveautés architecturales :
───────────────────────────
1. ObsGate sur chaque skip-connexion
   Gate σ(conv(mask_downsampled)) module les features du skip selon la
   densité locale d'observations. Le décodeur sait quelle zone est observée.

2. GroupNorm remplace BatchNorm
   Compatible avec batch_size=1 à l'inférence MC (BN crashe avec B=1).

3. Huber loss (δ=0.5) remplace MSE
   Robuste aux observations bruitées et fronts mal positionnés.
   Gradients bornés → convergence plus stable.

4. Retrait de L_spec et L_ts
   Dans v3 : L_spec ≈ 0.007 à la fin → contribution nulle.
   Ces termes diluaient les gradients de reconstruction principale.

5. FiLM conditioning et deep supervision conservés.

Usage :
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
from data.loader import load_ocean, add_data_args
from data.dataset import (build_datasets, BuoySampler,
                          masked_rmse, per_channel_rmse)


# =============================================================================
#  BLOCS DE BASE
# =============================================================================

class MCDropout2d(nn.Module):
    """
    Dropout spatial TOUJOURS actif (training et inférence).
    C'est le secret de MC-Dropout : à l'inférence, on fait N passes
    avec dropout ON → variance des prédictions = incertitude épistémique.
    (Gal & Ghahramani 2016 — Bayesian deep learning via dropout)
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        # training=True force le dropout même en mode eval
        return F.dropout2d(x, p=self.p, training=True)


class ResDoubleConv(nn.Module):
    """Double conv résiduelle + MC-Dropout spatial."""
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),   # GroupNorm > BatchNorm
            nn.GELU(),                               # pour MC-Dropout (batch size 1)
            MCDropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return self.net(x) + self.skip(x)

    # Note: GroupNorm remplace BatchNorm car MC-Dropout à l'inférence
    # peut être appelé avec batch_size=1, ce qui fait crasher BatchNorm.


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2),
                                 ResDoubleConv(in_ch, out_ch, dropout_p))
    def forward(self, x): return self.net(x)


class ObsGate(nn.Module):
    """
    Skip-gating conditionné sur la densité d'observations locale.

    Pour chaque skip-connexion (niveau k), on downsampe le masque
    d'observations à la résolution de ce niveau et on calcule un gate
    σ(conv(mask_ds)) ∈ [0,1]. Ce gate module les features du skip :
      - Gate≈1 dans les zones bien observées → skip passe fort
        (le décodeur peut faire confiance aux features encodées)
      - Gate≈0 dans les zones lacunaires → skip atténué
        (le décodeur doit interpoler depuis le bottleneck)

    Impact direct sur RMSE : le décodeur ne "confond" plus les zones
    observées avec les zones à interpoler.
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
    """Bloc Up avec FiLM conditioning (N_obs) + ObsGate sur skip."""
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
        skip = self.gate(skip, mask_ds)          # gate selon observations locales
        h    = self.conv(torch.cat([skip, x], dim=1))
        gam, bet = self.film(cond).chunk(2, dim=-1)
        gam  = gam.view(-1, h.shape[1], 1, 1)
        bet  = bet.view(-1, h.shape[1], 1, 1)
        return h * (1 + gam) + bet


class CBAM(nn.Module):
    """Attention canal + spatiale au bottleneck."""
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
#  AE-UNet v4 — MC-Dropout + ObsGate + FiLM
# =============================================================================

class ObservabilityVAE(nn.Module):
    """
    AE-UNet déterministe avec incertitude par MC-Dropout.

    Pourquoi abandonner la reparamétrisation VAE ?
    ─────────────────────────────────────────────────
    Dans le log v3 : RMSE stagne à 0.185 dès ep40, malgré 160 époques de
    plus. Cause : le bruit ε~N(0,I) injecté à CHAQUE forward pass de
    training crée un plancher de RMSE (~0.03-0.05 en absolu) que le modèle
    ne peut pas franchir.

    MC-Dropout (Gal & Ghahramani 2016) :
    ────────────────────────────────────
    - Training  : dropout actif → régularisation
    - Inférence : dropout TOUJOURS actif → N passes → variance = incertitude
    - Mathématiquement équivalent à une inférence variationnelle approximative
    - RMSE bien meilleur car pas de bruit latent injected

    Nouveautés v4 :
    ───────────────
    1. MC-Dropout remplace reparamétrisation VAE
    2. ObsGate sur chaque skip : gate conditionné sur densité observations locale
       → décodeur sait distinguer zones observées / lacunaires
    3. GroupNorm remplace BatchNorm (compatible batch_size=1 à l'inférence MC)
    4. FiLM conditioning maintenu (N_obs)
    5. Deep supervision maintenu
    """

    def __init__(self, in_ch=3, out_ch=2, base_ch=32, latent_ch=64,
                 dropout_p=0.1, cond_dim=32):
        super().__init__()
        bc = base_ch
        dp = dropout_p
        self.latent_ch = latent_ch
        self.cond_dim  = cond_dim
        self.dropout_p = dropout_p

        # ── Encodeur 4 niveaux ─────────────────────────────────────────────
        self.inc   = ResDoubleConv(in_ch, bc,    dp)
        self.down1 = Down(bc,    bc*2,  dp)
        self.down2 = Down(bc*2,  bc*4,  dp)
        self.down3 = Down(bc*4,  bc*8,  dp)
        self.down4 = Down(bc*8,  bc*16, dp)

        # ── FiLM embedding ─────────────────────────────────────────────────
        self.cond_embed = nn.Sequential(
            nn.Linear(1, cond_dim), nn.GELU(),
            nn.Linear(cond_dim, cond_dim), nn.GELU(),
        )

        # ── Bottleneck déterministe + CBAM ─────────────────────────────────
        # Pas de reparamétrisation — le "latent" est un simple vecteur de features
        self.cbam   = CBAM(bc*16)
        self.to_z   = nn.Conv2d(bc*16, latent_ch, 1)   # encode déterministe
        self.from_z = nn.Conv2d(latent_ch, bc*16, 1)

        # ── Décodeur FiLM 4 niveaux + ObsGate ─────────────────────────────
        self.up1 = FiLMUp(bc*16, bc*8,  bc*8,  cond_dim, dp)
        self.up2 = FiLMUp(bc*8,  bc*4,  bc*4,  cond_dim, dp)
        self.up3 = FiLMUp(bc*4,  bc*2,  bc*2,  cond_dim, dp)
        self.up4 = FiLMUp(bc*2,  bc,    bc,    cond_dim, dp)
        self.head = nn.Conv2d(bc, out_ch, 1)

        # ── Têtes de deep supervision ──────────────────────────────────────
        self.aux1 = nn.Conv2d(bc*8, out_ch, 1)
        self.aux2 = nn.Conv2d(bc*4, out_ch, 1)
        self.aux3 = nn.Conv2d(bc*2, out_ch, 1)

    def _get_cond(self, x):
        # Le masque est TOUJOURS le dernier canal d'entrée. L'ancien x[:, 2:3]
        # supposait implicitement in_ch=3 ; faux dès qu'on observe plus de deux
        # variables (OBSERVED_VARS = GLORYS_VARIABLES → in_ch=5).
        mask = x[:, -1:]
        return self.cond_embed(mask.mean(dim=[2, 3]))   # (B, cond_dim)

    def _downsample_mask(self, mask, target):
        """Downsampling NN du masque à la résolution d'une feature map cible."""
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
        """Décodage avec FiLM + ObsGate sur chaque skip."""
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
        # Pour compatibilité API avec le reste du code (score, figures)
        # On retourne mu=z, logvar=zeros (pas de KL)
        return pred, z, torch.zeros_like(z), aux

    @torch.no_grad()
    def reconstruct_with_uncertainty(self, x, n_samples=50):
        """
        Incertitude par MC-Dropout.

        Le modèle est en mode EVAL (BN/LN figés) mais MCDropout2d
        force dropout=True → chaque passe donne une prédiction légèrement
        différente. La variance = incertitude épistémique.

        Avantage vs VAE : pas de bruit latent → la moyenne des prédictions
        est beaucoup plus proche de la vérité terrain.
        """
        mask = x[:, 2:3]
        cond = self._get_cond(x)
        z, skips = self.encode(x)
        # MC passes : dropout actif via MCDropout2d
        samples = [self.decode(z, skips, cond, mask)[0]
                   for _ in range(n_samples)]
        stack = torch.stack(samples)
        return stack.mean(0), stack.std(0), z

    def get_latent(self, x):
        z, _ = self.encode(x)
        return z.flatten(1)


# =============================================================================
#  LOSS v4 : Huber + gradient + deep supervision  (spectral/T-S retirés)
# =============================================================================

class VAELoss(nn.Module):
    """
    Loss v4 — focus sur le RMSE_unobs :

        L = L_recon + λ_grad·L_grad + Σ w_k·L_aux_k   (pas de KL, pas de spectral)

    Huber loss (δ=0.5) remplace MSE :
        MSE pénalise très fortement les outliers (observations bruitées,
        front mal positionné). Huber est quadratique pour |e|<δ et linéaire
        au-delà → gradients bornés → convergence plus stable sur les fronts.

    Retrait de L_spec et L_ts :
        Dans le log v3, L_spec = 0.007 à la fin → contribution quasi-nulle,
        les gradients correspondants "polluent" l'optimisation principale.
        L_ts : même constat (contrainte utile en physique mais poids trop faible
        pour compenser son effet de dilution sur le gradient principal).
    """
    def __init__(self, w_obs=1.0, w_unobs=4.0, beta_max=0.0,
                 lambda_grad=0.5, lambda_spec=0.0, lambda_ts=0.0,
                 huber_delta=0.5):
        super().__init__()
        self.w_obs       = w_obs
        self.w_unobs     = w_unobs
        self.beta_max    = beta_max          # conservé à 0 — pas de KL
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

    def _recon_loss(self, pred, target, mask, sea=None):
        """
        `sea` : masque océanique (1, H, W) ou None.

        Sans lui, les pixels de terre — remplis par une constante, donc
        reconstruits parfaitement — feraient baisser artificiellement la loss
        et le RMSE. Inutile sur une fenêtre 100 % océanique, indispensable dès
        qu'on repasse sur un domaine côtier.
        """
        err = self._huber(pred - target)
        if sea is None:
            return (self.w_obs * (err * mask).mean()
                    + self.w_unobs * (err * (1 - mask)).mean())
        w_o = mask * sea
        w_u = (1 - mask) * sea
        return (self.w_obs * (err * w_o).sum() / w_o.sum().clamp_min(1.0)
                + self.w_unobs * (err * w_u).sum() / w_u.sum().clamp_min(1.0))

    def forward(self, pred, target, mask, mu, logvar, beta=1.0, aux_preds=None,
                sea=None):
        loss_recon = self._recon_loss(pred, target, mask, sea)

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
                sea_ds  = (None if sea is None
                           else F.interpolate(sea, size=(H, W), mode="nearest"))
                loss_aux = loss_aux + w * self._recon_loss(aux_up, target,
                                                           mask_ds, sea_ds)

        # KL = 0 (pas de reparamétrisation)
        kl       = torch.tensor(0.0, device=pred.device)
        loss_spec = torch.tensor(0.0, device=pred.device)

        total = loss_recon + self.lambda_grad * loss_grad + loss_aux
        return total, loss_recon, kl, loss_aux, loss_spec


# =============================================================================
#  ENTRAÎNEMENT
# =============================================================================

def train(args):
    print("=" * 62)
    print("  Brique 1 — Entraînement AE-UNet v4 (MC-Dropout + ObsGate)")
    print("=" * 62)

    print("\n[1/4] Chargement du champ oceanique...")
    fields, channels, sea_mask, data_info = load_ocean(args)
    print(f"  {data_info['source']} | {fields.shape} | canaux={channels}")

    # Pas d'augmentation par flip : avec un champ geophysique reel, un flip
    # produit une geometrie non physique (gradient meridien inverse, rotation
    # des courants incoherente avec Coriolis).
    train_ds, val_ds = build_datasets(fields, channels, split=0.8,
                                      sea_mask=sea_mask,
                                      observed_vars=OBSERVED_VARS,
                                      n_obs_min=args.n_obs_min,
                                      n_obs_max=args.n_obs_max)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    print(f"\n[2/4] Modele AE-UNet v4 "
          f"(base_ch={args.base_ch}, latent_ch={args.latent_ch}, "
          f"dropout={args.dropout_p}, cond_dim={args.cond_dim})...")
    model = ObservabilityVAE(in_ch=VAE_IN_CH, out_ch=VAE_OUT_CH,
                             base_ch=args.base_ch,
                             latent_ch=args.latent_ch,
                             dropout_p=args.dropout_p,
                             cond_dim=args.cond_dim).to(DEVICE)
    print(f"  Canaux : in={VAE_IN_CH} (observes {OBSERVED_VARS} + masque) "
          f"-> out={VAE_OUT_CH} ({channels})")
    npar = sum(p.numel() for p in model.parameters())
    print(f"  Parametres : {npar:,}")

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

    print(f"\n[3/4] Entrainement {args.epochs} ep | "
          f"Huber δ={args.huber_delta} | λ_grad={args.lambda_grad} | "
          f"MC-Dropout p={args.dropout_p} | augment=flip | "
          f"n_mc_val={args.n_mc_val}...")
    history = {"train_loss": [], "val_rmse_unobs": [], "kl": [],
               "loss_spec": [], "loss_aux": [], "lr": []}
    best_val = np.inf

    for epoch in range(1, args.epochs + 1):
        beta = min(1.0, epoch / warmup)

        model.train()
        ep_loss = ep_kl = ep_spec = ep_aux = 0.0
        for x, y, mask, sea in train_ld:
            x, y, mask, sea = (x.to(DEVICE), y.to(DEVICE),
                               mask.to(DEVICE), sea.to(DEVICE))
            pred, mu, logvar, aux_preds = model(x)
            loss, _, kl, l_aux, l_spec = criterion(
                pred, y, mask, mu, logvar, beta, aux_preds, sea=sea)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item(); ep_kl += kl.item()
            ep_spec += l_spec.item() if torch.is_tensor(l_spec) else float(l_spec)
            ep_aux  += l_aux.item()  if torch.is_tensor(l_aux)  else float(l_aux)
        scheduler.step()
        n = len(train_ld)
        ep_loss /= n; ep_kl /= n; ep_spec /= n; ep_aux /= n

        # Validation MC-moyennée : on moyenne n_mc_val passes pour avoir
        # le vrai RMSE du modèle (pas le biais d'un seul tirage dropout)
        model.eval()
        val_rmses = []
        val_rmse_by_density = {"sparse": [], "medium": [], "dense": []}
        ch_sq = np.zeros(VAE_OUT_CH); ch_n = 0.0
        with torch.no_grad():
            for x, y, mask, sea in val_ld:
                x, y, mask, sea = (x.to(DEVICE), y.to(DEVICE),
                                   mask.to(DEVICE), sea.to(DEVICE))
                preds = torch.stack([model(x)[0] for _ in range(args.n_mc_val)])
                pred_mean = preds.mean(0)
                sq = (pred_mean - y) ** 2
                w = (1 - mask) * sea            # non observe ET en mer
                # Accumulation par canal, en unites normalisees
                ch_sq += (sq * w).sum(dim=(0, 2, 3)).cpu().numpy()
                ch_n  += float(w.sum().item())
                for b in range(x.shape[0]):
                    n_obs_b = int(mask[b].sum().item())
                    wb = w[b]
                    rmse_b = float(torch.sqrt(
                        (sq[b] * wb).sum() / wb.sum().clamp_min(1.0)).item())
                    val_rmses.append(rmse_b)
                    if n_obs_b < 20:
                        val_rmse_by_density["sparse"].append(rmse_b)
                    elif n_obs_b < 50:
                        val_rmse_by_density["medium"].append(rmse_b)
                    else:
                        val_rmse_by_density["dense"].append(rmse_b)
        val_rmse = float(np.mean(val_rmses))

        # RMSE par canal en unites PHYSIQUES (degC, PSU, m/s).
        # Une RMSE agregee sur des canaux heterogenes n'a aucune
        # interpretation physique : c'est ce tableau qu'il faut presenter.
        val_rmse_phys = {c: float(np.sqrt(ch_sq[i] / max(ch_n, 1.0))
                                  * train_ds.std[i])
                         for i, c in enumerate(channels)}

        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(ep_loss)
        history["val_rmse_unobs"].append(val_rmse)
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
                  f"Laux={ep_aux:.4f} | lr={cur_lr:.2e}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_rmse_phys = dict(val_rmse_phys)
            torch.save({
                "model_state": model.state_dict(),
                "args":  vars(args),
                "channels": channels,
                "sea_mask": sea_mask,
                "data_info": data_info,
                "stats": {"mean": train_ds.mean, "std": train_ds.std},
                "rmse_phys": best_rmse_phys,
                # `norm` conserve pour compatibilite avec les fonctions de figures
                "norm":  {"T_mean": float(train_ds.mean[0]),
                          "T_std":  float(train_ds.std[0]),
                          "S_mean": float(train_ds.mean[1]),
                          "S_std":  float(train_ds.std[1])}
            }, out_dir / "vae_best.pt")

    print(f"\n  Meilleur RMSE val (non-obs) : {best_val:.4f}")
    print("  RMSE par canal, unites physiques :")
    _u = {"thetao": "degC", "so": "PSU", "uo": "m/s", "vo": "m/s"}
    for c, v in best_rmse_phys.items():
        var = c.rsplit("_z", 1)[0]
        print(f"    {c:<12} {v:8.4f} {_u.get(var, '')}")

    print("\n[4/4] Sauvegarde des courbes...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 4), facecolor="#0a1628")
    data = [("Loss totale",        "train_loss",      "#6baed6"),
            ("RMSE val (non-obs)", "val_rmse_unobs",  "#fc8d59"),
            ("KL divergence",      "kl",              "#74c476"),
            ("Loss spectrale",     "loss_spec",       "#ffd93d"),
            ("Deep supervision",   "loss_aux",        "#cc99ff")]
    for ax, (lbl, k, col) in zip(axes, data):
        ax.plot(history[k], color=col, lw=1.8)
        ax.set_title(lbl, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoque", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
        ax.grid(True, alpha=0.2, color="white")
    fig.tight_layout(pad=2)
    fig.savefig(out_dir / "vae_training_curves.png", dpi=130,
                facecolor="#0a1628", bbox_inches="tight")
    plt.close()
    print(f"  Courbes -> {out_dir}/vae_training_curves.png")
    print(f"  Checkpoint -> {out_dir}/vae_best.pt")


# =============================================================================
#  FIGURE 1 — Évaluation d'un réseau existant
#
#  Objectif : étant donné un réseau de N capteurs à positions fixes,
#  montrer où le réseau couvre bien / mal le domaine, et quantifier
#  la contribution de chaque capteur.
#
#  Layout (2 lignes × 4 colonnes) :
#    Ligne 1 SST : champ vrai + capteurs | reconstruction | sigma MC | zones lacunaires
#    Ligne 2 SSS : champ vrai + capteurs | reconstruction | sigma MC | score LOO capteurs
# =============================================================================

@torch.no_grad()
def plot_network_evaluation(model, T, S, norm, args,
                            positions=None, n_samples=80, n_loo_t=8):
    """
    Figure d'évaluation du réseau de capteurs.

    Paramètres
    ----------
    positions : liste de (x, y) en coordonnées pixel, ou None
        Si None, un réseau de N_BUOYS capteurs est généré avec args.seed_buoys.
    n_samples  : tirages MC pour l'incertitude
    n_loo_t    : nombre d'instants pour le calcul LOO
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

    # ── Réseau de référence ────────────────────────────────────────────────────
    if positions is None:
        seed_b = getattr(args, "seed_buoys", 42)
        sea    = getattr(args, "sea_mask", None)
        positions = BuoySampler(NX, NY, N_BUOYS, sea_mask=sea,
                                min_dist=MIN_BUOY_DIST, rng=seed_b).positions
        print(f"  Réseau généré : {N_BUOYS} bouées (seed_buoys={seed_b}, "
              f"dist_min={MIN_BUOY_DIST} px)")
    positions = list(positions)
    n_sensors = len(positions)

    # Masque fixe du réseau
    mask_np = np.zeros((NX, NY), dtype=np.float32)
    for (x, y) in positions:
        mask_np[x, y] = 1.0
    obs_pos = np.array(positions)

    # Instant de référence pour les cartes
    t_ref = len(T) // 3
    T_t, S_t = T_n[t_ref], S_n[t_ref]
    T_obs = T_t * mask_np
    S_obs = S_t * mask_np
    x_in = torch.from_numpy(np.stack([T_obs, S_obs, mask_np])[None]).to(DEVICE)

    # Reconstruction + incertitude MC
    recon_mean, recon_std, _ = model.reconstruct_with_uncertainty(x_in, n_samples=n_samples)
    rm = recon_mean[0].cpu().numpy()
    rs = recon_std[0].cpu().numpy()

    T_true  = T_t * norm["T_std"] + norm["T_mean"]
    S_true  = S_t * norm["S_std"] + norm["S_mean"]
    T_pred  = rm[0] * norm["T_std"] + norm["T_mean"]
    S_pred  = rm[1] * norm["S_std"] + norm["S_mean"]
    T_sigma = rs[0] * norm["T_std"]
    S_sigma = rs[1] * norm["S_std"]

    # ── Carte des zones lacunaires ─────────────────────────────────────────────
    # Une zone est "lacunaire" si sigma > seuil (percentile 75).
    # On construit une carte binaire + un masque de distance au capteur le plus proche.
    from scipy.ndimage import distance_transform_edt

    dist_to_sensor = distance_transform_edt(1 - mask_np)   # distance en pixels
    dist_to_sensor_n = dist_to_sensor / dist_to_sensor.max()   # normalisé [0,1]

    # Incertitude combinée T+S (moyenne normalisée)
    T_sigma_n = T_sigma / (T_sigma.max() + 1e-9)
    S_sigma_n = S_sigma / (S_sigma.max() + 1e-9)
    combined_sigma = 0.5 * (T_sigma_n + S_sigma_n)

    # Score de couverture : zones lacunaires = sigma élevé ET loin d'un capteur
    gap_map = combined_sigma * dist_to_sensor_n   # ∈ [0, 1]
    gap_threshold = np.percentile(gap_map, 80)
    gap_binary = (gap_map > gap_threshold).astype(float)

    # ── 3 bouées proposées — maximisent la couverture lacunaire ───────────────
    # Algorithme glouton : à chaque étape, on place la bouée au maximum de
    # gap_map résiduel, puis on met à jour la distance au capteur le plus proche.
    from scipy.ndimage import distance_transform_edt as _edt
    proposed_positions = []
    gap_residual = gap_map.copy()
    mask_augmented = mask_np.copy()
    for _ in range(3):
        flat_idx = np.argmax(gap_residual)
        px, py   = np.unravel_index(flat_idx, gap_residual.shape)  # px∈[0,NX), py∈[0,NY)
        proposed_positions.append((int(px), int(py)))
        mask_augmented[px, py] = 1.0
        dist_new = _edt(1 - mask_augmented) / (dist_to_sensor.max() + 1e-9)
        gap_residual = combined_sigma * dist_new
    proposed_arr = np.array(proposed_positions)  # (3, 2) — (x, y) en coords pixel

    # ── LOO scores — contribution de chaque capteur ───────────────────────────
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
        loo_delta[i] = rmse_i - rmse_full   # >0 : ce capteur apporte de l'info

    # Normaliser les deltas pour l'affichage
    loo_colors = np.clip(loo_delta / (loo_delta.max() + 1e-9), 0, 1)

    # ── Figure ─────────────────────────────────────────────────────────────────
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

    # ── Ligne 1 : SST ──────────────────────────────────────────────────────────
    # [0,0] SST vrai + positions capteurs (colorées par LOO delta)
    ax00 = fig.add_subplot(gs[0, 0])
    cell(ax00, T_true, ocean_cmap, *vT,
         f"SST vrai + réseau ({n_sensors} capteurs)\n"
         f"couleur = contribution LOO (vert=fort, rouge=faible)",
         "°C", pts=obs_pos, pts_c=loo_colors, pts_s=55)

    # [0,1] SST reconstruction
    ax01 = fig.add_subplot(gs[0, 1])
    cell(ax01, T_pred, ocean_cmap, *vT,
         f"SST reconstruction VAE\n(t={t_ref},  RMSE_unobs={rmse_full:.3f})",
         "°C", pts=obs_pos, pts_c="white", pts_s=12)

    # [0,2] Incertitude SST
    ax02 = fig.add_subplot(gs[0, 2])
    cell(ax02, T_sigma, "YlOrRd", 0, T_sigma.max(),
         f"SST incertitude σ MC  (N={n_samples} tirages)\n"
         "rouge = zone mal contrainte par le réseau",
         "°C", pts=obs_pos, pts_c="cyan", pts_s=12,
         contour=gap_binary)

    # [0,3] Carte des zones lacunaires + 3 bouées proposées
    ax03 = fig.add_subplot(gs[0, 3])
    cell(ax03, gap_map, "inferno", 0, gap_map.max(),
         f"Zones lacunaires + 3 bouées proposées\n"
         f"(σ élevé × distance capteur)  —  {int(gap_binary.sum())} px critiques",
         "score", pts=obs_pos, pts_c="cyan", pts_s=15,
         contour=gap_binary)
    # Bouées proposées : étoiles jaunes numérotées
    for k, (px, py) in enumerate(proposed_arr):
        ax03.scatter(px, py, marker="*", s=320, c="#ffd93d",
                     edgecolors="black", linewidths=0.8, zorder=8)
        ax03.annotate(f"P{k+1}", (px, py),
                      textcoords="offset points", xytext=(6, 4),
                      fontsize=8, color="#ffd93d", fontweight="bold")

    # ── Ligne 2 : SSS ──────────────────────────────────────────────────────────
    # [1,0] SSS vrai
    ax10 = fig.add_subplot(gs[1, 0])
    cell(ax10, S_true, sal_cmap, *vS,
         "SSS vrai + positions capteurs", "psu",
         pts=obs_pos, pts_c="white", pts_s=12)

    # [1,1] SSS reconstruction
    ax11 = fig.add_subplot(gs[1, 1])
    cell(ax11, S_pred, sal_cmap, *vS,
         "SSS reconstruction VAE", "psu",
         pts=obs_pos, pts_c="white", pts_s=12)

    # [1,2] Incertitude SSS
    ax12 = fig.add_subplot(gs[1, 2])
    cell(ax12, S_sigma, "YlOrRd", 0, S_sigma.max(),
         "SSS incertitude σ MC", "psu",
         pts=obs_pos, pts_c="cyan", pts_s=12,
         contour=gap_binary)

    # [1,3] LOO barplot — contribution de chaque capteur
    ax13 = fig.add_subplot(gs[1, 3])
    ax13.set_facecolor("#050d1a")
    for sp in ax13.spines.values(): sp.set_edgecolor("#1a3a5c")

    idx_sort = np.argsort(loo_delta)[::-1]   # trié par contribution décroissante
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
    ax13.set_xlabel("Δ RMSE  (LOO − complet)", color="white", fontsize=8)
    ax13.set_title("Contribution LOO par capteur\n"
                   "vert = indispensable  |  rouge = redondant",
                   color="white", fontsize=8.5, fontweight="bold", pad=5)
    ax13.tick_params(colors="white", labelsize=6)
    ax13.grid(True, alpha=0.2, color="white", axis="x")

    # Seuil de redondance (delta < 5% du max)
    thr = loo_delta.max() * 0.05
    n_redondant = (loo_delta < thr).sum()
    ax13.axvline(thr, color="#ffd93d", lw=1, linestyle="--", alpha=0.7,
                 label=f"seuil 5%  ({n_redondant} redondants)")
    ax13.legend(fontsize=6, labelcolor="white", facecolor="#0a1628", loc="lower right")

    fig.text(0.5, 0.97,
             f"AE-UNet — Évaluation Réseau  ({n_sensors} capteurs)  "
             f"|  contour rouge = zones lacunaires  |  ★ = bouées proposées",
             ha="center", color="white", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "cyan = capteur existant  |  ★ jaune = bouée proposée (greedy gap)  "
             "|  couleur capteur ligne 1 = contribution LOO",
             ha="center", color="#8ab4d4", fontsize=8)

    out = out_dir / "vae_network_evaluation.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Figure évaluation réseau -> {out}")
    print(f"  Bouées proposées : " +
          "  ".join([f"P{k+1}=({px},{py})" for k,(px,py) in enumerate(proposed_arr)]))
    return loo_delta, gap_map, positions, proposed_arr


# =============================================================================
#  FIGURE 2 — Incertitude comparée sur différentes densités de réseau
#
#  Même réseau, on enlève des capteurs progressivement et on montre
#  comment l'incertitude augmente dans les zones déjà lacunaires.
# =============================================================================

@torch.no_grad()
def plot_uncertainty_maps(model, T, S, norm, args, n_samples=60):
    """
    Figure 2 — Evolution de l'incertitude en fonction de la densité réseau.

    3 colonnes : réseau Dense (N=40), Moyen (N=20), Clairsemé (N=8)
    Pour chaque : SST sigma | SSS sigma | profil d'incertitude méridionale
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

    configs = [("Dense   (N=40)", 40), ("Moyen   (N=20)", 20), ("Clairsemé (N=8)", 8)]

    # Échelle commune pour comparaison
    unc_max_all = 0.0
    results = []
    for (_, n_obs) in configs:
        np.random.seed(n_obs * 7)
        flat = np.zeros(NX * NY, dtype=np.float32)
        flat[np.random.choice(NX*NY, n_obs, replace=False)] = 1.0
        mask = flat.reshape(NX, NY)
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

        # Ligne 2 : profil méridional d'incertitude (moyen sur x)
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        y_ax = np.arange(NY)
        ax.fill_betweenx(y_ax, 0, T_s.mean(axis=0),
                         color="#fc8d59", alpha=0.7, label="SST sigma")
        ax.fill_betweenx(y_ax, 0, S_s.mean(axis=0),
                         color="#6baed6", alpha=0.5, label="SSS sigma")
        # Points capteurs sur le profil
        for (_, yp) in obs_pos:
            ax.axhline(yp, color="cyan", lw=0.4, alpha=0.4)
        ax.set_title(f"Profil méridional σ (moyen en x)", color="white",
                     fontsize=8, fontweight="bold", pad=4)
        ax.set_xlabel("sigma moyen", color="white", fontsize=7)
        ax.set_ylabel("y (latitude)", color="white", fontsize=7)
        ax.tick_params(colors="white", labelsize=6)
        ax.legend(fontsize=7, labelcolor="white", facecolor="#0a1628")
        ax.grid(True, alpha=0.2, color="white", axis="x")

    fig.text(0.5, 0.97,
             "VAE-UNet — Incertitude vs Densité Réseau  "
             "(cyan = capteurs  |  même échelle de couleur)",
             ha="center", color="white", fontsize=12, fontweight="bold")

    out = out_dir / "vae_uncertainty_density.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Figure densité/incertitude -> {out}")
    """
    Figure 1 — Reconstruction vs Nature Run.

    4 lignes x 4 colonnes :
      Colonne :  Verite terrain | Reconstruction (mu) | Erreur |  Incertitude (sigma)
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

    # Deux instants : dense (30 obs) et clairseme (8 obs).
    # Les indices etaient codes en dur (50 et 150), ce qui plante des que la
    # serie compte moins de 151 dates — cas courant avec un jeu GLORYS reel
    # limite a quelques mois. On les derive maintenant de la longueur reelle.
    nt_ = len(T)
    t_a, t_b = nt_ // 4, (3 * nt_) // 4
    scenarios = [
        (f"t={t_a}  N=30 obs.", t_a, 30),
        (f"t={t_b}  N=8 obs.",  t_b, 8),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 18), facecolor="#0a1628")
    col_titles = ["Verite terrain", "Reconstruction (mu)", "Erreur |pred - vrai|", "Incertitude (sigma MC)"]

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

        # Masque aleatoire avec n_obs observations
        np.random.seed(t + 100)
        flat = np.zeros(NX * NY, dtype=np.float32)
        flat[np.random.choice(NX * NY, n_obs, replace=False)] = 1.0
        mask_np = flat.reshape(NX, NY)

        T_obs = T_t * mask_np
        S_obs = S_t * mask_np
        x_in  = torch.from_numpy(
            np.stack([T_obs, S_obs, mask_np])[None]).to(DEVICE)

        # Reconstruction MC
        recon_mean, recon_std, _ = model.reconstruct_with_uncertainty(
            x_in, n_samples=n_samples)
        recon_mean = recon_mean[0].cpu().numpy()  # (2, NX, NY)
        recon_std  = recon_std[0].cpu().numpy()

        # Denormalisation pour affichage
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
        # Overlay bouees
        obs_pos = np.argwhere(mask_np > 0.5)
        axes[row_T,0].scatter(obs_pos[:,0], obs_pos[:,1], c="white", s=8,
                              marker="x", linewidths=0.6, zorder=5, alpha=0.8)

        cell(axes[row_T,1], T_pred_phys, ocean_cmap, *vT,
             title=f"SST | Reconstruction  |  {desc}", label="degC")

        err_lim = T_err_phys.max()
        cell(axes[row_T,2], T_err_phys, "hot", 0, err_lim,
             title=f"SST | Erreur |predict-vrai|  RMSE={T_err_phys.mean():.3f}", label="degC")

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
             title=f"SSS | Erreur  RMSE={S_err_phys.mean():.3f}", label="psu")

        cell(axes[row_S,3], S_unc_phys, unc_cmap, 0, S_unc_phys.max(),
             title=f"SSS | Incertitude sigma MC (N={n_samples})", label="psu")

    # Titres colonnes
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
    """RMSE (pixels non observés) sur un seul instant, moyenne sur n_mc tirages VAE."""
    mask = np.zeros((NX, NY), dtype=np.float32)
    T_obs = np.zeros_like(mask); S_obs = np.zeros_like(mask)
    ns_T = OBS_NOISE_STD / (norm["T_std"] + 1e-9)
    ns_S = OBS_NOISE_STD / (norm["S_std"] + 1e-9)
    for (x, y) in positions:
        mask[x, y] = 1.0
        T_obs[x, y] = T_n_t[x, y] + np.random.normal(0, ns_T)
        S_obs[x, y] = S_n_t[x, y] + np.random.normal(0, ns_S)
    x_in = torch.from_numpy(np.stack([T_obs, S_obs, mask])[None]).to(DEVICE)
    rm, _, _ = model.reconstruct_with_uncertainty(x_in, n_samples=n_mc)
    pred = rm[0].cpu().numpy()
    # Le modèle reconstruit VAE_OUT_CH canaux (T, S, u, v) mais cette métrique
    # ne porte que sur T et S : on restreint aux deux premiers canaux, sinon
    # le broadcast échoue silencieusement ou compare des grandeurs sans rapport.
    y_true = np.stack([T_n_t, S_n_t])
    pred = pred[:y_true.shape[0]]
    sq = (pred - y_true) ** 2
    return float(np.sqrt((sq * (1 - mask[None])).mean()))


# =============================================================================
#  SCORING — Leave-One-Out


def score(args):
    print("=" * 62)
    print("  Brique 1 — Scoring VAE")
    print("=" * 62)

    ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = ObservabilityVAE(
        in_ch=VAE_IN_CH, out_ch=VAE_OUT_CH,
        base_ch=ckpt["args"]["base_ch"],
        latent_ch=ckpt["args"]["latent_ch"],
        dropout_p=ckpt["args"].get("dropout_p", 0.1),
        cond_dim=ckpt["args"].get("cond_dim", 32)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Modele charge : {args.checkpoint}")

    fields, channels, sea_mask, _ = load_ocean(args)
    T, S = fields[:, 0], fields[:, 1]      # thetao_z0, so_z0
    norm = ckpt["norm"]

    positions = BuoySampler(NX, NY, N_BUOYS, sea_mask=sea_mask,
                            min_dist=MIN_BUOY_DIST,
                            rng=getattr(args, "seed_buoys", 7)).positions

    # Leave-One-Out
    print("\n  Leave-One-Out...")
    t_idx = np.random.choice(len(T), 10, replace=False)
    rmse_full = np.mean([_rmse_unobs(model, T[t], S[t], positions, norm)
                         for t in t_idx])
    print(f"  RMSE reseau complet : {rmse_full:.4f}")

    loo_scores = {}
    for i, pos in enumerate(positions):
        sub = [p for j, p in enumerate(positions) if j != i]
        rmse_loo = np.mean([_rmse_unobs(model, T[t], S[t], sub, norm)
                            for t in t_idx])
        loo_scores[i] = {"position": list(pos), "delta_rmse": float(rmse_loo - rmse_full)}
        print(f"  Capteur {i:2d} @ {pos} | delta={loo_scores[i]['delta_rmse']:+.4f}")

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
                   help="Produit un rapport .txt avec les métriques clés")
    p.add_argument("--seed_ocean",   type=int,   default=42,
                   help="Seed du nature run (pour --report)")
    p.add_argument("--seed_buoys",   type=int,   default=7,
                   help="Seed du réseau de bouées (pour --report)")
    p.add_argument("--checkpoint",   type=str,   default="outputs/vae_best.pt")
    p.add_argument("--output_dir",   type=str,   default="outputs")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--base_ch",      type=int,   default=32)
    p.add_argument("--latent_ch",    type=int,   default=64)
    p.add_argument("--cond_dim",     type=int,   default=32)
    p.add_argument("--dropout_p",    type=float, default=0.1,
                   help="MC-Dropout p (actif aussi à l inférence)")
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
    add_data_args(p)
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

        print("  Chargement du champ pour les figures...")
        fields, channels, sea_mask, data_info = load_ocean(args)
        T, S = fields[:, 0], fields[:, 1]        # thetao_z0, so_z0
        args.sea_mask = sea_mask

        if args.figures:
            print("\n  Figure 1 : Evaluation du reseau existant (zones lacunaires + LOO)...")
            plot_network_evaluation(model, T, S, norm, args, n_samples=args.n_mc)
            print("\n  Figure 2 : Incertitude vs densite reseau...")
            plot_uncertainty_maps(model, T, S, norm, args, n_samples=args.n_mc)

        if args.score:
            score(args)

    if args.report:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output_dir)
        # Recharger métriques depuis checkpoint si disponible
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            saved_args = ckpt.get("args", {})
        except Exception:
            saved_args = {}
        lines = [
            "=" * 68,
            "  Brique 1 — AE-UNet MC-Dropout — Rapport",
            f"  Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 68, "",
            "── DONNÉES ──────────────────────────────────────────────────────────",
            f"  source      : {saved_args.get('data_source') or DATA_SOURCE}",
            f"  domaine     : {NX}×{NY}  |  canaux : {N_CHANNELS}",
            f"  observés    : {OBSERVED_VARS}",
            "",
            "── REPRODUCTIBILITÉ ─────────────────────────────────────────────────",
            f"  seed_ocean  : {args.seed_ocean}",
            f"  seed_buoys  : {args.seed_buoys}",
            "",
            "── HYPERPARAMÈTRES ──────────────────────────────────────────────────",
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
        rpt = out / f"rapport_ae_{ts}.txt"
        rpt.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  Rapport AE → {rpt}")
