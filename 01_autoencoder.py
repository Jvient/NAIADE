"""
==========================================================================
  BRIQUE 1 — AE-UNet v4 d'Observabilité (MC-Dropout + ObsGate + FiLM)
==========================================================================

Architecture : AE déterministe + MC-Dropout pour l'incertitude épistémique.
  - MC-Dropout (Gal & Ghahramani 2016) : dropout actif aussi à l'inférence
    → N passes forward → variance des prédictions = incertitude
  - ObsGate sur chaque skip-connexion : gate conditionné sur la densité
    locale d'observations
  - GroupNorm (compatible batch_size=1 à l'inférence MC)
  - FiLM conditioning (N_obs) + deep supervision
  - Huber loss (δ=0.5) robuste aux fronts mal positionnés

Usage :
  python 01_autoencoder.py --train
  python 01_autoencoder.py --score   --checkpoint outputs/ae_best.pt
  python 01_autoencoder.py --figures --checkpoint outputs/ae_best.pt
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
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from config import *
try:
    from dataset import SyntheticOceanGenerator, OceanOEDDataset, build_datasets
except ModuleNotFoundError:
    from data.dataset import SyntheticOceanGenerator, OceanOEDDataset, build_datasets


# =============================================================================
#  BLOCS DE BASE
# =============================================================================

class MCDropout2d(nn.Module):
    """Dropout spatial TOUJOURS actif (training et inférence) pour MC-Dropout."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout2d(x, p=self.p, training=True)


class ResDoubleConv(nn.Module):
    """Double conv résiduelle + MC-Dropout spatial."""
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            MCDropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return self.net(x) + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2),
                                 ResDoubleConv(in_ch, out_ch, dropout_p))
    def forward(self, x):
        return self.net(x)


class ObsGate(nn.Module):
    """
    Skip-gating conditionné sur la densité d'observations locale.
    Gate σ(conv(mask_ds)) ∈ [0,1] module les features du skip :
      Gate≈1 en zone observée → skip passe fort
      Gate≈0 en zone lacunaire → décodeur interpole depuis le bottleneck
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
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.gate = ObsGate(skip_ch)
        self.conv = ResDoubleConv(in_ch // 2 + skip_ch, out_ch, dropout_p)
        self.film = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x, skip, cond, mask_ds):
        x  = self.up(x)
        dy = skip.shape[2] - x.shape[2]
        dx = skip.shape[3] - x.shape[3]
        x  = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        skip = self.gate(skip, mask_ds)
        h = self.conv(torch.cat([skip, x], dim=1))
        gam, bet = self.film(cond).chunk(2, dim=-1)
        return h * (1 + gam.view(-1, h.shape[1], 1, 1)) + bet.view(-1, h.shape[1], 1, 1)


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

class ObservabilityAE(nn.Module):
    """
    AE-UNet déterministe avec incertitude par MC-Dropout.

    Entrée : (B, 3, NX, NY) — [T_obs, S_obs, mask]
    Sortie : (B, 2, NX, NY) — [T_pred, S_pred] + sorties auxiliaires deep supervision
    """
    def __init__(self, in_ch=3, out_ch=2, base_ch=32, latent_ch=64,
                 dropout_p=0.1, cond_dim=32):
        super().__init__()
        bc, dp = base_ch, dropout_p
        self.latent_ch = latent_ch
        self.cond_dim  = cond_dim
        self.dropout_p = dropout_p

        # Encodeur 4 niveaux
        self.inc   = ResDoubleConv(in_ch, bc, dp)
        self.down1 = Down(bc,    bc*2,  dp)
        self.down2 = Down(bc*2,  bc*4,  dp)
        self.down3 = Down(bc*4,  bc*8,  dp)
        self.down4 = Down(bc*8,  bc*16, dp)

        # FiLM embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(1, cond_dim), nn.GELU(),
            nn.Linear(cond_dim, cond_dim), nn.GELU(),
        )

        # Bottleneck déterministe + CBAM
        self.cbam   = CBAM(bc*16)
        self.to_z   = nn.Conv2d(bc*16, latent_ch, 1)
        self.from_z = nn.Conv2d(latent_ch, bc*16, 1)

        # Décodeur FiLM 4 niveaux + ObsGate
        self.up1 = FiLMUp(bc*16, bc*8,  bc*8,  cond_dim, dp)
        self.up2 = FiLMUp(bc*8,  bc*4,  bc*4,  cond_dim, dp)
        self.up3 = FiLMUp(bc*4,  bc*2,  bc*2,  cond_dim, dp)
        self.up4 = FiLMUp(bc*2,  bc,    bc,    cond_dim, dp)
        self.head = nn.Conv2d(bc, out_ch, 1)

        # Deep supervision
        self.aux1 = nn.Conv2d(bc*8, out_ch, 1)
        self.aux2 = nn.Conv2d(bc*4, out_ch, 1)
        self.aux3 = nn.Conv2d(bc*2, out_ch, 1)

    def _get_cond(self, x):
        mask = x[:, 2:3]
        return self.cond_embed(mask.mean(dim=[2, 3]))

    def _downsample_mask(self, mask, target):
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
        s1, s2, s3, s4 = skips
        h = self.from_z(z)
        h    = self.up1(h, s4, cond, self._downsample_mask(mask, s4))
        aux1 = self.aux1(h)
        h    = self.up2(h, s3, cond, self._downsample_mask(mask, s3))
        aux2 = self.aux2(h)
        h    = self.up3(h, s2, cond, self._downsample_mask(mask, s2))
        aux3 = self.aux3(h)
        h    = self.up4(h, s1, cond, self._downsample_mask(mask, s1))
        return self.head(h), [aux1, aux2, aux3]

    def forward(self, x):
        mask     = x[:, 2:3]
        cond     = self._get_cond(x)
        z, skips = self.encode(x)
        pred, aux = self.decode(z, skips, cond, mask)
        return pred, z, aux

    @torch.no_grad()
    def reconstruct_with_uncertainty(self, x, n_samples=50):
        """
        MC-Dropout : N passes avec dropout actif → variance = incertitude.
        Retourne (mean, std, z).
        """
        mask = x[:, 2:3]
        cond = self._get_cond(x)
        z, skips = self.encode(x)
        samples = [self.decode(z, skips, cond, mask)[0] for _ in range(n_samples)]
        stack = torch.stack(samples)
        return stack.mean(0), stack.std(0), z

    def get_latent(self, x):
        z, _ = self.encode(x)
        return z.flatten(1)


# =============================================================================
#  LOSS — Huber + gradient + deep supervision
# =============================================================================

class AELoss(nn.Module):
    """
    L = L_recon + λ_grad·L_grad + Σ w_k·L_aux_k
    Huber loss (δ=0.5) : quadratique pour |e|<δ, linéaire au-delà.
    """
    def __init__(self, w_obs=1.0, w_unobs=4.0, lambda_grad=0.5, huber_delta=0.5):
        super().__init__()
        self.w_obs       = w_obs
        self.w_unobs     = w_unobs
        self.lambda_grad = lambda_grad
        self.huber_delta = huber_delta
        self.aux_weights = [0.4, 0.3, 0.2]

    def _huber(self, diff):
        d = self.huber_delta
        abs_diff = diff.abs()
        return torch.where(abs_diff < d, 0.5 * diff**2, d * (abs_diff - 0.5 * d))

    def _recon_loss(self, pred, target, mask):
        err = self._huber(pred - target)
        return (self.w_obs * (err * mask).mean()
                + self.w_unobs * (err * (1 - mask)).mean())

    @staticmethod
    def _spatial_grad(f):
        return f[..., 1:, :] - f[..., :-1, :], f[..., :, 1:] - f[..., :, :-1]

    def forward(self, pred, target, mask, aux_preds=None):
        loss_recon = self._recon_loss(pred, target, mask)

        pgx, pgy = self._spatial_grad(pred)
        tgx, tgy = self._spatial_grad(target)
        loss_grad = self._huber(pgx - tgx).mean() + self._huber(pgy - tgy).mean()

        loss_aux = torch.tensor(0.0, device=pred.device)
        if aux_preds is not None:
            H, W = target.shape[2], target.shape[3]
            for aux, w in zip(aux_preds, self.aux_weights):
                aux_up  = F.interpolate(aux, size=(H, W), mode="bilinear", align_corners=False)
                mask_ds = F.interpolate(mask, size=(H, W), mode="nearest")
                loss_aux = loss_aux + w * self._recon_loss(aux_up, target, mask_ds)

        total = loss_recon + self.lambda_grad * loss_grad + loss_aux
        return total, loss_recon, loss_aux


# =============================================================================
#  ENTRAÎNEMENT
# =============================================================================

def train(args):
    print("=" * 62)
    print("  Brique 1 — Entraînement AE-UNet v4 (MC-Dropout + ObsGate)")
    print("=" * 62)

    set_global_seed(args.seed_ocean)

    print("\n[1/4] Génération du nature run...")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=NT)
    print(f"  T: {T.shape}  [{T.min():.1f}, {T.max():.1f}] °C")

    train_ds, val_ds = build_datasets(T, S, split=0.8,
                                      n_obs_min=args.n_obs_min,
                                      n_obs_max=args.n_obs_max,
                                      augment_train=True)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\n[2/4] Modèle AE-UNet v4 (base_ch={args.base_ch}, latent_ch={args.latent_ch}, "
          f"dropout={args.dropout_p})...")
    model = ObservabilityAE(base_ch=args.base_ch, latent_ch=args.latent_ch,
                            dropout_p=args.dropout_p, cond_dim=args.cond_dim).to(DEVICE)
    print(f"  Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(ep):
        warmup = max(1, args.epochs // 10)
        if ep < warmup:
            return float(ep + 1) / warmup
        return 0.5 * (1.0 + np.cos(np.pi * (ep - warmup) / max(1, args.epochs - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = AELoss(w_obs=1.0, w_unobs=args.w_unobs,
                       lambda_grad=args.lambda_grad, huber_delta=args.huber_delta)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[3/4] Entraînement {args.epochs} époques...")
    history = {"train_loss": [], "val_rmse_unobs": [], "loss_aux": [], "lr": []}
    best_val = np.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = ep_aux = 0.0
        for x, y, mask in train_ld:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            pred, z, aux_preds = model(x)
            loss, _, l_aux = criterion(pred, y, mask, aux_preds)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            ep_aux  += l_aux.item()
        scheduler.step()
        n = len(train_ld)
        ep_loss /= n; ep_aux /= n

        # Validation MC-moyennée
        model.eval()
        val_rmses = []
        with torch.no_grad():
            for x, y, mask in val_ld:
                x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                preds = torch.stack([model(x)[0] for _ in range(args.n_mc_val)])
                pred_mean = preds.mean(0)
                sq = (pred_mean - y) ** 2
                for b in range(x.shape[0]):
                    val_rmses.append(float(torch.sqrt((sq[b] * (1 - mask[b])).mean()).item()))
        val_rmse = float(np.mean(val_rmses))

        history["train_loss"].append(ep_loss)
        history["val_rmse_unobs"].append(val_rmse)
        history["loss_aux"].append(ep_aux)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}/{args.epochs} | Loss={ep_loss:.4f} | "
                  f"RMSE_unobs={val_rmse:.4f} | Laux={ep_aux:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save({
                "model_state": model.state_dict(),
                "args":  vars(args),
                "norm":  {"T_mean": train_ds.T_mean, "T_std": train_ds.T_std,
                          "S_mean": train_ds.S_mean, "S_std": train_ds.S_std}
            }, out_dir / "ae_best.pt")

    print(f"\n  Meilleur RMSE val (non-obs) : {best_val:.4f}")

    # Courbes d'apprentissage
    print("\n[4/4] Sauvegarde des courbes...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), facecolor="#0a1628")
    for ax, (lbl, k, col) in zip(axes, [
        ("Loss totale",        "train_loss",     "#6baed6"),
        ("RMSE val (non-obs)", "val_rmse_unobs", "#fc8d59"),
        ("Deep supervision",   "loss_aux",       "#cc99ff"),
    ]):
        ax.plot(history[k], color=col, lw=1.8)
        ax.set_title(lbl, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("Époque", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
        ax.grid(True, alpha=0.2, color="white")
    fig.tight_layout(pad=2)
    fig.savefig(out_dir / "ae_training_curves.png", dpi=130,
                facecolor="#0a1628", bbox_inches="tight")
    plt.close()
    print(f"  Courbes → {out_dir}/ae_training_curves.png")
    print(f"  Checkpoint → {out_dir}/ae_best.pt")


# =============================================================================
#  HELPERS — RMSE MC
# =============================================================================

@torch.no_grad()
def _compute_rmse_mc(model, T_n_t, S_n_t, positions, norm, n_mc=8):
    """RMSE (pixels non observés) sur un seul instant, moyenne sur n_mc tirages."""
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
    y_true = np.stack([T_n_t, S_n_t])
    sq = (pred - y_true) ** 2
    return float(np.sqrt((sq * (1 - mask[None])).mean()))


# =============================================================================
#  FIGURE 1 — Évaluation d'un réseau existant
# =============================================================================

@torch.no_grad()
def plot_network_evaluation(model, T, S, norm, args,
                            positions=None, n_samples=80, n_loo_t=8):
    """
    Figure d'évaluation du réseau : reconstruction, incertitude MC,
    zones lacunaires, LOO, et proposition de 3 bouées.
    """
    from scipy.ndimage import distance_transform_edt

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

    # Réseau de référence
    if positions is None:
        rng = np.random.default_rng(getattr(args, "seed_buoys", 42))
        positions = [(int(rng.integers(0, NX)), int(rng.integers(0, NY)))
                     for _ in range(N_BUOYS)]
    positions = list(positions)
    n_sensors = len(positions)

    mask_np = np.zeros((NX, NY), dtype=np.float32)
    for (x, y) in positions:
        mask_np[x, y] = 1.0
    obs_pos = np.array(positions)

    # Reconstruction + incertitude MC
    t_ref = len(T) // 3
    T_t, S_t = T_n[t_ref], S_n[t_ref]
    x_in = torch.from_numpy(np.stack([T_t * mask_np, S_t * mask_np, mask_np])[None]).to(DEVICE)
    recon_mean, recon_std, _ = model.reconstruct_with_uncertainty(x_in, n_samples=n_samples)
    rm = recon_mean[0].cpu().numpy()
    rs = recon_std[0].cpu().numpy()

    T_true  = T_t * norm["T_std"] + norm["T_mean"]
    S_true  = S_t * norm["S_std"] + norm["S_mean"]
    T_pred  = rm[0] * norm["T_std"] + norm["T_mean"]
    S_pred  = rm[1] * norm["S_std"] + norm["S_mean"]
    T_sigma = rs[0] * norm["T_std"]
    S_sigma = rs[1] * norm["S_std"]

    # Carte des zones lacunaires
    dist_to_sensor = distance_transform_edt(1 - mask_np)
    dist_n = dist_to_sensor / (dist_to_sensor.max() + 1e-9)
    combined_sigma = 0.5 * (T_sigma / (T_sigma.max() + 1e-9) + S_sigma / (S_sigma.max() + 1e-9))
    gap_map = combined_sigma * dist_n
    gap_threshold = np.percentile(gap_map, 80)
    gap_binary = (gap_map > gap_threshold).astype(float)

    # 3 bouées proposées — critère D-optimal (réduction de variance latente)
    # Pour chaque candidat, on simule l'ajout d'un capteur et on mesure
    # la réduction de variance prédictive MC-Dropout. Le candidat qui
    # maximise det(Σ_avant) / det(Σ_après) est D-optimal.
    # Approximation efficace : on utilise trace(Σ) (critère A-optimal)
    # sur un sous-ensemble de candidats issus des zones à forte incertitude.
    n_propose = 3
    n_candidates = 25  # positions évaluées par itération
    n_mc_fast = 6      # passes MC rapides pour le scoring

    proposed_positions = []
    mask_aug = mask_np.copy()
    for k_prop in range(n_propose):
        # Candidats : top-variance parmi les pixels non observés, sous-échantillonnés
        var_map = combined_sigma * (1 - mask_aug)  # 0 aux capteurs existants
        flat_idx = np.argsort(var_map.ravel())[::-1]
        # Sous-échantillonner en gardant un espacement minimal (~8 px)
        candidates = []
        for fi in flat_idx:
            if len(candidates) >= n_candidates:
                break
            cx, cy = np.unravel_index(fi, var_map.shape)
            if var_map[cx, cy] < 1e-6:
                continue
            # Espacement minimum avec les candidats déjà retenus
            if all(abs(cx - pc[0]) + abs(cy - pc[1]) > 8 for pc in candidates):
                candidates.append((int(cx), int(cy)))

        if not candidates:
            # Fallback : argmax simple
            px, py = np.unravel_index(np.argmax(var_map), var_map.shape)
            proposed_positions.append((int(px), int(py)))
            mask_aug[px, py] = 1.0
            continue

        # Évaluer chaque candidat : variance totale après ajout
        best_pos, best_var = None, np.inf
        for (cx, cy) in candidates:
            mask_test = mask_aug.copy()
            mask_test[cx, cy] = 1.0
            x_test = torch.from_numpy(
                np.stack([T_t * mask_test, S_t * mask_test, mask_test])[None]).to(DEVICE)
            _, rs_test, _ = model.reconstruct_with_uncertainty(x_test, n_samples=n_mc_fast)
            # Critère A-optimal : trace de la variance (somme des variances pixel)
            total_var = float(rs_test[0].sum().cpu())
            if total_var < best_var:
                best_var = total_var
                best_pos = (cx, cy)

        proposed_positions.append(best_pos)
        mask_aug[best_pos[0], best_pos[1]] = 1.0
        print(f"    Bouée P{k_prop+1} @ {best_pos} (Δvar={float(combined_sigma.sum()) - best_var:.2f})")

    proposed_arr = np.array(proposed_positions)

    # LOO scores
    t_idx = np.random.choice(len(T), min(n_loo_t, len(T)), replace=False)
    rmse_full = np.mean([_compute_rmse_mc(model, T_n[t], S_n[t], positions, norm, n_mc=6) for t in t_idx])
    loo_delta = np.zeros(n_sensors)
    for i in range(n_sensors):
        sub = [p for j, p in enumerate(positions) if j != i]
        loo_delta[i] = np.mean([_compute_rmse_mc(model, T_n[t], S_n[t], sub, norm, n_mc=6) for t in t_idx]) - rmse_full
    loo_colors = np.clip(loo_delta / (loo_delta.max() + 1e-9), 0, 1)

    # Figure 2×4
    fig = plt.figure(figsize=(22, 11), facecolor=BG)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.28,
                           left=0.04, right=0.97, top=0.91, bottom=0.05)

    def cell(ax, data, cmap, vmin, vmax, title, label, pts=None, pts_c=None,
             pts_s=40, pts_cmap="RdYlGn", contour=None):
        im = ax.imshow(data.T, cmap=cmap, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        if pts is not None:
            ax.scatter(pts[:,0], pts[:,1], c=pts_c if pts_c is not None else "white",
                       s=pts_s, cmap=pts_cmap, vmin=0, vmax=1,
                       edgecolors="black", linewidths=0.5, zorder=6)
        if contour is not None:
            ax.contour(contour.T, levels=[0.5], colors=["#ff6b6b"], linewidths=1.5, linestyles="--")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.set_title(title, color="white", fontsize=8.5, pad=5, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(label, color="white", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)

    vT = (T_true.min(), T_true.max())
    vS = (S_true.min(), S_true.max())

    # Ligne 1 : SST
    cell(fig.add_subplot(gs[0, 0]), T_true, ocean_cmap, *vT,
         f"SST vrai + réseau ({n_sensors} capteurs)", "°C",
         pts=obs_pos, pts_c=loo_colors, pts_s=55)
    cell(fig.add_subplot(gs[0, 1]), T_pred, ocean_cmap, *vT,
         f"SST reconstruction (RMSE={rmse_full:.3f})", "°C",
         pts=obs_pos, pts_c="white", pts_s=12)
    cell(fig.add_subplot(gs[0, 2]), T_sigma, "YlOrRd", 0, T_sigma.max(),
         f"SST incertitude σ MC (N={n_samples})", "°C",
         pts=obs_pos, pts_c="cyan", pts_s=12, contour=gap_binary)

    ax03 = fig.add_subplot(gs[0, 3])
    cell(ax03, gap_map, "inferno", 0, gap_map.max(),
         f"Zones lacunaires + {n_propose} bouées D-optimal", "score",
         pts=obs_pos, pts_c="cyan", pts_s=15, contour=gap_binary)
    for k, (px, py) in enumerate(proposed_arr):
        ax03.scatter(px, py, marker="*", s=320, c="#ffd93d", edgecolors="black", linewidths=0.8, zorder=8)
        ax03.annotate(f"P{k+1}", (px, py), textcoords="offset points", xytext=(6, 4),
                      fontsize=8, color="#ffd93d", fontweight="bold")

    # Ligne 2 : SSS + LOO barplot
    cell(fig.add_subplot(gs[1, 0]), S_true, sal_cmap, *vS,
         "SSS vrai + capteurs", "psu", pts=obs_pos, pts_c="white", pts_s=12)
    cell(fig.add_subplot(gs[1, 1]), S_pred, sal_cmap, *vS,
         "SSS reconstruction", "psu", pts=obs_pos, pts_c="white", pts_s=12)
    cell(fig.add_subplot(gs[1, 2]), S_sigma, "YlOrRd", 0, S_sigma.max(),
         "SSS incertitude σ MC", "psu", pts=obs_pos, pts_c="cyan", pts_s=12, contour=gap_binary)

    # LOO barplot
    ax13 = fig.add_subplot(gs[1, 3])
    ax13.set_facecolor("#050d1a")
    for sp in ax13.spines.values(): sp.set_edgecolor("#1a3a5c")
    idx_sort = np.argsort(loo_delta)[::-1]
    colors_bar = plt.cm.RdYlGn(np.clip((loo_delta[idx_sort] - loo_delta.min()) /
                                        (loo_delta.max() - loo_delta.min() + 1e-9), 0, 1))
    ax13.barh(np.arange(n_sensors), loo_delta[idx_sort], color=colors_bar, edgecolor="#1a3a5c", linewidth=0.5)
    ax13.axvline(0, color="white", lw=0.8, alpha=0.5)
    thr = loo_delta.max() * 0.05
    ax13.axvline(thr, color="#ffd93d", lw=1, linestyle="--", alpha=0.7, label=f"seuil 5% ({(loo_delta < thr).sum()} redondants)")
    ax13.set_yticks(np.arange(0, n_sensors, max(1, n_sensors//10)))
    ax13.set_yticklabels([f"C{idx_sort[i]}" for i in range(0, n_sensors, max(1, n_sensors//10))],
                         color="white", fontsize=6)
    ax13.set_xlabel("Δ RMSE (LOO − complet)", color="white", fontsize=8)
    ax13.set_title("Contribution LOO par capteur", color="white", fontsize=8.5, fontweight="bold", pad=5)
    ax13.tick_params(colors="white", labelsize=6)
    ax13.grid(True, alpha=0.2, color="white", axis="x")
    ax13.legend(fontsize=6, labelcolor="white", facecolor="#0a1628", loc="lower right")

    fig.text(0.5, 0.97,
             f"AE-UNet — Évaluation Réseau ({n_sensors} capteurs) | ★ = bouées D-optimal",
             ha="center", color="white", fontsize=12, fontweight="bold")

    out = out_dir / "ae_network_evaluation.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Figure évaluation réseau → {out}")
    print(f"  Bouées proposées : " + "  ".join([f"P{k+1}=({px},{py})" for k,(px,py) in enumerate(proposed_arr)]))
    return loo_delta, gap_map, positions, proposed_arr


# =============================================================================
#  FIGURE 2 — Incertitude vs densité réseau
# =============================================================================

@torch.no_grad()
def plot_uncertainty_maps(model, T, S, norm, args, n_samples=60):
    """Incertitude comparée : Dense (N=40), Moyen (N=20), Clairsemé (N=8)."""
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    BG = "#0a1628"

    model.eval()
    T_n = (T - norm["T_mean"]) / norm["T_std"]
    S_n = (S - norm["S_mean"]) / norm["S_std"]
    t = len(T) // 2

    configs = [("Dense (N=40)", 40), ("Moyen (N=20)", 20), ("Clairsemé (N=8)", 8)]

    unc_max = 0.0
    results = []
    for (_, n_obs) in configs:
        np.random.seed(n_obs * 7)
        flat = np.zeros(NX * NY, dtype=np.float32)
        flat[np.random.choice(NX*NY, n_obs, replace=False)] = 1.0
        mask = flat.reshape(NX, NY)
        x_in = torch.from_numpy(np.stack([T_n[t]*mask, S_n[t]*mask, mask])[None]).to(DEVICE)
        rm, rs, _ = model.reconstruct_with_uncertainty(x_in, n_samples=n_samples)
        T_s = rs[0, 0].cpu().numpy() * norm["T_std"]
        S_s = rs[0, 1].cpu().numpy() * norm["S_std"]
        unc_max = max(unc_max, T_s.max(), S_s.max())
        results.append((mask, T_s, S_s, np.argwhere(mask > 0.5)))

    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.28,
                           left=0.05, right=0.97, top=0.91, bottom=0.06)

    for col, ((desc, _), (_, T_s, S_s, obs_pos)) in enumerate(zip(configs, results)):
        for row, (data, lbl) in enumerate([(T_s, "SST sigma (°C)"), (S_s, "SSS sigma (psu)")]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(data.T, cmap="YlOrRd", origin="lower", aspect="auto", vmin=0, vmax=unc_max)
            ax.scatter(obs_pos[:,0], obs_pos[:,1], c="cyan", s=12, edgecolors="black", linewidths=0.3, zorder=5, alpha=0.8)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#050d1a")
            for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
            ax.set_title(f"{desc}\n{lbl}" if row == 0 else lbl, color="white", fontsize=9, fontweight="bold", pad=4)
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)

        # Profil méridional
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.fill_betweenx(np.arange(NY), 0, T_s.mean(axis=0), color="#fc8d59", alpha=0.7, label="SST σ")
        ax.fill_betweenx(np.arange(NY), 0, S_s.mean(axis=0), color="#6baed6", alpha=0.5, label="SSS σ")
        for (_, yp) in obs_pos:
            ax.axhline(yp, color="cyan", lw=0.4, alpha=0.4)
        ax.set_title("Profil méridional σ", color="white", fontsize=8, fontweight="bold", pad=4)
        ax.tick_params(colors="white", labelsize=6)
        ax.legend(fontsize=7, labelcolor="white", facecolor="#0a1628")
        ax.grid(True, alpha=0.2, color="white", axis="x")

    fig.text(0.5, 0.97, "AE-UNet — Incertitude vs Densité Réseau",
             ha="center", color="white", fontsize=12, fontweight="bold")
    out = out_dir / "ae_uncertainty_density.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Figure densité/incertitude → {out}")


# =============================================================================
#  SCORING — Leave-One-Out
# =============================================================================

def score(args):
    print("=" * 62)
    print("  Brique 1 — Scoring AE")
    print("=" * 62)

    set_global_seed(args.seed_ocean)

    ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = ObservabilityAE(
        base_ch=ckpt["args"]["base_ch"],
        latent_ch=ckpt["args"]["latent_ch"],
        dropout_p=ckpt["args"].get("dropout_p", 0.1),
        cond_dim=ckpt["args"].get("cond_dim", 32)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Modèle chargé : {args.checkpoint}")

    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=200)
    norm = ckpt["norm"]

    rng = np.random.default_rng(args.seed_buoys)
    positions = [(int(rng.integers(0, NX)), int(rng.integers(0, NY))) for _ in range(N_BUOYS)]

    print("\n  Leave-One-Out...")
    t_idx = np.random.choice(len(T), 10, replace=False)
    rmse_full = np.mean([_compute_rmse_mc(model, (T[t]-norm["T_mean"])/norm["T_std"],
                                          (S[t]-norm["S_mean"])/norm["S_std"], positions, norm) for t in t_idx])
    print(f"  RMSE réseau complet : {rmse_full:.4f}")

    loo_scores = {}
    for i, pos in enumerate(positions):
        sub = [p for j, p in enumerate(positions) if j != i]
        rmse_loo = np.mean([_compute_rmse_mc(model, (T[t]-norm["T_mean"])/norm["T_std"],
                                              (S[t]-norm["S_mean"])/norm["S_std"], sub, norm) for t in t_idx])
        loo_scores[i] = {"position": list(pos), "delta_rmse": float(rmse_loo - rmse_full)}
        print(f"  Capteur {i:2d} @ {pos} | delta={loo_scores[i]['delta_rmse']:+.4f}")

    out_dir = Path(args.output_dir)
    with open(out_dir / "ae_loo_scores.json", "w") as f:
        json.dump(loo_scores, f, indent=2)
    print(f"  LOO scores → {out_dir}/ae_loo_scores.json")


# =============================================================================
#  POINT D'ENTRÉE
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="AE-UNet v4 MC-Dropout OED")
    p.add_argument("--train",        action="store_true")
    p.add_argument("--score",        action="store_true")
    p.add_argument("--figures",      action="store_true")
    p.add_argument("--seed_ocean",   type=int,   default=42)
    p.add_argument("--seed_buoys",   type=int,   default=7)
    p.add_argument("--checkpoint",   type=str,   default="outputs/ae_best.pt")
    p.add_argument("--output_dir",   type=str,   default="outputs")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--base_ch",      type=int,   default=32)
    p.add_argument("--latent_ch",    type=int,   default=64)
    p.add_argument("--cond_dim",     type=int,   default=32)
    p.add_argument("--dropout_p",    type=float, default=0.1)
    p.add_argument("--w_unobs",      type=float, default=4.0)
    p.add_argument("--lambda_grad",  type=float, default=0.5)
    p.add_argument("--huber_delta",  type=float, default=0.5)
    p.add_argument("--n_obs_min",    type=int,   default=10)
    p.add_argument("--n_obs_max",    type=int,   default=80)
    p.add_argument("--n_mc_val",     type=int,   default=15)
    p.add_argument("--n_mc",         type=int,   default=60)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not any([args.train, args.score, args.figures]):
        print("Usage: python 01_autoencoder.py --train [--figures] [--score]")
        sys.exit(0)

    if args.train:
        train(args)

    if args.score or args.figures:
        ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        model = ObservabilityAE(
            base_ch=ckpt["args"]["base_ch"],
            latent_ch=ckpt["args"]["latent_ch"],
            dropout_p=ckpt["args"].get("dropout_p", 0.1),
            cond_dim=ckpt["args"].get("cond_dim", 32)).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        norm = ckpt["norm"]

        set_global_seed(args.seed_ocean)
        gen = SyntheticOceanGenerator()
        T, S = gen.generate_dataset(nt=200, seed=args.seed_ocean)

        if args.figures:
            print("\n  Figure 1 : Évaluation réseau (zones lacunaires + LOO)...")
            plot_network_evaluation(model, T, S, norm, args, n_samples=args.n_mc)
            print("\n  Figure 2 : Incertitude vs densité réseau...")
            plot_uncertainty_maps(model, T, S, norm, args, n_samples=args.n_mc)

        if args.score:
            score(args)
