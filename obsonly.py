"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BRIQUE 1b — RECONSTRUCTION SUPERVISÉE PAR LES OBSERVATIONS SEULES           ║
║                                                                              ║
║  Remplace la supervision par la vérité (AELoss, w_unobs=4.0 sur les pixels   ║
║  non observés) par une supervision auto-supervisée aux CAPTEURS MASQUÉS.     ║
║                                                                              ║
║    ancien :  L = w_obs·Huber(pred, truth)|obs + w_unobs·Huber(pred,truth)|¬obs║
║    nouveau : L = NLL_gaussienne(mu, sigma², obs)|capteurs tenus à l'écart    ║
║                 + lambda_tv · TV(mu)          (a priori de régularité,       ║
║                                                sans vérité)                  ║
║                                                                              ║
║  Le nature run n'apparaît nulle part dans cette boucle. Il ne sert qu'à      ║
║  valider l'estimateur a posteriori (validate_obsonly.py).                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Deux changements structurels par rapport à 01_autoencoder.py
-------------------------------------------------------------
1. Tête hétéroscédastique (mu, log sigma²) au lieu d'une sortie déterministe.
   L'utilité obs-only est une VARIANCE PRÉDICTIVE : elle doit être calibrée,
   et la calibration se vérifie sur observations tenues à l'écart (PIT).
   MC-Dropout donne une dispersion, pas une variance calibrée.

2. Masquage PAR CAPTEUR (et par groupe pour les trajectoires), jamais par
   pixel : sinon les points voisins d'un même glider fuient dans l'entrée.

Usage
-----
    python obsonly.py --train --obs outputs/obs_synth.npz --epochs 60
    python obsonly.py --lobo  --obs outputs/obs_synth.npz --ckpt outputs/ae_obsonly.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from naiade_compat import (get_device, resolve_seed_fn, find_ae_class,
                           check_ae_signature)
from obs_operator import ObsSet, split_sensors, VARIABLES

DEVICE = get_device()
set_global_seed = resolve_seed_fn(verbose=True)


# ══════════════════════════════════════════════════════════════════════════════
#  IMPORT DU MODULE 01 (nom commençant par un chiffre)
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_AE_ATTRS = ("cond_embed", "encode", "decode")


def _load_ae_module(path="01_autoencoder.py", ae_class=None):
    """Charge 01_autoencoder.py et résout la classe de l'autoencodeur.

    Le nom de la classe a divergé entre branches : on le résout à l'exécution
    (naiade_compat.find_ae_class) au lieu de l'imposer. Retourne (module, classe).
    """
    p = Path(__file__).parent / path
    if not p.exists():
        raise FileNotFoundError(
            f"{p} introuvable. Ce module se greffe sur l'AE existant ; "
            "vérifiez que vous êtes bien à la racine du dépôt.")
    spec = importlib.util.spec_from_file_location("ae_mod", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ae_mod"] = mod
    spec.loader.exec_module(mod)

    cls = find_ae_class(mod, override=ae_class)
    ok, missing, defaults = check_ae_signature(cls)
    if not ok:
        raise TypeError(
            f"{cls.__name__}.__init__ n'accepte pas {missing}.\n"
            "  La greffe obs-only a besoin de in_ch/out_ch paramétrables : "
            "4 canaux d'entrée (T, S, mask_T, mask_S) et 4 de sortie "
            "(mu_T, mu_S, logvar_T, logvar_S).\n"
            f"  Signature actuelle : {defaults}")

    # cond_embed est un attribut d'INSTANCE : instancier pour le voir.
    try:
        probe = cls(in_ch=4, out_ch=4, base_ch=4, latent_ch=4)
    except Exception as e:
        raise TypeError(
            f"{cls.__name__}(in_ch=4, out_ch=4) échoue : {e}") from e
    for a in REQUIRED_AE_ATTRS:
        if not hasattr(probe, a):
            raise AttributeError(
                f"{cls.__name__} n'a pas .{a} — architecture incompatible. "
                "obsonly.py remplace _get_cond et suppose un FiLM conditionné "
                "par cond_embed.")
    del probe
    mod._AE_CLASS = cls
    return mod


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLE — tête hétéroscédastique, masques par variable
# ══════════════════════════════════════════════════════════════════════════════

class ObservabilityAEHetero(nn.Module):
    """AE-UNet de la Brique 1, avec :
      · entrée 4 canaux  [T_obs, S_obs, mask_T, mask_S]
        (masques SÉPARÉS : un mouillage Pacifique mesure T sans S)
      · sortie 4 canaux  [mu_T, mu_S, logvar_T, logvar_S]

    Réutilise ObservabilityAE tel quel — on ne change que in_ch/out_ch et le
    canal lu comme masque de conditionnement.
    """

    def __init__(self, ae_module, base_ch=32, latent_ch=64, dropout_p=0.1,
                 cond_dim=32, logvar_min=-7.0, logvar_max=3.0):
        super().__init__()
        AE = getattr(ae_module, '_AE_CLASS',
                     getattr(ae_module, 'ObservabilityAE', None))
        self.net = AE(
            in_ch=4, out_ch=4, base_ch=base_ch, latent_ch=latent_ch,
            dropout_p=dropout_p, cond_dim=cond_dim)
        # conditionnement FiLM : moyenne des deux masques au lieu du canal 2
        self.net._get_cond = lambda x: self.net.cond_embed(
            x[:, 2:4].mean(dim=[1, 2, 3], keepdim=False).unsqueeze(-1))
        self.logvar_min, self.logvar_max = logvar_min, logvar_max
        # facteur de recalibration, ajusté après entraînement sur la validation
        self.register_buffer("calib_scale", torch.ones(1))

    def _split(self, out):
        mu = out[:, :2]
        logvar = torch.clamp(out[:, 2:], self.logvar_min, self.logvar_max)
        return mu, logvar

    def forward(self, x):
        gate = x[:, 2:4].max(dim=1, keepdim=True).values      # (B,1,H,W)
        cond = self.net._get_cond(x)
        z, skips = self.net.encode(x)
        out, aux = self.net.decode(z, skips, cond, gate)
        mu, logvar = self._split(out)
        return mu, logvar, z, [self._split(a) for a in aux]

    @torch.no_grad()
    def predict(self, x, n_mc=8):
        """Prédiction (mu, sigma). n_mc>1 combine l'incertitude aléatoire
        (tête hétéroscédastique) et épistémique (MC-Dropout) :
            sigma² = E[sigma²_alea] + Var[mu]
        """
        mus, vs = [], []
        for _ in range(max(1, n_mc)):
            mu, logvar, _, _ = self.forward(x)
            mus.append(mu); vs.append(logvar.exp())
        mu = torch.stack(mus).mean(0)
        var = torch.stack(vs).mean(0)
        if n_mc > 1:
            var = var + torch.stack(mus).var(0)
        return mu, var.sqrt() * self.calib_scale


# ══════════════════════════════════════════════════════════════════════════════
#  PERTE — NLL gaussienne aux capteurs tenus à l'écart
# ══════════════════════════════════════════════════════════════════════════════

class HeldOutNLL(nn.Module):
    """L = NLL(mu, sigma² ; obs)|held-out
          + w_in · NLL|entrée            (0 par défaut : évite la copie)
          + lambda_tv · variation totale de mu   (a priori, sans vérité)
          + Σ w_k · NLL_aux|held-out             (deep supervision conservée)

    Le terme TV remplace le `lambda_grad` de AELoss, qui comparait le gradient
    de la prédiction à celui de la VÉRITÉ — inutilisable ici.
    """

    def __init__(self, w_in=0.0, lambda_tv=0.02, aux_weights=(0.4, 0.3, 0.2),
                 ocean=None):
        super().__init__()
        self.w_in = w_in
        self.lambda_tv = lambda_tv
        self.aux_weights = aux_weights
        self.ocean = ocean

    @staticmethod
    def _nll(mu, logvar, y, m):
        """NLL gaussienne moyennée sur les points où m == 1."""
        n = m.sum()
        if n < 1:
            return mu.sum() * 0.0
        se = (y - mu) ** 2
        term = 0.5 * (logvar + se / logvar.exp().clamp_min(1e-8)
                      + float(np.log(2 * np.pi)))
        return (term * m).sum() / n

    def _tv(self, mu):
        gx = (mu[..., 1:, :] - mu[..., :-1, :]).abs()
        gy = (mu[..., :, 1:] - mu[..., :, :-1]).abs()
        if self.ocean is None:
            return gx.mean() + gy.mean()
        ox = self.ocean[..., 1:, :] * self.ocean[..., :-1, :]
        oy = self.ocean[..., :, 1:] * self.ocean[..., :, :-1]
        return ((gx * ox).sum() / (ox.sum() * mu.shape[1] + 1e-9)
                + (gy * oy).sum() / (oy.sum() * mu.shape[1] + 1e-9))

    def forward(self, mu, logvar, y, m_held, m_in, aux=None):
        l_held = self._nll(mu, logvar, y, m_held)
        loss = l_held
        if self.w_in > 0:
            loss = loss + self.w_in * self._nll(mu, logvar, y, m_in)
        if self.lambda_tv > 0:
            loss = loss + self.lambda_tv * self._tv(mu)
        l_aux = torch.zeros((), device=mu.device)
        if aux:
            H, W = y.shape[-2:]
            for (a_mu, a_lv), w in zip(aux, self.aux_weights):
                a_mu = F.interpolate(a_mu, (H, W), mode="bilinear",
                                     align_corners=False)
                a_lv = F.interpolate(a_lv, (H, W), mode="bilinear",
                                     align_corners=False)
                l_aux = l_aux + w * self._nll(a_mu, a_lv, y, m_held)
            loss = loss + l_aux
        return loss, l_held.detach(), l_aux.detach()


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET — masquage par capteur / par groupe
# ══════════════════════════════════════════════════════════════════════════════

class SensorMaskingDataset(Dataset):
    """Un échantillon = un pas de temps.

    À chaque tirage, les capteurs actifs sont scindés en deux :
      · entrée    -> peints dans x  [T_obs, S_obs, mask_T, mask_S]
      · held-out  -> peints dans y + m_held, jamais visibles en entrée

    Le tirage se fait PAR GROUPE (un glider entier, pas un point de glider).
    Les capteurs `forbidden` (split val/test) ne sont jamais mis en entrée :
    ils sont soit held-out, soit exclus, selon `use_forbidden_as_target`.
    """

    def __init__(self, obs: ObsSet, sids_input, sids_target=None,
                 drop_frac=(0.15, 0.40), rng_seed=0, min_input=3,
                 use_forbidden_as_target=True, augment="none"):
        self.obs = obs
        self.nx, self.ny, self.nt = obs.nx, obs.ny, obs.nt
        self.set_input = set(int(s) for s in sids_input)
        self.set_target = (set(int(s) for s in sids_target)
                           if sids_target is not None else set())
        self.drop_frac = drop_frac
        self.min_input = min_input
        self.use_forbidden = use_forbidden_as_target
        self.rng_seed = rng_seed
        # "flip" = retournement zonal (comme build_datasets de la Brique 1)
        # "roll" = translation zonale torique. Sans augmentation, l'AE
        # mémorise la climatologie de CHAQUE pixel-capteur au lieu
        # d'apprendre un opérateur d'interpolation transférable.
        self.augment = augment

        idx = obs.index_by_time()
        self.times = [t for t in range(self.nt)
                      if len(idx[t]) >= min_input + 1]
        # groupe de chaque capteur
        self.group_of = {s.sid: s.group for s in obs.sensors}

    def __len__(self):
        return len(self.times)

    def __getitem__(self, k):
        t = self.times[k]
        rows = self.obs.at(t)
        rng = np.random.default_rng(self.rng_seed * 1_000_003 + t)

        sid_row = self.obs.sid[rows]
        grp_row = np.array([self.group_of[int(s)] for s in sid_row])

        in_pool = np.array([int(s) in self.set_input for s in sid_row])
        tgt_pool = np.array([int(s) in self.set_target for s in sid_row])

        # groupes candidats au masquage : ceux de l'ensemble d'entrée
        groups = np.unique(grp_row[in_pool]) if in_pool.any() else np.array([])
        f = rng.uniform(*self.drop_frac)
        n_drop = int(np.clip(round(f * len(groups)), 1, max(0, len(groups) - 1))) \
            if len(groups) > 1 else 0
        dropped = set(rng.choice(groups, n_drop, replace=False).tolist()) \
            if n_drop > 0 else set()

        is_held = (np.array([g in dropped for g in grp_row]) & in_pool)
        if self.use_forbidden:
            is_held = is_held | tgt_pool
        is_in = in_pool & ~is_held

        # garde-fou : jamais moins de min_input points en entrée
        if is_in.sum() < self.min_input and is_held.any():
            move = np.where(is_held & in_pool)[0][: self.min_input]
            is_in[move] = True
            is_held[move] = False

        x = np.zeros((4, self.nx, self.ny), np.float32)
        y = np.zeros((2, self.nx, self.ny), np.float32)
        m_in = np.zeros((2, self.nx, self.ny), np.float32)
        m_held = np.zeros((2, self.nx, self.ny), np.float32)

        xs, ys = self.obs.x[rows], self.obs.y[rows]
        val, has = self.obs.val[rows], self.obs.has[rows]

        for v in range(2):
            sel = is_in & has[:, v]
            x[v, xs[sel], ys[sel]] = val[sel, v]
            x[2 + v, xs[sel], ys[sel]] = 1.0
            m_in[v, xs[sel], ys[sel]] = 1.0
            sel = is_held & has[:, v]
            y[v, xs[sel], ys[sel]] = val[sel, v]
            m_held[v, xs[sel], ys[sel]] = 1.0
        # les points d'entrée servent aussi de cible si w_in > 0
        for v in range(2):
            sel = is_in & has[:, v]
            y[v, xs[sel], ys[sel]] = val[sel, v]

        if self.augment != "none":
            arrs = [x, y, m_in, m_held]
            if "roll" in self.augment:
                sh = int(rng.integers(0, self.nx))
                arrs = [np.roll(a, sh, axis=-2) for a in arrs]
            if "flip" in self.augment and rng.random() < 0.5:
                arrs = [a[..., ::-1, :].copy() for a in arrs]
            x, y, m_in, m_held = arrs

        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m_held)),
                torch.from_numpy(np.ascontiguousarray(m_in)))


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION — vérifiable sur observations seules
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def calibration_report(model, loader, n_mc=8, device=DEVICE):
    """Diagnostics calculés UNIQUEMENT sur les capteurs tenus à l'écart :
      · RMSE et CRPS gaussien
      · z-score standardisé : sigma_emp de (y-mu)/sigma doit valoir 1
      · histogramme PIT (uniforme si calibré)
      · coverage à 95 %
    """
    model.eval()
    zs, errs, sig = [], [], []
    for x, y, m_held, _ in loader:
        x, y, m_held = x.to(device), y.to(device), m_held.to(device)
        mu, sd = model.predict(x, n_mc=n_mc)
        sel = m_held > 0.5
        if sel.sum() == 0:
            continue
        e = (y - mu)[sel]
        s = sd[sel].clamp_min(1e-6)
        zs.append((e / s).cpu().numpy())
        errs.append(e.cpu().numpy())
        sig.append(s.cpu().numpy())
    if not zs:
        return {}
    z = np.concatenate(zs); e = np.concatenate(errs); s = np.concatenate(sig)
    from math import erf, sqrt
    pit = 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    # CRPS gaussien analytique
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    Phi = pit
    crps = s * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return {
        "n": int(len(z)),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "crps": float(crps.mean()),
        "z_std": float(z.std()),                      # 1.0 = calibré
        "coverage_95": float((np.abs(z) < 1.96).mean()),
        "pit_hist": (hist / hist.sum()).round(3).tolist(),
        "sigma_mean": float(s.mean()),
    }


@torch.no_grad()
def fit_variance_scale(model, loader, n_mc=8, device=DEVICE):
    """Facteur d'échelle s tel que sigma_calibré = s · sigma_prédit.

    Une tête hétéroscédastique entraînée en NLL est systématiquement
    SUR-CONFIANTE hors distribution : elle apprend la dispersion des capteurs
    d'entraînement, pas celle des capteurs qu'elle n'a jamais vus. s = std(z)
    mesuré sur la validation corrige ce biais d'un seul paramètre.

    Ajusté sur VAL, à reporter sur TEST — sinon la correction est circulaire.
    """
    model.eval()
    zs = []
    for x, y, m_held, _ in loader:
        x, y, m_held = x.to(device), y.to(device), m_held.to(device)
        mu, sd = model.predict(x, n_mc=n_mc)
        sel = m_held > 0.5
        if sel.sum() == 0:
            continue
        zs.append((((y - mu)[sel]) / sd[sel].clamp_min(1e-6)).cpu().numpy())
    if not zs:
        return 1.0
    return float(np.concatenate(zs).std())


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════

def train_obsonly(args):
    print("=" * 70)
    print("  Brique 1b — AE obs-only (NLL sur capteurs tenus à l'écart)")
    print("=" * 70)
    set_global_seed(args.seed)

    obs = ObsSet.load(args.obs)
    print("\n[1/4] " + obs.summary().replace("\n", "\n      "))

    sp = split_sensors(obs, frac_fit=args.frac_fit, frac_val=args.frac_val,
                       seed=args.seed)
    print(f"\n[2/4] Split PAR CAPTEUR : fit={len(sp['fit'])} "
          f"val={len(sp['val'])} test={len(sp['test'])}")
    print("      (aucun capteur de val/test n'entre jamais en entrée)")

    ds_tr = SensorMaskingDataset(obs, sp["fit"], sids_target=None,
                                 drop_frac=tuple(args.drop_frac),
                                 rng_seed=args.seed, augment=args.augment)
    ds_va = SensorMaskingDataset(obs, sp["fit"], sids_target=sp["val"],
                                 drop_frac=(0.0, 0.0), rng_seed=args.seed + 1)
    ds_te = SensorMaskingDataset(obs, sp["fit"], sids_target=sp["test"],
                                 drop_frac=(0.0, 0.0), rng_seed=args.seed + 2)
    ld_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)
    ld_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)
    print(f"      {len(ds_tr)} pas de temps exploitables")

    ae_mod = _load_ae_module(ae_class=args.ae_class)
    model = ObservabilityAEHetero(ae_mod, base_ch=args.base_ch,
                                  latent_ch=args.latent_ch,
                                  dropout_p=args.dropout_p).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n[3/4] Modèle hétéroscédastique : {n_par:,} paramètres")

    oc = None
    if obs.ocean is not None:
        oc = torch.from_numpy(obs.ocean.astype(np.float32))[None, None].to(DEVICE)
        print(f"      masque océan actif : {100 * obs.ocean.mean():.1f} % "
              "de la grille (la TV ignore la terre)")
    crit = HeldOutNLL(w_in=args.w_in, lambda_tv=args.lambda_tv, ocean=oc)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best, hist = np.inf, []
    best_ep, since = 0, 0

    for ep in range(1, args.epochs + 1):
        model.train(); tot = nh = 0.0
        for x, y, mh, mi in ld_tr:
            x, y, mh, mi = (x.to(DEVICE), y.to(DEVICE),
                            mh.to(DEVICE), mi.to(DEVICE))
            mu, lv, _, aux = model(x)
            loss, l_h, _ = crit(mu, lv, y, mh, mi, aux)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nh += float(l_h)
        sched.step()
        n = max(1, len(ld_tr))

        rep = calibration_report(model, ld_va, n_mc=args.n_mc_val)
        hist.append({"epoch": ep, "loss": tot / n, "nll_held": nh / n, **rep})
        if ep % max(1, args.epochs // 10) == 0 or ep == 1:
            print(f"  ep {ep:3d}/{args.epochs} | L={tot/n:7.3f} "
                  f"| NLL_held={nh/n:7.3f} | RMSE_val={rep.get('rmse', 0):.4f} "
                  f"| z_std={rep.get('z_std', 0):.2f} "
                  f"| cov95={rep.get('coverage_95', 0):.2f}")

        crit_val = rep.get("crps", np.inf)
        if crit_val < best - 1e-6:
            best, best_ep, since = crit_val, ep, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "split": {k: v.tolist() for k, v in sp.items()},
                        "calib": rep},
                       out_dir / "ae_obsonly.pt")
        else:
            since += 1
            if args.patience and since >= args.patience:
                print(f"  arrêt anticipé (époque {ep}, pas d'amélioration "
                      f"depuis {since})")
                break

    print(f"\n[4/4] Meilleur CRPS val : {best:.4f} (époque {best_ep})")
    print(f"      Checkpoint → {out_dir}/ae_obsonly.pt")
    (out_dir / "obsonly_history.json").write_text(json.dumps(hist, indent=1))

    # ── recalibration sur VAL, contrôle indépendant sur TEST ──────────────
    ck = torch.load(out_dir / "ae_obsonly.pt", map_location=DEVICE,
                    weights_only=False)
    model.load_state_dict(ck["model_state"])
    before = calibration_report(model, ld_te, n_mc=args.n_mc_val)

    s_cal = fit_variance_scale(model, ld_va, n_mc=args.n_mc_val)
    model.calib_scale.fill_(s_cal)
    after = calibration_report(model, ld_te, n_mc=args.n_mc_val)

    ck["model_state"] = model.state_dict()
    ck["calib_scale"] = s_cal
    ck["calib_test"] = after
    torch.save(ck, out_dir / "ae_obsonly.pt")

    print(f"\n  Recalibration de la variance : sigma <- {s_cal:.2f} x sigma")
    print("  (ajusté sur VAL, mesuré ci-dessous sur TEST — jamais utilisé)")
    print(f"\n  {'':<14s} {'avant':>8s} {'après':>8s}   attendu")
    print(f"    z_std      {before['z_std']:>8.2f} {after['z_std']:>8.2f}   1.00")
    print(f"    cover 95%  {before['coverage_95']:>8.2f} "
          f"{after['coverage_95']:>8.2f}   0.95")
    print(f"    CRPS       {before['crps']:>8.4f} {after['crps']:>8.4f}   (plus bas = mieux)")
    if before["z_std"] > 2.0:
        print("\n  [!] z_std > 2 avant recalibration : le modèle généralise mal")
        print("      d'un capteur à l'autre. Il apprend la climatologie de")
        print("      chaque pixel plutôt qu'un opérateur d'interpolation.")
        print("      Essayez --augment flip_roll, --base_ch 16, --patience 10.")
    return model, obs, sp


# ══════════════════════════════════════════════════════════════════════════════
#  LOBO — contribution marginale par capteur, sans vérité
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def lobo_scores(model, obs: ObsSet, sids_active=None, n_mc=8, max_t=200,
                seed=0, device=DEVICE):
    """Leave-One-Buoy-Out : pour chaque capteur k, on le retire de l'entrée et
    on mesure la dégradation aux AUTRES observations disponibles.

        delta_k = CRPS(réseau sans k) − CRPS(réseau complet)

    Aucune vérité n'intervient : la référence est l'observation elle-même.
    delta_k élevé = capteur irremplaçable ; delta_k ≈ 0 = redondant.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    sids = np.array(sorted(sids_active if sids_active is not None
                           else [s.sid for s in obs.sensors]), np.int32)
    idx = obs.index_by_time()
    times = [t for t in range(obs.nt) if len(idx[t]) > 4]
    if len(times) > max_t:
        times = sorted(rng.choice(times, max_t, replace=False).tolist())

    group_of = {s.sid: s.group for s in obs.sensors}
    base_crps = np.zeros(len(times))
    sums = {int(k): 0.0 for k in sids}
    counts = {int(k): 0 for k in sids}

    def _paint(rows, keep_mask):
        x = np.zeros((4, obs.nx, obs.ny), np.float32)
        xs, ys = obs.x[rows], obs.y[rows]
        val, has = obs.val[rows], obs.has[rows]
        for v in range(2):
            sel = keep_mask & has[:, v]
            x[v, xs[sel], ys[sel]] = val[sel, v]
            x[2 + v, xs[sel], ys[sel]] = 1.0
        return torch.from_numpy(x)[None].to(device)

    def _crps_at(x, rows, target_mask):
        y = np.zeros((2, obs.nx, obs.ny), np.float32)
        m = np.zeros((2, obs.nx, obs.ny), np.float32)
        xs, ys = obs.x[rows], obs.y[rows]
        val, has = obs.val[rows], obs.has[rows]
        for v in range(2):
            sel = target_mask & has[:, v]
            y[v, xs[sel], ys[sel]] = val[sel, v]
            m[v, xs[sel], ys[sel]] = 1.0
        y = torch.from_numpy(y)[None].to(device)
        m = torch.from_numpy(m)[None].to(device)
        mu, sd = model.predict(x, n_mc=n_mc)
        sel = m > 0.5
        if sel.sum() == 0:
            return None
        z = ((y - mu)[sel] / sd[sel].clamp_min(1e-6))
        Phi = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        phi = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        crps = sd[sel] * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
        return float(crps.mean())

    for ti, t in enumerate(times):
        rows = obs.at(t)
        sid_row = obs.sid[rows]
        present = np.isin(sid_row, sids)
        if present.sum() < 4:
            continue
        # cible : un sous-ensemble tenu à l'écart, fixe pour ce pas de temps
        grp_row = np.array([group_of[int(s)] for s in sid_row])
        groups = np.unique(grp_row[present])
        held_g = set(rng.choice(groups, max(1, len(groups) // 5),
                                replace=False).tolist())
        is_held = np.array([g in held_g for g in grp_row]) & present
        is_in = present & ~is_held
        if is_in.sum() < 3 or is_held.sum() == 0:
            continue

        c0 = _crps_at(_paint(rows, is_in), rows, is_held)
        if c0 is None:
            continue
        base_crps[ti] = c0

        for k in np.unique(sid_row[is_in]):
            drop = is_in & (sid_row != k)
            if drop.sum() < 3:
                continue
            ck = _crps_at(_paint(rows, drop), rows, is_held)
            if ck is None:
                continue
            sums[int(k)] += ck - c0
            counts[int(k)] += 1

    out = {}
    for k in sids:
        k = int(k)
        out[k] = sums[k] / counts[k] if counts[k] > 0 else np.nan
    return out, float(np.mean(base_crps[base_crps > 0]))


def _stratified_times(obs, sids, max_t, min_per_sensor, rng):
    """Pas de temps garantissant une couverture MINIMALE de chaque capteur.

    Un tirage uniforme sur 3653 pas laisse sans score les capteurs souvent en
    panne ou de courte durée de vie (ils sortent en NaN). On tire d'abord
    min_per_sensor instants où chaque capteur est présent, puis on complète.
    """
    idx = obs.index_by_time()
    usable = [t for t in range(obs.nt) if len(idx[t]) > 4]
    alive = {int(k): [] for k in sids}
    keep = set(int(k) for k in sids)
    for t in usable:
        for sd in np.unique(obs.sid[idx[t]]):
            if int(sd) in keep:
                alive[int(sd)].append(t)
    chosen = set()
    for k in sids:
        av = alive[int(k)]
        if av:
            n = min(min_per_sensor, len(av))
            chosen.update(rng.choice(av, n, replace=False).tolist())
    rest = [t for t in usable if t not in chosen]
    if len(chosen) < max_t and rest:
        extra = rng.choice(rest, min(max_t - len(chosen), len(rest)),
                           replace=False)
        chosen.update(extra.tolist())
    return sorted(chosen)


@torch.no_grad()
def lobo_sigma_scores(model, obs: ObsSet, sids_active=None, n_mc=4, max_t=40,
                      seed=0, device=DEVICE, min_per_sensor=8):
    """Contribution marginale par RÉDUCTION DE VARIANCE PRÉDICTIVE.

        delta_k = <sigma | réseau sans k> − <sigma | réseau complet>
                  moyenné sur TOUT le domaine

    Différence essentielle avec lobo_scores (mode CRPS) : celui-ci n'évalue
    qu'aux points OBSERVÉS, donc il mesure « à quel point k aide à prédire les
    autres capteurs ». Un capteur isolé dans un vide de données aide énormément
    la reconstruction du champ mais n'a aucun voisin à prédire — son delta CRPS
    est nul alors que sa contribution réelle est grande.

    Le mode sigma évalue partout, y compris là où il n'y a aucune observation.
    C'est la quantité de l'OED bayésien (réduction d'entropie prédictive), et
    elle est directement comparable au delta RMSE d'une interpolation optimale.
    Elle ne requiert toujours aucune vérité — seulement un sigma CALIBRÉ.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    sids = np.array(sorted(sids_active if sids_active is not None
                           else [s.sid for s in obs.sensors]), np.int32)
    times = _stratified_times(obs, sids, max_t, min_per_sensor, rng)

    sums = {int(k): 0.0 for k in sids}
    counts = {int(k): 0 for k in sids}
    base_tot, base_n = 0.0, 0

    # Moyenne restreinte à l'océan : sur une boîte tropicale Atlantique, ~44 %
    # de la grille est continentale. Le modèle y prédit une anomalie nulle avec
    # un sigma minuscule, ce qui dilue tout écart entre configurations réseau.
    w, wsum = None, 1.0
    if obs.ocean is not None:
        w = torch.from_numpy(obs.ocean.astype(np.float32))[None, None].to(device)
        wsum = float(w.sum()) * 2      # 2 variables

    def _paint(rows, keep):
        x = np.zeros((4, obs.nx, obs.ny), np.float32)
        xs, ys = obs.x[rows], obs.y[rows]
        val, has = obs.val[rows], obs.has[rows]
        for v in range(2):
            sel = keep & has[:, v]
            x[v, xs[sel], ys[sel]] = val[sel, v]
            x[2 + v, xs[sel], ys[sel]] = 1.0
        return x

    for t in times:
        rows = obs.at(t)
        sid_row = obs.sid[rows]
        present = np.isin(sid_row, sids)
        uniq = np.unique(sid_row[present])
        if len(uniq) < 4:
            continue

        x0 = torch.from_numpy(_paint(rows, present))[None].to(device)
        _, sd0 = model.predict(x0, n_mc=n_mc)
        s0 = float((sd0 * w).sum() / wsum) if w is not None else float(sd0.mean())
        base_tot += s0; base_n += 1

        batch = np.stack([_paint(rows, present & (sid_row != k)) for k in uniq])
        for i in range(0, len(batch), 8):
            xb = torch.from_numpy(batch[i:i + 8]).to(device)
            _, sdb = model.predict(xb, n_mc=n_mc)
            sk = ((sdb * w).sum(dim=(1, 2, 3)) / wsum).cpu().numpy() \
                if w is not None else sdb.mean(dim=(1, 2, 3)).cpu().numpy()
            for j, k in enumerate(uniq[i:i + 8]):
                sums[int(k)] += float(sk[j]) - s0
                counts[int(k)] += 1

    out = {int(k): (sums[int(k)] / counts[int(k)] if counts[int(k)] else np.nan)
           for k in sids}
    miss = [int(k) for k in sids if counts[int(k)] == 0]
    if miss:
        print(f"  [!] {len(miss)} capteur(s) sans score : jamais actifs sur "
              f"les {len(times)} instants tirés. Augmentez --lobo_t.")
    return out, (base_tot / max(1, base_n))


@torch.no_grad()
def monotonicity_check(model, obs: ObsSet, sizes=(4, 8, 16, 24, 32, 44),
                       n_t=20, n_rep=4, n_mc=4, seed=0, device=DEVICE):
    """sigma moyen en fonction du NOMBRE de capteurs en entrée.

    Test de cohérence indispensable avant d'utiliser sigma comme mesure
    d'information : ajouter des observations ne peut pas AUGMENTER
    l'incertitude. Si la courbe n'est pas décroissante, le mode
    --lobo_mode sigma est invalide et les delta_sigma négatifs en sont le
    symptôme, pas la cause.

    Cause la plus probable si le test échoue : le conditionnement global du
    décodeur (_get_cond alimenté par la densité MOYENNE du masque, ObsGate sur
    chaque skip). Retirer un capteur modifie alors la reconstruction partout,
    y compris loin de lui, et cet effet global peut dominer le gain local.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    idx = obs.index_by_time()
    times = [t for t in range(obs.nt) if len(idx[t]) >= max(sizes)]
    if not times:
        times = [t for t in range(obs.nt) if len(idx[t]) >= min(sizes) + 2]
    if not times:
        return []
    times = rng.choice(times, min(n_t, len(times)), replace=False).tolist()

    out = []
    for n_s in sizes:
        vals = []
        for t in times:
            rows = obs.at(t)
            uniq = np.unique(obs.sid[rows])
            if len(uniq) < n_s:
                continue
            for _ in range(n_rep):
                keep = set(rng.choice(uniq, n_s, replace=False).tolist())
                sel = np.array([int(s) in keep for s in obs.sid[rows]])
                x = np.zeros((4, obs.nx, obs.ny), np.float32)
                xs, ys = obs.x[rows], obs.y[rows]
                val, has = obs.val[rows], obs.has[rows]
                for v in range(2):
                    m = sel & has[:, v]
                    x[v, xs[m], ys[m]] = val[m, v]
                    x[2 + v, xs[m], ys[m]] = 1.0
                _, sd = model.predict(
                    torch.from_numpy(x)[None].to(device), n_mc=n_mc)
                if obs.ocean is not None:
                    ocm = torch.from_numpy(
                        obs.ocean.astype(np.float32))[None, None].to(device)
                    vals.append(float((sd * ocm).sum() / (ocm.sum() * 2)))
                else:
                    vals.append(float(sd.mean()))
        if vals:
            out.append((n_s, float(np.mean(vals)), float(np.std(vals))))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("AE obs-only")
    p.add_argument("--train", action="store_true")
    p.add_argument("--lobo", action="store_true")
    p.add_argument("--monotonic", action="store_true",
                   help="sigma décroît-il avec le nombre de capteurs ?")
    p.add_argument("--obs", default="outputs/obs_synth.npz")
    p.add_argument("--ckpt", default="outputs/ae_obsonly.pt")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--latent_ch", type=int, default=64)
    p.add_argument("--dropout_p", type=float, default=0.1)
    p.add_argument("--ae_class", default=None,
                   help="nom de la classe AE si non standard")
    p.add_argument("--w_in", type=float, default=0.0,
                   help="poids de la NLL aux points d'ENTRÉE (>0 = risque de copie)")
    p.add_argument("--lambda_tv", type=float, default=0.02)
    p.add_argument("--drop_frac", type=float, nargs=2, default=[0.15, 0.40])
    p.add_argument("--frac_fit", type=float, default=0.70)
    p.add_argument("--frac_val", type=float, default=0.15)
    # MCDropout2d force training=True même en eval() : une passe unique n'est
    # qu'un tirage de dropout parmi d'autres. Avec n_mc_val=1 la calibration
    # mesure ce bruit de tirage et non celle du modèle.
    p.add_argument("--n_mc_val", type=int, default=8)
    p.add_argument("--augment", default="flip_roll",
                   choices=["none", "flip", "roll", "flip_roll"],
                   help="augmentation zonale contre la mémorisation")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lobo_mc", type=int, default=8)
    p.add_argument("--lobo_min_per_sensor", type=int, default=8,
                   help="instants minimum où chaque capteur doit "
                        "être présent pour recevoir un score")
    p.add_argument("--lobo_t", type=int, default=150)
    p.add_argument("--lobo_mode", default="sigma", choices=["sigma", "crps"],
                   help="sigma = réduction de variance sur tout le domaine "
                        "(OED bayésien) ; crps = dégradation aux capteurs "
                        "observés seulement")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.train:
        train_obsonly(a)
    if a.lobo:
        obs = ObsSet.load(a.obs)
        ck = torch.load(a.ckpt, map_location=DEVICE, weights_only=False)
        ae_mod = _load_ae_module(ae_class=a.ae_class)
        m = ObservabilityAEHetero(ae_mod, base_ch=ck["args"]["base_ch"],
                                  latent_ch=ck["args"]["latent_ch"],
                                  dropout_p=ck["args"]["dropout_p"]).to(DEVICE)
        m.load_state_dict(ck["model_state"])
        cs = float(m.calib_scale.item())
        if abs(cs - 1.0) > 1e-6:
            print(f"  variance recalibrée : sigma x {cs:.2f}")
        if a.lobo_mode == "sigma":
            sc, base = lobo_sigma_scores(m, obs, n_mc=max(2, a.lobo_mc // 2),
                                         max_t=a.lobo_t, seed=a.seed,
                                         min_per_sensor=a.lobo_min_per_sensor)
            lbl, unit = "sigma moyen", "delta_sigma"
        else:
            sc, base = lobo_scores(m, obs, n_mc=a.lobo_mc, max_t=a.lobo_t,
                                   seed=a.seed)
            lbl, unit = "CRPS", "delta_CRPS"
        rank = sorted(sc.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -9))
        print(f"\n{lbl} réseau complet : {base:.4f}   [mode {a.lobo_mode}]")
        print(f"\n  rang  sid  type        {unit:>11s}   interprétation")
        # Regroupement par TYPE : un delta est moyenné sur les instants où le
        # capteur est vivant. Un dériveur (250 j) et un mouillage (3653 j) ne
        # sont pas notés sur les mêmes situations — les comparer directement
        # avantage systématiquement les plateformes de courte durée de vie.
        by_kind = {}
        for k, d in rank:
            if d == d:
                by_kind.setdefault(obs.sensors[k].kind, []).append((k, d))
        for kind in sorted(by_kind):
            lst = by_kind[kind]
            print(f"\n  --- {kind} ({len(lst)} capteurs notés) ---")
            for r, (k, d) in enumerate(lst[:15], 1):
                tag = "irremplaçable" if d > 0.02 * base else (
                    "utile" if d > 0.005 * base else "redondant")
                print(f"  {r:4d} {k:4d}  {d:+10.5f}   {tag}")
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.output_dir) / "lobo_ae.json").write_text(
            json.dumps({"mode": a.lobo_mode, "base": base, "base_crps": base,
                        "delta": {str(k): v for k, v in sc.items()}}, indent=1))
        print(f"\n  → {a.output_dir}/lobo_ae.json")
    if a.monotonic:
        obs = ObsSet.load(a.obs)
        ck = torch.load(a.ckpt, map_location=DEVICE, weights_only=False)
        ae_mod = _load_ae_module(ae_class=a.ae_class)
        m = ObservabilityAEHetero(ae_mod, base_ch=ck["args"]["base_ch"],
                                  latent_ch=ck["args"]["latent_ch"],
                                  dropout_p=ck["args"]["dropout_p"]).to(DEVICE)
        m.load_state_dict(ck["model_state"])
        res = monotonicity_check(m, obs, seed=a.seed)
        print("\n  N capteurs   sigma moyen   ecart-type")
        prev, bad = None, 0
        for n_s, mu, sd in res:
            flag = ""
            if prev is not None and mu > prev + 1e-6:
                flag = "  <-- AUGMENTE"; bad += 1
            print(f"  {n_s:>10d}   {mu:>11.4f}   {sd:>10.4f}{flag}")
            prev = mu
        if bad:
            print(f"\n  [!] {bad} violation(s) de monotonie : sigma n'est PAS")
            print("      une mesure d'information valide pour ce modèle.")
            print("      --lobo_mode sigma est inutilisable en l'état ;")
            print("      revenez à --lobo_mode crps en attendant.")
        else:
            print("\n  Monotonie respectée : sigma est utilisable comme")
            print("  mesure d'information (OED bayésien).")

    if not (a.train or a.lobo or a.monotonic):
        print("rien à faire : --train et/ou --lobo")
