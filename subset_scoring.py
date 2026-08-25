"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SCORING PAR SOUS-ENSEMBLES                                                  ║
║                                                                              ║
║  Le protocole par capteur a atteint son plancher : la contribution d'une     ║
║  bouée sur 17 vaut ~0.4 % de la RMSE, sous l'erreur d'estimation de la       ║
║  référence (concordance inter-périodes ~0.14).                               ║
║                                                                              ║
║  On change d'unité d'analyse : au lieu de noter 17 capteurs, on note des     ║
║  CONFIGURATIONS de réseau.                                                   ║
║    · le signal grossit — retirer 5 bouées sur 17 déplace la RMSE d'un ordre  ║
║      de grandeur de plus que d'en retirer une                                ║
║    · l'échantillon grossit — 300 configurations au lieu de 17 capteurs       ║
║    · c'est la question opérationnelle réelle : un SNO arbitre entre des      ║
║      configurations, pas entre des bouées prises isolément (Gasparin 2023    ║
║      compare NOMINAL contre ENHANCED, jamais des mouillages un par un)       ║
║                                                                              ║
║  PIÈGE CENTRAL : la TAILLE de la configuration est un confondant massif.     ║
║  Plus de bouées => RMSE plus basse ET sigma plus bas. Une corrélation        ║
║  globale mesurerait surtout « savoir compter les bouées ». Le chiffre qui    ║
║  compte est la corrélation À TAILLE FIXE, rapportée ici séparément.          ║
╚══════════════════════════════════════════════════════════════════════════════╝

    python subset_scoring.py --obs outputs/split_train/obs_synth.npz \\
        --truth outputs/split_train/_truth.npz --ckpt outputs/ae_obsonly.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from obs_operator import ObsSet
from validate_obsonly import (_cov_blend, spearman, estimate_decorrelation_px)


def _load(mod_path, name):
    spec = importlib.util.spec_from_file_location(name, mod_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  VÉRITÉ : RMSE d'interpolation optimale par CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def true_rmse_by_config(T, positions, configs, eval_stride=4, split=0.5,
                        noise_var=4e-4, L_px=20.0, shrink=0.3, ocean=None):
    """RMSE hors échantillon de l'OI pour chaque sous-ensemble de capteurs.

    La covariance est estimée UNE FOIS sur la première moitié, l'évaluation
    porte sur la seconde. Chaque configuration ne coûte alors qu'un système
    linéaire de taille |config|.
    """
    nt, nx, ny = T.shape
    n_fit = int(split * nt)
    gx, gy = np.meshgrid(np.arange(0, nx, eval_stride),
                         np.arange(0, ny, eval_stride), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    if ocean is not None:
        grid = grid[ocean[grid[:, 0], grid[:, 1]]]

    pos = np.array(positions, float)
    Yg_fit = T[:n_fit][:, grid[:, 0], grid[:, 1]]
    Yg_ev = T[n_fit:][:, grid[:, 0], grid[:, 1]]
    Ys_fit = T[:n_fit][:, pos[:, 0].astype(int), pos[:, 1].astype(int)]
    Ys_ev = T[n_fit:][:, pos[:, 0].astype(int), pos[:, 1].astype(int)]

    Css = _cov_blend(Ys_fit, pos, pos, L_px=L_px, shrink=shrink)
    Cgs = _cov_blend(Yg_fit, grid, pos, Ys_fit, L_px=L_px, shrink=shrink)

    out = np.zeros(len(configs))
    for i, sub in enumerate(configs):
        sub = list(sub)
        A = Css[np.ix_(sub, sub)] + noise_var * np.eye(len(sub))
        W = np.linalg.solve(A, Cgs[:, sub].T).T
        out[i] = float(np.sqrt(((Yg_ev - Ys_ev[:, sub] @ W.T) ** 2).mean()))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  OBS-ONLY : sigma moyen par CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sigma_by_config(model, obs, sids, configs, times, n_mc=2, batch=12,
                    device="cpu"):
    """sigma prédictif moyen, réseau restreint à chaque configuration.

    Seuls les capteurs de `sids` sont peints. Les autres plateformes sont
    exclues : on veut l'effet de la CONFIGURATION DE MOUILLAGES, pas celui
    d'un fond Argo/dériveurs qui varie d'un instant à l'autre.
    """
    model.eval()
    w = None
    if obs.ocean is not None:
        w = torch.from_numpy(obs.ocean.astype(np.float32))[None, None].to(device)
        wsum = float(w.sum()) * 2

    def paint(rows, keep_sids):
        x = np.zeros((4, obs.nx, obs.ny), np.float32)
        xs, ys = obs.x[rows], obs.y[rows]
        val, has = obs.val[rows], obs.has[rows]
        sel0 = np.isin(obs.sid[rows], list(keep_sids))
        for v in range(2):
            m = sel0 & has[:, v]
            x[v, xs[m], ys[m]] = val[m, v]
            x[2 + v, xs[m], ys[m]] = 1.0
        return x

    tot = np.zeros(len(configs))
    cnt = np.zeros(len(configs))
    for t in times:
        rows = obs.at(t)
        present = set(int(s) for s in np.unique(obs.sid[rows]))
        stack, keep_i = [], []
        for i, cfg in enumerate(configs):
            live = [sids[j] for j in cfg if sids[j] in present]
            if len(live) < 2:
                continue
            stack.append(paint(rows, live)); keep_i.append(i)
        if not stack:
            continue
        stack = np.stack(stack)
        for b in range(0, len(stack), batch):
            xb = torch.from_numpy(stack[b:b + batch]).to(device)
            _, sd = model.predict(xb, n_mc=n_mc)
            v = ((sd * w).sum(dim=(1, 2, 3)) / wsum if w is not None
                 else sd.mean(dim=(1, 2, 3))).cpu().numpy()
            for j, val_ in enumerate(v):
                tot[keep_i[b + j]] += float(val_)
                cnt[keep_i[b + j]] += 1
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), cnt


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(a):
    print("=" * 70)
    print("  Scoring par sous-ensembles de mouillages")
    print("=" * 70)

    obs = ObsSet.load(a.obs)
    tr = np.load(a.truth)
    T = tr["T"][:obs.nt]
    fixed = [k for k, s in enumerate(obs.sensors) if s.is_fixed]
    pos = [obs.sensors[k].mean_pos for k in fixed]
    n = len(fixed)
    print(f"\n[1/4] {n} mouillages | nature run {T.shape}")
    if n < 8:
        sys.exit("moins de 8 mouillages : sous-ensembles inexploitables")

    sizes = [s for s in range(a.size_min, min(a.size_max, n - 1) + 1)]
    rng = np.random.default_rng(a.seed)
    configs, cfg_size = [], []
    for s in sizes:
        for _ in range(a.per_size):
            configs.append(tuple(sorted(rng.choice(n, s, replace=False))))
            cfg_size.append(s)
    # dédoublonnage
    seen, keep = set(), []
    for i, c in enumerate(configs):
        if c not in seen:
            seen.add(c); keep.append(i)
    configs = [configs[i] for i in keep]
    cfg_size = np.array([cfg_size[i] for i in keep])
    print(f"      {len(configs)} configurations, tailles {sizes[0]}–{sizes[-1]}")

    L_px = a.influence_px
    if L_px <= 0:
        L_px = estimate_decorrelation_px(T, ocean=obs.ocean, verbose=False)
        if not np.isfinite(L_px):
            L_px = 20.0
    print(f"\n[2/4] Vérité : RMSE d'OI par configuration (L={L_px:.1f} px)")
    rmse = true_rmse_by_config(T, pos, configs, eval_stride=a.eval_stride,
                               noise_var=a.noise_var, L_px=L_px,
                               shrink=a.shrinkage, ocean=obs.ocean)
    print(f"      RMSE : {rmse.min():.4f} – {rmse.max():.4f} "
          f"(étendue {100 * (rmse.max() - rmse.min()) / rmse.mean():.1f} % "
          "de la moyenne)")
    print("      Rappel : par capteur, l'étendue était de ~0.4 %.")

    print(f"\n[3/4] Obs-only : sigma par configuration")
    ck = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    ae = _load("obsonly.py", "obsonly_mod")
    ae_mod = ae._load_ae_module()
    model = ae.ObservabilityAEHetero(
        ae_mod, base_ch=ck["args"]["base_ch"], latent_ch=ck["args"]["latent_ch"],
        dropout_p=ck["args"]["dropout_p"]).to(a.device)
    model.load_state_dict(ck["model_state"])

    idx = obs.index_by_time()
    usable = [t for t in range(obs.nt) if len(idx[t]) > 4]
    times = sorted(rng.choice(usable, min(a.n_times, len(usable)),
                              replace=False).tolist())
    sig, cnt = sigma_by_config(model, obs, fixed, configs, times,
                               n_mc=a.n_mc, batch=a.batch, device=a.device)
    print(f"      {int((cnt > 0).sum())}/{len(configs)} configurations notées "
          f"sur {len(times)} instants")

    print(f"\n[4/4] Corrélation")
    ok = np.isfinite(sig) & np.isfinite(rmse)
    r_all, _ = spearman(sig[ok], rmse[ok])
    r_size, _ = spearman(cfg_size[ok].astype(float), rmse[ok])
    print(f"\n  {'':<28s} {'Spearman':>9s}   interprétation")
    print(f"  {'sigma obs-only vs RMSE':<28s} {r_all:>+9.3f}   "
          "global (confondu par la taille)")
    print(f"  {'taille seule vs RMSE':<28s} {r_size:>+9.3f}   "
          "ce que 'savoir compter' suffit à obtenir")

    print(f"\n  À TAILLE FIXE — le chiffre qui compte :")
    print(f"  {'taille':>7s} {'n cfg':>6s} {'Spearman':>9s}")
    rs, ws = [], []
    for s in sizes:
        m = ok & (cfg_size == s)
        if m.sum() < 8:
            continue
        r, _ = spearman(sig[m], rmse[m])
        if np.isfinite(r):
            rs.append(r); ws.append(int(m.sum()))
            print(f"  {s:>7d} {int(m.sum()):>6d} {r:>+9.3f}")
    if rs:
        mean_r = float(np.average(rs, weights=ws))
        print(f"\n  moyenne pondérée à taille fixe : {mean_r:+.3f} "
              f"({sum(ws)} configurations)")
        if mean_r > 0.5:
            print("  -> sigma obs-only classe correctement les configurations")
            print("     à effectif égal. C'est le résultat visé : le modèle")
            print("     capte OÙ sont les bouées, pas seulement combien.")
        elif mean_r > 0.25:
            print("  -> signal modéré mais réel à effectif égal.")
        else:
            print("  -> à effectif égal, sigma n'apporte rien : la corrélation")
            print("     globale ne reflétait que le nombre de bouées.")

    out = Path(a.output_dir) / "subset_scoring.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_configs": len(configs), "spearman_global": r_all,
         "spearman_size_only": r_size,
         "spearman_by_size": {int(s): float(r) for s, r in zip(sizes, rs)},
         "rmse": rmse.tolist(), "sigma": sig.tolist(),
         "size": cfg_size.tolist(),
         "configs": [list(map(int, c)) for c in configs]}, indent=1))
    print(f"\n  → {out}")


def parse_args():
    p = argparse.ArgumentParser("scoring par sous-ensembles")
    p.add_argument("--obs", default="outputs/split_train/obs_synth.npz")
    p.add_argument("--truth", default="outputs/split_train/_truth.npz")
    p.add_argument("--ckpt", default="outputs/ae_obsonly.pt")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--size_min", type=int, default=8)
    p.add_argument("--size_max", type=int, default=15)
    p.add_argument("--per_size", type=int, default=40)
    p.add_argument("--n_times", type=int, default=25)
    p.add_argument("--n_mc", type=int, default=2)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--eval_stride", type=int, default=4)
    p.add_argument("--noise_var", type=float, default=4e-4)
    p.add_argument("--shrinkage", type=float, default=0.3)
    p.add_argument("--influence_px", type=float, default=-1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
