"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  VALIDATION DE L'ESTIMATEUR — l'expérience décisive                          ║
║                                                                              ║
║  Question : le classement des capteurs produit SANS vérité retrouve-t-il la  ║
║  contribution marginale VRAIE, connue puisqu'on est dans un nature run ?     ║
║                                                                              ║
║  Référence « vérité » : interpolation optimale (OI) du champ complet depuis  ║
║  le réseau, évaluée contre le nature run.                                    ║
║        delta_k^vrai = RMSE(champ | réseau privé de k) − RMSE(champ | réseau) ║
║  Covariance estimée sur la première moitié du run, évaluation sur la seconde ║
║  (sinon la covariance empirique sur-apprend — cf. EVF_SHRINKAGE dans         ║
║  config.py).                                                                 ║
║                                                                              ║
║  Le livrable n'est PAS « voici le réseau optimal ». C'est cette corrélation  ║
║  de rang : elle mesure si l'estimateur obs-only est transférable à un SNO    ║
║  réel, où la vérité n'existera pas.                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
    python validate_obsonly.py --obs outputs/obs_synth.npz \\
        --truth outputs/_truth.npz \\
        --lobo_ae outputs/lobo_ae.json --lobo_gnn outputs/lobo_gnn.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from obs_operator import ObsSet

try:
    from config import INFLUENCE_RADIUS_KM, EVF_SHRINKAGE, DX_KM
except Exception:
    INFLUENCE_RADIUS_KM, EVF_SHRINKAGE, DX_KM = 90.0, 0.9, 5.0


# ══════════════════════════════════════════════════════════════════════════════
#  RÉFÉRENCE : CONTRIBUTION MARGINALE VRAIE PAR INTERPOLATION OPTIMALE
# ══════════════════════════════════════════════════════════════════════════════

def estimate_decorrelation_px(T, n_t=40, n_pairs=40000, seed=0,
                              verbose=True, ocean=None):
    """Longueur de décorrélation MESURÉE sur le champ, en pixels.

    INFLUENCE_RADIUS_KM / DX_KM est une valeur héritée de config.py, calibrée
    pour un autre usage. La référence OI en dépend directement : la mesurer
    évite de valider l'estimateur contre un a priori géométrique arbitraire.
    Ajuste rho(d) = exp(-d^2 / 2L^2) sur la corrélation empirique binnée.
    """
    rng = np.random.default_rng(seed)
    nt, nx, ny = T.shape
    ts = rng.choice(nt, min(n_t, nt), replace=False)
    A = T[ts].reshape(len(ts), -1)
    A = A - A.mean(0)
    pool = (np.flatnonzero(ocean.ravel()) if ocean is not None
            else np.arange(nx * ny))
    idx = rng.choice(pool, min(n_pairs, len(pool)), replace=False)
    i = rng.choice(idx, 4000); j = rng.choice(idx, 4000)
    xi, yi = np.unravel_index(i, (nx, ny)); xj, yj = np.unravel_index(j, (nx, ny))
    d = np.hypot(xi - xj, yi - yj)
    a, b = A[:, i], A[:, j]
    num = (a * b).mean(0)
    den = a.std(0) * b.std(0) + 1e-9
    r = num / den
    bins = np.linspace(0, d.max(), 25)
    k = np.digitize(d, bins)
    dd, rr = [], []
    for m in range(1, len(bins)):
        sel = k == m
        if sel.sum() > 30:
            dd.append(d[sel].mean()); rr.append(r[sel].mean())
    dd, rr = np.array(dd), np.array(rr)
    if verbose:
        k = max(1, len(dd) // 8)
        print("  profil rho(d) : " + "  ".join(
            f"{d_:.0f}px:{r_:+.2f}" for d_, r_ in zip(dd[::k], rr[::k])))

    # 1) PREMIÈRE traversée de 1/e : L = d / sqrt(2) pour un noyau gaussien.
    #    Critère prioritaire car local et robuste. L'ajustement log-linéaire
    #    sur tout le profil est dominé par la QUEUE : un plateau résiduel à
    #    rho ~ 0.1 (téléconnexion, mode de grande échelle) tire la pente vers
    #    zéro et produit un L absurde, plusieurs fois le vrai.
    below = np.where(rr < np.exp(-1.0))[0]
    if len(below):
        L_cross = float(dd[below[0]] / np.sqrt(2))
        # ajustement local, restreint aux points AVANT la traversée
        m = np.arange(len(rr)) <= below[0]
        m &= rr > 0.05
        if m.sum() >= 3:
            sl = np.polyfit(dd[m] ** 2, np.log(np.clip(rr[m], 1e-3, .999)), 1)[0]
            if sl < 0:
                L_fit = float(np.sqrt(-1.0 / (2 * sl)))
                if 0.5 * L_cross < L_fit < 2 * L_cross:
                    return L_fit          # les deux concordent
        if verbose:
            print(f"  L par traversée 1/e : {L_cross:.1f} px "
                  "(ajustement global écarté, queue non nulle)")
        return L_cross

    # 2) repli : ajustement global, faute de mieux
    m = rr > 0.05
    if m.sum() >= 3:
        slope = np.polyfit(dd[m] ** 2, np.log(np.clip(rr[m], 1e-3, 0.999)), 1)[0]
        if slope < 0:
            return float(np.sqrt(-1.0 / (2 * slope)))

    # 3) rho ne descend jamais sous 1/e : champ cohérent sur tout le domaine
    if verbose:
        print("  [!] rho(d) ne passe jamais sous 1/e sur le domaine — le champ")
        print("      est cohérent à grande échelle. L n'est pas identifiable ;")
        print("      la référence OI par noyau gaussien est mal posée ici.")
    return float("nan")


def _cov_blend(A, pos_a, pos_b, B=None, L_px=18.0, shrink=EVF_SHRINKAGE):
    """Covariance empirique mélangée à un modèle gaussien paramétrique.

    shrink = 1 -> modèle pur ; 0 -> empirique pure. La valeur par défaut
    (0.9) est celle diagnostiquée dans config.py : sur un an de nature run,
    la covariance d'échantillon sur-apprend.
    """
    B = A if B is None else B
    emp = (A - A.mean(0)) .T @ (B - B.mean(0)) / max(1, len(A) - 1)
    d2 = ((np.asarray(pos_a)[:, None] - np.asarray(pos_b)[None]) ** 2).sum(-1)
    rho = np.exp(-0.5 * d2 / L_px ** 2)
    sa = A.std(0)[:, None]; sb = B.std(0)[None, :]
    par = rho * sa * sb
    return (1 - shrink) * emp + shrink * par


def true_loo_contribution(T, positions, eval_stride=4, split=0.5,
                          noise_var=0.01, L_px=None, verbose=True,
                          shrink=None, ocean=None):
    """delta_k = RMSE_OI(sans k) − RMSE_OI(complet), calculé sur le champ vrai.

    T : (nt, nx, ny) nature run (une variable suffit pour le classement).
    """
    nt, nx, ny = T.shape
    L_px = L_px or (INFLUENCE_RADIUS_KM / max(DX_KM, 1e-6))
    n_fit = int(split * nt)
    gx, gy = np.meshgrid(np.arange(0, nx, eval_stride),
                         np.arange(0, ny, eval_stride), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    if ocean is not None:
        keep_g = ocean[grid[:, 0], grid[:, 1]]
        if keep_g.sum() < 20:
            raise ValueError("moins de 20 points d'évaluation en mer")
        grid = grid[keep_g]

    Yg_fit = T[:n_fit][:, grid[:, 0], grid[:, 1]]
    Yg_ev = T[n_fit:][:, grid[:, 0], grid[:, 1]]
    pos = np.array(positions, float)
    Ys_fit = T[:n_fit][:, pos[:, 0].astype(int), pos[:, 1].astype(int)]
    Ys_ev = T[n_fit:][:, pos[:, 0].astype(int), pos[:, 1].astype(int)]

    sh = EVF_SHRINKAGE if shrink is None else shrink
    Css = _cov_blend(Ys_fit, pos, pos, L_px=L_px, shrink=sh)
    Cgs = _cov_blend(Yg_fit, grid, pos, Ys_fit, L_px=L_px, shrink=sh)

    def rmse(sub):
        if len(sub) == 0:
            return float(np.sqrt((Yg_ev ** 2).mean()))
        A = Css[np.ix_(sub, sub)] + noise_var * np.eye(len(sub))
        W = np.linalg.solve(A, Cgs[:, sub].T).T
        pred = Ys_ev[:, sub] @ W.T
        return float(np.sqrt(((Yg_ev - pred) ** 2).mean()))

    n = len(positions)
    full = rmse(list(range(n)))
    delta = np.zeros(n)
    for k in range(n):
        delta[k] = rmse([i for i in range(n) if i != k]) - full
    if verbose:
        print(f"  OI référence : RMSE réseau complet = {full:.4f} "
              f"| L = {L_px:.0f} px | shrinkage = {sh}")
    return delta, full


# ══════════════════════════════════════════════════════════════════════════════
#  CORRÉLATIONS DE RANG
# ══════════════════════════════════════════════════════════════════════════════

def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan, int(m.sum())
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean()), int(m.sum())


def spearman_ci(a, b, n_boot=2000, seed=0):
    """IC 95 % par bootstrap sur les CAPTEURS + p-value par permutation.

    Avec 17 mouillages, l'IC d'un Spearman fait environ ±0.45 : un +0.30 et un
    -0.15 ne sont pas distinguables. Sans cette colonne, on lit des classements
    dans du bruit d'échantillonnage.
    """
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m], b[m]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    r0, _ = spearman(x, y)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(i)) < 4:
            continue
        r, _ = spearman(x[i], y[i])
        if np.isfinite(r):
            boots.append(r)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    perm = [spearman(x, y[rng.permutation(n)])[0] for _ in range(n_boot)]
    perm = np.array([p for p in perm if np.isfinite(p)])
    pval = float((np.abs(perm) >= abs(r0)).mean()) if len(perm) else np.nan
    return float(lo), float(hi), pval


def paired_diff_ci(a, b, truth, n_boot=4000, seed=0):
    """IC 95 % de la DIFFÉRENCE de Spearman entre deux estimateurs.

    Tester chaque estimateur contre zéro gaspille de la puissance : les deux
    sont évalués sur LES MÊMES 17 capteurs, donc leurs erreurs sont corrélées.
    Le bootstrap apparié sur les capteurs élimine cette variance commune et
    répond à la vraie question : l'estimateur appris porte-t-il une
    information que la géométrie seule ne porte pas ?
    """
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(truth)
    x, y, t = a[m], b[m], truth[m]
    n = len(t)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan
    d0 = spearman(x, t)[0] - spearman(y, t)[0]
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(i)) < 4:
            continue
        r1, r2 = spearman(x[i], t[i])[0], spearman(y[i], t[i])[0]
        if np.isfinite(r1) and np.isfinite(r2):
            ds.append(r1 - r2)
    if not ds:
        return d0, np.nan, np.nan, np.nan
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    # p bilatérale : proportion de rééchantillons de signe opposé
    pv = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return float(d0), float(lo), float(hi), float(min(1.0, pv))


def top_k_overlap(a, b, k):
    m = np.isfinite(a) & np.isfinite(b)
    idx = np.where(m)[0]
    ta = set(idx[np.argsort(-a[idx])[:k]])
    tb = set(idx[np.argsort(-b[idx])[:k]])
    return len(ta & tb) / max(1, k)


def _load_scores(path, key, n):
    if not path or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    out = np.full(n, np.nan)
    for k, v in d.get(key, {}).items():
        if v is not None:
            out[int(k)] = float(v)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 70)
    print("  Validation de l'estimateur obs-only contre la vérité du nature run")
    print("=" * 70)

    obs = ObsSet.load(args.obs)
    tr = np.load(args.truth)
    T = tr["T"][:obs.nt]
    n = len(obs.sensors)
    print(f"\n[1/3] {n} capteurs | nature run {T.shape}")
    if obs.ocean is not None:
        print(f"      masque océan : {100 * obs.ocean.mean():.1f} % — "
              "l'évaluation OI exclut les points à terre")

    # ── seuls les capteurs FIXES ont une contribution OI bien définie ──────
    fixed = [k for k, s in enumerate(obs.sensors) if s.is_fixed]
    print(f"      {len(fixed)} capteurs fixes retenus pour la référence OI "
          f"(les plateformes mobiles changent de position)")
    pos = [obs.sensors[k].mean_pos for k in fixed]

    print("\n[2/3] Contribution marginale VRAIE (interpolation optimale)")
    L_px = args.influence_px
    if L_px <= 0:
        L_meas = estimate_decorrelation_px(T, ocean=obs.ocean)
        L_cfg = INFLUENCE_RADIUS_KM / max(DX_KM, 1e-6)
        print(f"  L mesurée sur le champ : {L_meas:.1f} px "
              f"| L issue de config : {L_cfg:.1f} px")
        if np.isfinite(L_meas) and abs(L_meas - L_cfg) / max(L_cfg, 1e-9) > 0.3:
            print("  [!] écart > 30 % — la valeur de config.py ne décrit pas ce "
                  "champ ; c'est la valeur mesurée qui est utilisée")
        L_px = L_meas if np.isfinite(L_meas) else L_cfg
    d_true_f, full = true_loo_contribution(
        T, pos, eval_stride=args.eval_stride, split=args.split,
        noise_var=args.noise_var, L_px=L_px, shrink=args.shrinkage,
        ocean=obs.ocean)
    d_true = np.full(n, np.nan)
    d_true[fixed] = d_true_f

    print("\n[3/3] Comparaison avec les estimateurs obs-only")
    rows = []
    d_ae = _load_scores(args.lobo_ae, "delta", n)
    sk = _load_scores(args.lobo_gnn, "skill", n)
    d_gnn = _load_scores(args.lobo_gnn, "delta", n)

    if d_ae is not None:
        mode = "?"
        try:
            mode = json.loads(Path(args.lobo_ae).read_text()).get("mode", "?")
        except Exception:
            pass
        rows.append((f"AE obs-only  [{mode}]".ljust(24)[:24], d_ae))
    if d_gnn is not None:
        rows.append(("GNN obs-only delta_NLL ", d_gnn))
    if sk is not None:
        rows.append(("GNN obs-only −skill    ", -sk))

    # baseline naïve : distance au plus proche voisin (géométrie pure)
    P = np.array([s.mean_pos for s in obs.sensors], float)
    dmat = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    np.fill_diagonal(dmat, np.inf)
    rows.append(("baseline  d_plus_proche", dmat.min(1)))

    print(f"\n  {'estimateur':<24s} {'Spearman':>9s} {'IC 95%':>16s} "
          f"{'p':>6s} {'n':>4s} {'top-5':>7s}")
    print("  " + "-" * 72)
    res = {}
    for name, sc in rows:
        s_full, nn_ = spearman(sc, d_true)
        lo, hi, pv = spearman_ci(sc, d_true)
        t5 = top_k_overlap(sc, d_true, 5)
        t10 = top_k_overlap(sc, d_true, 10)
        res[name.strip()] = dict(spearman=s_full, ci=[lo, hi], p=pv,
                                 n=nn_, top5=t5, top10=t10)
        star = " *" if (pv == pv and pv < 0.05) else ""
        print(f"  {name:<24s} {s_full:>9.3f} [{lo:+.2f},{hi:+.2f}] "
              f"{pv:>6.3f} {nn_:>4d} {t5:>7.2f}{star}")
    print("  * = significatif au seuil 5 % (permutation). Sans étoile, le")
    print("    classement n'est pas distinguable du hasard à cette taille "
          "d'échantillon.")

    # ── comparaison APPARIÉE à la baseline : bien plus de puissance ────────
    base_sc = dict(rows).get("baseline  d_plus_proche")
    if base_sc is not None:
        print(f"\n  Écart à la baseline géométrique (bootstrap apparié)")
        print(f"  {'estimateur':<24s} {'delta rho':>10s} {'IC 95%':>16s} "
              f"{'p':>7s}")
        for name, sc in rows:
            if name.startswith("baseline"):
                continue
            d0, lo, hi, pv = paired_diff_ci(sc, base_sc, d_true)
            star = " *" if (pv == pv and pv < 0.05) else ""
            res.setdefault(name.strip(), {})["vs_baseline"] = \
                dict(delta=d0, ci=[lo, hi], p=pv)
            print(f"  {name:<24s} {d0:>+10.3f} [{lo:+.2f},{hi:+.2f}] "
                  f"{pv:>7.3f}{star}")
        print("  Les deux estimateurs sont notés sur LES MÊMES capteurs :")
        print("  l'appariement retire la variance commune et donne bien plus")
        print("  de puissance que deux tests séparés contre zéro.")

    print("\n  Lecture : Spearman > 0.6 et top-5 >= 0.6 = l'estimateur obs-only")
    print("  est transférable. En dessous, il ne l'est pas — et c'est un")
    print("  résultat, pas un échec : il faut alors corriger l'estimateur")
    print("  AVANT de toucher à un SNO réel.")
    print("\n  Attention : comparer aussi à la baseline géométrique. Un")
    print("  estimateur appris qui ne bat pas 'distance au plus proche voisin'")
    print("  n'a pas gagné son coût.")

    # ── sensibilité de la référence à ses propres hyperparamètres ─────────
    print("\n  Sensibilité de la RÉFÉRENCE (la vérité dépend-elle du prior ?)")
    print(f"  {'shrinkage':>10s} {'RMSE_oos':>9s} {'std(delta)':>11s} " +
          " ".join(f"{n.split()[0][:9]:>10s}" for n, _ in rows))
    best_sh, best_rmse = None, np.inf
    for sh in (0.0, 0.3, 0.6, 0.9):
        dt_f, rm = true_loo_contribution(
            T, pos, eval_stride=args.eval_stride, split=args.split,
            noise_var=args.noise_var, L_px=L_px, shrink=sh, verbose=False,
            ocean=obs.ocean)
        dt = np.full(n, np.nan); dt[fixed] = dt_f
        if rm < best_rmse:
            best_sh, best_rmse = sh, rm
        line = f"  {sh:>10.1f} {rm:>9.4f} {dt_f.std():>11.5f} "
        for _, sc in rows:
            r, _ = spearman(sc, dt)
            line += f" {r:>+10.3f}"
        print(line)
    print(f"\n  RMSE hors échantillon minimale à shrinkage = {best_sh} "
          f"({best_rmse:.4f})")
    print("  -> c'est CETTE ligne qui fait foi : elle reconstruit le mieux le")
    print("     champ sur la moitié non utilisée pour estimer la covariance.")
    print("  Si le classement de la baseline change de signe entre les lignes,")
    print("  sa 'victoire' venait du noyau géométrique, pas des données.")

    out = Path(args.output_dir) / "validation_obsonly.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"rmse_full_oi": full, "n_fixed": len(fixed), "results": res,
         "delta_true": {str(fixed[i]): float(d_true_f[i])
                        for i in range(len(fixed))}}, indent=1))
    print(f"\n  → {out}")


def parse_args():
    p = argparse.ArgumentParser("validation estimateur obs-only")
    p.add_argument("--obs", default="outputs/obs_synth.npz")
    p.add_argument("--truth", default="outputs/_truth.npz")
    p.add_argument("--lobo_ae", default="outputs/lobo_ae.json")
    p.add_argument("--lobo_gnn", default="outputs/lobo_gnn.json")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--eval_stride", type=int, default=4)
    p.add_argument("--split", type=float, default=0.5)
    # 0.01 était ~25x le bruit réel en unités normalisées (0.0192^2 = 3.7e-4)
    # -> l'OI sur-lissait et écrasait les écarts entre capteurs
    p.add_argument("--noise_var", type=float, default=4e-4)
    p.add_argument("--shrinkage", type=float, default=0.3,
                   help="mélange covariance empirique / noyau paramétrique. "
                        "0.9 (valeur config) rend la référence quasi géométrique")
    p.add_argument("--influence_px", type=float, default=-1,
                   help="<=0 : mesurer L sur le champ au lieu de config.py")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
