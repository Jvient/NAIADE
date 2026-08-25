"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RÉPLICATION SUR LES SPLITS TEMPORELS                                        ║
║                                                                              ║
║  Applique les modèles ENTRAÎNÉS SUR train aux observations de val et test,   ║
║  sans réentraînement. Trois estimations du même écart, sur des périodes      ║
║  disjointes.                                                                 ║
║                                                                              ║
║  Ce que ça teste, et que le split par capteur ne teste pas :                 ║
║    · transférabilité TEMPORELLE — l'estimateur tient-il sur des années       ║
║      qu'il n'a jamais vues, avec d'autres régimes ENSO/Atlantic Niño ?       ║
║    · robustesse du signe — un p à 0.09 répliqué trois fois vaut mieux        ║
║      qu'un p à 0.09 isolé (combinaison de Fisher)                            ║
║                                                                              ║
║  Le réseau de mouillages est IDENTIQUE d'un split à l'autre (positions       ║
║  PIRATA réelles, même graine), donc les scores sont comparables.             ║
╚══════════════════════════════════════════════════════════════════════════════╝

    python replicate_splits.py --splits train,val,test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from validate_obsonly import spearman as _spearman


def run(cmd, quiet=True):
    print(f"    $ {' '.join(cmd[:6])} ...")
    r = subprocess.run(cmd, capture_output=quiet, text=True)
    if r.returncode != 0:
        out = (r.stdout or "") + (r.stderr or "")
        print(out[-1500:])
        raise RuntimeError(f"échec : {' '.join(cmd)}")
    return r.stdout or ""


def do_split(split, a):
    d = Path(a.output_dir) / f"split_{split}"
    d.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    print(f"\n  [{split}] génération des observations")
    run([py, "00_make_obs.py", "--data", "glorys", "--split", split,
         "--glorys_cache", a.glorys_cache,
         "--n_argo", str(a.n_argo), "--n_drifters", str(a.n_drifters),
         "--seed_buoys", str(a.seed_buoys), "--output_dir", str(d)])

    obs = str(d / "obs_synth.npz")
    print(f"  [{split}] scores AE (modèle entraîné sur train)")
    run([py, "obsonly.py", "--lobo", "--obs", obs, "--ckpt", a.ae_ckpt,
         "--output_dir", str(d), "--lobo_t", str(a.lobo_t)])

    print(f"  [{split}] scores GNN (graphe reconstruit)")
    try:
        run([py, "gnn_lobo.py", "--lobo", "--obs", obs, "--ckpt", a.gnn_ckpt,
             "--output_dir", str(d), "--rebuild_graph",
             "--lobo_t", str(a.lobo_t)])
    except RuntimeError as e:
        print(f"    [!] GNN indisponible sur ce split : {e}")

    print(f"  [{split}] validation")
    run([py, "validate_obsonly.py", "--obs", obs,
         "--truth", str(d / "_truth.npz"),
         "--lobo_ae", str(d / "lobo_ae.json"),
         "--lobo_gnn", str(d / "lobo_gnn.json"),
         "--output_dir", str(d)])

    return json.loads((d / "validation_obsonly.json").read_text())


def fisher(pvals):
    """Combinaison de Fisher : -2 sum log p ~ chi2(2k) sous H0."""
    ps = [p for p in pvals if p is not None and np.isfinite(p) and p > 0]
    if len(ps) < 2:
        return np.nan
    stat = -2 * np.sum(np.log(ps))
    k = len(ps)
    # survie du chi2 à 2k ddl, sans scipy
    from math import exp, lgamma, log
    x, df = stat / 2.0, k
    term, tot = 1.0, 1.0
    for i in range(1, df):
        term *= x / i
        tot += term
    return float(min(1.0, exp(-x) * tot))


def main(a):
    print("=" * 70)
    print("  Réplication sur les splits temporels")
    print("=" * 70)
    print("\n  Modèles entraînés sur TRAIN, appliqués tels quels aux autres")
    print("  splits. Aucun réentraînement, aucune recalibration.")

    res = {}
    for sp in a.splits.split(","):
        try:
            res[sp] = do_split(sp.strip(), a)
        except Exception as e:
            print(f"  [!] split {sp} abandonné : {e}")

    if not res:
        sys.exit("aucun split exploitable")

    # ── table de synthèse ────────────────────────────────────────────────
    names = sorted({k for r in res.values() for k in r["results"]})
    print("\n" + "=" * 70)
    print("  SYNTHÈSE — Spearman contre la contribution vraie")
    print("=" * 70)
    print(f"\n  {'estimateur':<24s} " +
          " ".join(f"{s:>14s}" for s in res) + f" {'Fisher p':>10s}")
    for nm in names:
        line, ps = f"  {nm:<24s} ", []
        for sp, r in res.items():
            e = r["results"].get(nm)
            if e is None:
                line += f"{'—':>15s}"
                continue
            line += f"{e['spearman']:>+9.3f}(n{e['n']:>2d})"
            ps.append(e.get("p"))
        line += f" {fisher(ps):>10.4f}"
        print(line)

    print(f"\n  {'écart à la baseline':<24s} " +
          " ".join(f"{s:>14s}" for s in res) + f" {'Fisher p':>10s}")
    for nm in names:
        if nm.startswith("baseline"):
            continue
        line, ps, ds = f"  {nm:<24s} ", [], []
        for sp, r in res.items():
            e = (r["results"].get(nm) or {}).get("vs_baseline")
            if e is None:
                line += f"{'—':>15s}"
                continue
            line += f"{e['delta']:>+15.3f}"
            ps.append(e.get("p")); ds.append(e["delta"])
        fp = fisher(ps)
        line += f" {fp:>10.4f}"
        print(line + ("  *" if fp == fp and fp < 0.05 else ""))
        if ds:
            print(f"  {'':24s}   écart moyen {np.mean(ds):+.3f}, "
                  f"signe constant : {all(np.sign(d) == np.sign(ds[0]) for d in ds)}")

    # ── LA question préalable : la référence est-elle reproductible ? ─────
    print("\n" + "=" * 70)
    print("  PLANCHER DE BRUIT — la 'vérité' se corrèle-t-elle à elle-même ?")
    print("=" * 70)
    print("\n  delta_true est estimé indépendamment sur chaque période. Si les")
    print("  trois estimations ne concordent pas entre elles, aucune méthode")
    print("  ne peut les prédire : on mesurerait le bruit de la référence.")
    print("  C'est le PLAFOND de tout Spearman du tableau précédent.")

    truths = {}
    for sp, r in res.items():
        dt = r.get("delta_true", {})
        truths[sp] = {int(k): v for k, v in dt.items()}
    keys = sorted(set.intersection(*[set(t) for t in truths.values()])) \
        if len(truths) > 1 else []
    sps = list(truths)
    if len(sps) > 1 and len(keys) >= 5:
        print(f"\n  {'':>10s} " + " ".join(f"{s:>10s}" for s in sps))
        mat = {}
        for a_ in sps:
            line = f"  {a_:>10s} "
            for b_ in sps:
                va = np.array([truths[a_][k] for k in keys])
                vb = np.array([truths[b_][k] for k in keys])
                r_, _ = _spearman(va, vb)
                mat[(a_, b_)] = r_
                line += f"{r_:>+10.3f}" if a_ != b_ else f"{'—':>10s}"
            print(line)
        off = [v for (a_, b_), v in mat.items() if a_ != b_]
        med = float(np.median(off))
        print(f"\n  concordance médiane entre périodes : {med:+.3f}  "
              f"(n = {len(keys)} mouillages)")
        if med < 0.3:
            print("\n  [!] La référence ne se reproduit pas d'une période à")
            print("      l'autre. Le protocole par CAPTEUR a atteint son")
            print("      plancher de bruit : la contribution marginale d'une")
            print("      bouée sur 17, dans une boîte de 100 degrés, vaut")
            print("      quelques dixièmes de pour cent de la RMSE — sous")
            print("      l'erreur d'estimation de l'interpolation optimale.")
            print("      Aucun réglage d'estimateur ne franchira ce plafond.")
            print("      Passer au scoring par SOUS-ENSEMBLES (retirer 5")
            print("      bouées à la fois) : deltas d'un ordre de grandeur")
            print("      plus grands, et des centaines de points au lieu de 17.")
        elif med < 0.6:
            print("\n  [!] Reproductibilité partielle : le plafond atteignable")
            print(f"      est de l'ordre de {med:.2f}, pas de 1. Rapporter les")
            print("      Spearman RELATIVEMENT à ce plafond.")
        else:
            print("\n  Référence stable : les Spearman du tableau précédent")
            print("  sont interprétables tels quels.")

    print("\n  Lecture. Fisher combine des tests INDÉPENDANTS : les splits")
    print("  temporels le sont (années disjointes), les lignes de shrinkage")
    print("  ne l'étaient pas. Un signe constant sur les trois périodes est")
    print("  l'argument le plus fort disponible à cette taille d'échantillon.")
    print("\n  Attention : val et test sont plus COURTS que train. Moins de")
    print("  pas de temps => scores plus bruités, IC plus larges. Un Spearman")
    print("  plus faible n'y signifie pas forcément une moindre transférabilité.")

    out = Path(a.output_dir) / "replication_splits.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"\n  → {out}")


def parse_args():
    p = argparse.ArgumentParser("réplication sur splits")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--glorys_cache", default="data/glorys_cache")
    p.add_argument("--ae_ckpt", default="outputs/ae_obsonly.pt")
    p.add_argument("--gnn_ckpt", default="outputs/gnn_lobo.pt")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--n_argo", type=int, default=20)
    p.add_argument("--n_drifters", type=int, default=15)
    p.add_argument("--seed_buoys", type=int, default=7)
    p.add_argument("--lobo_t", type=int, default=120)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
