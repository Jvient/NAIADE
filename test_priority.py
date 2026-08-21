"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TEST DU COUPLAGE A  --  le GNN fournit la pertinence au planificateur       ║
║                                                                              ║
║  Script de VALIDATION, pas encore un livrable. Il répond à deux questions    ║
║  distinctes, qu'il ne faut pas confondre :                                   ║
║                                                                              ║
║   1. PREDICTION  -- le GNN classe-t-il les bouées comme le fait la           ║
║      contribution marginale exacte ? (rho de Spearman, hors échantillon)     ║
║      Comparé aux deux proxys existants.                                      ║
║                                                                              ║
║   2. DECISION    -- brancher le GNN sur le planificateur produit-il un       ║
║      meilleur réseau à budget donné ? (EVF effective, validée hors           ║
║      échantillon)                                                            ║
║                                                                              ║
║  Un bon score de prédiction sans gain de décision ne vaudrait rien : c'est   ║
║  la seconde question qui tranche.                                            ║
║                                                                              ║
║  Tout est répété sur plusieurs graines d'océan, parce qu'aucun des écarts    ║
║  mesurés jusqu'ici n'avait de barre d'erreur.                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Exemples
--------
    NAIADE_DOMAIN=large python test_priority.py --seeds 42 43 44
    NAIADE_DOMAIN=demo  python test_priority.py --seeds 1 2 3 --n_graphs 200
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

from config import (DOMAIN, NX, NY, DX_KM, PORT_XY_FRAC, DEVICE,
                    EVF_SHRINKAGE, NT)
from data.dataset import SyntheticOceanGenerator
import priority as P


def load_brick3():
    spec = importlib.util.spec_from_file_location(
        "brick3", Path(__file__).with_name("03_rl.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def brick2_proxy(env, idx):
    """Cible d'entraînement actuelle de la brique 2 : 1 - correlation moyenne.
    Reproduite ici telle quelle pour la comparer aux autres."""
    R = P.candidate_correlation(env)[np.ix_(idx, idx)]
    Rz = np.abs(R.copy())
    np.fill_diagonal(Rz, 0.0)
    n = len(idx)
    return 1.0 - Rz.sum(axis=1) / max(n - 1, 1)


def run_seed(b3, args, seed, verbose=True):
    t0 = time.time()
    print(f"\n{'='*72}\n  GRAINE {seed}   (domaine {DOMAIN}, "
          f"{NX}x{NY} @ {DX_KM:.0f} km)\n{'='*72}", flush=True)

    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=args.nt, seed=seed)

    port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM
    maint = b3.MaintenanceModel(b3.get_params(args.maintenance), port)
    env = b3.OceanNetworkEnv(
        T, S, grid_x=args.grid_x, grid_y=args.grid_y,
        n_min=args.n_min, n_max=args.n_max,
        fit_influence=True, evf_cv=True, shrinkage=args.evf_shrink,
        maintenance=maint)

    # ── 1. prediction ────────────────────────────────────────────────────────
    print("  [1/2] Jeu d'entrainement (contribution marginale exacte)...",
          flush=True)
    data = P.build_dataset(env, n_graphs=args.n_graphs,
                           n_range=(args.n_min, args.n_max + 1),
                           seed=seed, verbose=verbose)
    model = P.PriorityGNN().to(DEVICE)
    rho_gnn = P.train(model, data, epochs=args.epochs, verbose=verbose)

    n_val = max(1, int(len(data) * 0.2))
    val = data[-n_val:]
    rho_var = float(np.mean([P.spearman(env._maint_priority[idx], y.numpy())
                             for *_, y, idx in val]))
    rho_b2 = float(np.mean([P.spearman(brick2_proxy(env, idx), y.numpy())
                            for *_, y, idx in val]))
    print(f"\n  Spearman hors echantillon (vs contribution marginale exacte)")
    print(f"    GNN entraine sur la vraie cible      {rho_gnn:+.3f}")
    print(f"    proxy variabilite locale (actuel)    {rho_var:+.3f}")
    print(f"    cible actuelle de la brique 2        {rho_b2:+.3f}", flush=True)

    # ── 2. decision ──────────────────────────────────────────────────────────
    from campaign import greedy_under_budget, auto_budget_levels
    budgets = args.budgets
    if not budgets:
        budgets, b_viable = auto_budget_levels(env, n_ref=args.n_max)
        budgets = budgets[:3]
        print(f"\n  Budget minimum viable : {b_viable:.0f} k€/an")
    print(f"  [2/2] Reseaux sous budget : "
          + ", ".join(f"{b:.0f}" for b in budgets) + " k€/an", flush=True)

    gnn = P.GNNPriority(model, env)
    sources = [("proxy statique", None),
               ("GNN", gnn),
               ("LOO exact", lambda e, i: P.loo_contribution(e, i) + 1e-6)]
    table = []
    print(f"\n  {'budget':>8} | " + " | ".join(f"{n:>16}" for n, _ in sources))
    for B in budgets:
        row = {"budget": float(B)}
        cells = []
        for name, fn in sources:
            env.priority_fn = fn
            env._eval_cache.clear()
            t = time.time()
            r = greedy_under_budget(env, float(B), "effective", verbose=False)
            row[name] = {"info": float(r["ev"]["info"]),
                         "n": int(len(r["idx"])),
                         "availability": float(np.mean(r["ev"]["availability"])),
                         "seconds": round(time.time() - t, 1)}
            cells.append(f"{r['ev']['info']:.4f} (N={len(r['idx'])})")
        print(f"  {B:>8.0f} | " + " | ".join(f"{c:>16}" for c in cells),
              flush=True)
        table.append(row)
    env.priority_fn = None

    print(f"\n  graine {seed} terminee en {time.time()-t0:.0f}s", flush=True)
    return {"seed": seed, "domain": DOMAIN,
            "influence_km": float(env.influence_px * DX_KM),
            "spearman": {"gnn": rho_gnn, "proxy_variance": rho_var,
                         "brick2_target": rho_b2},
            "budgets": table}


def summarise(results):
    print(f"\n{'='*72}\n  SYNTHESE SUR {len(results)} GRAINE(S)\n{'='*72}")
    for k, lab in [("gnn", "GNN"), ("proxy_variance", "proxy variabilite"),
                   ("brick2_target", "cible brique 2")]:
        v = np.array([r["spearman"][k] for r in results])
        print(f"  Spearman {lab:<20} {v.mean():+.3f} +/- {v.std():.3f}")

    print(f"\n  Gain de DECISION du GNN sur le proxy statique "
          f"(EVF effective) :")
    print(f"  {'budget rang':>12} | {'moyenne':>9} | {'ecart-type':>10} | "
          f"{'graines gagnantes':>18}")
    n_lvl = min(len(r["budgets"]) for r in results)
    for i in range(n_lvl):
        g = np.array([(r["budgets"][i]["GNN"]["info"]
                       - r["budgets"][i]["proxy statique"]["info"])
                      / max(r["budgets"][i]["proxy statique"]["info"], 1e-9)
                      * 100 for r in results])
        print(f"  {i+1:>12} | {g.mean():>+8.1f}% | {g.std():>9.1f}% | "
              f"{int((g > 0).sum())}/{len(g)}")
    print("\n  Un gain moyen inferieur a son ecart-type n'est pas un resultat.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--nt", type=int, default=NT)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--n_min", type=int, default=10)
    p.add_argument("--n_max", type=int, default=30)
    p.add_argument("--n_graphs", type=int, default=300)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--evf_shrink", type=float, default=EVF_SHRINKAGE)
    p.add_argument("--maintenance", type=str, default="pirata",
                   choices=["regional", "pirata"])
    p.add_argument("--budgets", type=float, nargs="+", default=None,
                   help="k€/an. Par defaut : trois niveaux auto-calibres.")
    p.add_argument("--out", type=str, default="outputs/priority_test.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    b3 = load_brick3()
    results = [run_seed(b3, args, s) for s in args.seeds]
    summarise(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Resultats -> {out}")
