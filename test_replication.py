"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  REPLICATION DU RESULTAT PRINCIPAL SUR PLUSIEURS GRAINES D'OCEAN             ║
║                                                                              ║
║  Question posee, une seule :                                                 ║
║                                                                              ║
║      A budget de maintien donne, concevoir le reseau EN TENANT COMPTE du     ║
║      maintien apporte-t-il plus d'information que le concevoir sans ?        ║
║                                                                              ║
║  Les deux bras sont exposes au MEME ocean, au MEME budget, avec le MEME      ║
║  glouton. Seul l'objectif differe :                                          ║
║                                                                              ║
║      integre    : maximise l'information EFFECTIVE (disponibilite comprise)  ║
║      naif       : maximise l'information NOMINALE (bouees supposees toujours ║
║                   disponibles), puis on evalue ce qu'il delivre reellement   ║
║      entretenable : idem naif, MAIS interdit de deployer une bouee que le    ║
║                   budget ne permet pas de visiter au moins une fois par an   ║
║                                                                              ║
║  Le troisieme bras est le temoin qui compte. Le bras naif degenere aux       ║
║  budgets bas : une bouee non entretenue ne coutant que son amortissement,    ║
║  il en achete le maximum et n'a plus de quoi armer un navire (26 mouillages, ║
║  disponibilite 0,29, zero campagne). Le battre ne prouve rien. Le bras       ║
║  entretenable represente un operateur qui ignore la notion d'information     ║
║  effective mais refuse de deployer ce qu'il ne peut pas maintenir.           ║
║                                                                              ║
║  Le temoin retient sa configuration au sens de SON objectif : il n'a jamais  ║
║  acces a la disponibilite, qu'il est cense ignorer.                          ║
║                                                                              ║
║  Comparaison APPARIEE : les deux bras partagent la graine d'ocean, donc      ║
║  l'ecart par graine elimine la variabilite entre oceans. C'est ce qui rend   ║
║  quelques graines exploitables la ou des mesures independantes en            ║
║  demanderaient beaucoup plus.                                                ║
║                                                                              ║
║  Le chiffre publie jusqu'ici (+32,9 % sur `large`) repose sur UNE graine.    ║
║  Ce script existe pour savoir si c'est un resultat ou une anecdote.          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Exemples
--------
    NAIADE_DOMAIN=large python test_replication.py --seeds 42 43 44 45 46 \\
        --maintenance pirata --n_max 30

    NAIADE_DOMAIN=demo python test_replication.py --seeds 1 2 3 4 5 \\
        --maintenance regional --n_max 26
"""

from __future__ import annotations

import argparse, importlib.util, json, time
from pathlib import Path

import numpy as np

from config import (DOMAIN, NX, NY, DX_KM, PORT_XY_FRAC, EVF_SHRINKAGE,
                    NT)
from data.dataset import SyntheticOceanGenerator


def load_brick3():
    spec = importlib.util.spec_from_file_location(
        "brick3", Path(__file__).with_name("03_rl.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_seed(b3, args, seed):
    from campaign import greedy_under_budget, auto_budget_levels

    t0 = time.time()
    print(f"\n{'='*78}\n  GRAINE {seed}   (domaine {DOMAIN}, {NX}x{NY} @ "
          f"{DX_KM:.0f} km, profil {args.maintenance})\n{'='*78}", flush=True)

    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=args.nt, seed=seed)
    port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM
    maint = b3.MaintenanceModel(b3.get_params(args.maintenance), port)
    env = b3.OceanNetworkEnv(
        T, S, grid_x=args.grid_x, grid_y=args.grid_y,
        n_min=args.n_min, n_max=args.n_max,
        fit_influence=True, evf_cv=True, shrinkage=args.evf_shrink,
        maintenance=maint)

    budgets = args.budgets
    if not budgets:
        budgets, b_viable = auto_budget_levels(
            env, n_ref=args.n_max, fractions=tuple(args.fractions))
        print(f"  Budget minimum viable (N={args.n_max}) : {b_viable:.0f} k€/an")
    print(f"  Niveaux : " + ", ".join(f"{b:.0f}" for b in budgets) + " k€/an")
    print(f"  Rayon d influence ajuste : {env.influence_px * DX_KM:.0f} km\n",
          flush=True)

    print(f"  {'budget':>7} | {'integre':>21} | {'naif':>21} | "
          f"{'entretenable':>21} | {'vs naif':>7} {'vs ent.':>7}")
    print("  " + "-" * 96)
    rows = []
    def pack(r):
        return {"info": float(r["ev"]["info"]), "n": int(len(r["idx"])),
                "info_nominal": float(r["ev"]["info_nominal"]),
                "availability": float(np.mean(r["ev"]["availability"]))
                if len(r["idx"]) else 0.0,
                "cost": float(r["ev"]["cost_keur"])}

    for B in budgets:
        a = greedy_under_budget(env, float(B), "effective", verbose=False)
        n = greedy_under_budget(env, float(B), "nominal", verbose=False)
        s = greedy_under_budget(env, float(B), "nominal", verbose=False,
                                require_serviceable=True)
        gi, gn, gs = a["ev"]["info"], n["ev"]["info"], s["ev"]["info"]
        rel_n = (gi - gn) / max(gn, 1e-9) * 100
        rel_s = (gi - gs) / max(gs, 1e-9) * 100
        rows.append({
            "budget": float(B),
            "integrated": pack(a), "control": pack(n), "serviceable": pack(s),
            "gain_abs": float(gi - gn), "gain_rel_pct": float(rel_n),
            "gain_rel_pct_serviceable": float(rel_s)})
        print(f"  {B:>7.0f} | {gi:.4f} N={len(a['idx']):<2} "
              f"d={np.mean(a['ev']['availability']):.2f} | "
              f"{gn:.4f} N={len(n['idx']):<2} "
              f"d={np.mean(n['ev']['availability']):.2f} | "
              f"{gs:.4f} N={len(s['idx']):<2} "
              f"d={np.mean(s['ev']['availability']) if len(s['idx']) else 0:.2f} | "
              f"{rel_n:>+7.1f}% {rel_s:>+7.1f}%", flush=True)

    print(f"\n  graine {seed} terminee en {time.time()-t0:.0f}s", flush=True)
    return {"seed": seed, "domain": DOMAIN, "profile": args.maintenance,
            "influence_km": float(env.influence_px * DX_KM),
            "budgets": rows}


def summarise(results, args):
    print(f"\n{'='*78}\n  SYNTHESE APPARIEE SUR {len(results)} GRAINE(S)"
          f"\n{'='*78}")
    n_lvl = min(len(r["budgets"]) for r in results)

    for key, lab in [("gain_rel_pct", "contre le temoin NAIF"),
                     ("gain_rel_pct_serviceable",
                      "contre le temoin ENTRETENABLE (celui qui compte)")]:
        print(f"\n  Gain {lab}")
        print(f"  {'budget':>7} | {'moyen':>12} | {'ecart-type':>10} | "
              f"{'min':>7} | {'max':>7} | {'gagnantes':>9}")
        print("  " + "-" * 68)
        for i in range(n_lvl):
            B = results[0]["budgets"][i]["budget"]
            g = np.array([r["budgets"][i][key] for r in results])
            verdict = ""
            if len(g) > 1 and abs(g.mean()) < g.std():
                verdict = "  <- non concluant"
            print(f"  {B:>7.0f} | {g.mean():>+11.1f}% | {g.std():>9.1f}% | "
                  f"{g.min():>+6.1f}% | {g.max():>+6.1f}% | "
                  f"{int((g > 0).sum()):>4}/{len(g)}{verdict}")

    print(f"\n  Mecanisme (moyennes sur les graines) :")
    print(f"  {'budget':>7} | {'N int.':>7} {'d int.':>7} | "
          f"{'N naif':>7} {'d naif':>7} | {'N ent.':>7} {'d ent.':>7}")
    print("  " + "-" * 66)
    for i in range(n_lvl):
        B = results[0]["budgets"][i]["budget"]
        m = lambda arm, k: np.mean([r["budgets"][i][arm][k] for r in results])
        print(f"  {B:>7.0f} | {m('integrated','n'):>7.1f} "
              f"{m('integrated','availability'):>7.2f} | "
              f"{m('control','n'):>7.1f} {m('control','availability'):>7.2f} | "
              f"{m('serviceable','n'):>7.1f} "
              f"{m('serviceable','availability'):>7.2f}")

    print(f"\n  Lecture. Un ecart moyen inferieur a son ecart-type n'est pas un\n"
          f"  resultat. Un ecart de quelques pour cent reste sous la myopie du\n"
          f"  glouton et ne doit pas etre cite. Seuls les niveaux ou l'ecart\n"
          f"  domine nettement sa dispersion ET ou les graines sont unanimes\n"
          f"  peuvent etre presentes.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--nt", type=int, default=NT,
                   help="Defaut = NT de config.py, pour rester coherent "
                        "avec 03_rl.py (une valeur codee en dur ici avait "
                        "produit deux series de chiffres incomparables)")
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--n_min", type=int, default=10)
    p.add_argument("--n_max", type=int, default=30)
    p.add_argument("--evf_shrink", type=float, default=EVF_SHRINKAGE)
    p.add_argument("--maintenance", type=str, default="pirata",
                   choices=["regional", "pirata"])
    p.add_argument("--budgets", type=float, nargs="+", default=None)
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.25, 0.35, 0.55, 1.0],
                   help="Fractions du budget minimum viable balayees. Les "
                        "fractions basses sont le regime ou la contrainte "
                        "mord ; 1.0 sert de temoin sature.")
    p.add_argument("--out", type=str, default="outputs/replication.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    b3 = load_brick3()
    results = [run_seed(b3, args, s) for s in args.seeds]
    summarise(results, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Resultats -> {out}")
