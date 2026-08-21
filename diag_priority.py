"""
DIAGNOSTIC  --  la priorite a-t-elle seulement une prise sur le plan ?

Trois sources de pertinence donnant des resultats IDENTIQUES au dernier
chiffre ne signifient pas "aucun gain" : cela signifie que la priorite
n'influence rien. Avant de comparer des sources, il faut verifier que le
levier existe.

La priorite n'intervient qu'a un seul endroit du planificateur : le retrait
iteratif des bouees quand une campagne ne rentre pas dans le budget. Si le
budget couvre confortablement toutes les campagnes, ou s'il n'en couvre
aucune, ce retrait ne s'active jamais et TOUTE priorite donne le meme plan.

Ce script tire des priorites aleatoires et mesure combien de bouees changent
de niveau de service. Zero = levier inexistant, la comparaison des sources
n'a aucun sens a ce budget. Il ne demande aucun entrainement et tourne en
quelques secondes.

    NAIADE_DOMAIN=large python diag_priority.py --maintenance pirata \\
        --budgets 1530 2400 4360 --sizes 11 16 25 30
"""

from __future__ import annotations

import argparse, importlib.util
from pathlib import Path

import numpy as np

from config import DOMAIN, NX, NY, DX_KM, PORT_XY_FRAC
from data.dataset import SyntheticOceanGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nt", type=int, default=180)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--n_max", type=int, default=30)
    p.add_argument("--maintenance", type=str, default="pirata",
                   choices=["regional", "pirata"])
    p.add_argument("--budgets", type=float, nargs="+",
                   default=[1530, 2400, 4360])
    p.add_argument("--sizes", type=int, nargs="+", default=[11, 16, 25, 30])
    p.add_argument("--draws", type=int, default=6)
    a = p.parse_args()

    spec = importlib.util.spec_from_file_location(
        "brick3", Path(__file__).with_name("03_rl.py"))
    b3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b3)

    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=a.nt, seed=a.seed)
    port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM
    mm = b3.MaintenanceModel(b3.get_params(a.maintenance), port)
    env = b3.OceanNetworkEnv(T, S, grid_x=a.grid_x, grid_y=a.grid_y,
                             n_min=8, n_max=a.n_max, maintenance=mm)

    print(f"\n  domaine {DOMAIN} {NX}x{NY} @ {DX_KM:.0f} km | profil "
          f"{a.maintenance} | autonomie {mm.p.endurance_days:.0f} j")
    print(f"\n  {'budget':>7} {'N':>3} | {'repartition des visites':>26} | "
          f"{'hors portee':>11} | {'prise de la priorite':>20}")
    print("  " + "-" * 78)
    rng = np.random.default_rng(0)
    inert = 0
    total = 0
    for B in a.budgets:
        for n in a.sizes:
            idx = env.sample_feasible(n, rng=np.random.default_rng(1))
            if len(idx) < n:
                continue
            pts = env.positions_km(idx)
            oor = int(mm.unreachable(pts).sum())
            V = np.array([mm.plan(pts, B, rng.uniform(0.05, 1.0, len(idx)),
                                  refine=False).visits
                          for _ in range(a.draws)])
            varie = int((V.max(0) != V.min(0)).sum())
            rep = [int(v) for v in np.bincount(
                V[0], minlength=mm.p.max_visits_per_year + 1)]
            total += 1
            inert += (varie == 0)
            flag = "  INERTE" if varie == 0 else ""
            print(f"  {B:>7.0f} {len(idx):>3} | "
                  f"{str(rep):>26} | {oor:>11} | "
                  f"{varie:>3}/{len(idx)} bouees{flag}")

    print(f"\n  {inert}/{total} configurations ou la priorite n'a AUCUNE prise.")
    if inert == total:
        print("  -> A ces budgets, comparer des sources de pertinence est vide\n"
              "     de sens : le plan est le meme quelle que soit la priorite.\n"
              "     Chercher les budgets ou le retrait s'active (typiquement\n"
              "     entre 'une campagne complete' et 'deux campagnes').")
    elif inert:
        print("  -> Ne comparer les sources que sur les lignes non inertes.")


if __name__ == "__main__":
    main()
