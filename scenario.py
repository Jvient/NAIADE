"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SCENARIOS DE MAINTENANCE PLURIANNUELS                                       ║
║                                                                              ║
║  Jusqu'ici, la disponibilite etait moyennee : une bouee disponible a 61 %    ║
║  etait traitee comme une bouee en permanence plus bruitee, via               ║
║                                                                              ║
║      R_eff = R/a + C (1-a)/a                                                 ║
║                                                                              ║
║  C'est exact pour un gain constant, mais cela dit qu'une panne coute un      ║
║  montant FIXE. Or une bouee qui meurt en fevrier et n'est relevee qu'en      ║
║  novembre ne coute pas la meme chose selon ce qui se passe dans l'ocean      ║
║  pendant ces neuf mois.                                                      ║
║                                                                              ║
║  Ce module simule le calendrier reel :                                       ║
║                                                                              ║
║    - chaque bouee tombe en panne a un instant exponentiel (MTBF) ;           ║
║    - elle reste morte jusqu'a ce que le NAVIRE l'atteigne, a une date        ║
║      deduite de la trajectoire de campagne planifiee (transit + temps sur    ║
║      station, jambe apres jambe) ;                                           ║
║    - le critere d'information est evalue A CHAQUE PAS DE TEMPS avec le       ║
║      masque reellement disponible, sans moyennage.                           ║
║                                                                              ║
║  Deux resultats en sortent :                                                 ║
║                                                                              ║
║    1. la courbe en dents de scie -- l'erreur derive pendant que le reseau    ║
║       s'eteint, et chute a chaque campagne. A budget serre, la derive        ║
║       l'emporte et le reseau se degrade d'annee en annee ;                   ║
║    2. une VALIDATION de l'approximation R_eff, en comparant sa prediction    ║
║       a la moyenne des scenarios.                                            ║
║                                                                              ║
║  Aucun apprentissage, aucune assimilation : c'est l'etage 1, celui qui rend  ║
║  les deux suivants evaluables.                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

    NAIADE_DOMAIN=large python scenario.py --maintenance pirata --n_max 30
"""

from __future__ import annotations

import argparse, importlib.util, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (DOMAIN, NX, NY, DX_KM, PORT_XY_FRAC, NT, EVF_SHRINKAGE)
from data.dataset import SyntheticOceanGenerator
from kalman import make_evaluator

BG, PANEL, EDGE = "#0a1628", "#050d1a", "#2a4a7a"
COLORS = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]


# ══════════════════════════════════════════════════════════════════════════════
#  CALENDRIER DES VISITES
# ══════════════════════════════════════════════════════════════════════════════

def visit_calendar(plan, params, first_departure=45.0):
    """
    Date (en jour de l'annee) a laquelle le navire atteint chaque bouee.

    Les dates ne sont pas arbitraires : elles decoulent de la trajectoire
    planifiee. Le navire part du port, enchaine les stations dans l'ordre de
    la tournee, revient, repart pour la jambe suivante. Une bouee en fin de
    tournee est donc reparee plusieurs semaines apres la premiere -- ce qui
    est precisement le genre de detail que le moyennage efface.

    Retourne {position_dans_le_reseau: [jours de visite]}.

    ATTENTION : `Leg.buoys` contient des indices POSITIONNELS (rang dans le
    reseau passe au planificateur), pas les indices globaux de positions
    candidates. Les confondre donne un calendrier vide et des bouees jamais
    reparees -- panne silencieuse, sans exception ni message.
    """
    cal: dict[int, list[float]] = {}
    V = max(len(plan.campaigns), 1)
    for camp in plan.campaigns:
        # les campagnes sont reparties dans l'annee
        day = first_departure + 365.0 * (camp.index - 1) / V
        for leg in camp.legs:
            wp = leg.waypoints
            seg = np.linalg.norm(wp[1:] - wp[:-1], axis=1) / params.ship_km_per_day
            t = day
            for k, b in enumerate(leg.buoys):
                t += seg[k] + params.on_station_days
                cal.setdefault(int(b), []).append(t)
            t += seg[-1]                       # retour au port
            day = t
    return cal


def simulate_uptime(n_buoys, calendar, mtbf_days, years, rng,
                    days_per_year=365):
    """
    Etat marche/panne de chaque bouee, jour par jour, sur plusieurs annees.

    Panne exponentielle depuis la derniere reparation ; reparation seulement
    au passage du navire. Une bouee non inscrite au calendrier n'est jamais
    reparee : elle s'eteint et le reste.
    """
    n_days = int(years * days_per_year)
    up = np.ones((n_days, int(n_buoys)), dtype=bool)
    for j in range(int(n_buoys)):
        visits = sorted([d + y * days_per_year
                         for y in range(int(years))
                         for d in calendar.get(j, [])])
        t_repair = 0.0
        for v in visits + [float(n_days)]:
            tau = rng.exponential(mtbf_days)
            t_fail = t_repair + tau
            if t_fail < v:
                a, z = int(np.ceil(t_fail)), int(min(np.ceil(v), n_days))
                if a < n_days:
                    up[a:z, j] = False
            t_repair = v
            if t_repair >= n_days:
                break
    return up


# ══════════════════════════════════════════════════════════════════════════════
#  SERIE TEMPORELLE D'INFORMATION
# ══════════════════════════════════════════════════════════════════════════════

def evf_timeseries(env, buoy_ids, up):
    """
    Variance expliquee a chaque pas de temps, avec le masque REEL.

    Le masque ne change qu'aux evenements (panne ou reparation), donc on ne
    recalcule le critere que pour les configurations distinctes -- quelques
    dizaines par an au lieu de 365.
    """
    buoy_ids = np.asarray(buoy_ids, dtype=int)
    out = np.empty(len(up))
    cache: dict[tuple, float] = {}
    for t in range(len(up)):
        key = tuple(np.flatnonzero(up[t]).tolist())
        v = cache.get(key)
        if v is None:
            v = (env.explained_variance(buoy_ids[list(key)]) if key else 0.0)
            cache[key] = v
        out[t] = v
    return out, len(cache)


def run_scenarios(env, idx, budget, years=5, n_scenarios=12, seed=0,
                  priority=None, evaluate=None):
    """Plusieurs trajectoires de pannes sur le meme reseau et le meme plan."""
    ev = env.evaluate(idx, budget_keur=budget, refine=True,
                      priority=priority, with_plan=True)
    plan = ev["plan"]
    cal = visit_calendar(plan, env.maint.p)
    rng = np.random.default_rng(seed)
    series, n_conf = [], 0
    for _ in range(n_scenarios):
        up = simulate_uptime(len(idx), cal, env.maint.p.mtbf_days,
                             years, rng)
        if evaluate is None:
            s, nc = evf_timeseries(env, idx, up)
            n_conf += nc
        else:
            s = evaluate(idx, up)
        series.append(s)
    S = np.array(series)
    return {
        "budget": float(budget), "idx": idx, "plan": plan,
        "series": S,
        "mean": float(S.mean()), "p10": float(np.percentile(S, 10)),
        "static_approx": float(ev["info"]),      # ce que predit R_eff
        "nominal": float(ev["info_nominal"]),
        "availability_static": float(np.mean(ev["availability"])),
        "availability_scenario": float(np.mean([
            simulate_uptime(len(idx), cal, env.maint.p.mtbf_days, years,
                            np.random.default_rng(1000 + k)).mean()
            for k in range(3)])),
        "cost": float(ev["cost_keur"]), "n": int(len(idx)),
        "n_configs": n_conf,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _frame(ax, title="", xlab="", ylab=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
    if title:
        ax.set_title(title, color="white", fontsize=11, fontweight="bold",
                     pad=8)
    ax.set_xlabel(xlab, color="white", fontsize=9)
    ax.set_ylabel(ylab, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)


def plot_sawtooth(results, out_path, years):
    fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.32,
                          wspace=0.26)

    ax = fig.add_subplot(gs[0, :])
    _frame(ax, "Information delivree au fil du temps — les pannes erodent le "
               "reseau, les campagnes le retablissent",
           "Annees", "Variance expliquee (EVF)")
    for c, r in zip(COLORS, results):
        S = r["series"]
        x = np.arange(S.shape[1]) / 365.0
        ax.fill_between(x, np.percentile(S, 10, axis=0),
                        np.percentile(S, 90, axis=0), color=c, alpha=0.18)
        ax.plot(x, S.mean(0), color=c, lw=1.7,
                label=f"{r['budget']:.0f} k€/an  (N={r['n']}, "
                      f"moy. {r['mean']:.3f})")
        ax.axhline(r["static_approx"], color=c, ls=":", lw=1.1, alpha=0.8)
    for y in range(1, int(years)):
        ax.axvline(y, color="white", lw=0.6, alpha=0.22)
    ax.legend(fontsize=8.5, labelcolor="white", facecolor=BG, edgecolor=EDGE,
              loc="upper right", ncol=2)
    ax.grid(alpha=0.15, color="white")
    ax.text(0.008, 0.04, "traits pointilles : prediction de l'approximation "
                         "moyennee R_eff\nbandes : enveloppe 10-90 % sur les "
                         "scenarios de pannes",
            transform=ax.transAxes, color="white", fontsize=7.8,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=EDGE,
                      alpha=0.85))

    # -- validation de l'approximation ----------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _frame(ax, "L'approximation moyennee tient-elle ?", "Budget (k€/an)",
           "EVF")
    b = [r["budget"] for r in results]
    ax.plot(b, [r["mean"] for r in results], "o-", color="#6bcb77", lw=2,
            label="moyenne des scenarios (verite)")
    ax.plot(b, [r["static_approx"] for r in results], "s--", color="#ffd93d",
            lw=1.6, label="approximation R_eff")
    ax.plot(b, [r["nominal"] for r in results], "^:", color="#5a7ca8", lw=1.4,
            label="nominal (100 % dispo.)")
    ax.legend(fontsize=7.5, labelcolor="white", facecolor=BG, edgecolor=EDGE)
    ax.grid(alpha=0.15, color="white")

    ax = fig.add_subplot(gs[1, 1])
    _frame(ax, "Ecart de l'approximation", "Budget (k€/an)",
           "(R_eff - scenarios) / scenarios")
    err = [(r["static_approx"] - r["mean"]) / max(r["mean"], 1e-9) * 100
           for r in results]
    ax.bar(b, err, width=0.5 * (min(np.diff(b)) if len(b) > 1 else 100),
           color=["#ff6b6b" if e > 0 else "#4d96ff" for e in err])
    ax.axhline(0, color="white", lw=1)
    ax.set_ylabel("%", color="white", fontsize=9)
    ax.grid(alpha=0.15, color="white", axis="y")

    ax = fig.add_subplot(gs[1, 2])
    _frame(ax, "Degradation d'annee en annee", "Annee",
           "EVF moyenne annuelle")
    for c, r in zip(COLORS, results):
        S = r["series"]
        yr = [S[:, int(y * 365):int((y + 1) * 365)].mean()
              for y in range(int(S.shape[1] / 365))]
        ax.plot(range(1, len(yr) + 1), yr, "o-", color=c, lw=1.7)
    ax.grid(alpha=0.15, color="white")
    ax.text(0.03, 0.06, "une pente descendante signale un budget\nqui ne "
                        "rattrape pas les pannes",
            transform=ax.transAxes, color="white", fontsize=7.5,
            linespacing=1.5)

    fig.suptitle("Scenarios de maintenance pluriannuels — evaluation au "
                 "masque reel, sans moyennage",
                 color="white", fontsize=14, fontweight="bold", y=0.965)
    fig.savefig(out_path, dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Figure -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  PILOTE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nt", type=int, default=NT)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--n_min", type=int, default=10)
    p.add_argument("--n_max", type=int, default=30)
    p.add_argument("--evf_shrink", type=float, default=EVF_SHRINKAGE)
    p.add_argument("--maintenance", type=str, default="pirata",
                   choices=["regional", "pirata"])
    p.add_argument("--budgets", type=float, nargs="+", default=None)
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.25, 0.35, 0.55, 1.0])
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--scenarios", type=int, default=12)
    p.add_argument("--evaluator", type=str, default="static",
                   choices=["static", "kalman"],
                   help="kalman = filtre EOF/AR(1) avec memoire temporelle. "
                        "ATTENTION : la validation de l approximation R_eff "
                        "n a de sens qu avec l evaluateur statique, les deux "
                        "criteres n etant pas sur la meme echelle.")
    p.add_argument("--n_modes", type=int, default=50)
    p.add_argument("--propagator", type=str, default="lim",
                   choices=["lim", "ar1"])
    p.add_argument("--out_dir", type=str, default="outputs")
    a = p.parse_args()

    spec = importlib.util.spec_from_file_location(
        "brick3", Path(__file__).with_name("03_rl.py"))
    b3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b3)
    from campaign import greedy_under_budget, auto_budget_levels, marginal_info

    print(f"\n  Domaine {DOMAIN} {NX}x{NY} @ {DX_KM:.0f} km | profil "
          f"{a.maintenance} | {a.years} ans x {a.scenarios} scenarios")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=a.nt, seed=a.seed)
    port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM
    maint = b3.MaintenanceModel(b3.get_params(a.maintenance), port)
    env = b3.OceanNetworkEnv(T, S, grid_x=a.grid_x, grid_y=a.grid_y,
                             n_min=a.n_min, n_max=a.n_max,
                             fit_influence=True, evf_cv=True,
                             shrinkage=a.evf_shrink, maintenance=maint)

    budgets = a.budgets
    if not budgets:
        budgets, viable = auto_budget_levels(env, n_ref=a.n_max,
                                             fractions=tuple(a.fractions))
        print(f"  Budget minimum viable : {viable:.0f} k€/an")

    evaluate = (None if a.evaluator == "static"
                else make_evaluator(env, "kalman", n_modes=a.n_modes,
                                    propagator=a.propagator))
    results = []
    print(f"\n  {'budget':>8} | {'N':>3} | {'scenarios':>9} | "
          f"{'R_eff':>7} | {'ecart':>7} | {'dispo scen.':>11}")
    print("  " + "-" * 62)
    for B in budgets:
        g = greedy_under_budget(env, float(B), "effective", verbose=False)
        if len(g["idx"]) == 0:
            continue
        r = run_scenarios(env, g["idx"], float(B), years=a.years,
                          n_scenarios=a.scenarios, seed=a.seed,
                          priority=marginal_info(env, g["idx"]),
                          evaluate=evaluate)
        results.append(r)
        d = (r["static_approx"] - r["mean"]) / max(r["mean"], 1e-9) * 100
        print(f"  {B:>8.0f} | {r['n']:>3} | {r['mean']:>9.4f} | "
              f"{r['static_approx']:>7.4f} | {d:>+6.1f}% | "
              f"{r['availability_scenario']:>11.2f}", flush=True)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    plot_sawtooth(results, out / "maintenance_scenarios.png", a.years)
    payload = [{k: v for k, v in r.items()
                if k not in ("series", "plan", "idx")}
               | {"idx": [int(i) for i in r["idx"]],
                  "annual_mean": [float(r["series"][:, int(y*365):int((y+1)*365)].mean())
                                  for y in range(a.years)]}
               for r in results]
    (out / "maintenance_scenarios.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Donnees -> {out / 'maintenance_scenarios.json'}")

    errs = [abs(r["static_approx"] - r["mean"]) / max(r["mean"], 1e-9) * 100
            for r in results]
    print(f"\n  VALIDATION de l'approximation moyennee R_eff :")
    print(f"    ecart absolu moyen {np.mean(errs):.1f} % "
          f"(max {np.max(errs):.1f} %)")
    print(f"    -> au-dela de ~10 %, le moyennage de la disponibilite n'est\n"
          f"       plus defendable et il faut evaluer au masque reel.")


if __name__ == "__main__":
    main()
