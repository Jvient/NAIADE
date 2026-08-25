"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PIPELINE DE MAINTENANCE SUR DONNEES REELLES                                 ║
║                                                                              ║
║  Fait tourner la chaine complete — reseau sous budget, plan de campagnes,    ║
║  scenarios de pannes pluriannuels — sur GLORYS12 ou NATL60 plutot que sur    ║
║  l'ocean synthetique.                                                        ║
║                                                                              ║
║  L'evaluateur est le filtre de Kalman EOF/LIM, PAS le critere parametrique.  ║
║  Trois mesures independantes condamnent le noyau gaussien isotrope sur ces   ║
║  donnees :                                                                   ║
║      RMS residuel de l'ajustement      0,213 (NATL60) / 0,214 (GLORYS)       ║
║      anisotropie zonale/meridienne     0,82 sur le Gulf Stream               ║
║      correlations negatives            jusqu'a -0,78                         ║
║  Les EOF portent cette structure sans qu'on ait a la modeliser.              ║
║                                                                              ║
║  GRILLE. Le projet lit NX, NY et DX_KM depuis config au moment de l'import.  ║
║  Ce script lit donc d'abord l'entete NetCDF, pose les surcharges             ║
║  d'environnement, PUIS importe le reste. D'ou l'ordre inhabituel des imports ║
║  plus bas -- ce n'est pas un oubli.                                          ║
║                                                                              ║
║  AVERTISSEMENT SUR GLORYS. C'est une reanalyse : elle contient l'empreinte   ║
║  du reseau d'observation qu'on evalue, et sa mesoechelle est amortie -- sur  ║
║  la meme boite Gulf Stream, longueur de decorrelation 104 km contre 54 pour  ║
║  NATL60 en run libre, et correlation minimale -0,31 contre -0,78. Utiliser   ║
║  GLORYS pour les statistiques et la profondeur d'echantillon ; preferer      ║
║  NATL60 comme verite des que la mesoechelle compte.                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

    python run_real.py --source glorys --data_dir /path/glo12 --glob "*.nc" \\
        --box gulfstream --maintenance pirata
"""

from __future__ import annotations

import argparse, importlib.util, json, os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="glorys",
                   choices=["glorys", "natl60"])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--glob", type=str, default="*.nc")
    p.add_argument("--sst", type=str,
                   default="NATL60-CJM165_NATL_sst_y2013.1y.nc")
    p.add_argument("--ssh", type=str,
                   default="NATL60-CJM165_NATL_ssh_y2013.1y.nc")
    p.add_argument("--channels", type=str, nargs=2,
                   default=["thetao", "so"])
    p.add_argument("--box", type=str, default="gulfstream")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--lowpass_days", type=int, default=90)
    # reseau et maintien
    p.add_argument("--maintenance", type=str, default="pirata",
                   choices=["regional", "pirata"])
    p.add_argument("--grid_x", type=int, default=12)
    p.add_argument("--grid_y", type=int, default=12)
    p.add_argument("--n_min", type=int, default=6)
    p.add_argument("--n_max", type=int, default=20)
    p.add_argument("--fractions", type=float, nargs="+",
                   default=[0.25, 0.35, 0.55, 1.0])
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--scenarios", type=int, default=8)
    p.add_argument("--n_modes", type=int, default=150)
    p.add_argument("--optimize_with", type=str, default="kalman",
                   choices=["static", "kalman"],
                   help="Critere OPTIMISE par le glouton. 'static' optimise "
                        "l EVF instantanee alors que l evaluation finale se "
                        "fait au Kalman : optimiseur et evaluateur divergent, "
                        "et le reseau est sous-dimensionne parce que le "
                        "critere statique penalise la faible disponibilite "
                        "sans voir que l information persiste. 'kalman' "
                        "aligne les deux (plus lent).")
    p.add_argument("--opt_years", type=int, default=1)
    p.add_argument("--opt_scenarios", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def load_fields(a):
    """Charge la boite AVANT tout import de config, et rend (T, S, dx_km)."""
    from natl60 import load_box, mesoscale_anomaly_obs, BOXES
    d = Path(a.data_dir)
    bx = BOXES[a.box] if a.box in BOXES else a.box
    if a.source == "glorys":
        from glorys import load_box_glorys
        paths = sorted(d.glob(a.glob))
        if not paths:
            raise SystemExit(f"Aucun fichier '{a.glob}' dans {d}")
        box, _ = load_box_glorys(paths, box=bx, stride=a.stride,
                                 channels=tuple(a.channels))
        return box.sst, box.ssh, box
    box = load_box(d / a.sst, d / a.ssh, box=bx, stride=a.stride,
                   lowpass_days=a.lowpass_days)
    return (mesoscale_anomaly_obs(box.sst, a.lowpass_days),
            mesoscale_anomaly_obs(box.ssh, a.lowpass_days), box)


def main():
    a = parse_args()

    # ── 1. donnees, puis surcharge de grille, puis imports ───────────────────
    C1, C2, box = load_fields(a)
    nt, nx, ny = C1.shape
    os.environ["NAIADE_NX"] = str(nx)
    os.environ["NAIADE_NY"] = str(ny)
    os.environ["NAIADE_DX_KM"] = f"{box.dx_km:.6f}"

    from config import PORT_XY_FRAC, DX_KM, NX, NY
    spec = importlib.util.spec_from_file_location(
        "brick3", Path(__file__).with_name("03_rl.py"))
    b3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b3)
    from campaign import greedy_under_budget, auto_budget_levels, marginal_info
    from scenario import visit_calendar, simulate_uptime, plot_sawtooth
    from kalman import make_evaluator

    print(f"\n  Grille appliquee : {NX} x {NY} @ {DX_KM:.2f} km "
          f"({NX*DX_KM:.0f} x {NY*DX_KM:.0f} km, {nt} jours)")
    if abs(box.dx_ratio - 1) > 0.10:
        print(f"  [ATTENTION] maille anisotrope ({box.dx_ratio:.2f}) traitee "
              f"comme carree :\n              les distances, donc les couts de "
              f"desserte, sont faux de "
              f"{abs(box.dx_ratio-1)*100:.0f} % dans une direction.")

    # ── 2. environnement ─────────────────────────────────────────────────────
    port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM
    maint = b3.MaintenanceModel(b3.get_params(a.maintenance), port)
    env = b3.OceanNetworkEnv(C1, C2, grid_x=a.grid_x, grid_y=a.grid_y,
                             n_min=a.n_min, n_max=a.n_max,
                             fit_influence=True, evf_cv=True,
                             maintenance=maint)

    evaluate = make_evaluator(env, "kalman", n_modes=a.n_modes)

    def align_objective(budget):
        """
        Fait optimiser le glouton sur la MEME grandeur que l evaluation.

        Sans cela, le glouton maximise l EVF instantanee -- qui suppose qu une
        bouee morte cesse d informer immediatement -- puis on juge le resultat
        avec un filtre ou l information persiste plusieurs semaines. Le critere
        statique surestime alors le cout d une faible disponibilite, le glouton
        s arrete trop tot, et le reseau est SOUS-DIMENSIONNE. C est ce qui
        produit un ecart negatif au temoin aux budgets confortables.

        On enveloppe `evaluate` de l environnement : le plan de campagnes et
        les couts restent inchanges, seul le champ "info" est remplace par la
        moyenne d un scenario Kalman court. L original reste accessible sous
        "info_static".
        """
        orig = env.evaluate

        def wrapped(active_idx, *args, **kw):
            ev = orig(active_idx, *args, **kw)
            sel = np.asarray(active_idx, dtype=int)
            if len(sel) == 0:
                return ev
            plan = ev.get("plan")
            if plan is None:
                plan = orig(sel, budget_keur=budget, refine=False,
                            with_plan=True)["plan"]
            cal = visit_calendar(plan, maint.p)
            vals = [evaluate(sel, simulate_uptime(
                len(sel), cal, maint.p.mtbf_days, a.opt_years,
                np.random.default_rng(7000 + k))).mean()
                for k in range(a.opt_scenarios)]
            ev = dict(ev)
            ev["info_static"] = ev["info"]
            ev["info"] = float(np.mean(vals))
            return ev

        return orig, wrapped

    budgets, viable = auto_budget_levels(env, n_ref=a.n_max,
                                         fractions=tuple(a.fractions))
    print(f"  Budget minimum viable : {viable:.0f} k€/an")
    print(f"  Niveaux : " + ", ".join(f"{b:.0f}" for b in budgets) + " k€/an\n")

    # ── 3. reseau, campagnes, scenarios ──────────────────────────────────────
    results = []
    print(f"  {'budget':>8} | {'N':>3} | {'integre':>8} | {'temoin ent.':>11} "
          f"| {'ecart':>7} | {'dispo':>5}   (optimise et evalue : "
          f"{a.optimize_with} / kalman)")
    print("  " + "-" * 56)
    for B in budgets:
        restore = None
        if a.optimize_with == "kalman":
            restore, env.evaluate = align_objective(float(B))
        g = greedy_under_budget(env, float(B), "effective", verbose=False)
        if len(g["idx"]) == 0:
            continue
        ctl = greedy_under_budget(env, float(B), "nominal", verbose=False,
                                  require_serviceable=True)
        if restore is not None:
            env.evaluate = restore
            env._eval_cache.clear()
        # Les DEUX bras sont evalues par le MEME evaluateur, sur les MEMES
        # tirages de pannes. Comparer l'un au Kalman et l'autre au critere
        # statique reviendrait a comparer deux echelles sans rapport -- l'ecart
        # mesurerait le changement d'evaluateur, pas la difference de reseau.
        def scenario_mean(sel):
            if len(sel) == 0:
                return np.zeros((a.scenarios, int(a.years * 365)))
            e = env.evaluate(sel, budget_keur=float(B), refine=True,
                             priority=marginal_info(env, sel), with_plan=True)
            c = visit_calendar(e["plan"], maint.p)
            return np.array([
                evaluate(sel, simulate_uptime(len(sel), c, maint.p.mtbf_days,
                                              a.years,
                                              np.random.default_rng(1000 + k)))
                for k in range(a.scenarios)]), e

        idx = g["idx"]
        S, ev = scenario_mean(idx)
        Sc, _ = scenario_mean(ctl["idx"])
        ctl_mean = float(Sc.mean())
        d_ = (float(S.mean()) - ctl_mean) / max(ctl_mean, 1e-9)
        results.append({
            "budget": float(B), "series": S, "idx": idx, "plan": ev["plan"],
            "n": int(len(idx)), "mean": float(S.mean()),
            "static_approx": float(ev["info"]),
            "nominal": float(ev["info_nominal"]),
            "control_serviceable": ctl_mean,
            "control_n": int(len(ctl["idx"])),
            "availability_static": float(np.mean(ev["availability"])),
            "cost": float(ev["cost_keur"])})
        print(f"  {B:>8.0f} | {len(idx):>3} | {S.mean():>8.4f} | "
              f"{ctl_mean:>8.4f} | {d_*100:>+6.1f}% | "
              f"{np.mean(ev['availability']):>5.2f}", flush=True)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.source}_{box.name}"
    plot_sawtooth(results, out / f"real_scenarios_{tag}.png", a.years)
    (out / f"real_scenarios_{tag}.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("series", "plan", "idx")}
         | {"annual_mean": [float(r["series"][:, int(y*365):int((y+1)*365)].mean())
                            for y in range(a.years)]}
         for r in results], indent=2), encoding="utf-8")

    print(f"\n  Rayon d influence ajuste : {env.influence_px * DX_KM:.0f} km")
    print(f"  Evaluateur : Kalman EOF/LIM, {a.n_modes} modes "
          f"(pas de noyau parametrique)")
    print(f"  Donnees -> {out / f'real_scenarios_{tag}.json'}")
    print(f"\n  Rappel : l ecart au temoin ENTRETENABLE est la mesure qui "
          f"compte.\n  Sur le synthetique il valait +0,5 a +6,8 %, sous le "
          f"bruit du glouton.\n  S il change d ordre de grandeur ici, c est le "
          f"regime dynamique qui parle.")


if __name__ == "__main__":
    main()
