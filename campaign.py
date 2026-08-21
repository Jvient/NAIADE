"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CAMPAGNES DE MAINTENANCE — démo « réseau optimal par niveau de budget »     ║
║                                                                              ║
║  Pour chaque niveau de budget annuel, on produit :                           ║
║    1. le réseau qui maximise l'information EFFECTIVEMENT délivrée            ║
║       (information nominale dégradée par la disponibilité des bouées) ;      ║
║    2. le plan de campagnes finançable et la TRAJECTOIRE du navire ;          ║
║    3. le ratio information / coût de maintien, et sa comparaison au réseau   ║
║       conçu sans tenir compte du maintien.                                   ║
║                                                                              ║
║  Le résultat attendu — et c'est tout l'intérêt de la démo — est qu'à budget  ║
║  serré le réseau optimal n'est PAS le plus informatif sur le papier : c'est  ║
║  celui qu'on peut encore entretenir.                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import NX, NY, DX_KM, PORT_XY_FRAC
from data.dataset import mesoscale_anomaly

BG, PANEL, EDGE = "#0a1628", "#050d1a", "#2a4a7a"
CAMP_COLORS = ["#ffd93d", "#6bcb77", "#4d96ff", "#ff6b6b"]


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMISATION SOUS CONTRAINTE DE BUDGET
# ══════════════════════════════════════════════════════════════════════════════

def marginal_info(env, idx, availability=None):
    """Contribution marginale (leave-one-out) de chaque bouée à l'information."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) <= 1:
        return np.ones(len(idx))
    full = env.explained_variance(idx, availability)
    out = np.empty(len(idx))
    for k in range(len(idx)):
        keep = np.delete(np.arange(len(idx)), k)
        a = None if availability is None else np.asarray(availability)[keep]
        out[k] = full - env.explained_variance(idx[keep], a)
    return np.clip(out, 0.0, None)


def _insertion_km(env, sel, cands):
    """
    Surcoût kilométrique d'insertion de chaque candidat dans la tournée
    courante (port -> bouées sélectionnées -> port), en km. Vectorisé :
    pour chaque arête (a, b) du circuit, coût = d(a,c) + d(c,b) - d(a,b) ;
    on garde le minimum sur les arêtes.
    """
    port = env.maint.port
    C = env.positions_km(cands)                       # (nc, 2)
    if len(sel) == 0:
        return 2.0 * np.linalg.norm(C - port[None, :], axis=1)
    from maintenance import plan_route
    P = env.positions_km(np.asarray(sel, dtype=int))
    order, _ = plan_route(P, port, refine=False)
    path = np.vstack([port, P[order], port])          # (n+2, 2)
    A, B = path[:-1], path[1:]                        # arêtes
    dAB = np.linalg.norm(B - A, axis=1)               # (n+1,)
    dAC = np.linalg.norm(C[:, None, :] - A[None, :, :], axis=2)
    dCB = np.linalg.norm(C[:, None, :] - B[None, :, :], axis=2)
    return (dAC + dCB - dAB[None, :]).min(axis=1)


def greedy_under_budget(env, budget, objective="effective", shortlist=14,
                        n_max=None, patience=3, verbose=True,
                        require_serviceable=False):
    """
    Construction gloutonne du réseau sous contrainte de budget de maintien.

    objective = "effective" : maximise l'information réellement délivrée,
                              disponibilité comprise -> tient compte du fait
                              qu'ajouter une bouée dilue la maintenance.
    objective = "nominal"   : maximise l'information en supposant toutes les
                              bouées disponibles à 100 % -> c'est le réseau
                              qu'on dessine quand on ignore le maintien. Il
                              sert de témoin.

    require_serviceable     : n'admet que les configurations dont le plan
                              accorde AU MOINS UNE visite annuelle à chaque
                              bouée deployee.

    Pourquoi ce troisième régime. Le témoin "nominal" seul est un épouvantail
    aux budgets bas : comme une bouée non entretenue ne coûte que son
    amortissement, il en achète le maximum et n'a plus de quoi armer un seul
    navire (26 mouillages, disponibilité 0,29, zéro campagne). Battre cela ne
    prouve rien — aucun opérateur ne ferait ça. Combiné à objective="nominal",
    ce drapeau produit le témoin réaliste : quelqu'un qui ignore la notion
    d'information effective, mais qui refuse de déployer ce qu'il ne peut pas
    entretenir.

    Deux étages pour rester rapide : présélection de candidats, puis évaluation
    complète (plan de campagnes + disponibilité) de cette présélection.

    ATTENTION à la présélection. La classer sur la seule EVF nominale
    reviendrait à imposer au mode "effective" le vivier du mode "nominal" :
    les deux gloutons ne pourraient alors différer qu'à la marge, et la
    comparaison au témoin serait vide de sens. Le vivier réunit donc les
    meilleurs candidats sur l'information brute ET les meilleurs sur
    l'information par kilomètre de détour — une bouée médiocre mais sur la
    route reste dans la course.
    """
    n_max = int(n_max or env.n_max)
    sel: list[int] = []
    trace = []
    ev0 = env.evaluate([], budget_keur=budget)
    best = {"idx": np.array([], int), "ev": ev0, "score": -np.inf}
    since_best = 0

    for _ in range(n_max):
        cands = env.feasible_candidates(sel)
        if len(cands) == 0:
            break
        base = env.explained_variance(np.array(sel, dtype=int)) if sel else 0.0
        gains = np.array([env.explained_variance(np.array(sel + [int(c)]))
                          for c in cands]) - base
        top_info = cands[np.argsort(gains)[::-1][:shortlist]]
        if objective == "effective":
            detour = _insertion_km(env, sel, cands)
            eff = gains / np.maximum(detour, 1.0)     # information par km
            top_cheap = cands[np.argsort(eff)[::-1][:shortlist]]
            short = np.unique(np.concatenate([top_info, top_cheap]))
        else:
            short = top_info

        best_c, best_ev, best_score = None, None, -np.inf
        for c in short:
            cand = np.array(sorted(sel + [int(c)]), dtype=int)
            ev = env.evaluate(cand, budget_keur=budget, refine=False)
            if ev["over_budget"] > 1e-9:
                continue
            if require_serviceable and (len(ev["visits"]) == 0
                                        or int(np.min(ev["visits"])) < 1):
                continue
            score = ev["info_nominal"] if objective == "nominal" else ev["info"]
            if score > best_score:
                best_score, best_c, best_ev = score, int(c), ev
        if best_c is None:
            break

        sel.append(best_c)
        trace.append({"n": len(sel), "info": best_ev["info"],
                      "info_nominal": best_ev["info_nominal"],
                      "cost": best_ev["cost_keur"], "ratio": best_ev["ratio"]})
        # La configuration retenue est la meilleure AU SENS DE L'OBJECTIF
        # POURSUIVI. Retenir le témoin sur l'information effective lui
        # donnerait accès à une information qu'il est censé ignorer.
        if best_score > best["score"] + 1e-9:
            best = {"idx": np.array(sorted(sel), dtype=int), "ev": best_ev,
                    "score": best_score}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break     # l'information effective ne progresse plus

    idx = best["idx"]
    if len(idx) == 0:
        return {"idx": idx, "ev": best["ev"], "trace": trace, "plan": None}

    # Replanification finale : 2-opt complet, et priorité de maintenance donnée
    # par la contribution marginale réelle de chaque bouée (hors boucle, donc
    # pas de circularité).
    prio = marginal_info(env, idx)
    ev = env.evaluate(idx, budget_keur=budget, refine=True,
                      priority=prio, with_plan=True)
    if verbose:
        print(f"    budget {budget:>6.0f} k€ | N={len(idx):2d} | "
              f"info={ev['info']:.3f} (nominal {ev['info_nominal']:.3f}) | "
              f"coût={ev['cost_keur']:6.1f} k€ | "
              f"dispo={np.mean(ev['availability']):.2f} | "
              f"ratio={ev['ratio']:.3f} EVF/100k€")
    return {"idx": idx, "ev": ev, "trace": trace, "plan": ev["plan"]}


def policy_under_budget(env, policy, budget, n_max=None):
    """Séquence proposée par la politique PPO, tronquée au budget."""
    if policy is None:
        return None
    import torch
    from config import DEVICE
    n_max = int(n_max or env.n_max)
    mask = np.zeros(env.K, dtype=np.float32)
    sel, best = [], None
    policy.eval()
    saved = env.active_mask
    for _ in range(n_max):
        env.active_mask = mask.copy()
        try:
            obs = torch.from_numpy(env._get_obs().astype(np.float32))[None].to(DEVICE)
            with torch.no_grad():
                logits, _ = policy(obs)
        except Exception:
            break
        lg = logits[0].cpu().numpy()
        lg[mask > 0.5] = -np.inf
        lg[env.invalid_action_mask(mask)] = -np.inf
        if not np.isfinite(lg).any():
            break
        c = int(np.argmax(lg))
        mask[c] = 1.0
        sel.append(c)
        ev = env.evaluate(np.array(sorted(sel)), budget_keur=budget, refine=False)
        if ev["over_budget"] > 1e-9:
            sel.pop()
            break
        if best is None or ev["info"] > best["ev"]["info"]:
            best = {"idx": np.array(sorted(sel), dtype=int), "ev": ev}
    env.active_mask = saved
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _frame(ax, title="", xlab="", ylab=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
    if title:
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=7)
    ax.set_xlabel(xlab, color="white", fontsize=8.5)
    ax.set_ylabel(ylab, color="white", fontsize=8.5)
    ax.tick_params(colors="white", labelsize=7.5)


def plot_campaigns(env, results, out_dir, fname="campaign_budget_levels.png"):
    """
    Une carte par niveau de budget (réseau + trajectoire du navire), puis une
    ligne de synthèse : information vs budget, décomposition du coût, ratio
    information / coût.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [r["budget"] for r in results]
    nb = len(results)

    var_bg = mesoscale_anomaly(env.T).var(axis=0)
    port_px = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY])

    fig = plt.figure(figsize=(5.2 * nb, 10.4), facecolor=BG)
    gs = fig.add_gridspec(2, max(nb, 3), height_ratios=[1.35, 1.0],
                          hspace=0.26, wspace=0.24)

    for j, r in enumerate(results):
        ax = fig.add_subplot(gs[0, j])
        _frame(ax, f"Budget {r['budget']:.0f} k€/an   "
                   f"(N={len(r['idx'])}, EVF={r['ev']['info']:.3f})",
               "x (pixel)", "y (pixel)")
        ax.imshow(var_bg.T, cmap="Blues_r", origin="lower", aspect="auto",
                  extent=[0, NX, 0, NY], alpha=0.55)

        plan = r["plan"]
        if plan is not None:
            # Les campagnes sont emboîtées : la campagne 1 dessert tout le
            # monde, les suivantes ne repassent que sur les prioritaires. Les
            # tracer toutes en trait plein rendrait la carte illisible, d'où
            # trait plein pour la première et pointillés pour les repasses.
            for camp in plan.campaigns:
                col = CAMP_COLORS[(camp.index - 1) % len(CAMP_COLORS)]
                ls = "-" if camp.index == 1 else "--"
                lw = 1.7 if camp.index == 1 else 1.0
                for leg in camp.legs:
                    w = leg.waypoints / DX_KM          # km -> pixels
                    ax.plot(w[:, 0], w[:, 1], color=col, lw=lw, ls=ls,
                            alpha=0.9 if camp.index == 1 else 0.6,
                            zorder=4, solid_capstyle="round")
                    ax.plot(w[1:-1, 0], w[1:-1, 1], "o", color=col, ms=2.6,
                            zorder=5)
            if plan.campaigns and j == int(np.argmax(
                    [len(x["plan"].campaigns) if x["plan"] else 0
                     for x in results])):
                ax.legend(handles=[
                    Line2D([], [], color=CAMP_COLORS[(c.index - 1)
                                                     % len(CAMP_COLORS)],
                           ls="-" if c.index == 1 else "--",
                           lw=1.6, label=f"campagne {c.index} "
                                         f"({len(c.buoys)} bouées, "
                                         f"{c.days:.0f} j)")
                    for c in plan.campaigns],
                    fontsize=7, labelcolor="white", facecolor=BG,
                    edgecolor=EDGE, loc="lower right")

        pos = np.array([env.candidate_positions[i] for i in r["idx"]],
                       dtype=float) if len(r["idx"]) else np.zeros((0, 2))
        if len(pos):
            sc = ax.scatter(pos[:, 0], pos[:, 1],
                            c=r["ev"]["availability"], cmap="RdYlGn",
                            vmin=0.25, vmax=1.0, s=95, edgecolors="black",
                            linewidths=0.7, zorder=6)
            if j == nb - 1:
                cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
                cb.set_label("Disponibilité des données", color="white",
                             fontsize=8)
                cb.ax.yaxis.set_tick_params(color="white", labelcolor="white",
                                            labelsize=7)
        ax.plot(port_px[0], port_px[1], marker="*", ms=17, color="white",
                markeredgecolor="black", zorder=7)
        ax.annotate("port", port_px + np.array([4, 6]), color="white",
                    fontsize=8)
        ax.set_xlim(0, NX); ax.set_ylim(0, NY)

        txt = (f"{len(r['plan'].campaigns) if r['plan'] else 0} campagne(s), "
               f"{sum(len(c.legs) for c in r['plan'].campaigns) if r['plan'] else 0} jambe(s)\n"
               f"{r['ev']['km']:.0f} km · {r['ev']['days_at_sea']:.0f} j de mer · "
               f"{r['ev']['co2_t']:.0f} tCO2\n"
               f"dispo moy. {np.mean(r['ev']['availability']):.0%} · "
               f"ratio {r['ev']['ratio']:.3f} EVF/100k€")
        ax.text(0.02, 0.985, txt, transform=ax.transAxes, va="top", ha="left",
                color="white", fontsize=7.6, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=BG,
                          edgecolor=EDGE, alpha=0.88))

    # ── synthèse 1 : information vs budget ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _frame(ax, "Information délivrée vs budget de maintien",
           "Budget annuel (k€)", "Variance expliquée (EVF)")
    eff = [r["ev"]["info"] for r in results]
    nom = [r["ev"]["info_nominal"] for r in results]
    ax.plot(budgets, nom, "o--", color="#5a7ca8", lw=1.6, ms=6,
            label="nominale (bouées supposées 100 % disponibles)")
    ax.plot(budgets, eff, "o-", color="#6bcb77", lw=2.4, ms=7,
            label="effective (disponibilité comprise)")
    if any(r.get("naive") for r in results):
        nv = [r["naive"]["ev"]["info"] if r.get("naive") else np.nan
              for r in results]
        ax.plot(budgets, nv, "s:", color="#ff6b6b", lw=1.8, ms=6,
                label="réseau conçu sans le maintien, évalué au même budget")
    ax.fill_between(budgets, eff, nom, color="#ff6b6b", alpha=0.16)
    ax.legend(fontsize=7.2, labelcolor="white", facecolor=BG, edgecolor=EDGE,
              loc="lower right")
    ax.grid(True, alpha=0.2, color="white")

    # ── synthèse 2 : décomposition du coût ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _frame(ax, "Où part le budget", "Budget annuel (k€)", "k€/an")
    w = 0.62 * (min(np.diff(budgets)) if len(budgets) > 1 else 100)
    capex = [r["plan"].cost_capex_keur if r["plan"] else 0 for r in results]
    ship = [r["plan"].cost_ship_keur if r["plan"] else 0 for r in results]
    cons = [r["plan"].cost_consumable_keur if r["plan"] else 0 for r in results]
    ax.bar(budgets, capex, width=w, color="#4d96ff", label="matériel (amorti)")
    ax.bar(budgets, ship, width=w, bottom=capex, color="#ffd93d",
           label="jours de mer + mobilisation")
    ax.bar(budgets, cons, width=w,
           bottom=np.array(capex) + np.array(ship), color="#ff6b6b",
           label="consommables")
    ax.plot(budgets, budgets, ls=":", color="white", lw=1.2, alpha=0.7,
            label="budget disponible")
    ax.legend(fontsize=7.2, labelcolor="white", facecolor=BG, edgecolor=EDGE,
              loc="upper left")
    ax.grid(True, alpha=0.18, color="white", axis="y")

    # ── synthèse 3 : ratio information / coût ────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    _frame(ax, "Rendement du budget", "Budget annuel (k€)",
           "EVF pour 100 k€/an")
    ratio = [r["ev"]["ratio"] for r in results]
    ax.plot(budgets, ratio, "o-", color="#ffd93d", lw=2.4, ms=7)
    ax.grid(True, alpha=0.2, color="white")
    ax2 = ax.twinx()
    ax2.set_facecolor(PANEL)
    ax2.plot(budgets, [np.mean(r["ev"]["availability"]) for r in results],
             "s--", color="#6bcb77", lw=1.6, ms=5)
    ax2.set_ylabel("Disponibilité moyenne", color="#6bcb77", fontsize=8.5)
    ax2.tick_params(colors="#6bcb77", labelsize=7.5)
    ax2.set_ylim(0, 1.02)
    ax.text(0.02, 0.04,
            "Le ratio décroît : chaque k€ supplémentaire achète\n"
            "de moins en moins d'information (rendement décroissant).",
            transform=ax.transAxes, color="white", fontsize=7.2,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=EDGE,
                      alpha=0.85))

    fig.suptitle("Brique 3 — Réseau optimal et trajectoire de campagne "
                 "par niveau de budget de maintien",
                 color="white", fontsize=14, fontweight="bold", y=0.965)
    path = out_dir / fname
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Figure -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  PILOTE
# ══════════════════════════════════════════════════════════════════════════════

def auto_budget_levels(env, n_ref=None, fractions=(0.35, 0.55, 1.0, 1.8)):
    """
    Niveaux de budget calibrés sur le réseau lui-même.

    Le budget « minimum viable » est celui qui couvre l'amortissement du
    matériel plus UNE campagne annuelle complète. Les niveaux balayés en sont
    des fractions : en dessous de 1, la maintenance ne peut pas être assurée
    partout et les arbitrages commencent ; au-dessus, on achète des campagnes
    supplémentaires. Sans cette calibration, des budgets en dur ne veulent
    rien dire dès qu'on change la taille de la grille candidate.
    """
    n_ref = int(n_ref or env.n_max)
    idx = env.sample_feasible(n_ref, rng=np.random.default_rng(0))
    b_viable = env.maint.minimum_viable_budget(env.positions_km(idx))
    return [round(f * b_viable, -1) for f in fractions], b_viable


def run_campaign_demo(env, budgets, out_dir, policy=None, compare_naive=True,
                      shortlist=14, n_max=None, verbose=True):
    """Boucle sur les niveaux de budget et produit figure + JSON + résumé."""
    if env.maint is None:
        raise RuntimeError("Aucun modèle de maintien attaché à l'environnement "
                           "(utiliser --maintenance regional|pirata).")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    if budgets is None or len(budgets) == 0:
        budgets, b_viable = auto_budget_levels(env, n_ref=n_max)
        print(f"\n  Budget minimum viable (N={n_max or env.n_max}, capex + une "
              f"campagne annuelle complete) : {b_viable:.0f} k€/an")
        print(f"  Niveaux balayes (auto) : "
              + ", ".join(f"{b:.0f}" for b in budgets) + " k€/an")

    print("\n-- Réseau optimal et campagnes par niveau de budget ---------------")
    for b in sorted(budgets):
        r = greedy_under_budget(env, float(b), objective="effective",
                                shortlist=shortlist, n_max=n_max,
                                verbose=verbose)
        entry = {"budget": float(b), "idx": r["idx"], "ev": r["ev"],
                 "plan": r["plan"], "trace": r["trace"]}

        if compare_naive:
            nv = greedy_under_budget(env, float(b), objective="nominal",
                                     shortlist=shortlist, n_max=n_max,
                                     verbose=False)
            entry["naive"] = nv
            gain = r["ev"]["info"] - nv["ev"]["info"]
            if verbose:
                print(f"                  témoin « sans maintien » : "
                      f"N={len(nv['idx']):2d}, info={nv['ev']['info']:.3f} "
                      f"-> conception intégrée : {gain:+.3f} EVF "
                      f"({gain / max(nv['ev']['info'], 1e-9) * 100:+.1f} %)")

        if policy is not None:
            pol = policy_under_budget(env, policy, float(b), n_max=n_max)
            entry["policy"] = pol
            if pol is not None and verbose:
                print(f"                  politique PPO           : "
                      f"N={len(pol['idx']):2d}, info={pol['ev']['info']:.3f}")
        results.append(entry)

    fig_path = plot_campaigns(env, results, out_dir)

    # ── export JSON ──────────────────────────────────────────────────────────
    payload = []
    for r in results:
        plan = r["plan"]
        item = {
            "budget_keur": r["budget"],
            "n_buoys": int(len(r["idx"])),
            "positions_px": [[int(env.candidate_positions[i][0]),
                              int(env.candidate_positions[i][1])]
                             for i in r["idx"]],
            "info_effective": r["ev"]["info"],
            "info_nominal": r["ev"]["info_nominal"],
            "ratio_evf_per_100keur": r["ev"]["ratio"],
            "availability": [float(a) for a in r["ev"]["availability"]],
            "visits_per_year": [int(v) for v in r["ev"]["visits"]],
            "maintenance": plan.summary() if plan else None,
            "campaigns": [
                {"index": c.index, "km": c.km, "days": c.days,
                 "legs": [{"buoys": [int(b) for b in lg.buoys],
                           "km": lg.km, "days": lg.days,
                           "waypoints_km": lg.waypoints.round(1).tolist()}
                          for lg in c.legs]}
                for c in (plan.campaigns if plan else [])],
        }
        if r.get("naive"):
            item["naive_reference"] = {
                "n_buoys": int(len(r["naive"]["idx"])),
                "info_effective": r["naive"]["ev"]["info"],
                "info_nominal": r["naive"]["ev"]["info_nominal"]}
        payload.append(item)

    json_path = out_dir / "campaign_plan.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Plan détaillé -> {json_path}")
    return results, fig_path, json_path


def campaign_report_lines(env, results):
    """Lignes de rapport texte, à concaténer au rapport de la brique 3."""
    p = env.maint.p
    L = ["", "-- MAINTIEN EN CONDITION OPERATIONNELLE ------------------------------",
         f"  profil            : {p.name}",
         f"  navire            : {p.ship_day_rate_keur:.0f} k€/jour, "
         f"{p.ship_speed_kn:.0f} nds, {p.ship_co2_t_per_day:.0f} tCO2/jour",
         f"  station           : {p.on_station_days:.2f} jour/bouee, "
         f"consommables {p.consumable_keur_per_visit:.1f} k€",
         f"  materiel          : {p.buoy_capex_keur:.0f} k€ amortis sur "
         f"{p.buoy_life_years:.0f} ans",
         f"  fiabilite         : MTBF {p.mtbf_days:.0f} jours",
         "",
         "  budget |  N | info eff | info nom |  cout  | dispo | EVF/100k€ | tCO2",
         "  " + "-" * 68]
    for r in results:
        L.append(f"  {r['budget']:>6.0f} | {len(r['idx']):>2d} | "
                 f"{r['ev']['info']:>8.3f} | {r['ev']['info_nominal']:>8.3f} | "
                 f"{r['ev']['cost_keur']:>6.1f} | "
                 f"{np.mean(r['ev']['availability']):>5.2f} | "
                 f"{r['ev']['ratio']:>9.3f} | {r['ev']['co2_t']:>5.0f}")
    if any(r.get("naive") for r in results):
        L += ["", "  Temoin : reseau concu SANS modele de maintien, evalue au",
              "  meme budget (l'ecart mesure l'apport de la conception integree)",
              "  budget |  N | info eff | ecart"]
        for r in results:
            if not r.get("naive"):
                continue
            d = r["ev"]["info"] - r["naive"]["ev"]["info"]
            L.append(f"  {r['budget']:>6.0f} | {len(r['naive']['idx']):>2d} | "
                     f"{r['naive']['ev']['info']:>8.3f} | {d:+.3f}")
    return L
