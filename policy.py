"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  POLITIQUES DE MAINTENANCE ADAPTATIVES  —  quelle marge pour l'apprentissage ║
║                                                                              ║
║  Le planificateur actuel decide UNE FOIS pour toutes qui sera visite, et     ║
║  applique ce plan chaque annee quoi qu'il arrive. Une decision adaptative    ║
║  observe l'etat reel du reseau au depart de la campagne -- qui est mort,     ║
║  depuis quand -- et choisit en consequence. C'est un vrai probleme           ║
║  sequentiel sous incertitude, contrairement au placement statique.           ║
║                                                                              ║
║  Ce module NE cherche pas a apprendre une politique. Il mesure d'abord ce    ║
║  que valent des politiques simples, et surtout l'ecart qui les separe d'un   ║
║  oracle informe des pannes a venir. Cet ecart EST la marge disponible pour   ║
║  un agent : s'il vaut 3 %, il n'y a rien a apprendre ; s'il vaut 30 %, le    ║
║  sujet existe.                                                               ║
║                                                                              ║
║  La precaution n'est pas rhetorique. Sur ce projet, le RL a perdu d'un       ║
║  facteur deux contre un glouton, et le glouton n'a rien gagne contre une     ║
║  regle d'une ligne. Mesurer le plafond avant de courir apres coute une       ║
║  journee et evite de refaire le trajet.                                      ║
║                                                                              ║
║  Comparaison equitable : toutes les politiques rejouent LES MEMES tirages de ║
║  duree de vie (variables aleatoires communes). Les ecarts observes viennent  ║
║  des decisions, pas de la chance.                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

    NAIADE_DOMAIN=large python policy.py --maintenance pirata --n_max 30
"""

from __future__ import annotations

import argparse, importlib.util, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DOMAIN, NX, NY, DX_KM, PORT_XY_FRAC, NT, EVF_SHRINKAGE
from data.dataset import SyntheticOceanGenerator
from maintenance import plan_route
from kalman import make_evaluator

BG, PANEL, EDGE = "#0a1628", "#050d1a", "#2a4a7a"
PCOLORS = {"plan fixe": "#5a7ca8", "mortes d'abord": "#4d96ff",
           "plus anciennes": "#c77dff", "moins cheres": "#ffd93d",
           "contribution": "#6bcb77", "contribution dyn.": "#2f9e44",
           "contribution GNN": "#00c2a8", "oracle": "#ff6b6b"}


# ══════════════════════════════════════════════════════════════════════════════
#  TIRAGES COMMUNS
# ══════════════════════════════════════════════════════════════════════════════

def draw_lifetimes(n, mtbf, years, seed, max_repairs=40):
    """
    Durees de vie pre-tirees : la k-ieme est consommee apres la k-ieme
    reparation. Toutes les politiques partagent ces tirages, ce qui rend la
    comparaison appariee -- une politique ne peut pas gagner parce qu'elle a
    eu de la chance.
    """
    rng = np.random.default_rng(seed)
    return rng.exponential(mtbf, size=(max_repairs + 1, n))


# ══════════════════════════════════════════════════════════════════════════════
#  POLITIQUES
# ══════════════════════════════════════════════════════════════════════════════
#  Chacune recoit l'etat au depart de la campagne et retourne les bouees
#  candidates, PAR ORDRE DE PRIORITE. Le budget et l'autonomie tronquent
#  ensuite cette liste.

def pol_planned(st):
    """Plan fixe : on visite tout le monde, quel que soit l'etat reel.
    C'est ce que fait le planificateur aujourd'hui."""
    return list(np.argsort(-st["contribution"]))


def pol_dead_first(st):
    """Seules les bouees mortes, dans l'ordre de la tournee."""
    return [int(i) for i in np.flatnonzero(st["dead"])]


def pol_oldest(st):
    d = np.flatnonzero(st["dead"])
    return [int(i) for i in d[np.argsort(-st["days_dead"][d])]]


def pol_cheapest(st):
    d = np.flatnonzero(st["dead"])
    return [int(i) for i in d[np.argsort(st["detour_km"][d])]]


def pol_contribution(st):
    """
    Contribution FIGEE, calculee une fois sur le reseau complet.

    Defaut connu : au depart de la campagne, la moitie du reseau est morte.
    Une bouee jugee redondante parce qu'elle a deux voisines devient la seule
    source d'information de sa region si ces voisines sont mortes -- et cette
    politique ne peut pas le voir.
    """
    d = np.flatnonzero(st["dead"])
    return [int(i) for i in d[np.argsort(-st["contribution"][d])]]


def pol_contribution_dyn(st):
    """
    Contribution CONDITIONNEE aux bouees encore vivantes : gain reel de
    reparer j, soit EVF(vivantes + j) - EVF(vivantes). C'est la quantite que
    l'operateur voudrait connaitre, et elle change a chaque campagne.
    """
    d = np.flatnonzero(st["dead"])
    if len(d) == 0:
        return []
    g = st["gain_exact"]()
    return [int(i) for i in d[np.argsort(-g[d])]]


def pol_contribution_gnn(st):
    """
    Meme quantite, predite par le GNN de la brique 2 en une passe avant.

    Le GNN est inductif (GraphSAGE entraine sur des sous-reseaux tires au
    hasard), donc interrogeable sur n'importe quel ensemble de survivantes.
    A cette taille de reseau le calcul exact est abordable et le GNN ne gagne
    rien en vitesse : l'interet serait pour de plus grands reseaux ou dans une
    boucle d'apprentissage. Ce bras mesure donc le COUT de l'approximation,
    pas un gain.
    """
    d = np.flatnonzero(st["dead"])
    if len(d) == 0:
        return []
    g = st["gain_gnn"]()
    return [int(i) for i in d[np.argsort(-g[d])]]


def pol_oracle(st):
    """
    Oracle : connait la duree de vie que la bouee aura APRES reparation.

    Score = contribution x fraction de l'intervalle pendant laquelle elle
    sera effectivement en vie, DIVISE par le cout de desserte. Reparer une
    bouee qui remourra dans dix jours ne vaut rien, et seul l'oracle le sait.

    Le rapport au detour n'est pas cosmetique : une premiere version ignorait
    le cout, choisissait des bouees lointaines, en reparait moins, et se
    faisait battre de 5 % par la simple politique "moins cheres". Un oracle
    qui perd contre une heuristique ne mesure aucune marge.

    Ce n'est pas l'optimum exact -- c'est un oracle myope a une campagne, avec
    un score approche. Plafond indicatif, pas borne prouvee : un agent
    prevoyant pourrait le depasser.
    """
    d = np.flatnonzero(st["dead"])
    if len(d) == 0:
        return []
    horizon = max(st["days_to_next"], 1.0)
    frac = np.clip(st["next_life"][d] / horizon, 0.0, 1.0)
    score = st["contribution"][d] * frac / np.maximum(st["detour_km"][d], 1.0)
    return [int(i) for i in d[np.argsort(-score)]]


POLICIES = {
    "plan fixe":      pol_planned,
    "mortes d'abord": pol_dead_first,
    "plus anciennes": pol_oldest,
    "moins cheres":   pol_cheapest,
    "contribution":   pol_contribution,
    "contribution dyn.": pol_contribution_dyn,
    "oracle":         pol_oracle,
}


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate(env, idx, budget, policy_fn, lifetimes, years=5,
             n_campaigns=2, first_departure=45.0, contribution=None,
             days_per_year=365, gnn=None):
    """
    Boucle annuelle : a chaque depart de campagne, la politique ordonne les
    bouees a servir ; on construit la tournee, on la tronque au budget et a
    l'autonomie, et les reparations s'appliquent A LA DATE D'ARRIVEE du
    navire, pas au depart.
    """
    p = env.maint.p
    pts = env.positions_km(idx)
    n = len(idx)
    n_days = int(years * days_per_year)
    port = env.maint.port

    capex = n * p.buoy_capex_keur / max(p.buoy_life_years, 1e-6)
    contribution = (np.ones(n) if contribution is None
                    else np.asarray(contribution, float))

    up = np.ones((n_days, n), dtype=bool)
    alive = np.ones(n, dtype=bool)
    n_repairs = np.zeros(n, dtype=int)
    death_day = np.full(n, -1.0)
    # premiere vie
    t_fail = lifetimes[0, :n].copy()
    spent = {"ship": 0.0, "cons": 0.0}
    km_total, days_sea, n_visits = 0.0, 0.0, 0

    departures = sorted(first_departure + days_per_year * y
                        + days_per_year * k / n_campaigns
                        for y in range(int(years))
                        for k in range(n_campaigns))
    dep_idx = 0
    year_budget = budget - capex

    for t in range(n_days):
        if t % days_per_year == 0:                 # nouvelle annee budgetaire
            year_budget = budget - capex
        # pannes du jour
        for j in range(n):
            if alive[j] and t >= t_fail[j]:
                alive[j] = False
                death_day[j] = t
        # depart de campagne ?
        while dep_idx < len(departures) and departures[dep_idx] <= t:
            dep = departures[dep_idx]
            nxt = (departures[dep_idx + 1] if dep_idx + 1 < len(departures)
                   else n_days)
            order = _campaign(env, idx, pts, port, p, policy_fn, alive,
                              death_day, dep, nxt, contribution, lifetimes,
                              n_repairs, year_budget, gnn)
            for j, arrival in order:
                a = int(min(np.ceil(arrival), n_days))
                if a < n_days:
                    alive[j] = True
                    n_repairs[j] = min(n_repairs[j] + 1,
                                       lifetimes.shape[0] - 1)
                    t_fail[j] = arrival + lifetimes[n_repairs[j], j]
                    death_day[j] = -1.0
                    up[a:, j] = True
                n_visits += 1
            year_budget -= order.cost if hasattr(order, "cost") else 0.0
            year_budget = max(year_budget, 0.0)
            spent["ship"] += getattr(order, "cost", 0.0)
            km_total += getattr(order, "km", 0.0)
            days_sea += getattr(order, "days", 0.0)
            dep_idx += 1
        up[t] = alive

    return {"up": up, "km": km_total, "days_at_sea": days_sea,
            "n_visits": n_visits, "capex": capex}


class _Order(list):
    cost = 0.0
    km = 0.0
    days = 0.0


def _campaign(env, idx, pts, port, p, policy_fn, alive, death_day, dep, nxt,
              contribution, lifetimes, n_repairs, budget_left, gnn=None):
    """Construit une campagne : selection, tournee, troncature au budget."""
    n = len(idx)
    dead = ~alive
    detour = np.linalg.norm(pts - port[None, :], axis=1)
    nl = np.array([lifetimes[min(n_repairs[j] + 1, lifetimes.shape[0] - 1), j]
                   for j in range(n)])
    idx = np.asarray(idx, dtype=int)
    alive_idx = idx[alive]

    def gain_exact():
        base = env.explained_variance(alive_idx) if len(alive_idx) else 0.0
        g = np.zeros(n)
        for j in np.flatnonzero(dead):
            g[j] = env.explained_variance(
                np.sort(np.append(alive_idx, idx[j]))) - base
        return g

    def gain_gnn():
        g = np.zeros(n)
        if gnn is None:
            return gain_exact()
        for j in np.flatnonzero(dead):
            sub = np.sort(np.append(alive_idx, idx[j]))
            sc = np.asarray(gnn(env, sub))
            g[j] = float(sc[int(np.searchsorted(sub, idx[j]))])
        return g

    st = {"dead": dead, "alive": alive,
          "gain_exact": gain_exact, "gain_gnn": gain_gnn,
          "days_dead": np.where(death_day >= 0, dep - death_day, 0.0),
          "detour_km": detour, "contribution": contribution,
          "days_to_next": nxt - dep, "next_life": nl}

    cand = [j for j in policy_fn(st) if 0 <= j < n]
    out = _Order()
    if not cand:
        return out

    # troncature : on retire par la queue de la liste de priorite tant que la
    # campagne ne rentre pas dans le budget restant
    while cand:
        sub = np.array(cand, dtype=int)
        keep = [j for j in sub
                if p.on_station_days + 2 * np.linalg.norm(pts[j] - port)
                / p.ship_km_per_day <= p.endurance_days]
        if not keep:
            return out
        sub = np.array(keep, dtype=int)
        order, _ = plan_route(pts[sub], port, refine=len(sub) <= 12)
        legs = env.maint.split_into_legs(pts[sub][order],
                                         [int(sub[o]) for o in order])
        cost, _ = env.maint.campaign_cost(legs)
        cost += sum(len(l.buoys) for l in legs) * p.consumable_keur_per_visit
        if cost <= budget_left or len(cand) == 1:
            t = dep
            for leg in legs:
                wp = leg.waypoints
                seg = np.linalg.norm(wp[1:] - wp[:-1], axis=1) / p.ship_km_per_day
                for k, b in enumerate(leg.buoys):
                    t += seg[k] + p.on_station_days
                    out.append((int(b), t))
                t += seg[-1]
            out.cost = float(cost if cost <= budget_left else 0.0)
            out.km = float(sum(l.km for l in legs))
            out.days = float(sum(l.days for l in legs))
            if cost > budget_left:
                return _Order()
            return out
        cand = cand[:-1]
    return out


def budget_envelope(env, idx):
    """
    Bornes de budget annuel qui ont un sens pour ce reseau.

    plancher = amortissement + la campagne la moins chere possible (une seule
               bouee, la plus proche). En dessous, AUCUNE reparation n'est
               finançable et toutes les politiques sont identiquement nulles.
    plafond  = amortissement + une campagne servant tout le monde.

    Choisir le budget comme une fraction arbitraire du budget de
    dimensionnement conduit droit au regime degenere : sur un essai,
    l'amortissement mangeait 144 k€ sur 160, laissant 16 k€ quand la campagne
    minimale en coutait 31.
    """
    p = env.maint.p
    pts = env.positions_km(idx)
    n = len(idx)
    capex = n * p.buoy_capex_keur / max(p.buoy_life_years, 1e-6)
    single = []
    for j in range(n):
        legs = env.maint.split_into_legs(pts[[j]], [j])
        if not legs:
            continue
        c, _ = env.maint.campaign_cost(legs)
        single.append(c + p.consumable_keur_per_visit)
    floor = capex + (min(single) if single else 0.0)
    legs = env.maint.split_into_legs(
        pts[plan_route(pts, env.maint.port, refine=True)[0]],
        list(range(n)))
    full, _ = env.maint.campaign_cost(legs)
    full += n * p.consumable_keur_per_visit
    return capex, float(floor), float(capex + full)


def evf_series(env, idx, up):
    idx = np.asarray(idx, dtype=int)
    out = np.empty(len(up))
    cache: dict[tuple, float] = {}
    for t in range(len(up)):
        key = tuple(np.flatnonzero(up[t]).tolist())
        v = cache.get(key)
        if v is None:
            v = env.explained_variance(idx[list(key)]) if key else 0.0
            cache[key] = v
        out[t] = v
    return out


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


def plot_policies(res, out_path, years, budget):
    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.0], hspace=0.34,
                          wspace=0.26)
    ax = fig.add_subplot(gs[0, :])
    _frame(ax, f"Politiques de maintenance adaptatives — budget "
               f"{budget:.0f} k€/an", "Annees", "Variance expliquee (EVF)")
    for name, r in res.items():
        ax.plot(np.arange(len(r["mean_series"])) / 365.0, r["mean_series"],
                color=PCOLORS.get(name, "white"), lw=1.6,
                ls="--" if name == "oracle" else "-",
                label=f"{name}  (aire {r['auc']:.4f})")
    ax.legend(fontsize=8.5, labelcolor="white", facecolor=BG, edgecolor=EDGE,
              ncol=3, loc="lower left")
    ax.grid(alpha=0.15, color="white")

    names = list(res)
    ref = res["plan fixe"]["auc"]
    ax = fig.add_subplot(gs[1, 0])
    _frame(ax, "Gain sur le plan fixe", "", "%")
    g = [(res[n]["auc"] - ref) / max(ref, 1e-9) * 100 for n in names]
    ax.barh(names, g, color=[PCOLORS.get(n, "white") for n in names])
    ax.axvline(0, color="white", lw=1)
    ax.grid(alpha=0.15, color="white", axis="x")

    ax = fig.add_subplot(gs[1, 1])
    _frame(ax, "Marge restante jusqu'a l'oracle", "", "%")
    orc = res["oracle"]["auc"]
    m = [(orc - res[n]["auc"]) / max(res[n]["auc"], 1e-9) * 100 for n in names]
    ax.barh(names, m, color=[PCOLORS.get(n, "white") for n in names])
    ax.axvline(0, color="white", lw=1)
    ax.grid(alpha=0.15, color="white", axis="x")
    ax.text(0.5, 0.06, "c'est cette marge, et elle seule,\nqu'un agent "
                       "pourrait recuperer", transform=ax.transAxes,
            color="white", fontsize=7.6, ha="center", linespacing=1.5)

    ax = fig.add_subplot(gs[1, 2])
    _frame(ax, "Effort de maintenance", "", "jours de mer / an")
    ax.barh(names, [res[n]["days_at_sea"] / years for n in names],
            color=[PCOLORS.get(n, "white") for n in names])
    ax.grid(alpha=0.15, color="white", axis="x")

    fig.suptitle("Marge disponible pour une politique apprise",
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
    p.add_argument("--budget", type=float, default=None,
                   help="Budget annuel d EXPLOITATION (k€/an). Defaut : "
                        "0.35 x budget minimum viable.")
    p.add_argument("--budget_frac", type=float, default=0.45,
                   help="Position du budget d exploitation dans l enveloppe "
                        "utile (0 = une seule reparation finançable, 1 = tout "
                        "le reseau servi a chaque campagne). Le regime "
                        "interessant est au milieu.")
    p.add_argument("--budget_design", type=float, default=None,
                   help="Budget ayant servi a DIMENSIONNER le reseau. Par "
                        "defaut le budget minimum viable, ce qui represente "
                        "le cas courant d un reseau herite dont le budget "
                        "d entretien s est ensuite reduit. Le confondre avec "
                        "le budget d exploitation donne un reseau si petit "
                        "que toutes les bouees mortes sont reparables a "
                        "chaque campagne, et toutes les politiques "
                        "coincident.")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--scenarios", type=int, default=8)
    p.add_argument("--campaigns", type=int, default=2)
    p.add_argument("--evaluator", type=str, default="static",
                   choices=["static", "kalman"],
                   help="static = critere BLUE instantane (sans memoire) ; "
                        "kalman = filtre EOF/AR(1), l information d une bouee "
                        "persiste apres sa panne. Les niveaux absolus des deux "
                        "evaluateurs ne sont PAS comparables entre eux : ne "
                        "comparer des politiques qu a evaluateur fixe.")
    p.add_argument("--n_modes", type=int, default=50)
    p.add_argument("--propagator", type=str, default="lim",
                   choices=["lim", "ar1"],
                   help="lim = modele inverse lineaire (matrice pleine, "
                        "defaut) ; ar1 = diagonal, conserve pour comparaison "
                        "mais faux d un facteur ~10 sur la memoire.")
    p.add_argument("--gnn", action="store_true",
                   help="Entrainer le GNN de pertinence (priority.py) et "
                        "ajouter le bras 'contribution GNN'.")
    p.add_argument("--gnn_graphs", type=int, default=250)
    p.add_argument("--gnn_epochs", type=int, default=80)
    p.add_argument("--out_dir", type=str, default="outputs")
    a = p.parse_args()

    spec = importlib.util.spec_from_file_location(
        "brick3", Path(__file__).with_name("03_rl.py"))
    b3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b3)
    from campaign import greedy_under_budget, auto_budget_levels, marginal_info

    print(f"\n  Domaine {DOMAIN} | profil {a.maintenance} | {a.years} ans x "
          f"{a.scenarios} scenarios x {a.campaigns} campagne(s)/an")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=a.nt, seed=a.seed)
    port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM
    maint = b3.MaintenanceModel(b3.get_params(a.maintenance), port)
    env = b3.OceanNetworkEnv(T, S, grid_x=a.grid_x, grid_y=a.grid_y,
                             n_min=a.n_min, n_max=a.n_max,
                             fit_influence=True, evf_cv=True,
                             shrinkage=a.evf_shrink, maintenance=maint)

    lv, viable = auto_budget_levels(env, n_ref=a.n_max, fractions=(0.35,))
    b_design = a.budget_design if a.budget_design is not None else viable
    g = greedy_under_budget(env, float(b_design), "effective", verbose=False)
    idx = g["idx"]
    contrib = marginal_info(env, idx)

    capex, floor, ceil = budget_envelope(env, idx)
    budget = (a.budget if a.budget is not None
              else floor + a.budget_frac * (ceil - floor))
    print(f"  Dimensionnement du reseau : {b_design:.0f} k€/an -> N={len(idx)}")
    print(f"  Amortissement             : {capex:.0f} k€/an")
    print(f"  Enveloppe utile           : {floor:.0f} (une reparation) .. "
          f"{ceil:.0f} (tout le monde) k€/an")
    print(f"  Exploitation retenue      : {budget:.0f} k€/an")
    if budget < floor:
        print(f"\n  [ABANDON] Budget sous le plancher : aucune reparation "
              f"n'est finançable,\n            toutes les politiques seraient "
              f"identiquement nulles.\n            Relancer avec --budget "
              f">= {floor:.0f}.")
        return
    print()

    evaluate = make_evaluator(env, a.evaluator, n_modes=a.n_modes,
                              propagator=a.propagator)
    print(f"  Evaluateur : {a.evaluator}")

    policies = dict(POLICIES)
    gnn = None
    if a.gnn:
        import priority as P
        print(f"  Entrainement du GNN de pertinence "
              f"({a.gnn_graphs} graphes)...", flush=True)
        data = P.build_dataset(env, n_graphs=a.gnn_graphs,
                               n_range=(max(a.n_min, 6), len(idx) + 1),
                               seed=a.seed, verbose=False)
        from config import DEVICE
        model = P.PriorityGNN().to(DEVICE)
        rho = P.train(model, data, epochs=a.gnn_epochs, verbose=False)
        print(f"  GNN : Spearman validation {rho:+.3f}")
        gnn = P.GNNPriority(model, env)
        policies["contribution GNN"] = pol_contribution_gnn
        policies = {k: policies[k] for k in
                    [x for x in policies if x != "oracle"] + ["oracle"]}
    print()

    res = {}
    print(f"  {'politique':>16} | {'aire sous courbe':>16} | "
          f"{'vs plan fixe':>12} | {'j. mer/an':>9}")
    print("  " + "-" * 62)
    for name, fn in policies.items():
        series, days = [], []
        for s in range(a.scenarios):
            lifetimes = draw_lifetimes(len(idx), maint.p.mtbf_days, a.years,
                                       seed=1000 + s)
            sim = simulate(env, idx, float(budget), fn, lifetimes,
                           years=a.years, n_campaigns=a.campaigns,
                           contribution=contrib, gnn=gnn)
            series.append(evaluate(idx, sim["up"]))
            days.append(sim["days_at_sea"])
        S_ = np.array(series)
        res[name] = {"mean_series": S_.mean(0), "auc": float(S_.mean()),
                     "std": float(S_.mean(1).std()),
                     "days_at_sea": float(np.mean(days))}
        print(f"  {name:>16} | {res[name]['auc']:>16.4f} | ", end="")
        ref = res["plan fixe"]["auc"]
        print(f"{(res[name]['auc']-ref)/max(ref,1e-9)*100:>+11.1f}% | "
              f"{res[name]['days_at_sea']/a.years:>9.1f}", flush=True)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    plot_policies(res, out / "maintenance_policies.png", a.years, budget)
    (out / "maintenance_policies.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "mean_series"}
         for k, v in res.items()}, indent=2), encoding="utf-8")

    best = max((v["auc"], k) for k, v in res.items() if k != "oracle")
    marge = (res["oracle"]["auc"] - best[0]) / max(best[0], 1e-9) * 100
    print(f"\n  Meilleure heuristique : {best[1]} ({best[0]:.4f})")
    print(f"  Oracle                : {res['oracle']['auc']:.4f}")
    print(f"  MARGE POUR UN AGENT   : {marge:+.1f} %")
    if marge < 5:
        print("  -> marge trop faible pour justifier un apprentissage.")
    elif marge < 15:
        print("  -> marge modeste ; un agent devra faire nettement mieux que\n"
              "     l'heuristique pour valoir son cout.")
    else:
        print("  -> marge substantielle : le probleme merite un agent.")
    print("  (oracle myope a une campagne : plafond indicatif, pas une borne\n"
          "   prouvee -- un agent prevoyant pourrait le depasser)")


if __name__ == "__main__":
    main()
