"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MAINTENANCE — modèle de maintien en condition opérationnelle des bouées     ║
║                                                                              ║
║  Ce module remplace le proxy « coût ∝ nombre de bouées » par un modèle de    ║
║  maintien explicite, et — c'est le point clé — il RELIE ce modèle à          ║
║  l'information via la DISPONIBILITÉ des données :                            ║
║                                                                              ║
║      budget  ->  plan de campagnes  ->  intervalle de visite par bouée       ║
║              ->  disponibilité a_i  ->  variance d'erreur effective          ║
║              ->  variance expliquée par le réseau (EVF)                      ║
║                                                                              ║
║  Une bouée non maintenue n'est pas une bouée gratuite : c'est une bouée qui  ║
║  tombe en panne et cesse de porter de l'information. C'est ce couplage qui   ║
║  rend le ratio information / coût de maintien non trivial.                   ║
║                                                                              ║
║  Trois objets :                                                              ║
║    MaintenanceParams  — paramètres physiques et économiques (profils)        ║
║    MaintenancePlan    — sortie du planificateur (visites, routes, coûts)     ║
║    MaintenanceModel   — planification sous contrainte de budget              ║
╚══════════════════════════════════════════════════════════════════════════════╝

AVERTISSEMENT DE CALIBRATION
----------------------------
Les valeurs par défaut sont des ORDRES DE GRANDEUR documentés (jour-navire,
vitesse de transit, temps sur station, MTBF des mouillages). Elles sont
plausibles mais ne remplacent pas les chiffres réels des SNO. Tout est
regroupé dans `MAINT_PROFILES` (config.py) pour être remplacé par les coûts
réels d'exploitation sans toucher au reste du code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaintenanceParams:
    """Paramètres de maintien. Un jeu de valeurs = un contexte opérationnel."""

    name: str = "regional"

    # -- Navire ----------------------------------------------------------------
    ship_day_rate_keur: float = 12.0    # k€ / jour de mer (affrètement + équipage)
    ship_speed_kn: float = 10.0         # nœuds -> km/jour = 1.852 * 24 * kn
    ship_co2_t_per_day: float = 8.0     # tCO2 / jour de mer
    mobilisation_keur: float = 15.0     # k€ par jambe de campagne (départ port)
    endurance_days: float = 12.0        # autonomie max d'une jambe avant retour

    # -- Intervention sur une bouée --------------------------------------------
    on_station_days: float = 0.35       # jours sur station (relève + redéploiement)
    consumable_keur_per_visit: float = 3.5   # k€ (capteurs, batteries, lignes)

    # -- Matériel ---------------------------------------------------------------
    buoy_capex_keur: float = 45.0       # k€ — coût d'acquisition d'un mouillage
    buoy_life_years: float = 5.0        # années d'amortissement

    # -- Fiabilité --------------------------------------------------------------
    mtbf_days: float = 550.0            # temps moyen avant panne / perte / vandalisme
    deploy_horizon_days: float = 1825.0 # horizon si la bouée n'est JAMAIS visitée
    max_visits_per_year: int = 3        # nombre max de campagnes annuelles

    @property
    def ship_km_per_day(self) -> float:
        return self.ship_speed_kn * 1.852 * 24.0

    def as_dict(self) -> dict:
        return asdict(self)


# Profils prêts à l'emploi. `regional` correspond au domaine synthétique de la
# démo (800 x 1200 km, navire côtier). `pirata` donne les ordres de grandeur
# d'un réseau de mouillages hauturiers en Atlantique tropical : navire
# hauturier, campagnes longues, matériel lourd, MTBF dégradé par le vandalisme
# (déprédations sur les mouillages, cf. Bourlès et al. 2019, BAMS).
MAINT_PROFILES = {
    "regional": dict(
        name="regional",
        ship_day_rate_keur=12.0, ship_speed_kn=10.0, ship_co2_t_per_day=8.0,
        mobilisation_keur=15.0, endurance_days=12.0,
        on_station_days=0.35, consumable_keur_per_visit=3.5,
        buoy_capex_keur=45.0, buoy_life_years=5.0,
        mtbf_days=550.0, deploy_horizon_days=1825.0, max_visits_per_year=3,
    ),
    "pirata": dict(
        name="pirata",
        ship_day_rate_keur=30.0, ship_speed_kn=11.0, ship_co2_t_per_day=22.0,
        mobilisation_keur=60.0, endurance_days=35.0,
        on_station_days=1.0, consumable_keur_per_visit=18.0,
        buoy_capex_keur=120.0, buoy_life_years=4.0,
        mtbf_days=420.0, deploy_horizon_days=1460.0, max_visits_per_year=2,
    ),
}


def get_params(profile: str = "regional", **overrides) -> MaintenanceParams:
    """Construit un jeu de paramètres depuis un profil nommé."""
    if profile not in MAINT_PROFILES:
        raise KeyError(f"Profil inconnu : {profile}. "
                       f"Disponibles : {list(MAINT_PROFILES)}")
    cfg = dict(MAINT_PROFILES[profile])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return MaintenanceParams(**cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPONIBILITÉ  —  le pont entre maintenance et information
# ══════════════════════════════════════════════════════════════════════════════

def availability(interval_days: float, mtbf_days: float) -> float:
    """
    Fraction du temps pendant laquelle une bouée délivre effectivement de la
    donnée, quand elle est visitée tous les `interval_days`.

    Modèle : la panne (ou la perte) survient à un instant exponentiel de
    moyenne MTBF ; elle n'est réparée qu'à la visite suivante. Sur un
    intervalle Delta, la durée de fonctionnement attendue vaut

        E[min(tau, Delta)] = MTBF * (1 - exp(-Delta / MTBF))

    d'où la disponibilité

        a(Delta) = (1 - exp(-x)) / x        avec  x = Delta / MTBF

    Propriétés : a -> 1 quand Delta -> 0 (visites très fréquentes), a décroît
    de façon monotone, a ~ MTBF/Delta pour Delta >> MTBF. Croissante et
    saturante en nombre de visites : exactement la forme attendue d'un
    rendement décroissant de la maintenance.
    """
    x = max(float(interval_days), 1e-6) / max(float(mtbf_days), 1e-6)
    if x < 1e-6:
        return 1.0
    return float((1.0 - math.exp(-x)) / x)


def availability_from_visits(n_visits: int, p: MaintenanceParams) -> float:
    """Disponibilité annuelle pour `n_visits` campagnes de maintenance par an."""
    if n_visits <= 0:
        # Jamais visitée : elle fonctionne jusqu'à sa panne, puis reste morte
        # jusqu'à la fin de son déploiement nominal.
        return availability(p.deploy_horizon_days, p.mtbf_days)
    return availability(365.0 / n_visits, p.mtbf_days)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTAGE  —  tournée de maintenance
# ══════════════════════════════════════════════════════════════════════════════

def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def nearest_neighbour_tour(pts: np.ndarray, port: np.ndarray) -> list[int]:
    """Tournée gloutonne depuis le port (indices locaux dans `pts`)."""
    n = len(pts)
    remaining = list(range(n))
    cur = port
    order = []
    while remaining:
        d = np.linalg.norm(pts[remaining] - cur, axis=1)
        k = int(np.argmin(d))
        j = remaining.pop(k)
        order.append(j)
        cur = pts[j]
    return order


def two_opt(order: list[int], pts: np.ndarray, port: np.ndarray,
            max_passes: int = 8) -> list[int]:
    """
    Amélioration 2-opt de la tournée (port -> ... -> port).
    Bornée en nombre de passes : on veut une bonne tournée, pas l'optimum —
    la fonction est appelée des dizaines de milliers de fois pendant le RL.
    """
    n = len(order)
    if n < 4:
        return list(order)
    ordr = list(order)
    P = np.vstack([port, pts[ordr], port])                # (n+2, 2)
    for _ in range(max_passes):
        improved = False
        seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                # échange des arêtes (i-1, i) et (j, j+1)
                delta = (np.linalg.norm(P[i - 1] - P[j])
                         + np.linalg.norm(P[i] - P[j + 1])
                         - seg[i - 1] - seg[j])
                if delta < -1e-9:
                    P[i:j + 1] = P[i:j + 1][::-1]
                    ordr[i - 1:j] = ordr[i - 1:j][::-1]
                    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
                    improved = True
        if not improved:
            break
    return ordr


def plan_route(pts: np.ndarray, port: np.ndarray, refine: bool = True
               ) -> tuple[list[int], float]:
    """Tournée port -> bouées -> port. Retourne (ordre, longueur en km)."""
    if len(pts) == 0:
        return [], 0.0
    order = nearest_neighbour_tour(pts, port)
    if refine and len(order) >= 4:
        order = two_opt(order, pts, port)
    path = np.vstack([port, pts[order], port])
    length = float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())
    return order, length


# ══════════════════════════════════════════════════════════════════════════════
#  PLAN DE MAINTENANCE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Leg:
    """Une jambe de campagne : sortie du port, N stations, retour au port."""
    buoys: list[int]                    # indices GLOBAUX des bouées visitées
    waypoints: np.ndarray               # (n+2, 2) en km, port aux deux bouts
    km: float
    days: float


@dataclass
class Campaign:
    """Une campagne annuelle de maintenance (une ou plusieurs jambes)."""
    index: int
    legs: list[Leg] = field(default_factory=list)

    @property
    def buoys(self) -> list[int]:
        return [b for lg in self.legs for b in lg.buoys]

    @property
    def km(self) -> float:
        return sum(lg.km for lg in self.legs)

    @property
    def days(self) -> float:
        return sum(lg.days for lg in self.legs)


@dataclass
class MaintenancePlan:
    """Résultat de la planification sous contrainte de budget."""
    buoy_ids: np.ndarray                # indices globaux des bouées du réseau
    visits: np.ndarray                  # (n,) visites/an accordées à chaque bouée
    availability: np.ndarray            # (n,) disponibilité résultante
    campaigns: list[Campaign]
    cost_capex_keur: float              # amortissement du matériel
    cost_ship_keur: float               # jours de mer + mobilisation
    cost_consumable_keur: float         # consommables des interventions
    co2_t: float
    days_at_sea: float
    km: float
    budget_keur: float | None
    feasible: bool                      # le budget couvre-t-il au moins le capex ?
    unreachable: np.ndarray | None = None   # (n,) hors de portee du navire

    @property
    def total_cost_keur(self) -> float:
        return (self.cost_capex_keur + self.cost_ship_keur
                + self.cost_consumable_keur)

    @property
    def mean_availability(self) -> float:
        return float(self.availability.mean()) if len(self.availability) else 0.0

    def summary(self) -> dict:
        return {
            "n_buoys": int(len(self.buoy_ids)),
            "n_campaigns": len(self.campaigns),
            "n_legs": sum(len(c.legs) for c in self.campaigns),
            "visits_mean": float(self.visits.mean()) if len(self.visits) else 0.0,
            "availability_mean": self.mean_availability,
            "availability_min": (float(self.availability.min())
                                 if len(self.availability) else 0.0),
            "cost_total_keur": self.total_cost_keur,
            "cost_capex_keur": self.cost_capex_keur,
            "cost_ship_keur": self.cost_ship_keur,
            "cost_consumable_keur": self.cost_consumable_keur,
            "co2_t": self.co2_t,
            "days_at_sea": self.days_at_sea,
            "km": self.km,
            "budget_keur": self.budget_keur,
            "feasible": self.feasible,
            "n_unreachable": (int(self.unreachable.sum())
                              if self.unreachable is not None else 0),
        }


class MaintenanceModel:
    """
    Planificateur de maintenance sous contrainte de budget annuel.

    Politique de maintenance retenue (simple, lisible, et défendable devant un
    opérateur) : les campagnes sont EMBOÎTÉES. La campagne k visite toutes les
    bouées dont le niveau de service v_i >= k. La campagne 1 est donc la plus
    large, les suivantes ne servent que les bouées prioritaires.

    Allocation du budget :
        1. le capex (amortissement) est incompressible ;
        2. on tente de financer la campagne 1 complète. Si elle ne rentre pas,
           on retire itérativement la bouée de plus mauvais rapport
           priorité / surcoût de détour, jusqu'à tenir dans le budget ;
        3. on répète pour la campagne 2, puis 3, tant qu'il reste du budget.

    C'est exactement le ratio information / coût de maintien appliqué au niveau
    de la décision opérationnelle : sous budget serré, les bouées coûteuses à
    desservir et peu informatives sont visitées moins souvent — donc moins
    disponibles — donc contribuent moins à la variance expliquée.
    """

    def __init__(self, params: MaintenanceParams, port_km: Sequence[float]):
        self.p = params
        self.port = np.asarray(port_km, dtype=np.float64).reshape(2)

    # -- coûts élémentaires ----------------------------------------------------

    def leg_days(self, km: float, n_stations: int) -> float:
        return km / self.p.ship_km_per_day + n_stations * self.p.on_station_days

    def solo_leg_days(self, pt: np.ndarray) -> float:
        """Durée d'une jambe dédiée à une seule station (aller-retour port)."""
        km = 2.0 * float(np.linalg.norm(np.asarray(pt).reshape(2) - self.port))
        return self.leg_days(km, 1)

    def unreachable(self, pts: np.ndarray) -> np.ndarray:
        """
        Stations hors de portée du navire : même seules, leur aller-retour
        dépasse l'autonomie. Sur un grand domaine ce cas n'est pas théorique —
        c'est lui qui crée le vrai gradient spatial de coût de desserte, et le
        taire reviendrait à planifier des campagnes impossibles.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        return np.array([self.solo_leg_days(p) > self.p.endurance_days
                         for p in pts])

    def split_into_legs(self, order_pts: np.ndarray, ids: list[int]
                        ) -> list[Leg]:
        """
        Découpe une tournée en jambes respectant l'autonomie du navire.
        Les stations sont ajoutées dans l'ordre de la tournée tant que la jambe
        (transit aller + inter-stations + retour + temps sur station) tient dans
        `endurance_days`. Une station hors de portée est écartée.
        """
        legs: list[Leg] = []
        cur_ids: list[int] = []
        cur_pts: list[np.ndarray] = []
        for k, pt in enumerate(order_pts):
            if self.solo_leg_days(pt) > self.p.endurance_days:
                continue                     # hors de portee du navire
            trial_pts = cur_pts + [pt]
            path = np.vstack([self.port, np.array(trial_pts), self.port])
            km = float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())
            days = self.leg_days(km, len(trial_pts))
            if days > self.p.endurance_days and cur_pts:
                path0 = np.vstack([self.port, np.array(cur_pts), self.port])
                km0 = float(np.linalg.norm(path0[1:] - path0[:-1], axis=1).sum())
                legs.append(Leg(buoys=list(cur_ids), waypoints=path0, km=km0,
                                days=self.leg_days(km0, len(cur_ids))))
                cur_ids, cur_pts = [ids[k]], [pt]
            else:
                cur_ids = cur_ids + [ids[k]]
                cur_pts = trial_pts
        if cur_pts:
            path0 = np.vstack([self.port, np.array(cur_pts), self.port])
            km0 = float(np.linalg.norm(path0[1:] - path0[:-1], axis=1).sum())
            legs.append(Leg(buoys=list(cur_ids), waypoints=path0, km=km0,
                            days=self.leg_days(km0, len(cur_ids))))
        return legs

    def campaign_cost(self, legs: list[Leg]) -> tuple[float, float]:
        """(coût navire k€, CO2 t) d'une campagne, mobilisation par jambe incluse."""
        days = sum(lg.days for lg in legs)
        cost = days * self.p.ship_day_rate_keur + len(legs) * self.p.mobilisation_keur
        co2 = days * self.p.ship_co2_t_per_day
        return float(cost), float(co2)

    def _build_campaign(self, pts: np.ndarray, ids: list[int], index: int,
                        refine: bool) -> tuple[Campaign, float, float]:
        if len(ids) == 0:
            return Campaign(index=index, legs=[]), 0.0, 0.0
        order, _ = plan_route(pts, self.port, refine=refine)
        legs = self.split_into_legs(pts[order], [ids[o] for o in order])
        camp = Campaign(index=index, legs=legs)
        cost, co2 = self.campaign_cost(legs)
        # consommables : seules les stations effectivement desservies comptent
        cost += len(camp.buoys) * self.p.consumable_keur_per_visit
        return camp, cost, co2

    # -- planification ---------------------------------------------------------

    def plan(self, positions_km: np.ndarray, budget_keur: float | None = None,
             priority: np.ndarray | None = None, buoy_ids: np.ndarray | None = None,
             refine: bool = True) -> MaintenancePlan:
        """
        Parameters
        ----------
        positions_km : (n, 2) positions des bouées, en km dans le repère du port
        budget_keur  : budget annuel disponible. None = budget illimité
                       (toutes les bouées reçoivent `max_visits_per_year`)
        priority     : (n,) score d'intérêt de chaque bouée (variance locale,
                       EVF marginale...). Sert à arbitrer qui garde sa visite
                       quand le budget manque. Uniforme si None.
        """
        pts = np.asarray(positions_km, dtype=np.float64).reshape(-1, 2)
        n = len(pts)
        ids = (np.arange(n) if buoy_ids is None
               else np.asarray(buoy_ids, dtype=int))
        prio = (np.ones(n) if priority is None
                else np.asarray(priority, dtype=np.float64).reshape(n))
        prio = np.clip(prio, 1e-9, None)

        capex = n * self.p.buoy_capex_keur / max(self.p.buoy_life_years, 1e-6)
        if n == 0:
            return MaintenancePlan(np.array([], int), np.array([]), np.array([]),
                                   [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   budget_keur, True)

        visits = np.zeros(n, dtype=int)
        oor = self.unreachable(pts)      # bouees hors de portee du navire
        campaigns: list[Campaign] = []
        spent_ship, spent_cons, co2, days, km = 0.0, 0.0, 0.0, 0.0, 0.0
        feasible = budget_keur is None or capex <= budget_keur

        for k in range(1, self.p.max_visits_per_year + 1):
            # candidats de la campagne k : bouées déjà servies k-1 fois
            cand = [i for i in range(n) if visits[i] == k - 1 and not oor[i]]
            if not cand:
                break
            camp, cost_k, co2_k = self._build_campaign(pts[cand], cand, k, refine)

            if budget_keur is not None:
                remaining = budget_keur - (capex + spent_ship + spent_cons)
                # Retrait itératif de la bouée de plus mauvais rapport
                # priorité / économie réalisée. L'économie est estimée par le
                # DÉTOUR qu'elle impose sur la tournée courante — évaluation en
                # O(n) au lieu de reconstruire n tournées, ce qui rend la
                # planification utilisable dans la boucle du RL.
                for _ in range(len(cand) + 1):
                    if cost_k <= remaining or not cand:
                        break
                    order, _ = plan_route(pts[cand], self.port, refine=False)
                    seq = [cand[o] for o in order]
                    path = np.vstack([self.port, pts[seq], self.port])
                    d = np.linalg.norm(path[1:] - path[:-1], axis=1)
                    # économie kilométrique du retrait de la station de rang k
                    save_km = np.array([
                        d[t] + d[t + 1]
                        - float(np.linalg.norm(path[t + 2] - path[t]))
                        for t in range(len(seq))])
                    save_keur = (save_km / self.p.ship_km_per_day
                                 * self.p.ship_day_rate_keur
                                 + self.p.on_station_days * self.p.ship_day_rate_keur
                                 + self.p.consumable_keur_per_visit)
                    score = np.array([prio[i] for i in seq]) / np.maximum(save_keur, 1e-6)
                    n_drop = 1
                    if save_keur.sum() > 0:            # combien faut-il couper ?
                        deficit = cost_k - remaining
                        n_drop = int(np.searchsorted(
                            np.cumsum(np.sort(save_keur)[::-1]), deficit) + 1)
                    # décroissance géométrique : on ne retire jamais plus d'un
                    # tiers de la campagne d'un coup, sinon un budget serré
                    # supprime la campagne entière au lieu de la réduire.
                    n_drop = int(np.clip(n_drop, 1, max(1, len(seq) // 3)))
                    drop = set(np.array(seq)[np.argsort(score)[:n_drop]].tolist())
                    cand = [j for j in cand if j not in drop]
                    if cand:
                        camp, cost_k, co2_k = self._build_campaign(
                            pts[cand], cand, k, refine)
                    else:
                        camp, cost_k, co2_k = Campaign(k, []), 0.0, 0.0
                if not cand or cost_k > remaining:
                    break

            served = camp.buoys          # les hors-portee en sont exclus
            for i in served:
                visits[i] = k
            campaigns.append(camp)
            spent_ship += cost_k - len(served) * self.p.consumable_keur_per_visit
            spent_cons += len(served) * self.p.consumable_keur_per_visit
            co2 += co2_k
            days += camp.days
            km += camp.km

        avail = np.array([availability_from_visits(int(v), self.p) for v in visits])

        return MaintenancePlan(
            buoy_ids=ids, visits=visits, availability=avail, campaigns=campaigns,
            cost_capex_keur=float(capex), cost_ship_keur=float(spent_ship),
            cost_consumable_keur=float(spent_cons), co2_t=float(co2),
            days_at_sea=float(days), km=float(km), budget_keur=budget_keur,
            feasible=bool(feasible), unreachable=oor)

    # -- utilitaire pour la démo ----------------------------------------------

    def minimum_viable_budget(self, positions_km: np.ndarray) -> float:
        """Budget en dessous duquel le réseau ne peut même pas être entretenu
        une fois par an (capex + une campagne complète)."""
        pts = np.asarray(positions_km, dtype=np.float64).reshape(-1, 2)
        capex = len(pts) * self.p.buoy_capex_keur / max(self.p.buoy_life_years, 1e-6)
        if len(pts) == 0:
            return 0.0
        _, cost1, _ = self._build_campaign(pts, list(range(len(pts))), 1, True)
        return float(capex + cost1)
