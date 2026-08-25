"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BRIQUE 0 — OPÉRATEUR D'OBSERVATION                                          ║
║                                                                              ║
║  Extrait d'un nature run (synthétique ou GLORYS12) un jeu d'observations     ║
║  typées par nature de plateforme, avec les erreurs qui rendent le problème   ║
║  honnête :                                                                   ║
║    · erreur instrumentale        (par variable, cf. OBS_NOISE_T / _S)        ║
║    · erreur de représentativité  (décalage temporel ±k jours, Gasparin 2023) ║
║    · manquants NON aléatoires    (hasard de panne corrélé à la variance)     ║
║                                                                              ║
║  Le nature run n'est JAMAIS relu après cet appel : tout ce qui suit dans le  ║
║  pipeline obs-only ne voit que l'objet ObsSet.                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage minimal
-------------
    from obs_operator import ObsNetwork, Platform, PRESETS

    net = ObsNetwork(nx=NX, ny=NY, nt=len(T), rng=np.random.default_rng(7))
    net.add_moorings(n=20, **PRESETS["mooring"])
    net.add_argo(n=15,     **PRESETS["argo"])
    net.add_glider(waypoints=[(20,30),(120,180)], **PRESETS["glider"])
    obs = net.sample(T, S)          # -> ObsSet
    obs.save("outputs/obs_synth.npz")

Convention de coordonnées : indices grille (x, y) avec x in [0, nx), y in [0, ny),
identique au reste de NAIADE (positions = liste de tuples (x, y)).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# Import PAR NOM avec repli individuel. Un `from config import (A, B, C)`
# global échouerait en bloc si un seul nom manque — or main n'a pas
# OBS_NOISE_T/_S ni DX_KM (ajoutés sur glo12), et on perdrait silencieusement
# les vrais NX/NY au profit des valeurs par défaut.
_DEFAULTS = dict(NX=160, NY=240, OBS_NOISE_STD=0.05,
                 OBS_NOISE_T=None, OBS_NOISE_S=None, DX_KM=5.0)
try:
    import config as _cfg
except Exception:
    _cfg = None

def _cfg_get(name):
    v = getattr(_cfg, name, None) if _cfg is not None else None
    return _DEFAULTS[name] if v is None else v

NX = _cfg_get("NX")
NY = _cfg_get("NY")
DX_KM = _cfg_get("DX_KM")
# Sur main il n'existe qu'un OBS_NOISE_STD unique. Repli explicite : 0.05 vaut
# ~2 % du signal en température mais ~25 % en salinité — d'où la scission.
OBS_NOISE_T = _cfg_get("OBS_NOISE_T") or _cfg_get("OBS_NOISE_STD")
OBS_NOISE_S = _cfg_get("OBS_NOISE_S") or 0.4 * _cfg_get("OBS_NOISE_STD")
MISSING_CFG = [k for k in ("OBS_NOISE_T", "OBS_NOISE_S", "DX_KM")
               if _cfg is not None and getattr(_cfg, k, None) is None]


VARIABLES = ("T", "S")
KINDS = ("mooring", "drifter", "argo", "glider", "ship", "satellite")


# ══════════════════════════════════════════════════════════════════════════════
#  SPÉCIFICATION DE PLATEFORME
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Platform:
    """Caractéristiques d'échantillonnage d'un type de plateforme.

    period_days   : intervalle nominal entre deux observations
    variables     : sous-ensemble de ('T', 'S') effectivement mesuré
                    (PIRATA Pacifique = température seule → ('T',))
    noise_T/S     : écart-type instrumental, unités physiques
    repr_shift_d  : amplitude du décalage temporel de représentativité (±)
    return_rate   : taux de retour de données nominal (0-1)
    hazard_daily  : probabilité de panne définitive par jour
    hazard_var_amp: amplification du hasard de panne par la variance locale.
                    0 = manquants aléatoires ; >0 = manquants NON aléatoires,
                    les capteurs meurent davantage là où c'est énergétique.
    service_days  : intervalle de maintenance (remise en service). 0 = jamais.
    """
    kind: str = "mooring"
    variables: tuple = VARIABLES
    period_days: float = 1.0
    noise_T: float = OBS_NOISE_T
    noise_S: float = OBS_NOISE_S
    repr_shift_d: int = 3
    return_rate: float = 1.0
    hazard_daily: float = 0.0
    hazard_var_amp: float = 0.0
    service_days: int = 0
    # spécifiques
    speed_px_day: float = 0.0      # dériveur / glider / navire
    cell_px: int = 0               # argo : demi-côté de la cellule de dérive
    coverage: float = 1.0          # satellite : fraction de pixels clairs


PRESETS = {
    # Mouillage : haute fréquence, position fixe, forte contrainte logistique.
    "mooring": dict(kind="mooring", period_days=1.0, return_rate=0.80,
                    hazard_daily=2.0e-3, hazard_var_amp=2.0, service_days=365),
    # Dériveur : advecté, T seule le plus souvent, durée de vie limitée.
    "drifter": dict(kind="drifter", variables=("T",), period_days=1.0,
                    return_rate=0.90, hazard_daily=4.0e-3, speed_px_day=1.0),
    # Argo : cycle 10 j, position aléatoire dans une cellule (≈3°x3°).
    "argo":    dict(kind="argo", period_days=10.0, return_rate=0.95,
                    hazard_daily=5.0e-4, cell_px=12),
    # Glider : transect répété, pilotable.
    "glider":  dict(kind="glider", period_days=1.0, return_rate=0.95,
                    hazard_daily=1.0e-3, speed_px_day=4.0),
    # Navire d'opportunité : ligne fixe, passages irréguliers.
    "ship":    dict(kind="ship", period_days=1.0, return_rate=0.98,
                    hazard_daily=0.0, speed_px_day=40.0),
    # Satellite : champ quasi complet, trous nuageux corrélés.
    "satellite": dict(kind="satellite", variables=("T",), period_days=1.0,
                      noise_T=0.35, repr_shift_d=0, coverage=0.70),
}


@dataclass
class Sensor:
    """Un capteur instancié : métadonnées + trajectoire (peut être immobile)."""
    sid: int
    kind: str
    variables: tuple
    times: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    xs: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    ys: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    group: int = -1          # segment logique (trajectoire) pour le masquage
    noise: tuple = (OBS_NOISE_T, OBS_NOISE_S)

    @property
    def is_fixed(self):
        return self.kind in ("mooring",)

    @property
    def mean_pos(self):
        if len(self.xs) == 0:
            return (0, 0)
        return (int(np.round(self.xs.mean())), int(np.round(self.ys.mean())))


# ══════════════════════════════════════════════════════════════════════════════
#  JEU D'OBSERVATIONS (format long)
# ══════════════════════════════════════════════════════════════════════════════

class ObsSet:
    """Observations en format long + index par pas de temps.

    Colonnes : t, x, y, sid, group, kind_id, val_T, val_S, has_T, has_S
    Les valeurs sont dans l'unité du champ passé à sample() — donc normalisées
    si T, S le sont. Aucun accès au nature run n'est conservé.
    """

    def __init__(self, t, x, y, sid, group, kind_id, val, has,
                 sensors, nx, ny, nt, meta=None, ocean=None):
        self.t = np.asarray(t, np.int32)
        self.x = np.asarray(x, np.int32)
        self.y = np.asarray(y, np.int32)
        self.sid = np.asarray(sid, np.int32)
        self.group = np.asarray(group, np.int32)
        self.kind_id = np.asarray(kind_id, np.int8)
        self.val = np.asarray(val, np.float32)      # (n_obs, 2)
        self.has = np.asarray(has, np.bool_)        # (n_obs, 2)
        self.sensors = sensors
        self.nx, self.ny, self.nt = int(nx), int(ny), int(nt)
        self.meta = meta or {}
        # Masque océan (nx, ny) ou None. Indispensable dès que le domaine
        # contient de la terre : sans lui, toutes les moyennes spatiales
        # (TV, sigma, RMSE de référence) sont diluées par des pixels
        # continentaux que le modèle « reconstruit » à zéro sans effort.
        self.ocean = None if ocean is None else np.asarray(ocean, bool)
        self._index = None

    # ── accès ─────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.t)

    @property
    def n_sensors(self):
        return len(self.sensors)

    def index_by_time(self):
        """Liste de tableaux d'indices, un par pas de temps."""
        if self._index is None:
            order = np.argsort(self.t, kind="stable")
            splits = np.searchsorted(self.t[order], np.arange(self.nt + 1))
            self._index = [order[splits[k]:splits[k + 1]] for k in range(self.nt)]
        return self._index

    def at(self, t):
        return self.index_by_time()[t]

    def series(self, sid, var="T"):
        """Série temporelle d'un capteur : (times, values) — trous exclus."""
        v = VARIABLES.index(var)
        m = (self.sid == sid) & self.has[:, v]
        return self.t[m], self.val[m, v]

    def gridded_series(self, var="T", fill=np.nan):
        """Matrice (n_sensors, nt) avec NaN sur les manquants. Base de tous
        les diagnostics obs-only (corrélations, LOBO)."""
        v = VARIABLES.index(var)
        out = np.full((self.n_sensors, self.nt), fill, np.float32)
        m = self.has[:, v]
        out[self.sid[m], self.t[m]] = self.val[m, v]
        return out

    def positions(self):
        """Position moyenne par capteur — pour le graphe et les cartes."""
        return [s.mean_pos for s in self.sensors]

    # ── persistance ───────────────────────────────────────────────────────
    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, t=self.t, x=self.x, y=self.y, sid=self.sid,
            group=self.group, kind_id=self.kind_id, val=self.val, has=self.has,
            shape=np.array([self.nx, self.ny, self.nt]),
            sensors=json.dumps([
                dict(sid=s.sid, kind=s.kind, variables=list(s.variables),
                     times=s.times.tolist(), xs=s.xs.tolist(), ys=s.ys.tolist(),
                     group=s.group, noise=list(s.noise)) for s in self.sensors]),
            meta=json.dumps(self.meta),
            ocean=(np.zeros(0, np.bool_) if self.ocean is None
                   else self.ocean))
        return path

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        nx, ny, nt = d["shape"]
        sensors = [Sensor(sid=s["sid"], kind=s["kind"],
                          variables=tuple(s["variables"]),
                          times=np.array(s["times"], np.int32),
                          xs=np.array(s["xs"], np.int32),
                          ys=np.array(s["ys"], np.int32),
                          group=s["group"], noise=tuple(s["noise"]))
                   for s in json.loads(str(d["sensors"]))]
        oc = d["ocean"] if "ocean" in d.files else np.zeros(0, np.bool_)
        return cls(d["t"], d["x"], d["y"], d["sid"], d["group"], d["kind_id"],
                   d["val"], d["has"], sensors, nx, ny, nt,
                   json.loads(str(d["meta"])),
                   ocean=(None if oc.size == 0 else oc))

    def summary(self):
        by_kind = {}
        for s in self.sensors:
            by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
        lines = [f"ObsSet : {len(self):,} observations | {self.n_sensors} capteurs "
                 f"| grille {self.nx}x{self.ny} | {self.nt} pas de temps"]
        for k, n in sorted(by_kind.items()):
            m = np.array([s.kind == k for s in self.sensors])
            nobs = int(np.isin(self.sid, np.where(m)[0]).sum())
            lines.append(f"  {k:10s} : {n:3d} capteurs, {nobs:7,d} obs")
        cov = len(self) / max(1, self.nt)
        lines.append(f"  couverture moyenne : {cov:.1f} obs/pas de temps")
        if self.ocean is not None:
            lines.append(f"  masque océan : {100 * self.ocean.mean():.1f} % "
                         "de la grille")
        # Deux quantités DIFFÉRENTES, qu'il ne faut pas confondre :
        #   retour  = obs reçues / obs programmées pendant que le capteur vit
        #   actifs  = nombre de capteurs vivants à un instant donné
        # Un dériveur mort et non remplacé donne un "retour" catastrophique
        # alors que sa fin de vie est normale : c'est la couverture qui manque.
        lines.append(f"  {'type':10s} {'retour':>7s} {'actifs médians':>15s}")
        for kind in sorted({s.kind for s in self.sensors}):
            ids = [s.sid for s in self.sensors if s.kind == kind]
            planned = sum(len(self.sensors[i].times) for i in ids)
            got = int(np.isin(self.sid, ids).sum())
            live = np.zeros(self.nt, np.int32)
            for i in ids:
                tt = self.sensors[i].times
                if len(tt):
                    live[tt.min():tt.max() + 1] += 1
            if planned:
                lines.append(f"  {kind:10s} {100 * got / planned:>6.0f}% "
                             f"{int(np.median(live)):>15d}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  RÉSEAU — construction puis échantillonnage
# ══════════════════════════════════════════════════════════════════════════════

class ObsNetwork:
    """Déclare les plateformes, puis prélève dans le nature run."""

    def __init__(self, nx=NX, ny=NY, nt=1000, rng=None, ocean=None):
        self.nx, self.ny, self.nt = int(nx), int(ny), int(nt)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.ocean = ocean                       # (nx, ny) bool ou None
        self.sensors: list[Sensor] = []
        self.specs: dict[int, Platform] = {}
        self._next_group = 0

    # ── utilitaires ───────────────────────────────────────────────────────
    def _clip(self, x, y):
        return int(np.clip(x, 0, self.nx - 1)), int(np.clip(y, 0, self.ny - 1))

    def _is_ocean(self, x, y):
        return True if self.ocean is None else bool(self.ocean[x, y])

    def _rand_ocean_pos(self):
        for _ in range(500):
            x = int(self.rng.integers(0, self.nx))
            y = int(self.rng.integers(0, self.ny))
            if self._is_ocean(x, y):
                return x, y
        raise RuntimeError("aucune position océan trouvée")

    def _new_sensor(self, spec: Platform, times, xs, ys, group=None):
        sid = len(self.sensors)
        s = Sensor(sid=sid, kind=spec.kind, variables=tuple(spec.variables),
                   times=np.asarray(times, np.int32),
                   xs=np.asarray(xs, np.int32), ys=np.asarray(ys, np.int32),
                   group=self._next_group if group is None else group,
                   noise=(spec.noise_T, spec.noise_S))
        if group is None:
            self._next_group += 1
        self.sensors.append(s)
        self.specs[sid] = spec
        return s

    # ── plateformes ───────────────────────────────────────────────────────
    def add_moorings(self, n=20, positions=None, **kw):
        """Mouillages fixes. positions : liste (x, y) explicite (ex. PIRATA)."""
        spec = Platform(**{**PRESETS["mooring"], **kw})
        if positions is None:
            positions = [self._rand_ocean_pos() for _ in range(n)]
        step = max(1, int(round(spec.period_days)))
        times = np.arange(0, self.nt, step)
        for (x, y) in positions:
            x, y = self._clip(x, y)
            self._new_sensor(spec, times, np.full(len(times), x),
                             np.full(len(times), y))
        return self

    def _draw_lifetime(self, spec):
        """Durée de vie d'une plateforme dérivante, tirée à la création.

        Pour un mouillage, le hasard de panne dépend de la variance locale et
        est appliqué dans sample(). Pour une plateforme MOBILE, ça n'a pas de
        sens (elle change de lieu) et surtout il faut connaître la durée de vie
        AVANT de construire la trajectoire, pour pouvoir programmer le
        remplacement.
        """
        h = float(np.clip(spec.hazard_daily, 1e-6, 0.5))
        return int(self.rng.geometric(h)) if spec.hazard_daily > 0 else self.nt

    def add_argo(self, n=15, maintain=True, **kw):
        """Flotteurs : cycle ~10 j, position aléatoire dans une cellule.

        maintain=True : n flotteurs ACTIFS SIMULTANÉMENT, remplacés à leur
        mort. Sans ça, sur un run décennal la flotte s'éteint la première
        année et la couverture s'effondre sans que rien ne le signale.
        """
        spec = Platform(**{**PRESETS["argo"], **kw})
        step = max(1, int(round(spec.period_days)))
        c = max(1, spec.cell_px)
        dead = Platform(**{**PRESETS["argo"], **kw, "hazard_daily": 0.0})

        for _slot in range(n):
            t0 = 0
            while t0 < self.nt:
                life = self._draw_lifetime(spec)
                t_end = min(self.nt, t0 + life)
                cx, cy = self._rand_ocean_pos()
                times = np.arange(t0 + int(self.rng.integers(0, step)),
                                  t_end, step)
                if len(times) == 0:
                    t0 = t_end + 1
                    continue
                xs, ys = [], []
                for _t in times:
                    x, y = self._clip(cx + self.rng.integers(-c, c + 1),
                                      cy + self.rng.integers(-c, c + 1))
                    if not self._is_ocean(x, y):
                        x, y = cx, cy
                    xs.append(x); ys.append(y)
                self._new_sensor(dead, times, xs, ys)
                t0 = t_end
                if not maintain:
                    break
        return self

    def add_drifters(self, n=10, u=None, v=None, maintain=True, **kw):
        """Dériveurs advectés. u, v : vitesses (nx, ny) en px/jour ou scalaires.

        maintain=True : n dériveurs ACTIFS SIMULTANÉMENT. Un SVP vit ~400 j ;
        sur 10 ans, sans remplacement, la flotte disparaît après la 1re année.
        """
        spec = Platform(**{**PRESETS["drifter"], **kw})
        dead = Platform(**{**PRESETS["drifter"], **kw, "hazard_daily": 0.0})
        for _slot in range(n):
            t0 = 0
            while t0 < self.nt:
                life = self._draw_lifetime(spec)
                self._one_drifter(dead, u, v, t0, min(self.nt, t0 + life))
                t0 = t0 + life
                if not maintain:
                    break
        return self

    def _one_drifter(self, spec, u, v, t_start, t_end):
        x, y = self._rand_ocean_pos()
        times, xs, ys = [], [], []
        fx, fy = float(x), float(y)
        for t in range(t_start, t_end):
            ux = self._vel(u, fx, fy, 0.0)
            uy = self._vel(v, fx, fy, 0.0)
            fx += ux + self.rng.normal(0, spec.speed_px_day * 0.3)
            fy += uy + self.rng.normal(0, spec.speed_px_day * 0.3)
            fx = float(np.clip(fx, 0, self.nx - 1))
            fy = float(np.clip(fy, 0, self.ny - 1))
            xi, yi = self._clip(round(fx), round(fy))
            if not self._is_ocean(xi, yi):
                break                       # échoué à la côte
            times.append(t); xs.append(xi); ys.append(yi)
        if len(times) > 10:
            return self._new_sensor(spec, times, xs, ys)
        return None

    def _vel(self, f, x, y, default):
        if f is None:
            return default
        if np.isscalar(f):
            return float(f)
        return float(f[int(np.clip(x, 0, self.nx - 1)),
                       int(np.clip(y, 0, self.ny - 1))])

    def add_glider(self, waypoints, n_repeat=None, **kw):
        """Transect répété en dents de scie entre waypoints.
        Chaque *passage* est un groupe distinct (masquage par segment)."""
        spec = Platform(**{**PRESETS["glider"], **kw})
        path = self._densify(waypoints, spec.speed_px_day)
        return self._trajectory_platform(spec, path, n_repeat)

    def add_ship(self, waypoints, n_repeat=None, **kw):
        """Navire d'opportunité : ligne fixe, passages rapides et irréguliers."""
        spec = Platform(**{**PRESETS["ship"], **kw})
        path = self._densify(waypoints, spec.speed_px_day)
        return self._trajectory_platform(spec, path, n_repeat, irregular=True)

    def _densify(self, waypoints, speed):
        """Discrétise une polyligne au pas `speed` px/jour."""
        pts = []
        wp = list(waypoints) + [waypoints[0]]        # aller-retour
        for (x0, y0), (x1, y1) in zip(wp[:-1], wp[1:]):
            d = np.hypot(x1 - x0, y1 - y0)
            k = max(2, int(np.ceil(d / max(1e-6, speed))))
            for a in np.linspace(0, 1, k, endpoint=False):
                pts.append(self._clip(round(x0 + a * (x1 - x0)),
                                      round(y0 + a * (y1 - y0))))
        return [p for p in pts if self._is_ocean(*p)]

    def _trajectory_platform(self, spec, path, n_repeat, irregular=False):
        if not path:
            return self
        L = len(path)
        n_repeat = n_repeat if n_repeat is not None else max(1, self.nt // L)
        t = 0
        for _r in range(n_repeat):
            if irregular:
                t += int(self.rng.integers(0, max(1, self.nt // (2 * n_repeat))))
            if t >= self.nt:
                break
            times = np.arange(t, min(self.nt, t + L))
            seg = path[:len(times)]
            g = self._next_group; self._next_group += 1
            self._new_sensor(spec, times, [p[0] for p in seg],
                             [p[1] for p in seg], group=g)
            t += L
        return self

    def add_satellite(self, **kw):
        """Champ quasi complet avec trous nuageux corrélés. Un seul « capteur »
        logique, mais des dizaines de milliers d'observations par pas de temps.
        Représente ce que l'in situ doit compléter, pas remplacer."""
        spec = Platform(**{**PRESETS["satellite"], **kw})
        self._satellite = spec
        return self

    # ── prélèvement ───────────────────────────────────────────────────────
    def sample(self, T, S, verbose=True):
        """Prélève dans le nature run. T, S : (nt, nx, ny).

        Applique dans l'ordre : décalage de représentativité, lecture du champ,
        bruit instrumental, taux de retour, hasard de panne (corrélé à la
        variance locale si hazard_var_amp > 0).
        """
        T = np.asarray(T); S = np.asarray(S)
        nt = min(self.nt, len(T))
        var_rank = self._variance_rank(T)

        rows_t, rows_x, rows_y, rows_sid = [], [], [], []
        rows_g, rows_k, rows_v, rows_h = [], [], [], []

        for s in self.sensors:
            spec = self.specs[s.sid]
            alive_until = self._death_time(s, spec, var_rank, nt)
            keep = (s.times < min(nt, alive_until))
            if not keep.any():
                continue
            tt, xx, yy = s.times[keep], s.xs[keep], s.ys[keep]

            # taux de retour (manquants ponctuels)
            ok = self.rng.random(len(tt)) < spec.return_rate
            tt, xx, yy = tt[ok], xx[ok], yy[ok]
            if len(tt) == 0:
                continue

            # erreur de représentativité : lecture décalée de ±k jours
            if spec.repr_shift_d > 0:
                sh = self.rng.integers(-spec.repr_shift_d,
                                       spec.repr_shift_d + 1, size=len(tt))
                t_read = np.clip(tt + sh, 0, nt - 1)
            else:
                t_read = tt

            vT = T[t_read, xx, yy].astype(np.float32)
            vS = S[t_read, xx, yy].astype(np.float32)
            vT = vT + self.rng.normal(0, spec.noise_T, len(tt)).astype(np.float32)
            vS = vS + self.rng.normal(0, spec.noise_S, len(tt)).astype(np.float32)

            hT = "T" in s.variables
            hS = "S" in s.variables
            rows_t.append(tt); rows_x.append(xx); rows_y.append(yy)
            rows_sid.append(np.full(len(tt), s.sid, np.int32))
            rows_g.append(np.full(len(tt), s.group, np.int32))
            rows_k.append(np.full(len(tt), KINDS.index(s.kind), np.int8))
            rows_v.append(np.stack([vT, vS], 1))
            rows_h.append(np.tile(np.array([hT, hS], bool), (len(tt), 1)))

        if not rows_t:
            raise RuntimeError("aucune observation générée")

        obs = ObsSet(np.concatenate(rows_t), np.concatenate(rows_x),
                     np.concatenate(rows_y), np.concatenate(rows_sid),
                     np.concatenate(rows_g), np.concatenate(rows_k),
                     np.concatenate(rows_v), np.concatenate(rows_h),
                     self.sensors, self.nx, self.ny, nt,
                     meta={"specs": {int(k): asdict(v)
                                     for k, v in self.specs.items()}},
                     ocean=self.ocean)
        if verbose:
            print(obs.summary())
        return obs

    def satellite_field(self, T, S=None, t=None, rng=None):
        """Champ satellite masqué par les nuages, pour le mode « apport
        conditionnel de l'in situ sachant le satellite ». Retourne (val, mask)."""
        spec = getattr(self, "_satellite", None)
        if spec is None:
            return None, None
        rng = rng or self.rng
        cloud = _smooth_noise(T.shape[-2:], rng, sigma=8.0)
        thr = np.quantile(cloud, 1.0 - spec.coverage)
        mask = (cloud <= thr)
        if self.ocean is not None:
            mask &= self.ocean
        field = T[t] + rng.normal(0, spec.noise_T, T.shape[-2:])
        return field.astype(np.float32), mask

    # ── manquants non aléatoires ──────────────────────────────────────────
    def _variance_rank(self, T):
        """Rang normalisé [0,1] de la variance temporelle locale."""
        v = T.var(axis=0)
        flat = v.ravel()
        order = flat.argsort().argsort().astype(np.float32)
        return (order / max(1, len(flat) - 1)).reshape(v.shape)

    def _death_time(self, s: Sensor, spec: Platform, var_rank, nt):
        """Instant de panne définitive (ou de la dernière maintenance)."""
        if spec.hazard_daily <= 0:
            return nt
        x, y = s.mean_pos
        r = float(var_rank[min(x, var_rank.shape[0] - 1),
                           min(y, var_rank.shape[1] - 1)])
        h = spec.hazard_daily * (1.0 + spec.hazard_var_amp * (r - 0.5) * 2.0)
        h = float(np.clip(h, 1e-6, 0.5))
        t_death = int(self.rng.geometric(h))
        if spec.service_days > 0:
            # remise en service : on ne meurt que si la panne survient et n'est
            # pas réparée avant la fin de la période courante
            cycles = nt // spec.service_days + 1
            for c in range(cycles):
                if t_death > spec.service_days:
                    t_death = (c + 1) * spec.service_days + int(
                        self.rng.geometric(h))
                else:
                    break
        return t_death


def estimate_advection(T, max_shift=6, n_t=60, seed=0):
    """Estime (u, v) en PIXELS PAR PAS DE TEMPS par corrélation croisée décalée.

    À utiliser pour semer les dériveurs : config.U_MEAN / V_MEAN sont exprimés
    dans les unités internes du générateur, pas en px/pas, et l'erreur d'unité
    est invisible (les dériveurs partent simplement au mauvais endroit).
    Cette estimation est mesurée sur le champ lui-même, donc sans ambiguïté.
    """
    rng = np.random.default_rng(seed)
    nt = len(T)
    ts = rng.choice(nt - 1, min(n_t, nt - 1), replace=False)
    best, bu, bv = -np.inf, 0.0, 0.0
    a = np.stack([T[t] for t in ts])
    a = (a - a.mean((1, 2), keepdims=True))
    for du in range(-max_shift, max_shift + 1):
        for dv in range(-max_shift, max_shift + 1):
            b = np.stack([np.roll(np.roll(T[t + 1], -du, 0), -dv, 1)
                          for t in ts])
            b = b - b.mean((1, 2), keepdims=True)
            c = float((a * b).mean()
                      / (a.std() * b.std() + 1e-9))
            if c > best:
                best, bu, bv = c, float(du), float(dv)
    return bu, bv, best


def _smooth_noise(shape, rng, sigma=8.0):
    """Bruit gaussien lissé (nuages corrélés) sans dépendre de scipy."""
    w = rng.normal(size=shape).astype(np.float32)
    k = int(max(1, round(sigma)))
    ker = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
    ker /= ker.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, "same"), 0, w)
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, "same"), 1, out)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT PAR CAPTEUR — jamais par pixel
# ══════════════════════════════════════════════════════════════════════════════

def split_sensors(obs: ObsSet, frac_fit=0.7, frac_val=0.15, seed=0,
                  stratify_by_kind=True):
    """Partition des CAPTEURS en fit / val / test.

    Le split est au niveau capteur (et par groupe pour les trajectoires) :
    un split par pixel laisserait fuir les points voisins d'un même glider.
    """
    rng = np.random.default_rng(seed)
    groups = np.array([s.group for s in obs.sensors])
    kinds = np.array([s.kind for s in obs.sensors])
    uniq = np.unique(groups)

    assign = {}
    strata = [None] if not stratify_by_kind else sorted(set(kinds))
    for k in strata:
        sel = uniq if k is None else np.unique(
            groups[kinds == k]) if (kinds == k).any() else np.array([])
        sel = rng.permutation(sel)
        n = len(sel)
        n_fit = int(round(frac_fit * n))
        n_val = int(round(frac_val * n))
        for g in sel[:n_fit]:
            assign[int(g)] = "fit"
        for g in sel[n_fit:n_fit + n_val]:
            assign[int(g)] = "val"
        for g in sel[n_fit + n_val:]:
            assign[int(g)] = "test"

    out = {"fit": [], "val": [], "test": []}
    for s in obs.sensors:
        out[assign.get(int(s.group), "fit")].append(s.sid)
    return {k: np.array(v, np.int32) for k, v in out.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOTEST
# ══════════════════════════════════════════════════════════════════════════════

def _toy_ocean(nt=120, nx=64, ny=96, seed=0):
    """Champ advecté minimal, uniquement pour l'autotest de ce module."""
    rng = np.random.default_rng(seed)
    base = _smooth_noise((nx, ny), rng, sigma=6.0)
    T = np.zeros((nt, nx, ny), np.float32)
    for t in range(nt):
        T[t] = np.roll(np.roll(base, int(0.5 * t), axis=1), int(0.15 * t), axis=0)
        T[t] += 0.15 * _smooth_noise((nx, ny), rng, sigma=4.0)
    T = (T - T.mean()) / (T.std() + 1e-9)
    S = 0.7 * T + 0.3 * np.roll(T, 5, axis=2)
    return T.astype(np.float32), S.astype(np.float32)


if __name__ == "__main__":
    T, S = _toy_ocean()
    nt, nx, ny = T.shape
    net = ObsNetwork(nx=nx, ny=ny, nt=nt, rng=np.random.default_rng(7))
    net.add_moorings(n=12)
    net.add_argo(n=8)
    net.add_drifters(n=6, u=0.15, v=0.5)
    net.add_glider(waypoints=[(10, 10), (50, 80)], n_repeat=2)
    net.add_ship(waypoints=[(5, 5), (60, 90)])
    obs = net.sample(T, S)
    sp = split_sensors(obs, seed=1)
    print("\nsplit capteurs :", {k: len(v) for k, v in sp.items()})
    ser = obs.gridded_series("T")
    print("séries (n_sensors, nt) :", ser.shape,
          f"| manquants {100 * np.isnan(ser).mean():.1f} %")
    p = obs.save("/tmp/obs_test.npz")
    print("roundtrip :", len(ObsSet.load(p)) == len(obs))
