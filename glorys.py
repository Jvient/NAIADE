r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NAIADE — Chargeur GLORYS12V1 (CMEMS) — multi-variables / multi-niveaux      ║
║                                                                              ║
║  Configuration courante : golfe de Gascogne, fenêtre 100 % océanique.        ║
║      variables      : thetao, so, uo, vo                                     ║
║      niveaux        : 0 (~0.49 m) et 1 (~1.54 m)                             ║
║      → 8 canaux physiques                                                    ║
║                                                                              ║
║  Sortie principale :                                                          ║
║      fields    (nt, n_ch, nx, ny) float32                                    ║
║      channels  ['thetao_z0', 'thetao_z1', 'so_z0', ..., 'vo_z1']             ║
║                                                                              ║
║  Convention d'axes NAIADE :                                                   ║
║      x ←→ longitude (0 = ouest)   |   y ←→ latitude (0 = sud)                ║
║  Les fichiers CMEMS sont (time, depth, latitude, longitude) → transposés.    ║
║                                                                              ║
║  Usage :                                                                      ║
║      # 1. trouver la plus grande fenêtre sans terre                           ║
║      python -m data.glorys --find-box data/raw/glorys_gascogne                ║
║                                                                              ║
║      # 2. inspecter la configuration retenue                                  ║
║      python -m data.glorys --probe data/raw/glorys_gascogne \                 ║
║             --lon -10 -4.5 --lat 43.5 47.0 --cache data/cache                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:  # pragma: no cover
    XARRAY_AVAILABLE = False


# Variables CMEMS par défaut et leurs unités (pour les rapports et les figures)
DEFAULT_VARIABLES = ("thetao", "so", "uo", "vo")
VAR_UNITS = {"thetao": "°C", "so": "PSU", "uo": "m/s", "vo": "m/s", "zos": "m"}
VAR_LABELS = {"thetao": "Température", "so": "Salinité",
              "uo": "Courant zonal", "vo": "Courant méridien",
              "zos": "Hauteur de mer"}


# ══════════════════════════════════════════════════════════════════════════════
#  Conteneur de domaine
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GlorysDomain:
    """
    Champs GLORYS12 multi-variables prêts pour NAIADE.

    Attributs
    ---------
    fields    : (nt, n_ch, nx, ny) float32 — tous les canaux physiques.
    channels  : list[str] — noms des canaux, ex. 'thetao_z0'.
    sea_mask  : (nx, ny) bool — True = océan.
    lon, lat  : coordonnées géographiques des axes x et y.
    times     : (nt,) datetime64.
    depths    : (n_depth,) float32 — profondeurs réelles des niveaux retenus.
    meta      : dict — provenance et traitements.
    """
    fields: np.ndarray
    channels: list
    sea_mask: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    times: np.ndarray
    depths: np.ndarray
    meta: dict = field(default_factory=dict)

    # ── Dimensions ───────────────────────────────────────────────────────────
    @property
    def nt(self) -> int:
        return self.fields.shape[0]

    @property
    def n_ch(self) -> int:
        return self.fields.shape[1]

    @property
    def nx(self) -> int:
        return self.fields.shape[2]

    @property
    def ny(self) -> int:
        return self.fields.shape[3]

    @property
    def n_sea(self) -> int:
        return int(self.sea_mask.sum())

    @property
    def sea_fraction(self) -> float:
        return float(self.sea_mask.mean())

    @property
    def is_full_sea(self) -> bool:
        return bool(self.sea_mask.all())

    # ── Accès aux canaux ─────────────────────────────────────────────────────
    def channel(self, name) -> np.ndarray:
        """Renvoie le champ (nt, nx, ny) du canal `name`."""
        if name not in self.channels:
            raise KeyError(f"Canal '{name}' absent. Disponibles : {self.channels}")
        return self.fields[:, self.channels.index(name)]

    @property
    def T(self) -> np.ndarray:
        """Rétro-compatibilité : SST au niveau le plus proche de la surface."""
        return self.channel(self._first("thetao"))

    @property
    def S(self) -> np.ndarray:
        """Rétro-compatibilité : SSS au niveau le plus proche de la surface."""
        return self.channel(self._first("so"))

    def _first(self, var):
        for c in self.channels:
            if c.startswith(var + "_"):
                return c
        raise KeyError(f"Variable '{var}' absente. Canaux : {self.channels}")

    # ── Utilitaires géographiques ────────────────────────────────────────────
    def sea_indices(self) -> np.ndarray:
        xs, ys = np.where(self.sea_mask)
        return np.stack([xs, ys], axis=1)

    def sample_sea_positions(self, n, rng=None, min_dist=0):
        """Tire n positions (x, y) en mer, avec espacement minimal optionnel."""
        rng = np.random.default_rng(rng)
        idx = self.sea_indices()
        if len(idx) < n:
            raise ValueError(f"Seulement {len(idx)} pixels mer pour {n} capteurs.")
        if min_dist <= 0:
            sel = rng.choice(len(idx), n, replace=False)
            return [tuple(int(v) for v in idx[i]) for i in sel]

        chosen = []
        for i in rng.permutation(len(idx)):
            p = idx[i]
            if all((p[0]-q[0])**2 + (p[1]-q[1])**2 >= min_dist**2 for q in chosen):
                chosen.append(p)
                if len(chosen) == n:
                    break
        if len(chosen) < n:
            raise ValueError(f"min_dist={min_dist} trop grand : "
                             f"{len(chosen)}/{n} positions trouvées.")
        return [tuple(int(v) for v in p) for p in chosen]

    def pixel_to_lonlat(self, x, y):
        return float(self.lon[x]), float(self.lat[y])

    def lonlat_to_pixel(self, lon, lat):
        return (int(np.abs(self.lon - lon).argmin()),
                int(np.abs(self.lat - lat).argmin()))

    # ── Diagnostics ──────────────────────────────────────────────────────────
    def channel_stats(self) -> dict:
        """Moyenne / écart-type par canal, calculés sur la mer uniquement."""
        sm = self.sea_mask
        return {c: (float(self.fields[:, i][:, sm].mean()),
                    float(self.fields[:, i][:, sm].std()))
                for i, c in enumerate(self.channels)}

    def level_redundancy(self) -> dict:
        """
        Corrélation entre niveaux verticaux successifs, par variable.

        DIAGNOSTIC IMPORTANT : les niveaux GLORYS 0 (0.49 m) et 1 (1.54 m) sont
        séparés de ~1 m, donc systématiquement dans la couche de mélange. Une
        corrélation > 0.99 signifie que le second niveau n'apporte quasiment
        aucune information et ne fait que doubler le coût de calcul.
        """
        out = {}
        by_var = {}
        for i, c in enumerate(self.channels):
            var, z = c.rsplit("_z", 1)
            by_var.setdefault(var, []).append((int(z), i))
        for var, lst in by_var.items():
            lst.sort()
            for (za, ia), (zb, ib) in zip(lst[:-1], lst[1:]):
                a = self.fields[:, ia][:, self.sea_mask].ravel()
                b = self.fields[:, ib][:, self.sea_mask].ravel()
                if a.std() < 1e-12 or b.std() < 1e-12:
                    out[f"{var}: z{za}↔z{zb}"] = float("nan")   # canal constant
                else:
                    out[f"{var}: z{za}↔z{zb}"] = float(np.corrcoef(a, b)[0, 1])
        return out

    def degenerate_channels(self, tol=1e-10) -> list:
        """Canaux de variance quasi nulle — inutiles et sources de NaN une fois
        normalisés. À vérifier systématiquement après un changement de domaine."""
        sm = self.sea_mask
        return [c for i, c in enumerate(self.channels)
                if float(self.fields[:, i][:, sm].std()) < tol]

    def summary(self) -> str:
        km = self.meta.get("dx_km", float("nan"))
        stats = self.channel_stats()
        lines = [
            f"GLORYS12 | {self.nt} pas de temps | grille {self.nx}×{self.ny} "
            f"(~{km:.1f} km/px)",
            f"  lon [{self.lon[0]:.3f}, {self.lon[-1]:.3f}]  "
            f"lat [{self.lat[0]:.3f}, {self.lat[-1]:.3f}]",
            f"  niveaux : {', '.join(f'{d:.3f} m' for d in self.depths)}",
            f"  mer : {self.n_sea}/{self.nx*self.ny} px "
            f"({100*self.sea_fraction:.2f} %)"
            + ("  ✓ FENÊTRE 100 % OCÉANIQUE" if self.is_full_sea
               else "  ⚠ contient de la terre"),
            f"  désaisonnalisé : {self.meta.get('remove_seasonal', False)}",
            f"  {self.n_ch} canaux :",
        ]
        for c in self.channels:
            var = c.rsplit("_z", 1)[0]
            m, s = stats[c]
            lines.append(f"    {c:<12} {VAR_LABELS.get(var, var):<16} "
                         f"moy={m:>8.3f}  std={s:>7.3f}  [{VAR_UNITS.get(var, '')}]")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Recherche de la plus grande fenêtre sans terre
# ══════════════════════════════════════════════════════════════════════════════

def largest_sea_rectangle(sea_mask):
    """
    Plus grand rectangle entièrement océanique dans `sea_mask`.

    Algorithme classique du « plus grand rectangle dans un histogramme »,
    appliqué ligne par ligne. Complexité O(nx · ny).

    Retour
    ------
    (x0, x1, y0, y1) : bornes INCLUSIVES en indices pixel.
    """
    sea = np.asarray(sea_mask, dtype=bool)
    nx, ny = sea.shape
    heights = np.zeros(ny, dtype=int)
    best = (0, -1, 0, -1)
    best_area = 0

    for x in range(nx):
        heights = np.where(sea[x], heights + 1, 0)

        # Plus grand rectangle dans l'histogramme `heights`
        stack = []          # indices de colonnes à hauteur croissante
        for y in range(ny + 1):
            h = heights[y] if y < ny else 0
            start = y
            while stack and stack[-1][1] >= h:
                sy, sh = stack.pop()
                area = sh * (y - sy)
                if area > best_area and sh > 0:
                    best_area = area
                    best = (x - sh + 1, x, sy, y - 1)
                start = sy
            stack.append((start, h))

    return best


def suggest_sea_boxes(sea_mask, lon, lat, n=3, shrink=0.85):
    """
    Propose plusieurs fenêtres océaniques : la maximale, puis des variantes
    resserrées (utile pour garder une marge par rapport au talus continental).
    """
    x0, x1, y0, y1 = largest_sea_rectangle(sea_mask)
    boxes = []
    for k in range(n):
        f = shrink ** k
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) * f / 2, (y1 - y0) * f / 2
        a, b = int(round(cx - hw)), int(round(cx + hw))
        c, d = int(round(cy - hh)), int(round(cy + hh))
        boxes.append(dict(
            px=(a, b, c, d),
            shape=(b - a + 1, d - c + 1),
            lon=(float(lon[a]), float(lon[b])),
            lat=(float(lat[c]), float(lat[d])),
            all_sea=bool(sea_mask[a:b+1, c:d+1].all()),
        ))
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
#  Chargement
# ══════════════════════════════════════════════════════════════════════════════

def _list_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .nc trouvé dans {path}")
    return files


def _cache_key(path, kwargs) -> str:
    files = sorted(str(p.name) + str(p.stat().st_size) for p in _list_files(path))
    payload = json.dumps({"files": files, "opts": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _open_files(files, preprocess, verbose=True):
    """
    Ouvre les .nc en appliquant `preprocess` À CHAQUE FICHIER avant
    concaténation : sinon on charge les 50 niveaux × 5 variables de chaque
    journalier (plusieurs Go sur une série longue).
    """
    if len(files) == 1:
        with xr.open_dataset(files[0], decode_times=True) as d:
            return preprocess(d).load()
    try:
        import dask  # noqa: F401
        return xr.open_mfdataset(files, combine="by_coords", parallel=False,
                                 decode_times=True, preprocess=preprocess)
    except ImportError:
        if verbose:
            print(f"  [glorys] dask absent → lecture séquentielle de "
                  f"{len(files)} fichiers (pip install dask pour du lazy-loading)")
        parts = []
        for i, f in enumerate(files):
            with xr.open_dataset(f, decode_times=True) as d:
                parts.append(preprocess(d).load())
            if verbose and (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(files)}")
        ds = xr.concat(parts, dim="time", data_vars="minimal",
                       coords="minimal", compat="override")
        return ds.sortby("time")


def load_glorys(path,
                variables=DEFAULT_VARIABLES,
                depth_indices=(0, 1),
                lon_range=None,
                lat_range=None,
                time_range=None,
                coarsen=1,
                grid_multiple=None,
                remove_seasonal=False,
                seasonal_window=15,
                require_full_sea=False,
                auto_sea_box=False,
                cache=None,
                verbose=True) -> GlorysDomain:
    """
    Charge un jeu GLORYS12 multi-variables / multi-niveaux au format NAIADE.

    Paramètres
    ----------
    variables : tuple[str]
        Variables CMEMS à charger. Défaut : (thetao, so, uo, vo).
    depth_indices : tuple[int]
        Indices des niveaux verticaux. Défaut (0, 1) ≈ 0.49 m et 1.54 m.
        → n_ch = len(variables) × len(depth_indices)
    lon_range, lat_range : (min, max) | None
        Fenêtre géographique. C'est ici qu'on restreint le domaine pour
        éliminer la terre.
    require_full_sea : bool
        Lève une erreur si la fenêtre contient le moindre point de terre.
        À activer une fois la fenêtre calée — garantit qu'aucune régression
        ne réintroduira silencieusement du continent.
    auto_sea_box : bool
        Rogne automatiquement au plus grand rectangle 100 % océanique.
        Pratique pour explorer ; préférer des bornes explicites en production
        (le résultat dépend du jeu de fichiers fourni).
    cache : str | Path | None
        Répertoire de cache .npz — fortement recommandé.

    Retour
    ------
    GlorysDomain
    """
    variables = tuple(variables)
    depth_indices = tuple(int(z) for z in depth_indices)

    opts = dict(variables=variables, depth_indices=depth_indices,
                lon_range=lon_range, lat_range=lat_range, time_range=time_range,
                coarsen=coarsen, grid_multiple=grid_multiple,
                remove_seasonal=remove_seasonal,
                seasonal_window=seasonal_window, auto_sea_box=auto_sea_box)

    # ── Cache ────────────────────────────────────────────────────────────────
    cache_file = None
    if cache is not None:
        cache_dir = Path(cache); cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"glorys_{_cache_key(path, opts)}.npz"
        if cache_file.exists():
            if verbose:
                print(f"  [glorys] cache → {cache_file.name}")
            dom = _from_npz(cache_file)
            _check_full_sea(dom, require_full_sea)
            return dom

    if not XARRAY_AVAILABLE:
        raise ImportError("xarray et netcdf4 requis : pip install xarray netcdf4")

    files = _list_files(path)
    if verbose:
        print(f"  [glorys] lecture de {len(files)} fichier(s) depuis {path}")
        print(f"  [glorys] variables={list(variables)}  niveaux={list(depth_indices)}"
              f"  → {len(variables)*len(depth_indices)} canaux")

    depths_holder = {}

    def preprocess(d):
        missing = [v for v in variables if v not in d]
        if missing:
            raise KeyError(f"Variables {missing} absentes. "
                           f"Présentes : {list(d.data_vars)}")
        d = d[list(variables)]
        if lon_range is not None:
            d = d.sel(longitude=slice(*lon_range))
        if lat_range is not None:
            d = d.sel(latitude=slice(*lat_range))
        if "depth" in d.dims:
            depths_holder["depths"] = np.asarray(
                d["depth"].values[list(depth_indices)], dtype=np.float32)
            d = d.isel(depth=list(depth_indices))
        else:
            depths_holder["depths"] = np.array([0.0], dtype=np.float32)
        return d

    ds = _open_files(files, preprocess=preprocess, verbose=verbose)

    if time_range is not None:
        ds = ds.sel(time=slice(*time_range))
    ds = ds.sortby("time")

    if coarsen > 1:
        ds = ds.coarsen(latitude=coarsen, longitude=coarsen,
                        boundary="trim").mean()

    if remove_seasonal:
        ds = _remove_seasonal(ds, seasonal_window)

    # ── Empilement des canaux : (nt, n_ch, nx=lon, ny=lat) ───────────────────
    depths = depths_holder.get("depths", np.array([0.0], dtype=np.float32))
    has_depth = "depth" in ds.dims

    arrays, channels = [], []
    for var in variables:
        da = ds[var]
        if has_depth and "depth" in da.dims:
            for k in range(len(depth_indices)):
                sub = da.isel(depth=k).transpose("time", "longitude", "latitude")
                arrays.append(np.asarray(sub.values, dtype=np.float32))
                channels.append(f"{var}_z{k}")
        else:
            sub = da.transpose("time", "longitude", "latitude")
            arrays.append(np.asarray(sub.values, dtype=np.float32))
            channels.append(f"{var}_z0")

    fields = np.stack(arrays, axis=1)          # (nt, n_ch, nx, ny)
    lon = np.asarray(ds["longitude"].values, dtype=np.float32)
    lat = np.asarray(ds["latitude"].values, dtype=np.float32)
    times = np.asarray(ds["time"].values)
    ds.close()

    # ── Masque terre / mer : valide sur TOUS les canaux et TOUS les temps ────
    sea_mask = np.isfinite(fields).all(axis=(0, 1))
    if sea_mask.sum() == 0:
        raise ValueError("Aucun pixel océanique valide — vérifier le domaine.")

    # ── Rognage automatique au plus grand rectangle océanique ────────────────
    if auto_sea_box and not sea_mask.all():
        x0, x1, y0, y1 = largest_sea_rectangle(sea_mask)
        if verbose:
            print(f"  [glorys] auto_sea_box → px x[{x0}:{x1}] y[{y0}:{y1}]  "
                  f"lon[{lon[x0]:.3f}, {lon[x1]:.3f}]  lat[{lat[y0]:.3f}, {lat[y1]:.3f}]")
        fields = fields[:, :, x0:x1+1, y0:y1+1]
        sea_mask = sea_mask[x0:x1+1, y0:y1+1]
        lon, lat = lon[x0:x1+1], lat[y0:y1+1]

    # ── Rognage à un multiple de `grid_multiple` ─────────────────────────────
    # Le VAE enchaîne 4 sous-échantillonnages (bc → bc*2 → bc*4 → bc*8).
    # Si NX ou NY n'est pas divisible par 16, les tailles ne se recollent pas
    # entre l'encodeur et le décodeur : soit ConvTranspose2d renvoie une carte
    # d'un pixel de trop pour la concaténation du skip, soit la dimension
    # tombe à 1 ou 2 et le goulot latent devient dégénéré.
    if grid_multiple and grid_multiple > 1:
        m = int(grid_multiple)
        nx_t = (fields.shape[2] // m) * m
        ny_t = (fields.shape[3] // m) * m
        if nx_t < m or ny_t < m:
            raise ValueError(
                f"Domaine {fields.shape[2]}×{fields.shape[3]} trop petit pour "
                f"un multiple de {m}. Élargir la fenêtre ou réduire coarsen.")
        ox, oy = (fields.shape[2] - nx_t) // 2, (fields.shape[3] - ny_t) // 2
        if (ox, oy) != (0, 0) or (nx_t, ny_t) != fields.shape[2:]:
            if verbose:
                print(f"  [glorys] rognage centré {fields.shape[2]}×"
                      f"{fields.shape[3]} → {nx_t}×{ny_t} (multiple de {m})")
            fields = fields[:, :, ox:ox+nx_t, oy:oy+ny_t]
            sea_mask = sea_mask[ox:ox+nx_t, oy:oy+ny_t]
            lon, lat = lon[ox:ox+nx_t], lat[oy:oy+ny_t]

    # ── Remplissage éventuel de la terre résiduelle ──────────────────────────
    if not sea_mask.all():
        fields = _fill_land(fields, sea_mask)
    fields[~np.isfinite(fields)] = 0.0

    dx_km = float(np.abs(np.diff(lon)).mean() * 111.32 * np.cos(np.deg2rad(lat.mean()))) \
        if len(lon) > 1 else float("nan")

    meta = dict(source="GLORYS12V1", path=str(path), n_files=len(files),
                variables=list(variables), depth_indices=list(depth_indices),
                coarsen=coarsen, grid_multiple=grid_multiple,
                remove_seasonal=bool(remove_seasonal),
                dx_km=dx_km, lon_range=lon_range, lat_range=lat_range,
                auto_sea_box=bool(auto_sea_box))

    dom = GlorysDomain(fields=fields, channels=channels, sea_mask=sea_mask,
                       lon=lon, lat=lat, times=times, depths=depths, meta=meta)

    if cache_file is not None:
        _to_npz(dom, cache_file)
        if verbose:
            print(f"  [glorys] cache écrit → {cache_file.name}")
    if verbose:
        print("  " + dom.summary().replace("\n", "\n  "))

    _check_full_sea(dom, require_full_sea)
    return dom


def _check_full_sea(dom, require):
    if require and not dom.is_full_sea:
        n_land = dom.nx * dom.ny - dom.n_sea
        raise ValueError(
            f"\n  require_full_sea=True mais la fenêtre contient {n_land} points "
            f"de terre ({100*(1-dom.sea_fraction):.2f} %)."
            f"\n  → Lancer :  python -m data.glorys --find-box <dir>"
            f"\n     pour obtenir des bornes lon/lat entièrement océaniques.")


def _remove_seasonal(ds, window=15):
    """
    Retire la climatologie jour-de-l'année, lissée circulairement.

    ⚠ MODE D'ÉCHEC SILENCIEUX
    Si la série ne couvre qu'une seule année, chaque jour-de-l'année n'a qu'UN
    échantillon : la climatologie est alors égale à la donnée elle-même et
    l'anomalie vaut exactement zéro (au lissage près). Le champ résultant est
    du bruit, tout entraînement converge vers la moyenne, et aucune métrique
    ne le signale — on croit simplement que le modèle apprend mal.
    D'où le contrôle explicite ci-dessous.
    """
    counts = ds.groupby("time.dayofyear").count(dim="time")
    first = list(counts.data_vars)[0]
    c = np.asarray(counts[first].values)
    # Compter par jour-de-l'année, indépendamment des dimensions spatiales
    while c.ndim > 1:
        c = c.max(axis=-1)
    n_med = float(np.median(c))

    if n_med < 2:
        n_years = len(np.unique(ds["time.year"].values))
        raise ValueError(
            f"\n  remove_seasonal=True mais la climatologie ne dispose que de "
            f"{n_med:.0f} échantillon par jour-de-l'année"
            f"\n  ({n_years} année(s) de données)."
            f"\n  L'anomalie serait nulle par construction et le champ ne "
            f"contiendrait que du bruit."
            f"\n  → soit fournir au moins 3 ans de données,"
            f"\n  → soit passer GLORYS_SEASONAL = False et l'assumer "
            f"explicitement.")
    if n_med < 3:
        print(f"  [glorys] ⚠ climatologie estimée sur {n_med:.0f} échantillons "
              f"par jour seulement — anomalie bruitée. 3 ans minimum recommandés.")

    clim = ds.groupby("time.dayofyear").mean("time")
    if window > 1 and clim.sizes.get("dayofyear", 0) > window:
        clim = (clim.pad(dayofyear=window, mode="wrap")
                    .rolling(dayofyear=window, center=True, min_periods=1).mean()
                    .isel(dayofyear=slice(window, -window)))
    out = ds.groupby("time.dayofyear") - clim
    return out.drop_vars("dayofyear", errors="ignore")


def _fill_land(F, sea_mask):
    """Remplace les NaN terre par la moyenne océanique (par canal, par date)."""
    F = F.copy()
    land = ~sea_mask
    for t in range(F.shape[0]):
        for c in range(F.shape[1]):
            m = float(np.nanmean(F[t, c][sea_mask]))
            F[t, c][land] = m
    return F


def _to_npz(dom, path):
    np.savez_compressed(
        path, fields=dom.fields, sea_mask=dom.sea_mask, lon=dom.lon, lat=dom.lat,
        depths=dom.depths,
        times=dom.times.astype("datetime64[s]").astype(np.int64),
        channels=json.dumps(dom.channels), meta=json.dumps(dom.meta, default=str))


def _from_npz(path) -> GlorysDomain:
    z = np.load(path, allow_pickle=False)
    return GlorysDomain(
        fields=z["fields"], channels=json.loads(str(z["channels"])),
        sea_mask=z["sea_mask"].astype(bool), lon=z["lon"], lat=z["lat"],
        depths=z["depths"], times=z["times"].astype("datetime64[s]"),
        meta=json.loads(str(z["meta"])))


# ══════════════════════════════════════════════════════════════════════════════
#  Adaptateur — interface SyntheticOceanGenerator
# ══════════════════════════════════════════════════════════════════════════════

class GlorysOceanGenerator:
    """Drop-in replacement de SyntheticOceanGenerator (mode 2 canaux T/S)."""

    def __init__(self, path, **kwargs):
        self.domain = load_glorys(path, **kwargs)
        self.nx, self.ny = self.domain.nx, self.domain.ny
        self.sea_mask = self.domain.sea_mask

    def generate_dataset(self, nt=None, seed=None):
        """`seed` ignoré (données déterministes) ; `nt` tronque la série."""
        T, S = self.domain.T, self.domain.S
        if nt is not None and nt < len(T):
            return T[:nt], S[:nt]
        if nt is not None and nt > len(T):
            print(f"  [glorys] ⚠ nt={nt} demandé, {len(T)} pas de temps disponibles.")
        return T, S


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _cmd_find_box(args):
    """Cherche la plus grande fenêtre 100 % océanique du domaine."""
    dom = load_glorys(args.path, variables=args.variables,
                      depth_indices=(0,), cache=args.cache, verbose=False)
    print("\n" + "=" * 74)
    print("  NAIADE — recherche d'une fenêtre 100 % océanique")
    print("=" * 74)
    print(f"  Domaine source : {dom.nx}×{dom.ny} px, "
          f"mer = {100*dom.sea_fraction:.2f} %")
    print(f"  lon [{dom.lon[0]:.3f}, {dom.lon[-1]:.3f}]  "
          f"lat [{dom.lat[0]:.3f}, {dom.lat[-1]:.3f}]\n")

    boxes = suggest_sea_boxes(dom.sea_mask, dom.lon, dom.lat, n=3)
    for i, b in enumerate(boxes):
        tag = "MAXIMALE" if i == 0 else f"marge -{int((1-0.85**i)*100)} %"
        ok = "✓" if b["all_sea"] else "✗"
        print(f"  [{i}] {tag:<14} {ok} {b['shape'][0]}×{b['shape'][1]} px   "
              f"lon [{b['lon'][0]:.3f}, {b['lon'][1]:.3f}]   "
              f"lat [{b['lat'][0]:.3f}, {b['lat'][1]:.3f}]")

    b = boxes[args.pick]
    print("\n" + "-" * 74)
    print(f"  Option [{args.pick}] → à reporter dans config.py :")
    print("-" * 74)
    print(f"""
GLORYS_LON_RANGE = ({b['lon'][0]:.4f}, {b['lon'][1]:.4f})
GLORYS_LAT_RANGE = ({b['lat'][0]:.4f}, {b['lat'][1]:.4f})

NX = {b['shape'][0]}
NY = {b['shape'][1]}
""")
    print("  Vérifier ensuite avec :")
    print(f"    python -m data.glorys --probe {args.path} "
          f"--lon {b['lon'][0]:.4f} {b['lon'][1]:.4f} "
          f"--lat {b['lat'][0]:.4f} {b['lat'][1]:.4f}")
    print("=" * 74 + "\n")

    if args.plot:
        _plot_boxes(dom, boxes, args.plot)


def _cmd_probe(args):
    dom = load_glorys(args.path, variables=args.variables,
                      depth_indices=tuple(args.depths),
                      lon_range=tuple(args.lon) if args.lon else None,
                      lat_range=tuple(args.lat) if args.lat else None,
                      coarsen=args.coarsen,
                      grid_multiple=args.grid_multiple,
                      remove_seasonal=args.remove_seasonal,
                      require_full_sea=args.require_full_sea,
                      auto_sea_box=args.auto_sea_box,
                      cache=args.cache, verbose=False)

    print("\n" + "=" * 74)
    print("  NAIADE — inspection GLORYS12")
    print("=" * 74)
    print("  " + dom.summary().replace("\n", "\n  "))

    # ── Redondance verticale ─────────────────────────────────────────────────
    deg = dom.degenerate_channels()
    if deg:
        print(f"\n  ⚠ Canaux de variance quasi nulle : {deg}")
        print("    Ils produiront des NaN à la normalisation — les retirer.")

    red = dom.level_redundancy()
    if red:
        print("\n  Corrélation entre niveaux verticaux :")
        worst = 0.0
        for k, v in red.items():
            if not np.isfinite(v):
                print(f"    {k:<20} r =    n/a  (canal constant)")
                continue
            flag = "  ⚠ quasi-redondant" if abs(v) > 0.99 else ""
            worst = max(worst, abs(v))
            print(f"    {k:<20} r = {v:+.4f}{flag}")
        if worst > 0.99:
            print("\n    → Les deux niveaux portent quasiment la même information.")
            print("      Ils doublent le coût de calcul sans gain d'observabilité.")
            print("      Envisager depth_indices=(0,) ou un second niveau plus")
            print("      profond (~20–50 m) pour capturer la stratification.")

    print("\n" + "-" * 74)
    print("  À reporter dans config.py :")
    print("-" * 74)
    lon_s = (f"({dom.lon[0]:.4f}, {dom.lon[-1]:.4f})" if args.lon else "None")
    lat_s = (f"({dom.lat[0]:.4f}, {dom.lat[-1]:.4f})" if args.lat else "None")
    print(f"""
DATA_SOURCE       = "glorys"
GLORYS_DIR        = "{Path(args.path).as_posix()}"
GLORYS_VARIABLES  = {tuple(args.variables)}
GLORYS_DEPTHS     = {tuple(args.depths)}
GLORYS_LON_RANGE  = {lon_s}
GLORYS_LAT_RANGE  = {lat_s}

NX      = {dom.nx}
NY      = {dom.ny}
NT      = {dom.nt}
N_CHANNELS = {dom.n_ch}
""")
    print(f"  ⚠ n_obs_max doit rester < {dom.n_sea} (pixels disponibles)")
    try:
        from config import OBSERVED_VARS
        n_obs_ch = len([c for c in dom.channels
                        if c.rsplit("_z", 1)[0] in OBSERVED_VARS])
        print(f"  ⚠ VAE : in_ch = {n_obs_ch + 1}  /  out_ch = {dom.n_ch}")
        print(f"      ({n_obs_ch} canaux observés {tuple(OBSERVED_VARS)} + masque "
              f"→ {dom.n_ch} canaux reconstruits)")
    except Exception:
        print(f"  ⚠ VAE : in_ch = n_canaux_observés + 1  /  out_ch = {dom.n_ch}")
    print("=" * 74 + "\n")

    if args.plot:
        _quicklook(dom, args.plot)


def _plot_boxes(dom, boxes, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0a1628")
    ext = [dom.lon[0], dom.lon[-1], dom.lat[0], dom.lat[-1]]
    ax.imshow(dom.sea_mask.T.astype(float), origin="lower", extent=ext,
              cmap="Blues", vmin=0, vmax=1.4, aspect="auto")
    cols = ["#ffd93d", "#fc8d59", "#74c476"]
    for i, b in enumerate(boxes):
        (lo0, lo1), (la0, la1) = b["lon"], b["lat"]
        ax.add_patch(mpatches.Rectangle((lo0, la0), lo1-lo0, la1-la0,
                                        fill=False, lw=2.2, ec=cols[i % 3],
                                        label=f"[{i}] {b['shape'][0]}×{b['shape'][1]}"))
    ax.legend(loc="lower left", framealpha=0.3, labelcolor="white")
    ax.set_title("Fenêtres 100 % océaniques candidates", color="white",
                 fontweight="bold")
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="#0a1628", bbox_inches="tight")
    plt.close()
    print(f"  Figure → {out_path}")


def _quicklook(dom, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = dom.n_ch
    ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.8*nrow),
                             facecolor="#0a1628", squeeze=False)
    ext = [dom.lon[0], dom.lon[-1], dom.lat[0], dom.lat[-1]]
    cmaps = {"thetao": "RdYlBu_r", "so": "viridis", "uo": "RdBu_r", "vo": "RdBu_r"}
    for k, c in enumerate(dom.channels):
        ax = axes[k // ncol][k % ncol]
        var = c.rsplit("_z", 1)[0]
        D = np.where(dom.sea_mask, dom.fields[0, k], np.nan)
        im = ax.imshow(D.T, origin="lower", extent=ext,
                       cmap=cmaps.get(var, "viridis"), aspect="auto")
        ax.set_title(f"{c}  [{VAR_UNITS.get(var,'')}]", color="white",
                     fontsize=10, fontweight="bold")
        ax.tick_params(colors="white", labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)
    for k in range(n, nrow*ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"GLORYS12 — {dom.nx}×{dom.ny}, {dom.nt} dates, "
                 f"{'100 % mer' if dom.is_full_sea else 'avec terre'}",
                 color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="#0a1628", bbox_inches="tight")
    plt.close()
    print(f"  Quicklook → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Chargeur / inspecteur GLORYS12 pour NAIADE")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--probe", dest="path_probe", help="Inspecter une configuration")
    g.add_argument("--find-box", dest="path_box", help="Chercher une fenêtre sans terre")

    p.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    p.add_argument("--depths", nargs="+", type=int, default=[0, 1],
                   help="Indices des niveaux verticaux (défaut : 0 1)")
    p.add_argument("--lon", nargs=2, type=float, default=None)
    p.add_argument("--lat", nargs=2, type=float, default=None)
    p.add_argument("--coarsen", type=int, default=1)
    p.add_argument("--grid_multiple", type=int, default=16,
                   help="Rogner la grille à un multiple de N (0 = désactivé)")
    p.add_argument("--remove_seasonal", action="store_true")
    p.add_argument("--require_full_sea", action="store_true")
    p.add_argument("--auto_sea_box", action="store_true")
    p.add_argument("--pick", type=int, default=0,
                   help="Fenêtre à retenir dans --find-box (0 = maximale)")
    p.add_argument("--cache", type=str, default=None)
    p.add_argument("--plot", type=str, default=None)

    a = p.parse_args()
    if a.path_box:
        a.path = a.path_box
        _cmd_find_box(a)
    else:
        a.path = a.path_probe
        _cmd_probe(a)


if __name__ == "__main__":
    main()
