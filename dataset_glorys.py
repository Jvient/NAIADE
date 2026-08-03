"""
==============================================================================
  dataset_glorys.py — Nature run GLORYS12 (boîte PIRATA) pour NAIADE
==============================================================================

Remplace le générateur synthétique (dataset.py) par la réanalyse GLORYS12
téléchargée via copernicusmarine :

    copernicusmarine subset \
      --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \
      --variable thetao --variable so --variable zos --variable mlotst \
      --minimum-longitude -30 --maximum-longitude -10 \
      --minimum-latitude -5  --maximum-latitude 12 \
      --minimum-depth 0.4 --maximum-depth 0.6 \
      --start-datetime 2005-01-01 --end-datetime 2020-12-31 \
      -o data/ -f glorys12_pirata_surface.nc

Principes d'intégration (cf. discussion de conception) :
  1. ANOMALIES climatologiques journalières — la climatologie (cycle
     saisonnier) est calculée sur les années TRAIN uniquement, lissée
     circulairement (fenêtre 31 j), puis retirée de toute la série.
     Sans cela l'AE apprend surtout la climatologie.
  2. SPLITS PAR ANNÉES ENTIÈRES (défaut : train 2005-2016, val 2017-2018,
     test 2019-2020) — supprime la fuite temporelle du split 80/20.
  3. STATS DE NORMALISATION calculées sur le train seul, stockées dans le
     cache et réutilisées telles quelles pour val/test.
  4. MASQUE OCÉAN — les pixels terre/NaN sont exclus de l'échantillonnage
     d'observations et fournis au reste du pipeline (loss, RL, GNN).
  5. BRUITS D'OBSERVATION INDÉPENDANTS pour T et S (corrige le bruit
     partagé de dataset.py, qui corrélait artificiellement les erreurs).
  6. COORDONNÉES RÉELLES — lat/lon conservées ; helpers latlon<->indices
     et distances haversine pour le GNN et l'environnement RL.

Usage :
    # 1. Prétraitement (une fois) : NetCDF -> cache .npy memmap + meta.json
    python dataset_glorys.py --preprocess --nc data/glorys12_pirata_surface.nc

    # 2. Résumé du cache
    python dataset_glorys.py --info

    # 3. Figure diagnostique (équivalent plot_nature_run)
    python dataset_glorys.py --figures

Intégration dans les briques (exemple Brique 1) :

    from dataset_glorys import GlorysData, GlorysOEDDataset

    data     = GlorysData()                    # lit data/glorys_cache/
    train_ds = GlorysOEDDataset(data, "train", n_obs_min=10, n_obs_max=80)
    val_ds   = GlorysOEDDataset(data, "val",   n_obs_min=10, n_obs_max=80)
    # -> mêmes sorties (x, y, mask) que OceanOEDDataset : drop-in.
    # Grille : data.nlat x data.nlon (remplace config.NX x config.NY).
    # Loss : multiplier par data.ocean_torch() pour ignorer la terre.

Dépendances : xarray, netCDF4, scipy, pandas (en plus de torch/numpy).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# =============================================================================
#  Constantes par défaut
# =============================================================================

DEFAULT_NC    = "data/glorys12_pirata_surface.nc"
DEFAULT_CACHE = "data/glorys_cache"

# Nom NAIADE -> nom GLORYS. T et S obligatoires, le reste optionnel.
VAR_MAP = {"T": "thetao", "S": "so", "Z": "zos", "MLD": "mlotst"}
REQUIRED_VARS = ("T", "S")

# Splits par années entières (bornes incluses)
DEFAULT_SPLITS = {"train": (2005, 2016), "val": (2017, 2018), "test": (2019, 2020)}

CLIM_WINDOW = 31          # lissage circulaire de la climatologie (jours)

# Bruit d'observation OSSE en unités physiques (modifiable par brique)
OBS_NOISE_PHYS = {"T": 0.10,    # °C  (précision mouillage + représentativité)
                  "S": 0.02}    # psu

# Positions nominales de mouillages PIRATA dans/près de la boîte.
# ATTENTION : positions NOMINALES de déploiement — à vérifier/compléter
# depuis les métadonnées GTMBA/PMEL (https://www.pmel.noaa.gov/gtmba/)
# avant toute utilisation quantitative (scoring LOO du réseau réel).
PIRATA_NOMINAL = {
    "0N23W":  (0.0,  -23.0),
    "0N10W":  (0.0,  -10.0),
    # Hors boîte par défaut (30W-10W, 5S-12N) mais utiles si élargie :
    "0N35W":  (0.0,  -35.0),
    "0N0E":   (0.0,    0.0),
    "6S10W":  (-6.0, -10.0),
    "4N38W":  (4.0,  -38.0),
    "8N38W":  (8.0,  -38.0),
    "12N38W": (12.0, -38.0),
}


# =============================================================================
#  Prétraitement NetCDF -> cache
# =============================================================================

def _smooth_climatology(clim, window=CLIM_WINDOW):
    """Lissage circulaire (sur le jour de l'année) robuste aux NaN.

    clim : (366, nlat, nlon) — moyenne brute par jour de l'année.
    Le jour 366 n'est alimenté que par les années bissextiles : le lissage
    circulaire comble ce déficit d'échantillonnage.
    """
    from scipy.ndimage import uniform_filter1d
    pad = window // 2
    ext = np.concatenate([clim[-pad:], clim, clim[:pad]], axis=0)
    finite = np.isfinite(ext)
    num = uniform_filter1d(np.where(finite, ext, 0.0), size=window,
                           axis=0, mode="nearest")
    den = uniform_filter1d(finite.astype(np.float32), size=window,
                           axis=0, mode="nearest")
    sm = num / np.maximum(den, 1e-6)
    sm[den < 1e-6] = np.nan
    return sm[pad:pad + 366]


def _build_climatology(arr, doy_idx, train_sel):
    """Climatologie journalière lissée, calculée sur le train uniquement.

    arr       : (nt, nlat, nlon) champ brut (NaN sur terre)
    doy_idx   : (nt,) jour de l'année - 1  (0..365)
    train_sel : (nt,) booléen — pas de temps appartenant au train
    """
    import warnings
    _, nlat, nlon = arr.shape
    clim = np.full((366, nlat, nlon), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        # nanmean sur les pixels 100 % NaN (terre) -> warning attendu
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for d in range(366):
            sel = train_sel & (doy_idx == d)
            if sel.any():
                clim[d] = np.nanmean(arr[sel], axis=0)
    return _smooth_climatology(clim).astype(np.float32)


def preprocess(nc_path=DEFAULT_NC, cache_dir=DEFAULT_CACHE,
               splits=None, detrend=False, variables=None):
    """NetCDF GLORYS -> cache : anomalies float32 (.npy memmap-ables),
    climatologies, masque océan, coordonnées, stats de normalisation.

    detrend : si True, retire aussi la tendance linéaire par pixel
              (ajustée sur le train). Utile si le réchauffement 2005-2020
              domine la variance ; désactivé par défaut pour garder la
              variabilité interannuelle basse fréquence.
    """
    import pandas as pd
    import xarray as xr

    splits = splits or DEFAULT_SPLITS
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    print(f"  Ouverture : {nc_path}")
    ds = xr.open_dataset(nc_path)

    # Noms de coordonnées (copernicusmarine : latitude/longitude/time/depth)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat = ds[lat_name].values.astype(np.float32)
    lon = ds[lon_name].values.astype(np.float32)
    times = pd.to_datetime(ds["time"].values)
    years = times.year.values.astype(np.int32)
    doy_idx = (times.dayofyear.values - 1).astype(np.int32)

    train_sel = (years >= splits["train"][0]) & (years <= splits["train"][1])
    if not train_sel.any():
        raise ValueError(f"Aucun pas de temps train dans {splits['train']} — "
                         f"années disponibles : {years.min()}–{years.max()}")

    keep = variables or [k for k in VAR_MAP if VAR_MAP[k] in ds.data_vars]
    missing = [v for v in REQUIRED_VARS if v not in keep]
    if missing:
        raise ValueError(f"Variables obligatoires absentes du NetCDF : {missing}")
    print(f"  Variables : {keep}  |  {len(times)} jours "
          f"({years.min()}–{years.max()})  |  grille {len(lat)}x{len(lon)}")

    meta = {
        "source": str(nc_path),
        "variables": keep,
        "splits": {k: list(v) for k, v in splits.items()},
        "detrend": bool(detrend),
        "clim_window_days": CLIM_WINDOW,
        "nlat": int(len(lat)), "nlon": int(len(lon)), "nt": int(len(times)),
        "norm": {},
    }

    ocean_mask = None
    for key in keep:
        var = VAR_MAP[key]
        print(f"  [{key}] chargement de '{var}'...")
        da = ds[var]
        if "depth" in da.dims:                       # thetao/so : niveau 0.49 m
            da = da.isel(depth=0)
        arr = da.transpose("time", lat_name, lon_name).values.astype(np.float32)

        # Masque océan : pixels finis sur toute la série, commun aux variables
        var_ocean = np.isfinite(arr).all(axis=0)
        ocean_mask = var_ocean if ocean_mask is None else (ocean_mask & var_ocean)

        print(f"  [{key}] climatologie journalière (train "
              f"{splits['train'][0]}–{splits['train'][1]}, lissage {CLIM_WINDOW} j)...")
        clim = _build_climatology(arr, doy_idx, train_sel)
        np.save(cache / f"clim_{key}.npy", clim)

        anom = arr - clim[doy_idx]

        if detrend:
            t_axis = np.arange(len(times), dtype=np.float32)
            tc = t_axis - t_axis[train_sel].mean()
            num = np.tensordot(tc[train_sel], anom[train_sel], axes=(0, 0))
            den = float((tc[train_sel] ** 2).sum())
            slope = (num / den).astype(np.float32)          # (nlat, nlon)
            anom = anom - tc[:, None, None] * slope[None]
            np.save(cache / f"trend_{key}.npy", slope)

        # Stats de normalisation : train + océan uniquement
        tr = anom[train_sel][:, var_ocean]
        mean, std = float(np.nanmean(tr)), float(np.nanstd(tr))
        meta["norm"][key] = {"mean": mean, "std": std}
        print(f"  [{key}] anomalies train : mean={mean:+.4f}  std={std:.4f}")

        # Écriture memmap (terre -> 0 pour éviter les NaN dans les tenseurs)
        anom = np.nan_to_num(anom, nan=0.0)
        mm = np.lib.format.open_memmap(
            cache / f"{key}_anom.npy", mode="w+",
            dtype=np.float32, shape=anom.shape)
        mm[:] = anom
        mm.flush()
        del arr, anom, mm

    np.save(cache / "ocean_mask.npy", ocean_mask)
    np.save(cache / "lat.npy", lat)
    np.save(cache / "lon.npy", lon)
    np.save(cache / "years.npy", years)
    np.save(cache / "doy_idx.npy", doy_idx)
    np.save(cache / "time.npy", ds["time"].values)  # datetime64

    meta["ocean_fraction"] = float(ocean_mask.mean())
    with open(cache / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    ds.close()

    print(f"  Fraction océan : {100 * meta['ocean_fraction']:.1f} %")
    print(f"  Cache écrit -> {cache}/")
    return meta


# =============================================================================
#  Accès aux données prétraitées
# =============================================================================

class GlorysData:
    """Accès memmap au cache GLORYS + helpers géographiques.

    Attributs principaux :
        nlat, nlon         : dimensions de la grille (remplacent NX, NY)
        lat, lon           : coordonnées 1D (degrés)
        ocean              : (nlat, nlon) bool — masque océan
        norm               : {"T": {"mean","std"}, ...} stats train
        anomalies["T"|...] : memmap (nt, nlat, nlon), anomalies NON normalisées
    """

    def __init__(self, cache_dir=DEFAULT_CACHE):
        self.cache = Path(cache_dir)
        if not (self.cache / "meta.json").exists():
            raise FileNotFoundError(
                f"Cache absent : {self.cache}/meta.json — lancer d'abord "
                f"`python dataset_glorys.py --preprocess`")
        with open(self.cache / "meta.json") as f:
            self.meta = json.load(f)

        self.variables = self.meta["variables"]
        self.norm      = self.meta["norm"]
        self.splits    = {k: tuple(v) for k, v in self.meta["splits"].items()}
        self.lat   = np.load(self.cache / "lat.npy")
        self.lon   = np.load(self.cache / "lon.npy")
        self.years = np.load(self.cache / "years.npy")
        self.time  = np.load(self.cache / "time.npy")
        self.ocean = np.load(self.cache / "ocean_mask.npy")
        self.nlat, self.nlon = self.ocean.shape

        self.anomalies = {
            key: np.load(self.cache / f"{key}_anom.npy", mmap_mode="r")
            for key in self.variables}

    # -- splits ---------------------------------------------------------------
    def split_indices(self, split):
        y0, y1 = self.splits[split]
        return np.where((self.years >= y0) & (self.years <= y1))[0]

    def get_arrays(self, split, variables=("T", "S"), normalized=True, step=1):
        """Charge en RAM les anomalies d'un split, ordre (nt, nlat, nlon).

        step : sous-échantillonnage temporel (ex. step=5 pour la matrice de
               corrélation du GNN ou les stats de l'environnement RL).
        """
        idx = self.split_indices(split)[::step]
        out = []
        for key in variables:
            a = np.asarray(self.anomalies[key][idx], dtype=np.float32)
            if normalized:
                n = self.norm[key]
                a = (a - n["mean"]) / (n["std"] + 1e-9)
                a *= self.ocean[None]          # terre -> 0 exactement
            out.append(a)
        return out if len(out) > 1 else out[0]

    def ocean_torch(self, device="cpu"):
        """Masque océan (1, 1, nlat, nlon) à multiplier dans la loss AE."""
        return torch.from_numpy(
            self.ocean.astype(np.float32))[None, None].to(device)

    def denormalize(self, arr, key):
        n = self.norm[key]
        return arr * (n["std"] + 1e-9) + n["mean"]

    def climatology(self, key):
        return np.load(self.cache / f"clim_{key}.npy")

    # -- géographie -----------------------------------------------------------
    def latlon_to_ij(self, lat, lon, require_ocean=True):
        """(lat, lon) -> indices (i, j) du pixel le plus proche.
        Si require_ocean, renvoie le pixel océan le plus proche."""
        i = int(np.abs(self.lat - lat).argmin())
        j = int(np.abs(self.lon - lon).argmin())
        if require_ocean and not self.ocean[i, j]:
            ii, jj = np.where(self.ocean)
            d2 = (self.lat[ii] - lat) ** 2 + (self.lon[jj] - lon) ** 2
            k = int(d2.argmin())
            i, j = int(ii[k]), int(jj[k])
        return i, j

    def ij_to_latlon(self, i, j):
        return float(self.lat[i]), float(self.lon[j])

    def distance_km(self, ij_a, ij_b):
        """Distance haversine (km) entre deux pixels — pour arêtes GNN / RL."""
        la1, lo1 = self.ij_to_latlon(*ij_a)
        la2, lo2 = self.ij_to_latlon(*ij_b)
        la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
        a = (np.sin((la2 - la1) / 2) ** 2
             + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
        return float(2 * 6371.0 * np.arcsin(np.sqrt(a)))

    def pirata_positions(self, in_box_only=True):
        """Positions (i, j) des mouillages PIRATA nominaux (cf. PIRATA_NOMINAL,
        à vérifier sur GTMBA/PMEL avant usage quantitatif)."""
        out = {}
        for name, (la, lo) in PIRATA_NOMINAL.items():
            if in_box_only and not (self.lat.min() <= la <= self.lat.max()
                                    and self.lon.min() <= lo <= self.lon.max()):
                continue
            out[name] = self.latlon_to_ij(la, lo)
        return out

    def summary(self):
        y = self.years
        lines = [
            f"Grille   : {self.nlat} lat x {self.nlon} lon  "
            f"({self.lat.min():.2f}..{self.lat.max():.2f}N, "
            f"{self.lon.min():.2f}..{self.lon.max():.2f}E)",
            f"Temps    : {len(y)} jours  ({y.min()}–{y.max()})",
            f"Océan    : {100 * self.ocean.mean():.1f} % des pixels",
            f"Variables: {self.variables}",
        ]
        for k, (y0, y1) in self.splits.items():
            lines.append(f"  split {k:<5}: {y0}–{y1}  "
                         f"({len(self.split_indices(k))} jours)")
        for k in self.variables:
            n = self.norm[k]
            lines.append(f"  norm  {k:<5}: mean={n['mean']:+.4f}  std={n['std']:.4f}")
        return "\n".join(lines)


# =============================================================================
#  Dataset PyTorch — interface identique à OceanOEDDataset
# =============================================================================

class GlorysOEDDataset(Dataset):
    """Drop-in pour OceanOEDDataset (Brique 1), sur anomalies GLORYS.

    __getitem__ -> (x, y, mask) :
        x    : (3, nlat, nlon)  [T_obs, S_obs, mask]   (obs = anomalie + bruit)
        y    : (2, nlat, nlon)  [T, S] anomalies normalisées (0 sur terre)
        mask : (1, nlat, nlon)

    Différences avec la version synthétique :
      - normalisation par les stats TRAIN du cache (pas du split courant) ;
      - observations tirées uniquement sur les pixels océan ;
      - bruits d'observation indépendants T / S, en unités physiques ;
      - pas d'augmentation par flip par défaut (l'asymétrie beta-plane /
        ITCZ rend les flips physiquement injustifiés sur données réelles).
    """

    def __init__(self, data: GlorysData, split="train",
                 n_obs_min=10, n_obs_max=80,
                 obs_noise_T=OBS_NOISE_PHYS["T"],
                 obs_noise_S=OBS_NOISE_PHYS["S"],
                 step=1):
        self.data  = data
        self.split = split
        self.idx   = data.split_indices(split)[::step]
        self.n_obs_min, self.n_obs_max = n_obs_min, n_obs_max

        self.nlat, self.nlon = data.nlat, data.nlon
        self.ocean = data.ocean
        self.ocean_flat = np.where(self.ocean.ravel())[0]

        # Bruit physique -> unités normalisées
        self.ns_T = obs_noise_T / (data.norm["T"]["std"] + 1e-9)
        self.ns_S = obs_noise_S / (data.norm["S"]["std"] + 1e-9)
        self._nT = data.norm["T"]
        self._nS = data.norm["S"]

    def __len__(self):
        return len(self.idx)

    def _random_mask(self, n_obs):
        flat = np.zeros(self.nlat * self.nlon, dtype=np.float32)
        pick = np.random.choice(self.ocean_flat, n_obs, replace=False)
        flat[pick] = 1.0
        return flat.reshape(self.nlat, self.nlon)

    def _load_norm(self, t):
        T = np.asarray(self.data.anomalies["T"][t], dtype=np.float32)
        S = np.asarray(self.data.anomalies["S"][t], dtype=np.float32)
        T = (T - self._nT["mean"]) / (self._nT["std"] + 1e-9) * self.ocean
        S = (S - self._nS["mean"]) / (self._nS["std"] + 1e-9) * self.ocean
        return T, S

    def __getitem__(self, k):
        t = int(self.idx[k])
        T, S = self._load_norm(t)
        n_obs = np.random.randint(self.n_obs_min, self.n_obs_max + 1)
        mask  = self._random_mask(n_obs)

        # Bruits indépendants par variable (corrige le bruit partagé de
        # dataset.py qui corrélait artificiellement les erreurs T/S)
        eT = np.random.randn(self.nlat, self.nlon).astype(np.float32) * self.ns_T
        eS = np.random.randn(self.nlat, self.nlon).astype(np.float32) * self.ns_S

        x = np.stack([(T + eT) * mask, (S + eS) * mask, mask])
        y = np.stack([T, S])
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(mask[None]))


def build_glorys_datasets(cache_dir=DEFAULT_CACHE, **kwargs):
    """Miroir de build_datasets() : (train_ds, val_ds) + l'objet GlorysData."""
    data = GlorysData(cache_dir)
    return (GlorysOEDDataset(data, "train", **kwargs),
            GlorysOEDDataset(data, "val", **kwargs),
            data)


# =============================================================================
#  Helpers d'intégration NAIADE (briques 1/2/3, run_demo)
# =============================================================================

def identity_norm(data: GlorysData):
    """Dict `norm` pour les briques en mode GLORYS.

    Les champs fournis aux briques (get_arrays / GlorysOEDDataset) sont DÉJÀ
    des anomalies normalisées par les stats train : la normalisation aval doit
    être l'identité. On y adjoint le bruit d'observation en unités normalisées
    (obs_ns_*) et les stats physiques pour la dénormalisation des figures.
    """
    return {
        "T_mean": 0.0, "T_std": 1.0, "S_mean": 0.0, "S_std": 1.0,
        "obs_ns_T": OBS_NOISE_PHYS["T"] / (data.norm["T"]["std"] + 1e-9),
        "obs_ns_S": OBS_NOISE_PHYS["S"] / (data.norm["S"]["std"] + 1e-9),
        "phys": {k: dict(v) for k, v in data.norm.items()},
    }


def sample_ocean_positions(ocean, n, rng=None, seed=None):
    """n positions (i, j) aléatoires sur les pixels océan uniquement."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    flat = np.where(np.asarray(ocean).ravel() > 0.5)[0]
    ncols = ocean.shape[1]
    pick = rng.choice(flat, size=n, replace=False)
    return [(int(k // ncols), int(k % ncols)) for k in pick]


# =============================================================================
#  Figure diagnostique (équivalent plot_nature_run)
# =============================================================================

def plot_glorys_summary(data: GlorysData, out_path="outputs/glorys_nature_run.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()]
    land = np.where(data.ocean, np.nan, 1.0)

    clim_T = data.climatology("T")
    T_val  = data.get_arrays("val", variables=("T",), normalized=False, step=2)
    S_val  = data.get_arrays("val", variables=("S",), normalized=False, step=2)

    fig, axes = plt.subplots(2, 3, figsize=(17, 8), facecolor="#0a1628")
    fig.subplots_adjust(hspace=0.35, wspace=0.30, left=0.05, right=0.97,
                        top=0.90, bottom=0.07)

    def styled(ax, im, title, label):
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a4a7a")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white", labelsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(label, color="white", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
        ax.imshow(land, extent=ext, origin="lower", cmap="Greys",
                  vmin=0, vmax=1.5, aspect="auto", zorder=3)

    with np.errstate(all="ignore"):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            clim_T_mean = np.nanmean(clim_T, axis=0)
    im = axes[0, 0].imshow(clim_T_mean, extent=ext,
                           origin="lower", cmap="RdYlBu_r", aspect="auto")
    styled(axes[0, 0], im, "SST — climatologie annuelle (train)", "°C")

    im = axes[0, 1].imshow(T_val.std(axis=0), extent=ext, origin="lower",
                           cmap="plasma", aspect="auto")
    styled(axes[0, 1], im, "Anomalie SST — sigma (val)", "°C")

    snap = len(T_val) // 2
    lim = np.nanstd(T_val) * 3
    im = axes[0, 2].imshow(T_val[snap], extent=ext, origin="lower",
                           cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    styled(axes[0, 2], im, "Anomalie SST — instantané (val)", "°C")

    im = axes[1, 0].imshow(S_val.std(axis=0), extent=ext, origin="lower",
                           cmap="viridis", aspect="auto")
    styled(axes[1, 0], im, "Anomalie SSS — sigma (val)", "psu")

    # Série temporelle au mouillage 0N23W (ou centre de boîte)
    ax = axes[1, 1]
    ax.set_facecolor("#050d1a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a4a7a")
    pirata = data.pirata_positions()
    if pirata:
        name, (i, j) = next(iter(pirata.items()))
    else:
        name, (i, j) = "centre", (data.nlat // 2, data.nlon // 2)
    idx = data.split_indices("val")
    ts = np.asarray(data.anomalies["T"][idx, i, j])
    ax.plot(ts, color="#fc8d59", lw=0.8)
    ax.set_title(f"Anomalie SST @ {name}", color="white",
                 fontsize=10, fontweight="bold")
    ax.tick_params(colors="white", labelsize=7)
    ax.grid(alpha=0.2, color="white")

    im = axes[1, 2].imshow(data.ocean, extent=ext, origin="lower",
                           cmap="Blues", aspect="auto")
    for nm, (pi, pj) in pirata.items():
        la, lo = data.ij_to_latlon(pi, pj)
        axes[1, 2].plot(lo, la, "o", color="#ffd93d", ms=7,
                        mec="black", zorder=5)
        axes[1, 2].annotate(nm, (lo, la), color="white", fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
    styled(axes[1, 2], im, "Masque océan + mouillages PIRATA nominaux", "")

    fig.suptitle("Nature run GLORYS12 — boîte PIRATA — anomalies climatologiques",
                 color="white", fontsize=13, fontweight="bold")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor="#0a1628", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {out_path}")


# =============================================================================
#  CLI
# =============================================================================

def _parse_years(txt):
    a, b = txt.split("-")
    return int(a), int(b)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GLORYS12 -> NAIADE")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--info",       action="store_true")
    p.add_argument("--figures",    action="store_true")
    p.add_argument("--nc",     type=str, default=DEFAULT_NC)
    p.add_argument("--cache",  type=str, default=DEFAULT_CACHE)
    p.add_argument("--detrend", action="store_true",
                   help="Retire la tendance linéaire par pixel (fit train)")
    p.add_argument("--train_years", type=str, default="2005-2016")
    p.add_argument("--val_years",   type=str, default="2017-2018")
    p.add_argument("--test_years",  type=str, default="2019-2020")
    p.add_argument("--out", type=str, default="outputs/glorys_nature_run.png")
    args = p.parse_args()

    if not any([args.preprocess, args.info, args.figures]):
        p.print_help()
        raise SystemExit(0)

    if args.preprocess:
        splits = {"train": _parse_years(args.train_years),
                  "val":   _parse_years(args.val_years),
                  "test":  _parse_years(args.test_years)}
        preprocess(args.nc, args.cache, splits=splits, detrend=args.detrend)

    if args.info or args.figures:
        data = GlorysData(args.cache)
        print(data.summary())
        if args.figures:
            plot_glorys_summary(data, out_path=args.out)
