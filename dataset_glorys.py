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

# Boîte par défaut : emprise du réseau PIRATA (lat -18.9..+20.5, lon -38.0..-2.7)
# élargie d'environ 5 degrés. La marge n'est pas cosmétique : l'EVF évalue la
# reconstruction sur TOUTE la boîte, donc une boîte collée aux bouées les
# flatterait, et une boîte trop large (Atlantique entier) mesurerait surtout
# des régions qu'aucun mouillage PIRATA ne peut contraindre.
DEFAULT_BOX = {"lat": (-25.0, 26.0), "lon": (-45.0, 3.0)}

# Dégradation de la grille 1/12° par moyenne de blocs COARSEN x COARSEN.
# 2 -> 1/6° (18.5 km) : ~10 points par longueur de décorrélation (L ~ 180 km),
# cache de 3.3 Go, charge GPU 1.8x celle d'une grille 205x241 déjà validée.
# Moyenne de blocs et non sous-échantillonnage : filtre passe-bas, pas de
# repliement de la mésoéchelle sur les grandes échelles.
DEFAULT_COARSEN = 2

# Nom NAIADE -> nom GLORYS. T et S obligatoires, le reste optionnel.
VAR_MAP = {"T": "thetao", "S": "so", "Z": "zos", "MLD": "mlotst"}
REQUIRED_VARS = ("T", "S")

# Splits par années entières (bornes incluses) — jeu 2007-2019
DEFAULT_SPLITS = {"train": (2007, 2016), "val": (2017, 2018), "test": (2019, 2019)}

CLIM_WINDOW = 31          # lissage circulaire de la climatologie (jours)

# Bruit d'observation OSSE en unités physiques (modifiable par brique)
OBS_NOISE_PHYS = {"T": 0.10,    # °C  (précision mouillage + représentativité)
                  "S": 0.02}    # psu

# Réseau PIRATA — positions de campagne relevées (lat, lon).
# Source : fichier bouées du projet. Ce sont des positions RÉELLES de
# mouillages, et non plus les positions nominales arrondies utilisées
# auparavant : 0N23W nominal devient PT076 à (0.0017, -22.9883).
# PT075 est ambigu selon la campagne (parfois référencé PI287A) — le doublon
# n'est pas dupliqué ici, une seule position est retenue.
PIRATA_NOMINAL = {
    "PI289A": (  0.0000,  -2.6850),
    "PI288A": (  0.0200,  -9.8467),
    "PI280A": (-18.8517, -34.6583),
    "PI285A": (  0.0100, -34.9967),
    "PI284A": (  7.9467, -38.0300),
    "PI283A": (  4.0083, -37.9367),
    "PT077":  ( -6.0333,  -9.9983),
    "PT078":  ( -9.9067,  -9.9817),
    "PT065":  ( 20.4517, -23.1417),
    "PT068":  ( 11.4883, -22.9867),
    "PT069":  (  4.0450, -22.9867),
    "PT076":  (  0.0017, -22.9883),
    "PT070":  ( -8.0083, -30.6333),
    "PT062":  (-13.5233, -32.5967),
    "PT063":  ( 20.0250, -37.8467),
    "PT072":  ( 15.0033, -37.9917),
    "PT075":  (  2.4133,  -4.6300),
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

    fields = {}
    for key in keep:
        var = VAR_MAP[key]
        print(f"  [{key}] chargement de '{var}'...")
        da = ds[var]
        if "depth" in da.dims:                       # thetao/so : niveau 0.49 m
            da = da.isel(depth=0)
        fields[key] = da.transpose("time", lat_name, lon_name
                                   ).values.astype(np.float32)

    times64 = ds["time"].values
    ds.close()
    return write_cache(fields, lat, lon, times64, years, doy_idx,
                       cache_dir=cache, splits=splits, detrend=detrend,
                       source=str(nc_path))


def write_cache(fields, lat, lon, times64, years, doy_idx,
                cache_dir=DEFAULT_CACHE, splits=None, detrend=False,
                source="unknown"):
    """Écrit le cache NAIADE à partir de champs bruts en mémoire.

    Séparé de preprocess() pour que la fixture de test (make_fixture) emprunte
    EXACTEMENT le même chemin de code : climatologie, détrend, stats de
    normalisation, masque océan, format des .npy et de meta.json. Toute dérive
    entre le cache réel et le cache de test devient ainsi impossible.

    fields : dict {"T": (nt, nlat, nlon) float32, "S": ..., ...}
             NaN sur les pixels terre — c'est ce qui définit le masque océan.
    """
    splits = splits or DEFAULT_SPLITS
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    keep = list(fields)
    missing = [v for v in REQUIRED_VARS if v not in keep]
    if missing:
        raise ValueError(f"Variables obligatoires absentes : {missing}")

    train_sel = (years >= splits["train"][0]) & (years <= splits["train"][1])
    if not train_sel.any():
        raise ValueError(f"Aucun pas de temps train dans {splits['train']} — "
                         f"années disponibles : {years.min()}–{years.max()}")
    nt = len(years)

    meta = {
        "source": source,
        "variables": keep,
        "splits": {k: list(v) for k, v in splits.items()},
        "detrend": bool(detrend),
        "clim_window_days": CLIM_WINDOW,
        "nlat": int(len(lat)), "nlon": int(len(lon)), "nt": int(nt),
        "norm": {},
    }

    ocean_mask = None
    for key in keep:
        arr = fields[key]

        # Masque océan : pixels finis sur toute la série, commun aux variables
        var_ocean = np.isfinite(arr).all(axis=0)
        ocean_mask = var_ocean if ocean_mask is None else (ocean_mask & var_ocean)

        print(f"  [{key}] climatologie journalière (train "
              f"{splits['train'][0]}–{splits['train'][1]}, lissage {CLIM_WINDOW} j)...")
        clim = _build_climatology(arr, doy_idx, train_sel)
        np.save(cache / f"clim_{key}.npy", clim)

        anom = arr - clim[doy_idx]

        if detrend:
            t_axis = np.arange(nt, dtype=np.float32)
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
        del anom, mm

    np.save(cache / "ocean_mask.npy", ocean_mask)
    np.save(cache / "lat.npy", np.asarray(lat, dtype=np.float32))
    np.save(cache / "lon.npy", np.asarray(lon, dtype=np.float32))
    np.save(cache / "years.npy", np.asarray(years, dtype=np.int32))
    np.save(cache / "doy_idx.npy", np.asarray(doy_idx, dtype=np.int32))
    np.save(cache / "time.npy", times64)             # datetime64

    meta["ocean_fraction"] = float(ocean_mask.mean())
    with open(cache / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Fraction océan : {100 * meta['ocean_fraction']:.1f} %")
    print(f"  Cache écrit -> {cache}/")
    return meta


# =============================================================================
#  Fixture de test — cache GLORYS synthétique, sans NetCDF ni téléchargement
# =============================================================================

FIXTURE_CACHE = "data/glorys_fixture"

FIXTURE_SPLITS = {"train": (2005, 2008), "val": (2009, 2009),
                  "test": (2010, 2010)}


def make_fixture(cache_dir=FIXTURE_CACHE, years=(2005, 2010),
                 lat_range=(-8.0, 14.0), lon_range=(-40.0, -8.0),
                 resolution=0.5, step_days=2, seed=0, splits=None):
    """Écrit un cache au FORMAT GLORYS12 rempli de champs SYNTHÉTIQUES.

    But : exercer en quelques secondes tout le code GLORYS-spécifique —
    masque terre, haversine, latlon_to_ij, pirata_positions, splits par
    années, climatologie journalière, stats de normalisation train-only —
    sans compte Copernicus, sans NetCDF, sans les 40 Go de réanalyse.

    ┌──────────────────────────────────────────────────────────────────┐
    │ CE N'EST PAS UN JUMEAU PHYSIQUE DE L'ATLANTIQUE TROPICAL.        │
    │ Les champs sont des blobs advectés + AR(1) : ils n'ont ni ondes  │
    │ équatoriales, ni TIWs, ni upwelling, ni bilan de chaleur. Aucun  │
    │ résultat scientifique ne doit être produit sur cette fixture.    │
    │ Elle sert à répondre à « le code tourne-t-il ? », jamais à       │
    │ « la réponse est-elle physiquement juste ? ».                    │
    └──────────────────────────────────────────────────────────────────┘

    Propriétés délibérément conservées du vrai domaine, parce que le code
    testé en dépend :
      - coordonnées réelles à cheval sur l'équateur (cos(lat) non trivial) ;
      - un trait de côte, donc des pixels terre en NaN -> masque océan ;
      - plusieurs mouillages PIRATA nominaux à l'intérieur de la boîte ;
      - un cycle saisonnier marqué, pour que retirer la climatologie change
        réellement le signal ;
      - une décorrélation spatiale finie, pour que corrélations et EVF ne
        soient pas dégénérées.
    """
    import pandas as pd

    splits = splits or FIXTURE_SPLITS
    rng = np.random.default_rng(seed)

    lat = np.arange(lat_range[0], lat_range[1], resolution, dtype=np.float32)
    lon = np.arange(lon_range[0], lon_range[1], resolution, dtype=np.float32)
    nlat, nlon = len(lat), len(lon)

    times = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31",
                          freq=f"{step_days}D")
    nt = len(times)
    yr = times.year.values.astype(np.int32)
    doy = (times.dayofyear.values - 1).astype(np.int32)

    print(f"  Fixture : grille {nlat}x{nlon} @ {resolution} deg "
          f"({lat[0]:.1f}..{lat[-1]:.1f}N, {lon[0]:.1f}..{lon[-1]:.1f}E)")
    print(f"            {nt} pas de temps ({years[0]}-{years[1]}, "
          f"1 sur {step_days} jours)")

    LA = lat[:, None] * np.ones((1, nlon), dtype=np.float32)
    LO = np.ones((nlat, 1), dtype=np.float32) * lon[None, :]

    # ── Trait de côte : Brésil au SO, Afrique de l'Ouest au NE ──────────────
    # Approximation grossière mais calée pour que TOUS les mouillages PIRATA
    # nominaux tombent en mer : sinon latlon_to_ij(require_ocean=True) les
    # rabat silencieusement de plusieurs centaines de km et le scénario
    # `--fixed pirata` teste une géométrie qui n'est pas celle qu'on croit.
    land = (((LA < -2.5) & (LO < -35.0 - 1.0 * (LA + 2.5)))       # Brésil
            | ((LA > 3.0) & (LO > -6.5 - 0.85 * (LA - 3.0))))      # Afrique
    if land.mean() > 0.40 or land.mean() < 0.02:
        print(f"  [ATTENTION] fraction terre = {100*land.mean():.0f} % — "
              f"géométrie de côte peu réaliste, ajuster lat/lon_range")

    # ── Climatologie : gradient méridien + cycle saisonnier + langue froide ──
    doy_all = np.arange(366, dtype=np.float32)
    phase = 2 * np.pi * (doy_all - 60.0) / 365.25
    clim_T = (27.5 - 0.10 * np.abs(LA)[None] ** 1.6
              + 1.8 * np.sin(phase)[:, None, None] * (1 + 0.05 * LA)[None]
              - 1.2 * np.exp(-(LA / 2.5) ** 2)[None]
              * np.clip(np.sin(phase - 1.0), 0, 1)[:, None, None])
    clim_S = (35.6 - 0.6 * np.exp(-((LA - 6) / 4.0) ** 2)[None]
              - 0.35 * np.cos(phase)[:, None, None])

    # ── Anomalies : blobs advectés vers l'ouest + AR(1) lissé ───────────────
    tt = np.arange(nt, dtype=np.float32)

    def _blobs(n_blob, amp, radius_deg, drift):
        """Blobs gaussiens advectés vers l'ouest. Vectorisé sur l'axe temps et
        restreint à la fenêtre de vie de chaque blob : une boucle Python sur nt
        coûtait une minute pour un cache de 30 Mo."""
        out = np.zeros((nt, nlat, nlon), dtype=np.float32)
        rlat, rlon = radius_deg, radius_deg * 1.6
        for _ in range(n_blob):
            la0 = rng.uniform(lat[0], lat[-1])
            lo0 = rng.uniform(lon[0], lon[-1])
            t0 = rng.uniform(0, nt)
            life = rng.uniform(nt / 30, nt / 8)
            a = amp * rng.normal()
            env = np.exp(-((tt - t0) / life) ** 2)
            k = np.where(env > 1e-3)[0]
            if len(k) == 0:
                continue
            lo_t = (lo0 + drift * (tt[k] - t0)).astype(np.float32)
            dlat2 = ((lat[None, :, None] - la0) / rlat) ** 2      # (1,nlat,1)
            dlon2 = ((lon[None, None, :] - lo_t[:, None, None])
                     / rlon) ** 2                                  # (k,1,nlon)
            out[k] += (a * env[k, None, None]
                       * np.exp(-(dlat2 + dlon2))).astype(np.float32)
        return out

    from scipy.ndimage import gaussian_filter

    def _red_noise():
        """Bruit rouge : AR(1) en temps + lissage spatial. Beaucoup moins cher
        qu'un gaussian_filter 3D sur (nt, nlat, nlon)."""
        w = rng.normal(size=(nt, nlat, nlon)).astype(np.float32)
        w = gaussian_filter(w, sigma=(0.0, 2.0, 2.0))
        rho = 0.85
        for t in range(1, nt):
            w[t] += rho * w[t - 1]
        w /= w.std() + 1e-9
        return w

    noise = _red_noise()

    anom_T = _blobs(40, 0.9, 2.2, -0.012) + 0.35 * noise
    # S partiellement corrélée à T (compensation partielle, comme en vrai)
    anom_S = (0.15 * anom_T
              + 0.10 * _blobs(30, 0.9, 2.8, -0.008)
              + 0.05 * _red_noise())

    T = (clim_T[doy] + anom_T).astype(np.float32)
    S = (clim_S[doy] + anom_S).astype(np.float32)
    T[:, land] = np.nan          # la terre définit le masque océan
    S[:, land] = np.nan

    meta = write_cache({"T": T, "S": S}, lat, lon,
                       times.values, yr, doy,
                       cache_dir=cache_dir, splits=splits,
                       source="FIXTURE SYNTHETIQUE — pas de donnees reelles")

    # Trace explicite dans le cache : un run lancé par erreur sur la fixture
    # doit être identifiable a posteriori dans meta.json.
    meta["fixture"] = True
    meta["warning"] = ("Cache de TEST genere par make_fixture(). Champs "
                       "synthetiques sans validite physique. Ne jamais "
                       "utiliser pour un resultat scientifique.")
    with open(Path(cache_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    d = GlorysData(cache_dir)
    pir = d.pirata_positions()
    print(f"  Mouillages PIRATA dans la boîte : {len(pir)} "
          f"({', '.join(pir) if pir else 'aucun — élargir lon/lat_range'})")

    # Contrôle : un mouillage tombé sur la terre est rabattu en silence par
    # latlon_to_ij(require_ocean=True). On veut le savoir.
    tol = 1.5 * resolution
    for name, (i, j) in pir.items():
        la_n, lo_n = PIRATA_NOMINAL[name]
        la_g, lo_g = d.ij_to_latlon(i, j)
        if abs(la_g - la_n) > tol or abs(lo_g - lo_n) > tol:
            print(f"  [ATTENTION] {name} rabattu de "
                  f"({la_n:+.1f},{lo_n:+.1f}) vers ({la_g:+.1f},{lo_g:+.1f}) "
                  f"— il tombe sur la terre de la fixture, ajuster le "
                  f"trait de côte")
    size_mb = sum(f.stat().st_size for f in Path(cache_dir).glob("*")) / 1e6
    print(f"  Taille du cache : {size_mb:.1f} Mo")
    return meta


def _block_mean(a, f, min_frac=0.5):
    """Moyenne par blocs f x f sur les deux derniers axes, robuste aux NaN.

    Une cellule dégradée est déclarée MER si au moins min_frac de ses cellules
    fines le sont, et vaut alors la moyenne des seules cellules mer. Sans ce
    seuil, une cellule côtière contenant un unique pixel d'eau deviendrait un
    point de mer à part entière, et le masque océan grossi mordrait sur la
    terre.
    """
    if f == 1:
        return a
    nt, nlat, nlon = a.shape
    h, w = nlat // f, nlon // f
    b = a[:, :h * f, :w * f].reshape(nt, h, f, w, f)
    finite = np.isfinite(b)
    cnt = finite.sum(axis=(2, 4))
    s = np.where(finite, b, 0.0).sum(axis=(2, 4))
    out = np.where(cnt >= max(1, int(min_frac * f * f)),
                   s / np.maximum(cnt, 1), np.nan)
    return out.astype(np.float32)


def _block_mean_1d(x, f):
    if f == 1:
        return np.asarray(x)
    n = (len(x) // f) * f
    return np.asarray(x)[:n].reshape(-1, f).mean(1)


def preprocess_multi(nc_dir, cache_dir=DEFAULT_CACHE, box=None,
                     coarsen=DEFAULT_COARSEN, splits=None, variables=("T", "S"),
                     years=None, detrend=False, pattern="*.nc"):
    """Prétraite un RÉPERTOIRE de NetCDF GLORYS annuels vers un cache NAIADE.

    Conçu pour le jeu Atlantique complet 2007-2019 : chaque fichier annuel pèse
    ~14 Go avec ses 5 variables, et l'ensemble au 1/12° sur le domaine complet
    représenterait 35 Go de cache pour deux variables. Trois réductions sont
    donc appliquées AVANT tout chargement en mémoire :

      1. sélection des seules variables demandées (thetao, so) au niveau de
         surface (depth = 0.494 m) ;
      2. découpe sur la boîte, par sel() paresseux — xarray ne lit que le
         sous-domaine sur le disque ;
      3. dégradation par moyenne de blocs coarsen x coarsen.

    Le pic mémoire est ainsi d'environ 1 Go (une année, une variable, le
    sous-domaine avant dégradation), quel que soit le nombre d'années.

    L'écriture du cache passe par write_cache(), donc par exactement le même
    code que preprocess() et make_fixture() : climatologie journalière lissée,
    stats de normalisation train uniquement, masque océan, meta.json.
    """
    import pandas as pd
    import xarray as xr

    box = box or DEFAULT_BOX
    splits = splits or DEFAULT_SPLITS
    keep = [v for v in variables if v in VAR_MAP]
    missing = [v for v in REQUIRED_VARS if v not in keep]
    if missing:
        raise ValueError(f"Variables obligatoires absentes : {missing}")

    files = sorted(Path(nc_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier {pattern} dans {nc_dir}")
    print(f"  {len(files)} fichiers NetCDF dans {nc_dir}")
    print(f"  Boîte : lat {box['lat'][0]:+.1f}..{box['lat'][1]:+.1f} | "
          f"lon {box['lon'][0]:+.1f}..{box['lon'][1]:+.1f} | "
          f"dégradation 1/{12 // coarsen}° (blocs {coarsen}x{coarsen})")

    lat = lon = None
    chunks = {k: [] for k in keep}
    times = []

    for f in files:
        ds = xr.open_dataset(f)
        sub = ds.sel(latitude=slice(*box["lat"]), longitude=slice(*box["lon"]))
        t = pd.to_datetime(sub["time"].values)
        if years is not None:
            m = (t.year >= years[0]) & (t.year <= years[1])
            if not m.any():
                ds.close()
                continue
            sub, t = sub.isel(time=np.where(m)[0]), t[m]

        if lat is None:
            lat = _block_mean_1d(sub["latitude"].values, coarsen).astype(np.float32)
            lon = _block_mean_1d(sub["longitude"].values, coarsen).astype(np.float32)
            print(f"  Grille dégradée : {len(lat)}x{len(lon)} "
                  f"({lat[0]:+.2f}..{lat[-1]:+.2f}N, {lon[0]:+.2f}..{lon[-1]:+.2f}E)")

        for k in keep:
            da = sub[VAR_MAP[k]]
            if "depth" in da.dims:
                da = da.isel(depth=0)
            arr = da.transpose("time", "latitude", "longitude").values
            chunks[k].append(_block_mean(arr.astype(np.float32), coarsen))
            del arr
        times.append(t.values)
        ds.close()
        print(f"    {f.name} : {len(t)} dates", flush=True)

    fields = {k: np.concatenate(chunks[k], axis=0) for k in keep}
    for k in keep:
        chunks[k] = None
    times64 = np.concatenate(times)
    tt = pd.to_datetime(times64)
    yr = tt.year.values.astype(np.int32)
    doy = (tt.dayofyear.values - 1).astype(np.int32)

    order = np.argsort(times64)          # les fichiers peuvent être désordonnés
    if not (order == np.arange(len(order))).all():
        print("  Fichiers non chronologiques : réordonnancement")
        for k in keep:
            fields[k] = fields[k][order]
        times64, yr, doy = times64[order], yr[order], doy[order]

    gb = sum(v.nbytes for v in fields.values()) / 1e9
    print(f"  {len(times64)} dates | {gb:.1f} Go en mémoire "
          f"| années {yr.min()}-{yr.max()}")

    return write_cache(fields, lat, lon, times64, yr, doy,
                       cache_dir=cache_dir, splits=splits, detrend=detrend,
                       source=f"{nc_dir} ({len(files)} fichiers), "
                              f"boîte {box}, coarsen {coarsen}")


# =============================================================================
#  Diagnostic des échelles de décorrélation spatiale
# =============================================================================

def _fit_scale(centers, prof, max_km):
    """Ajuste la corrélation binnée par une gaussienne ET une exponentielle.

    Gaussienne exp(-d^2/2L^2) : c'est la forme EXACTE du noyau paramétrique de
    l'EVF (03_rl.py). Exponentielle exp(-d/Le) : forme concurrente classique en
    géostatistique. Si l'exponentielle ajuste nettement mieux, c'est le noyau
    de l'EVF lui-même qui est mal spécifié, pas seulement sa portée.

    Recherche sur grille plutôt que régression sur log(rho) : les profils sont
    bruités et changent de signe en queue, où le log n'est pas défini.
    Renvoie aussi le e-folding lu sur la courbe, valeur SANS modèle.
    """
    ok = np.isfinite(prof)
    if ok.sum() < 5:
        return dict(L=np.nan, Le=np.nan, e_fold=np.nan, forme="indéterminée")
    c, p = centers[ok], prof[ok]
    grid = np.linspace(20.0, max(3000.0, 1.5 * max_km), 500)
    sse_g = np.array([np.sum((p - np.exp(-c ** 2 / (2 * L ** 2))) ** 2)
                      for L in grid])
    sse_e = np.array([np.sum((p - np.exp(-c / Le)) ** 2) for Le in grid])
    L, Le = float(grid[sse_g.argmin()]), float(grid[sse_e.argmin()])

    e_fold = np.nan
    below = np.where(p < np.exp(-1.0))[0]
    if len(below) and below[0] > 0:
        i = below[0]
        f = (p[i - 1] - np.exp(-1.0)) / (p[i - 1] - p[i] + 1e-12)
        e_fold = float(c[i - 1] + f * (c[i] - c[i - 1]))

    g, e = float(sse_g.min()), float(sse_e.min())
    forme = ("gaussienne" if g < 0.8 * e else
             "EXPONENTIELLE" if e < 0.8 * g else "indiscernables")
    return dict(L=L, Le=Le, e_fold=e_fold, forme=forme)


def decorrelation_scales(data, split="train", n_sites=700, step=1,
                         max_km=2500.0, n_bins=40, seed=0,
                         bands=((-3.0, 3.0), (3.0, 15.0))):
    """Estime L, l'échelle de décorrélation spatiale des anomalies.

    Pourquoi ce diagnostic n'est pas optionnel
    ------------------------------------------
    INFLUENCE_RADIUS_KM pilote le modèle paramétrique vers lequel la covariance
    de l'EVF est contractée. Avec EVF_SHRINKAGE = 0.9, c'est donc à 90 % CE
    MODÈLE qui décide du score, et L en est le seul paramètre de forme. La
    valeur 90 km héritée de main provient d'un nature run de moyenne latitude.

    Trois décompositions, parce que trois biais différents guettent :
      - L global : la valeur à mettre dans --influence_km ;
      - zonal contre méridien : en régime équatorial l'anisotropie est forte
        (ondes de Kelvin, TIWs). Le noyau de l'EVF est ISOTROPE — ce
        diagnostic dit à quel point c'est une approximation ;
      - par bande de latitude : le rayon de déformation croît vers l'équateur.
        Si les bandes divergent, un L unique est un compromis, pas une mesure.

    La moyenne spatiale de domaine est retirée à chaque pas de temps, comme
    dans mesoscale_anomaly() (03_rl.py) : sans ça le mode global corrèle tous
    les sites entre eux et L part à l'infini.
    """
    R_KM = 6371.0
    rng = np.random.default_rng(seed)
    idx = data.split_indices(split)[::step]
    nt = len(idx)
    ocean = data.ocean
    sites = np.asarray(sample_ocean_positions(
        ocean, min(n_sites, int(ocean.sum())), rng=rng), dtype=int)

    print(f"  {nt} pas de temps x {len(sites)} sites océan (split {split})")
    if nt < 100:
        print(f"  [ATTENTION] {nt} pas de temps : corrélations très bruitées, "
              f"réduire --scales_step")

    la = np.radians(data.lat[sites[:, 0]])
    lo = np.radians(data.lon[sites[:, 1]])
    lat_deg = data.lat[sites[:, 0]]
    dla, dlo = la[:, None] - la[None, :], lo[:, None] - lo[None, :]
    hav = (np.sin(dla / 2) ** 2
           + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2)
    D = 2 * R_KM * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    # composante purement zonale (même latitude) de la séparation
    DZ = 2 * R_KM * np.arcsin(np.sqrt(np.clip(
        np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2, 0, 1)))

    iu = np.triu_indices(len(sites), 1)
    d = D[iu]
    frac_z = np.divide(DZ[iu], d, out=np.zeros_like(d), where=d > 0)
    lat_a, lat_b = lat_deg[iu[0]], lat_deg[iu[1]]
    edges = np.linspace(0, max_km, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_range = d <= max_km

    def _profile(c, mask):
        m = mask & in_range & np.isfinite(c)
        if m.sum() < 200:
            return np.full(n_bins, np.nan)
        k = np.clip(np.digitize(d[m], edges) - 1, 0, n_bins - 1)
        cnt = np.bincount(k, minlength=n_bins)
        s = np.bincount(k, weights=c[m], minlength=n_bins)
        return np.where(cnt >= 20, s / np.maximum(cnt, 1), np.nan)

    subsets = {"global": np.ones(len(d), bool),
               "zonal": frac_z > 0.9,
               "méridien": frac_z < 0.4}
    for lo_b, hi_b in bands:
        subsets[f"lat [{lo_b:+.0f},{hi_b:+.0f}]"] = (
            (lat_a >= lo_b) & (lat_a < hi_b) & (lat_b >= lo_b) & (lat_b < hi_b))

    results, curves = {}, {}
    for key in data.variables:
        A = np.asarray(data.anomalies[key][idx], dtype=np.float32)
        w = ocean.astype(np.float32)
        A = A - (A * w[None]).reshape(nt, -1).sum(1)[:, None, None] / (w.sum() + 1e-9)
        X = A[:, sites[:, 0], sites[:, 1]]
        X = X - X.mean(0)
        X /= (X.std(0) + 1e-9)
        C = ((X.T @ X) / nt)[iu]

        curves[key], results[key] = {}, {}
        print(f"\n  --- {key} ---")
        for name, mask in subsets.items():
            prof = _profile(C, mask)
            fit = _fit_scale(centers, prof, max_km)
            curves[key][name], results[key][name] = prof, fit
            if np.isfinite(fit["L"]):
                print(f"    {name:<16s} L ={fit['L']:7.0f} km | "
                      f"e-folding ={fit['e_fold']:7.0f} km | "
                      f"meilleure forme : {fit['forme']}")
            else:
                print(f"    {name:<16s} pas assez de couples")

        z, m = results[key]["zonal"]["L"], results[key]["méridien"]["L"]
        if np.isfinite(z) and np.isfinite(m) and m > 0:
            results[key]["global"]["anisotropie"] = z / m
            print(f"    anisotropie zonal/méridien = {z / m:.2f}" +
                  ("  -> noyau isotrope acceptable" if z / m < 1.5 else
                   "  -> ANISOTROPIE FORTE : le noyau isotrope de l'EVF est "
                   "une approximation à justifier"))

    # ── Recommandation, PAR VARIABLE ────────────────────────────────────────
    # Une portée unique est un mauvais compromis dès que L_T et L_S diffèrent
    # nettement (facteur 2.7 sur la boîte PIRATA). On sort donc les deux, dans
    # l'ordre attendu par --influence_km, et pour les deux formes de noyau.
    order = [k for k in ("T", "S") if k in results]
    order += [k for k in results if k not in order]
    Lg = [results[k]["global"]["L"] for k in order]
    Le = [results[k]["global"]["e_fold"] for k in order]
    if Lg and np.isfinite(Lg[0]):
        print("\n  Portées globales par variable :")
        for k, a, b in zip(order, Lg, Le):
            print(f"    {k:<4s} gaussienne L = {a:6.0f} km | "
                  f"exponentielle (e-folding) = {b:6.0f} km")
        gauss = " ".join(f"{v:.0f}" for v in Lg[:2])
        expo = " ".join(f"{v:.0f}" for v in Le[:2] if np.isfinite(v))
        print(f"\n  --influence_km {gauss}                      (noyau gaussien)")
        if expo:
            print(f"  --influence_km {expo} --evf_kernel exp   (noyau exponentiel)")
        reco = float(np.nanmean(Lg))
        print(f"  INFLUENCE_RADIUS_KM recommandé : {reco:.0f} km"
              f"   (scalaire, compromis)")
        if "EXPONENTIELLE" in {results[k]["global"]["forme"] for k in results}:
            print("  [!] La décroissance est mieux décrite par exp(-d/Le) que "
                  "par la gaussienne\n      du noyau EVF. La portée reste "
                  "utilisable, mais la FORME du noyau\n      paramétrique est "
                  "discutable — à mentionner dans les limites.")
        if getattr(data, "meta", {}).get("fixture"):
            print("  [!] Diagnostic mené sur la FIXTURE : aucune signification "
                  "physique.\n      Relancer sur le cache GLORYS réel.")
    return dict(centers=centers, curves=curves, fits=results)


def plot_decorrelation(res, out_path="outputs/decorrelation_scales.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers, curves, fits = res["centers"], res["curves"], res["fits"]
    keys = list(curves)
    fig, axes = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 4.6),
                             squeeze=False)
    for ax, key in zip(axes[0], keys):
        for name, prof in curves[key].items():
            ax.plot(centers, prof, marker="o", ms=3, lw=1.3, label=name)
        L = fits[key]["global"]["L"]
        if np.isfinite(L):
            ax.plot(centers, np.exp(-centers ** 2 / (2 * L ** 2)), "k--", lw=1.3,
                    label=f"gaussienne L={L:.0f} km")
        ax.axhline(np.exp(-1), color="grey", lw=0.8, ls=":")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel("distance (km)")
        ax.set_ylabel("corrélation moyenne")
        ax.set_title(f"Décorrélation spatiale — {key}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Figure -> {out_path}")


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
    p.add_argument("--preprocess_multi", action="store_true",
                   help="Prétraite un RÉPERTOIRE de NetCDF annuels (jeu "
                        "Atlantique 2007-2019) : découpe boîte + dégradation "
                        "par moyenne de blocs, en flux pour tenir la RAM")
    p.add_argument("--nc_dir", type=str, default="data/glorys_nc",
                   help="Répertoire des NetCDF annuels")
    p.add_argument("--box", type=float, nargs=4, default=None,
                   metavar=("LAT0", "LAT1", "LON0", "LON1"),
                   help="Boîte (défaut : emprise PIRATA + 5 deg)")
    p.add_argument("--coarsen", type=int, default=DEFAULT_COARSEN,
                   help="Facteur de dégradation (2 -> 1/6 deg, 3 -> 1/4 deg)")
    p.add_argument("--years", type=int, nargs=2, default=None,
                   metavar=("Y0", "Y1"))
    p.add_argument("--make-fixture", dest="make_fixture", action="store_true",
                   help="Génère un cache de TEST synthétique au format GLORYS "
                        "(aucun téléchargement). Champs sans validité "
                        "physique — pour tester le code, pas la science.")
    p.add_argument("--fixture_cache", type=str, default=FIXTURE_CACHE)
    p.add_argument("--fixture_step",  type=int, default=2,
                   help="1 pas de temps tous les N jours (2 = cache ~20 Mo)")
    p.add_argument("--fixture_res",   type=float, default=0.5,
                   help="Résolution en degrés (0.5 par défaut ; le vrai "
                        "GLORYS12 est à 1/12 deg)")
    p.add_argument("--info",       action="store_true")
    p.add_argument("--scales",     action="store_true",
                   help="Diagnostic des échelles de décorrélation spatiale "
                        "-> valeur à donner à --influence_km (brique 3)")
    p.add_argument("--scales_sites", type=int, default=700,
                   help="Sites océan échantillonnés (coût en n^2)")
    p.add_argument("--scales_step",  type=int, default=1,
                   help="Sous-échantillonnage temporel du diagnostic")
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

    if not any([args.preprocess, args.preprocess_multi, args.make_fixture,
                args.info, args.figures, args.scales]):
        p.print_help()
        raise SystemExit(0)

    if args.make_fixture:
        make_fixture(cache_dir=args.fixture_cache,
                     step_days=args.fixture_step,
                     resolution=args.fixture_res)
        if not (args.info or args.figures):
            args.cache = args.fixture_cache
            args.info = True

    if args.preprocess_multi:
        box = None
        if args.box:
            box = {"lat": (args.box[0], args.box[1]),
                   "lon": (args.box[2], args.box[3])}
        preprocess_multi(args.nc_dir, cache_dir=args.cache, box=box,
                         coarsen=args.coarsen, years=args.years)
        if not (args.info or args.figures or args.scales):
            args.info = True

    if args.preprocess:
        splits = {"train": _parse_years(args.train_years),
                  "val":   _parse_years(args.val_years),
                  "test":  _parse_years(args.test_years)}
        preprocess(args.nc, args.cache, splits=splits, detrend=args.detrend)

    if args.info or args.figures or args.scales:
        data = GlorysData(args.cache)
        print(data.summary())
        if args.figures:
            plot_glorys_summary(data, out_path=args.out)
        if args.scales:
            res = decorrelation_scales(data, n_sites=args.scales_sites,
                                       step=args.scales_step)
            plot_decorrelation(res, out_path=str(
                Path(args.cache) / "decorrelation_scales.png"))
