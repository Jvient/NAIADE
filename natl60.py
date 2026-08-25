"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CHARGEUR NATL60  —  passage du nature run synthetique a un ocean reel       ║
║                                                                              ║
║  Fichiers attendus (challenge 2023 OSSE SSC NATL60 GF) :                     ║
║      NATL60-CJM165_NATL_sst_y2013.1y.nc                                      ║
║      NATL60-CJM165_NATL_ssh_y2013.1y.nc                                      ║
║  grille 1/20°, journaliere, 365 pas, 27-65°N / 79°W-7°E.                     ║
║                                                                              ║
║  Ce que ce changement apporte, et qui manquait au synthetique :              ║
║  NATL60 resout les equations primitives, donc la vitesse depend de l'etat.   ║
║  Le systeme est CHAOTIQUE, la ou notre generateur advectait des traceurs     ║
║  passifs dans un champ prescrit. Un filtre lineaire cesse d'y etre optimal   ║
║  par construction, et le plafond calculable de `ceiling.py` disparait -- ce  ║
║  qui rouvre les questions fermees pour cause de trop grande previsibilite.   ║
║                                                                              ║
║  Quatre pieges specifiques aux donnees reelles, traites ici :                ║
║                                                                              ║
║  1. GRILLE NON CARREE. A 0,05° la maille meridienne fait 5,6 km partout,     ║
║     mais la maille zonale vaut 5,6 x cos(lat) : 4,7 km a 33°N, 4,1 a 43°N.   ║
║     Le code suppose une maille carree unique ; on retient la moyenne sur la  ║
║     boite et on signale l'ecart. Au-dela d'une quinzaine de degres           ║
║     d'extension meridienne, il faudrait reprojeter.                          ║
║                                                                              ║
║  2. TERRE. Les points continentaux sont des NaN, que ni les EOF ni les       ║
║     correlations ne supportent. On refuse une boite qui en contient plutot   ║
║     que de les combler en silence.                                           ║
║                                                                              ║
║  3. CYCLE SAISONNIER SUR UNE SEULE ANNEE. Impossible d'estimer une           ║
║     climatologie par moyenne inter-annuelle. On retire la carte moyenne puis ║
║     un passe-bas temporel : ce qui reste est la mesoechelle. La coupure est  ║
║     un choix, `lowpass_days` l'expose.                                       ║
║                                                                              ║
║  4. CONVENTION D'AXES. Le code NAIADE attend (temps, x=lon, y=lat) ; les     ║
║     fichiers sont en (time, lat, lon).                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

R_EARTH_KM = 6371.0

# Boites usuelles : (lat_min, lat_max, lon_min, lon_max)
BOXES = {
    # Boite du challenge : meandre principal du Gulf Stream, tourbillons et
    # filaments sous-mesoechelle. Pleine mer, aucun point de terre.
    "gulfstream": (33.0, 43.0, -65.0, -55.0),
    # Interieur de gyre subtropicale : faible EKE, champ homogene. Sert de
    # temoin -- si la conception integree n'apporte rien la ou le champ est
    # uniforme, c'est attendu, pas un echec.
    "subtropical": (28.0, 36.0, -50.0, -40.0),
    # Atlantique nord-est, regime intermediaire.
    "northeast": (45.0, 55.0, -25.0, -15.0),
}


@dataclass
class BoxData:
    sst: np.ndarray          # (nt, nx, ny) float32
    ssh: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    dx_km: float             # maille moyenne retenue (compatibilite NAIADE)
    dx_zonal_km: float       # maille zonale reelle
    dy_merid_km: float       # maille meridienne reelle
    dx_ratio: float          # anisotropie zonale/meridienne sur la boite
    name: str

    @property
    def shape(self):
        return self.sst.shape


def _grid_metrics(lat, lon):
    dlat = float(np.abs(np.diff(lat)).mean())
    dlon = float(np.abs(np.diff(lon)).mean())
    dy = dlat * np.pi / 180.0 * R_EARTH_KM
    dx = dlon * np.pi / 180.0 * R_EARTH_KM * np.cos(np.deg2rad(lat)).mean()
    return dx, dy


def mesoscale_anomaly_obs(F, lowpass_days=90, detrend_space=True):
    """
    Anomalie mesoechelle d'un champ observe sur UNE seule annee.

    On retire la carte moyenne temporelle (structure permanente : position
    moyenne du jet, gradient nord-sud), puis un passe-bas temporel par moyenne
    glissante (cycle saisonnier et derive basse frequence). Ce qui reste est la
    variabilite de quelques jours a quelques semaines, c'est-a-dire ce qu'un
    reseau de mouillages a vocation a contraindre.

    `lowpass_days` est un CHOIX, pas une constante physique : trop court, il
    mange la mesoechelle ; trop long, il laisse du saisonnier. A 90 jours on
    garde les echelles de 10 a 40 jours qui portent les tourbillons.
    """
    F = np.asarray(F, dtype=np.float32)
    A = F - F.mean(axis=0, keepdims=True)
    w = int(max(lowpass_days, 3))
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=np.float32) / w
    pad = w // 2
    Ap = np.concatenate([A[:pad][::-1], A, A[-pad:][::-1]], axis=0)
    low = np.empty_like(A)
    flat = Ap.reshape(len(Ap), -1)
    out = np.empty((len(A), flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.convolve(flat[:, j], k, mode="valid")
    low = out.reshape(A.shape)
    A = A - low
    if detrend_space:
        A -= A.mean(axis=(1, 2), keepdims=True)
    return A


def load_box(sst_path, ssh_path, box="gulfstream", stride=1,
             t_slice=None, lowpass_days=90, verbose=True):
    """
    Extrait une boite, verifie qu'elle est exploitable, et rend les champs dans
    la convention NAIADE.

    `stride` sous-echantillonne spatialement : 1 garde le 1/20° (~5 km), 2
    donne du 1/10°. Attention, la resolution effective du champ de surface de
    NATL60 est d'environ 7 km : au-dela de stride=1 on commence a jeter
    precisement la mesoechelle qui justifie d'utiliser ce jeu.
    """
    import xarray as xr

    if isinstance(box, str):
        if box not in BOXES:
            raise KeyError(f"Boite inconnue : {box}. Choix : {list(BOXES)}")
        name, (la0, la1, lo0, lo1) = box, BOXES[box]
    else:
        name, (la0, la1, lo0, lo1) = "custom", box

    sel = dict(lat=slice(la0, la1), lon=slice(lo0, lo1))
    dss = xr.open_dataset(sst_path).sel(**sel)
    dsh = xr.open_dataset(ssh_path).sel(**sel)
    if t_slice is not None:
        dss = dss.isel(time=slice(*t_slice))
        dsh = dsh.isel(time=slice(*t_slice))

    lat = dss["lat"].values[::stride]
    lon = dss["lon"].values[::stride]
    sst = dss["sst"].values[:, ::stride, ::stride].astype(np.float32)
    ssh = dsh["ssh"].values[:, ::stride, ::stride].astype(np.float32)
    dss.close(); dsh.close()

    n_nan = int(np.isnan(sst).any(axis=0).sum() + np.isnan(ssh).any(axis=0).sum())
    if n_nan:
        raise ValueError(
            f"{n_nan} points de terre (NaN) dans la boite '{name}'. Ni les EOF "
            f"ni les correlations ne les supportent, et les combler en silence "
            f"fabriquerait de la structure. Choisir une boite de pleine mer, "
            f"ou implementer un masque de bout en bout.")

    dx, dy = _grid_metrics(lat, lon)
    dx_km = 0.5 * (dx + dy)
    ratio = dx / dy

    # (time, lat, lon) -> (time, x=lon, y=lat)
    sst = np.transpose(sst, (0, 2, 1)).copy()
    ssh = np.transpose(ssh, (0, 2, 1)).copy()

    if verbose:
        nt, nx, ny = sst.shape
        print(f"  Boite '{name}' : {la0}-{la1}°N, {lo0}-{lo1}°E")
        print(f"    grille        {nx} x {ny} points, {nt} pas de temps")
        print(f"    maille        zonale {dx:.2f} km | meridienne {dy:.2f} km "
              f"| retenue {dx_km:.2f} km")
        if abs(ratio - 1) > 0.10:
            print(f"    [ATTENTION] anisotropie de maille {ratio:.2f} : le code "
                  f"suppose une maille carree.\n"
                  f"                Les distances sont fausses de "
                  f"{abs(ratio-1)*100:.0f} % dans une direction. Reduire "
                  f"l'extension\n"
                  f"                meridienne ou reprojeter avant toute "
                  f"conclusion quantitative.")
        print(f"    SST           {np.nanmean(sst):.2f} +/- {np.nanstd(sst):.2f} °C")
        print(f"    SSH           {np.nanmean(ssh):.3f} +/- {np.nanstd(ssh):.3f} m")

    return BoxData(sst=sst, ssh=ssh, lon=lon, lat=lat, dx_km=float(dx_km),
                   dx_zonal_km=float(dx), dy_merid_km=float(dy),
                   dx_ratio=float(ratio), name=name)
