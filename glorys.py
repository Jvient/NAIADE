"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CHARGEUR GLORYS12  —  la profondeur d'echantillon qui manquait              ║
║                                                                              ║
║  GLORYS12V1, global 1/12°, journalier, surface (0,494 m), 2007-2019 :        ║
║  thetao (T), so (S), zos (SSH), uo, vo.                                      ║
║                                                                              ║
║  Ce que treize ans debloquent, et qu'une annee de NATL60 interdisait :       ║
║                                                                              ║
║  1. ~4750 jours au lieu de 365. Le propagateur a 150 modes devient estimable ║
║     (30 echantillons par parametre de ligne au lieu de 1,2), et le           ║
║     surajustement diagnostique sur NATL60 disparait.                         ║
║                                                                              ║
║  2. CLIMATOLOGIE VRAIE. Sur une annee il fallait un passe-bas arbitraire     ║
║     pour retirer le saisonnier, au risque de manger la mesoechelle. Ici on   ║
║     moyenne par jour de l'annee sur treize ans : l'anomalie mesoechelle      ║
║     n'est plus un choix de coupure.                                          ║
║                                                                              ║
║  3. VARIABILITE INTERANNUELLE, et donc la saisonnalite de l'EKE -- la seule  ║
║     facon de repondre a « vaut-il mieux armer la campagne au printemps ou a  ║
║     l'automne », question que nos scenarios ne pouvaient pas poser.          ║
║                                                                              ║
║  4. SEPARATION PAR ANNEES. Ajuster sur 2007-2015, valider sur 2016-2019 :    ║
║     un protocole propre, sans fuite temporelle.                              ║
║                                                                              ║
║  5. T ET S disponibles, donc les MEMES deux canaux que le nature run         ║
║     synthetique : aucun refactoring de canaux.                               ║
║                                                                              ║
║  CE QUE GLORYS N'EST PAS : une verite. C'est une reanalyse, elle contient    ║
║  l'empreinte du reseau d'observation qu'on cherche a evaluer, et sa          ║
║  mesoechelle est amortie. Usage recommande :                                 ║
║                                                                              ║
║      GLORYS  ->  statistiques, propagateur, scenarios, saisonnalite          ║
║      NATL60  ->  verite de l'OSSE                                            ║
║                                                                              ║
║  Ce montage a un merite inattendu : un propagateur estime sur GLORYS         ║
║  applique a une verite NATL60, ce sont DEUX modeles differents. L'erreur de  ║
║  modele devient reelle au lieu d'etre fabriquee -- c'est la sortie du piege  ║
║  du modele parfait, celui qui donnait 13 % de perte a dix jours et rendait   ║
║  l'assimilation inutile.                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from natl60 import BOXES, BoxData, _grid_metrics


def _open(paths, box, stride, verbose):
    import xarray as xr

    la0, la1, lo0, lo1 = box
    # `chunks` exigerait dask ; on concatene a la main pour n'imposer aucune
    # dependance supplementaire. Une boite regionale tient largement en memoire
    # (13 ans a 1/12° sur 10°x10° font moins d'un Go en float32).
    try:
        ds = xr.open_mfdataset(sorted(str(p) for p in paths),
                               combine="by_coords", chunks={"time": 60})
    except ImportError:
        parts = []
        for p in sorted(str(x) for x in paths):
            d = xr.open_dataset(p)
            parts.append(d.sel(latitude=slice(box[0], box[1]),
                               longitude=slice(box[2], box[3])).load())
            d.close()
        ds = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    ds = ds.sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
    if "depth" in ds.dims:
        ds = ds.isel(depth=0, drop=True)
    if verbose:
        print(f"    {len(ds.time)} pas de temps, "
              f"{str(ds.time.values[0])[:10]} -> {str(ds.time.values[-1])[:10]}")
    return ds


def climatology_anomaly(F, times, smooth_days=15):
    """
    Anomalie par rapport a une VRAIE climatologie journaliere.

    Moyenne par jour de l'annee sur toutes les annees disponibles, lissee sur
    `smooth_days` pour absorber le bruit d'echantillonnage (treize valeurs par
    jour calendaire, ce n'est pas enorme). Contrairement au passe-bas utilise
    sur une annee unique, cette anomalie ne repose sur aucune hypothese de
    separation d'echelles : le saisonnier est retire parce qu'il est
    saisonnier, pas parce qu'il est lent.
    """
    F = np.asarray(F, dtype=np.float32)
    doy = np.array([t.astype("datetime64[D]").astype(object).timetuple().tm_yday
                    for t in times])
    doy = np.clip(doy, 1, 365)
    clim = np.zeros((365,) + F.shape[1:], dtype=np.float32)
    for d in range(1, 366):
        m = doy == d
        clim[d - 1] = F[m].mean(0) if m.any() else np.nan
    # lissage circulaire sur le jour de l'annee
    w = int(max(smooth_days, 1))
    if w > 1:
        k = np.ones(w, dtype=np.float32) / w
        pad = w // 2
        ext = np.concatenate([clim[-pad:], clim, clim[:pad]], axis=0)
        flat = ext.reshape(len(ext), -1)
        out = np.empty((365, flat.shape[1]), dtype=np.float32)
        for j in range(flat.shape[1]):
            out[:, j] = np.convolve(flat[:, j], k, mode="valid")[:365]
        clim = out.reshape(clim.shape)
    A = F - clim[doy - 1]
    A -= A.mean(axis=(1, 2), keepdims=True)
    return A


def load_box_glorys(paths, box="gulfstream", stride=1, channels=("thetao", "so"),
                    smooth_days=15, verbose=True):
    """
    Extrait une boite sur toutes les annees fournies.

    `channels` : deux variables parmi thetao / so / zos. Le couple par defaut
    (T, S) reproduit exactement les canaux du nature run synthetique, donc
    aucun changement de code en aval. (zos, thetao) donnerait le couple le plus
    proche de NATL60 pour une comparaison directe.
    """
    if isinstance(box, str):
        if box not in BOXES:
            raise KeyError(f"Boite inconnue : {box}. Choix : {list(BOXES)}")
        name, bb = box, BOXES[box]
    else:
        name, bb = "custom", box

    paths = [Path(p) for p in (paths if isinstance(paths, (list, tuple))
                               else [paths])]
    if verbose:
        print(f"  GLORYS12, boite '{name}' : {len(paths)} fichier(s)")
    ds = _open(paths, bb, stride, verbose)

    lat = ds["latitude"].values[::stride].astype(np.float64)
    lon = ds["longitude"].values[::stride].astype(np.float64)
    times = ds["time"].values
    fields = []
    for c in channels:
        if c not in ds:
            raise KeyError(f"Variable '{c}' absente. Disponibles : "
                           f"{list(ds.data_vars)}")
        F = ds[c].values[:, ::stride, ::stride].astype(np.float32)
        fields.append(F)
    ds.close()

    n_nan = int(sum(np.isnan(F).any(axis=0).sum() for F in fields))
    if n_nan:
        raise ValueError(
            f"{n_nan} points de terre (NaN) dans la boite '{name}'. Choisir "
            f"une boite de pleine mer, ou implementer un masque de bout en "
            f"bout : combler fabriquerait de la structure.")

    dx, dy = _grid_metrics(lat, lon)
    A = [np.transpose(climatology_anomaly(F, times, smooth_days), (0, 2, 1))
         for F in fields]

    if verbose:
        nt, nx, ny = A[0].shape
        yrs = nt / 365.25
        print(f"    grille        {nx} x {ny}, {nt} jours ({yrs:.1f} ans)")
        print(f"    maille        zonale {dx:.2f} km | meridienne {dy:.2f} km")
        print(f"    canaux        {channels[0]} / {channels[1]}")
        print(f"    anomalie      climatologie journaliere lissee "
              f"{smooth_days} j (vraie climatologie, pas un passe-bas)")
        for c, F in zip(channels, A):
            print(f"      {c:<8} ecart-type {F.std():.4f}")
        print(f"    echantillons  {nt} jours ; un propagateur a k modes exige "
              f"~3k jours\n                  -> k <= {nt//3} modes sans "
              f"surajustement (contre 60 sur NATL60)")

    return BoxData(sst=A[0], ssh=A[1], lon=lon, lat=lat,
                   dx_km=float(0.5 * (dx + dy)), dx_zonal_km=float(dx),
                   dy_merid_km=float(dy), dx_ratio=float(dx / dy),
                   name=f"{name}_glorys"), times


def eke_seasonality(u, v, times):
    """
    Cycle saisonnier de l'energie cinetique turbulente, moyenne par mois.

    Repond a une question que nos scenarios ne pouvaient pas poser : le cout
    d'une panne depend-il de la saison ? Si l'EKE varie d'un facteur deux entre
    printemps et automne, la DATE des campagnes devient un levier -- et c'est
    le seul angle ou le RL de maintenance garde une chance, puisque l'oracle
    myope ne teste que l'ordonnancement a l'interieur d'une campagne.
    """
    eke = 0.5 * (np.asarray(u, np.float32) ** 2 + np.asarray(v, np.float32) ** 2)
    mon = np.array([t.astype("datetime64[M]").astype(object).month
                    for t in times])
    return np.array([eke[mon == m].mean() if (mon == m).any() else np.nan
                     for m in range(1, 13)])
