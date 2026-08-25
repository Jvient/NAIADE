"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DÉCOUPAGE BOÎTE PIRATA depuis les fichiers GLORYS12 ANNUELS                 ║
║                                                                              ║
║  Produit le NetCDF unique attendu par dataset_glorys.preprocess() :          ║
║      data/glorys12_pirata_surface.nc                                         ║
║                                                                              ║
║  Contraintes reprises de preprocess() pour que le cache se construise sans   ║
║  retouche :                                                                  ║
║    · coordonnées nommées latitude/longitude/time (ou lat/lon)                ║
║    · variables aux noms GLORYS : thetao, so, zos, mlotst                     ║
║    · thetao et so OBLIGATOIRES ; zos et mlotst repris s'ils sont là          ║
║    · une seule dimension depth au plus, tranchée au niveau 0 en aval         ║
╚══════════════════════════════════════════════════════════════════════════════╝

    python cut_pirata_box.py --src /chemin/glorys12/surface --pattern "*.nc"
    python cut_pirata_box.py --src ... --box wide      # boîte PIRATA_buoys.txt
    python cut_pirata_box.py --src ... --inspect       # n'écrit rien

Nécessite xarray + dask (`pip install xarray dask netCDF4`).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

# Boîte large, celle discutée dans PIRATA_buoys.txt : contient les 17 bouées.
BOX_WIDE = dict(lat_min=-34.03791187836783, lat_max=30.247730978775017,
                lon_min=-69.55051546282085, lon_max=31.29758676932199)

# Boîte resserrée sur le réseau dense (moins de pixels, plus de pas de temps
# tenables en RAM). Écarte PT062, PT063, PT065, PI280A.
BOX_CORE = dict(lat_min=-20.0, lat_max=22.0, lon_min=-45.0, lon_max=0.0)

BOXES = {"wide": BOX_WIDE, "core": BOX_CORE}

VAR_GLORYS = {"thetao": "T", "so": "S", "zos": "Z", "mlotst": "MLD"}
REQUIRED = ("thetao", "so")


def _coord_names(ds):
    lat = "latitude" if "latitude" in ds.coords else (
        "lat" if "lat" in ds.coords else None)
    lon = "longitude" if "longitude" in ds.coords else (
        "lon" if "lon" in ds.coords else None)
    if lat is None or lon is None or "time" not in ds.coords:
        raise KeyError(
            f"coordonnées introuvables. Présentes : {list(ds.coords)}. "
            "Attendu latitude/longitude/time (ou lat/lon/time).")
    return lat, lon


def _normalise_lon(ds, lon_name, box):
    """GLORYS est en -180..180, mais certains exports sont en 0..360.
    La boîte PIRATA traverse le méridien de Greenwich : sans conversion, la
    sélection renverrait un domaine vide ou coupé en deux."""
    lon = ds[lon_name]
    if float(lon.max()) > 180.0:
        ds = ds.assign_coords({lon_name: (((lon + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)
        print("  longitudes converties 0..360 -> -180..180")
    return ds


def inspect(files, n=1):
    import xarray as xr
    print(f"\n  {len(files)} fichier(s). Inspection du premier :")
    ds = xr.open_dataset(files[0])
    lat, lon = _coord_names(ds)
    print(f"    coords    : {list(ds.coords)}")
    print(f"    variables : {list(ds.data_vars)}")
    print(f"    dims      : {dict(ds.sizes)}")
    print(f"    lat  {float(ds[lat].min()):+.2f} .. {float(ds[lat].max()):+.2f}"
          f"   (pas {float(abs(ds[lat][1] - ds[lat][0])):.4f}°)")
    print(f"    lon  {float(ds[lon].min()):+.2f} .. {float(ds[lon].max()):+.2f}")
    print(f"    time {str(ds['time'].values[0])[:10]} .. "
          f"{str(ds['time'].values[-1])[:10]}")
    miss = [v for v in REQUIRED if v not in ds.data_vars]
    if miss:
        print(f"    [!] variables obligatoires absentes : {miss}")
        print(f"        VAR_MAP de dataset_glorys attend {list(VAR_GLORYS)}")
    ds.close()


def main(a):
    try:
        import xarray as xr
    except ImportError:
        sys.exit("xarray requis : pip install xarray dask netCDF4")

    files = sorted(glob.glob(str(Path(a.src) / a.pattern)))
    if not files:
        sys.exit(f"aucun fichier pour {Path(a.src) / a.pattern}")

    if a.inspect:
        inspect(files)
        return

    box = BOXES[a.box] if a.box in BOXES else None
    if box is None:
        box = dict(lat_min=a.lat_min, lat_max=a.lat_max,
                   lon_min=a.lon_min, lon_max=a.lon_max)

    print("=" * 70)
    print("  Découpage boîte PIRATA depuis les fichiers annuels GLORYS12")
    print("=" * 70)
    print(f"\n  {len(files)} fichier(s) : {Path(files[0]).name} ... "
          f"{Path(files[-1]).name}")
    print(f"  boîte '{a.box}' : lat [{box['lat_min']:.2f}, {box['lat_max']:.2f}] "
          f"lon [{box['lon_min']:.2f}, {box['lon_max']:.2f}]")

    ds = xr.open_mfdataset(files, combine="by_coords", parallel=False,
                           chunks={"time": a.chunk_time})
    lat_name, lon_name = _coord_names(ds)
    ds = _normalise_lon(ds, lon_name, box)

    keep = [v for v in VAR_GLORYS if v in ds.data_vars]
    miss = [v for v in REQUIRED if v not in keep]
    if miss:
        sys.exit(f"variables obligatoires absentes : {miss} "
                 f"(présentes : {list(ds.data_vars)})")
    print(f"  variables retenues : {keep}")
    ds = ds[keep]

    # latitudes parfois décroissantes selon l'export
    if float(ds[lat_name][0]) > float(ds[lat_name][-1]):
        ds = ds.sortby(lat_name)
        print("  latitudes réordonnées croissantes")

    sub = ds.sel({lat_name: slice(box["lat_min"], box["lat_max"]),
                  lon_name: slice(box["lon_min"], box["lon_max"])})
    if sub.sizes[lat_name] == 0 or sub.sizes[lon_name] == 0:
        sys.exit("sélection vide — vérifiez les conventions de longitude "
                 "avec --inspect")

    # sous-échantillonnage spatial : 1/12° sur la boîte large = ~1200x1200 px,
    # hors de portée d'un AE-UNet. coarsen fait une MOYENNE de bloc, ce qui
    # filtre correctement avant décimation (un simple isel::k repliera la
    # mésoéchelle non résolue sur les grandes échelles).
    if a.coarsen > 1:
        sub = sub.coarsen({lat_name: a.coarsen, lon_name: a.coarsen},
                          boundary="trim").mean()
        print(f"  coarsen x{a.coarsen} (moyenne de bloc, pas décimation)")

    if a.time_step > 1:
        sub = sub.isel(time=slice(None, None, a.time_step))

    if "depth" in sub.dims:
        sub = sub.isel(depth=0)
        print("  niveau de surface (depth=0) retenu")

    nt = sub.sizes["time"]
    nla, nlo = sub.sizes[lat_name], sub.sizes[lon_name]
    gb = nt * nla * nlo * len(keep) * 4 / 1e9
    print(f"\n  sortie : {nt} pas de temps x {nla} lat x {nlo} lon "
          f"x {len(keep)} var  ~= {gb:.2f} Go")
    yrs = sub["time"].dt.year.values
    print(f"  années : {yrs.min()}–{yrs.max()}")
    if gb > a.max_gb:
        sys.exit(f"  [STOP] {gb:.1f} Go > --max_gb {a.max_gb}. Augmentez "
                 "--coarsen ou --time_step, ou passez à --box core.")
    if nla * nlo > 400_000:
        print("  [!] plus de 400 000 pixels : l'AE-UNet sera très lourd, "
              "augmentez --coarsen")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in keep}
    print(f"\n  écriture -> {out}  (compression zlib, peut prendre du temps)")
    sub.to_netcdf(out, encoding=enc)
    ds.close()

    print(f"\n  Terminé. Étape suivante :")
    print(f"    python dataset_glorys.py --preprocess --nc {out} \\")
    print(f"        --train_years 2005-2016 --val_years 2017-2018 "
          f"--test_years 2019-2020")
    print("  (adaptez les années à la couverture affichée ci-dessus ; "
          "preprocess échoue si le split train est vide)")


def parse_args():
    p = argparse.ArgumentParser("découpage boîte PIRATA")
    p.add_argument("--src", required=True, help="dossier des fichiers annuels")
    p.add_argument("--pattern", default="*.nc")
    p.add_argument("--out", default="data/glorys12_pirata_surface.nc")
    p.add_argument("--box", default="wide",
                   help="wide | core | custom (avec --lat_min etc.)")
    p.add_argument("--lat_min", type=float, default=BOX_WIDE["lat_min"])
    p.add_argument("--lat_max", type=float, default=BOX_WIDE["lat_max"])
    p.add_argument("--lon_min", type=float, default=BOX_WIDE["lon_min"])
    p.add_argument("--lon_max", type=float, default=BOX_WIDE["lon_max"])
    p.add_argument("--coarsen", type=int, default=6,
                   help="facteur de moyenne de bloc (6 : 1/12° -> 0.5°)")
    p.add_argument("--time_step", type=int, default=1)
    p.add_argument("--chunk_time", type=int, default=30)
    p.add_argument("--max_gb", type=float, default=25.0)
    p.add_argument("--inspect", action="store_true",
                   help="affiche la structure du premier fichier, n'écrit rien")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
