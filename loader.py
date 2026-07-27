"""
NAIADE — point d'entrée unique pour les données.

Toutes les briques appellent `load_ocean(args)` et reçoivent le même quadruplet,
quelle que soit la source :

    fields, channels, sea_mask, info = load_ocean(args)

    fields   : (nt, n_ch, nx, ny) float32
    channels : list[str]
    sea_mask : (nx, ny) bool
    info     : dict de provenance, à verser dans les rapports
"""
import numpy as np

import config as C


def load_ocean(args=None, verbose=True):
    """Charge le champ océanique selon config.DATA_SOURCE."""
    source = getattr(args, "data_source", None) or C.DATA_SOURCE
    nt = getattr(args, "nt", None) or C.NT

    if source == "glorys":
        from data.glorys import load_glorys

        path = getattr(args, "glorys_dir", None) or C.GLORYS_DIR
        dom = load_glorys(
            path,
            variables=C.GLORYS_VARIABLES,
            depth_indices=C.GLORYS_DEPTHS,
            lon_range=C.GLORYS_LON_RANGE,
            lat_range=C.GLORYS_LAT_RANGE,
            coarsen=C.GLORYS_COARSEN,
            grid_multiple=C.GLORYS_GRID_MULT,
            remove_seasonal=C.GLORYS_SEASONAL,
            require_full_sea=C.GLORYS_FULL_SEA,
            cache=C.GLORYS_CACHE,
            verbose=verbose,
        )

        # ── Garde-fous : une géométrie ou un nombre de canaux incohérent
        #    ferait tourner tout le pipeline sur des dimensions fausses
        #    sans jamais lever d'erreur explicite.
        if (dom.nx, dom.ny) != (C.NX, C.NY):
            raise ValueError(
                f"\n  Grille GLORYS {dom.nx}×{dom.ny} ≠ config NX×NY "
                f"= {C.NX}×{C.NY}."
                f"\n  → python -m data.glorys --probe {path} "
                f"--lon {C.GLORYS_LON_RANGE[0]} {C.GLORYS_LON_RANGE[1]} "
                f"--lat {C.GLORYS_LAT_RANGE[0]} {C.GLORYS_LAT_RANGE[1]}"
                f"\n     puis reporter NX/NY dans config.py.")

        if dom.n_ch != C.N_CHANNELS:
            raise ValueError(
                f"\n  {dom.n_ch} canaux chargés ≠ config N_CHANNELS "
                f"= {C.N_CHANNELS}."
                f"\n  → Vérifier GLORYS_VARIABLES et GLORYS_DEPTHS.")

        fields = dom.fields[:nt] if nt < dom.nt else dom.fields

        info = dict(
            source="GLORYS12V1",
            variables=list(C.GLORYS_VARIABLES),
            depths_m=[round(float(d), 3) for d in dom.depths],
            grid_multiple=C.GLORYS_GRID_MULT,
            channels=list(dom.channels),
            observed_vars=list(C.OBSERVED_VARS),
            n_times=len(fields),
            date_start=str(dom.times[0])[:10],
            date_end=str(dom.times[min(len(fields), len(dom.times)) - 1])[:10],
            seasonal_removed=C.GLORYS_SEASONAL,
            full_sea=dom.is_full_sea,
            sea_fraction=round(dom.sea_fraction, 5),
            dx_km=round(dom.meta.get("dx_km", float("nan")), 2),
            lon_range=[float(dom.lon[0]), float(dom.lon[-1])],
            lat_range=[float(dom.lat[0]), float(dom.lat[-1])],
            level_redundancy={k: (None if not np.isfinite(v) else round(v, 4))
                              for k, v in dom.level_redundancy().items()},
        )
        return fields, dom.channels, dom.sea_mask, info

    # ── Mode synthétique (legacy) ────────────────────────────────────────────
    from data.dataset import SyntheticOceanGenerator

    seed = getattr(args, "seed_ocean", 42)
    gen = SyntheticOceanGenerator(nx=C.NX, ny=C.NY)
    T, S = gen.generate_dataset(nt=nt, seed=seed)
    fields = np.stack([T, S], axis=1).astype(np.float32)
    channels = ["thetao_z0", "so_z0"]
    sea_mask = np.ones((C.NX, C.NY), dtype=bool)
    info = dict(source="synthetic", n_times=len(fields), seed_ocean=seed,
                channels=channels, full_sea=True, sea_fraction=1.0)
    return fields, channels, sea_mask, info


def add_data_args(parser):
    """Arguments CLI communs aux quatre scripts."""
    parser.add_argument("--data_source", choices=["synthetic", "glorys"],
                        default=None, help="Surcharge config.DATA_SOURCE")
    parser.add_argument("--glorys_dir", type=str, default=None,
                        help="Surcharge config.GLORYS_DIR")
    return parser
