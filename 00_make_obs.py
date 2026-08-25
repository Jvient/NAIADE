"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BRIQUE 0 — GÉNÉRATION DU JEU D'OBSERVATIONS                                 ║
║                                                                              ║
║  Produit outputs/obs_synth.npz (les observations) et outputs/_truth.npz      ║
║  (le nature run, SCELLÉ jusqu'à validate_obsonly.py).                        ║
║                                                                              ║
║  Trois pièges traités ici, chacun silencieux si on l'ignore :                ║
║                                                                              ║
║  1. UNITÉS DU BRUIT. config.OBS_NOISE_T/S sont en °C et psu. Les champs      ║
║     donnés au modèle sont NORMALISÉS. Passer le bruit physique tel quel à    ║
║     un champ normalisé le multiplie par ~1/sigma — soit un facteur 2.6 en    ║
║     T et 0.18 en S sur ce nature run. On convertit explicitement.            ║
║                                                                              ║
║  2. SÉPARATION MINIMALE. La Brique 3 s'interdit les bouées adjacentes        ║
║     (MIN_BUOY_SEP_KM). Un tirage uniforme produirait un réseau que le RL     ║
║     n'aurait jamais le droit de proposer — comparaison faussée.              ║
║                                                                              ║
║  3. ADVECTION. config.U_MEAN/V_MEAN sont dans les unités internes du         ║
║     générateur, pas en px/pas. On la mesure sur le champ.                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

    python 00_make_obs.py --nt 800 --seed_ocean 42 --seed_buoys 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from obs_operator import ObsNetwork, estimate_advection, PRESETS


# ── import du générateur, quel que soit son emplacement ──────────────────────
def _load_generator():
    last = None
    for mod in ("data.dataset", "dataset"):
        try:
            m = __import__(mod, fromlist=["*"])
            gen = getattr(m, "SyntheticOceanGenerator")
            sep = getattr(m, "sample_separated_positions", None)
            return gen, sep, mod
        except Exception as e:
            last = e
    raise ImportError(
        f"SyntheticOceanGenerator introuvable (data.dataset ni dataset) : {last}")


def _load_glorys(a):
    """Nature run GLORYS12 sur la boîte PIRATA.

    get_arrays(normalized=True) rend déjà des ANOMALIES climatologiques
    normalisées : pas de renormalisation ici, sinon on écrase les stats du
    split train par celles du split chargé.
    """
    from dataset_glorys import GlorysData
    d = GlorysData(a.glorys_cache)
    Tn, Sn = d.get_arrays(a.split, ("T", "S"), normalized=True, step=a.step)
    return d, np.asarray(Tn, np.float32), np.asarray(Sn, np.float32)


def main(a):
    import config as cfg

    print("=" * 70)
    print("  Brique 0 — génération des observations")
    print("=" * 70)

    glorys = None
    if a.data == "glorys":
        glorys, Tn, Sn = _load_glorys(a)
        nt, nx, ny = Tn.shape
        # anomalies déjà normalisées par les stats du split TRAIN
        norm = dict(T_mean=0.0, T_std=float(glorys.norm["T"]["std"]),
                    S_mean=0.0, S_std=float(glorys.norm["S"]["std"]))
        print(f"\n[1/5] GLORYS12 split={a.split} step={a.step}")
        print(f"      grille {nx}x{ny} | {nt} pas de temps | "
              f"océan {100 * glorys.ocean.mean():.1f} %")
        print(f"      lat [{glorys.lat.min():.1f}, {glorys.lat.max():.1f}] "
              f"lon [{glorys.lon.min():.1f}, {glorys.lon.max():.1f}]")
        print(f"      sigma_T = {norm['T_std']:.2f} °C | "
              f"sigma_S = {norm['S_std']:.3f} psu (stats train)")
        sample_sep = None
    else:
        gen_cls, sample_sep, src = _load_generator()
        print(f"\n[1/5] Nature run via {src} (nt={a.nt}, seed={a.seed_ocean})")
        T, S = gen_cls().generate_dataset(nt=a.nt, seed=a.seed_ocean)
        T = np.asarray(T, np.float32); S = np.asarray(S, np.float32)
        nt, nx, ny = T.shape
        print(f"      T {T.shape} [{T.min():.1f}, {T.max():.1f}] °C  sigma={T.std():.2f}")
        print(f"      S {S.shape} [{S.min():.2f}, {S.max():.2f}] psu sigma={S.std():.3f}")
        norm = dict(T_mean=float(T.mean()), T_std=float(T.std()),
                    S_mean=float(S.mean()), S_std=float(S.std()))
        Tn = (T - norm["T_mean"]) / norm["T_std"]
        Sn = (S - norm["S_mean"]) / norm["S_std"]

    nT_phys = float(getattr(cfg, "OBS_NOISE_T", 0.05))
    nS_phys = float(getattr(cfg, "OBS_NOISE_S", 0.02))
    nT = nT_phys / norm["T_std"]
    nS = nS_phys / norm["S_std"]
    print(f"\n[2/5] Bruit instrumental converti en unités normalisées")
    print(f"      T : {nT_phys:.3f} °C  -> {nT:.4f}  "
          f"({100 * nT_phys / norm['T_std']:.1f} % de sigma_T)")
    print(f"      S : {nS_phys:.3f} psu -> {nS:.4f}  "
          f"({100 * nS_phys / norm['S_std']:.1f} % de sigma_S)")
    if nS > 0.15:
        print("      [!] le bruit de salinité dépasse 15 % du signal — "
              "vérifiez OBS_NOISE_S avant d'interpréter les scores SSS")

    # ── advection mesurée ────────────────────────────────────────────────────
    u, v, c = estimate_advection(Tn, max_shift=a.max_shift)
    if glorys is not None and a.step > 1:
        u, v = u / a.step, v / a.step   # advection par pas de temps réel
    print(f"\n[3/5] Advection mesurée : u={u:+.0f} v={v:+.0f} px/pas "
          f"(corr={c:.3f})")
    if abs(u) < 1e-9 and abs(v) < 1e-9:
        print("      [!] déplacement nul à la résolution entière. Les dériveurs "
              "seront dominés par la marche aléatoire ; augmentez --max_shift "
              "ou acceptez-le (le champ advecte moins d'un pixel par pas).")

    # ── réseau ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(a.seed_buoys)
    ocean = glorys.ocean.astype(bool) if glorys is not None else None
    net = ObsNetwork(nx=nx, ny=ny, nt=nt, rng=rng, ocean=ocean)

    if glorys is not None:
        # POSITIONS RÉELLES des mouillages, pas un tirage aléatoire.
        # PIRATA_NOMINAL de dataset_glorys ne contient que 8 positions et
        # porte un avertissement « à vérifier/compléter » ; pirata_real.py
        # fournit les 17 de PIRATA_buoys.txt.
        print("\n[4/5] Mouillages")
        if a.pirata == "real":
            from pirata_real import pirata_positions_real
            pir = pirata_positions_real(glorys)
        else:
            pir = glorys.pirata_positions(in_box_only=True)
            print(f"  PIRATA nominal : {len(pir)} mouillages "
                  "(sous-ensemble de dataset_glorys)")
        pos = list(pir.values())
        if not pos:
            raise RuntimeError(
                "aucun mouillage dans la boîte — vérifiez l'emprise du cache")
        a_n_moorings = len(pos)
    elif sample_sep is not None and not a.no_separation:
        pos = sample_sep(nx, ny, a.n_moorings, rng=rng)
        a_n_moorings = a.n_moorings
        print(f"\n[4/5] {a.n_moorings} mouillages via sample_separated_positions "
              f"(sep >= {getattr(cfg, 'MIN_BUOY_SEP_KM', '?')} km)")
    else:
        pos = None
        a_n_moorings = a.n_moorings
        print(f"\n[4/5] {a.n_moorings} mouillages en tirage libre "
              "(séparation minimale NON respectée)")

    net.add_moorings(n=a_n_moorings, positions=pos, noise_T=nT, noise_S=nS,
                     hazard_var_amp=a.hazard_var_amp,
                     hazard_daily=a.hazard_daily,
                     service_days=a.service_days,
                     return_rate=a.return_rate)
    net.add_argo(n=a.n_argo, noise_T=nT, noise_S=nS,
                 maintain=not a.no_redeploy)
    if a.n_drifters:
        net.add_drifters(n=a.n_drifters, u=u, v=v, noise_T=nT, noise_S=nS,
                         maintain=not a.no_redeploy)
    if a.n_glider_repeat:
        net.add_glider(waypoints=[(nx // 8, ny // 8),
                                  (7 * nx // 8, 7 * ny // 8)],
                       n_repeat=a.n_glider_repeat, noise_T=nT, noise_S=nS)

    obs = net.sample(Tn, Sn)

    # ── contrôles de sanité ──────────────────────────────────────────────────
    print("\n[5/5] Contrôles")
    ser = obs.gridded_series("T")
    miss = float(np.isnan(ser).mean())
    print(f"      manquants dans les séries : {100 * miss:.1f} %")
    if miss > 0.9:
        print("      [!] plus de 90 % de manquants — le GNN aura trop peu de "
              "chevauchement pour estimer les corrélations")
    per_t = np.array([len(obs.at(t)) for t in range(obs.nt)])
    print(f"      obs par pas de temps : min={per_t.min()} "
          f"médiane={int(np.median(per_t))} max={per_t.max()}")
    half = per_t[:len(per_t) // 2].mean(), per_t[len(per_t) // 2:].mean()
    if half[0] > 1.3 * half[1]:
        print(f"      [!] couverture en chute : {half[0]:.1f} obs/pas sur la "
              f"1re moitié contre {half[1]:.1f} sur la 2e. Les plateformes "
              "dérivantes meurent sans remplacement (--no_redeploy ?).")
    if np.median(per_t) < 8:
        print("      [!] médiane < 8 : la reconstruction de champ sera très "
              "sous-contrainte, augmentez le nombre de plateformes")

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    obs.save(out / "obs_synth.npz")
    np.savez_compressed(out / "_truth.npz", T=Tn, S=Sn, **norm)
    print(f"\n      observations -> {out}/obs_synth.npz")
    print(f"      vérité SCELLÉE -> {out}/_truth.npz  "
          "(ne pas rouvrir avant validate_obsonly.py)")


def parse_args():
    p = argparse.ArgumentParser("génération des observations")
    p.add_argument("--nt", type=int, default=800)
    p.add_argument("--seed_ocean", type=int, default=42)
    p.add_argument("--seed_buoys", type=int, default=7)
    p.add_argument("--n_moorings", type=int, default=24)
    p.add_argument("--n_argo", type=int, default=15)
    p.add_argument("--n_drifters", type=int, default=10)
    p.add_argument("--n_glider_repeat", type=int, default=6)
    p.add_argument("--hazard_daily", type=float, default=5e-4,
                   help="panne quotidienne des MOUILLAGES. 2e-3 sur 10 ans "
                        "donne ~30 %% de retour, loin des 70-80 %% de PIRATA")
    p.add_argument("--service_days", type=int, default=180)
    p.add_argument("--no_redeploy", action="store_true",
                   help="pas de remplacement des plateformes "
                        "dérivantes mortes (flotte décroissante)")
    p.add_argument("--return_rate", type=float, default=0.80)
    p.add_argument("--hazard_var_amp", type=float, default=2.0,
                   help="0 = manquants aléatoires ; >0 = pannes corrélées "
                        "à la variance locale (cas réaliste)")
    p.add_argument("--max_shift", type=int, default=6)
    p.add_argument("--no_separation", action="store_true")
    p.add_argument("--data", default="synthetic",
                   choices=["synthetic", "glorys"])
    p.add_argument("--glorys_cache", default="data/glorys_cache")
    p.add_argument("--pirata", default="real",
                   choices=["real", "nominal"],
                   help="real = les 17 de PIRATA_buoys.txt")
    p.add_argument("--split", default="train",
                   choices=["train", "val", "test"])
    p.add_argument("--step", type=int, default=1,
                   help="sous-échantillonnage temporel GLORYS")
    p.add_argument("--output_dir", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
