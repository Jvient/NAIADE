"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  POSITIONS RÉELLES DES MOUILLAGES PIRATA                                     ║
║                                                                              ║
║  dataset_glorys.PIRATA_NOMINAL ne contient que 8 positions nominales, avec   ║
║  un avertissement explicite dans le code source :                            ║
║     « positions NOMINALES de déploiement — à vérifier/compléter depuis les   ║
║       métadonnées GTMBA/PMEL avant toute utilisation quantitative »          ║
║                                                                              ║
║  Ce module fournit les 17 positions effectivement listées dans               ║
║  PIRATA_buoys.txt, et les convertit en indices de grille sur le cache        ║
║  GLORYS chargé.                                                              ║
║                                                                              ║
║  Deux d'entre elles sortent de la boîte GLORYS habituelle (PT065 à 20.45N,   ║
║  PI280A à -18.85N) : elles sont filtrées avec un message, jamais projetées   ║
║  silencieusement sur le bord du domaine.                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# nom -> (latitude, longitude), depuis PIRATA_buoys.txt
PIRATA_REAL = {
    "PI289A": (0.0000, -2.6850),
    "PI288A": (0.0200, -9.8467),
    "PI280A": (-18.8517, -34.6583),
    "PI285A": (0.0100, -34.9967),
    "PI284A": (7.9467, -38.0300),
    "PI283A": (4.0083, -37.9367),
    "PT077": (-6.0333, -9.9983),
    "PT078": (-9.9067, -9.9817),
    "PT065": (20.4517, -23.1417),
    "PT068": (11.4883, -22.9867),
    "PT069": (4.0450, -22.9867),
    "PT076": (0.0017, -22.9883),
    "PT070": (-8.0083, -30.6333),
    "PT062": (-13.5233, -32.5967),
    "PT063": (20.0250, -37.8467),
    "PT072": (15.0033, -37.9917),
    "PT075": (2.4133, -4.6300),      # ou PI287A selon la campagne
}

# Boîte de travail discutée dans PIRATA_buoys.txt
PIRATA_BOX = dict(lon_min=-69.55051546282085, lon_max=31.29758676932199,
                  lat_min=-34.03791187836783, lat_max=30.247730978775017)


def pirata_positions_real(glorys, verbose=True, max_snap_deg=0.75):
    """Positions réelles -> indices (i, j) sur la grille du cache GLORYS.

    Une bouée hors domaine est ÉCARTÉE, jamais rabattue sur le bord : la
    projection silencieuse de latlon_to_ij(require_ocean=True) déplacerait
    un mouillage de plusieurs centaines de kilomètres et fausserait sa
    contribution marginale sans le signaler.

    max_snap_deg : déplacement maximal toléré lors du recalage sur un pixel
    océan (bouée tombant sur un pixel terre du masque).
    """
    lat_min, lat_max = float(glorys.lat.min()), float(glorys.lat.max())
    lon_min, lon_max = float(glorys.lon.min()), float(glorys.lon.max())

    kept, dropped, snapped = {}, [], []
    for name, (la, lo) in sorted(PIRATA_REAL.items()):
        if not (lat_min <= la <= lat_max and lon_min <= lo <= lon_max):
            dropped.append((name, la, lo))
            continue
        i, j = glorys.latlon_to_ij(la, lo, require_ocean=True)
        la_g, lo_g = glorys.ij_to_latlon(i, j)
        shift = max(abs(la_g - la), abs(lo_g - lo))
        if shift > max_snap_deg:
            dropped.append((name, la, lo))
            snapped.append((name, shift))
            continue
        kept[name] = (int(i), int(j))

    if verbose:
        print(f"  PIRATA réel : {len(kept)}/{len(PIRATA_REAL)} mouillages "
              f"dans la boîte [{lat_min:.1f},{lat_max:.1f}]N "
              f"[{lon_min:.1f},{lon_max:.1f}]E")
        if dropped:
            print("    écartés : " + ", ".join(n for n, _, _ in dropped))
        for n, sh in snapped:
            print(f"    [!] {n} : recalage océan de {sh:.2f}° — écarté plutôt "
                  "que déplacé")
        if len(kept) < 10:
            print("    [!] moins de 10 mouillages : un split fixe par capteur "
                  "laissera trop peu de nœuds en validation. Utilisez une "
                  "validation croisée par plis.")
    return kept


if __name__ == "__main__":
    print(f"{len(PIRATA_REAL)} mouillages réels")
    for n, (la, lo) in sorted(PIRATA_REAL.items()):
        print(f"  {n:8s} {la:+9.4f}N {lo:+9.4f}E")
