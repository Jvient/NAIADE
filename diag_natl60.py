"""
DIAGNOSTIC NATL60  —  le test qui decide, avant tout portage

Trois questions, dans cet ordre. Chacune peut invalider la suite.

1. PREVISIBILITE. Ajuster un modele inverse lineaire sur les EOF et mesurer la
   variance perdue en prevision libre a 1, 5, 10 et 30 jours. Sur l'ocean
   synthetique elle valait 13 % a dix jours : un modele qui perd si peu n'a
   presque pas besoin d'observations, l'assimilation sature, et toute
   evaluation de reseau conclut que n'importe quel reseau suffit. C'est ce qui
   a ferme l'etage 3. Si le Gulf Stream perd 40-50 %, tout se rouvre.

2. NOYAU. Ajuster exp(-d^2/2L^2) sur la correlation empirique et regarder le
   RESIDU. Sur le synthetique il valait 0,17-0,19 et aucune famille ne faisait
   mieux, ce qui rendait ininterpretable tout ecart de quelques pour cent.

3. ANISOTROPIE. Longueur de decorrelation zonale contre meridienne. Le jet du
   Gulf Stream correle sur des centaines de kilometres dans le sens du courant
   et sur quelques dizaines en travers. Un noyau isotrope y sera pire
   qu'ailleurs -- argument supplementaire pour l'evaluateur de Kalman, qui
   n'a pas besoin de noyau.

    python diag_natl60.py --data_dir data --box gulfstream
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from natl60 import load_box, mesoscale_anomaly_obs, BOXES

BG, PANEL, EDGE = "#0a1628", "#050d1a", "#2a4a7a"


def _frame(ax, title="", xlab="", ylab=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
    if title:
        ax.set_title(title, color="white", fontsize=10.5, fontweight="bold",
                     pad=7)
    ax.set_xlabel(xlab, color="white", fontsize=9)
    ax.set_ylabel(ylab, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)


def eofs(X, n_modes, ntr):
    mu = X[:ntr].mean(0, keepdims=True)
    Xtr = X[:ntr] - mu
    G = Xtr @ Xtr.T
    w, U = np.linalg.eigh(G.astype(np.float64))
    o = np.argsort(w)[::-1][:n_modes]
    w, U = np.clip(w[o], 1e-12, None), U[:, o]
    E = (Xtr.T @ U) / np.sqrt(w)[None, :]
    return E.astype(np.float32), (X - mu) @ E, float(Xtr.var(0).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="natl60",
                   choices=["natl60", "glorys"],
                   help="natl60 = verite haute resolution, une annee ; "
                        "glorys = reanalyse, treize ans. La profondeur "
                        "d echantillon de glorys est ce qui rend le "
                        "propagateur estimable.")
    p.add_argument("--glob", type=str, default="*.nc",
                   help="Motif des fichiers GLORYS dans --data_dir")
    p.add_argument("--channels", type=str, nargs=2,
                   default=["thetao", "so"],
                   help="Canaux GLORYS (thetao/so/zos)")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--sst", type=str,
                   default="NATL60-CJM165_NATL_sst_y2013.1y.nc")
    p.add_argument("--ssh", type=str,
                   default="NATL60-CJM165_NATL_ssh_y2013.1y.nc")
    p.add_argument("--box", type=str, default="gulfstream")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--n_modes", type=int, default=60)
    p.add_argument("--lowpass_days", type=int, default=90)
    p.add_argument("--max_lag_km", type=float, default=600.0)
    p.add_argument("--out_dir", type=str, default="outputs")
    a = p.parse_args()

    d = Path(a.data_dir)
    if a.source == "glorys":
        from glorys import load_box_glorys
        paths = sorted(d.glob(a.glob))
        if not paths:
            raise SystemExit(f"Aucun fichier '{a.glob}' dans {d}")
        bx = BOXES[a.box] if a.box in BOXES else a.box
        box, _times = load_box_glorys(paths, box=bx, stride=a.stride,
                                      channels=tuple(a.channels))
        # l anomalie est deja retiree par climatologie journaliere
        Ta, Ha = box.sst.copy(), box.ssh.copy()
    else:
        box = load_box(d / a.sst, d / a.ssh, box=a.box, stride=a.stride,
                       lowpass_days=a.lowpass_days)
        Ta = mesoscale_anomaly_obs(box.sst, a.lowpass_days)
        Ha = mesoscale_anomaly_obs(box.ssh, a.lowpass_days)
        print(f"\n  Anomalie mesoechelle : passe-bas {a.lowpass_days} j")
    nt, nx, ny = Ta.shape
    dx = box.dx_km
    Ta = Ta / (Ta.std() + 1e-9)
    Ha = Ha / (Ha.std() + 1e-9)

    X = np.concatenate([Ta.reshape(nt, -1), Ha.reshape(nt, -1)], axis=1)
    ntr = nt // 2
    E, Z, var_tot = eofs(X, a.n_modes, ntr)
    lam = Z[:ntr].var(0)
    print(f"  EOF : {E.shape[1]} modes, {lam.sum()/var_tot*100:.0f} % de la "
          f"variance")

    # ── 1. previsibilite ─────────────────────────────────────────────────────
    Z0, Z1 = Z[:ntr - 1], Z[1:ntr]
    C0 = Z0.T @ Z0 / len(Z0)
    A = (Z1.T @ Z0 / len(Z0)) @ np.linalg.pinv(C0)
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho >= 1.0:
        A = A * (0.999 / rho)
    Q = C0 - A @ C0 @ A.T
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    Q = V @ np.diag(np.clip(w, 0, None)) @ V.T

    P = np.zeros_like(Q)
    horizons, curve = [1, 5, 10, 30], {}
    series = []
    for h in range(1, 61):
        P = A @ P @ A.T + Q
        series.append(np.trace(P) / lam.sum())
        if h in horizons:
            curve[h] = series[-1]
    print(f"\n  1. PREVISIBILITE — erreur de prevision libre / variance")
    print("     " + "   ".join(f"{h} j = {curve[h]*100:.0f} %"
                               for h in horizons))
    print(f"     (ocean synthetique NAIADE : 1 %, 6 %, 13 %, 35 %)")
    p10 = curve[10]
    if p10 > 0.35:
        verdict1 = ("     -> Previsibilite REALISTE. Les observations comptent, "
                    "l'assimilation ne\n        saturera pas. L'etage 3 et le RL "
                    "de maintenance redeviennent des\n        questions ouvertes.")
    elif p10 > 0.20:
        verdict1 = ("     -> Previsibilite intermediaire. Mieux que le "
                    "synthetique, mais verifier\n        la saturation avant de "
                    "conclure quoi que ce soit.")
    else:
        verdict1 = (
            "     -> Toujours tres previsible, MAIS c'est une propriete de la\n"
            "        REPRESENTATION autant que de l'ocean : moyennes "
            "journalieres a 1/20°\n"
            "        et troncature EOF retirent precisement les echelles "
            "rapides et peu\n"
            "        previsibles. Tester la sensibilite avant de conclure :\n"
            "          --lowpass_days 30     (garder plus de variabilite "
            "rapide)\n"
            "          --n_modes 150         (moins tronquer l'etat)\n"
            "          --stride 1            (garder le 1/20° complet)")
    print(verdict1)

    # ── 2. noyau ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    npts = min(700, nx * ny)
    flat = rng.choice(nx * ny, npts, replace=False)
    ix, iy = np.unravel_index(flat, (nx, ny))
    sT = Ta[:ntr].reshape(ntr, -1)[:, flat]
    sH = Ha[:ntr].reshape(ntr, -1)[:, flat]
    R = 0.5 * (np.corrcoef(sT, rowvar=False) + np.corrcoef(sH, rowvar=False))
    R = np.nan_to_num(R)
    # Distances avec les mailles REELLES, pas la maille moyenne. Utiliser dx
    # dans les deux directions injecte exactement l'anisotropie de grille dans
    # la mesure d'anisotropie du champ -- une premiere version rapportait un
    # rapport de 1,00 sur un champ dont l'anisotropie vraie etait 0,79.
    dxk = (ix[:, None] - ix[None, :]) * box.dx_zonal_km
    dyk = (iy[:, None] - iy[None, :]) * box.dy_merid_km
    dd = np.sqrt(dxk ** 2 + dyk ** 2)
    iu = np.triu_indices(npts, 1)
    dv, rv = dd[iu], R[iu]
    keep = dv < a.max_lag_km
    dv, rv = dv[keep], rv[keep]

    Ls = np.linspace(5, 400, 300)
    err = [np.sqrt(np.mean((np.exp(-dv ** 2 / (2 * L ** 2)) - rv) ** 2))
           for L in Ls]
    i = int(np.argmin(err))
    L_fit, res = float(Ls[i]), float(err[i])
    print(f"\n  2. NOYAU — L ajuste = {L_fit:.0f} km, RMS residuel = {res:.3f}")
    print(f"     (synthetique : 0,17-0,19 ; correlation minimale ici "
          f"{rv.min():+.2f})")
    if res > 0.15:
        print("     -> Erreur de modele comparable au synthetique : les ecarts "
              "de quelques\n        pour cent resteront ininterpretables. "
              "Privilegier l'evaluateur Kalman.")
    else:
        print("     -> Noyau nettement mieux specifie qu'attendu.")

    # ── 3. anisotropie ───────────────────────────────────────────────────────
    def efold(axis_km, corr):
        o = np.argsort(axis_km)
        b = np.linspace(0, a.max_lag_km, 40)
        idx = np.digitize(axis_km[o], b)
        prof = np.array([corr[o][idx == k].mean() if (idx == k).any() else np.nan
                         for k in range(1, len(b))])
        c = 0.5 * (b[1:] + b[:-1])
        ok = ~np.isnan(prof)
        below = np.flatnonzero(prof[ok] < 1 / np.e)
        return (float(c[ok][below[0]]) if len(below) else float(a.max_lag_km),
                c[ok], prof[ok])

    zon = np.abs(dyk[iu][keep]) < 1.5 * box.dy_merid_km
    mer = np.abs(dxk[iu][keep]) < 1.5 * box.dx_zonal_km
    Lz, cz, pz = efold(np.abs(dxk[iu][keep])[zon], rv[zon])
    Lm, cm, pm = efold(np.abs(dyk[iu][keep])[mer], rv[mer])
    print(f"\n  3. ANISOTROPIE — decorrelation zonale {Lz:.0f} km | "
          f"meridienne {Lm:.0f} km | rapport {Lz/max(Lm,1e-9):.2f}")
    print(f"     (mailles reelles {box.dx_zonal_km:.2f} / "
          f"{box.dy_merid_km:.2f} km, l anisotropie de grille est corrigee)")
    if abs(Lz / max(Lm, 1e-9) - 1) > 0.3:
        print("     -> Champ franchement anisotrope. Un noyau isotrope est "
              "structurellement\n        inadapte ; ne pas fonder de conclusion "
              "sur le critere parametrique seul.")

    # ── figure ───────────────────────────────────────────────────────────────
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)

    _frame(ax[0], "Erreur de prevision libre", "Horizon (jours)",
           "fraction de la variance")
    ax[0].plot(range(1, 61), series, color="#6bcb77", lw=2, label="NATL60")
    ax[0].plot([1, 5, 10, 30], [0.01, 0.06, 0.13, 0.35], "s--",
               color="#ff6b6b", lw=1.4, label="synthetique NAIADE")
    ax[0].axhline(0.5, color="white", ls=":", lw=1, alpha=0.6)
    ax[0].legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor=EDGE)
    ax[0].grid(alpha=0.15, color="white")

    _frame(ax[1], f"Correlation vs distance (L={L_fit:.0f} km, "
                  f"RMS={res:.3f})", "Distance (km)", "Correlation")
    s = rng.choice(len(dv), min(12000, len(dv)), replace=False)
    ax[1].scatter(dv[s], rv[s], s=3, alpha=0.15, color="#6bcb77")
    xx = np.linspace(0, a.max_lag_km, 300)
    ax[1].plot(xx, np.exp(-xx ** 2 / (2 * L_fit ** 2)), color="#ffd93d", lw=2.2)
    ax[1].axhline(0, color="white", lw=0.8, alpha=0.5)
    ax[1].grid(alpha=0.15, color="white")

    _frame(ax[2], "Anisotropie", "Separation (km)", "Correlation")
    ax[2].plot(cz, pz, color="#4d96ff", lw=2, label=f"zonale ({Lz:.0f} km)")
    ax[2].plot(cm, pm, color="#ff6b6b", lw=2, label=f"meridienne ({Lm:.0f} km)")
    ax[2].axhline(1 / np.e, color="white", ls=":", lw=1, alpha=0.7)
    ax[2].legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor=EDGE)
    ax[2].grid(alpha=0.15, color="white")

    fig.suptitle(f"NATL60 — boite '{box.name}' : ce que change un ocean reel",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    path = out / f"diag_natl60_{box.name}.png"
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Figure -> {path}")


if __name__ == "__main__":
    main()
