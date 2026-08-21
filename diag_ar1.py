"""
DIAGNOSTIC  --  l'AR(1) par mode decrit-il la dynamique, ou la caricature-t-il ?

Le filtre de l'etage 2 suppose z_{t+1} = diag(a) z_t + w. Le temps de
decorrelation qui en sort vaut 6 j sur `demo` mais 38 j sur `large`, alors que
les tourbillons y ont un temps de retournement de l'ordre de la dizaine de
jours. Deux hypotheses :

  H1  la variance est dominee par des modes lents (basse frequence mal retiree
      par mesoscale_anomaly a cette echelle) -- l'AR(1) est alors fidele, et
      c'est le champ lui-meme qui est lent ;

  H2  l'AR(1) absorbe en "persistance" ce qui est en realite de la
      PROPAGATION. Un tourbillon qui se deplace fait osciller les EOF les unes
      dans les autres ; un modele diagonal, incapable de representer ce
      transfert, ne peut l'imiter que par une decroissance lente.

Le test qui les separe :

  * l'autocorrelation empirique d'un PC contre celle de l'AR(1) ajuste. Si
    l'empirique decroit vite puis oscille ou change de signe, c'est de la
    propagation (H2). Si elle decroit lentement et regulierement, c'est
    vraiment lent (H1).

  * la part de variance expliquee a un jour par un modele DIAGONAL contre un
    modele COUPLE A = C(1) C(0)^-1. Si le couple fait nettement mieux, la
    dynamique passe par les termes hors-diagonale que l'AR(1) jette.

Enjeu : si H2 est vraie, l'evaluateur Kalman SURESTIME la memoire, donc
sous-estime le cout d'une panne -- et c'est exactement l'effet qui fait gagner
la politique "moins cheres". Le classement des politiques serait alors un
artefact du propagateur.

    NAIADE_DOMAIN=large python diag_ar1.py --maintenance pirata
"""

from __future__ import annotations

import argparse, importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DOMAIN, NX, NY, DX_KM, PORT_XY_FRAC, NT
from data.dataset import SyntheticOceanGenerator, mesoscale_anomaly

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nt", type=int, default=NT)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--n_modes", type=int, default=50)
    p.add_argument("--maintenance", type=str, default="pirata",
                   choices=["regional", "pirata"])
    p.add_argument("--max_lag", type=int, default=90)
    p.add_argument("--out_dir", type=str, default="outputs")
    a = p.parse_args()

    spec = importlib.util.spec_from_file_location(
        "brick3", Path(__file__).with_name("03_rl.py"))
    b3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b3)
    from kalman import KalmanEOF

    print(f"\n  Domaine {DOMAIN} {NX}x{NY} @ {DX_KM:.0f} km")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=a.nt, seed=a.seed)
    port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM
    maint = b3.MaintenanceModel(b3.get_params(a.maintenance), port)
    env = b3.OceanNetworkEnv(T, S, grid_x=a.grid_x, grid_y=a.grid_y,
                             n_min=10, n_max=30, maintenance=maint)
    # Les DEUX propagateurs sont construits explicitement. Se contenter de
    # celui charge par defaut reviendrait a comparer LIM a lui-meme sous
    # l'etiquette "AR(1)" -- le diagnostic validerait alors ce qu'il teste.
    kf_ar1 = KalmanEOF(env, n_modes=a.n_modes, propagator="ar1")
    kf = KalmanEOF(env, n_modes=a.n_modes, propagator="lim")

    # ── reconstruire les PC sur toute la periode ─────────────────────────────
    Ta = mesoscale_anomaly(env.T) / (env.T.std() + 1e-9)
    Sa = mesoscale_anomaly(env.S) / (env.S.std() + 1e-9)
    st, nt = env.eval_stride, len(Ta)
    X = np.concatenate([Ta[:, ::st, ::st].reshape(nt, -1),
                        Sa[:, ::st, ::st].reshape(nt, -1)], axis=1)
    X = X - X.mean(0, keepdims=True)
    Z = X @ kf.E                                    # (nt, k)
    k = Z.shape[1]

    # ── 1. autocorrelation empirique vs AR(1) ────────────────────────────────
    lags = np.arange(0, a.max_lag + 1)
    def implied_acf(A, C0, max_lag):
        """r_i(lag) = [A^lag C0]_ii / C0_ii : autocorrelation IMPLIQUEE par un
        propagateur. C'est la seule quantite comparable a l'empirique."""
        d = np.clip(np.diag(C0), 1e-12, None)
        M = np.eye(len(d))
        out = [np.ones(len(d))]
        for _ in range(max_lag):
            M = A @ M
            out.append(np.diag(M @ C0) / d)
        return np.array(out).T                      # (k, max_lag+1)

    Z0, Z1 = Z[:-1], Z[1:]
    C0 = Z0.T @ Z0 / len(Z0)
    acf_ar1 = implied_acf(kf_ar1.A, C0, a.max_lag)
    acf_lim = implied_acf(kf.A, C0, a.max_lag)

    def efold(r):
        b = np.flatnonzero(r < 1 / np.e)
        return float(b[0]) if len(b) else float(a.max_lag)

    print(f"\n  Temps de decorrelation (passage sous 1/e), en jours")
    print(f"  {'mode':>5} | {'part var.':>9} | {'empirique':>9} | "
          f"{'AR(1)':>7} | {'LIM':>7} | {'signe negatif ?':>15}")
    print("  " + "-" * 68)
    emp_tau, ar_tau, lim_tau, rows = [], [], [], []
    for i in range(min(k, 8)):
        z = Z[:, i]
        ac = np.array([1.0 if l == 0 else
                       float(np.corrcoef(z[l:], z[:-l])[0, 1]) for l in lags])
        te = efold(ac)
        ta, tl = efold(acf_ar1[i]), efold(acf_lim[i])
        neg = "oui" if ac.min() < -0.15 else "non"
        emp_tau.append(te); ar_tau.append(ta); lim_tau.append(tl)
        rows.append((i, ac, te, ta, tl, neg))
        print(f"  {i:>5} | {kf.lam[i]/kf.lam.sum()*100:>8.1f}% | {te:>9.1f} | "
              f"{ta:>7.1f} | {tl:>7.1f} | {neg:>15}")

    # ── 2. diagonal contre couple ────────────────────────────────────────────
    C1 = Z1.T @ Z0 / len(Z0)
    A_full = C1 @ np.linalg.pinv(C0)                 # modele inverse lineaire
    A_diag = np.diag(np.diag(A_full))
    var = (Z1 ** 2).mean()
    r_diag = 1.0 - ((Z1 - Z0 @ A_diag.T) ** 2).mean() / var
    r_full = 1.0 - ((Z1 - Z0 @ A_full.T) ** 2).mean() / var
    off = (np.abs(A_full - A_diag).sum() /
           max(np.abs(A_full).sum(), 1e-12) * 100)
    print(f"\n  Prevision a 1 jour, variance expliquee")
    print(f"    AR(1) diagonal              {r_diag:.4f}")
    print(f"    modele couple C(1)C(0)^-1   {r_full:.4f}")
    print(f"    poids des termes hors-diagonale : {off:.0f} % de |A|")

    # ── verdict ──────────────────────────────────────────────────────────────
    r_ar1 = float(np.median(np.array(ar_tau) / np.maximum(emp_tau, 1e-9)))
    r_lim = float(np.median(np.array(lim_tau) / np.maximum(emp_tau, 1e-9)))
    gain = (r_full - r_diag) / max(1.0 - r_diag, 1e-9) * 100
    print(f"\n  VERDICT")
    print(f"    tau implique / tau empirique : AR(1) x{r_ar1:.1f}  |  "
          f"LIM x{r_lim:.1f}")
    print(f"    le couplage recupere {gain:.0f} % de l'erreur de prevision a "
          f"1 jour")
    if r_ar1 > 1.5 and r_lim < 1.5:
        print("    -> L'AR(1) surestime la memoire ; le modele couple corrige.\n"
              "       Garder --propagator lim (defaut).")
    elif r_ar1 > 1.5:
        print("    -> Les DEUX propagateurs surestiment la memoire. Ni l'un ni\n"
              "       l'autre n'est utilisable en l'etat : chercher du cote de\n"
              "       la basse frequence residuelle du champ.")
    else:
        print("    -> Les deux propagateurs sont fideles a cette echelle ;\n"
              "       le choix importe peu ici.")

    # ── figure ───────────────────────────────────────────────────────────────
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=BG)
    ax = axes[0]
    _frame(ax, "Autocorrelation des composantes principales",
           "Decalage (jours)", "Correlation")
    for i, ac, te, ta, tl, neg in rows[:4]:
        c = plt.cm.viridis(i / 4)
        ax.plot(lags, ac, color=c, lw=1.9, label=f"mode {i} (empirique)")
        ax.plot(lags, acf_ar1[i][:len(lags)], color=c, lw=1.0, ls="--",
                alpha=0.75)
        ax.plot(lags, acf_lim[i][:len(lags)], color=c, lw=1.3, ls=":",
                alpha=0.95)
    ax.axhline(1 / np.e, color="white", ls=":", lw=1, alpha=0.7)
    ax.axhline(0, color="white", lw=0.8, alpha=0.5)
    ax.legend(fontsize=7.5, labelcolor="white", facecolor=BG, edgecolor=EDGE)
    ax.grid(alpha=0.15, color="white")
    ax.text(0.98, 0.95, "plein : empirique\ntirets : AR(1)\npointille : LIM",
            transform=ax.transAxes, ha="right", va="top", color="white",
            fontsize=7.8, linespacing=1.5)

    ax = axes[1]
    _frame(ax, "Structure du propagateur a 1 jour", "mode", "mode")
    im = ax.imshow(np.abs(A_full[:20, :20]), cmap="magma")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white",
                                labelsize=7)
    ax.text(0.02, 0.02, f"hors-diagonale = {off:.0f} % de |A|\n"
                        f"var. expliquee : diagonal {r_diag:.3f} "
                        f"vs couple {r_full:.3f}",
            transform=ax.transAxes, color="white", fontsize=8,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=EDGE,
                      alpha=0.9))

    fig.suptitle("L'AR(1) par mode decrit-il la dynamique ?", color="white",
                 fontsize=13, fontweight="bold", y=1.0)
    path = out / "diag_ar1.png"
    fig.savefig(path, dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Figure -> {path}")


if __name__ == "__main__":
    main()
