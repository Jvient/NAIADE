"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ETAGE 3a  —  LE PLAFOND EST CALCULABLE                                      ║
║                                                                              ║
║  Le nature run advecte des traceurs PASSIFS : `_velocity(t)` ne depend que   ║
║  du temps, jamais de T ni de S. Le systeme est donc                          ║
║                                                                              ║
║      T_{t+1} = M_t T_t + f_t                                                 ║
║                                                                              ║
║  lineaire, a operateur VARIABLE dans le temps. Consequence : un filtre de    ║
║  Kalman utilisant le vrai M_t est l'estimateur OPTIMAL. Aucun modele appris  ║
║  ne peut faire mieux, par construction.                                      ║
║                                                                              ║
║  Ce qui manque au LIM de l'etage 2 n'est donc pas la non-linearite -- il n'y ║
║  en a pas -- mais la dependance a l'ecoulement : son A est stationnaire,     ║
║  moyenne sur la periode d'entrainement, alors que le vrai propagateur change ║
║  chaque jour avec la position des tourbillons.                               ║
║                                                                              ║
║  Ce module mesure exactement cet ecart :                                     ║
║                                                                              ║
║      LIM stationnaire   <   AE recurrent appris   <   KF a operateur exact   ║
║                                                                              ║
║  Si l'ecart entre les deux bornes vaut 2 %, l'etage 3 est sans objet sur cet ║
║  ocean. S'il vaut 25 %, on sait quoi viser avant d'avoir entraine quoi que   ║
║  ce soit. Meme logique que l'oracle des politiques : mesurer le plafond      ║
║  avant de courir apres.                                                      ║
║                                                                              ║
║  COMMENT on obtient M_t sans reecrire le modele : on remplace temporairement ║
║  `_departure_stencil` de l'instance par une enveloppe qui, en plus de rendre ║
║  le stencil au generateur, l'applique a une banque d'EOF. La trajectoire du  ║
║  nature run est donc reproduite a l'identique -- meme graine, memes tirages  ║
║  -- et l'operateur est capture au passage, sans risque de divergence entre   ║
║  une reimplementation et le modele reel.                                     ║
║                                                                              ║
║  Le rappel climatologique est affine ; sa partie constante s'annule pour une ║
║  perturbation, il ne reste que le facteur 1/(1+r), different pour T et S.    ║
╚══════════════════════════════════════════════════════════════════════════════╝

    NAIADE_DOMAIN=large python ceiling.py --maintenance pirata --n_max 30
"""

from __future__ import annotations

import argparse, importlib.util, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (DOMAIN, NX, NY, DX_KM, PORT_XY_FRAC, NT, DT_DAYS,
                    N_SUBSTEPS, KAPPA, TAU_T_DAYS, TAU_S_DAYS, SPINUP_DAYS,
                    EVF_SHRINKAGE)
from data.dataset import (SyntheticOceanGenerator, mesoscale_anomaly,
                          _laplacian)

DAY = 86400.0
BG, PANEL, EDGE = "#0a1628", "#050d1a", "#2a4a7a"


# ══════════════════════════════════════════════════════════════════════════════
#  EOF SUR LA GRILLE COMPLETE
# ══════════════════════════════════════════════════════════════════════════════

def full_grid_eofs(T, S, n_modes, train_frac=0.5):
    """
    EOF sur la grille COMPLETE (et non la grille d'evaluation sous-echantillonnee)
    parce que l'operateur reel agit sur la grille complete.

    Passage par la matrice de Gram (nt x nt) : la SVD directe d'une matrice
    (nt, 2*nx*ny) serait inutilement couteuse alors que nt est petit.
    """
    nt = len(T)
    Ta = (mesoscale_anomaly(T) / (T.std() + 1e-9)).astype(np.float32)
    Sa = (mesoscale_anomaly(S) / (S.std() + 1e-9)).astype(np.float32)
    X = np.concatenate([Ta.reshape(nt, -1), Sa.reshape(nt, -1)], axis=1)
    ntr = max(int(nt * train_frac), 8)
    mu = X[:ntr].mean(0, keepdims=True)
    Xtr = X[:ntr] - mu
    G = Xtr @ Xtr.T
    w, U = np.linalg.eigh(G.astype(np.float64))
    o = np.argsort(w)[::-1][:n_modes]
    w, U = np.clip(w[o], 1e-12, None), U[:, o]
    E = (Xtr.T @ U) / np.sqrt(w)[None, :]          # (2*nx*ny, k) orthonormee
    Z = (X - mu) @ E                               # (nt, k) sur toute la periode
    return E.astype(np.float32), Z, ntr, float(Xtr.var(axis=0).sum())


# ══════════════════════════════════════════════════════════════════════════════
#  OPERATEUR EXACT, CAPTURE PENDANT LA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def capture_propagators(E, nt, seed, t_stride=1, verbose=True):
    """
    A_t = E^T M_t E, capture en rejouant la generation du nature run.

    t_stride > 1 n'enregistre qu'un pas sur t_stride (l'operateur varie
    lentement, a l'echelle de la duree de vie des tourbillons) ; les pas
    intermediaires reutilisent le dernier A capture.
    """
    nx, ny = NX, NY
    k = E.shape[1]
    bank = E.T.reshape(k, 2, nx, ny).astype(np.float64)   # (k, [T,S], nx, ny)

    dt_out = DT_DAYS
    nsub = max(1, int(N_SUBSTEPS))
    dt_s = dt_out * DAY / nsub
    dx = DX_KM * 1000.0
    kdt = KAPPA * dt_s / dx ** 2
    rT = dt_s / (TAU_T_DAYS * DAY)
    rS = dt_s / (TAU_S_DAYS * DAY)

    gen = SyntheticOceanGenerator()
    orig = gen._departure_stencil
    state = {"n": 0, "A": [], "idx": []}
    spin = int(SPINUP_DAYS / dt_out)

    def wrapper(u, v, _dt_s):
        st = orig(u, v, _dt_s)
        n = state["n"]; state["n"] += 1
        rec = n - spin
        if rec < 0 or rec % t_stride != 0:
            return st
        M = np.empty((2 * nx * ny, k), dtype=np.float32)
        for i in range(k):
            fT, fS = bank[i, 0].copy(), bank[i, 1].copy()
            for _ in range(nsub):
                fT = st.apply(fT); fT += kdt * _laplacian(fT); fT /= (1 + rT)
                fS = st.apply(fS); fS += kdt * _laplacian(fS); fS /= (1 + rS)
            M[:nx * ny, i] = fT.ravel()
            M[nx * ny:, i] = fS.ravel()
        state["A"].append((E.T @ M).astype(np.float64))
        state["idx"].append(rec)
        if verbose and len(state["A"]) % 50 == 0:
            print(f"    {len(state['A'])} propagateurs captures", flush=True)
        return st

    gen._departure_stencil = wrapper
    gen.generate_dataset(nt=nt, seed=seed)
    gen._departure_stencil = orig
    return np.array(state["A"]), np.array(state["idx"])


# ══════════════════════════════════════════════════════════════════════════════
#  FILTRE
# ══════════════════════════════════════════════════════════════════════════════

class Filter:
    """Filtre de Kalman en espace EOF, propagateur fixe ou variable."""

    def __init__(self, E, Z, ntr, var_total, H, R, lam, A, Q, A_seq=None,
                 idx_seq=None):
        self.lam, self.H, self.R = lam, H, R
        self.A, self.Q = A, Q
        self.A_seq, self.idx_seq = A_seq, idx_seq
        self.tr_tot = float(lam.sum())

    def _prop(self, t):
        if self.A_seq is None:
            return self.A, self.Q
        j = int(np.searchsorted(self.idx_seq, t % (self.idx_seq[-1] + 1),
                                side="right") - 1)
        return self.A_seq[max(j, 0)], self.Q

    def series(self, buoy_pos, up, spinup=60):
        n = up.shape[1]
        P = np.diag(self.lam).copy()
        out = np.empty(len(up))
        for t in list(range(-spinup, 0)) + list(range(len(up))):
            A, Q = self._prop(max(t, 0))
            P = A @ P @ A.T + Q
            act = np.flatnonzero(up[max(t, 0)])
            if len(act):
                sel = np.concatenate([buoy_pos[act], buoy_pos[act] + n])
                H = self.H[sel]
                Sm = H @ P @ H.T + np.diag(self.R[sel])
                K = np.linalg.solve(Sm, H @ P).T
                P = P - K @ (H @ P)
                P = 0.5 * (P + P.T)
            if t >= 0:
                out[t] = 1.0 - float(np.trace(P)) / self.tr_tot
        return out


def _stabilise(A):
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    return A * (0.999 / rho) if rho >= 1.0 else A


def _psd(Q):
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    return V @ np.diag(np.clip(w, 0, None)) @ V.T


# ══════════════════════════════════════════════════════════════════════════════
#  PILOTE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nt", type=int, default=NT)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--n_min", type=int, default=10)
    p.add_argument("--n_max", type=int, default=30)
    p.add_argument("--n_modes", type=int, default=120,
                   help="Doit etre assez grand pour que le reseau NE contraigne "
                        "PAS entierement l etat, sinon la variance expliquee "
                        "sature pres de 1 et ecrase les ecarts.")
    p.add_argument("--t_stride", type=int, default=3,
                   help="Un propagateur capture tous les t_stride jours. "
                        "L operateur varie a l echelle de la duree de vie des "
                        "tourbillons, donc 3 jours suffisent largement.")
    p.add_argument("--maintenance", type=str, default="pirata",
                   choices=["regional", "pirata"])
    p.add_argument("--budget_frac", type=float, default=0.45)
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--scenarios", type=int, default=6)
    p.add_argument("--out_dir", type=str, default="outputs")
    a = p.parse_args()

    spec = importlib.util.spec_from_file_location(
        "brick3", Path(__file__).with_name("03_rl.py"))
    b3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b3)
    from campaign import greedy_under_budget, auto_budget_levels, marginal_info
    from scenario import visit_calendar, simulate_uptime

    print(f"\n  Domaine {DOMAIN} {NX}x{NY} @ {DX_KM:.0f} km | "
          f"{a.n_modes} modes | operateur tous les {a.t_stride} j")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=a.nt, seed=a.seed)

    E, Z, ntr, var_tot = full_grid_eofs(T, S, a.n_modes)
    lam = Z[:ntr].var(axis=0) + 1e-12
    print(f"  EOF grille complete : {E.shape[1]} modes, "
          f"{lam.sum()/var_tot*100:.0f} % de la variance")

    # ── propagateur stationnaire (etage 2) ───────────────────────────────────
    Z0, Z1 = Z[:ntr - 1], Z[1:ntr]
    C0 = Z0.T @ Z0 / len(Z0)
    A_lim = _stabilise((Z1.T @ Z0 / len(Z0)) @ np.linalg.pinv(C0))
    Q_lim = _psd(C0 - A_lim @ C0 @ A_lim.T)

    # ── propagateur exact ─────────────────────────────────────────────────────
    print("  Capture de l operateur exact (rejeu du nature run)...", flush=True)
    A_seq, idx_seq = capture_propagators(E, a.nt, a.seed, a.t_stride)
    print(f"  {len(A_seq)} propagateurs captures")

    # bruit de modele du bras exact : residus reels z_{t+1} - A_t z_t
    res = []
    for t in range(ntr - 1):
        j = int(np.searchsorted(idx_seq, t, side="right") - 1)
        res.append(Z[t + 1] - A_seq[max(j, 0)] @ Z[t])
    res = np.array(res)
    Q_exact = _psd(res.T @ res / len(res))
    print(f"  Bruit de modele : LIM tr(Q)={np.trace(Q_lim):.3f}  |  "
          f"exact tr(Q)={np.trace(Q_exact):.3f}  "
          f"({np.trace(Q_exact)/max(np.trace(Q_lim),1e-9)*100:.0f} %)")

    # ── observations ──────────────────────────────────────────────────────────
    port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM
    maint = b3.MaintenanceModel(b3.get_params(a.maintenance), port)
    env = b3.OceanNetworkEnv(T, S, grid_x=a.grid_x, grid_y=a.grid_y,
                             n_min=a.n_min, n_max=a.n_max, fit_influence=True,
                             evf_cv=True, maintenance=maint)
    Ta = mesoscale_anomaly(T) / (T.std() + 1e-9)
    Sa = mesoscale_anomaly(S) / (S.std() + 1e-9)
    oT = np.stack([Ta[:ntr, x, y] for x, y in env.candidate_positions], 1)
    oS = np.stack([Sa[:ntr, x, y] for x, y in env.candidate_positions], 1)
    Y = np.concatenate([oT, oS], 1)
    Y = Y - Y.mean(0, keepdims=True)
    Hs, _, _, _ = np.linalg.lstsq(Z[:ntr] - Z[:ntr].mean(0), Y, rcond=None)
    H_all = Hs.T
    R_all = env._R_diag + (Y - (Z[:ntr] - Z[:ntr].mean(0)) @ Hs).var(axis=0)

    # ── reseau et scenarios ───────────────────────────────────────────────────
    lv, viable = auto_budget_levels(env, n_ref=a.n_max, fractions=(0.35,))
    g = greedy_under_budget(env, float(viable), "effective", verbose=False)
    idx = g["idx"]
    from policy import budget_envelope
    capex, floor, ceilb = budget_envelope(env, idx)
    budget = floor + a.budget_frac * (ceilb - floor)
    ev = env.evaluate(idx, budget_keur=budget, refine=True,
                      priority=marginal_info(env, idx), with_plan=True)
    cal = visit_calendar(ev["plan"], maint.p)
    print(f"  Reseau N={len(idx)} | budget {budget:.0f} k€/an\n")

    f_lim = Filter(E, Z, ntr, var_tot, H_all, R_all, lam, A_lim, Q_lim)
    f_exa = Filter(E, Z, ntr, var_tot, H_all, R_all, lam, A_lim, Q_exact,
                   A_seq=A_seq, idx_seq=idx_seq)

    s_lim, s_exa = [], []
    for k in range(a.scenarios):
        up = simulate_uptime(len(idx), cal, maint.p.mtbf_days, a.years,
                             np.random.default_rng(1000 + k))
        s_lim.append(f_lim.series(idx, up))
        s_exa.append(f_exa.series(idx, up))
    s_lim, s_exa = np.array(s_lim), np.array(s_exa)
    m_lim, m_exa = float(s_lim.mean()), float(s_exa.mean())
    marge = (m_exa - m_lim) / max(m_lim, 1e-9) * 100

    # L'aire sous courbe est une variance EXPLIQUEE : elle sature pres de 1
    # des que le reseau contraint bien les modes retenus, et ecrase alors les
    # ecarts. L'erreur RESIDUELLE (1 - EVF) ne sature pas et donne la seconde
    # lecture, souvent tres differente.
    e_lim, e_exa = 1.0 - m_lim, 1.0 - m_exa
    red = (e_lim - e_exa) / max(e_lim, 1e-9) * 100

    print(f"  {'estimateur':>28} | {'variance expliquee':>18} | "
          f"{'erreur residuelle':>17}")
    print("  " + "-" * 70)
    print(f"  {'LIM stationnaire (etage 2)':>28} | {m_lim:>18.4f} | "
          f"{e_lim:>17.4f}")
    print(f"  {'KF a operateur exact':>28} | {m_exa:>18.4f} | "
          f"{e_exa:>17.4f}")
    print(f"\n  MARGE POUR UN MODELE APPRIS")
    print(f"    en variance expliquee : {marge:+.1f} %")
    print(f"    en erreur residuelle  : -{red:.0f} %")

    if m_lim > 0.90:
        print(f"\n  [ATTENTION] Variance expliquee > 0,90 : le filtre est en\n"
              f"  regime SATURE. Avec {a.n_modes} modes et {len(idx)} bouees,\n"
              f"  l'etat est quasi entierement contraint et la metrique ecrase\n"
              f"  les ecarts. Les deux lectures ci-dessus divergent fortement,\n"
              f"  ce qui est le symptome. Relancer avec --n_modes 100 ou plus\n"
              f"  pour que le systeme soit sous-determine, comme il l'est en\n"
              f"  realite, avant de conclure quoi que ce soit sur la marge.")

    if m_lim > 0.90:
        # Rendre un verdict ici reviendrait a conclure juste apres avoir
        # explique qu'on ne peut pas conclure. Le script se contredisait.
        print("\n  -> VERDICT SUSPENDU tant que la mesure est saturee.\n"
              "     Relancer avec --n_modes 120 (le cout de capture croit\n"
              "     lineairement avec le nombre de modes).")
    elif abs(marge) < 5:
        print("  -> L'etage 3 est sans objet sur cet ocean : le propagateur\n"
              "     stationnaire capture deja presque tout. Un AE recurrent ne\n"
              "     pourrait qu'egaler le LIM, au prix d'un entrainement.")
    elif marge < 15:
        print("  -> Marge modeste. Un AE recurrent devrait recuperer l'essentiel\n"
              "     de cet ecart pour valoir son cout.")
    else:
        print("  -> Marge substantielle : la dependance a l'ecoulement compte,\n"
              "     l'etage 3 a un objectif clair et chiffre.")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=BG)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
    x = np.arange(s_lim.shape[1]) / 365.0
    ax.plot(x, s_lim.mean(0), color="#ffd93d", lw=1.6,
            label=f"LIM stationnaire ({m_lim:.4f})")
    ax.plot(x, s_exa.mean(0), color="#6bcb77", lw=1.6,
            label=f"operateur exact — plafond ({m_exa:.4f})")
    ax.fill_between(x, s_lim.mean(0), s_exa.mean(0), color="#6bcb77",
                    alpha=0.18)
    ax.set_title("Ce qu'un modele appris pourrait au mieux recuperer",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Annees", color="white")
    ax.set_ylabel("Variance expliquee", color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=9, labelcolor="white", facecolor=BG, edgecolor=EDGE)
    ax.grid(alpha=0.15, color="white")
    fig.savefig(out / "ceiling_stage3.png", dpi=145, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    (out / "ceiling_stage3.json").write_text(json.dumps(
        {"lim": m_lim, "exact": m_exa, "margin_pct": marge,
         "residual_lim": e_lim, "residual_exact": e_exa,
         "residual_reduction_pct": red, "saturated": bool(m_lim > 0.90),
         "n_modes": int(a.n_modes), "n_buoys": int(len(idx)),
         "budget_keur": float(budget)}, indent=2), encoding="utf-8")
    print(f"\n  Figure -> {out / 'ceiling_stage3.png'}")


if __name__ == "__main__":
    main()
