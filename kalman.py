"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ETAGE 2  —  EVALUATEUR DE KALMAN SUR EOF (propagateur AR(1) par mode)       ║
║                                                                              ║
║  Le critere statique evalue chaque jour independamment : une bouee morte     ║
║  cesse d'informer INSTANTANEMENT, et une bouee reparee informe pleinement    ║
║  des le premier jour. Les deux sont faux. L'information d'une observation    ║
║  persiste le temps que l'ocean se decorrele (~12 jours ici), et l'erreur     ║
║  d'analyse recroit progressivement vers la climatologie apres une panne.     ║
║                                                                              ║
║  Ce module remplace l'evaluateur, pas les politiques :                       ║
║                                                                              ║
║      etat        z_t  =  coefficients des k premieres EOF du nature run      ║
║      propagation z_{t+1} = A z_t + w,   A = C(1) C(0)^-1 (modele inverse     ║
║                                          lineaire, matrice PLEINE)           ║
║      observation y_t = H z_t + v,       H = regression des bouees sur EOF    ║
║      metrique    EVF_t = 1 - trace(P_t) / trace(Lambda)                      ║
║                                                                              ║
║  Deux proprietes rendent la chose peu couteuse et exacte :                   ║
║                                                                              ║
║  1. En lineaire-gaussien, la covariance d'erreur P ne depend QUE de la       ║
║     sequence de masques, pas des valeurs observees. On propage donc P seul,  ║
║     sans simuler la moindre mesure. Matrices k x k : quelques secondes pour  ║
║     tous les scenarios.                                                      ║
║                                                                              ║
║  2. Les bouees ne tombent pas sur les mailles d'evaluation. Plutot que       ║
║     d'interpoler, on regresse la valeur au point de la bouee sur les EOF :   ║
║     la ligne de regression donne H, et la variance residuelle devient une    ║
║     erreur de representativite ajoutee a R. Ce que les EOF retenues ne       ║
║     savent pas representer est ainsi compte comme du bruit, pas ignore.      ║
║                                                                              ║
║  BONUS non trivial : contrairement au critere statique, celui-ci n'a PAS     ║
║  besoin du noyau gaussien isotrope dont on a mesure l'erreur de modele       ║
║  (RMS 0,18 en correlation, incapable de representer les anticorrelations).   ║
║  Les EOF portent la vraie structure spatiale, anticorrelations comprises.    ║
║                                                                              ║
║  POURQUOI PAS UN AR(1) PAR MODE. C'etait la premiere version, et elle est    ║
║  fausse d'un facteur 10 sur le temps de decorrelation (diag_ar1.py) :        ║
║                                                                              ║
║   - ajuste au decalage 1, il lit une autocorrelation de 0,999 sur un champ   ║
║     spatialement lisse et en extrapole une memoire de plusieurs annees, la   ║
║     ou la decroissance reelle se fait en quelques semaines ;                 ║
║   - sa fonction d'autocorrelation a^lag est positive par construction, donc  ║
║     incapable de representer les autocorrelations NEGATIVES observees sur    ║
║     la majorite des modes -- signature d'une structure qui se propage et     ║
║     reexcite l'EOF en sens inverse.                                          ║
║                                                                              ║
║  Une matrice pleine peut avoir des valeurs propres complexes, donc           ║
║  representer rotation et propagation, pas seulement relaxation. Le cout est  ║
║  identique (k x k).                                                          ║
║                                                                              ║
║  LIMITES restantes. Systeme suppose lineaire-gaussien : pas d'erreur du      ║
║  jour, pas de dependance a l'ecoulement. Il faudrait un EnKF pour cela.      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import numpy as np

from config import DX_KM
from data.dataset import mesoscale_anomaly


class KalmanEOF:
    """Evaluateur sequentiel. Interface calquee sur `evf_series`."""

    def __init__(self, env, n_modes=50, train_frac=0.5, verbose=True,
                 propagator="lim"):
        self.env = env
        Ta = mesoscale_anomaly(env.T) / (env.T.std() + 1e-9)
        Sa = mesoscale_anomaly(env.S) / (env.S.std() + 1e-9)
        st, nt = env.eval_stride, len(Ta)

        # ── champ a reconstruire : mailles d'evaluation, T puis S ────────────
        X = np.concatenate([Ta[:, ::st, ::st].reshape(nt, -1),
                            Sa[:, ::st, ::st].reshape(nt, -1)], axis=1)
        ntr = max(int(nt * train_frac), 8)
        Xtr = X[:ntr] - X[:ntr].mean(0, keepdims=True)

        # ── EOF ───────────────────────────────────────────────────────────────
        k = int(min(n_modes, Xtr.shape[0] - 2, Xtr.shape[1]))
        _, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        self.E = Vt[:k].T                       # (2M, k) orthonormee
        Z = Xtr @ self.E                        # (ntr, k) composantes
        self.lam = Z.var(axis=0) + 1e-12        # variance par mode
        self.k = k
        self.var_total = float(Xtr.var(axis=0).sum())
        self.frac_resolved = float(self.lam.sum() / max(self.var_total, 1e-12))

        # ── propagateur ───────────────────────────────────────────────────────
        Z0, Z1 = Z[:-1], Z[1:]
        C0 = Z0.T @ Z0 / len(Z0)
        if propagator == "ar1":
            a = np.array([np.corrcoef(Z1[:, i], Z0[:, i])[0, 1]
                          for i in range(k)])
            a = np.clip(np.nan_to_num(a), 0.0, 0.999)
            self.A = np.diag(a)
            self.Q = np.diag(self.lam * (1.0 - a ** 2))
        else:
            C1 = Z1.T @ Z0 / len(Z0)
            A = C1 @ np.linalg.pinv(C0)
            # stabilite : le propagateur ne doit pas amplifier
            rho = np.max(np.abs(np.linalg.eigvals(A)))
            if rho >= 1.0:
                A = A * (0.999 / rho)
            Q = C0 - A @ C0 @ A.T
            Q = 0.5 * (Q + Q.T)
            w, V = np.linalg.eigh(Q)                 # projection PSD
            self.A, self.Q = A, V @ np.diag(np.clip(w, 0, None)) @ V.T
        self.propagator = propagator
        self.tau = self._implied_tau(C0)

        # ── operateur d'observation : regression des bouees sur les EOF ──────
        oT = np.stack([Ta[:ntr, x, y] for x, y in env.candidate_positions], 1)
        oS = np.stack([Sa[:ntr, x, y] for x, y in env.candidate_positions], 1)
        Y = np.concatenate([oT, oS], axis=1)          # (ntr, 2K)
        Y = Y - Y.mean(0, keepdims=True)
        Hs, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)   # (k, 2K)
        self.H_all = Hs.T                                  # (2K, k)
        resid = Y - Z @ Hs
        self.r_repr = resid.var(axis=0)                    # representativite
        self.R_all = env._R_diag + self.r_repr             # instrument + repr.

        if verbose:
            print(f"  Kalman EOF    : {k} modes ({propagator}), "
                  f"{self.frac_resolved*100:.0f} % de la variance | tau "
                  f"median {np.median(self.tau):.1f} j | representativite "
                  f"mediane {np.median(self.r_repr):.3f}")

    def _implied_tau(self, C0, max_lag=200):
        """
        Temps de decorrelation IMPLIQUE par le propagateur, mode par mode :
        r_i(lag) = [A^lag C0]_ii / C0_ii, premier passage sous 1/e.

        C'est la quantite a comparer a l'autocorrelation empirique. La juger
        sur la prevision a un jour ne dit rien : l'AR(1) y est excellent
        (0,97) tout en se trompant d'un facteur 10 sur la memoire, parce que
        c'est l'extrapolation aux longs decalages qui derape.
        """
        d = np.clip(np.diag(C0), 1e-12, None)
        M = np.eye(len(d))
        tau = np.full(len(d), float(max_lag))
        seen = np.zeros(len(d), dtype=bool)
        for lag in range(1, max_lag + 1):
            M = self.A @ M
            r = np.diag(M @ C0) / d
            hit = (~seen) & (r < 1 / np.e)
            tau[hit] = lag
            seen |= hit
            if seen.all():
                break
        return tau

    # -------------------------------------------------------------------------

    def _rows(self, idx):
        """Lignes de H et bruits R pour les canaux T et S des bouees `idx`."""
        idx = np.asarray(idx, dtype=int)
        sel = np.concatenate([idx, idx + self.env.K])
        return self.H_all[sel], self.R_all[sel]

    def series(self, idx, up, spinup=60):
        """
        EVF jour par jour pour la sequence de disponibilite `up` (T, n).

        Un rodage de `spinup` jours precede la periode evaluee pour que P
        parte d'un regime etabli plutot que de la climatologie -- sans quoi
        les premieres semaines de chaque scenario seraient artificiellement
        mauvaises et biaiseraient l'aire sous la courbe.
        """
        idx = np.asarray(idx, dtype=int)
        H_all, R_all = self._rows(idx)
        n = len(idx)
        A, Q, lam = self.A, self.Q, self.lam
        tr_tot = float(lam.sum())

        P = np.diag(lam).copy()
        out = np.empty(len(up))
        order = list(range(-spinup, 0)) + list(range(len(up)))
        for t in order:
            P = A @ P @ A.T + Q                              # propagation
            row = up[max(t, 0)]
            act = np.flatnonzero(row)
            if len(act):
                sel = np.concatenate([act, act + n])
                H = H_all[sel]
                S = H @ P @ H.T + np.diag(R_all[sel])
                K = np.linalg.solve(S, H @ P).T                # P H^T S^-1
                P = P - K @ (H @ P)
                P = 0.5 * (P + P.T)
            if t >= 0:
                out[t] = 1.0 - float(np.trace(P)) / tr_tot
        return out


def make_evaluator(env, kind="static", n_modes=50, verbose=True,
                   propagator="lim"):
    """
    Fabrique l'evaluateur demande.

    "static" : critere BLUE instantane (etage 0/1), sans memoire temporelle
    "kalman" : filtre de Kalman EOF/AR(1) (etage 2)

    Les deux exposent la meme signature `f(idx, up) -> serie`, ce qui permet
    de rejouer EXACTEMENT les memes scenarios et de mesurer ce que le critere
    statique manquait.
    """
    if kind == "kalman":
        kf = KalmanEOF(env, n_modes=n_modes, verbose=verbose,
                       propagator=propagator)
        return lambda idx, up: kf.series(idx, up)

    def static(idx, up):
        idx = np.asarray(idx, dtype=int)
        out = np.empty(len(up))
        cache: dict[tuple, float] = {}
        for t in range(len(up)):
            key = tuple(np.flatnonzero(up[t]).tolist())
            v = cache.get(key)
            if v is None:
                v = env.explained_variance(idx[list(key)]) if key else 0.0
                cache[key] = v
            out[t] = v
        return out
    return static
