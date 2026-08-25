"""
oed_core.py — Noyau OED/DA commun aux briques NAIADE.

Remplace le critere d'information heuristique par l'ERREUR D'ANALYSE EXACTE
d'une assimilation lineaire, calculee dans une base EOF reduite.

    x  = U a           x : anomalies (T,S) normalisees sur la grille d'eval
                       U : (m, k) EOF orthonormees, a ~ N(0, diag(lam))
    y  = H x + eps     H : selection aux positions de bouees, eps ~ N(0, R)

    Pa_red = ( diag(1/lam) + (HU)^T R^-1 (HU) )^-1                        (1)

Tout le vocabulaire OED en decoule, pour un cout de k x k :

    A-optimalite    tr(Pa)                     variance d'analyse residuelle
    fraction resolue 1 - tr(Pa)/sum(lam)       equivalent EVF, dans [0,1]
    D-optimalite    logdet(diag(lam)) - logdet(Pa)   gain d'information
    DFS             k - tr(Pa / lam)           degres de liberte du signal

Interet pour NAIADE : R est DIAGONALE ET DEPEND DE L'AGE de chaque bouee.
Entretenir une bouee = remettre son age a zero = reduire tr(Pa). Le benefice
d'une intervention de maintenance devient donc une quantite physique
calculable, dans la meme unite que le benefice d'un deploiement neuf.

Numpy pur : aucune dependance a torch, la brique tourne en CI.
"""

import numpy as np
from pathlib import Path

from config import (DX_KM, PORT_XY_FRAC, COST_BUOY_FIXED, COST_SHIP_PER_KM,
                    CO2_SHIP_PER_KM, OBS_NOISE_T, OBS_NOISE_S,
                    INFLUENCE_RADIUS_KM, MIN_BUOY_SEP_KM)

DAYS_PER_YEAR = 365.0


# =============================================================================
#  1. Nature run long, avec cache
# =============================================================================
def build_nature_run(nx=192, ny=288, nt=13 * 365, stride=4, n_eddies=45,
                     seed=42, pert_amp=None, cache_dir="data/cache"):
    """
    Nature run pluriannuel, plus grand et plus chaotique que la demo.

    `pert_amp` surcharge PERT_AMP (amplitude de la perturbation stochastique
    non resolue sur psi) : c'est le bouton "chaos". `data.dataset` fait
    `from config import *`, donc la constante doit etre patchee dans le module
    lui-meme, pas dans config.

    Retourne dict(T, S, stride, dx_km, shape) avec T, S de forme (nt, nxc, nyc).
    """
    import data.dataset as dsm

    tag = f"nr_{nx}x{ny}_nt{nt}_st{stride}_ed{n_eddies}_s{seed}"
    if pert_amp is not None:
        tag += f"_pa{pert_amp:g}"
    cache = Path(cache_dir) / f"{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return dict(T=z["T"], S=z["S"], stride=int(z["stride"]),
                    dx_km=float(z["dx_km"]), cached=True)

    if pert_amp is not None:
        dsm.PERT_AMP = float(pert_amp)
    gen = dsm.SyntheticOceanGenerator(nx=nx, ny=ny, n_eddies=n_eddies, seed=seed)
    T, S = gen.generate_light(nt=nt, seed=seed, stride=stride)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, T=T, S=S, stride=stride, dx_km=DX_KM * stride)
    return dict(T=T, S=S, stride=stride, dx_km=DX_KM * stride, cached=False)


# =============================================================================
#  2. Deseasonnalisation
# =============================================================================
def harmonic_climatology(F, t_days, n_harm=2, coefs=None):
    """
    Climatologie par pixel = regression harmonique (annuelle + semi-annuelle).

    `coefs` permet d'AJUSTER SUR LA PERIODE D'APPRENTISSAGE et de retirer la
    meme climatologie sur la periode de test : sans cela le cycle saisonnier
    du test fuiterait dans l'evaluation.
    """
    X = [np.ones_like(t_days)]
    for h in range(1, n_harm + 1):
        w = 2 * np.pi * h / DAYS_PER_YEAR
        X += [np.cos(w * t_days), np.sin(w * t_days)]
    X = np.stack(X, axis=1)                                  # (nt, 1+2h)
    flat = F.reshape(len(F), -1)
    if coefs is None:
        coefs = np.linalg.lstsq(X, flat, rcond=None)[0]      # (1+2h, m)
    return (X @ coefs).reshape(F.shape), coefs


# =============================================================================
#  3. Base EOF
# =============================================================================
def randomized_svd(A, k, n_oversample=15, n_iter=3, rng=None):
    """SVD tronquee randomisee (Halko et al.) — evite la SVD complete (nt, m)."""
    rng = rng or np.random.default_rng(0)
    n = A.shape[1]
    Q = rng.standard_normal((n, k + n_oversample))
    Q, _ = np.linalg.qr(A @ Q)
    for _ in range(n_iter):                                  # power iterations
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


class EOFBasis:
    """
    Base reduite construite sur les anomalies (T, S) de la periode d'apprentissage.

    T et S sont normalises SEPAREMENT par leur ecart-type local avant empilement :
    sans cela var(SST) ~ 5 degC2 ecrase var(SSS) ~ 0.08 psu2 et la salinite ne
    pese pour rien dans le critere — le meme piege que celui documente dans
    03_rl.py.
    """

    def __init__(self, T, S, train_slice, k=80, n_harm=2, seed=0):
        self.shape = T.shape[1:]
        self.m_cells = int(np.prod(self.shape))
        t = np.arange(len(T), dtype=np.float64)

        # --- climatologie ajustee sur le train, retiree partout -------------
        clim_T, cT = harmonic_climatology(T[train_slice], t[train_slice], n_harm)
        clim_S, cS = harmonic_climatology(S[train_slice], t[train_slice], n_harm)
        AT = T - harmonic_climatology(T, t, n_harm, coefs=cT)[0]
        AS = S - harmonic_climatology(S, t, n_harm, coefs=cS)[0]

        # --- normalisation par la variabilite locale du train ---------------
        self.sig_T = AT[train_slice].std(0).ravel() + 1e-9
        self.sig_S = AS[train_slice].std(0).ravel() + 1e-9
        self.A = np.concatenate([AT.reshape(len(T), -1) / self.sig_T,
                                 AS.reshape(len(S), -1) / self.sig_S],
                                axis=1).astype(np.float64)        # (nt, 2*m_cells)
        self.m = self.A.shape[1]

        Atr = self.A[train_slice]
        self.mean = Atr.mean(0)
        Atr = Atr - self.mean
        k = int(min(k, min(Atr.shape) - 1))
        _, s, Vt = randomized_svd(Atr, k, rng=np.random.default_rng(seed))
        self.U = np.ascontiguousarray(Vt.T)                       # (m, k)
        self.lam = (s ** 2) / max(len(Atr) - 1, 1)                # variances modales
        self.k = k
        self.train_slice = train_slice
        self.var_total_train = float(((Atr) ** 2).sum(1).mean())
        self.var_explained = float(self.lam.sum() / self.var_total_train)

    # -- indices de lignes observees par une bouee posee sur la cellule c ----
    def obs_rows(self, cells):
        cells = np.asarray(cells, dtype=int)
        return np.concatenate([cells, cells + self.m_cells])

    def coeffs(self, sl):
        """Coefficients EOF verite des instants `sl` (pour l'evaluation OOS)."""
        return (self.A[sl] - self.mean) @ self.U


# =============================================================================
#  4. Modele d'erreur d'observation vieillissante
# =============================================================================
class SensorAgeing:
    """
    Une bouee ne tombe pas en panne d'un coup : biofouling, derive de capteur,
    perte de la bouee de surface. On fait donc croitre sigma_obs depuis la
    derniere intervention, et on superpose un risque de perte totale (Weibull).

        sigma(age)^2 = [sigma0 (1 + (age/tau)^p)]^2 + sigma_repr^2

    sigma_repr : erreur de representativite (une bouee est un point, la maille
    en fait 20 km) — souvent dominante et systematiquement oubliee.
    """

    def __init__(self, sig_T_map, sig_S_map,
                 tau_days=300.0, p=1.6, repr_frac=0.25,
                 weibull_k=1.8, weibull_lam=550.0):
        self.rT0 = (OBS_NOISE_T / sig_T_map) ** 2      # bruit instrumental, unites normalisees
        self.rS0 = (OBS_NOISE_S / sig_S_map) ** 2
        self.repr2 = repr_frac ** 2                    # fraction de la variance locale
        self.tau, self.p = tau_days, p
        self.wk, self.wlam = weibull_k, weibull_lam

    def r_diag(self, cells, ages):
        """Diagonale de R pour des bouees d'ages donnes (lignes T puis S)."""
        cells = np.asarray(cells, int)
        g = (1.0 + (np.asarray(ages, float) / self.tau) ** self.p) ** 2
        return np.concatenate([self.rT0[cells] * g + self.repr2,
                               self.rS0[cells] * g + self.repr2])

    def survival(self, age0, dt):
        """P(survivre dt jours de plus | age atteint age0), loi de Weibull."""
        a0, a1 = np.asarray(age0, float) / self.wlam, (np.asarray(age0, float) + dt) / self.wlam
        return np.exp(a0 ** self.wk - a1 ** self.wk)


# =============================================================================
#  5. Posterieur d'assimilation et criteres OED
# =============================================================================
class AnalysisError:
    """Evalue (1) pour un reseau donne. Cout O(k^3), k ~ 80 -> ~0.5 ms."""

    def __init__(self, basis: EOFBasis, ageing: SensorAgeing):
        self.b, self.ag = basis, ageing
        self.inv_lam = 1.0 / basis.lam

    def posterior(self, cells, ages):
        """Retourne (Pa_red (k,k), HU (2n,k), Rinv (2n,))."""
        if len(cells) == 0:
            return np.diag(self.b.lam), np.zeros((0, self.b.k)), np.zeros(0)
        rows = self.b.obs_rows(cells)
        HU = self.b.U[rows]                                   # (2n, k)
        Rinv = 1.0 / self.ag.r_diag(cells, ages)
        P = np.diag(self.inv_lam) + HU.T @ (Rinv[:, None] * HU)
        L = np.linalg.cholesky(P)
        Pa = np.linalg.inv(L).T @ np.linalg.inv(L)            # = P^-1, SPD
        return Pa, HU, Rinv

    # ---- criteres -----------------------------------------------------------
    def metrics(self, cells, ages):
        Pa, HU, _ = self.posterior(cells, ages)
        tr = float(np.trace(Pa))
        lam_sum = float(self.b.lam.sum())
        sign, logdet = np.linalg.slogdet(Pa)
        return dict(
            trace=tr,
            resolved=1.0 - tr / lam_sum,                              # ~ EVF
            dfs=float(self.b.k - (np.diag(Pa) * self.inv_lam).sum()),
            logdet_gain=float(np.log(self.b.lam).sum() - logdet),
            n=len(cells),
        )

    def resolved(self, cells, ages):
        Pa, _, _ = self.posterior(cells, ages)
        return 1.0 - float(np.trace(Pa)) / float(self.b.lam.sum())

    # ---- verification hors echantillon --------------------------------------
    def resolved_oos(self, cells, ages, test_slice, rng=None):
        """
        Variance reellement expliquee sur une periode JAMAIS VUE : on simule les
        observations depuis la verite, on applique le BLUE, on mesure le residu.
        C'est le chiffre honnete, celui qui devenait negatif avec la covariance
        empirique brute (cf. commentaire EVF_SHRINKAGE de config.py).
        """
        rng = rng or np.random.default_rng(0)
        a_true = self.b.coeffs(test_slice)                            # (nt, k)
        if len(cells) == 0:
            return 0.0
        Pa, HU, Rinv = self.posterior(cells, ages)
        y = a_true @ HU.T
        y = y + rng.standard_normal(y.shape) * np.sqrt(1.0 / Rinv)
        a_hat = (Pa @ (HU.T @ (Rinv[:, None] * y.T))).T
        num = ((a_true - a_hat) ** 2).sum()
        den = (a_true ** 2).sum()
        return float(1.0 - num / den)

    def marginal_gain(self, cells, ages, candidates, new_age=0.0):
        """Gain de variance resolue de l'ajout de chaque candidat (boucle exacte)."""
        base = self.resolved(cells, ages)
        out = np.empty(len(candidates))
        for i, c in enumerate(candidates):
            out[i] = self.resolved(list(cells) + [c], list(ages) + [new_age]) - base
        return out

    def marginal_gain_fast(self, cells, ages, cand_cells, cand_age=None,
                           new_age=0.0, Pa=None):
        """
        Gain marginal de TOUS les candidats en une passe, par mise a jour de
        Kalman de rang 1 :

            delta_tr = tr[ (D^-1 + H Pa H^T)^-1 (H Pa)(H Pa)^T ]

        Le meme calcul couvre les deux decisions, ce qui est exactement ce qui
        rend l'entretien et le deploiement comparables dans la meme unite :

            deploiement -> d = 1 / r(age_neuf)              (obs nouvelle)
            entretien   -> d = 1 / r(0) - 1 / r(age)        (gain de precision)

        Vectorise : ~0.2 ms pour 200 candidats, contre ~50 ms pour la boucle
        naive candidat par candidat. L'approximation ne porte que sur la non-orthogonalite des deux
        lignes (T et S) d'une meme bouee.
        """
        if Pa is None:
            Pa, _, _ = self.posterior(cells, ages)
        cand_cells = np.asarray(cand_cells, int)
        rows = self.b.obs_rows(cand_cells)
        H = self.b.U[rows]                                    # (2K, k)
        r_new = self.ag.r_diag(cand_cells, np.full(len(cand_cells), new_age))
        d = 1.0 / r_new
        if cand_age is not None:                              # bouees deja en place
            occ = np.isfinite(cand_age)
            if occ.any():
                a = np.where(occ, cand_age, 0.0)
                r_old = self.ag.r_diag(cand_cells, a)
                d2 = 1.0 / r_new - 1.0 / r_old
                m = np.concatenate([occ, occ])
                d = np.where(m, np.maximum(d2, 0.0), d)
        K = len(cand_cells)
        H1, H2 = H[:K], H[K:]                                 # lignes T et S
        d1, d2 = d[:K], d[K:]
        M1, M2 = H1 @ Pa, H2 @ Pa
        A11 = (M1 * H1).sum(1); A12 = (M1 * H2).sum(1); A22 = (M2 * H2).sum(1)
        G11 = (M1 * M1).sum(1); G12 = (M1 * M2).sum(1); G22 = (M2 * M2).sum(1)
        big = 1e12
        S11 = A11 + np.where(d1 > 0, 1.0 / np.maximum(d1, 1e-300), big)
        S22 = A22 + np.where(d2 > 0, 1.0 / np.maximum(d2, 1e-300), big)
        det = np.maximum(S11 * S22 - A12 ** 2, 1e-300)
        drop = (S22 * G11 - 2.0 * A12 * G12 + S11 * G22) / det
        return drop / float(self.b.lam.sum())


# =============================================================================
#  6. Geometrie, contraintes, cout de campagne
# =============================================================================
class Domain:
    """Grille de candidats + distances physiques + tournee navire."""

    def __init__(self, shape, dx_km, min_sep_km=MIN_BUOY_SEP_KM):
        self.shape = shape
        self.nxc, self.nyc = shape
        self.dx_km = dx_km
        gx, gy = np.meshgrid(np.arange(self.nxc), np.arange(self.nyc), indexing="ij")
        self.xy_km = np.stack([gx.ravel(), gy.ravel()], 1).astype(float) * dx_km
        self.port = np.array([PORT_XY_FRAC[0] * self.nxc,
                              PORT_XY_FRAC[1] * self.nyc]) * dx_km
        self.min_sep_km = min_sep_km
        self.d_port = np.linalg.norm(self.xy_km - self.port, axis=1)

    def dist(self, a, b):
        return np.linalg.norm(self.xy_km[np.asarray(a, int)][:, None]
                              - self.xy_km[np.asarray(b, int)][None], axis=2)

    def feasible(self, cand, active):
        """Masque des candidats respectant la separation minimale."""
        ok = np.ones(len(cand), bool)
        if len(active):
            d = self.dist(cand, active)
            ok &= d.min(1) >= self.min_sep_km
        return ok

    def tour_length(self, cells):
        """Tournee port -> bouees -> port : plus proche voisin puis 2-opt."""
        if len(cells) == 0:
            return 0.0
        pts = self.xy_km[np.asarray(cells, int)]
        n = len(pts)
        rest, cur, order = list(range(n)), self.port, []
        while rest:
            d = np.linalg.norm(pts[rest] - cur, axis=1)
            j = int(np.argmin(d)); order.append(rest[j]); cur = pts[rest[j]]; rest.pop(j)
        route = np.vstack([self.port, pts[order], self.port])

        def L(r):
            return float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum())

        best = L(route)
        improved = True
        while improved and n > 3:                       # 2-opt
            improved = False
            for i in range(1, n):
                for j in range(i + 1, n + 1):
                    cand = route.copy(); cand[i:j] = cand[i:j][::-1]
                    lc = L(cand)
                    if lc < best - 1e-9:
                        route, best, improved = cand, lc, True
        return best

    def campaign_cost(self, serviced, n_active, n_new=0):
        """
        Cout d'une campagne (k EUR) et CO2 (t).
        Le navire ne visite QUE les bouees traitees : grouper geographiquement
        les interventions devient donc payant, ce qui rend l'arbitrage reel.
        """
        km = self.tour_length(serviced)
        cost = (0.5 * COST_BUOY_FIXED * n_active            # amortissement semestriel
                + km * COST_SHIP_PER_KM
                + 0.35 * COST_BUOY_FIXED * n_new)           # materiel neuf
        return float(cost), float(km * CO2_SHIP_PER_KM), float(km)
