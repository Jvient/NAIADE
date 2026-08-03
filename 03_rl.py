"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BRIQUE 3 — Reinforcement Learning : Optimisation du Réseau          ║
║                                                                              ║
║  Formalisation MDP :                                                         ║
║    État    s_t : masque binaire actuel (grille grossière) + stats champ     ║
║    Action  a_t : activer / désactiver une des K positions candidates        ║
║    Récompense  : gain de reconstruction RMSE − pénalité budget bouées      ║
║                                                                              ║
║  Algorithme : PPO (Proximal Policy Optimization) implémenté en PyTorch pur  ║
║  Multi-objectif : front de Pareto information vs nombre de capteurs         ║
║                                                                              ║
║  Usage :                                                                     ║
║    python 03_rl.py --train                                                   ║
║    python 03_rl.py --pareto           (front Pareto info/nb capteurs)       ║
║    python 03_rl.py --train --pareto                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, argparse
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data.dataset import (SyntheticOceanGenerator, local_variance_map,
                          mesoscale_anomaly)

# ─── Import optionnel de la Brique 1 pour la récompense dense ─────────────────
try:
    from brique1_autoencoder import ObservabilityVAE
    AE_AVAILABLE = True
except ImportError:
    AE_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONNEMENT MDP
# ══════════════════════════════════════════════════════════════════════════════

class OceanNetworkEnv:
    """
    Environnement RL pour l'optimisation du réseau d'observation.

    Grille candidate (coarse grid) :
        On discrétise l'espace en une grille GX × GY de positions candidates.
        L'espace d'action est donc de taille K = GX × GY (toggle par position).
        Le budget de bouées actives est contraint : [n_min, n_max].

    État s_t : (K + n_stats,) float32
        - K premiers éléments : masque binaire des positions actives
        - n_stats derniers : statistiques globales du champ nature run
          (variance locale agrégée, gradient moyen...)

    Récompense r_t :
        r_t = w_info * r_info − w_budget * r_budget

        r_info    = amélioration de la couverture pondérée variance
                    (proxy : couverture variance locale)
        r_budget  = pénalité si hors de la plage [n_min, n_max] bouées actives

    Épisode :
        T_ep actions consécutives → à la fin, évaluation de la configuration finale
    """

    def __init__(self, T, S,
                 grid_x=16, grid_y=24,    # résolution de la grille candidate
                 n_min=10, n_max=40,       # plage de bouées actives
                 episode_len=20,           # actions par épisode
                 w_info=1.0, w_budget=0.5,
                 info_mode="evf",          # "evf" (defaut) | "coverage" | "legacy"
                 influence_km=INFLUENCE_RADIUS_KM,
                 info_gain=RL_INFO_GAIN,
                 eval_stride=8,            # sous-echantillonnage de la grille
                 evf_cv=False,             # score EVF valide hors echantillon
                 min_sep=MIN_SEP_CELLS,    # separation mini en cases de grille
                 ae_model=None):           # autoencoder optionnel (Brique 1)
        self.T = T.astype(np.float32)
        self.S = S.astype(np.float32)
        self.grid_x  = grid_x
        self.grid_y  = grid_y
        self.K       = grid_x * grid_y    # nb de positions candidates
        self.n_min   = n_min
        self.n_max   = n_max
        self.ep_len  = episode_len
        self.w_info, self.w_budget = w_info, w_budget
        self.info_mode    = info_mode
        self.influence_px = influence_km / DX_KM
        self.info_gain    = info_gain if info_mode in ("coverage", "evf") else 1.0
        self.eval_stride  = eval_stride
        self.evf_cv       = bool(evf_cv)
        self.min_sep      = int(min_sep)
        self.ae_model = ae_model
        self.nt = len(T)

        # Positions physiques des K candidats (centre de chaque cellule)
        self.candidate_positions = []
        sx = NX / grid_x
        sy = NY / grid_y
        for gx in range(grid_x):
            for gy in range(grid_y):
                px = int(gx * sx + sx / 2)
                py = int(gy * sy + sy / 2)
                self.candidate_positions.append((min(px, NX-1), min(py, NY-1)))

        # Contrainte : deux bouées ne peuvent pas occuper des cases adjacentes
        self._build_conflict_matrix()

        # Statistiques globales du nature run (pré-calculées une fois)
        self._precompute_field_stats()

        # État courant
        self.active_mask = None    # (K,) binaire
        self.step_count  = 0
        self.t_current   = 0
        self.obs_dim = self.K + len(self.field_stats)

    def _build_conflict_matrix(self):
        """
        `_conflict[i, j] = True` si les candidats i et j sont trop proches pour
        être actifs simultanément (distance de Tchebychev < min_sep sur la
        grille candidate, ou de Manhattan si MIN_SEP_DIAGONAL est False).

        Plafond de faisabilité : avec min_sep = 2 on ne peut activer qu'une
        case sur deux dans chaque direction, d'où
            n_feasible_max = ceil(grid_x/min_sep) * ceil(grid_y/min_sep)
        Ce plafond borne n_max et l'étendue des balayages de Pareto : demander
        40 bouées sur une grille 8x12 contrainte est tout simplement infaisable.
        """
        gi = np.array([i // self.grid_y for i in range(self.K)])
        gj = np.array([i %  self.grid_y for i in range(self.K)])
        di = np.abs(gi[:, None] - gi[None, :])
        dj = np.abs(gj[:, None] - gj[None, :])
        dist = np.maximum(di, dj) if MIN_SEP_DIAGONAL else (di + dj)
        self._conflict = (dist < self.min_sep)
        np.fill_diagonal(self._conflict, False)
        self._grid_ij = np.stack([gi, gj], 1)

        s = max(1, self.min_sep)
        self.n_feasible_max = int(np.ceil(self.grid_x / s)
                                  * np.ceil(self.grid_y / s))
        if self.n_max > self.n_feasible_max:
            print(f"  [CONTRAINTE] n_max={self.n_max} > maximum faisable "
                  f"({self.n_feasible_max}) avec séparation {self.min_sep} "
                  f"case(s) — ramené à {self.n_feasible_max}")
            self.n_max = self.n_feasible_max
        self.n_min = int(min(self.n_min, self.n_max))

    # -------------------------------------------------------------------------
    def feasible_candidates(self, active_idx):
        """Indices activables sans violer la séparation minimale."""
        active_idx = np.asarray(active_idx, dtype=int)
        ok = np.ones(self.K, dtype=bool)
        if len(active_idx):
            ok &= ~self._conflict[active_idx].any(axis=0)
            ok[active_idx] = False
        return np.where(ok)[0]

    def is_feasible(self, active_idx):
        active_idx = np.asarray(active_idx, dtype=int)
        if len(active_idx) < 2:
            return True
        return not self._conflict[np.ix_(active_idx, active_idx)].any()

    def sample_feasible(self, n, rng=None):
        """
        Tire n candidats respectant la séparation minimale (insertion aléatoire
        gloutonne). Si n dépasse ce que le domaine peut accueillir, retourne le
        plus grand ensemble faisable trouvé.
        """
        # np.random n'expose pas .integers : on force un Generator
        if rng is None or not hasattr(rng, "integers"):
            rng = np.random.default_rng()
        n = int(min(n, self.n_feasible_max))
        if n <= 0:
            return np.array([], dtype=int)

        # Insertion gloutonne aleatoire : configurations bien diversifiees,
        # mais elle sature avant le plafond theorique (77/96 sur 16x24).
        best = np.array([], dtype=int)
        for _ in range(20):
            sel = []
            for c in rng.permutation(self.K):
                if len(sel) >= n:
                    break
                if not sel or not self._conflict[c, sel].any():
                    sel.append(int(c))
            if len(sel) >= n:
                return np.array(sel[:n], dtype=int)
            if len(sel) > len(best):
                best = np.array(sel, dtype=int)

        # Repli : reseau regulier a decalage aleatoire. Il atteint exactement
        # le plafond de faisabilite, on en tire ensuite un sous-ensemble.
        sp = max(1, self.min_sep)
        oi, oj = int(rng.integers(0, sp)), int(rng.integers(0, sp))
        lat = np.where(((self._grid_ij[:, 0] - oi) % sp == 0)
                       & ((self._grid_ij[:, 1] - oj) % sp == 0))[0]
        if len(lat) >= n:
            return np.array(sorted(rng.choice(lat, n, replace=False)), dtype=int)
        return lat if len(lat) > len(best) else best

    def invalid_action_mask(self, active=None):
        """
        Masque (K,) ou (B, K) des actions interdites : activer un candidat en
        conflit avec une bouée déjà posée. Désactiver reste toujours permis.

        Fonction déterministe du masque actif — donc recalculable depuis
        l'observation stockée dans le buffer PPO, ce qui permet de masquer les
        logits À LA FOIS au tirage et à la mise à jour sans stocker le masque.
        """
        a = self.active_mask if active is None else np.asarray(active)
        single = (a.ndim == 1)
        A = a.reshape(1, -1) if single else a
        conflicts = (A > 0.5) @ self._conflict          # (B, K)
        invalid = (conflicts > 0) & (A <= 0.5)
        return invalid[0] if single else invalid

    def _precompute_field_stats(self):
        """
        Deux objets précalculés :

        1. `field_stats` (K,) — variabilité mésoéchelle locale par candidat,
           utilisée comme features d'état de la politique.
           SST et SSS sont standardisées SÉPARÉMENT avant d'être mélangées.
           Sans cela, var(SST) ~ 3 °C² écrase var(SSS) ~ 0.03 psu² et la
           salinité contribue pour 0.0 % à la variance du mélange : l'agent
           optimisait un réseau purement thermique.

        2. `_kernel` (K, M) et `_weights` (M,) — noyau de couverture utilisé
           par la récompense d'information. Chaque candidat « explique » une
           tache gaussienne de rayon = échelle de décorrélation du nature run
           (INFLUENCE_RADIUS_KM ~ 90 km, diagnostiquée sur le champ), pondérée
           par la variabilité locale de la cellule.
        """
        # ── 1. features d'état ────────────────────────────────────────────────
        self.field_stats, vT, vS = local_variance_map(
            self.T, self.S, self.candidate_positions, half_win=2)
        self.var_T_cand, self.var_S_cand = vT, vS

        # ── 2. grille d'évaluation + poids de variabilité ─────────────────────
        st = self.eval_stride
        xs = np.arange(0, NX, st); ys = np.arange(0, NY, st)
        GX, GY = np.meshgrid(xs, ys, indexing="ij")
        self._eval_xy = np.stack([GX.ravel(), GY.ravel()], axis=1).astype(np.float32)

        Ta = mesoscale_anomaly(self.T); Sa = mesoscale_anomaly(self.S)
        wT = Ta.var(axis=0)[::st, ::st].ravel()
        wS = Sa.var(axis=0)[::st, ::st].ravel()
        w = 0.6 * wT / (wT.mean() + 1e-12) + 0.4 * wS / (wS.mean() + 1e-12)
        self._weights = (w / w.sum()).astype(np.float32)

        # ── 3. noyau candidat -> cellule ──────────────────────────────────────
        cand = np.array(self.candidate_positions, dtype=np.float32)
        d2 = ((cand[:, None, 0] - self._eval_xy[None, :, 0]) ** 2
              + (cand[:, None, 1] - self._eval_xy[None, :, 1]) ** 2)
        self._kernel = np.exp(-d2 / (2.0 * self.influence_px ** 2)).astype(np.float32)

        # ── 4. base d'information (mode "evf") ────────────────────────────────
        self._precompute_information_basis()

    def _precompute_information_basis(self, shrinkage=None, cv=None):
        """
        Critère de VARIANCE EXPLIQUÉE — la « qualité de l'information » d'un
        réseau n'est pas une couverture géométrique mais la fraction de la
        variabilité du système qu'on sait reconstruire à partir des seules
        observations. Estimation linéaire optimale (BLUE / interpolation
        optimale), critère standard en OSSE :

            EVF = Σ_c C_cO (C_OO + R)⁻¹ C_Oc  /  Σ_c C_cc

        Le vecteur d'observation contient SST **et** SSS en chaque bouée
        (2n observations), chacune normalisée par son écart-type et affectée
        de son propre bruit instrumental : la salinité pèse réellement.

        Pourquoi la covariance n'est pas empirique
        ------------------------------------------
        Le temps de décorrélation mésoéchelle du nature run est de ~12 jours.
        Une série d'un an ne contient donc que ~30 réalisations INDÉPENDANTES,
        pour 2n = 40 paramètres à estimer dès 20 bouées. La covariance
        empirique brute sur-apprend : mesurée hors échantillon, la variance
        expliquée devient NÉGATIVE (-0.23 à N=20). Le score « en échantillon »
        correspondant (0.62) n'était que du bruit ajusté.

        On contracte donc vers un modèle paramétrique construit sur les
        diagnostics du nature run lui-même :

            C[(i,v),(j,w)] = sigma_v(i)·sigma_w(j) · exp(-d_ij²/2L²) · c_vw

        avec sigma = carte d'écart-type mésoéchelle, L = échelle de
        décorrélation (INFLUENCE_RADIUS_KM) et c_vw = corrélation T-S locale.
        Contraction EVF_SHRINKAGE = 0.9 : mesurée hors échantillon, la
        variance expliquée redevient positive et croissante en N
        (+0.03 / +0.07 / +0.12 / +0.20 pour N = 5 / 10 / 20 / 40).

        Le mode `evf_cv` mesure directement cette valeur hors échantillon
        (statistiques sur la 1re moitié, score sur la 2nde) : c'est le chiffre
        honnête à communiquer, un peu plus bruité.
        """
        d = EVF_SHRINKAGE if shrinkage is None else float(shrinkage)
        cv = getattr(self, "evf_cv", False) if cv is None else bool(cv)

        Ta = mesoscale_anomaly(self.T) / (self.T.std() + 1e-9)
        Sa = mesoscale_anomaly(self.S) / (self.S.std() + 1e-9)
        st, nt = self.eval_stride, len(Ta)

        yT = Ta[:, ::st, ::st].reshape(nt, -1)
        yS = Sa[:, ::st, ::st].reshape(nt, -1)
        Y  = np.concatenate([yT, yS], axis=1)
        oT = np.stack([Ta[:, x, y] for (x, y) in self.candidate_positions], 1)
        oS = np.stack([Sa[:, x, y] for (x, y) in self.candidate_positions], 1)
        O  = np.concatenate([oT, oS], axis=1)
        Y = Y - Y.mean(0); O = O - O.mean(0)

        tr = slice(0, nt // 2) if cv else slice(0, nt)
        Otr, Ytr = O[tr], Y[tr]; n_tr = len(Ytr)

        # ── modèle paramétrique ───────────────────────────────────────────────
        cnd  = np.array(self.candidate_positions, dtype=np.float64)
        cell = self._eval_xy.astype(np.float64)
        L2   = 2.0 * self.influence_px ** 2

        def _rho(A, B):
            d2 = ((A[:, None, 0] - B[None, :, 0]) ** 2
                  + (A[:, None, 1] - B[None, :, 1]) ** 2)
            return np.exp(-d2 / L2)

        sT_o = oT[tr].std(0); sS_o = oS[tr].std(0)
        sT_c = yT[tr].std(0); sS_c = yS[tr].std(0)
        rTS  = float(np.clip(np.mean([
            np.corrcoef(oT[tr, k], oS[tr, k])[0, 1] for k in range(self.K)]), -1, 1))
        Roo, Roc = _rho(cnd, cnd), _rho(cnd, cell)

        def _b(Rm, sa, sb, cross):
            return (sa[:, None] * sb[None, :]) * Rm * (rTS if cross else 1.0)

        C_OO_p = np.block([[_b(Roo, sT_o, sT_o, 0), _b(Roo, sT_o, sS_o, 1)],
                           [_b(Roo, sS_o, sT_o, 1), _b(Roo, sS_o, sS_o, 0)]])
        C_OY_p = np.block([[_b(Roc, sT_o, sT_c, 0), _b(Roc, sT_o, sS_c, 1)],
                           [_b(Roc, sS_o, sT_c, 1), _b(Roc, sS_o, sS_c, 0)]])

        # ── contraction empirique -> paramétrique ─────────────────────────────
        C_OO_s = Otr.T @ Otr / n_tr
        C_OY_s = Otr.T @ Ytr / n_tr
        self._C_OO = ((1 - d) * C_OO_s + d * C_OO_p).astype(np.float64)
        self._C_OY = ((1 - d) * C_OY_s + d * C_OY_p).astype(np.float64)
        self._shrinkage = d
        self._rho_TS = rTS
        self._evf_cv = cv

        var_par = np.concatenate([sT_c, sS_c]) ** 2
        var_smp = (Ytr ** 2).mean(0)
        if cv:
            self._O_va = O[nt // 2:].astype(np.float64)
            self._Y_va = Y[nt // 2:].astype(np.float64)
            self._var_total = float((self._Y_va ** 2).mean(0).sum())
        else:
            self._var_total = float(((1 - d) * var_smp + d * var_par).sum())

        # Bruit instrumental, dans les unités normalisées de chaque variable
        rT = (OBS_NOISE_T / (self.T.std() + 1e-9)) ** 2
        rS = (OBS_NOISE_S / (self.S.std() + 1e-9)) ** 2
        self._R_diag = np.concatenate([np.full(self.K, rT), np.full(self.K, rS)])

    def explained_variance(self, active_idx):
        """Fraction de la variabilité mésoéchelle expliquée par le réseau."""
        active_idx = np.asarray(active_idx, dtype=int)
        if len(active_idx) == 0:
            return 0.0
        idx = np.concatenate([active_idx, active_idx + self.K])
        G = self._C_OO[np.ix_(idx, idx)] + np.diag(self._R_diag[idx])
        C = self._C_OY[idx]
        try:
            B = np.linalg.solve(G, C)              # gain d'interpolation optimale
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(G, C, rcond=None)[0]

        if not self._evf_cv:
            return float((C * B).sum() / (self._var_total + 1e-12))

        resid = self._Y_va - self._O_va[:, idx] @ B
        expl = (self._Y_va ** 2).mean(0).sum() - (resid ** 2).mean(0).sum()
        return float(expl / (self._var_total + 1e-12))

    def network_cost(self, active_idx):
        """
        Coût opérationnel annuel d'une configuration réseau.

            coût = N · COST_BUOY_FIXED
                 + longueur_tournée · COST_SHIP_PER_KM · N_CAMPAIGNS_YEAR

        La tournée de maintenance part du port, visite toutes les bouées par
        plus proche voisin et revient. Le coût n'est donc PAS proportionnel à
        N : un réseau compact près du port coûte moins cher qu'un réseau
        dispersé au large de même taille. C'est cette non-proportionnalité qui
        rend l'arbitrage information/coût réellement antagoniste.

        Retourne (cout_keur, co2_tonnes, longueur_km).
        """
        active_idx = np.asarray(active_idx, dtype=int)
        n = len(active_idx)
        if n == 0:
            return 0.0, 0.0, 0.0
        pts = np.array([self.candidate_positions[i] for i in active_idx],
                       dtype=np.float64) * DX_KM
        port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM

        # tournée plus proche voisin depuis le port, retour au port
        remaining = list(range(n)); cur = port; length = 0.0
        while remaining:
            d = np.linalg.norm(pts[remaining] - cur, axis=1)
            k = int(np.argmin(d)); length += float(d[k])
            cur = pts[remaining[k]]; remaining.pop(k)
        length += float(np.linalg.norm(cur - port))

        km_an = length * N_CAMPAIGNS_YEAR
        cost = n * COST_BUOY_FIXED + km_an * COST_SHIP_PER_KM
        co2  = km_an * CO2_SHIP_PER_KM
        return float(cost), float(co2), float(length)

    def reset(self):
        """Initialise un épisode : placement aléatoire de n_init bouées."""
        n_init = np.random.randint(self.n_min, self.n_max + 1)
        self.active_mask = np.zeros(self.K, dtype=np.float32)
        self.active_mask[self.sample_feasible(n_init)] = 1.0
        self.step_count = 0
        self.t_current  = np.random.randint(0, self.nt)
        return self._get_obs()

    def _get_obs(self):
        """Vecteur d'état : masque actif ∥ statistiques du champ."""
        return np.concatenate([self.active_mask, self.field_stats])

    def _compute_info_reward(self):
        """
        Score d'information du réseau courant, dans [0, 1].

        Mode "coverage" (défaut)
        ------------------------
            I(mask) = Σ_c  w_c · max_{i actif} exp(-d(c,i)² / 2L²)   /  Σ_c w_c

        Fraction de la variabilité du domaine effectivement couverte par au
        moins un capteur, à l'échelle de décorrélation L du nature run.
        Propriétés : croissant en N, saturant, sous-modulaire — donc un vrai
        coude, situé près de aire_domaine / (2πL²) ≈ 19 capteurs pour ce
        nature run. Deux bouées collées n'ajoutent presque rien (le max ne
        double pas), ce qui rend le bonus anti-clustering inutile.

        Mode "legacy"
        -------------
        Ancienne formule 0.7·moyenne(variance aux positions actives)
                       + 0.3·distance moyenne au plus proche voisin.
        Elle est conservée pour comparaison mais elle est NON MONOTONE en N
        (mesuré : 0.111 à N=3, 0.090 à N=5, -0.003 à N=12, 0.030 à N=40) :
        le « coude » N★ extrait du front de Pareto n'y était pas
        interprétable.
        """
        active_idx = np.where(self.active_mask > 0.5)[0]
        if len(active_idx) == 0:
            return 0.0

        if self.info_mode == "evf":
            return self.explained_variance(active_idx)

        if self.info_mode == "coverage":
            cov = self._kernel[active_idx].max(axis=0)      # (M,)
            return float((self._weights * cov).sum())

        # ── legacy ────────────────────────────────────────────────────────────
        coverage_score = float(self.field_stats[active_idx].mean())
        if len(active_idx) > 1:
            positions_active = np.array([self.candidate_positions[i]
                                         for i in active_idx], dtype=np.float32)
            tree = KDTree(positions_active)
            nn_dists, _ = tree.query(positions_active, k=2)
            spread_bonus = float(nn_dists[:, 1].mean() / np.sqrt(NX**2 + NY**2))
        else:
            spread_bonus = 0.0
        return 0.7 * coverage_score + 0.3 * spread_bonus

    def step(self, action):
        """
        Action : toggle de la position candidate `action` (0..K-1).
        
        Retourne : (obs, reward, done, info)
        """
        assert 0 <= action < self.K, f"Action invalide : {action}"
        prev_info = self._compute_info_reward()
        prev_n_active = int(self.active_mask.sum())

        # Toggle. Activer un candidat en conflit est refusé : la politique est
        # normalement empêchée par le masquage des logits, ce test protège les
        # appels hors PPO (rollouts manuels, chargement d'un vieux checkpoint).
        was_active = self.active_mask[action] > 0.5
        if not was_active:
            act = np.where(self.active_mask > 0.5)[0]
            if len(act) and self._conflict[action, act].any():
                self.step_count += 1
                return (self._get_obs(), -0.25,
                        self.step_count >= self.ep_len,
                        {"n_active": int(self.active_mask.sum()),
                         "delta_info": 0.0, "budget_penalty": 0.0,
                         "infaisable": True,
                         "total_info": prev_info})
        self.active_mask[action] = 0.0 if was_active else 1.0
        n_active = int(self.active_mask.sum())

        # Information après action
        new_info = self._compute_info_reward()
        delta_info = new_info - prev_info

        # Pénalité budget (hors de la plage autorisée)
        budget_penalty = 0.0
        if n_active < self.n_min:
            budget_penalty = float(self.n_min - n_active) / self.n_min
        elif n_active > self.n_max:
            budget_penalty = float(n_active - self.n_max) / self.n_max

        # info_gain remet delta_info (typiquement 1e-3..1e-2 en mode coverage)
        # a l'echelle de la penalite budget, sinon le budget domine tout.
        reward = (self.w_info   * self.info_gain * delta_info
                  - self.w_budget * budget_penalty)

        self.step_count += 1
        done = (self.step_count >= self.ep_len)

        info = {
            "n_active":       n_active,
            "delta_info":     delta_info,
            "budget_penalty": budget_penalty,
            "total_info":     new_info,
        }
        return self._get_obs(), float(reward), done, info


# ══════════════════════════════════════════════════════════════════════════════
#  POLITIQUE PPO — Actor-Critic
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Réseau actor-critic partagé pour PPO.

    Architecture :
        Tronc commun MLP (obs_dim → 256 → 256)
        ├── Actor  → logits (K actions) → distribution catégorielle
        └── Critic → valeur d'état V(s) (scalaire)

    L'entrée mélange deux types d'information :
        - masque binaire actif (sparse) : traité via embedding
        - statistiques continues du champ
    On les concatène et on passe dans le MLP commun.
    """
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        # Initialisation orthogonale (recommandée pour PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, obs, deterministic=False, invalid_mask=None):
        logits, value = self(obs)
        if invalid_mask is not None:
            logits = logits.masked_fill(invalid_mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Stocke les transitions (obs, action, reward, done, log_prob, value)
    pour un mini-batch PPO.
    """
    def __init__(self, buffer_size, obs_dim):
        self.obs       = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions   = np.zeros(buffer_size, dtype=np.int64)
        self.rewards   = np.zeros(buffer_size, dtype=np.float32)
        self.dones     = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values    = np.zeros(buffer_size, dtype=np.float32)
        self.ptr       = 0
        self.size      = buffer_size

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr = (self.ptr + 1) % self.size

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        """GAE-λ : advantage estimé par Generalized Advantage Estimation."""
        advantages = np.zeros(self.size, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_val = last_value
                next_done = 0.0
            else:
                next_val  = self.values[t + 1]
                next_done = self.dones[t + 1]
            delta = (self.rewards[t]
                     + gamma * next_val * (1 - next_done)
                     - self.values[t])
            gae = delta + gamma * lam * (1 - next_done) * gae
            advantages[t] = gae
        returns = advantages + self.values
        return advantages, returns

    def get_tensors(self, advantages, returns, device):
        return {
            "obs":       torch.tensor(self.obs,       device=device),
            "actions":   torch.tensor(self.actions,   device=device),
            "log_probs": torch.tensor(self.log_probs, device=device),
            "advantages":torch.tensor(advantages,     device=device),
            "returns":   torch.tensor(returns,        device=device),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT PPO
# ══════════════════════════════════════════════════════════════════════════════

def train_ppo(args, env):
    """
    Boucle d'entraînement PPO complète.

    Hyperparamètres clés :
        clip_eps     : clipping du ratio de politique (0.2 standard)
        entropy_coef : encourage l'exploration (0.01)
        vf_coef      : pondération de la value loss (0.5)
        n_epochs_ppo : passes sur chaque mini-batch (4)
    """
    print("═" * 60)
    print(" Brique 3 — PPO : Optimisation du Réseau d'Observation")
    print("═" * 60)

    obs_dim   = env.obs_dim
    n_actions = env.K
    policy    = ActorCritic(obs_dim, n_actions).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\n  Politique PPO — {n_params:,} paramètres")
    print(f"  Espace d'état  : {obs_dim} dim")
    print(f"  Espace d'action: {n_actions} positions candidates")

    buffer     = RolloutBuffer(args.buffer_size, obs_dim)
    clip_eps   = 0.2
    vf_coef    = 0.5
    ent_coef   = 0.01
    n_ppo_ep   = 4
    mini_batch = 64

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"episode_reward": [], "n_active": [], "info_score": []}
    ep_rewards = deque(maxlen=20)
    best_reward = -np.inf

    obs = env.reset()
    ep_reward = 0.0
    global_step = 0

    print(f"\n  Entraînement : {args.rl_steps} steps | buffer={args.buffer_size}")
    print("─" * 60)

    for step in range(args.rl_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        inv_t = torch.as_tensor(env.invalid_action_mask()[None], device=DEVICE)
        with torch.no_grad():
            action, log_prob, entropy, value = policy.get_action(
                obs_t, invalid_mask=inv_t)
        a = action.item()
        lp = log_prob.item()
        v  = value.item()

        next_obs, reward, done, info = env.step(a)
        buffer.add(obs, a, reward, done, lp, v)
        ep_reward += reward
        global_step += 1
        obs = next_obs

        if done:
            ep_rewards.append(ep_reward)
            history["episode_reward"].append(ep_reward)
            history["n_active"].append(info["n_active"])
            history["info_score"].append(info["total_info"])
            

            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save({"policy_state": policy.state_dict(),
                            "args": vars(args),
                            "active_mask": env.active_mask.copy()},
                           out_dir / "rl_best.pt")

            obs = env.reset()
            ep_reward = 0.0

        # ── Mise à jour PPO tous les buffer_size steps ───────────────────────
        if (step + 1) % args.buffer_size == 0:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                _, _, _, last_value = policy.get_action(
                    obs_t, invalid_mask=torch.as_tensor(
                        env.invalid_action_mask()[None], device=DEVICE))
                last_v = last_value.item()

            advantages, returns = buffer.compute_returns(last_v)
            # Normalisation des avantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            batch = buffer.get_tensors(advantages, returns, DEVICE)

            # PPO updates
            idx = np.arange(args.buffer_size)
            for _ in range(n_ppo_ep):
                np.random.shuffle(idx)
                for start in range(0, args.buffer_size, mini_batch):
                    end  = start + mini_batch
                    mb   = idx[start:end]
                    obs_mb = batch["obs"][mb]
                    act_mb = batch["actions"][mb]
                    lp_old = batch["log_probs"][mb]
                    adv_mb = batch["advantages"][mb]
                    ret_mb = batch["returns"][mb]

                    logits, values = policy(obs_mb)
                    # Le masque est une fonction deterministe du masque actif,
                    # qui occupe les K premieres composantes de l'observation :
                    # on le recalcule ici pour que log_prob soit coherent avec
                    # celui stocke au rollout (ratio PPO exact).
                    inv_mb = torch.as_tensor(
                        env.invalid_action_mask(
                            obs_mb[:, :env.K].detach().cpu().numpy()),
                        device=logits.device)
                    logits  = logits.masked_fill(inv_mb, -1e9)
                    dist    = torch.distributions.Categorical(logits=logits)
                    lp_new  = dist.log_prob(act_mb)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(lp_new - lp_old)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
                    actor_loss  = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values, ret_mb)
                    loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

            if len(ep_rewards) > 0 and (step + 1) % (args.buffer_size * 5) == 0:
                print(f"  Step {step+1:6d} | "
                      f"Mean reward (20 ep) = {np.mean(ep_rewards):+.3f} | "
                      f"Best = {best_reward:+.3f}")

    print(f"\n  ✓ Meilleure récompense : {best_reward:.4f}")
    print(f"  ✓ Checkpoint → {out_dir}/rl_best.pt")

    # ── Courbes d'apprentissage ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Brique 3 — PPO : Courbes d'entraînement", fontsize=14, fontweight="bold")

    axes[0, 0].plot(history["episode_reward"], alpha=0.4, color="steelblue")
    # Moyenne glissante
    w = max(1, len(history["episode_reward"]) // 20)
    smooth = np.convolve(history["episode_reward"],
                         np.ones(w)/w, mode="valid")
    axes[0, 0].plot(range(w-1, len(history["episode_reward"])), smooth, color="navy", lw=2)
    axes[0, 0].set_title("Récompense par épisode"); axes[0, 0].set_xlabel("Épisode")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["n_active"], color="orange", alpha=0.6)
    axes[0, 1].axhline(env.n_min, color="red", linestyle="--", label=f"n_min={env.n_min}")
    axes[0, 1].axhline(env.n_max, color="red", linestyle=":",  label=f"n_max={env.n_max}")
    axes[0, 1].set_title("Nombre de capteurs actifs en fin d'épisode")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["info_score"], color="green", alpha=0.6)
    axes[1, 0].set_title("Score d'information (couverture pondérée variance)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["info_score"], color="#74c476", alpha=0.8)
    axes[1, 1].set_title("Score d information (couverture pondérée variance)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "rl_training_curves.png", dpi=150)
    plt.close()
    print(f"  ✓ Courbes → {out_dir}/rl_training_curves.png")

    return policy


# ══════════════════════════════════════════════════════════════════════════════
#  FRONT DE PARETO — Optimisation sous contrainte budgétaire
# ══════════════════════════════════════════════════════════════════════════════

def _elbow_index(n_vals, info_vals):
    """
    Indice du coude : point le plus eloigne de la corde (premier -> dernier),
    apres normalisation des deux axes dans [0, 1].
    """
    n = np.asarray(n_vals, dtype=float); v = np.asarray(info_vals, dtype=float)
    if len(n) < 3:
        return 0
    # lissage leger : le sweep Monte-Carlo laisse du bruit residuel
    k = min(5, len(v) // 4 * 2 + 1)
    if k >= 3:
        ker = np.ones(k) / k
        v = np.convolve(np.pad(v, k // 2, mode="edge"), ker, mode="valid")[:len(n)]
    nx = (n - n.min()) / (np.ptp(n) + 1e-12)
    vy = (v - v.min()) / (np.ptp(v) + 1e-12)
    # distance verticale a la corde y = x (apres normalisation)
    return int(np.argmax(vy - nx))


def _config_info(env, idx):
    """Score d'information d'un jeu d'indices candidats."""
    mask = np.zeros(env.K, dtype=np.float32); mask[list(idx)] = 1.0
    env.active_mask = mask
    return float(env._compute_info_reward())


def _greedy_sequence(env, n_max):
    """
    Construction gloutonne : à chaque étape on ajoute le candidat qui maximise
    le gain d'information. Le critère étant sous-modulaire, cette suite est
    garantie à (1 − 1/e) de l'optimum (Nemhauser 1978) et sert de référence
    haute. Elle est emboîtée : un seul passage donne toutes les tailles.
    """
    sel, seqs = [], []
    for _ in range(min(n_max, env.n_feasible_max)):
        cands = env.feasible_candidates(sel)      # respecte la separation mini
        if len(cands) == 0:
            break
        best_c, best_v = None, -np.inf
        for c in cands:
            v = _config_info(env, sel + [int(c)])
            if v > best_v:
                best_v, best_c = v, int(c)
        sel = sel + [best_c]
        seqs.append((list(sel), best_v))
    return seqs


def _policy_sequence(env, policy, n_max):
    """
    Construction guidée par la politique PPO : on part d'un réseau vide et on
    active à chaque étape le candidat le mieux noté par l'acteur parmi les
    inactifs. Emboîtée elle aussi.

    C'est ici que la politique entraînée est réellement évaluée. L'ancien
    code recevait `policy` en argument et ne s'en servait jamais : le front
    mesurait du placement aléatoire, pas l'apport du RL.
    """
    if policy is None:
        return []
    seqs, sel = [], []
    mask = np.zeros(env.K, dtype=np.float32)
    policy.eval()
    for _ in range(min(n_max, env.n_feasible_max)):
        env.active_mask = mask.copy()
        obs = torch.from_numpy(env._get_obs().astype(np.float32))[None].to(DEVICE)
        with torch.no_grad():
            logits, _ = policy(obs)
        logits = logits[0].cpu().numpy()
        logits[mask > 0.5] = -np.inf                       # deja selectionne
        logits[env.invalid_action_mask(mask)] = -np.inf    # separation mini
        if not np.isfinite(logits).any():
            break
        c = int(np.argmax(logits))
        mask[c] = 1.0; sel = sel + [c]
        seqs.append((list(sel), _config_info(env, sel)))
    return seqs


def _non_dominated(n_vals, info_vals):
    """
    Masque des configurations non dominées au sens (info max, N min).
    Contrairement au balayage par N moyenné, ce test porte sur des
    configurations INDIVIDUELLES : un réseau de 22 bouées mal placées est
    dominé par un réseau de 17 bouées bien placées si son score est inférieur.
    C'est ce qui rend le front non trivial malgré la monotonie en N.
    """
    n = np.asarray(n_vals); v = np.asarray(info_vals)
    order = np.argsort(n, kind="stable")
    mask = np.zeros(len(n), dtype=bool)
    best = -np.inf
    for i in order:                      # N croissant : garder si strictement meilleur
        if v[i] > best + 1e-12:
            mask[i] = True; best = v[i]
    return mask


def compute_pareto_front(env, policy, args, n_random=25):
    """
    Front de Pareto  qualité de l'information  vs  nombre de bouées.

    Trois sources de configurations sont mises en concurrence à chaque N :
        - aléatoire   : ligne de base (ce que vaut un réseau non optimisé)
        - politique   : construction guidée par l'acteur PPO entraîné
        - glouton     : référence haute, garantie (1 − 1/e) de l'optimum

    Le front est l'ensemble des configurations non dominées parmi TOUTES
    celles évaluées, pas la moyenne par N. L'enveloppe supérieure donne le
    meilleur score atteignable pour chaque budget de bouées.

    Le « meilleur compromis » est identifié de deux façons complémentaires :
        - le coude de l'enveloppe (distance maximale à la corde)
        - la scalarisation  max_N [ info(N) − lambda·N ]  balayée en lambda,
          qui montre comment N★ se déplace avec le coût marginal d'une bouée.
    """
    print("\n── Front de Pareto : information vs nombre de capteurs ───────────")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if env.info_mode == "evf":
        print(f"  Critère : variance expliquée (interpolation optimale) | "
              f"L = {env.influence_px * DX_KM:.0f} km | "
              f"contraction = {env._shrinkage:.2f} | rho_TS = {env._rho_TS:+.2f}"
              + ("  [validé hors échantillon]" if env._evf_cv else ""))
    else:
        print(f"  Critère : {env.info_mode}")

    n_lo = max(1, env.n_min - 5)
    n_hi = min(env.n_feasible_max, env.n_max + 10)
    n_range = list(range(n_lo, n_hi + 1))

    cloud_n, cloud_v, cloud_src = [], [], []

    # ── 1. tirages aléatoires ─────────────────────────────────────────────────
    print(f"  Tirages aléatoires ({n_random} par N)...")
    rand_stats = {}
    for n_target in n_range:
        vals = []
        for _ in range(n_random):
            idx = env.sample_feasible(n_target)
            v = _config_info(env, idx)
            vals.append(v)
            cloud_n.append(n_target); cloud_v.append(v); cloud_src.append(0)
        rand_stats[n_target] = (float(np.mean(vals)), float(np.std(vals)))

    # ── 2. construction guidée par la politique PPO ───────────────────────────
    print("  Construction guidée par la politique PPO...")
    pol_seq = _policy_sequence(env, policy, n_hi)
    pol_by_n = {}
    for sel, v in pol_seq:
        if len(sel) in n_range:
            pol_by_n[len(sel)] = (list(sel), v)
            cloud_n.append(len(sel)); cloud_v.append(v); cloud_src.append(1)

    # ── 3. référence gloutonne ────────────────────────────────────────────────
    print("  Référence gloutonne (sous-modulaire, garantie 1-1/e)...")
    gre_seq = _greedy_sequence(env, n_hi)
    gre_by_n = {}
    for sel, v in gre_seq:
        if len(sel) in n_range:
            gre_by_n[len(sel)] = (list(sel), v)
            cloud_n.append(len(sel)); cloud_v.append(v); cloud_src.append(2)

    cloud_n = np.array(cloud_n); cloud_v = np.array(cloud_v)
    cloud_src = np.array(cloud_src)
    pareto_mask = _non_dominated(cloud_n, cloud_v)

    # ── 4. enveloppe supérieure par N ─────────────────────────────────────────
    pareto_points = []
    for n_target in n_range:
        m = cloud_n == n_target
        r_mean, r_std = rand_stats[n_target]
        best_src = int(cloud_src[m][np.argmax(cloud_v[m])])
        pareto_points.append({
            "n_buoys":        n_target,
            "info_mean":      float(cloud_v[m].max()),   # enveloppe = atteignable
            "info_std":       r_std,
            "info_random":    r_mean,
            "info_policy":    float(pol_by_n.get(n_target, ([], np.nan))[1]),
            "info_greedy":    float(gre_by_n.get(n_target, ([], np.nan))[1]),
            "best_source":    ["aleatoire", "politique", "glouton"][best_src],
            "n_dominated":    int((~pareto_mask[m]).sum()),
        })

    info_vals = np.array([p["info_mean"] for p in pareto_points])
    n_vals    = np.array([p["n_buoys"]   for p in pareto_points])
    rnd_vals  = np.array([p["info_random"] for p in pareto_points])
    elbow_idx = _elbow_index(n_vals, info_vals)
    n_star    = int(n_vals[elbow_idx])

    # ── 5. scalarisation : N★ en fonction du coût marginal d'une bouée ───────
    marg = np.gradient(info_vals, n_vals)
    lambdas = np.linspace(0.0, max(marg.max(), 1e-6) * 1.05, 160)
    n_of_lambda = np.array([n_vals[int(np.argmax(info_vals - lam * n_vals))]
                            for lam in lambdas])
    lam_star = float(np.interp(n_star, n_of_lambda[::-1], lambdas[::-1]))

    # ── 6. Figure ─────────────────────────────────────────────────────────────
    BG, PANEL, EDGE = "#0a1628", "#050d1a", "#2a4a7a"
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), facecolor=BG)

    def frame(ax, title, xlab, ylab):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(EDGE)
        ax.set_title(title, color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel(xlab, color="white", fontsize=9)
        ax.set_ylabel(ylab, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, alpha=0.2, color="white")

    # (a) nuage + front
    ax = axes[0]
    frame(ax, "Front de Pareto  —  information vs nombre de bouées",
          "Nombre de bouées actives",
          "Variance mésoéchelle expliquée" if env.info_mode == "evf"
          else "Score d'information")
    styles = [("aléatoire", "#5a7ca8", 14, 0.35, "o"),
              ("politique PPO", "#ffd93d", 42, 0.95, "s"),
              ("glouton (réf.)", "#6bcb77", 30, 0.85, "^")]
    for k, (lbl, col, sz, al, mk) in enumerate(styles):
        m = cloud_src == k
        if m.sum():
            ax.scatter(cloud_n[m], cloud_v[m], s=sz, c=col, alpha=al,
                       marker=mk, edgecolors="none", label=lbl, zorder=3)
    ax.scatter(cloud_n[pareto_mask], cloud_v[pareto_mask], s=95,
               facecolors="none", edgecolors="#ff6b6b", linewidths=1.4,
               zorder=6, label=f"non dominées ({pareto_mask.sum()})")
    ax.plot(n_vals, info_vals, color="#ff6b6b", lw=1.4, alpha=0.8, zorder=4)
    ax.axvline(n_star, color="#ff6b6b", ls="--", lw=1.5, alpha=0.8)
    ax.annotate(f"N★ = {n_star}\n{info_vals[elbow_idx]:.1%} de variance",
                (n_star, info_vals[elbow_idx]), textcoords="offset points",
                xytext=(12, -34), fontsize=9, color="#ff6b6b", fontweight="bold")
    n_eq = int(n_vals[np.argmin(np.abs(rnd_vals - info_vals[elbow_idx]))])
    if n_eq > n_star:
        ax.annotate(f"{n_star} bouées optimisées ≡ {n_eq} au hasard",
                    xy=(0.03, 0.94), xycoords="axes fraction",
                    color="#8ab4d4", fontsize=8.5)
    ax.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor=EDGE,
              loc="lower right")

    # (b) gain marginal
    ax = axes[1]
    frame(ax, "Gain marginal par bouée ajoutée\n(sur l'enveloppe supérieure)",
          "Nombre de bouées actives", "Δ information / Δ N")
    cols = ["#6bcb77" if n <= n_star else "#5a7ca8" for n in n_vals]
    ax.bar(n_vals, marg, color=cols, alpha=0.9, edgecolor=EDGE, lw=0.4)
    thr = marg.max() * 0.20
    ax.axhline(thr, color="#ffd93d", ls="--", lw=1.4,
               label=f"20 % du gain initial")
    ax.axvline(n_star, color="#ff6b6b", ls="--", lw=1.5, alpha=0.8,
               label=f"N★ = {n_star}")
    ax.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor=EDGE)

    # (c) compromis : N* en fonction du cout marginal
    ax = axes[2]
    frame(ax, "Compromis  —  N★ en fonction du coût d'une bouée\n"
              "max$_N$ [ info(N) − λ·N ]",
          "λ  (coût marginal d'une bouée, en unité d'information)",
          "N★ optimal")
    ax.plot(lambdas, n_of_lambda, color="#6baed6", lw=2.2)
    ax.axhline(n_star, color="#ff6b6b", ls="--", lw=1.3, alpha=0.8)
    ax.axvline(lam_star, color="#ff6b6b", ls="--", lw=1.3, alpha=0.8)
    ax.annotate(f"λ ≈ {lam_star:.4f}\n→ N★ = {n_star}", (lam_star, n_star),
                textcoords="offset points", xytext=(14, 14),
                fontsize=9, color="#ff6b6b", fontweight="bold")
    ax.annotate("bouée coûteuse →\nréseau léger",
                xy=(0.62, 0.80), xycoords="axes fraction",
                color="#8ab4d4", fontsize=8)

    fig.suptitle("Brique 3 — Front de Pareto information / nombre de bouées",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "rl_pareto_front.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Figure → {out_dir}/rl_pareto_front.png")

    # ── 7. Recommandations ────────────────────────────────────────────────────
    print("\n── Recommandations ────────────────────────────────────────────────")
    print(f"  N★ (coude)         : {n_star} bouées  |  "
          f"info = {info_vals[elbow_idx]:.3f}  "
          f"(source : {pareto_points[elbow_idx]['best_source']})")
    if n_eq > n_star:
        print(f"  Équivalence        : {n_star} bouées optimisées valent "
              f"{n_eq} bouées placées au hasard")
    print(f"  Coût-seuil         : lambda* = {lam_star:.4f} — au-dessus, "
          f"le réseau optimal se réduit")
    print(f"  Configurations non dominées : {int(pareto_mask.sum())} "
          f"sur {len(cloud_n)} évaluées")
    print(f"\n  {'N':>4} | {'aléatoire':>10} | {'politique':>10} | "
          f"{'glouton':>10} | {'enveloppe':>10}")
    for pt in pareto_points:
        if pt["n_buoys"] % max(1, len(pareto_points)//12) and pt["n_buoys"] != n_star:
            continue
        star = " ★" if pt["n_buoys"] == n_star else "  "
        print(f"  {pt['n_buoys']:>4} | {pt['info_random']:>10.3f} | "
              f"{pt['info_policy']:>10.3f} | {pt['info_greedy']:>10.3f} | "
              f"{pt['info_mean']:>10.3f}{star}")

    return pareto_points, pareto_mask, n_star


def _pareto_2d(cost, info):
    """Non-domination pour (coût minimal, information maximale)."""
    c = np.asarray(cost); v = np.asarray(info)
    order = np.argsort(c, kind="stable")
    mask = np.zeros(len(c), dtype=bool); best = -np.inf
    for i in order:
        if v[i] > best + 1e-12:
            mask[i] = True; best = v[i]
    return mask


def _greedy_weighted(env, n_max, lam_cost):
    """
    Glouton sur le critère scalarisé  info − lam_cost · coût.
    Balayer lam_cost engendre l'ensemble du front information/coût :
    lam = 0 donne le réseau le plus informatif, lam élevé le plus sobre.
    """
    sel, out = [], []
    for _ in range(min(n_max, env.n_feasible_max)):
        cands = env.feasible_candidates(sel)
        if len(cands) == 0:
            break
        best_c, best_u = None, -np.inf
        for c in cands:
            cand = sel + [int(c)]
            u = _config_info(env, cand) - lam_cost * env.network_cost(cand)[0]
            if u > best_u:
                best_u, best_c = u, int(c)
        sel = sel + [best_c]
        out.append(list(sel))
    return out


def compute_multiobjective_front(env, policy, args, n_random=20, n_lambda=6):
    """
    Front de Pareto bi-objectif : information vs COÛT opérationnel.

    Le nombre de bouées est un proxy grossier du coût. Ici le coût intègre la
    tournée de maintenance depuis le port : à N fixé il varie d'un facteur 1.3
    à 1.6 selon la dispersion du réseau. Les deux objectifs sont alors
    véritablement antagonistes et la non-domination discrimine réellement.

    Configurations mises en concurrence :
        - aléatoires (ligne de base)
        - politique PPO
        - gloutons scalarisés  max [info − lambda·coût]  pour plusieurs lambda
          -> balaye l'ensemble du front

    Sorties : figure 3 panneaux + recommandation sous contrainte de budget.
    """
    print("\n── Front bi-objectif : information vs coût opérationnel ──────────")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    n_hi = min(env.n_feasible_max, env.n_max + 10)
    C_n, C_c, C_v, C_co2, C_src = [], [], [], [], []

    def _add(idx, src):
        cost, co2, _ = env.network_cost(idx)
        C_n.append(len(idx)); C_c.append(cost); C_co2.append(co2)
        C_v.append(_config_info(env, idx)); C_src.append(src)

    print(f"  Tirages aléatoires ({n_random} par N)...")
    for n_t in range(max(1, min(env.n_min, n_hi) - 5), n_hi + 1):
        for _ in range(n_random):
            _add(env.sample_feasible(n_t), 0)

    print("  Politique PPO...")
    for sel, _ in _policy_sequence(env, policy, n_hi):
        _add(np.array(sel), 1)

    # échelle de lambda : gain d'info typique rapporté au coût d'une bouée
    lam_ref = 1.0 / max(COST_BUOY_FIXED * 20.0, 1e-6)
    lambdas = np.concatenate([[0.0], np.geomspace(0.2 * lam_ref, 8 * lam_ref,
                                                  n_lambda - 1)])
    print(f"  Gloutons scalarisés ({n_lambda} valeurs de lambda)...")
    for lam in lambdas:
        for sel in _greedy_weighted(env, n_hi, lam):
            _add(np.array(sel), 2)

    C_n = np.array(C_n); C_c = np.array(C_c); C_v = np.array(C_v)
    C_co2 = np.array(C_co2); C_src = np.array(C_src)
    front = _pareto_2d(C_c, C_v)

    # ── Figure ────────────────────────────────────────────────────────────────
    BG, PANEL, EDGE = "#0a1628", "#050d1a", "#2a4a7a"
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), facecolor=BG)

    def frame(ax, t, xl, yl):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(EDGE)
        ax.set_title(t, color="white", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel(xl, color="white", fontsize=9)
        ax.set_ylabel(yl, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, alpha=0.2, color="white")

    ax = axes[0]
    frame(ax, "Front de Pareto  —  information vs coût",
          "Coût opérationnel (k€/an)", "Variance mésoéchelle expliquée")
    sc = ax.scatter(C_c, C_v, c=C_n, s=16, cmap="viridis", alpha=0.55,
                    edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("N bouées", color="white",
                                                          fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)
    o = np.argsort(C_c[front])
    ax.plot(C_c[front][o], C_v[front][o], color="#ff6b6b", lw=2.0, zorder=5)
    ax.scatter(C_c[front], C_v[front], s=70, facecolors="none",
               edgecolors="#ff6b6b", linewidths=1.3, zorder=6,
               label=f"front ({front.sum()} configs)")
    ax.legend(fontsize=8, labelcolor="white", facecolor=BG, edgecolor=EDGE,
              loc="lower right")

    ax = axes[1]
    frame(ax, "Dispersion du coût à N fixé\n(deux réseaux de même taille ne "
              "coûtent pas pareil)", "Nombre de bouées", "Coût (k€/an)")
    ax.scatter(C_n, C_c, c=C_v, s=16, cmap="magma", alpha=0.6, edgecolors="none")
    ax.scatter(C_n[front], C_c[front], s=60, facecolors="none",
               edgecolors="#ff6b6b", linewidths=1.2, zorder=5)

    ax = axes[2]
    frame(ax, "Meilleure information atteignable\nsous contrainte de budget",
          "Budget annuel (k€)", "Variance expliquée atteignable")
    budgets = np.linspace(C_c.min(), C_c.max(), 90)
    best_v = np.array([C_v[C_c <= b].max() if (C_c <= b).any() else np.nan
                       for b in budgets])
    best_n = np.array([C_n[C_c <= b][np.argmax(C_v[C_c <= b])]
                       if (C_c <= b).any() else 0 for b in budgets])
    ax.plot(budgets, best_v, color="#6bcb77", lw=2.4)
    ax2 = ax.twinx(); ax2.set_facecolor(PANEL)
    ax2.plot(budgets, best_n, color="#ffd93d", lw=1.4, ls="--", alpha=0.85)
    ax2.set_ylabel("N bouées du réseau optimal", color="#ffd93d", fontsize=9)
    ax2.tick_params(colors="#ffd93d", labelsize=8)
    for b in (600, 900, 1200):
        if C_c.min() <= b <= C_c.max():
            ax.axvline(b, color="#5a7ca8", ls=":", lw=1.0, alpha=0.8)

    fig.suptitle("Brique 3 — Front bi-objectif information / coût opérationnel",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "rl_pareto_cost.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Figure → {out_dir}/rl_pareto_cost.png")

    print("\n── Réseaux optimaux sous contrainte de budget ─────────────────────")
    print(f"  {'budget':>9} | {'N':>3} | {'info':>6} | {'coût réel':>10} | {'tCO2/an':>8}")
    recos = []
    for b in (500, 700, 900, 1100, 1400):
        m = C_c <= b
        if not m.any():
            continue
        k = int(np.where(m)[0][np.argmax(C_v[m])])
        recos.append({"budget_keur": b, "n_buoys": int(C_n[k]),
                      "info": float(C_v[k]), "cost_keur": float(C_c[k]),
                      "co2_t": float(C_co2[k])})
        print(f"  {b:>7} k€ | {C_n[k]:>3d} | {C_v[k]:>6.3f} | "
              f"{C_c[k]:>8.0f} k€ | {C_co2[k]:>8.1f}")

    return {"cost": C_c, "info": C_v, "n": C_n, "co2": C_co2,
            "source": C_src, "front_mask": front, "recommandations": recos}


def mark_retained_config_on_pareto(n_retained, info_retained, out_dir):
    """
    Ajoute une étoile ★ sur le graphe rl_pareto_front.png pour montrer
    la configuration effectivement retenue (depuis le best checkpoint).
    Produit rl_pareto_front_pipeline.png pour ne pas écraser l'original.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    out_dir = Path(out_dir)
    src = out_dir / "rl_pareto_front.png"
    if not src.exists():
        return

    img = mpimg.imread(str(src))
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.imshow(img)
    ax.axis("off")

    # Annoter en overlay — position textuelle en bas de l'image
    fig.text(0.5, 0.01,
             f"★ Config retenue (best checkpoint) : N={n_retained}  |  "
             f"info={info_retained:.3f}",
             ha="center", color="#ffd93d", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a1628",
                       edgecolor="#ffd93d", alpha=0.9))

    out = out_dir / "rl_pareto_front_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a1628")
    plt.close()
    print(f"  Pareto annoté → {out}")


def visualize_two_configs(env, pareto_points, n_star, policy, args,
                          best_mask=None):
    """
    Compare deux configurations réseau :

    Config Dense  : best_mask du checkpoint (si fourni) ou simulation depuis N★
                    → c'est la configuration RETENUE transmise à GNN et AE
    Config Légère : N ≈ N★ // 2, simulée par la politique

    best_mask : np.ndarray (K,) float32 — active_mask du meilleur épisode RL.
                Quand fourni (mode pipeline), le panneau Dense montre exactement
                la configuration qui sera évaluée par GNN et AE.
    """
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib.colors import LinearSegmentedColormap
    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    BG = "#0a1628"

    n_light = max(env.n_min, int(n_star) // 2)

    def _run_config_policy(n_target):
        """Simule une configuration à n_target bouées avec la politique."""
        env.active_mask[:] = 0.0
        idx = env.sample_feasible(min(n_target, env.n_feasible_max))
        env.active_mask[idx] = 1.0
        obs = env._get_obs()
        policy.eval()
        with torch.no_grad():
            for _ in range(env.ep_len):
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=DEVICE).unsqueeze(0)
                action, _, _, _ = policy.get_action(obs_t, deterministic=True)
                obs, _, done, _ = env.step(action.item())
                if done:
                    break
        active_idx = np.where(env.active_mask > 0.5)[0]
        return active_idx, float(env._compute_info_reward())

    # Config dense : best_mask si fourni, sinon simulation depuis N★
    if best_mask is not None:
        env.active_mask = best_mask.copy()
        dense_idx  = np.where(best_mask > 0.5)[0]
        dense_info = float(env._compute_info_reward())
        dense_label = "Dense  (config retenue)"
        dense_note  = "★ configuration transmise au GNN & AE"
    else:
        dense_idx, dense_info = _run_config_policy(int(n_star))
        dense_label = "Dense  (N★ simulée)"
        dense_note  = f"N★={n_star} (coude Pareto)"

    light_idx, light_info = _run_config_policy(n_light)
    light_label = f"Légère  (N★ ÷ 2 ≈ {n_light})"

    T_bg    = env.T[0]
    vTmin, vTmax = float(env.T.min()), float(env.T.max())
    all_pos = np.array(env.candidate_positions)

    fig = plt.figure(figsize=(18, 8), facecolor=BG)
    title = ("Brique 3 RL — Config retenue (best checkpoint) vs Légère"
             if best_mask is not None
             else "Brique 3 RL — Dense (N★) vs Légère (N★÷2)")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=0.99)

    for col, (active_idx, info_score, label, note, col_c) in enumerate([
        (dense_idx,  dense_info,  dense_label, dense_note,  "#6bcb77"),
        (light_idx,  light_info,  light_label, f"N={len(light_idx)} capteurs", "#ffd93d"),
    ]):
        env.active_mask[:] = 0.0
        env.active_mask[active_idx] = 1.0
        inactive_idx = np.where(env.active_mask <= 0.5)[0]

        ax = fig.add_axes([0.05 + col*0.47, 0.10, 0.40, 0.80])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")

        ax.imshow(T_bg.T, cmap=ocean_cmap, origin="lower", aspect="auto",
                  vmin=vTmin, vmax=vTmax, alpha=0.5, extent=[0, NX, 0, NY])
        ax.set_xlim(0, NX); ax.set_ylim(0, NY)
        ax.scatter(all_pos[inactive_idx, 0], all_pos[inactive_idx, 1],
                   c="#1a3a5c", s=14, alpha=0.35, zorder=2)
        sc = ax.scatter(all_pos[active_idx, 0], all_pos[active_idx, 1],
                        c=env.field_stats[active_idx], cmap="plasma",
                        s=90, vmin=0, vmax=1,
                        edgecolors="white", linewidths=0.8, zorder=6)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label("Variance locale", color="white", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
        ax.set_title(f"{label}\nN={len(active_idx)} bouées  |  Info={info_score:.3f}",
                     color=col_c, fontsize=11, fontweight="bold", pad=6)
        ax.text(0.02, 0.03, note, transform=ax.transAxes,
                color=col_c, fontsize=8, alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])

    loss_info = (dense_info - light_info) / (dense_info + 1e-9) * 100
    fig.text(0.5, 0.02,
             f"{dense_label}: N={len(dense_idx)} | info={dense_info:.3f}     "
             f"{light_label}: N={len(light_idx)} | info={light_info:.3f}     "
             f"Perte info: {loss_info:.1f}%  pour {len(dense_idx)-len(light_idx)} capteurs en moins",
             ha="center", color="#8ab4d4", fontsize=9)

    out = out_dir / "rl_two_configs.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"\n  ── Deux configurations ──────────────────────────────────────")
    print(f"  Dense  : N={len(dense_idx)} bouées  info={dense_info:.3f}  [{dense_note}]")
    print(f"  Légère : N={len(light_idx)} bouées  info={light_info:.3f}")
    print(f"  Figure → {out}")




# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION CONFIGURATION FINALE
# ══════════════════════════════════════════════════════════════════════════════

def visualize_final_config(env, active_mask, args, title="Configuration optimale RL"):
    """Visualise la configuration de réseau trouvée par l'agent RL."""
    out_dir = Path(args.output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Brique 3 — {title}", fontsize=13, fontweight="bold")

    active_idx = np.where(active_mask > 0.5)[0]
    inactive_idx = np.where(active_mask <= 0.5)[0]
    all_pos = np.array(env.candidate_positions)

    ax = axes[0]
    ax.scatter(all_pos[inactive_idx, 0], all_pos[inactive_idx, 1],
               c="lightgray", s=30, alpha=0.4, label="Positions inactives")
    sc = ax.scatter(all_pos[active_idx, 0], all_pos[active_idx, 1],
                    c=env.field_stats[active_idx], cmap="YlOrRd",
                    s=120, edgecolors="black", linewidths=0.8, zorder=5,
                    label=f"Bouées actives ({len(active_idx)})")
    plt.colorbar(sc, ax=ax, label="Variance locale (importance OED)")
    ax.set_xlim(0, NX); ax.set_ylim(0, NY)
    ax.set_title(f"Réseau optimal — {len(active_idx)}/{env.K} positions actives")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")

    # Heatmap de la variance du champ
    ax2 = axes[1]
    variance_grid = env.field_stats.reshape(env.grid_x, env.grid_y)
    im = ax2.imshow(variance_grid.T, cmap="YlOrRd", origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax2, label="Variance locale normalisée")
    # Overlay des bouées
    for ai in active_idx:
        gx_idx = ai // env.grid_y
        gy_idx = ai % env.grid_y
        ax2.scatter(gx_idx, gy_idx, c="blue", s=60, marker="*", zorder=5)
    ax2.set_title("Grille de variance + positionnement RL\n(étoile bleue = bouée active)")
    ax2.set_xlabel(f"x (cellule grille {env.grid_x})")
    ax2.set_ylabel(f"y (cellule grille {env.grid_y})")

    fig.tight_layout()
    fig.savefig(out_dir / "rl_optimal_network.png", dpi=150)
    plt.close()
    print(f"  ✓ Configuration finale → {out_dir}/rl_optimal_network.png")


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

# =============================================================================
#  GIF — Progression de l'agent RL
# =============================================================================

def save_rl_gif(env, policy, args, n_frames=80):
    """
    Genere un GIF animant la progression de l agent RL.

    Chaque frame = 1 step de l agent dans l environnement.
    On rejoue depuis un etat initial aleatoire et on laisse tourner
    l agent avec la politique entrainees (mode deterministe).

    Panneau gauche  : champ de variance locale (OED target) + bouees actives
    Panneau central : graphe du reseau (positions + aretes kNN)
    Panneau droit   : courbe de recompense cumulee en temps reel
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap
    from collections import deque
    from scipy.spatial import KDTree as _KDTree

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    var_cmap = LinearSegmentedColormap.from_list("vc",
        ["#050d1a","#1a3a5c","#2e75b6","#ffd93d","#ff6b6b"], N=256)

    BG = "#0a1628"
    cands = np.array(env.candidate_positions)

    # Rejouer avec la politique
    obs = env.reset()
    all_states   = [env.active_mask.copy()]
    all_rewards  = [0.0]
    cum_rewards  = [0.0]
    all_infos    = [{"n_active": int(env.active_mask.sum()),
                     "total_info": env._compute_info_reward()}]

    policy.eval()
    with torch.no_grad():
        for _ in range(n_frames - 1):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action, _, _, _ = policy.get_action(obs_t, deterministic=False)
            obs, reward, done, info = env.step(action.item())
            all_states.append(env.active_mask.copy())
            all_rewards.append(reward)
            cum_rewards.append(cum_rewards[-1] + reward)
            all_infos.append(info)
            if done:
                obs = env.reset()

    # Variance de fond sur la grille candidate
    var_grid = env.field_stats.reshape(env.grid_x, env.grid_y)

    # Coordonnées pixel des positions candidates — doivent correspondre
    # à l'extent de l'imshow SST (0→NX, 0→NY) pour que les points
    # soient bien placés sur la carte.
    cands_px_x = cands[:, 0].astype(float)   # ∈ [0, NX]
    cands_px_y = cands[:, 1].astype(float)   # ∈ [0, NY]

    # Construction du GIF
    fig = plt.figure(figsize=(18, 7), facecolor=BG)
    ax1 = fig.add_axes([0.03, 0.10, 0.28, 0.78])    # carte SST + variance + bouees
    ax2 = fig.add_axes([0.36, 0.10, 0.28, 0.78])    # graphe reseau
    ax3 = fig.add_axes([0.70, 0.10, 0.27, 0.78])    # courbe recompense

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.tick_params(colors="#8ab4d4", labelsize=7)

    # Fond fixe ax1 : champ SST du nature run (coordonnées pixel NX×NY)
    T_bg = env.T[0]
    vTmin, vTmax = float(env.T.min()), float(env.T.max())
    ax1.imshow(T_bg.T, cmap=ocean_cmap, origin="lower", aspect="auto",
               vmin=vTmin, vmax=vTmax, extent=[0, NX, 0, NY])
    ax1.set_xlim(0, NX); ax1.set_ylim(0, NY)
    ax1.set_title("Variance locale (OED target)", color="white",
                  fontsize=9, fontweight="bold", pad=5)
    ax1.set_xticks([]); ax1.set_yticks([])

    # Fond fixe ax2 : même étendue pixel pour cohérence
    ax2.set_xlim(0, NX); ax2.set_ylim(0, NY)
    ax2.imshow(T_bg.T, cmap=ocean_cmap, origin="lower", aspect="auto",
               vmin=vTmin, vmax=vTmax, alpha=0.3, extent=[0, NX, 0, NY])
    ax2.set_title("Graphe du reseau (kNN)", color="white",
                  fontsize=9, fontweight="bold", pad=5)
    ax2.set_xticks([]); ax2.set_yticks([])

    # Courbe recompense
    ax3.set_xlim(0, n_frames); ax3.set_ylim(min(cum_rewards)*1.1, max(cum_rewards)*1.1)
    ax3.set_title("Recompense cumulee", color="white", fontsize=9, fontweight="bold", pad=5)
    ax3.set_xlabel("Etape", color="#8ab4d4", fontsize=8)
    ax3.grid(True, alpha=0.15, color="white")
    reward_line, = ax3.plot([], [], color="#6bcb77", lw=2)
    step_vline   = ax3.axvline(0, color="#ffd93d", lw=1, alpha=0.7)

    # Elements dynamiques — les offsets utilisent les coordonnées pixel (0→NX, 0→NY)
    sc_inactive = ax1.scatter([], [], c="#1a3a5c", s=20, alpha=0.3, zorder=2)
    sc_active1  = ax1.scatter([], [], s=90, zorder=5, edgecolors="white",
                               linewidths=0.7)
    sc_inactive2 = ax2.scatter([], [], c="#1a3a5c", s=15, alpha=0.3, zorder=2)
    sc_active2   = ax2.scatter([], [], s=70, zorder=5, edgecolors="white",
                                linewidths=0.7)

    edge_lines = []   # lignes d aretes (recreees a chaque frame)

    txt_step = fig.text(0.5, 0.96, "", ha="center", color="white",
                         fontsize=11, fontweight="bold")
    txt_n    = ax1.text(0.02, 0.02, "", transform=ax1.transAxes,
                         color="#ffd93d", fontsize=9, va="bottom", fontweight="bold")
    txt_r    = ax3.text(0.98, 0.05, "", transform=ax3.transAxes,
                         color="#6bcb77", fontsize=9, ha="right", fontweight="bold")

    reward_x, reward_y = [], []

    def update(frame):
        nonlocal edge_lines
        mask    = all_states[frame]
        info    = all_infos[frame]
        cum_r   = cum_rewards[frame]
        n_active = int(mask.sum())

        active_idx   = np.where(mask > 0.5)[0]
        inactive_idx = np.where(mask <= 0.5)[0]

        # ── Panneau 1 : carte SST + variance ───────────────────────────────
        if len(inactive_idx) > 0:
            sc_inactive.set_offsets(
                np.c_[cands_px_x[inactive_idx], cands_px_y[inactive_idx]])
        if len(active_idx) > 0:
            colors = plt.cm.plasma(env.field_stats[active_idx])
            sc_active1.set_offsets(
                np.c_[cands_px_x[active_idx], cands_px_y[active_idx]])
            sc_active1.set_color(colors)
        else:
            sc_active1.set_offsets(np.empty((0, 2)))

        # ── Panneau 2 : graphe reseau ─────────────────────────────────────────
        for ln in edge_lines: ln.remove()
        edge_lines = []

        if len(active_idx) > 1:
            pos_active = cands[active_idx].astype(float)
            tree_ = _KDTree(pos_active)
            for i in range(len(pos_active)):
                dists, idxs = tree_.query(pos_active[i], k=min(4, len(pos_active)))
                for j in idxs[1:]:
                    alpha_ = float(np.clip(1 - dists[list(idxs).index(j)] /
                                           (0.5*np.sqrt(NX**2+NY**2)), 0.05, 0.8))
                    ln_, = ax2.plot([pos_active[i,0], pos_active[j,0]],
                                    [pos_active[i,1], pos_active[j,1]],
                                    color="#2e75b6", alpha=alpha_, lw=1.2)
                    edge_lines.append(ln_)

        if len(inactive_idx) > 0:
            sc_inactive2.set_offsets(cands[inactive_idx])
        if len(active_idx) > 0:
            colors2 = plt.cm.YlOrRd(env.field_stats[active_idx])
            sc_active2.set_offsets(cands[active_idx])
            sc_active2.set_color(colors2)
        else:
            sc_active2.set_offsets(np.empty((0, 2)))

        # ── Panneau 3 : courbe ────────────────────────────────────────────────
        reward_x.append(frame); reward_y.append(cum_r)
        reward_line.set_data(reward_x, reward_y)
        step_vline.set_xdata([frame, frame])

        # ── Textes ────────────────────────────────────────────────────────────
        eps = max(0.05, 1.0 - frame / n_frames)
        txt_step.set_text(
            f"Brique 3 — RL  |  Etape {frame+1}/{n_frames}  "
            f"|  epsilon={eps:.2f}  |  N actives={n_active}")
        txt_step.set_color(plt.cm.cool(frame / n_frames))
        txt_n.set_text(f"Bouees: {n_active} [{env.n_min}-{env.n_max}]")
        txt_r.set_text(f"R_cum = {cum_r:+.3f}")

        return (sc_inactive, sc_active1, sc_inactive2, sc_active2,
                reward_line, step_vline, txt_step, txt_n, txt_r)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=True)
    gif_path = out_dir / "rl_progression.gif"
    writer = PillowWriter(fps=6, metadata={"title": "OED-IA RL SNO"})
    anim.save(str(gif_path), writer=writer, dpi=110,
              savefig_kwargs={"facecolor": BG})
    plt.close()
    print(f"  GIF RL -> {gif_path}")


# =============================================================================
#  ARGUMENTS + MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Brique 3 — RL pour OED")
    p.add_argument("--train",        action="store_true", help="Lancer PPO")
    p.add_argument("--pareto",       action="store_true", help="Front info / N")
    p.add_argument("--multiobj",     action="store_true",
                   help="Front bi-objectif information / cout operationnel")
    p.add_argument("--gif",          action="store_true", help="Genere le GIF")
    p.add_argument("--report",       action="store_true",
                   help="Produit un rapport .txt avec les métriques clés")
    p.add_argument("--seed_ocean",   type=int, default=42)
    p.add_argument("--seed_buoys",   type=int, default=7)
    p.add_argument("--checkpoint",   type=str, default="outputs/rl_best.pt")
    p.add_argument("--output_dir",   type=str, default="outputs")
    p.add_argument("--rl_steps",     type=int, default=50000)
    p.add_argument("--buffer_size",  type=int, default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--grid_x",       type=int, default=16)
    p.add_argument("--grid_y",       type=int, default=24)
    p.add_argument("--n_min",        type=int, default=10)
    p.add_argument("--n_max",        type=int, default=40)
    p.add_argument("--episode_len",  type=int, default=20)
    p.add_argument("--nt",           type=int, default=NT,
                   help="Longueur du nature run (jours)")
    p.add_argument("--info_mode",    type=str, default="evf",
                   choices=["evf", "coverage", "legacy"],
                   help="evf = variance expliquee (BLUE sur les EOF du nature "
                        "run) | coverage = couverture geometrique (plus rapide)")
    p.add_argument("--min_sep",      type=int, default=MIN_SEP_CELLS,
                   help="Separation mini entre bouees, en cases de la grille "
                        "candidate (2 = pas de cases adjacentes)")
    p.add_argument("--evf_shrink",   type=float, default=EVF_SHRINKAGE,
                   help="Contraction vers la covariance parametrique (0..1)")
    p.add_argument("--evf_cv",       type=int, default=0,
                   help="1 = score EVF valide hors echantillon (2 moities "
                        "temporelles) : moins optimiste, un peu plus bruite")
    p.add_argument("--n_random",     type=int, default=25,
                   help="Tirages aleatoires par N pour le nuage de Pareto")
    p.add_argument("--influence_km", type=float, default=INFLUENCE_RADIUS_KM,
                   help="Rayon d influence d un capteur (km)")
    p.add_argument("--w_info",       type=float, default=1.0)
    p.add_argument("--w_budget",     type=float, default=0.5)
    p.add_argument("--gif_frames",   type=int, default=80)
    return p.parse_args()


if __name__ == "__main__":
    from datetime import datetime
    args = parse_args()

    if not (args.train or args.pareto or args.gif or args.multiobj):
        print("Usage: python 03_rl.py --train [--pareto] [--multiobj] "
              "[--gif] [--report]")
        sys.exit(0)

    print(f"\n[1/2] Generation du nature run (seed_ocean={args.seed_ocean})...")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=args.nt, seed=args.seed_ocean)

    print("[2/2] Initialisation de l environnement MDP...")
    if args.nt < 365:
        print(f"  [ATTENTION] nt={args.nt} < 365 : cycle saisonnier "
              f"incomplet, statistiques biaisees.")

    env = OceanNetworkEnv(
        T, S,
        grid_x=args.grid_x, grid_y=args.grid_y,
        n_min=args.n_min, n_max=args.n_max,
        episode_len=args.episode_len,
        w_info=args.w_info, w_budget=args.w_budget,
        info_mode=args.info_mode, influence_km=args.influence_km,
        evf_cv=bool(args.evf_cv), min_sep=args.min_sep)
    print(f"  K = {env.K} positions candidates ({args.grid_x}x{args.grid_y})")
    print(f"  Budget bouees : [{env.n_min}, {env.n_max}]")
    print(f"  Separation    : >= {env.min_sep} case(s) "
          f"({'8-voisinage' if MIN_SEP_DIAGONAL else '4-voisinage'}) "
          f"-> maximum faisable = {env.n_feasible_max} bouees")
    print(f"  Score info    : {env.info_mode} | rayon d influence "
          f"{args.influence_km:.0f} km ({env.influence_px:.1f} px)")

    policy = None; pareto_data = {}
    if args.train:
        policy = train_ppo(args, env)
        ckpt = torch.load(Path(args.output_dir) / "rl_best.pt",
                          map_location=DEVICE, weights_only=False)
        visualize_final_config(env, ckpt["active_mask"], args)
        print("\n  Generation automatique du GIF post-entrainement...")
        save_rl_gif(env, policy, args, n_frames=args.gif_frames)

    if args.pareto:
        if policy is None:
            policy = ActorCritic(env.obs_dim, env.K).to(DEVICE)
            ckpt_path = Path(args.output_dir) / "rl_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                policy.load_state_dict(ckpt["policy_state"])
        pareto_points, pareto_mask, n_star = compute_pareto_front(
            env, policy, args, n_random=args.n_random)
        visualize_two_configs(env, pareto_points, n_star, policy, args)
        info_vals = np.array([p["info_mean"] for p in pareto_points])
        n_vals    = np.array([p["n_buoys"]   for p in pareto_points])
        n_light   = max(env.n_min, n_star // 2)
        pareto_data = {
            "n_star": int(n_star),
            "info_star": float(info_vals[np.argmin(np.abs(n_vals - n_star))]),
            "info_max":  float(info_vals.max()),
            "n_light":   int(n_light),
            "info_light": float(info_vals[np.argmin(np.abs(n_vals - n_light))]),
            "n_pareto_opt": int(pareto_mask.sum()),
        }

    multiobj_data = {}
    if args.multiobj:
        if policy is None:
            policy = ActorCritic(env.obs_dim, env.K).to(DEVICE)
            ck = Path(args.output_dir) / "rl_best.pt"
            if ck.exists():
                policy.load_state_dict(torch.load(ck, map_location=DEVICE,
                                                  weights_only=False)["policy_state"])
        multiobj_data = compute_multiobjective_front(
            env, policy, args, n_random=max(5, args.n_random // 2))

    if args.gif:
        print("\n  Generation du GIF de progression RL...")
        if policy is None:
            policy = ActorCritic(env.obs_dim, env.K).to(DEVICE)
            ckpt_path = Path(args.output_dir) / "rl_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                policy.load_state_dict(ckpt["policy_state"])
                print(f"  Politique chargee depuis {ckpt_path}")
        save_rl_gif(env, policy, args, n_frames=args.gif_frames)

    if args.report:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output_dir)
        perte_pct = 0.0
        if pareto_data:
            i_s = pareto_data.get("info_star", 0)
            i_l = pareto_data.get("info_light", 0)
            perte_pct = (i_s - i_l) / (i_s + 1e-9) * 100
        lines = [
            "=" * 68,
            "  Brique 3 — RL — Rapport",
            f"  Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 68, "",
            "── REPRODUCTIBILITÉ ─────────────────────────────────────────────────",
            f"  seed_ocean    : {args.seed_ocean}",
            f"  seed_buoys    : {args.seed_buoys}",
            "",
            "── PARAMÈTRES RL ────────────────────────────────────────────────────",
            f"  rl_steps      : {args.rl_steps}",
            f"  grid          : {args.grid_x}×{args.grid_y}  ({env.K} candidats)",
            f"  n_min / n_max : {args.n_min} / {args.n_max}",
            f"  w_info        : {args.w_info}",
        ]
        if pareto_data:
            lines += [
                "",
                "── RÉSULTATS PARETO ─────────────────────────────────────────────────",
                f"  N★ (coude)              : {pareto_data['n_star']} capteurs",
                f"  Score info N★           : {pareto_data['info_star']:.3f}",
                f"  Score info maximum      : {pareto_data['info_max']:.3f}",
                f"  Config légère N         : {pareto_data['n_light']} capteurs",
                f"  Score info légère       : {pareto_data['info_light']:.3f}",
                f"  Perte info dense→légère : {perte_pct:.1f} %",
                f"  Configs Pareto-optimales: {pareto_data['n_pareto_opt']}",
            ]
        if multiobj_data.get("recommandations"):
            lines += ["", "── RÉSEAUX OPTIMAUX SOUS BUDGET ─────────────────────────────────────",
                      f"  {'budget':>9} | {'N':>3} | {'info':>6} | {'coût':>8} | tCO2/an"]
            for r in multiobj_data["recommandations"]:
                lines.append(f"  {r['budget_keur']:>7} k€ | {r['n_buoys']:>3d} | "
                             f"{r['info']:>6.3f} | {r['cost_keur']:>6.0f} k€ | "
                             f"{r['co2_t']:>7.1f}")
            lines.append(f"  Configurations non dominées : "
                         f"{int(multiobj_data['front_mask'].sum())}")
        lines += ["", "── FICHIERS PRODUITS ────────────────────────────────────────────────"]
        for f in sorted(out.iterdir()):
            if f.suffix in {".pt", ".png", ".gif"}:
                lines.append(f"  {f.name:<44} {f.stat().st_size//1024:>5} KB")
        lines += ["", "=" * 68]
        rpt = out / f"rapport_rl_{ts}.txt"
        rpt.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  Rapport RL → {rpt}")

    print("\n  Brique 3 terminee.")
