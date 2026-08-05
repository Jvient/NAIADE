"""
===========================================================================
  BRIQUE 3 -- Reinforcement Learning : Optimisation du Reseau
  
  3 methodes de selection du N* optimal :
    pareto      : front de Pareto info vs N (sweep + Kneedle)
    efficiency  : eta(N) = info(N) / (1+log(N)), score unique
    scalarized  : PPO avec cout marginal integre, sweep sur lambda
  
  Usage :
    python 03_rl.py --train --evaluate --rl_method pareto
    python 03_rl.py --train --evaluate --rl_method efficiency
    python 03_rl.py --train --evaluate --rl_method scalarized
===========================================================================
"""

import sys, argparse
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from config import *
try:
    from dataset import SyntheticOceanGenerator
except ModuleNotFoundError:
    from data.dataset import SyntheticOceanGenerator

# ── Source de données : synthétique (défaut) ou GLORYS12 (--data glorys) ─────

OCEAN  = None
GLORYS = None


def _load_module(path, name):
    """Charge un module par chemin (les briques commencent par un chiffre)."""
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod          # picklabilité des classes du module
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
#  COUPLAGE — RewardModel : AE (vérité terrain) + émulateur SAGE (Brique 2)
# ══════════════════════════════════════════════════════════════════════════════
#  Trois usages :
#    skill(positions)          : 1 − RMSE_AE, potentiel réseau ABSOLU, coûteux
#                                mais déterministe (CRN : instants, bruits
#                                d'obs et seeds dropout fixés à l'init).
#    sage_score(pos, réseau)   : contribution marginale RELATIVE prédite du
#                                capteur dans son contexte (z-unités de la
#                                config) — l'émulateur entraîné en Brique 2.
#                                Quasi gratuit → shaping dense du RL.
#  NB : SAGE prédit des scores standardisés PAR CONFIG (moyenne nulle) ; leur
#  somme ne mesure PAS la qualité absolue du réseau — c'est le rôle de skill().


class RewardModel:
    def __init__(self, T, S, ae_checkpoint=None, sage_checkpoint=None,
                 n_t=4, n_mc=4, seed=123, corr_timestamps=200):
        self.T, self.S = T, S
        self.ae_model, self.norm = None, None
        self.sage, self.b2 = None, None
        rng = np.random.default_rng(seed)

        if ae_checkpoint and Path(ae_checkpoint).exists():
            b1 = _load_module("01_autoencoder.py", "brick1_ae")
            b1.NX, b1.NY, b1.OCEAN, b1.GLORYS = NX, NY, OCEAN, GLORYS
            ckpt = torch.load(ae_checkpoint, map_location=DEVICE,
                              weights_only=False)
            ck_args = ckpt["args"]
            self.ae_model = b1.ObservabilityAE(
                base_ch=ck_args.get("base_ch", 32),
                latent_ch=ck_args.get("latent_ch", 64),
                dropout_p=ck_args.get("dropout_p", 0.1)).to(DEVICE)
            self.ae_model.load_state_dict(ckpt["model_state"])
            self.ae_model.eval()
            self.norm = ckpt["norm"]
            self._b1 = b1
            # Jeu d'évaluation CRN figé (déterminisme du potentiel)
            self._t_idx = rng.choice(len(T), min(n_t, len(T)), replace=False)
            Tn = (T[self._t_idx] - self.norm["T_mean"]) / (self.norm["T_std"] + 1e-9)
            Sn = (S[self._t_idx] - self.norm["S_mean"]) / (self.norm["S_std"] + 1e-9)
            if OCEAN is not None:
                Tn *= OCEAN[None]; Sn *= OCEAN[None]
            self._Tn, self._Sn = Tn.astype(np.float32), Sn.astype(np.float32)
            ns_T = self.norm.get("obs_ns_T", OBS_NOISE_STD / (self.norm["T_std"] + 1e-9))
            ns_S = self.norm.get("obs_ns_S", OBS_NOISE_STD / (self.norm["S_std"] + 1e-9))
            self._nT = (rng.standard_normal(Tn.shape) * ns_T).astype(np.float32)
            self._nS = (rng.standard_normal(Sn.shape) * ns_S).astype(np.float32)
            self._n_mc = n_mc
            self._skill_cache = {}
            print(f"  [RewardModel] AE chargé ({ae_checkpoint}) | "
                  f"éval CRN : {len(self._t_idx)} instants, n_mc={n_mc}")

        # Profilage
        self.n_sage = 0; self.t_sage = 0.0
        self.n_skill = 0; self.t_skill = 0.0

        if sage_checkpoint and Path(sage_checkpoint).exists():
            b2 = _load_module("02_gnn.py", "brick2_gnn")
            b2.NX, b2.NY, b2.OCEAN, b2.GLORYS = NX, NY, OCEAN, GLORYS
            state = torch.load(sage_checkpoint, map_location=DEVICE,
                               weights_only=False)
            self.sage = None
            for in_dim in (10, 8):        # 10 = features v2 ; 8 = anciens ckpts
                try:
                    m = b2.GraphSAGEInductive(in_dim=in_dim).to(DEVICE)
                    m.load_state_dict(state)
                    self.sage = m
                    break
                except RuntimeError:
                    continue
            if self.sage is None:
                raise RuntimeError(f"Checkpoint SAGE incompatible : {sage_checkpoint}")
            self.sage.eval()
            self.b2 = b2
            self._sage_cache = {}
            # ── Cache par position : le goulot d'étranglement historique était
            # le recalcul par appel des corrélations (boucle sur les nœuds) et
            # surtout de np.gradient sur le CHAMP COMPLET (5 instants x N
            # nœuds x chaque appel). Tout est ramené à un précalcul unique :
            #   - séries z-scorées par position -> corrélation = produit
            #     scalaire (mathématiquement identique) ;
            #   - |grad T| précalculé une fois sur les 5 instants de référence
            #     (mêmes instants que _compute_node_features : 0,10,...,40) ;
            #   - var 5x5 par position calculée au premier accès puis cachée.
            self._pos_cache = {}
            rng_c = np.random.default_rng(seed + 7)
            self._corr_t_idx = rng_c.choice(
                len(T), min(corr_timestamps, len(T)), replace=False)
            g_t = list(range(0, min(len(T), 50), 10))
            gm = []
            for t_i in g_t:
                gx = np.gradient(T[t_i], axis=0)
                gy = np.gradient(T[t_i], axis=1)
                gm.append(np.sqrt(gx ** 2 + gy ** 2))
            self._gradmag = np.mean(gm, axis=0).astype(np.float32)  # (NX, NY)
            print(f"  [RewardModel] Émulateur SAGE chargé ({sage_checkpoint})")
            self._parity_check()

    # ── Potentiel absolu : skill AE (1 − RMSE non-observé) ──────────────────
    @torch.no_grad()
    def skill(self, positions):
        if self.ae_model is None:
            return None
        key = frozenset(positions)
        if key in self._skill_cache:
            return self._skill_cache[key]
        import time as _time
        t0 = _time.time()
        rmse = self._b1._eval_config_rmse_crn(
            self.ae_model, self._Tn, self._Sn, list(positions),
            self._nT, self._nS, n_mc=self._n_mc, mc_seed=777)
        s = 1.0 - rmse
        if len(self._skill_cache) > 4096:
            self._skill_cache.clear()
        self._skill_cache[key] = s
        self.n_skill += 1; self.t_skill += _time.time() - t0
        return s

    # ── Cache par position : série z-scorée + stats locales ─────────────────
    def _pos_data(self, pos):
        d = self._pos_cache.get(pos)
        if d is not None:
            return d
        px, py = pos
        tT = self.T[self._corr_t_idx, px, py]
        tS = self.S[self._corr_t_idx, px, py]
        ts_T = (tT - self.T[:, px, py].mean()) / (self.T[:, px, py].std() + 1e-9)
        ts_S = (tS - self.S[:, px, py].mean()) / (self.S[:, px, py].std() + 1e-9)
        serie = 0.6 * ts_T + 0.4 * ts_S
        serie = (serie - serie.mean()) / (serie.std() + 1e-9)   # corr = s·s'/n
        x0, x1 = max(0, px - 2), min(NX, px + 3)
        y0, y1 = max(0, py - 2), min(NY, py + 3)
        d = {"serie": serie.astype(np.float32),
             "var_T": float(self.T[:, x0:x1, y0:y1].var()),
             "var_S": float(self.S[:, x0:x1, y0:y1].var()),
             "grad":  float(self._gradmag[x0:x1, y0:y1].mean()),
             "border": min(px, NX - 1 - px, py, NY - 1 - py) / (max(NX, NY) / 2)}
        self._pos_cache[pos] = d
        return d

    def _fast_graph(self, positions):
        """Réplique build_graph (arêtes) + _compute_node_features (10 dim)
        de la Brique 2, à partir du cache par position. Parité vérifiée à
        l'init (_parity_check)."""
        from scipy.spatial import KDTree
        n = len(positions)
        data = [self._pos_data(p) for p in positions]
        M = np.stack([d["serie"] for d in data])                  # (n, nt)
        corr = (M @ M.T) / M.shape[1]
        np.clip(corr, -1.0, 1.0, out=corr)

        # Arêtes : seuil 0.5 + kNN 4 (identique à build_graph, défauts)
        pos_arr = np.array(positions, dtype=np.float32)
        edges = set()
        thr = np.argwhere(np.abs(corr) > 0.5)
        for i, j in thr:
            if i != j:
                edges.add((int(i), int(j)))
        tree = KDTree(pos_arr)
        _, knn = tree.query(pos_arr, k=min(5, n))
        for i in range(n):
            for j in np.atleast_1d(knn[i])[1:]:
                edges.add((int(i), int(j))); edges.add((int(j), int(i)))
        src_l = [e[0] for e in edges]; dst_l = [e[1] for e in edges]
        edge_index = torch.tensor([src_l, dst_l], dtype=torch.long)
        degree = np.bincount(src_l, minlength=n).astype(np.float32)

        # Features (mêmes définitions que _compute_node_features)
        cm = corr.copy(); np.fill_diagonal(cm, 0)
        corr_max = np.abs(cm).max(axis=1, keepdims=True)
        deg_n = (degree.reshape(-1, 1) / (degree.max() + 1e-9))
        vT = np.array([[d["var_T"]] for d in data], dtype=np.float32)
        vS = np.array([[d["var_S"]] for d in data], dtype=np.float32)
        gr = np.array([[d["grad"]]  for d in data], dtype=np.float32)
        vT = (vT - vT.mean()) / (vT.std() + 1e-9)
        vS = (vS - vS.mean()) / (vS.std() + 1e-9)
        gr = (gr - gr.mean()) / (gr.std() + 1e-9)
        bd = np.array([[d["border"]] for d in data], dtype=np.float32)
        d_nn = np.zeros((n, 1), dtype=np.float32)
        if n > 1:
            dd, _ = tree.query(pos_arr, k=2)
            d_nn[:, 0] = dd[:, 1] / (np.sqrt(NX ** 2 + NY ** 2) / 2)
        nf = np.full((n, 1), np.log1p(n) / np.log(100.0), dtype=np.float32)
        x = torch.tensor(np.hstack([
            pos_arr[:, 0:1] / NX, pos_arr[:, 1:2] / NY, corr_max, deg_n,
            vT, vS, gr, bd, d_nn, nf]).astype(np.float32))
        return x, edge_index

    def _parity_check(self, n_test=20):
        """Vérifie que le chemin rapide reproduit les features/arêtes de la
        Brique 2 (mêmes définitions, corrélation près — t_idx distincts)."""
        rng = np.random.default_rng(0)
        if OCEAN is not None:
            flat = np.where(OCEAN.ravel() > 0.5)[0]
            pick = rng.choice(flat, n_test, replace=False)
            positions = [(int(k // NY), int(k % NY)) for k in pick]
        else:
            positions = [(int(rng.integers(0, NX)), int(rng.integers(0, NY)))
                         for _ in range(n_test)]
        x_fast, ei_fast = self._fast_graph(positions)
        # Référence : mêmes corr (fast), features/arêtes de la Brique 2
        M = np.stack([self._pos_data(p)["serie"] for p in positions])
        corr = np.clip((M @ M.T) / M.shape[1], -1, 1)
        g_ref = self.b2.build_graph(positions, corr, T=self.T, S=self.S)
        e_fast = set(map(tuple, ei_fast.t().tolist()))
        e_ref  = set(map(tuple, g_ref["edge_index"].t().tolist()))
        feat_diff = float((x_fast - g_ref["x"]).abs().max())
        assert e_fast == e_ref, "Parité arêtes fast/Brique2 rompue"
        assert feat_diff < 1e-4, f"Parité features rompue (diff={feat_diff:.2e})"
        print(f"  [RewardModel] Parité chemin rapide vs Brique 2 : OK "
              f"(diff features max {feat_diff:.1e}, arêtes identiques)")

    # ── Contribution marginale relative prédite (émulateur Brique 2) ────────
    @torch.no_grad()
    def sage_score(self, pos, network_positions):
        if self.sage is None:
            return 0.0
        import time as _time
        t0 = _time.time()
        network_positions = list(network_positions)
        if pos not in network_positions:
            network_positions = network_positions + [pos]
        if len(network_positions) < 3:
            return 0.0     # graphe/corrélation dégénérés sous 3 nœuds
        key = (pos, frozenset(network_positions))
        if key in self._sage_cache:
            return self._sage_cache[key]
        x, edge_index = self._fast_graph(network_positions)
        out = self.sage(x.to(DEVICE), edge_index.to(DEVICE))
        scores = (out[0] if isinstance(out, tuple) else out).cpu().numpy()
        s = float(scores[network_positions.index(pos)])
        if len(self._sage_cache) > 65536:
            self._sage_cache.clear()
        self._sage_cache[key] = s
        self.n_sage += 1; self.t_sage += _time.time() - t0
        return s

    def report(self):
        if self.n_sage or self.n_skill:
            msg = "  [RewardModel] profil :"
            if self.n_sage:
                msg += (f" sage_score x{self.n_sage} "
                        f"({1e3 * self.t_sage / self.n_sage:.1f} ms/appel)")
            if self.n_skill:
                msg += (f" | skill x{self.n_skill} "
                        f"({1e3 * self.t_skill / self.n_skill:.1f} ms/appel)")
            print(msg)


def setup_data_source(args):
    global NX, NY, OCEAN, GLORYS
    if getattr(args, "data", "synthetic") == "glorys":
        from dataset_glorys import GlorysData
        GLORYS = GlorysData(getattr(args, "glorys_cache", "data/glorys_cache"))
        NX, NY = GLORYS.nlat, GLORYS.nlon
        OCEAN = GLORYS.ocean.astype(np.float32)
        print(f"  Source : GLORYS12 ({NX}x{NY}, océan {100*OCEAN.mean():.1f} %)")
        return GLORYS
    return None

try:
    from importlib.util import spec_from_file_location, module_from_spec
    _spec = spec_from_file_location("autoencoder", Path(__file__).parent / "01_autoencoder.py")
    _ae_mod = module_from_spec(_spec)
    _spec.loader.exec_module(_ae_mod)
    ObservabilityAE = _ae_mod.ObservabilityAE
    AE_AVAILABLE = True
except Exception:
    AE_AVAILABLE = False


# =========================================================================
#  GÉOMÉTRIE — distances physiques entre positions candidates
# =========================================================================

def pairwise_km(positions, geo=None, dx_km=DX_KM):
    """Matrice (n, n) des distances en km entre positions pixel (i, j).

    geo = GlorysData -> haversine vectorisée sur geo.lat / geo.lon.
                        Convention GLORYS : axe 0 = latitude, axe 1 = longitude
                        (cf. GlorysData.latlon_to_ij).
    geo = None       -> grille plane régulière de pas dx_km (synthétique).

    Vectorisé à dessein : GlorysData.distance_km() est correcte mais scalaire,
    et K~350 candidats font ~120k paires — inutilisable en boucle Python.
    """
    P = np.asarray(positions, dtype=np.float64)
    if len(P) == 0:
        return np.zeros((0, 0))
    if geo is None:
        d = P[:, None, :] - P[None, :, :]
        return np.sqrt((d ** 2).sum(-1)) * dx_km

    la = np.radians(np.asarray(geo.lat)[P[:, 0].astype(int)])
    lo = np.radians(np.asarray(geo.lon)[P[:, 1].astype(int)])
    dla, dlo = la[:, None] - la[None, :], lo[:, None] - lo[None, :]
    a = (np.sin(dla / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# =========================================================================
#  ENVIRONNEMENT MDP
# =========================================================================

class OceanNetworkEnv:
    """
    Grille candidate GX*GY. Action = toggle d'une position.
    Supporte deux modes de recompense :
      - standard  : delta_info - budget_penalty
      - scalarized: delta_info - lambda * cout marginal
    """
    def __init__(self, T, S, grid_x=16, grid_y=24,
                 n_min=10, n_max=40, episode_len=20,
                 w_info=1.0, w_budget=0.5, marginal_cost=0.0,
                 ocean_mask=None,
                 reward_mode="heuristic", reward_model=None,
                 sage_scale=0.3, w_terminal=5.0,
                 fixed_positions=None, init_mode="auto", mdp="toggle",
                 min_sep_km=MIN_BUOY_SEP_KM, geo=None):
        self.T = T.astype(np.float32)
        self.S = S.astype(np.float32)
        self.grid_x, self.grid_y = grid_x, grid_y
        self.n_min, self.n_max = n_min, n_max
        self.ep_len = episode_len
        self.w_info, self.w_budget = w_info, w_budget
        self.marginal_cost = marginal_cost
        self.nt = len(T)
        # Couplage : "heuristic" (historique) | "sage" | "ae" | "hybrid"
        self.reward_mode = reward_mode
        self.rm = reward_model
        self.sage_scale = sage_scale
        self.w_terminal = w_terminal
        # MDP : "toggle" (historique, édition sous pénalités budget) ou
        # "additive" (conception : épisode = EXACTEMENT k ajouts, pas de
        # retrait, budget structurel — supprime le réglage des pénalités ET
        # l'oscillation argmax du rollout déterministe en toggle).
        self.mdp = mdp
        self.eval_k = None      # force k à l'évaluation (comparaison N égal)
        if mdp == "additive":
            print(f"  MDP additif : épisode = k ajouts exacts, "
                  f"k ~ U[{n_min}, {n_max}]")
        if reward_mode in ("ae",) and (reward_model is None
                                       or reward_model.ae_model is None):
            raise ValueError("reward_mode='ae' requiert un RewardModel avec AE")
        if reward_mode in ("sage", "hybrid") and (reward_model is None
                                                  or reward_model.sage is None):
            raise ValueError(f"reward_mode='{reward_mode}' requiert l'émulateur SAGE")
        if reward_mode == "hybrid" and reward_model.ae_model is None:
            raise ValueError("reward_mode='hybrid' requiert aussi l'AE (bonus terminal)")

        sx, sy = NX / grid_x, NY / grid_y
        cand = [
            (min(int(gx*sx + sx/2), NX-1), min(int(gy*sy + sy/2), NY-1))
            for gx in range(grid_x) for gy in range(grid_y)
        ]
        # Mode GLORYS : seules les cellules dont le centre est en mer sont
        # candidates -> l'espace d'action K se réduit d'autant.
        if ocean_mask is not None:
            n0 = len(cand)
            cand = [(px, py) for (px, py) in cand if ocean_mask[px, py] > 0.5]
            if len(cand) < n0:
                print(f"  Grille candidate : {len(cand)}/{n0} cellules en mer")
        if len(cand) < max(2, n_min):
            raise ValueError("Grille candidate trop petite après masque océan")
        self.candidate_positions = cand
        self.K = len(cand)

        # ── Scénario contraint : réseau existant IMPOSÉ (ex. PIRATA) ─────────
        # Les positions fixes sont toujours actives, non togglables, hors
        # espace d'action. Le budget n_min/n_max porte sur les AJOUTS seuls
        # (le coût du réseau existant est un coût irrécupérable).
        self.fixed_positions = [tuple(map(int, p))
                                for p in (fixed_positions or [])]
        self.init_mode = ("empty" if self.fixed_positions else "random") \
            if init_mode == "auto" else init_mode
        if self.fixed_positions:
            print(f"  Réseau imposé : {len(self.fixed_positions)} stations "
                  f"fixes | budget d'AJOUTS [{n_min}, {n_max}] "
                  f"| init={self.init_mode}")
            # Dédup : une cellule candidate confondue avec une station fixe
            # serait un doublon achetable — on la retire de l'espace d'action.
            fset = set(self.fixed_positions)
            n0 = len(self.candidate_positions)
            self.candidate_positions = [p for p in self.candidate_positions
                                        if p not in fset]
            self.K = len(self.candidate_positions)
            if self.K < n0:
                print(f"  Candidats dédupliqués vs stations fixes : "
                      f"{self.K}/{n0}")

        # ── Séparation minimale entre bouées ─────────────────────────────────
        self.geo = geo
        self.min_sep_km = float(min_sep_km) if MIN_SEP_ENABLED else 0.0

        # Un candidat trop proche d'une station IMPOSÉE est définitivement
        # inachetable : on le sort de l'espace d'action plutôt que de le
        # masquer à chaque pas (K plus petit = exploration plus efficace).
        if self.fixed_positions and self.min_sep_km > 0:
            D = pairwise_km(self.candidate_positions + self.fixed_positions,
                            self.geo)
            nf = len(self.fixed_positions)
            blocked = (D[:-nf, -nf:] < self.min_sep_km).any(axis=1)
            n0 = len(self.candidate_positions)
            self.candidate_positions = [
                p for p, b in zip(self.candidate_positions, blocked) if not b]
            self.K = len(self.candidate_positions)
            if self.K < n0:
                print(f"  Séparation {self.min_sep_km:.0f} km : {n0 - self.K} "
                      f"candidats retirés (trop près d'une station imposée) "
                      f"-> K={self.K}")
            if self.K < max(2, self.n_min):
                raise ValueError(
                    f"Espace d'action vide après contrainte de séparation "
                    f"({self.min_sep_km:.0f} km) : réduire --min_sep_km ou "
                    f"augmenter --grid_x / --grid_y")

        self._build_conflict_matrix()
        self._precompute_field_stats()
        self.active_mask = None
        self.step_count = 0
        self.obs_dim = self.K + len(self.field_stats) \
            + (1 if self.mdp == "additive" else 0)

    # ---------------------------------------------------------------------
    #  Contrainte de séparation minimale
    # ---------------------------------------------------------------------
    def _build_conflict_matrix(self):
        """_conflict[i, j] = True si les candidats i et j sont trop proches
        pour être actifs simultanément (distance physique < min_sep_km)."""
        if self.min_sep_km <= 0:
            self._conflict = np.zeros((self.K, self.K), dtype=bool)
            self.n_feasible_max = self.K
            return

        D = pairwise_km(self.candidate_positions, self.geo)
        self._conflict = D < self.min_sep_km
        np.fill_diagonal(self._conflict, False)

        # Plafond de faisabilité : pas de formule fermée ici (grille irrégulière
        # après masque océan), on l'estime par packing glouton.
        self.n_feasible_max = self._estimate_packing()
        n_conf = int(self._conflict.sum() // 2)
        print(f"  Séparation minimale : {self.min_sep_km:.0f} km | "
              f"{n_conf} paires en conflit | packing max ~ "
              f"{self.n_feasible_max} bouées")
        if n_conf == 0:
            print(f"  [ATTENTION] aucun conflit : min_sep_km est inférieur au "
                  f"pas de la grille candidate, la contrainte est inopérante")
        if self.n_max > self.n_feasible_max:
            print(f"  [CONTRAINTE] n_max={self.n_max} > max faisable "
                  f"({self.n_feasible_max}) -> clippé")
            self.n_max = self.n_feasible_max
        self.n_min = int(min(self.n_min, self.n_max))

    def _estimate_packing(self, n_trials=12, seed=0):
        """Borne inférieure du plus grand ensemble indépendant (problème
        NP-difficile) : glouton par degré croissant + tirages aléatoires,
        on garde le meilleur."""
        rng = np.random.default_rng(seed)
        orders = [np.argsort(self._conflict.sum(1))]
        orders += [rng.permutation(self.K) for _ in range(n_trials)]
        best = 0
        for order in orders:
            sel = []
            for c in order:
                if not sel or not self._conflict[c, sel].any():
                    sel.append(int(c))
            best = max(best, len(sel))
        return best

    def feasible_candidates(self, active_idx):
        """Candidats activables sans violer la séparation."""
        a = np.asarray(active_idx, dtype=int)
        ok = np.ones(self.K, dtype=bool)
        if len(a):
            ok &= ~self._conflict[a].any(axis=0)
            ok[a] = False
        return np.where(ok)[0]

    def is_feasible(self, active_idx):
        a = np.asarray(active_idx, dtype=int)
        if len(a) < 2:
            return True
        return not self._conflict[np.ix_(a, a)].any()

    def sample_feasible(self, n, rng=None):
        """Tirage de n candidats respectant la séparation (insertion gloutonne
        randomisée). Si n dépasse le plafond, renvoie le plus grand ensemble
        faisable trouvé."""
        if rng is None or not hasattr(rng, "integers"):
            rng = np.random.default_rng()
        n = int(min(n, self.n_feasible_max))
        if n <= 0:
            return np.array([], dtype=int)
        best = np.array([], dtype=int)
        for _ in range(30):
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
        # repli : glouton par degré croissant, atteint le packing max estimé
        sel = []
        for c in np.argsort(self._conflict.sum(1)):
            if not sel or not self._conflict[c, sel].any():
                sel.append(int(c))
        sel = np.array(sel, dtype=int)
        if len(sel) >= n:
            return np.array(sorted(rng.choice(sel, n, replace=False)),
                            dtype=int)
        return sel if len(sel) > len(best) else best

    def invalid_action_mask(self, active=None):
        """Masque (K,) ou (B, K) des actions interdites : activer un candidat
        en conflit avec une bouée déjà posée. Désactiver reste TOUJOURS permis.

        Fonction déterministe du masque actif -> recalculable depuis l'obs
        stockée dans le buffer PPO, donc identique au rollout et à l'update
        (ratio PPO exact, pas de biais)."""
        a = self.active_mask if active is None else np.asarray(active)
        single = (a.ndim == 1)
        A = a.reshape(1, -1) if single else a
        act = (A > 0.5)
        conflicts = act.astype(np.float32) @ self._conflict.astype(np.float32)
        invalid = (conflicts > 0) & ~act
        return invalid[0] if single else invalid

    def _precompute_field_stats(self):
        stats = []
        for (px, py) in self.candidate_positions:
            x0, x1 = max(0, px-2), min(NX, px+3)
            y0, y1 = max(0, py-2), min(NY, py+3)
            stats.append(0.6*float(self.T[:, x0:x1, y0:y1].var())
                         + 0.4*float(self.S[:, x0:x1, y0:y1].var()))
        stats = np.array(stats, dtype=np.float32)
        s_min, s_max = stats.min(), stats.max()
        self.field_stats = (stats - s_min) / (s_max - s_min + 1e-9)
        # Stats des stations fixes (même recette, même normalisation)
        fs = []
        for (px, py) in self.fixed_positions:
            x0, x1 = max(0, px-2), min(NX, px+3)
            y0, y1 = max(0, py-2), min(NY, py+3)
            v = (0.6*float(self.T[:, x0:x1, y0:y1].var())
                 + 0.4*float(self.S[:, x0:x1, y0:y1].var()))
            fs.append((v - s_min) / (s_max - s_min + 1e-9))
        self.fixed_stats = np.array(fs, dtype=np.float32)

    def reset(self):
        self.active_mask = np.zeros(self.K, dtype=np.float32)
        if self.init_mode == "random":
            n_init = np.random.randint(self.n_min, self.n_max + 1)
            # sample_feasible remplace le tirage uniforme : un réseau initial
            # infaisable rendrait la quasi-totalité des actions illégales dès
            # le premier pas de l'épisode.
            self.active_mask[self.sample_feasible(n_init)] = 1.0
        # init_mode "empty" : la politique construit ses ajouts de zéro
        if self.mdp == "additive":
            self.active_mask[:] = 0.0          # additif : toujours de zéro
            self.k_target = int(self.eval_k) if self.eval_k is not None \
                else int(np.random.randint(self.n_min, self.n_max + 1))
        self.step_count = 0
        # Ledger anti-farming du shaping SAGE : chaque ajout enregistre le
        # shaping accordé ; le retrait rembourse EXACTEMENT ce montant.
        # Sans cela, un cycle ajout(+s)/retrait(-s') avec s' < s (le score
        # contextuel a baissé entre-temps) rapporte du shaping net sans
        # modifier le réseau final — canal de reward hacking observé
        # empiriquement (rewards croissantes, réseaux finaux dégénérés).
        self._sage_ledger = {}
        if self.reward_mode == "ae":
            self._phi = self.rm.skill(self._active_positions())
        return self._get_obs()

    def _active_positions(self):
        """Réseau complet : stations fixes + ajouts togglés."""
        return self.fixed_positions + [
            self.candidate_positions[i]
            for i in np.where(self.active_mask > 0.5)[0]]

    def _get_obs(self):
        base = np.concatenate([self.active_mask, self.field_stats])
        if self.mdp == "additive":
            remain = (self.k_target - self.active_mask.sum()) / max(self.n_max, 1)
            return np.concatenate([base, [np.float32(remain)]]).astype(np.float32)
        return base

    def _compute_info_reward(self):
        active_idx = np.where(self.active_mask > 0.5)[0]
        n_fixed = len(self.fixed_positions)
        if len(active_idx) == 0 and n_fixed == 0:
            return 0.0

        # Couverture avec saturation (Michaelis-Menten) :
        # alpha calibré sur n_max → demi-saturation quand ~n_max capteurs actifs.
        # Les stations fixes contribuent à la couverture (elles observent).
        sum_var = float(self.field_stats[active_idx].sum())
        if n_fixed:
            sum_var += float(self.fixed_stats.sum())
        alpha = float(self.n_max + n_fixed) * float(self.field_stats.mean())
        coverage = sum_var / (sum_var + alpha + 1e-9)

        # Bonus espacement : pénalise le clustering (fixes incluses —
        # ajouter près d'une station existante est pénalisé)
        pos_all = ([self.candidate_positions[i] for i in active_idx]
                   + self.fixed_positions)
        if len(pos_all) > 1:
            pos = np.array(pos_all, dtype=np.float32)
            nn_d, _ = KDTree(pos).query(pos, k=2)
            spread = float(nn_d[:, 1].mean() / np.sqrt(NX**2 + NY**2))
        else:
            spread = 0.0

        return 0.7 * coverage + 0.3 * spread

    def _step_additive(self, action):
        """MDP additif : chaque pas AJOUTE un candidat ; l'épisode se termine
        quand k_target ajouts sont posés. Pas de retrait, pas de pénalité
        budget (le budget est structurel). Reward : marginal selon le mode
        (heuristique / sage / Δ skill AE) + terminal AE en hybrid."""
        if self.active_mask[action] > 0.5:      # interdit par le masque ;
            self.step_count += 1                 # garde-fou : no-op pénalisé
            return self._get_obs(), -0.05, self.step_count >= 4 * self.n_max, {
                "n_active": int(self.active_mask.sum()),
                "total_info": float("nan"), "delta_info": 0.0}

        pos_a = self.candidate_positions[action]
        if self.reward_mode == "ae":
            self.active_mask[action] = 1.0
            phi_new = self.rm.skill(self._active_positions())
            delta_info, new_info = phi_new - self._phi, phi_new
            self._phi = phi_new
        elif self.reward_mode in ("sage", "hybrid"):
            self.active_mask[action] = 1.0
            s = self.rm.sage_score(pos_a, self._active_positions())
            delta_info = s * self.sage_scale
            new_info = float("nan")
        else:
            prev_info = self._compute_info_reward()
            self.active_mask[action] = 1.0
            new_info = self._compute_info_reward()
            delta_info = new_info - prev_info

        reward = self.w_info * delta_info
        if self.marginal_cost > 0:               # sweep scalarisé Pareto
            reward -= self.marginal_cost

        self.step_count += 1
        n_active = int(self.active_mask.sum())
        done = (n_active >= self.k_target) or (self.step_count >= 4 * self.n_max)

        if done and self.reward_mode == "hybrid":
            skill_f = self.rm.skill(self._active_positions())
            reward += self.w_terminal * skill_f
            new_info = skill_f
        elif done and self.reward_mode == "sage" and self.rm.ae_model is not None:
            new_info = self.rm.skill(self._active_positions())

        return self._get_obs(), float(reward), done, {
            "n_active": n_active, "total_info": new_info,
            "delta_info": delta_info}

    def step(self, action):
        assert 0 <= action < self.K
        # Garde-fou : la politique PPO est déjà empêchée par le masquage des
        # logits. Ce test protège les appels HORS PPO (rollouts manuels,
        # baselines, chargement d'un checkpoint entraîné sans contrainte).
        if self.min_sep_km > 0 and self.active_mask[action] <= 0.5:
            act = np.where(self.active_mask > 0.5)[0]
            if len(act) and self._conflict[action, act].any():
                self.step_count += 1
                done = (self.step_count >= (4 * self.n_max
                                            if self.mdp == "additive"
                                            else self.ep_len))
                return self._get_obs(), -0.25, done, {
                    "n_active": int(self.active_mask.sum()),
                    "total_info": float("nan"), "delta_info": 0.0,
                    "infaisable": True}
        if self.mdp == "additive":
            return self._step_additive(action)
        was_active = self.active_mask[action] > 0.5
        if self.reward_mode == "heuristic":
            prev_info = self._compute_info_reward()
        prev_positions = (self._active_positions()
                          if self.reward_mode in ("sage", "hybrid") else None)
        self.active_mask[action] = 0.0 if was_active else 1.0
        n_active = int(self.active_mask.sum())

        # ── Terme d'information selon le mode de couplage ────────────────────
        if self.reward_mode == "ae":
            # Potentiel absolu : Δ skill AE (une seule évaluation par pas,
            # le potentiel courant est en cache)
            phi_new = self.rm.skill(self._active_positions())
            delta_info, new_info = phi_new - self._phi, phi_new
            self._phi = phi_new
        elif self.reward_mode in ("sage", "hybrid"):
            # Contribution marginale RELATIVE prédite du capteur togglé.
            # Ajout : score dans le réseau après ajout, montant LEDGERISÉ.
            # Retrait : remboursement exact du montant accordé à l'ajout
            # (anti-farming : tout cycle ajout/retrait est net-zéro).
            pos_a = self.candidate_positions[action]
            if was_active:
                granted = self._sage_ledger.pop(action, None)
                if granted is None:      # capteur issu de l'init aléatoire
                    s = self.rm.sage_score(pos_a, prev_positions)
                    granted = s * self.sage_scale
                delta_info = -granted
            else:
                s = self.rm.sage_score(pos_a, self._active_positions())
                delta_info = s * self.sage_scale
                self._sage_ledger[action] = delta_info
            new_info = float("nan")     # pas de potentiel absolu en mode sage
        else:
            new_info = self._compute_info_reward()
            delta_info = new_info - prev_info

        if self.marginal_cost > 0:
            cost = self.marginal_cost if not was_active else -self.marginal_cost * 0.5
            reward = self.w_info * delta_info - cost
        else:
            penalty = 0.0
            if n_active < self.n_min:
                penalty = float(self.n_min - n_active) / self.n_min
            elif n_active > self.n_max:
                penalty = float(n_active - self.n_max) / self.n_max
            reward = self.w_info * delta_info - self.w_budget * penalty

        self.step_count += 1
        done = self.step_count >= self.ep_len

        # Mode hybrid : ancrage terminal par la vérité terrain AE
        if done and self.reward_mode == "hybrid":
            skill_f = self.rm.skill(self._active_positions())
            reward += self.w_terminal * skill_f
            new_info = skill_f
        elif done and self.reward_mode == "sage" and self.rm.ae_model is not None:
            new_info = self.rm.skill(self._active_positions())   # logging seul

        return self._get_obs(), float(reward), done, {
            "n_active": n_active, "total_info": new_info, "delta_info": delta_info}


# =========================================================================
#  BASELINES DE PLACEMENT + COMPARAISON (scorée par l'AE, jeu d'éval séparé)
# =========================================================================
#  Sans ces comparaisons, le RL/Pareto ne démontre rien : le greedy est
#  quasi-optimal pour les critères de couverture sous-modulaires (Krause et
#  al. 2008). Toutes les méthodes sont évaluées par le MÊME juge — le skill
#  AE — sur un jeu CRN distinct de celui de la reward (pas d'auto-notation).


def _baseline_random(env, n, rng, fixed=None):
    """n ajouts aléatoires (les stations fixes ne comptent pas dans n).
    Respecte la séparation minimale : sinon la comparaison opposerait un RL
    contraint à une baseline libre, ce qui biaise le verdict."""
    idx = env.sample_feasible(n, rng)
    return [env.candidate_positions[i] for i in idx]


def _baseline_maximin(env, n, rng, fixed=None):
    """Farthest-point sampling. Si un réseau fixe est imposé, la distance
    est initialisée aux stations existantes : les ajouts comblent d'abord
    les lacunes du réseau imposé."""
    pos = np.array(env.candidate_positions, dtype=np.float32)
    chosen = []
    if fixed:
        d = np.min([np.linalg.norm(pos - np.array(f, dtype=np.float32),
                                   axis=1) for f in fixed], axis=0)
    else:
        chosen = [int(rng.integers(0, env.K))]
        d = np.linalg.norm(pos - pos[chosen[0]], axis=1)
    while len(chosen) < n:
        # Restriction aux candidats faisables (séparation minimale).
        ok = np.zeros(env.K, dtype=bool)
        ok[env.feasible_candidates(chosen)] = True
        if not ok.any():
            break
        dd = np.where(ok, d, -np.inf)
        k = int(dd.argmax())
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(pos - pos[k], axis=1))
    return [env.candidate_positions[i] for i in chosen[:n]]


def _baseline_greedy_variance(env, n, fixed=None):
    """Top-n des cellules par variance locale, avec espacement glouton
    (par rapport aux choisies ET aux stations fixes imposées)."""
    order = np.argsort(-env.field_stats)
    pos = np.array(env.candidate_positions, dtype=np.float32)
    fx = [np.array(f, dtype=np.float32) for f in (fixed or [])]
    d_min = 0.5 * np.sqrt(NX * NY / max(n + len(fx), 1))
    chosen = []
    for dm in (d_min, d_min / 2, 0.0):
        for k in order:
            if len(chosen) >= n:
                break
            if int(k) in chosen:
                continue
            # Séparation minimale : contrainte DURE, elle ne se relâche pas
            # quand on descend dans le barème d_min -> d_min/2 -> 0.
            if chosen and env._conflict[int(k), chosen].any():
                continue
            far_chosen = all(np.linalg.norm(pos[k] - pos[c]) > dm
                             for c in chosen)
            far_fixed = all(np.linalg.norm(pos[k] - f) > dm for f in fx)
            if far_chosen and far_fixed:
                chosen.append(int(k))
        if len(chosen) >= n:
            break
    return [env.candidate_positions[i] for i in chosen[:n]]


def _baseline_greedy_sage(env, rm, n, cand_stride=1, verbose=True, fixed=None):
    """Placement séquentiel glouton guidé par l'émulateur SAGE : à chaque
    étape, ajoute le candidat de contribution marginale prédite maximale
    DANS LE CONTEXTE du réseau courant (fixes + ajouts précédents).
    C'est LA baseline naturelle du RL couplé (même signal, sans politique)."""
    fixed = list(fixed or [])
    chosen = []
    # Amorce uniquement si le contexte total est < 2 (score SAGE indéfini)
    if len(fixed) < 2:
        chosen = _baseline_greedy_variance(env, min(2 - len(fixed), n),
                                           fixed=fixed)
    pos2idx = {p: i for i, p in enumerate(env.candidate_positions)}
    for i in range(len(chosen), n):
        # Le vivier de candidats est recalculé à chaque ajout : la séparation
        # minimale dépend du réseau courant.
        chosen_idx = [pos2idx[p] for p in chosen if p in pos2idx]
        cand = env.feasible_candidates(chosen_idx)[::cand_stride]
        if len(cand) == 0:
            print(f"    greedy-SAGE : plus de candidat faisable à "
                  f"{len(chosen)} ajouts (séparation "
                  f"{env.min_sep_km:.0f} km)")
            break
        best_k, best_s = None, -np.inf
        for k in cand:
            p = env.candidate_positions[k]
            if p in chosen or p in fixed:
                continue
            s = rm.sage_score(p, fixed + chosen)
            if s > best_s:
                best_k, best_s = k, s
        if best_k is None:
            break
        chosen.append(env.candidate_positions[best_k])
        if verbose and ((i + 1) % 10 == 0 or i == n - 1):
            print(f"    greedy-SAGE : {i + 1}/{n} ajouts "
                  f"(dernier score {best_s:+.3f})")
    return chosen


def _rollout_policy_network(env, policy, deterministic=True, k=None):
    """Déroule la politique et retourne le réseau final.
    k : en MDP additif, force exactement k ajouts (comparaison à N égal)."""
    if k is not None and env.mdp == "additive":
        env.eval_k = int(k)
    obs = env.reset()
    env.eval_k = None
    done = False
    while not done:
        with torch.no_grad():
            a, _, _, _ = policy.get_action(
                torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE),
                deterministic=deterministic,
                forbid_active=(env.mdp == "additive"))
        obs, _, done, info = env.step(int(a.item()))
    return env._active_positions()


def _make_judge(args, env):
    """Juge indépendant de la reward : autres instants, autres bruits d'obs,
    autres seeds MC — évite toute auto-notation."""
    return RewardModel(env.T, env.S,
                       ae_checkpoint=args.ae_checkpoint,
                       sage_checkpoint=None,
                       n_t=args.reward_nt * 2, n_mc=args.reward_mc,
                       seed=args.seed_ocean + 999)


def _bootstrap_paired(diffs, n_boot=10000, seed=0):
    """IC95 bootstrap + fraction de rééchantillonnages > 0 sur des
    différences appariées."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    means = np.array([rng.choice(diffs, len(diffs), replace=True).mean()
                      for _ in range(n_boot)])
    return (float(diffs.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
            float((means > 0).mean()))


def compare_baselines(args, env, rm, policy=None, policies=None):
    """Compare les stratégies de placement à budget N fixé, jugées par le
    skill AE sur un jeu CRN INDÉPENDANT de la reward. Figure + JSON."""
    import json as _json
    assert rm.ae_model is not None, "--compare requiert --ae_checkpoint"
    n = args.compare_n or (args.n_min + args.n_max) // 2
    rng = np.random.default_rng(args.seed_buoys + 100)

    judge = _make_judge(args, env)
    if policies is None:
        policies = [policy] if policy is not None else []
    fixed = list(getattr(env, "fixed_positions", []))

    lbl = (f"réseau imposé {len(fixed)} stations + {n} AJOUTS"
           if fixed else f"N={n}")
    print(f"\n══ Comparaison des placements ({lbl}, juge = skill AE, "
          f"jeu d'éval indépendant) ══")
    results = {"fixed_positions": [list(p) for p in fixed]} if fixed else {}
    if fixed:
        s_fx = judge.skill(fixed)
        results["fixed_only"] = {"mean": float(s_fx), "std": 0.0}
        print(f"  réseau imposé seul : skill = {s_fx:.4f} "
              f"(la valeur marginale des ajouts se lit par rapport à ceci)")

    sk = [judge.skill(fixed + _baseline_random(env, n, rng, fixed=fixed))
          for _ in range(args.compare_seeds)]
    results["random"] = {"mean": float(np.mean(sk)), "std": float(np.std(sk)),
                         "runs": [float(s) for s in sk]}
    print(f"  random (x{args.compare_seeds})  : "
          f"skill = {np.mean(sk):.4f} ± {np.std(sk):.4f}")

    sk = [judge.skill(fixed + _baseline_maximin(env, n, rng, fixed=fixed))
          for _ in range(args.compare_seeds)]
    results["maximin"] = {"mean": float(np.mean(sk)), "std": float(np.std(sk)),
                          "runs": [float(s) for s in sk]}
    print(f"  maximin (x{args.compare_seeds}) : "
          f"skill = {np.mean(sk):.4f} ± {np.std(sk):.4f}")

    net = _baseline_greedy_variance(env, n, fixed=fixed)
    s = judge.skill(fixed + net)
    results["greedy_variance"] = {"mean": float(s), "std": 0.0}
    print(f"  greedy-variance : skill = {s:.4f}")

    if rm.sage is not None:
        print(f"  greedy-SAGE ({n} ajouts séquentiels)...")
        net = _baseline_greedy_sage(env, rm, n,
                                    cand_stride=args.sage_stride,
                                    fixed=fixed)
        s = judge.skill(fixed + net)
        results["greedy_sage"] = {"mean": float(s), "std": 0.0,
                                  "positions": [list(p) for p in net]}
        print(f"  greedy-SAGE     : skill = {s:.4f}")

    if policies:
        # Chaque rollout est APPARIÉ à un maximin frais au même N : le
        # bootstrap sur les différences appariées teste "le RL place-t-il
        # mieux que l'espacement pur, à budget strictement égal ?"
        sk, ns, per_pol = [], [], []
        seen_nets = {}          # réseau unique -> (skill, n_add, policy)
        for p_i, pol in enumerate(policies):
            sk_p = []
            for _ in range(args.compare_seeds):
                net = _rollout_policy_network(
                    env, pol, k=(n if env.mdp == "additive" else None))
                n_add = len(net) - len(fixed)
                s = judge.skill(net)
                sk.append(s); sk_p.append(s); ns.append(n_add)
                seen_nets.setdefault(frozenset(net), (s, n_add, p_i))
            per_pol.append({"mean": float(np.mean(sk_p)),
                            "std": float(np.std(sk_p))})
        # ── Test apparié sur les RÉSEAUX UNIQUES ────────────────────────────
        # Une politique déterministe (init=empty) produit le même réseau à
        # chaque rollout : compter chaque rollout comme une observation
        # gonflerait artificiellement le n du bootstrap. L'unité statistique
        # est le réseau distinct (en pratique : la politique/seed).
        diffs = []
        for (s, n_add, p_i) in seen_nets.values():
            if n_add >= 2:
                mm = _baseline_maximin(env, n_add, rng, fixed=fixed)
                diffs.append(s - judge.skill(fixed + mm))
        results[f"rl_{env.reward_mode}"] = {
            "mean": float(np.mean(sk)), "std": float(np.std(sk)),
            "n_policies": len(policies), "rollouts": len(sk),
            "n_add_mean": float(np.mean(ns)),
            "per_policy": per_pol, "runs": [float(s) for s in sk]}
        print(f"  RL ({env.reward_mode}, {len(policies)} pol x "
              f"{args.compare_seeds} rollouts) : "
              f"skill = {np.mean(sk):.4f} ± {np.std(sk):.4f} "
              f"({'ajouts' if fixed else 'N final'} moyen : {np.mean(ns):.1f})")
        if len(policies) > 1:
            for p_i, pp in enumerate(per_pol):
                print(f"    politique {p_i} : {pp['mean']:.4f} ± {pp['std']:.4f}")
        if diffs:
            n_pos = sum(d > 0 for d in diffs)
            if len(diffs) >= 5:
                m, lo, hi, frac = _bootstrap_paired(diffs,
                                                    seed=args.seed_buoys)
                results["rl_vs_maximin_paired"] = {
                    "mean_diff": m, "ci95": [lo, hi], "frac_positive": frac,
                    "n_unique_networks": len(diffs)}
                verdict = ("significativement MEILLEUR que maximin" if lo > 0
                           else "significativement MOINS BON que maximin"
                           if hi < 0 else
                           "statistiquement indistinguable de maximin")
                print(f"  RL vs maximin à N ÉGAL "
                      f"({len(diffs)} réseaux uniques) : "
                      f"Δskill = {np.mean(diffs):+.4f}  "
                      f"IC95 [{lo:+.4f}, {hi:+.4f}] | P(Δ>0) = {frac:.2f}")
                print(f"  → à budget égal, le RL est {verdict}")
            else:
                results["rl_vs_maximin_paired"] = {
                    "mean_diff": float(np.mean(diffs)),
                    "n_unique_networks": len(diffs),
                    "n_positive": int(n_pos)}
                print(f"  RL vs maximin à N ÉGAL : seulement {len(diffs)} "
                      f"réseau(x) unique(s) — {n_pos}/{len(diffs)} en faveur "
                      f"du RL, Δskill moyen {np.mean(diffs):+.4f}")
                print(f"  ⚠ Trop peu de réseaux distincts pour un IC : "
                      f"augmenter --n_policies (les politiques init=empty "
                      f"sont déterministes, --compare_seeds ne diversifie pas)")

    # Référence : réseau vide (climatologie)
    if not fixed:
        results["empty_reference"] = {"mean": float(judge.skill([])), "std": 0.0}
        print(f"  (référence réseau vide : skill = "
              f"{results['empty_reference']['mean']:.4f})")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rl_baselines_comparison.json", "w") as f:
        _json.dump({"n": int(n), "results": results}, f, indent=2)

    # Figure
    BG = "#0a1628"
    names = [k for k in results
             if k not in ("empty_reference", "fixed_only", "fixed_positions")
             and "mean" in results[k]]
    means = [results[k]["mean"] for k in names]
    stds  = [results[k].get("std", 0.0) for k in names]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor("#050d1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
    colors = ["#6baed6", "#9ecae1", "#fdae6b", "#74c476", "#fc8d59"][:len(names)]
    ax.bar(names, means, yerr=stds, capsize=4, color=colors,
           edgecolor="black", linewidth=0.6)
    ref_key = "fixed_only" if fixed else "empty_reference"
    ref_lbl = (f"réseau imposé seul ({len(fixed)} stations)" if fixed
               else "réseau vide (climatologie)")
    ax.axhline(results[ref_key]["mean"], ls="--", color="white",
               lw=1, alpha=0.6, label=ref_lbl)
    ax.set_ylabel("Skill AE (1 − RMSE) — jeu d'éval indépendant",
                  color="white", fontsize=9)
    ttl = (f"Ajout de {n} capteurs au réseau imposé — comparaison"
           if fixed else f"Placement de {n} capteurs — comparaison des stratégies")
    ax.set_title(ttl,
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    ax.legend(facecolor="#050d1a", labelcolor="white", fontsize=8)
    ax.grid(alpha=0.2, color="white", axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "rl_baselines_comparison.png", dpi=140,
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Comparaison → {out_dir}/rl_baselines_comparison.png")
    return results


def sweep_baselines(args, env, rm, policies=None):
    """Balayage skill(N) pour chaque stratégie + points RL (N choisi par la
    politique). greedy-SAGE et maximin sont des constructions EMBOÎTÉES :
    la chaîne est calculée une fois jusqu'à N_max, les préfixes donnent
    tous les N intermédiaires — le sweep coûte à peine plus qu'un point.
    Figure rl_skill_vs_n.png + JSON."""
    import json as _json
    assert rm.ae_model is not None, "--compare_sweep requiert --ae_checkpoint"
    Ns = sorted(int(x) for x in args.compare_sweep.split(","))
    n_max = max(Ns)
    rng = np.random.default_rng(args.seed_buoys + 200)
    judge = _make_judge(args, env)
    policies = policies or []
    fixed = list(getattr(env, "fixed_positions", []))

    lbl = f"AJOUTS au réseau imposé ({len(fixed)} st.)" if fixed else "N"
    print(f"\n══ Sweep skill vs {lbl} — N ∈ {Ns} (juge = skill AE indépendant) ══")
    out = {"Ns": Ns, "curves": {}, "rl_points": [],
           "fixed_positions": [list(p) for p in fixed]}
    if fixed:
        out["fixed_only_skill"] = float(judge.skill(fixed))
        print(f"  réseau imposé seul : skill = {out['fixed_only_skill']:.4f}")

    # random : indépendant par N
    cur = {N: [judge.skill(fixed + _baseline_random(env, N, rng, fixed=fixed))
               for _ in range(args.compare_seeds)] for N in Ns}
    out["curves"]["random"] = {str(N): {"mean": float(np.mean(v)),
                                        "std": float(np.std(v))}
                               for N, v in cur.items()}
    print("  random          : " + "  ".join(
        f"N{N}={np.mean(v):.3f}" for N, v in cur.items()))

    # maximin : chaînes emboîtées (une par seed)
    cur = {N: [] for N in Ns}
    for _ in range(args.compare_seeds):
        chain = _baseline_maximin(env, n_max, rng, fixed=fixed)
        for N in Ns:
            cur[N].append(judge.skill(fixed + chain[:N]))
    out["curves"]["maximin"] = {str(N): {"mean": float(np.mean(v)),
                                         "std": float(np.std(v))}
                                for N, v in cur.items()}
    print("  maximin         : " + "  ".join(
        f"N{N}={np.mean(v):.3f}" for N, v in cur.items()))

    # greedy-variance : par N (d_min dépend de N, non emboîté)
    cur = {N: [judge.skill(fixed + _baseline_greedy_variance(env, N,
                                                             fixed=fixed))]
           for N in Ns}
    out["curves"]["greedy_variance"] = {str(N): {"mean": float(np.mean(v)),
                                                 "std": 0.0}
                                        for N, v in cur.items()}
    print("  greedy-variance : " + "  ".join(
        f"N{N}={np.mean(v):.3f}" for N, v in cur.items()))

    # greedy-SAGE : une chaîne emboîtée jusqu'à n_max
    if rm.sage is not None:
        print(f"  greedy-SAGE : chaîne jusqu'à N={n_max}...")
        chain = _baseline_greedy_sage(env, rm, n_max,
                                      cand_stride=args.sage_stride,
                                      verbose=False, fixed=fixed)
        cur = {N: [judge.skill(fixed + chain[:N])] for N in Ns}
        out["curves"]["greedy_sage"] = {str(N): {"mean": float(np.mean(v)),
                                                 "std": 0.0}
                                        for N, v in cur.items()}
        out["greedy_sage_chain"] = [list(p) for p in chain]
        print("  greedy-SAGE     : " + "  ".join(
            f"N{N}={np.mean(v):.3f}" for N, v in cur.items()))

    # RL : points (N choisi, skill) par rollout
    for p_i, pol in enumerate(policies):
        Ns_rl = Ns if env.mdp == "additive" else [None] * args.compare_seeds
        for k_force in Ns_rl:
            net = _rollout_policy_network(env, pol, k=k_force)
            out["rl_points"].append({"policy": p_i,
                                     "n": len(net) - len(fixed),
                                     "skill": float(judge.skill(net))})
    if out["rl_points"]:
        print(f"  RL              : {len(out['rl_points'])} rollouts, "
              f"N ∈ [{min(p['n'] for p in out['rl_points'])}, "
              f"{max(p['n'] for p in out['rl_points'])}]")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rl_skill_vs_n.json", "w") as f:
        _json.dump(out, f, indent=2)

    # ── Figure ───────────────────────────────────────────────────────────────
    BG = "#0a1628"
    fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=BG)
    ax.set_facecolor("#050d1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
    styles = {"random":          ("#6baed6", "o", "random"),
              "maximin":         ("#9ecae1", "s", "maximin"),
              "greedy_variance": ("#fdae6b", "^", "greedy-variance"),
              "greedy_sage":     ("#74c476", "D", "greedy-SAGE (émulateur)")}
    for key, (c, mk, lbl) in styles.items():
        if key not in out["curves"]:
            continue
        m = np.array([out["curves"][key][str(N)]["mean"] for N in Ns])
        s = np.array([out["curves"][key][str(N)]["std"]  for N in Ns])
        ax.plot(Ns, m, "-", marker=mk, color=c, lw=1.6, ms=5, label=lbl)
        if s.max() > 0:
            ax.fill_between(Ns, m - s, m + s, color=c, alpha=0.15)
    if out["rl_points"]:
        xs = [p["n"] for p in out["rl_points"]]
        ys = [p["skill"] for p in out["rl_points"]]
        ax.scatter(xs, ys, s=40, c="#fc8d59", edgecolors="black",
                   linewidths=0.6, zorder=5,
                   label=f"RL {env.reward_mode} (N choisi par la politique)")
    if fixed:
        ax.axhline(out["fixed_only_skill"], ls="--", color="white", lw=1,
                   alpha=0.6, label=f"réseau imposé seul ({len(fixed)} st.)")
    ax.set_xlabel("Nombre d'ajouts N" if fixed else "Nombre de capteurs N",
                  color="white", fontsize=10)
    ax.set_ylabel("Skill AE (1 − RMSE) — jeu d'éval indépendant",
                  color="white", fontsize=10)
    ax.set_title("Skill vs nombre d'ajouts au réseau imposé, par stratégie"
                 if fixed else
                 "Skill de reconstruction vs taille du réseau, par stratégie",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    ax.legend(facecolor="#050d1a", labelcolor="white", fontsize=8)
    ax.grid(alpha=0.2, color="white")
    fig.tight_layout()
    fig.savefig(out_dir / "rl_skill_vs_n.png", dpi=140, facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Sweep → {out_dir}/rl_skill_vs_n.png")
    return out


def plot_synthesis(args, env, rm, policies):
    """Figure de synthèse du scénario contraint : carte des ajouts de chaque
    stratégie à k fixé, sur fond de sigma d'anomalies SST, stations imposées
    en étoiles, skills (juge indépendant) en légende. Les réseaux de TOUTES
    les politiques RL sont superposés : les recouvrements matérialisent les
    zones de consensus inter-seeds."""
    fixed = env.fixed_positions
    k = args.compare_n or (env.n_min + env.n_max) // 2
    rng = np.random.default_rng(args.seed_buoys + 300)
    judge = _make_judge(args, env)

    print(f"\n══ Figure de synthèse (k={k} ajouts) ══")
    nets = {}
    net = _baseline_maximin(env, k, rng, fixed=fixed)
    nets["maximin"] = (net, judge.skill(fixed + net))
    net = _baseline_greedy_variance(env, k, fixed=fixed)
    nets["greedy-variance"] = (net, judge.skill(fixed + net))
    if rm is not None and rm.sage is not None:
        net = _baseline_greedy_sage(env, rm, k, cand_stride=args.sage_stride,
                                    fixed=fixed, verbose=False)
        nets["greedy-SAGE"] = (net, judge.skill(fixed + net))
    rl_nets = []
    for pol in policies:
        full = _rollout_policy_network(
            env, pol, k=(k if env.mdp == "additive" else None))
        adds = [p for p in full if p not in fixed]
        rl_nets.append((adds, judge.skill(full)))

    # ── Carte ────────────────────────────────────────────────────────────────
    BG = "#0a1628"
    sigma = env.T.std(axis=0)
    if OCEAN is not None:
        sigma = np.where(OCEAN > 0.5, sigma, np.nan)
    if GLORYS is not None:
        ext = [float(GLORYS.lon.min()), float(GLORYS.lon.max()),
               float(GLORYS.lat.min()), float(GLORYS.lat.max())]
        def xy(p):
            la, lo = GLORYS.ij_to_latlon(*p)
            return lo, la
        xl, yl = "Longitude (°E)", "Latitude (°N)"
    else:
        ext = [0, NY, 0, NX]
        def xy(p):
            return p[1], p[0]
        xl, yl = "y (px)", "x (px)"

    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=BG)
    ax.set_facecolor("#050d1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
    im = ax.imshow(sigma, extent=ext, origin="lower", cmap="cividis",
                   aspect="auto", alpha=0.85)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("σ anomalies SST (unités normalisées)", color="white",
                 fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    for p in fixed:
        x, y = xy(p)
        ax.plot(x, y, "*", ms=22, color="#ffd93d", mec="black", mew=1.2,
                zorder=8)
    styles = {"maximin":         ("#9ecae1", "s", 95),
              "greedy-variance": ("#fdae6b", "^", 95),
              "greedy-SAGE":     ("#74c476", "D", 85)}
    for name, (net, sk) in nets.items():
        c, mk, ms = styles[name]
        xs, ys = zip(*[xy(p) for p in net])
        ax.scatter(xs, ys, s=ms, c=c, marker=mk, edgecolors="black",
                   linewidths=0.8, zorder=6,
                   label=f"{name} (skill {sk:.3f})")
    if rl_nets:
        sk_all = [s for _, s in rl_nets]
        for adds, _ in rl_nets:
            xs, ys = zip(*[xy(p) for p in adds])
            ax.scatter(xs, ys, s=60, c="#fc8d59", marker="o",
                       edgecolors="black", linewidths=0.5, alpha=0.55,
                       zorder=7)
        ax.scatter([], [], s=60, c="#fc8d59", marker="o", edgecolors="black",
                   label=f"RL {env.reward_mode} x{len(rl_nets)} politiques "
                         f"(skill {np.mean(sk_all):.3f} ± {np.std(sk_all):.3f})")
    ax.scatter([], [], s=180, c="#ffd93d", marker="*", edgecolors="black",
               label="stations imposées (PIRATA nominal)")

    ax.set_xlabel(xl, color="white", fontsize=10)
    ax.set_ylabel(yl, color="white", fontsize=10)
    ax.set_title(f"Extension du réseau imposé — {k} ajouts par stratégie "
                 f"(juge : skill AE, jeu indépendant)",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    leg = ax.legend(facecolor="#050d1a", labelcolor="white", fontsize=8.5,
                    loc="upper right", framealpha=0.85)
    leg.get_frame().set_edgecolor("#2a4a7a")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"rl_pirata_synthesis_k{k}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Synthèse → {out}")
    for name, (net, sk) in nets.items():
        print(f"    {name:16s}: skill {sk:.4f}")
    if rl_nets:
        print(f"    RL (x{len(rl_nets)})       : skill "
              f"{np.mean(sk_all):.4f} ± {np.std(sk_all):.4f}")


# =========================================================================
#  POLITIQUE PPO
# =========================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256, conflict=None):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

        # Matrice de conflit en buffer : la séparation minimale devient une
        # fonction de la seule obs, comme forbid_active. Sauvegardée avec le
        # state_dict -> un checkpoint reste rejouable à l'identique.
        if conflict is None:
            conflict = np.zeros((n_actions, n_actions), dtype=bool)
        self.register_buffer("conflict",
                             torch.as_tensor(np.asarray(conflict),
                                             dtype=torch.bool))

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def masked_logits(self, obs, forbid_active=False):
        """Deux masques cumulés, tous deux dérivés de l'obs
        (obs[..., :K] = active_mask) : identiques à l'échantillonnage et à
        l'update PPO — pas de biais de ratio.
          1. MDP additif : les candidats déjà actifs sont interdits.
          2. Séparation minimale : un candidat en conflit avec une bouée déjà
             posée est interdit. Le DÉSACTIVER reste permis (MDP toggle).
        """
        logits, value = self(obs)
        K = logits.shape[-1]
        active = obs[..., :K] > 0.5
        invalid = torch.zeros_like(active)
        if forbid_active:
            invalid = invalid | active
        if bool(self.conflict.any()):
            conf = active.float() @ self.conflict.float()
            invalid = invalid | ((conf > 0) & ~active)
        # Filet de sécurité : une ligne entièrement masquée produit des NaN
        # dans Categorical. Ne devrait pas arriver (n_max <= n_feasible_max),
        # mais un checkpoint rechargé avec d'autres n_min/n_max le pourrait.
        invalid = invalid & ~invalid.all(dim=-1, keepdim=True)
        logits = logits.masked_fill(invalid, -1e9)
        return logits, value

    def get_action(self, obs, deterministic=False, forbid_active=False):
        logits, value = self.masked_logits(obs, forbid_active)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class RolloutBuffer:
    def __init__(self, sz, obs_dim):
        self.obs = np.zeros((sz, obs_dim), np.float32)
        self.actions = np.zeros(sz, np.int64)
        self.rewards = np.zeros(sz, np.float32)
        self.dones = np.zeros(sz, np.float32)
        self.log_probs = np.zeros(sz, np.float32)
        self.values = np.zeros(sz, np.float32)
        self.ptr = 0; self.size = sz

    def add(self, obs, a, r, d, lp, v):
        i = self.ptr
        self.obs[i]=obs; self.actions[i]=a; self.rewards[i]=r
        self.dones[i]=float(d); self.log_probs[i]=lp; self.values[i]=v
        self.ptr = (self.ptr+1) % self.size

    def compute_returns(self, last_v, gamma=0.99, lam=0.95):
        adv = np.zeros(self.size, np.float32); gae = 0.0
        for t in reversed(range(self.size)):
            nv = last_v if t==self.size-1 else self.values[t+1]
            nd = 0.0 if t==self.size-1 else self.dones[t+1]
            delta = self.rewards[t] + gamma*nv*(1-nd) - self.values[t]
            gae = delta + gamma*lam*(1-nd)*gae; adv[t] = gae
        return adv, adv + self.values

    def get_tensors(self, adv, ret, dev):
        return {"obs": torch.tensor(self.obs, device=dev),
                "actions": torch.tensor(self.actions, device=dev),
                "log_probs": torch.tensor(self.log_probs, device=dev),
                "advantages": torch.tensor(adv, device=dev),
                "returns": torch.tensor(ret, device=dev)}


# =========================================================================
#  ENTRAINEMENT PPO (commun)
# =========================================================================

def train_ppo(args, env, label=""):
    prefix = f" [{label}]" if label else ""
    print(f"  PPO{prefix} : {args.rl_steps} steps")

    policy = ActorCritic(env.obs_dim, env.K, conflict=env._conflict).to(DEVICE)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    buf = RolloutBuffer(args.buffer_size, env.obs_dim)
    clip_eps, vf_c, ent_c, n_ep, mb = 0.2, 0.5, 0.01, 4, 64
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hist = {"episode_reward": [], "n_active": [], "info_score": []}
    ep_rews = deque(maxlen=20); best_rew = -np.inf
    obs = env.reset(); ep_r = 0.0

    for step in range(args.rl_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            act, lp, _, val = policy.get_action(obs_t, forbid_active=(env.mdp == "additive"))
        nobs, rew, done, info = env.step(act.item())
        buf.add(obs, act.item(), rew, done, lp.item(), val.item())
        ep_r += rew; obs = nobs
        if done:
            ep_rews.append(ep_r)
            hist["episode_reward"].append(ep_r)
            hist["n_active"].append(info["n_active"])
            hist["info_score"].append(info["total_info"])
            if ep_r > best_rew:
                best_rew = ep_r
                torch.save({"policy_state": policy.state_dict(),
                            "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                            "active_mask": env.active_mask.copy()},
                           out_dir / "rl_best.pt")
            obs = env.reset(); ep_r = 0.0
        if (step+1) % args.buffer_size == 0:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                _, _, _, lv = policy.get_action(obs_t, forbid_active=(env.mdp == "additive"))
            adv, ret = buf.compute_returns(lv.item())
            adv = (adv-adv.mean())/(adv.std()+1e-8)
            batch = buf.get_tensors(adv, ret, DEVICE)
            idx = np.arange(args.buffer_size)
            for _ in range(n_ep):
                np.random.shuffle(idx)
                for s in range(0, args.buffer_size, mb):
                    m = idx[s:s+mb]
                    lo, va = policy.masked_logits(
                        batch["obs"][m], forbid_active=(env.mdp == "additive"))
                    dist = torch.distributions.Categorical(logits=lo)
                    lp_ = dist.log_prob(batch["actions"][m])
                    ent = dist.entropy().mean()
                    ratio = torch.exp(lp_ - batch["log_probs"][m])
                    a = batch["advantages"][m]
                    loss = (-torch.min(ratio*a, torch.clamp(ratio,1-clip_eps,1+clip_eps)*a).mean()
                            + vf_c*F.mse_loss(va, batch["returns"][m]) - ent_c*ent)
                    optim.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optim.step()
            if len(ep_rews) > 0 and (step+1) % args.buffer_size == 0:
                import time as _t
                if not hasattr(args, "_t0"):
                    args._t0, args._s0 = _t.time(), step + 1
                    rate_txt = "   -- st/s | ETA    -- min"
                else:
                    rate = (step + 1 - args._s0) / max(_t.time() - args._t0, 1e-6)
                    eta = (args.rl_steps - step - 1) / max(rate, 1e-6)
                    rate_txt = f"{rate:5.1f} st/s | ETA {eta/60:5.1f} min"
                print(f"    Step {step+1:6d}/{args.rl_steps} "
                      f"| R={np.mean(list(ep_rews)[-20:]):+.3f} "
                      f"| Best={best_rew:+.3f} | {rate_txt}", flush=True)
    print(f"  Best reward: {best_rew:.4f}")

    # Courbes
    sfx = f"_{label}" if label else ""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"PPO{prefix}", fontsize=14, fontweight="bold")
    axes[0,0].plot(hist["episode_reward"], alpha=0.4, color="steelblue")
    w = max(1, len(hist["episode_reward"])//20)
    if len(hist["episode_reward"]) >= w:
        sm = np.convolve(hist["episode_reward"], np.ones(w)/w, mode="valid")
        axes[0,0].plot(range(w-1, len(hist["episode_reward"])), sm, color="navy", lw=2)
    axes[0,0].set_title("Reward/ep"); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(hist["n_active"], color="orange", alpha=0.6)
    axes[0,1].axhline(env.n_min, color="red", ls="--"); axes[0,1].axhline(env.n_max, color="red", ls=":")
    axes[0,1].set_title("N actifs"); axes[0,1].grid(True, alpha=0.3)
    axes[1,0].plot(hist["info_score"], color="green", alpha=0.6)
    axes[1,0].set_title("Info score"); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(np.cumsum(hist["episode_reward"]), color="#9b59b6", alpha=0.8)
    axes[1,1].set_title("Reward cumulee"); axes[1,1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"rl_training_curves{sfx}.png", dpi=150); plt.close()
    return policy, hist


# =========================================================================
#  HELPERS
# =========================================================================

def _sweep_info(env, policy, n_range, n_trials=20):
    policy.eval(); points = []
    for nt_ in n_range:
        scores = []
        for trial in range(n_trials):
            env.active_mask[:] = 0.0
            env.active_mask[np.random.choice(env.K, min(nt_, env.K), replace=False)] = 1.0
            if trial < n_trials//2:
                obs = env._get_obs()
                with torch.no_grad():
                    for _ in range(env.ep_len):
                        ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        a, _, _, _ = policy.get_action(ot, deterministic=False)
                        obs, _, d, _ = env.step(a.item())
                        if d: break
            scores.append(env._compute_info_reward())
        points.append({"n_buoys": nt_, "info_mean": float(np.mean(scores)),
                        "info_std": float(np.std(scores))})
    return points

def _run_policy_config(env, policy, n_target):
    env.active_mask[:] = 0.0
    env.active_mask[np.random.choice(env.K, min(n_target, env.K), replace=False)] = 1.0
    obs = env._get_obs(); policy.eval()
    with torch.no_grad():
        for _ in range(env.ep_len):
            ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a, _, _, _ = policy.get_action(ot, deterministic=True)
            obs, _, d, _ = env.step(a.item())
            if d: break
    return np.where(env.active_mask > 0.5)[0], float(env._compute_info_reward())

def _n_light(n_star):
    nl = max(2, int(n_star)//2)
    if nl >= int(n_star): nl = max(2, int(n_star) - max(3, int(n_star)//3))
    return nl


# =========================================================================
#  METHODE 1 -- PARETO (Kneedle)
# =========================================================================

def compute_pareto(env, policy, args):
    print("\n-- Methode PARETO --")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_range = range(env.n_min, min(env.K, env.n_max + 1))
    points = _sweep_info(env, policy, n_range)
    iv = np.array([p["info_mean"] for p in points])
    nv = np.array([p["n_buoys"] for p in points])
    ist = np.array([p["info_std"] for p in points])
    # Kneedle
    x0,y0 = float(nv[0]),float(iv[0]); x1,y1 = float(nv[-1]),float(iv[-1])
    nn_ = (nv-x0)/(x1-x0+1e-9); ii_ = (iv-y0)/(y1-y0+1e-9)
    dist = np.abs(ii_-nn_)/np.sqrt(2)
    conc = ii_ >= nn_-0.05
    if conc.any():
        cands = np.where(conc)[0]; elbow = cands[int(np.argmax(dist[cands]))]
    else:
        elbow = int(np.argmax(dist))
    if elbow <= 1: elbow = len(nv)//3
    elif elbow >= len(nv)-2: elbow = 2*len(nv)//3
    n_star = int(np.clip(nv[elbow], env.n_min, env.n_max))
    # Pareto mask
    pmask = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        pmask[i] = not any((iv[j]>=iv[i] and nv[j]<=nv[i]) and (iv[j]>iv[i] or nv[j]<nv[i])
                           for j in range(len(points)) if j!=i)
    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PARETO -- Info vs N", fontsize=14, fontweight="bold")
    axes[0].fill_between(nv, iv-ist, iv+ist, alpha=0.2, color="steelblue")
    axes[0].plot(nv, iv, "o-", color="steelblue", ms=4, label="Info")
    axes[0].scatter(nv[pmask], iv[pmask], c=nv[pmask], cmap="plasma", s=120, zorder=5,
                    edgecolors="black", lw=0.8, label="Pareto")
    axes[0].axvline(n_star, color="red", lw=1.5, ls="--", label=f"N*={n_star}")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("N"); axes[0].set_ylabel("Info")
    mg = np.gradient(iv, nv)
    axes[1].bar(nv, mg, color=["#2ecc71" if g>0 else "#e74c3c" for g in mg], alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("N"); axes[1].set_ylabel("Gain marginal"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir/"rl_pareto_front.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"  N* = {n_star} (Kneedle)")
    return points, n_star


# =========================================================================
#  METHODE 2 -- EFFICIENCY : gain net d'information
# =========================================================================

def compute_efficiency(env, policy, args):
    """
    Gain net : eta(N) = info(N) - info(n_min) - beta*(N - n_min).
    beta = 70% de la pente moyenne info → pic vers 60-70% de [n_min, n_max].
    N* = n_min + argmax(eta).
    """
    print("\n-- Methode EFFICIENCY --")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_range = range(env.n_min, min(env.K, env.n_max + 1))
    points = _sweep_info(env, policy, n_range)
    iv = np.array([p["info_mean"] for p in points])
    nv = np.array([p["n_buoys"] for p in points])
    ist = np.array([p["info_std"] for p in points])

    # Gain net : info relative a n_min, penalisee par cout lineaire
    info_base = iv[0]
    avg_slope = (iv[-1] - iv[0]) / (nv[-1] - nv[0] + 1e-9)
    beta = avg_slope * 0.9  # pénalité = 90% de la pente moyenne
    eta = (iv - info_base) - beta * (nv - nv[0])
    best = int(np.argmax(eta))
    n_star = int(nv[best])
    eta_per_buoy = iv / nv.astype(float)

    # Figure 1x3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EFFICIENCY -- Gain net d'information", fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.fill_between(nv, iv-ist, iv+ist, alpha=0.2, color="steelblue")
    ax.plot(nv, iv, "o-", color="steelblue", ms=4, label="Info(N)")
    cost_line = info_base + beta * (nv - nv[0])
    ax.plot(nv, cost_line, "--", color="#e74c3c", lw=1.5, alpha=0.7, label=f"Cout (beta={beta:.4f})")
    ax.axvline(n_star, color="red", lw=1.5, ls="--", label=f"N*={n_star}")
    ax.set_xlabel("N"); ax.set_ylabel("Info"); ax.set_title("Info vs Cout")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(nv, eta, "s-", color="#e67e22", ms=5, lw=2, label="Gain net")
    ax.axvline(n_star, color="red", lw=1.5, ls="--")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.scatter([n_star], [eta[best]], c="red", s=200, zorder=6, marker="*", label=f"N*={n_star}")
    ax.fill_between(nv, 0, eta, where=eta>0, alpha=0.15, color="#2ecc71")
    ax.fill_between(nv, 0, eta, where=eta<0, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("N"); ax.set_ylabel("Gain net"); ax.set_title("eta = gain - cout")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax = axes[2]
    ax.plot(nv, eta_per_buoy*1000, "^-", color="#9b59b6", ms=4, lw=1.5, label="Info/N (x1000)")
    ax.axvline(n_star, color="red", lw=1.5, ls="--")
    ax.set_xlabel("N"); ax.set_ylabel("Info/N (x1000)"); ax.set_title("Rendement par capteur")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir/"rl_efficiency.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"  N* = {n_star} | eta* = {eta[best]:.4f} | info = {iv[best]:.3f}")
    return points, n_star


# =========================================================================
#  METHODE 3 -- SCALARIZED (sweep lambda)
# =========================================================================

def compute_scalarized(env_T, env_S, policy_std, args):
    print("\n-- Methode SCALARIZED (sweep lambda) --")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lambdas = [0.001, 0.005, 0.01, 0.02]
    results = []; steps_lam = max(1000, args.rl_steps // 4)
    for lam in lambdas:
        print(f"\n  lambda = {lam} ({steps_lam} steps)...")
        env_lam = OceanNetworkEnv(env_T, env_S, grid_x=args.grid_x, grid_y=args.grid_y,
                                   n_min=2, n_max=args.n_max+20,
                                   episode_len=args.episode_len, marginal_cost=lam,
                                   ocean_mask=OCEAN,
                                   reward_mode=getattr(args, "reward", "heuristic"),
                                   reward_model=getattr(args, "_rm", None),
                                   mdp=getattr(args, "mdp", "toggle"),
                                   sage_scale=getattr(args, "sage_scale", 0.3),
                                   w_terminal=getattr(args, "w_terminal", 5.0),
                                   fixed_positions=getattr(args, "_fixed", None),
                                   init_mode=getattr(args, "init", "auto"),
                                   min_sep_km=getattr(args, "min_sep_km",
                                                      MIN_BUOY_SEP_KM),
                                   geo=GLORYS)
        args_lam = argparse.Namespace(**vars(args)); args_lam.rl_steps = steps_lam
        pol_lam, _ = train_ppo(args_lam, env_lam, label=f"lam={lam}")
        idx, info = _run_policy_config(env_lam, pol_lam, env_lam.n_max)
        n_act = len(idx); eta = info / (1+np.log(max(2, n_act)))
        results.append({"lambda": lam, "n_active": n_act, "info": info, "eta": eta,
                         "policy": pol_lam, "active_idx": idx})
        print(f"    -> N={n_act} | info={info:.3f} | eta={eta:.4f}")
    best = max(results, key=lambda r: r["eta"]); n_star = best["n_active"]
    torch.save({"policy_state": best["policy"].state_dict(), "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                "active_mask": np.zeros(0)}, out_dir/"rl_best.pt")
    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SCALARIZED -- Sweep lambda", fontsize=14, fontweight="bold")
    lams = [r["lambda"] for r in results]
    ns = [r["n_active"] for r in results]
    infos = [r["info"] for r in results]
    etas = [r["eta"] for r in results]
    axes[0].bar(range(len(lams)), ns, color=["#3498db","#2ecc71","#e67e22","#e74c3c"], alpha=0.8)
    axes[0].set_xticks(range(len(lams))); axes[0].set_xticklabels([f"l={l}" for l in lams])
    axes[0].set_ylabel("N capteurs"); axes[0].set_title("N par lambda"); axes[0].grid(True, alpha=0.3)
    sc = axes[1].scatter(ns, infos, c=lams, cmap="RdYlGn_r", s=200, zorder=5, edgecolors="black", lw=1.2)
    for r in results:
        axes[1].annotate(f"l={r['lambda']}", (r["n_active"], r["info"]),
                         textcoords="offset points", xytext=(8,5), fontsize=8)
    axes[1].scatter([best["n_active"]], [best["info"]], marker="*", c="red", s=400, zorder=6)
    plt.colorbar(sc, ax=axes[1], label="lambda")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("Info"); axes[1].set_title("Info vs N (*=best eta)"); axes[1].grid(True, alpha=0.3)
    colors = ["#e74c3c" if r is best else "#3498db" for r in results]
    axes[2].bar(range(len(lams)), etas, color=colors, alpha=0.8)
    axes[2].set_xticks(range(len(lams))); axes[2].set_xticklabels([f"l={l}" for l in lams])
    axes[2].set_ylabel("eta"); axes[2].set_title("eta = info/(1+log N)"); axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir/"rl_scalarized.png", dpi=150, bbox_inches="tight"); plt.close()
    points = [{"n_buoys": r["n_active"], "info_mean": r["info"], "info_std": 0.0} for r in results]
    print(f"\n  Best: lam={best['lambda']} -> N*={n_star} | eta={best['eta']:.4f}")
    return points, n_star


# =========================================================================
#  DISPATCH
# =========================================================================

def run_rl_method(env, policy, args):
    method = getattr(args, "rl_method", "pareto")
    if method == "efficiency":
        return compute_efficiency(env, policy, args)
    elif method == "scalarized":
        return compute_scalarized(env.T, env.S, policy, args)
    else:
        return compute_pareto(env, policy, args)


# =========================================================================
#  VISUALISATIONS
# =========================================================================

def visualize_two_configs(env, n_star, policy, args, best_mask=None):
    from matplotlib.colors import LinearSegmentedColormap
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    oc = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    BG = "#0a1628"; nl = _n_light(n_star)
    if best_mask is not None:
        env.active_mask = best_mask.copy()
        di = np.where(best_mask>0.5)[0]; dinf = float(env._compute_info_reward())
        dl, dn = "Dense (retenue)", "-> GNN & AE"
    else:
        di, dinf = _run_policy_config(env, policy, int(n_star))
        dl, dn = "Dense (N*)", f"N*={n_star}"
    li, linf = _run_policy_config(env, policy, nl)
    ap = np.array(env.candidate_positions); Tb = env.T[0]; vm,vM = float(env.T.min()),float(env.T.max())
    method = getattr(args, "rl_method", "pareto").upper()
    fig = plt.figure(figsize=(18,8), facecolor=BG)
    fig.suptitle(f"RL [{method}] -- Dense vs Legere", color="white", fontsize=13, fontweight="bold", y=0.99)
    for col,(idx,inf,lb,clr) in enumerate([(di,dinf,dl,"#6bcb77"),(li,linf,f"Legere (N~{nl})","#ffd93d")]):
        inact = np.setdiff1d(range(env.K), idx)
        ax = fig.add_axes([0.05+col*0.47, 0.10, 0.40, 0.80])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.imshow(Tb.T, cmap=oc, origin="lower", aspect="auto", vmin=vm, vmax=vM, alpha=0.5, extent=[0,NX,0,NY])
        ax.scatter(ap[inact,0], ap[inact,1], c="#1a3a5c", s=14, alpha=0.35)
        sc = ax.scatter(ap[idx,0], ap[idx,1], c=env.field_stats[idx], cmap="plasma",
                        s=90, vmin=0, vmax=1, edgecolors="white", lw=0.8, zorder=6)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label("Var", color="white", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
        ax.set_title(f"{lb}\nN={len(idx)} | Info={inf:.3f}", color=clr, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    lp = max(0, (dinf-linf)/max(dinf,1e-3)*100)
    fig.text(0.5, 0.02, f"Dense: N={len(di)} info={dinf:.3f} | Legere: N={len(li)} info={linf:.3f} | Perte: {lp:.1f}%",
             ha="center", color="#8ab4d4", fontsize=9)
    fig.savefig(out_dir/"rl_two_configs.png", dpi=150, facecolor=BG, bbox_inches="tight"); plt.close()
    print(f"  Dense: N={len(di)} info={dinf:.3f}")
    print(f"  Legere: N={len(li)} info={linf:.3f} (perte {lp:.1f}%)")

def visualize_final_config(env, active_mask, args):
    out_dir = Path(args.output_dir)
    ai = np.where(active_mask>0.5)[0]; ii = np.where(active_mask<=0.5)[0]
    ap = np.array(env.candidate_positions)
    method = getattr(args, "rl_method", "pareto").upper()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"RL [{method}] -- Config optimale", fontsize=13, fontweight="bold")
    axes[0].scatter(ap[ii,0], ap[ii,1], c="lightgray", s=30, alpha=0.4)
    if getattr(env, "fixed_positions", None):
        fp = np.array(env.fixed_positions)
        axes[0].scatter(fp[:,0], fp[:,1], marker="*", c="gold", s=260,
                        edgecolors="black", lw=0.8, zorder=6,
                        label=f"imposées ({len(fp)})")
        axes[0].legend(loc="upper right", fontsize=8)
    sc = axes[0].scatter(ap[ai,0], ap[ai,1], c=env.field_stats[ai], cmap="YlOrRd",
                         s=120, edgecolors="black", lw=0.8, zorder=5)
    plt.colorbar(sc, ax=axes[0], label="Var locale")
    axes[0].set_xlim(0,NX); axes[0].set_ylim(0,NY)
    axes[0].set_title(f"Reseau ({len(ai)} bouees)"); axes[0].grid(True, alpha=0.2)
    axes[1].bar(range(len(ai)), np.sort(env.field_stats[ai])[::-1], color="steelblue")
    axes[1].set_title("Variance (decroissant)"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir/"rl_final_config.png", dpi=150); plt.close()

def save_rl_gif(env, policy, args, n_frames=80):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap
    out_dir = Path(args.output_dir)
    oc = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    BG = "#0a1628"; Tb = env.T[0]; vm,vM = float(env.T.min()),float(env.T.max())
    ca = np.array(env.candidate_positions, dtype=float)
    method = getattr(args, "rl_method", "pareto").upper()
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,5.5),facecolor=BG)
    for ax in (ax1,ax2,ax3):
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
    ax1.imshow(Tb.T, cmap=oc, origin="lower", aspect="auto", vmin=vm, vmax=vM, alpha=0.5, extent=[0,NX,0,NY])
    si = ax1.scatter([],[],c="#1a3a5c",s=14,alpha=0.4)
    sa = ax1.scatter([],[],c="white",s=60,edgecolors="black",lw=0.5,zorder=5)
    ax1.set_xlim(0,NX); ax1.set_ylim(0,NY); ax1.set_title("Actions",color="white",fontsize=9,fontweight="bold")
    ax2.imshow(Tb.T, cmap=oc, origin="lower", aspect="auto", vmin=vm, vmax=vM, alpha=0.5, extent=[0,NX,0,NY])
    si2 = ax2.scatter([],[],c="#1a3a5c",s=14,alpha=0.3)
    sa2 = ax2.scatter([],[],c="white",s=70,edgecolors="white",lw=0.6,zorder=5)
    ax2.set_xlim(0,NX); ax2.set_ylim(0,NY); ax2.set_title("Reseau",color="white",fontsize=9,fontweight="bold")
    ax3.set_xlim(0,n_frames); ax3.set_ylim(0,0.5)
    info_line, = ax3.plot([],[],color="#6bcb77",lw=2,label="Info score")
    ax3_n = ax3.twinx()
    ax3_n.set_ylim(0, env.n_max+5)
    n_line, = ax3_n.plot([],[],color="#ffd93d",lw=1.5,alpha=0.7,label="N actifs")
    ax3_n.tick_params(colors="#ffd93d",labelsize=6)
    vl = ax3.axvline(0,color="white",lw=0.5,alpha=0.3)
    ax3.set_title("Info & N actifs",color="white",fontsize=9,fontweight="bold")
    ax3.tick_params(colors="#6bcb77",labelsize=6)
    ax3.set_ylabel("Info", color="#6bcb77", fontsize=7)
    ax3_n.set_ylabel("N", color="#ffd93d", fontsize=7)
    txt = fig.text(0.5,0.97,"",ha="center",color="white",fontsize=10,fontweight="bold")
    obs = env.reset(); rx,ry_info,ry_n=[],[],[]; el=[]
    def update(f):
        nonlocal obs,el
        if f==0: obs=env.reset(); rx.clear(); ry_info.clear(); ry_n.clear()
        ot = torch.tensor(obs,dtype=torch.float32,device=DEVICE).unsqueeze(0)
        with torch.no_grad(): a,_,_,_ = policy.get_action(ot)
        obs,r,d,info = env.step(a.item())
        cur_info = env._compute_info_reward()
        if d: obs=env.reset()
        ai=np.where(env.active_mask>0.5)[0]; ii=np.where(env.active_mask<=0.5)[0]
        n_active = len(ai)
        si.set_offsets(ca[ii] if len(ii) else np.empty((0,2)))
        sa.set_offsets(ca[ai] if n_active else np.empty((0,2)))
        for ln in el: ln.remove()
        el=[]
        if n_active>1:
            pa=ca[ai]; tree=KDTree(pa)
            for i in range(len(pa)):
                _,idxs=tree.query(pa[i],k=min(3,len(pa)))
                for j in idxs[1:]:
                    ln,=ax2.plot([pa[i,0],pa[j,0]],[pa[i,1],pa[j,1]],color="#2e75b6",alpha=0.5,lw=1.2)
                    el.append(ln)
        si2.set_offsets(ca[ii] if len(ii) else np.empty((0,2)))
        if n_active: sa2.set_offsets(ca[ai]); sa2.set_color(plt.cm.YlOrRd(env.field_stats[ai]))
        else: sa2.set_offsets(np.empty((0,2)))
        rx.append(f); ry_info.append(cur_info); ry_n.append(n_active)
        info_line.set_data(rx,ry_info); n_line.set_data(rx,ry_n)
        vl.set_xdata([f,f])
        # Auto-scale Y
        if len(ry_info) > 1:
            ax3.set_ylim(0, max(ry_info)*1.3+0.01)
        txt.set_text(f"RL [{method}] | Step {f+1}/{n_frames} | N={n_active} | Info={cur_info:.3f}")
        return (si,sa,si2,sa2,info_line,n_line,vl,txt)
    anim = FuncAnimation(fig,update,frames=n_frames,interval=200,blit=False)
    anim.save(str(out_dir/"rl_progression.gif"), writer=PillowWriter(fps=6), dpi=110,
              savefig_kwargs={"facecolor":BG})
    plt.close(); print(f"  GIF -> {out_dir}/rl_progression.gif")

def mark_retained_config_on_pareto(n_ret, info_ret, out_dir):
    import matplotlib.image as mpimg
    out_dir = Path(out_dir)
    for src_name in ["rl_pareto_front.png","rl_efficiency.png","rl_scalarized.png"]:
        src = out_dir/src_name
        if src.exists():
            img = mpimg.imread(str(src))
            fig,ax = plt.subplots(figsize=(14,6),dpi=150)
            ax.imshow(img); ax.axis("off")
            fig.text(0.5,0.01,f"* Config retenue: N={n_ret} | info={info_ret:.3f}",
                     ha="center",color="#ffd93d",fontsize=10,fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3",facecolor="#0a1628",edgecolor="#ffd93d",alpha=0.9))
            fig.savefig(out_dir/f"{src.stem}_pipeline.png",dpi=150,bbox_inches="tight",facecolor="#0a1628")
            plt.close(); break


# =========================================================================
#  POINT D'ENTREE
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Brique 3 -- RL pour OED")
    p.add_argument("--train", action="store_true")
    p.add_argument("--evaluate", action="store_true", help="Evalue avec la methode choisie")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--rl_method", choices=["pareto","efficiency","scalarized"],
                   default="pareto", help="Methode de selection N*")
    p.add_argument("--seed_ocean", type=int, default=42)
    p.add_argument("--seed_buoys", type=int, default=7)
    p.add_argument("--checkpoint", type=str, default="outputs/rl_best.pt")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--rl_steps", type=int, default=50000)
    p.add_argument("--buffer_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--n_min", type=int, default=10)
    p.add_argument("--n_max", type=int, default=40)
    p.add_argument("--episode_len", type=int, default=20)
    p.add_argument("--min_sep_km", type=float, default=MIN_BUOY_SEP_KM,
                   help="Séparation minimale entre bouées, en km. 0 désactive. "
                        "Sans effet si inférieure au pas de la grille "
                        "candidate (synthétique 16x24 : 50 km).")
    p.add_argument("--w_info", type=float, default=1.0)
    p.add_argument("--w_budget", type=float, default=0.5)
    p.add_argument("--gif_frames", type=int, default=80)
    p.add_argument("--data", choices=["synthetic", "glorys"], default="synthetic")
    p.add_argument("--glorys_cache", type=str, default="data/glorys_cache")
    p.add_argument("--time_step", type=int, default=3,
                   help="Sous-échantillonnage temporel du split train GLORYS")
    # Couplage AE / émulateur SAGE
    p.add_argument("--reward", choices=["heuristic", "sage", "ae", "hybrid"],
                   default="heuristic",
                   help="heuristic : historique | sage : shaping dense par "
                        "l'émulateur Brique 2 | ae : Δ skill AE | hybrid : "
                        "sage dense + bonus terminal AE (couplage complet)")
    p.add_argument("--ae_checkpoint",   type=str, default="outputs/ae_best.pt")
    p.add_argument("--sage_checkpoint", type=str, default="outputs/sage_best.pt")
    p.add_argument("--sage_scale",  type=float, default=0.3,
                   help="Échelle du shaping SAGE (scores en z-unités)")
    p.add_argument("--w_terminal",  type=float, default=5.0,
                   help="Poids du bonus terminal skill AE (mode hybrid)")
    p.add_argument("--reward_nt",   type=int,   default=4,
                   help="Instants du jeu CRN de la reward AE")
    p.add_argument("--reward_mc",   type=int,   default=4,
                   help="Passes MC de la reward AE")
    p.add_argument("--sage_stride", type=int,   default=2,
                   help="Stride sur les candidats pour greedy-SAGE (vitesse)")
    # Comparaison aux baselines
    p.add_argument("--compare", action="store_true",
                   help="Compare random/maximin/greedy-var/greedy-SAGE/RL "
                        "à budget fixe, jugés par le skill AE (jeu séparé)")
    p.add_argument("--compare_n",     type=int, default=None)
    p.add_argument("--compare_seeds", type=int, default=5)
    p.add_argument("--synthesis", action="store_true",
                   help="Figure de synthèse : carte des ajouts par stratégie "
                        "à k=compare_n, fond sigma SST, skills en légende")
    p.add_argument("--compare_sweep", type=str, default=None,
                   help="Balayage skill vs N, ex. '10,15,20,25,30,40'")
    p.add_argument("--mdp", choices=["toggle", "additive"], default="toggle",
                   help="additive : épisode = k ajouts exacts (conception), "
                        "sans retraits ni pénalités budget — recommandé "
                        "pour le scénario contraint")
    p.add_argument("--n_policies",    type=int, default=1,
                   help="Politiques entraînées avec des seeds différents "
                        "(consolidation statistique)")
    # Scénario contraint : réseau existant imposé
    p.add_argument("--fixed", type=str, default=None,
                   help="Réseau imposé : 'pirata' (mouillages réels, mode "
                        "glorys) ou liste littérale '[(i,j),...]'. Le budget "
                        "n_min/n_max porte alors sur les AJOUTS.")
    p.add_argument("--init", choices=["auto", "random", "empty"],
                   default="auto",
                   help="Réseau initial des épisodes : auto = empty si "
                        "--fixed, sinon random (historique)")
    return p.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    if not any([args.train, args.evaluate, args.gif, args.compare,
                args.compare_sweep, args.synthesis]):
        print("Usage: python 03_rl.py --train [--evaluate] [--gif] [--compare] [--rl_method pareto|efficiency|scalarized]")
        sys.exit(0)
    set_global_seed(args.seed_ocean)
    print(f"\n  Methode : {args.rl_method.upper()}")
    data = setup_data_source(args)
    if data is not None:
        T, S = data.get_arrays("train", ("T", "S"), normalized=True,
                               step=args.time_step)
        print(f"  Nature run GLORYS12 : {len(T)} jours (split train)")
    else:
        gen = SyntheticOceanGenerator()
        T, S = gen.generate_dataset(nt=NT, seed=args.seed_ocean)
    # ── Scénario contraint : parsing du réseau imposé ───────────────────────
    fixed_positions = None
    if args.fixed:
        if args.fixed.strip().lower() == "pirata":
            if GLORYS is None:
                raise SystemExit("--fixed pirata requiert --data glorys")
            pir = GLORYS.pirata_positions()
            fixed_positions = list(pir.values())
            print(f"  Réseau imposé PIRATA : "
                  + ", ".join(f"{k}@{v}" for k, v in pir.items()))
            print("  (positions NOMINALES — à vérifier sur GTMBA/PMEL "
                  "pour un usage quantitatif)")
        else:
            import ast as _ast
            fixed_positions = [tuple(map(int, p))
                               for p in _ast.literal_eval(args.fixed)]
            if GLORYS is not None:   # rabattre sur l'océan
                fixed_positions = [
                    GLORYS.latlon_to_ij(*GLORYS.ij_to_latlon(i, j),
                                        require_ocean=True)
                    for (i, j) in fixed_positions]
    args._fixed = fixed_positions

    # ── Couplage : RewardModel (AE + émulateur SAGE) si demandé ─────────────
    rm = None
    if args.reward != "heuristic" or args.compare:
        need_ae   = args.reward in ("ae", "hybrid") or args.compare
        need_sage = args.reward in ("sage", "hybrid") or args.compare
        rm = RewardModel(
            T, S,
            ae_checkpoint=args.ae_checkpoint if need_ae else None,
            sage_checkpoint=args.sage_checkpoint if need_sage else None,
            n_t=args.reward_nt, n_mc=args.reward_mc,
            seed=args.seed_ocean)

    env = OceanNetworkEnv(T, S, grid_x=args.grid_x, grid_y=args.grid_y,
                          n_min=args.n_min, n_max=args.n_max,
                          episode_len=args.episode_len, w_info=args.w_info,
                          w_budget=args.w_budget, ocean_mask=OCEAN,
                          reward_mode=args.reward, reward_model=rm,
                          sage_scale=args.sage_scale,
                          w_terminal=args.w_terminal, mdp=args.mdp,
                          fixed_positions=fixed_positions, init_mode=args.init,
                          min_sep_km=args.min_sep_km, geo=GLORYS)
    args._rm = rm
    if args.reward != "heuristic":
        print(f"  Reward couplée : mode '{args.reward}'"
              + (f" (sage_scale={args.sage_scale})"
                 if args.reward in ("sage", "hybrid") else "")
              + (f" (w_terminal={args.w_terminal})"
                 if args.reward == "hybrid" else ""))
    print(f"  K={env.K} | Budget [{args.n_min}, {args.n_max}]")
    policy = None
    policies = []
    if args.train:
        import shutil
        for p_i in range(args.n_policies):
            if args.n_policies > 1:
                set_global_seed(args.seed_ocean + 1000 * (p_i + 1))
            pol, _ = train_ppo(args, env,
                               label=f"seed{p_i}" if args.n_policies > 1 else "")
            policies.append(pol)
            if args.n_policies > 1:
                shutil.copy(Path(args.output_dir) / "rl_best.pt",
                            Path(args.output_dir) / f"rl_best_seed{p_i}.pt")
        policy = policies[0]
        ckpt = torch.load(Path(args.output_dir)/"rl_best.pt", map_location=DEVICE, weights_only=False)
        visualize_final_config(env, ckpt["active_mask"], args)
        save_rl_gif(env, policy, args, n_frames=args.gif_frames)
    if args.evaluate:
        if policy is None:
            policy = ActorCritic(env.obs_dim, env.K, conflict=env._conflict).to(DEVICE)
            cp = Path(args.output_dir)/"rl_best.pt"
            if cp.exists():
                policy.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=False)["policy_state"])
        pts, n_star = run_rl_method(env, policy, args)
        visualize_two_configs(env, n_star, policy, args)
    if args.gif and policy is None:
        policy = ActorCritic(env.obs_dim, env.K, conflict=env._conflict).to(DEVICE)
        cp = Path(args.output_dir)/"rl_best.pt"
        if cp.exists():
            policy.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=False)["policy_state"])
        save_rl_gif(env, policy, args, n_frames=args.gif_frames)

    if args.compare or args.compare_sweep or args.synthesis:
        if not policies:
            # Charge rl_best_seed*.pt (multi) sinon rl_best.pt
            cps = sorted(Path(args.output_dir).glob("rl_best_seed*.pt")) or \
                  [Path(args.output_dir) / "rl_best.pt"]
            for cp in cps:
                if cp.exists():
                    pol = ActorCritic(env.obs_dim, env.K, conflict=env._conflict).to(DEVICE)
                    pol.load_state_dict(torch.load(
                        cp, map_location=DEVICE,
                        weights_only=False)["policy_state"])
                    policies.append(pol)
            if policies:
                print(f"  {len(policies)} politique(s) chargée(s)")
        if args.compare:
            compare_baselines(args, env, rm, policies=policies)
        if args.compare_sweep:
            sweep_baselines(args, env, rm, policies=policies)
        if args.synthesis:
            plot_synthesis(args, env, rm, policies)

    if rm is not None:
        rm.report()
    print("\n  Brique 3 terminee.")
