"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BRIQUE 2b — GNN OBS-ONLY : PRÉDICTION DE NŒUD MASQUÉ (option A)             ║
║                                                                              ║
║  Pas de grille, pas de champ, pas de vérité. Le graphe des capteurs est      ║
║  construit à partir des seules séries observées ; on masque un nœud et on    ║
║  prédit sa mesure depuis ses voisins.                                        ║
║                                                                              ║
║    skill_k faible  -> capteur irremplaçable                                  ║
║    skill_k élevé   -> capteur redondant                                      ║
║                                                                              ║
║  NON CIRCULAIRE, contrairement à compute_proxy_targets de 02_gnn.py : la     ║
║  cible est une quantité PRÉDICTIVE (la mesure réelle du nœud masqué), pas    ║
║  une fonction de la matrice de corrélation qui a servi à bâtir le graphe.    ║
║                                                                              ║
║  Deux différences par rapport à build_spatial_correlation :                  ║
║    · corrélation à LAG OPTIMAL, pas au lag 0 — l'information est advectée    ║
║    · arêtes DIRIGÉES amont -> aval, le lag devient un attribut d'arête et    ║
║      le message de j vers i lit la valeur de j à t - lag_ij                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
    python gnn_lobo.py --train --obs outputs/obs_synth.npz --epochs 150
    python gnn_lobo.py --lobo  --obs outputs/obs_synth.npz --ckpt outputs/gnn_lobo.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from naiade_compat import get_device, resolve_seed_fn
from obs_operator import ObsSet, split_sensors, KINDS, VARIABLES

DEVICE = get_device()
set_global_seed = resolve_seed_fn()


# ══════════════════════════════════════════════════════════════════════════════
#  SÉRIES ET CORRÉLATION À LAG OPTIMAL
# ══════════════════════════════════════════════════════════════════════════════

def sensor_series(obs: ObsSet):
    """(n_sensors, nt, 2) avec NaN sur les manquants — 100 % obs-only."""
    V = np.stack([obs.gridded_series(v) for v in VARIABLES], -1)
    return V.astype(np.float32)


def _standardize(V):
    """Centre-réduit chaque série sur ses points valides."""
    out = V.copy()
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(V, axis=1, keepdims=True)
            sd = np.nanstd(V, axis=1, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return (out - mu) / sd, mu[:, 0], sd[:, 0]


def lagged_correlation(V, max_lag=10, min_overlap=20):
    """Corrélation croisée à lag optimal entre toutes les paires de capteurs.

    Retourne (C, L) : C[i,j] = corrélation maximale en valeur absolue,
    L[i,j] = lag correspondant (>0 : j précède i, donc j informe i).
    Robuste aux NaN (paires complètes uniquement).
    """
    n, nt, nv = V.shape
    # combinaison T/S pondérée, comme build_spatial_correlation
    w = np.array([0.6, 0.4], np.float32)[:nv]
    x = np.nansum(V * w, axis=-1)
    valid = np.isfinite(V).any(-1)
    x = np.where(valid, x, np.nan)

    C = np.zeros((n, n), np.float32)
    L = np.zeros((n, n), np.int16)
    xm = np.where(np.isfinite(x), x, 0.0)
    vm = np.isfinite(x).astype(np.float32)

    for lag in range(-max_lag, max_lag + 1):
        a = xm if lag >= 0 else np.roll(xm, -lag, axis=1)
        b = np.roll(xm, lag, axis=1) if lag >= 0 else xm
        va = vm if lag >= 0 else np.roll(vm, -lag, axis=1)
        vb = np.roll(vm, lag, axis=1) if lag >= 0 else vm
        m = va * vb
        cnt = m @ m.T * 0 + (m @ m.T)          # (n,n) chevauchement
        sab = (a * m) @ (b * m).T
        sa = (a * m) @ m.T
        sb = m @ (b * m).T
        saa = (a * a * m) @ m.T
        sbb = m @ (b * b * m).T
        with np.errstate(invalid="ignore", divide="ignore"):
            num = sab - sa * sb / np.maximum(cnt, 1)
            den = np.sqrt(np.maximum(saa - sa ** 2 / np.maximum(cnt, 1), 0)
                          * np.maximum(sbb - sb ** 2 / np.maximum(cnt, 1), 0))
            c = np.where((den > 1e-9) & (cnt >= min_overlap), num / den, 0.0)
        c = np.nan_to_num(c, 0.0)
        upd = np.abs(c) > np.abs(C)
        C = np.where(upd, c, C)
        L = np.where(upd, lag, L).astype(np.int16)
    np.fill_diagonal(C, 0.0)
    np.fill_diagonal(L, 0)
    return C.astype(np.float32), L


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURES NODALES — aucune ne lit le champ complet
# ══════════════════════════════════════════════════════════════════════════════

def node_features(obs: ObsSet, V, C):
    """Features (n, F) calculables sur observations seules :
      [x_norm, y_norm, one-hot type (6), has_T, has_S, return_rate,
       log_period, var_own, var_hf, var_lf, decorr_time, corr_max, noise]

    Remplace var_T/var_S/grad_mean de 02_gnn.py, qui lisaient des voisinages
    5x5 du nature run — fuite de vérité.
    """
    n, nt, _ = V.shape
    feats = []
    for k, s in enumerate(obs.sensors):
        x, y = s.mean_pos
        oh = np.zeros(len(KINDS), np.float32)
        oh[KINDS.index(s.kind)] = 1.0

        ser = V[k, :, 0]
        ok = np.isfinite(ser)
        ret = ok.mean()
        z = ser[ok]
        if len(z) > 8:
            z = (z - z.mean()) / (z.std() + 1e-9)
            hf = np.diff(z)
            var_hf = float(hf.var())
            var_lf = float(np.convolve(z, np.ones(5) / 5, "valid").var())
            r1 = float(np.corrcoef(z[:-1], z[1:])[0, 1]) if len(z) > 2 else 0.0
            r1 = float(np.clip(np.nan_to_num(r1), -0.999, 0.999))
            tau = -1.0 / np.log(max(abs(r1), 1e-3))
        else:
            var_hf = var_lf = 0.0; r1 = 0.0; tau = 0.0

        var_own = float(np.nanvar(V[k, :, 0])) if ok.any() else 0.0
        period = np.median(np.diff(s.times)) if len(s.times) > 1 else 1.0

        feats.append(np.concatenate([
            [x / obs.nx, y / obs.ny], oh,
            [float("T" in s.variables), float("S" in s.variables),
             ret, np.log1p(period) / 3.0,
             var_own, var_hf, var_lf, np.tanh(tau / 10.0),
             float(np.abs(C[k]).max()), float(s.noise[0])]]).astype(np.float32))
    X = np.stack(feats)
    # standardisation colonne (hors one-hot et positions)
    sl = slice(2 + len(KINDS), None)
    m, sd = X[:, sl].mean(0), X[:, sl].std(0) + 1e-6
    X[:, sl] = (X[:, sl] - m) / sd
    return X


def build_edges(obs: ObsSet, C, L, corr_threshold=0.35, k_nearest=5):
    """Arêtes dirigées j -> i : seuil de corrélation ∪ kNN géographique.
    Attributs : [|corr|, signe, lag normalisé, distance normalisée]."""
    n = len(obs.sensors)
    pos = np.array(obs.positions(), np.float32)
    d = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1))
    diag = np.hypot(obs.nx, obs.ny)
    src, dst, attr, lags = [], [], [], []
    seen = set()

    def add(j, i):
        if j == i or (j, i) in seen:
            return
        seen.add((j, i))
        lag = int(L[i, j])
        # orientation : la source est celle qui PRÉCÈDE
        if lag < 0:
            return
        src.append(j); dst.append(i); lags.append(lag)
        attr.append([abs(C[i, j]), np.sign(C[i, j]),
                     lag / 10.0, d[i, j] / diag])

    for i in range(n):
        for j in range(n):
            if i != j and abs(C[i, j]) > corr_threshold:
                add(j, i)
        for j in np.argsort(d[i])[1:k_nearest + 1]:
            add(int(j), i)
            add(i, int(j))                     # kNN symétrique si lag nul
    if not src:                                # garde-fou : graphe complet kNN
        for i in range(n):
            for j in np.argsort(d[i])[1:3]:
                src.append(int(j)); dst.append(i); lags.append(0)
                attr.append([0.0, 0.0, 0.0, d[i, int(j)] / diag])
    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(attr, dtype=torch.float),
            torch.tensor(lags, dtype=torch.long))


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLE — message passing avec attribut d'arête et décalage temporel
# ══════════════════════════════════════════════════════════════════════════════

class EdgeAttnLayer(nn.Module):
    """Attention sur arêtes, conditionnée par les attributs (corr, lag, dist)."""

    def __init__(self, dim, edge_dim=4):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.att = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1))
        self.upd = nn.GRUCell(dim, dim)

    def forward(self, h, edge_index, edge_attr, return_attn=False):
        s, d = edge_index[0], edge_index[1]
        e = torch.cat([h[s], h[d], edge_attr], -1)
        a = self.att(e).squeeze(-1)
        a = a - a.max()
        ex = a.exp()
        den = torch.zeros(h.size(0), device=h.device).scatter_add_(0, d, ex)
        alpha = ex / (den[d] + 1e-9)
        m = self.msg(e) * alpha.unsqueeze(-1)
        agg = torch.zeros_like(h).scatter_add_(
            0, d.unsqueeze(-1).expand_as(m), m)
        out = self.upd(agg, h)
        return (out, alpha) if return_attn else (out, None)


class MaskedNodePredictor(nn.Module):
    """Prédit (mu, logvar) de chaque nœud à partir de ses voisins.

    L'entrée d'un nœud combine ses features statiques et sa valeur courante ;
    un nœud MASQUÉ reçoit une valeur nulle et un drapeau `observed = 0`, donc
    sa prédiction ne peut venir que du graphe.
    """

    def __init__(self, in_dim, hidden=64, n_layers=2, edge_dim=4, n_var=2,
                 feat_dropout=0.3):
        super().__init__()
        # Dropout SUR LES FEATURES nodales : sans lui, le modèle identifie
        # chaque nœud par sa position et sa variance propre, et mémorise sa
        # série au lieu d'apprendre à la déduire du voisinage.
        self.feat_drop = nn.Dropout(feat_dropout)
        self.enc = nn.Sequential(
            nn.Linear(in_dim + 2 * n_var, hidden), nn.GELU(),
            nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(
            [EdgeAttnLayer(hidden, edge_dim) for _ in range(n_layers)])
        self.head_mu = nn.Linear(hidden, n_var)
        self.head_lv = nn.Linear(hidden, n_var)
        self.n_var = n_var

    def forward(self, feats, vals, observed, edge_index, edge_attr,
                return_attn=False):
        """feats (n,F) · vals (B,n,V) · observed (B,n,V)"""
        B = vals.shape[0]
        f = feats.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([self.feat_drop(f), vals * observed, observed], -1)
        h = self.enc(x)
        attn = None
        for lay in self.layers:
            hs = []
            for b in range(B):
                hb, a = lay(h[b], edge_index, edge_attr, return_attn)
                hs.append(hb)
                if return_attn:
                    attn = a
            h = torch.stack(hs)
        return self.head_mu(h), torch.clamp(self.head_lv(h), -7.0, 3.0), attn


# ══════════════════════════════════════════════════════════════════════════════
#  BATCHS — valeurs décalées par le lag d'arête
# ══════════════════════════════════════════════════════════════════════════════

class GraphBatcher:
    """Fabrique les tenseurs (vals, observed, target) pour un lot d'instants.

    Le décalage temporel des messages est appliqué en amont : la valeur portée
    par le nœud source j pour l'arête (j -> i) de lag L est V[j, t - L].
    Comme un nœud peut être source de plusieurs arêtes de lags différents, on
    approxime en donnant à chaque nœud sa valeur au lag MÉDIAN de ses arêtes
    sortantes — suffisant pour l'option A, exact dans l'option B.
    """

    def __init__(self, V, edge_index, lags, sids_input, sids_hold):
        self.V = V                                    # (n, nt, 2) standardisé
        self.n, self.nt, self.nv = V.shape
        self.obs_ok = np.isfinite(V)
        self.node_lag = np.zeros(self.n, np.int32)
        # torch.load(map_location=DEVICE) déplace tous les tenseurs du
        # checkpoint, y compris edge_index/lags qui restent des index CPU.
        src = edge_index[0].detach().cpu().numpy()
        lg = lags.detach().cpu().numpy()
        for j in range(self.n):
            m = src == j
            if m.any():
                self.node_lag[j] = int(np.median(lg[m]))
        self.in_set = np.zeros(self.n, bool); self.in_set[list(sids_input)] = True
        self.hold_set = np.zeros(self.n, bool); self.hold_set[list(sids_hold)] = True

    def usable_times(self, min_obs=4):
        avail = self.obs_ok.any(-1) & self.in_set[:, None]
        return [t for t in range(int(self.node_lag.max()), self.nt)
                if avail[:, t].sum() >= min_obs]

    def make(self, times, mask_frac=0.25, rng=None, force_mask=None):
        """Retourne vals, observed, target, tmask (B, n, V)."""
        rng = rng or np.random.default_rng(0)
        B = len(times)
        vals = np.zeros((B, self.n, self.nv), np.float32)
        obsd = np.zeros((B, self.n, self.nv), np.float32)
        targ = np.zeros((B, self.n, self.nv), np.float32)
        tmsk = np.zeros((B, self.n, self.nv), np.float32)

        for b, t in enumerate(times):
            for k in range(self.n):
                tk = t - int(self.node_lag[k])
                if tk < 0:
                    continue
                for v in range(self.nv):
                    if not self.obs_ok[k, tk, v]:
                        continue
                    vals[b, k, v] = self.V[k, tk, v]
                    obsd[b, k, v] = 1.0
            # cible : nœuds tenus à l'écart + une fraction tirée de l'entrée
            live = np.where(obsd[b].any(-1) > 0)[0]
            cand = np.array([k for k in live if self.in_set[k]])
            drop = set()
            if len(cand) > 3 and mask_frac > 0:
                nd = max(1, int(round(mask_frac * len(cand))))
                drop = set(rng.choice(cand, min(nd, len(cand) - 3),
                                      replace=False).tolist())
            if force_mask is not None:
                drop = set(force_mask)
            drop |= set(k for k in live if self.hold_set[k])
            for k in drop:
                targ[b, k] = vals[b, k]
                tmsk[b, k] = obsd[b, k]
                vals[b, k] = 0.0
                obsd[b, k] = 0.0
        return (torch.from_numpy(vals), torch.from_numpy(obsd),
                torch.from_numpy(targ), torch.from_numpy(tmsk))


def gaussian_nll(mu, logvar, y, m):
    n = m.sum()
    if n < 1:
        return mu.sum() * 0.0
    t = 0.5 * (logvar + (y - mu) ** 2 / logvar.exp().clamp_min(1e-8)
               + float(np.log(2 * np.pi)))
    return (t * m).sum() / n


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT + LOBO
# ══════════════════════════════════════════════════════════════════════════════

def select_nodes(obs, V, kinds=None, min_presence=0.10, min_pairs=0.25,
                 verbose=True):
    """Retient les capteurs assez PERSISTANTS pour porter un nœud de graphe.

    Un dériveur qui vit 250 jours sur un run de 3653 ne chevauche presque
    aucun autre capteur : la corrélation par paire n'est pas estimable, le
    nœud reste isolé, et il pollue le graphe sans jamais recevoir de message.
    Le problème n'est pas le bruit, c'est l'absence de recouvrement temporel.

    Les plateformes écartées ici restent utilisées par l'AE, qui travaille
    pas de temps par pas de temps et n'a pas besoin de recouvrement.
    """
    ok = np.isfinite(V).any(-1)
    # ÉTENDUE temporelle (premier -> dernier échantillon), pas le nombre
    # d'observations : un Argo au cycle de 10 jours plafonne à 10 % de pas
    # de temps observés même s'il vit tout le run. Un critère fondé sur le
    # comptage l'exclut mécaniquement, alors qu'il est parfaitement
    # persistant. Ce qui compte pour le graphe, c'est le RECOUVREMENT.
    span = np.zeros(len(ok))
    n_obs = ok.sum(1)
    for i in range(len(ok)):
        w = np.flatnonzero(ok[i])
        if len(w):
            span[i] = (w[-1] - w[0] + 1) / ok.shape[1]
    keep = (span >= min_presence) & (n_obs >= 20)
    if kinds:
        keep &= np.array([s.kind in kinds for s in obs.sensors])

    idx = np.flatnonzero(keep)
    if verbose:
        okf = ok.astype(np.float32)
        ovl = okf @ okf.T
        np.fill_diagonal(ovl, 0)
        sub = ovl[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1) if len(idx) > 1 else ([], [])
        frac = float((sub[iu] >= 20).mean()) if len(idx) > 1 else 0.0
        print(f"      nœuds retenus : {len(idx)}/{obs.n_sensors} "
              f"(étendue >= {100 * min_presence:.0f} % du run, >= 20 obs)")
        by = {}
        for i in idx:
            by[obs.sensors[i].kind] = by.get(obs.sensors[i].kind, 0) + 1
        print("      " + ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
        print(f"      paires avec >= 20 pas communs : {100 * frac:.0f} %")
        if frac < min_pairs:
            print("      [!] recouvrement temporel insuffisant : la matrice de")
            print("          corrélation sera creuse et le graphe fragmenté.")
            print("          Restreignez --kinds aux plateformes persistantes.")
    return idx


class _SubObs:
    """Vue d'un ObsSet restreinte à un sous-ensemble de capteurs."""

    def __init__(self, obs, idx):
        self.idx = np.asarray(idx)
        self.sensors = [obs.sensors[i] for i in self.idx]
        self.nx, self.ny, self.nt = obs.nx, obs.ny, obs.nt
        self.ocean = getattr(obs, "ocean", None)
        self.n_sensors = len(self.sensors)
        self._orig = obs
        # sid d'origine -> indice local. split_sensors() lit Sensor.sid, qui
        # reste l'identifiant GLOBAL : sans remappage, les indices renvoyés
        # débordent la taille du sous-graphe.
        self.local = {int(s.sid): k for k, s in enumerate(self.sensors)}

    def to_local(self, sids):
        return np.array([self.local[int(s)] for s in sids
                         if int(s) in self.local], np.int32)

    def positions(self):
        return [s.mean_pos for s in self.sensors]

    def orig_sid(self, k):
        return int(self.idx[k])


def build_all(obs, args):
    V_full = sensor_series(obs)
    kinds = set(args.kinds.split(",")) if args.kinds else None
    idx = select_nodes(obs, V_full, kinds=kinds,
                       min_presence=args.min_presence)
    if len(idx) < 6:
        raise RuntimeError(
            f"seulement {len(idx)} nœuds retenus — baissez --min_presence ou "
            "élargissez --kinds")
    sub = _SubObs(obs, idx)
    V = V_full[idx]
    Vz, mu, sd = _standardize(V)
    C, L = lagged_correlation(Vz, max_lag=args.max_lag)
    X = node_features(sub, Vz, C)
    ei, ea, lags = build_edges(sub, C, L, args.corr_threshold, args.k_nearest)
    return Vz, C, L, X, ei, ea, lags, sub


def train_gnn_lobo(args):
    print("=" * 70)
    print("  Brique 2b — GNN obs-only, prédiction de nœud masqué")
    print("=" * 70)
    set_global_seed(args.seed)
    obs = ObsSet.load(args.obs)
    print("\n[1/4] " + obs.summary().replace("\n", "\n      "))

    Vz, C, L, X, ei, ea, lags, sub = build_all(obs, args)
    n_dir = int((lags.cpu().numpy() > 0).sum())
    print(f"\n[2/4] Graphe : {sub.n_sensors} nœuds, {ei.shape[1]} arêtes "
          f"dont {n_dir} à lag non nul")
    print(f"      |corr| médiane = {np.median(np.abs(C[C != 0])):.3f} "
          f"| lag médian (arêtes dirigées) = "
          f"{int(np.median(lags.cpu().numpy()[lags.cpu().numpy() > 0])) if n_dir else 0} j")

    sp = split_sensors(sub, frac_fit=args.frac_fit,
                       frac_val=args.frac_val, seed=args.seed)
    sp = {k: sub.to_local(v) for k, v in sp.items()}
    print(f"      split capteurs : fit={len(sp['fit'])} val={len(sp['val'])} "
          f"test={len(sp['test'])}")

    bt_tr = GraphBatcher(Vz, ei, lags, sp["fit"], [])
    bt_va = GraphBatcher(Vz, ei, lags, sp["fit"], sp["val"])
    t_tr = bt_tr.usable_times(); t_va = bt_va.usable_times()
    print(f"      instants exploitables : {len(t_tr)}")

    model = MaskedNodePredictor(X.shape[1], hidden=args.hidden,
                                n_layers=args.n_layers,
                                feat_dropout=args.feat_dropout).to(DEVICE)
    feats = torch.from_numpy(X).to(DEVICE)
    ei_d, ea_d = ei.to(DEVICE), ea.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(args.seed)

    print(f"\n[3/4] Entraînement {args.epochs} époques "
          f"({sum(p.numel() for p in model.parameters()):,} paramètres)")
    best, hist = np.inf, []
    best_ep, since = 0, 0
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0; nb = 0
        perm = rng.permutation(t_tr)
        for i in range(0, len(perm), args.batch_size):
            tt = perm[i:i + args.batch_size].tolist()
            v, o, y, m = bt_tr.make(tt, args.mask_frac, rng)
            v, o, y, m = v.to(DEVICE), o.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            mu, lv, _ = model(feats, v, o, ei_d, ea_d)
            loss = gaussian_nll(mu, lv, y, m)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss.detach()); nb += 1
        sch.step()

        model.eval()
        with torch.no_grad():
            v, o, y, m = bt_va.make(t_va, 0.0, rng)
            v, o, y, m = v.to(DEVICE), o.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            mu, lv, _ = model(feats, v, o, ei_d, ea_d)
            sel = m > 0.5
            if sel.sum() > 0:
                e = (y - mu)[sel]
                z = (e / lv[sel].exp().sqrt().clamp_min(1e-6))
                rmse = float((e ** 2).mean().sqrt())
                # skill = 1 - MSE / var(obs) : séries standardisées -> var=1
                skill = float(1.0 - (e ** 2).mean())
                zstd = float(z.std())
                vnll = float(gaussian_nll(mu, lv, y, m))
            else:
                rmse = skill = zstd = vnll = float("nan")
        hist.append(dict(epoch=ep, loss=tot / max(1, nb), val_nll=vnll,
                         rmse=rmse, skill=skill, z_std=zstd))
        if ep % max(1, args.epochs // 10) == 0 or ep == 1:
            print(f"  ep {ep:3d}/{args.epochs} | L={tot/max(1,nb):7.3f} "
                  f"| NLL_val={vnll:7.3f} | RMSE={rmse:.3f} "
                  f"| skill={skill:+.3f} | z_std={zstd:.2f}")
        if vnll == vnll and vnll < best - 1e-4:
            best, best_ep, since = vnll, ep, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "X": X, "edge_index": ei, "edge_attr": ea,
                        "lags": lags, "split": {k: v.tolist()
                                                for k, v in sp.items()}},
                       out_dir / "gnn_lobo.pt")
        else:
            since += 1
            if args.patience and since >= args.patience:
                print(f"  arrêt anticipé (époque {ep}, "
                      f"pas d'amélioration depuis {since})")
                break

    print(f"\n[4/4] Meilleure NLL val : {best:.4f} (époque {best_ep})")
    if best_ep <= 3:
        print("      [!] le meilleur modèle est quasi non entraîné : la NLL de")
        print("          validation se dégrade dès les premières époques. Le")
        print("          GNN mémorise les nœuds d'entraînement au lieu")
        print("          d'apprendre une règle transférable. Augmentez")
        print("          --feat_dropout, réduisez --hidden.")
    print(f"      Checkpoint → {out_dir}/gnn_lobo.pt")
    (out_dir / "gnn_lobo_history.json").write_text(json.dumps(hist, indent=1))
    return model, obs


@torch.no_grad()
def lobo_gnn(model, obs, Vz, X, ei, ea, lags, sids=None, n_t=300, seed=0):
    """Deux scores complémentaires, tous deux obs-only :

      skill_k   : qualité de reconstruction du capteur k depuis ses voisins.
                  ÉLEVÉ = redondant. C'est le score « puis-je l'arrêter ? »
      delta_k   : dégradation de la prédiction des AUTRES nœuds quand on
                  retire k du graphe d'entrée. ÉLEVÉ = irremplaçable.

    Les deux ne disent pas la même chose : un capteur isolé est à la fois mal
    prédit (skill bas) et peu utile aux autres (delta bas) — c'est justement
    la signature d'une lacune du réseau, pas d'un capteur précieux.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    feats = torch.from_numpy(X).to(DEVICE)
    ei_d, ea_d = ei.to(DEVICE), ea.to(DEVICE)
    n = len(obs.sensors)
    sids = list(range(n)) if sids is None else list(sids)

    bt = GraphBatcher(Vz, ei, lags, range(n), [])
    times = bt.usable_times()
    if len(times) > n_t:
        times = sorted(rng.choice(times, n_t, replace=False).tolist())

    # ── skill : chaque nœud masqué seul, à tour de rôle ────────────────────
    skill = np.full(n, np.nan); sse = np.zeros(n); cnt = np.zeros(n)
    for k in sids:
        v, o, y, m = bt.make(times, 0.0, rng, force_mask=[k])
        v, o, y, m = v.to(DEVICE), o.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
        mu, lv, _ = model(feats, v, o, ei_d, ea_d)
        sel = m[:, k] > 0.5
        if sel.sum() == 0:
            continue
        e = (y[:, k] - mu[:, k])[sel]
        sse[k] = float((e ** 2).sum()); cnt[k] = float(sel.sum())
        skill[k] = 1.0 - sse[k] / max(cnt[k], 1)      # séries standardisées

    # ── delta : retrait de k, impact sur les autres nœuds masqués ──────────
    v0, o0, y0, m0 = bt.make(times, 0.25, rng)
    v0, o0, y0, m0 = (v0.to(DEVICE), o0.to(DEVICE),
                      y0.to(DEVICE), m0.to(DEVICE))
    mu0, lv0, _ = model(feats, v0, o0, ei_d, ea_d)
    base = float(gaussian_nll(mu0, lv0, y0, m0))
    delta = np.full(n, np.nan)
    for k in sids:
        v1, o1 = v0.clone(), o0.clone()
        if float(o1[:, k].sum()) == 0:
            continue
        v1[:, k] = 0.0; o1[:, k] = 0.0
        mu1, lv1, _ = model(feats, v1, o1, ei_d, ea_d)
        delta[k] = float(gaussian_nll(mu1, lv1, y0, m0)) - base
    return skill, delta, base


def run_lobo(args):
    obs = ObsSet.load(args.obs)
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    Vz, C, L, X, ei, ea, lags, sub = build_all(obs, args)
    if args.rebuild_graph:
        # Graphe reconstruit sur CE jeu d'observations : indispensable pour
        # appliquer un modèle entraîné sur un split à un autre split, où les
        # capteurs et leurs corrélations diffèrent. Le GNN est invariant au
        # nombre de nœuds, seule la dimension des features doit correspondre.
        if X.shape[1] != ck["X"].shape[1]:
            raise ValueError(
                f"features nodales incompatibles : {X.shape[1]} ici contre "
                f"{ck['X'].shape[1]} à l'entraînement")
        print(f"  graphe reconstruit : {X.shape[0]} nœuds, "
              f"{ei.shape[1]} arêtes")
    else:
        X = ck["X"]
        ei = ck["edge_index"].cpu()
        ea = ck["edge_attr"].cpu()
        lags = ck["lags"].cpu()
    model = MaskedNodePredictor(
        X.shape[1], hidden=ck["args"]["hidden"],
        n_layers=ck["args"]["n_layers"],
        feat_dropout=ck["args"].get("feat_dropout", 0.3)).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    skill, delta, base = lobo_gnn(model, sub, Vz, X, ei, ea, lags,
                                  n_t=args.lobo_t, seed=args.seed)

    print(f"\nNLL réseau complet : {base:.4f}")
    print("\n  sid  type        skill   delta_NLL   lecture")
    order = np.argsort(-np.nan_to_num(skill, nan=-9))
    for k in order:
        s, d = skill[k], delta[k]
        if s != s:
            continue
        if s > 0.5 and (d != d or d < 0.02):
            tag = "REDONDANT — candidat au retrait"
        elif s < 0.1 and (d != d or d < 0.02):
            tag = "ISOLÉ — lacune du réseau autour"
        elif d == d and d > 0.05:
            tag = "PIVOT — irremplaçable pour les voisins"
        else:
            tag = "utile"
        print(f"  {k:4d} {obs.sensors[k].kind:10s} {s:+6.3f}  "
              f"{d if d == d else float('nan'):+9.4f}   {tag}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "lobo_gnn.json").write_text(json.dumps({
        "base_nll": base,
        "skill": {str(sub.orig_sid(i)): (None if s != s else float(s))
                  for i, s in enumerate(skill)},
        "delta": {str(sub.orig_sid(i)): (None if d != d else float(d))
                  for i, d in enumerate(delta)}}, indent=1))
    print(f"\n  → {args.output_dir}/lobo_gnn.json")


def parse_args():
    p = argparse.ArgumentParser("GNN obs-only — nœud masqué")
    p.add_argument("--train", action="store_true")
    p.add_argument("--lobo", action="store_true")
    p.add_argument("--obs", default="outputs/obs_synth.npz")
    p.add_argument("--ckpt", default="outputs/gnn_lobo.pt")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--feat_dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--mask_frac", type=float, default=0.25)
    p.add_argument("--kinds", default="mooring,argo,ship",
                   help="types portant un nœud. Les dériveurs sont exclus "
                        "par défaut : trop courts pour se recouvrir")
    p.add_argument("--min_presence", type=float, default=0.10,
                   help="fraction minimale du run pendant laquelle un capteur "
                        "doit être actif pour devenir un nœud")
    p.add_argument("--max_lag", type=int, default=10)
    p.add_argument("--corr_threshold", type=float, default=0.35)
    p.add_argument("--k_nearest", type=int, default=5)
    p.add_argument("--frac_fit", type=float, default=0.70)
    p.add_argument("--frac_val", type=float, default=0.15)
    p.add_argument("--lobo_t", type=int, default=300)
    p.add_argument("--rebuild_graph", action="store_true",
                   help="reconstruire le graphe sur ce jeu d'obs "
                        "au lieu de reprendre celui du checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.train:
        train_gnn_lobo(a)
    if a.lobo:
        run_lobo(a)
    if not (a.train or a.lobo):
        print("rien à faire : --train et/ou --lobo")
