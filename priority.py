"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  COUPLAGE BRIQUE 2 -> BRIQUE 3                                               ║
║  Le GNN fournit la PERTINENCE des capteurs au planificateur de campagnes     ║
║                                                                              ║
║  Répartition des rôles, volontairement étanche :                             ║
║                                                                              ║
║     GNN  ->  « quelle information cette bouée apporte-t-elle, compte tenu    ║
║               de toutes les autres ? »           (pertinence, sans notion    ║
║                                                   de coût)                   ║
║     Plan ->  « combien coûte-t-il d'aller la voir ? »   (détour, autonomie,  ║
║                                                          jours de mer)       ║
║                                                                              ║
║  Le planificateur arbitre déjà sur le rapport pertinence / économie de       ║
║  détour. Il lui manquait seulement une pertinence digne de ce nom. Garder    ║
║  le coût HORS du GNN rend la décision explicable à un opérateur : on peut    ║
║  toujours dire si une bouée a été déclassée parce qu'elle informe peu, ou    ║
║  parce qu'elle coûte cher à desservir.                                       ║
║                                                                              ║
║  Ce que le couplage change réellement                                        ║
║  ------------------------------------                                        ║
║  Le planificateur disposait de deux sources de pertinence, toutes deux       ║
║  insatisfaisantes :                                                          ║
║    - la variabilité locale au point considéré : ponctuelle, aveugle à la     ║
║      redondance entre bouées (deux bouées voisines très variables reçoivent  ║
║      le même score élevé alors qu'elles se doublonnent) ;                    ║
║    - la contribution marginale exacte (leave-one-out) : juste, mais coûte    ║
║      n résolutions du critère à chaque replanification, donc inutilisable    ║
║      dans une boucle.                                                        ║
║                                                                              ║
║  Le GNN apprend hors ligne à prédire la seconde et se substitue à la         ║
║  première, en une passe avant. Pas de circularité : l'entraînement est       ║
║  découplé de la planification.                                               ║
║                                                                              ║
║  NB : la brique 2 s'entraîne aujourd'hui sur une cible proxy                 ║
║  (1 - moyenne des corrélations), c'est-à-dire sur une formule d'une ligne.   ║
║  Ici la cible est la vraie contribution marginale mesurée par le critère de  ║
║  la brique 3. C'est ce qui rend l'apprentissage utile plutôt que décoratif.  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import NX, NY, DEVICE


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPHE
# ══════════════════════════════════════════════════════════════════════════════

def _load_brick2():
    """Charge 02_gnn.py (nom de module non importable directement)."""
    path = Path(__file__).with_name("02_gnn.py")
    spec = importlib.util.spec_from_file_location("brick2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidate_correlation(env, n_timestamps=None):
    """
    Corrélation entre toutes les positions candidates, calculée UNE fois.

    Les graphes des réseaux échantillonnés en sont ensuite de simples
    sous-matrices : reconstruire la corrélation à chaque tirage coûterait
    plusieurs secondes par échantillon et rendrait l'entraînement absurde.
    """
    if getattr(env, "_cand_corr", None) is not None:
        return env._cand_corr
    from data.dataset import mesoscale_anomaly
    Ta = mesoscale_anomaly(env.T)
    Sa = mesoscale_anomaly(env.S)
    if n_timestamps and n_timestamps < len(Ta):
        sel = np.linspace(0, len(Ta) - 1, n_timestamps).astype(int)
        Ta, Sa = Ta[sel], Sa[sel]
    oT = np.stack([Ta[:, x, y] for x, y in env.candidate_positions], axis=1)
    oS = np.stack([Sa[:, x, y] for x, y in env.candidate_positions], axis=1)
    K = env.K
    R = np.corrcoef(np.concatenate([oT, oS], axis=1), rowvar=False)
    R = 0.5 * (R[:K, :K] + R[K:, K:])
    env._cand_corr = np.nan_to_num(R)
    return env._cand_corr


def build_graph_fast(env, idx, corr_full, corr_threshold=0.3, k_nearest=4):
    """
    Graphe d'un réseau : noeuds = bouées, arêtes = forte corrélation OU
    proximité géographique (les k plus proches garantissent la connexité).

    Features nodales, identiques à celles de la brique 2 pour rester
    interchangeable : position normalisée, corrélation maximale avec un
    voisin (proxy de redondance), degré normalisé, variabilité locale.
    """
    idx = np.asarray(idx, dtype=int)
    n = len(idx)
    pos = np.array([env.candidate_positions[i] for i in idx], dtype=np.float32)
    R = corr_full[np.ix_(idx, idx)]

    src, dst, attr = [], [], []
    seen = set()
    iu = np.argwhere(np.triu(np.abs(R) > corr_threshold, k=1))
    for i, j in iu:
        src += [int(i), int(j)]; dst += [int(j), int(i)]
        attr += [R[i, j], R[i, j]]
        seen.add((int(i), int(j))); seen.add((int(j), int(i)))
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    for i in range(n):
        for j in np.argsort(d[i])[:min(k_nearest, n - 1)]:
            if (i, int(j)) not in seen:
                src += [i, int(j)]; dst += [int(j), i]
                attr += [max(R[i, j], 0.1)] * 2
                seen.add((i, int(j))); seen.add((int(j), i))

    deg = np.bincount(src, minlength=n).astype(np.float32).reshape(-1, 1)
    Rz = R.copy(); np.fill_diagonal(Rz, 0.0)
    stats = env.field_stats[idx] if len(env.field_stats) >= env.K else None
    var = (stats.reshape(-1, 1) if stats is not None and len(stats) == n
           else np.zeros((n, 1), np.float32))
    x = np.concatenate([
        pos[:, 0:1] / NX, pos[:, 1:2] / NY,
        np.abs(Rz).max(axis=1, keepdims=True),
        deg / (deg.max() + 1e-9),
        var, var], axis=1).astype(np.float32)
    return (torch.tensor(x),
            torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(attr, dtype=torch.float).unsqueeze(-1))


# ══════════════════════════════════════════════════════════════════════════════
#  CIBLE : contribution marginale réelle
# ══════════════════════════════════════════════════════════════════════════════

def loo_contribution(env, idx):
    """
    Contribution marginale exacte de chaque bouée : chute de variance
    expliquée quand on la retire. C'est la quantité que le planificateur
    voudrait connaître, et qu'il ne peut pas se payer en boucle.
    """
    idx = np.asarray(idx, dtype=int)
    full = env.explained_variance(idx)
    out = np.empty(len(idx))
    for k in range(len(idx)):
        out[k] = full - env.explained_variance(np.delete(idx, k))
    return np.clip(out, 0.0, None)


def _standardise(y):
    """Centrage-réduction PAR GRAPHE : seul le classement interne compte,
    pas le niveau absolu qui dépend de la taille du réseau."""
    y = np.asarray(y, dtype=np.float64)
    return (y - y.mean()) / (y.std() + 1e-9)


def build_dataset(env, n_graphs=400, n_range=(10, 26), seed=0, verbose=True):
    """Tire des réseaux faisables et calcule leur contribution marginale."""
    rng = np.random.default_rng(seed)
    corr = candidate_correlation(env)
    data = []
    for g in range(n_graphs):
        n = int(rng.integers(*n_range))
        idx = env.sample_feasible(n, rng=rng)
        if len(idx) < 4:
            continue
        y = _standardise(loo_contribution(env, idx))
        x, ei, ea = build_graph_fast(env, idx, corr)
        data.append((x, ei, ea, torch.tensor(y, dtype=torch.float), idx))
        if verbose and (g + 1) % 50 == 0:
            print(f"    {g + 1}/{n_graphs} graphes", flush=True)
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  MODELE
# ══════════════════════════════════════════════════════════════════════════════

class PriorityGNN(nn.Module):
    """
    GraphSAGE inductif : agrège le voisinage, donc un même modèle traite des
    réseaux de tailles différentes et des bouées jamais vues. C'est
    indispensable ici, puisque le planificateur l'interroge sur des
    configurations qui changent à chaque pas.
    """

    def __init__(self, in_dim=6, hidden=64, layers=3):
        super().__init__()
        self.lin_self, self.lin_neigh = nn.ModuleList(), nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.lin_self.append(nn.Linear(d, hidden))
            self.lin_neigh.append(nn.Linear(d, hidden))
            d = hidden
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        h = x
        for ls, ln in zip(self.lin_self, self.lin_neigh):
            agg = torch.zeros(h.size(0), h.size(1), device=h.device,
                              dtype=h.dtype)
            agg.index_add_(0, dst, h[src])
            cnt = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
            cnt.index_add_(0, dst, torch.ones_like(src, dtype=h.dtype)
                           .unsqueeze(-1))
            h = torch.relu(ls(h) + ln(agg / cnt.clamp(min=1)))
        return self.head(h).squeeze(-1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def train(model, data, epochs=120, lr=2e-3, val_frac=0.2, verbose=True):
    """
    Perte : erreur quadratique sur la cible standardisée par graphe. Le
    critère suivi est en revanche le rho de Spearman, parce que le
    planificateur n'utilise que l'ORDRE des pertinences, pas leur valeur.
    """
    n_val = max(1, int(len(data) * val_frac))
    tr, va = data[:-n_val], data[-n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        np.random.shuffle(tr)
        tot = 0.0
        for x, ei, ea, y, _ in tr:
            opt.zero_grad()
            loss = ((model(x.to(DEVICE), ei.to(DEVICE)) - y.to(DEVICE)) ** 2).mean()
            loss.backward(); opt.step(); tot += loss.item()
        model.eval()
        with torch.no_grad():
            rho = np.mean([spearman(model(x.to(DEVICE), ei.to(DEVICE)).cpu().numpy(),
                                    y.numpy()) for x, ei, ea, y, _ in va])
        if rho > best:
            best = rho
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % 20 == 0:
            print(f"    epoch {ep+1:3d} | loss {tot/max(len(tr),1):.4f} "
                  f"| Spearman val {rho:+.3f} (meilleur {best:+.3f})", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFACE POUR LE PLANIFICATEUR
# ══════════════════════════════════════════════════════════════════════════════

class GNNPriority:
    """
    Objet appelable à passer au planificateur :  priority = gnn(env, idx).

    Retourne des poids strictement positifs, d'échelle arbitraire : le
    planificateur les utilise en rapport pertinence / économie de détour, donc
    seule leur échelle relative compte.
    """

    def __init__(self, model, env):
        self.model = model.eval()
        self.env = env
        self.corr = candidate_correlation(env)

    def __call__(self, env, idx):
        idx = np.asarray(idx, dtype=int)
        if len(idx) < 4:
            return np.ones(len(idx))
        x, ei, _ = build_graph_fast(env, idx, self.corr)
        with torch.no_grad():
            s = self.model(x.to(DEVICE), ei.to(DEVICE)).cpu().numpy()
        s = s - s.min()
        return s / (s.max() + 1e-9) + 0.05      # strictement positif

    def save(self, path):
        torch.save({"state": self.model.state_dict()}, path)

    @staticmethod
    def load(path, env, in_dim=6):
        m = PriorityGNN(in_dim=in_dim).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE,
                                     weights_only=False)["state"])
        return GNNPriority(m, env)
