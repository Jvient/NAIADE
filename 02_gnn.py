"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BRIQUE 2 — Graph Neural Network pour la Structure du Réseau         ║
║                                                                              ║
║  Pipeline :                                                                  ║
║    1. Construction du graphe (corrélation spatiale + kNN géographique)       ║
║    2. GAT : poids d'attention = proxy de redondance                          ║
║    3. GraphSAGE inductif : évalue des capteurs hypothétiques                 ║
║    4. Analyse : redondance, lacunes, ranking des capteurs                    ║
║                                                                              ║
║  Usage :                                                                     ║
║    python 02_gnn.py --train --analyze                                        ║
║    python 02_gnn.py --inductive --new_positions "[(10,20),(80,150)]"         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, argparse, ast
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent))
from config import *
try:
    from dataset import SyntheticOceanGenerator
except ModuleNotFoundError:
    from data.dataset import SyntheticOceanGenerator

# ── Import PyTorch Geometric ───────────────────────────────────────────────────
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[WARN] torch-geometric non disponible — implémentation manuelle utilisée.")


# ══════════════════════════════════════════════════════════════════════════════
#  IMPLÉMENTATION MANUELLE (fallback si torch_geometric absent)
# ══════════════════════════════════════════════════════════════════════════════

class ManualGATLayer(nn.Module):
    """Graph Attention Layer manuel (single-head)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, h, edge_index):
        Wh = self.W(h)
        src, dst = edge_index[0], edge_index[1]
        e = torch.cat([Wh[src], Wh[dst]], dim=-1)
        alpha = F.leaky_relu(self.a(e), 0.2).squeeze(-1)
        alpha_exp = torch.exp(alpha - alpha.max())
        alpha_sum = torch.zeros(h.size(0), device=h.device)
        alpha_sum.scatter_add_(0, dst, alpha_exp)
        alpha_norm = alpha_exp / (alpha_sum[dst] + 1e-9)
        out = torch.zeros_like(Wh)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(Wh[src]),
                         alpha_norm.unsqueeze(-1) * Wh[src])
        return F.elu(out), alpha_norm


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DU GRAPHE
# ══════════════════════════════════════════════════════════════════════════════

def build_spatial_correlation(T, S, positions, n_timestamps=200):
    """Matrice de corrélation de Pearson entre capteurs (séries T+S normalisées)."""
    n = len(positions)
    t_idx = np.random.choice(len(T), min(n_timestamps, len(T)), replace=False)
    series = np.zeros((n, len(t_idx)))
    for k, (x, y) in enumerate(positions):
        ts_T = (T[t_idx, x, y] - T[:, x, y].mean()) / (T[:, x, y].std() + 1e-9)
        ts_S = (S[t_idx, x, y] - S[:, x, y].mean()) / (S[:, x, y].std() + 1e-9)
        series[k] = 0.6 * ts_T + 0.4 * ts_S
    return np.corrcoef(series)


def build_graph(positions, corr_matrix, corr_threshold=0.5, k_nearest=4,
                T=None, S=None):
    """
    Construit le graphe du réseau : arêtes par seuil de corrélation + kNN.
    Features nodaux enrichis (8 dim) :
      [x_norm, y_norm, corr_max, degré_norm,
       var_T_locale, var_S_locale, grad_mean, dist_bord_norm]
    T, S optionnels : si fournis, calcule les features dynamiques.
    """
    from scipy.spatial import KDTree

    n = len(positions)
    pos_arr = np.array(positions, dtype=np.float32)
    edges = set()
    edge_weights = {}

    # Arêtes par seuil de corrélation
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > corr_threshold:
                edges.add((i, j)); edges.add((j, i))
                edge_weights[(i, j)] = corr_matrix[i, j]
                edge_weights[(j, i)] = corr_matrix[i, j]

    # kNN géographique
    tree = KDTree(pos_arr)
    for i in range(n):
        _, idxs = tree.query(pos_arr[i], k=min(k_nearest + 1, n))
        for j in idxs[1:]:
            if (i, j) not in edges:
                edges.add((i, j)); edges.add((j, i))
                edge_weights[(i, j)] = max(corr_matrix[i, j], 0.1)
                edge_weights[(j, i)] = max(corr_matrix[i, j], 0.1)

    src_list = [e[0] for e in edges]
    dst_list = [e[1] for e in edges]
    attr_list = [edge_weights.get(e, 0.1) for e in edges]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float).unsqueeze(-1)

    # ── Features nodaux enrichis ──────────────────────────────────────────────
    x_norm = pos_arr[:, 0:1] / NX
    y_norm = pos_arr[:, 1:2] / NY
    corr_max = np.array([corr_matrix[i].copy() for i in range(n)])
    np.fill_diagonal(corr_max, 0)
    corr_max_vals = corr_max.max(axis=1, keepdims=True)
    degree = np.bincount(src_list, minlength=n).reshape(-1, 1).astype(np.float32)
    degree_norm = degree / (degree.max() + 1e-9)

    # Features dynamiques (si T et S fournis)
    if T is not None and S is not None:
        var_T = np.zeros((n, 1), dtype=np.float32)
        var_S = np.zeros((n, 1), dtype=np.float32)
        grad_mean = np.zeros((n, 1), dtype=np.float32)
        for k, (px, py) in enumerate(positions):
            # Variance temporelle locale (fenêtre 5×5)
            x0, x1 = max(0, px-2), min(NX, px+3)
            y0, y1 = max(0, py-2), min(NY, py+3)
            var_T[k] = T[:, x0:x1, y0:y1].var()
            var_S[k] = S[:, x0:x1, y0:y1].var()
            # Gradient moyen du champ (norme) sur quelques instants
            grads = []
            for t_i in range(0, min(len(T), 50), 10):
                gx = np.gradient(T[t_i], axis=0)[x0:x1, y0:y1]
                gy = np.gradient(T[t_i], axis=1)[x0:x1, y0:y1]
                grads.append(np.sqrt(gx**2 + gy**2).mean())
            grad_mean[k] = np.mean(grads)
        # Normalisation
        var_T = (var_T - var_T.mean()) / (var_T.std() + 1e-9)
        var_S = (var_S - var_S.mean()) / (var_S.std() + 1e-9)
        grad_mean = (grad_mean - grad_mean.mean()) / (grad_mean.std() + 1e-9)
    else:
        var_T = np.zeros((n, 1), dtype=np.float32)
        var_S = np.zeros((n, 1), dtype=np.float32)
        grad_mean = np.zeros((n, 1), dtype=np.float32)

    # Distance au bord du domaine (min des 4 distances)
    dist_border = np.zeros((n, 1), dtype=np.float32)
    for k, (px, py) in enumerate(positions):
        dist_border[k] = min(px, NX - 1 - px, py, NY - 1 - py)
    dist_border = dist_border / (max(NX, NY) / 2)  # normalisé ~[0, 1]

    x_nodes = torch.tensor(
        np.hstack([x_norm, y_norm, corr_max_vals, degree_norm,
                   var_T, var_S, grad_mean, dist_border]),
        dtype=torch.float)

    return {
        "x": x_nodes,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "positions": positions,
        "corr_matrix": corr_matrix,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLES — GAT + GraphSAGE
# ══════════════════════════════════════════════════════════════════════════════

class OceanNetworkGAT(nn.Module):
    """
    GAT pour l'analyse du réseau. Prédit un score de contribution par nœud.
    Les poids d'attention alpha_{ij} sont le signal principal de redondance.
    """
    def __init__(self, in_dim=8, hidden_dim=32, out_dim=1, n_heads=4):
        super().__init__()
        if PYG_AVAILABLE:
            self.gat1 = GATConv(in_dim, hidden_dim, heads=n_heads, dropout=0.1)
            self.gat2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=1, concat=False)
        else:
            self.gat1 = ManualGATLayer(in_dim, hidden_dim * n_heads)
            self.gat2 = ManualGATLayer(hidden_dim * n_heads, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(32, out_dim),
        )
        self._attention_weights = None

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        if PYG_AVAILABLE:
            h, (_, alpha1) = self.gat1(x, edge_index, return_attention_weights=True)
            h = F.elu(h)
            h, (_, alpha2) = self.gat2(h, edge_index, return_attention_weights=True)
            h = F.elu(h)
            self._attention_weights = alpha2.detach()
        else:
            h, alpha1 = self.gat1(x, edge_index)
            h, alpha2 = self.gat2(h, edge_index)
            self._attention_weights = alpha2.detach()

        node_scores = self.head(h).squeeze(-1)
        if return_attention:
            return node_scores, alpha2
        return node_scores


class GraphSAGEInductive(nn.Module):
    """
    GraphSAGE inductif : évalue des capteurs non vus à l'entraînement.
    Entraîné sur les mêmes targets proxy que le GAT.
    """
    def __init__(self, in_dim=8, hidden_dim=64, out_dim=1):
        super().__init__()
        if PYG_AVAILABLE:
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        else:
            self.conv1 = ManualGATLayer(in_dim, hidden_dim)
            self.conv2 = ManualGATLayer(hidden_dim, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32), nn.GELU(), nn.Linear(32, out_dim),
        )

    def forward(self, x, edge_index):
        if PYG_AVAILABLE:
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
        else:
            h, _ = self.conv1(x, edge_index)
            h = F.relu(h)
            h, _ = self.conv2(h, edge_index)
            h = F.relu(h)
        return self.head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  TARGETS (proxy rapide sans Brique 1)
# ══════════════════════════════════════════════════════════════════════════════

def compute_proxy_targets(positions, corr_matrix):
    """contribution_i = 1 − mean(|corr(i, j)|) pour j ≠ i. Normalisé [0,1]."""
    n = len(positions)
    targets = np.zeros(n)
    for i in range(n):
        targets[i] = 1.0 - np.mean(np.abs(np.delete(corr_matrix[i], i)))
    targets = (targets - targets.min()) / (targets.max() - targets.min() + 1e-9)
    return torch.tensor(targets, dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT GAT
# ══════════════════════════════════════════════════════════════════════════════

def train_gnn(args, graph_dict, targets):
    """Entraîne le GAT sur la tâche de scoring nodal (split 80/20)."""
    print("\n── Entraînement GAT ───────────────────────────────────────────────")
    model = OceanNetworkGAT(in_dim=graph_dict["x"].shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x          = graph_dict["x"].to(DEVICE)
    edge_index = graph_dict["edge_index"].to(DEVICE)
    y          = targets.to(DEVICE)

    n = x.shape[0]
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:int(0.8 * n)]] = True
    test_mask = ~train_mask

    best_loss = np.inf
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.gnn_epochs + 1):
        model.train()
        scores = model(x, edge_index)
        loss = F.mse_loss(scores[train_mask], y[train_mask])
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_loss = F.mse_loss(model(x, edge_index)[test_mask], y[test_mask])
            print(f"  Époque {epoch:3d} | Train MSE={loss.item():.4f} | Test MSE={test_loss.item():.4f}")
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                torch.save(model.state_dict(), out_dir / "gnn_best.pt")

    print(f"  ✓ Checkpoint → {out_dir}/gnn_best.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT GraphSAGE
# ══════════════════════════════════════════════════════════════════════════════

def train_sage(args, graph_dict, targets):
    """Entraîne le GraphSAGE sur les mêmes targets proxy (pour l'inférence inductive)."""
    print("\n── Entraînement GraphSAGE ─────────────────────────────────────────")
    model = GraphSAGEInductive(in_dim=graph_dict["x"].shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x          = graph_dict["x"].to(DEVICE)
    edge_index = graph_dict["edge_index"].to(DEVICE)
    y          = targets.to(DEVICE)

    n = x.shape[0]
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:int(0.8 * n)]] = True
    test_mask = ~train_mask

    best_loss = np.inf
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = getattr(args, 'gnn_epochs', 200)

    for epoch in range(1, n_epochs + 1):
        model.train()
        scores = model(x, edge_index)
        loss = F.mse_loss(scores[train_mask], y[train_mask])
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_loss = F.mse_loss(model(x, edge_index)[test_mask], y[test_mask])
            print(f"  Époque {epoch:3d} | Train MSE={loss.item():.4f} | Test MSE={test_loss.item():.4f}")
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                torch.save(model.state_dict(), out_dir / "sage_best.pt")

    print(f"  ✓ Checkpoint SAGE → {out_dir}/sage_best.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSE DU RÉSEAU
# ══════════════════════════════════════════════════════════════════════════════

def analyze_network(model, graph_dict, targets, args, T=None, label=""):
    """Diagnostic complet : scores, attention, redondance, couverture, figures."""
    from matplotlib.colors import LinearSegmentedColormap

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    x          = graph_dict["x"].to(DEVICE)
    edge_index = graph_dict["edge_index"].to(DEVICE)
    positions  = graph_dict["positions"]
    pos_arr    = np.array(positions)
    corr_matrix = graph_dict["corr_matrix"]
    n = len(positions)

    with torch.no_grad():
        scores, attention = model(x, edge_index, return_attention=True)
    scores = scores.cpu().numpy()

    # Matrice d'attention
    attention_matrix = np.zeros((n, n))
    ei = graph_dict["edge_index"].numpy()
    a_vals = attention.cpu().squeeze().numpy()
    if a_vals.ndim > 1:
        a_vals = a_vals.mean(axis=-1)
    for k, (s, d) in enumerate(zip(ei[0], ei[1])):
        if k < len(a_vals):
            attention_matrix[s, d] = max(attention_matrix[s, d], float(a_vals[k]))

    # Score de redondance — basé sur les poids d'attention GAT
    # L'attention GAT capture les dépendances non-linéaires apprises,
    # pas seulement la corrélation pairwise brute.
    # redondance_i = attention entrante moyenne normalisée :
    #   un nœud fortement "attendu" par ses voisins porte une info
    #   déjà disponible via eux → redondant.
    redundancy_score = np.zeros(n)
    for i in range(n):
        incoming = attention_matrix[:, i]  # attention de tous les nœuds vers i
        incoming[i] = 0.0
        neighbors = np.where(incoming > 0)[0]
        if len(neighbors) > 0:
            redundancy_score[i] = incoming[neighbors].mean()
        else:
            redundancy_score[i] = 0.0
    r_min, r_max = redundancy_score.min(), redundancy_score.max()
    redundancy_score = (redundancy_score - r_min) / (r_max - r_min + 1e-9) if r_max > r_min else np.full(n, 0.5)

    # Couverture spatiale
    grid_res = 16
    coverage_grid = np.zeros((NX // grid_res + 1, NY // grid_res + 1))
    for (x_p, y_p) in positions:
        coverage_grid[x_p // grid_res, y_p // grid_res] += 1

    # Seuil de redondance
    unicite = 1 - redundancy_score
    is_redundant = unicite < np.percentile(unicite, 25)

    # Fond SST optionnel
    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    sst_bg   = T.mean(axis=0) if T is not None else None
    sst_vmin = sst_bg.min() if sst_bg is not None else 0
    sst_vmax = sst_bg.max() if sst_bg is not None else 1

    def _bg(ax):
        if sst_bg is not None:
            ax.imshow(sst_bg.T, cmap=ocean_cmap, origin="lower", aspect="auto",
                      vmin=sst_vmin, vmax=sst_vmax, alpha=0.45, extent=[0, NX, 0, NY])
        ax.set_xlim(0, NX); ax.set_ylim(0, NY)

    # Visualisation 2×3
    suffix = f"_{label}" if label else ""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Brique 2 — GNN : Analyse Structurelle" + (f" [{label}]" if label else ""),
                 fontsize=14, fontweight="bold")

    def _scatter(ax, colors, cmap, vmin, vmax, title, cbar_label, mark_red=False):
        _bg(ax)
        sc = ax.scatter(pos_arr[:,0], pos_arr[:,1], c=colors, cmap=cmap, s=130,
                        vmin=vmin, vmax=vmax, zorder=5, edgecolors="white", linewidths=0.8)
        if mark_red and is_redundant.any():
            ax.scatter(pos_arr[is_redundant,0], pos_arr[is_redundant,1], s=286,
                       facecolors="none", edgecolors="#ff4444", linewidths=2.0, zorder=7,
                       label=f"Redondant ({is_redundant.sum()})")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7, facecolor="#111")
        plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, label=cbar_label)
        ax.set_title(title, fontsize=10, fontweight="bold")

    _scatter(axes[0,0], scores, "RdYlGn", scores.min(), scores.max(),
             "Score contribution GAT\n(vert=fort | ○ rouge=redondant)", "Contribution", mark_red=True)
    _scatter(axes[0,1], unicite, "RdYlGn", 0, 1,
             "Unicité (1−redondance)\n(vert=unique | ○ rouge=redondant)", "Unicité", mark_red=True)

    # Matrice de corrélation
    im = axes[0,2].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=axes[0,2])
    axes[0,2].set_title(f"Corrélation spatiale (seuil={args.corr_threshold})")

    # Graphe réseau
    ax = axes[1,0]; _bg(ax)
    for k, (s, d) in enumerate(zip(ei[0], ei[1])):
        if s < d:
            alpha_val = float(attention_matrix[s, d])
            ax.plot([pos_arr[s,0], pos_arr[d,0]], [pos_arr[s,1], pos_arr[d,1]],
                    color="white", alpha=min(alpha_val * 5, 0.8), linewidth=1.5, zorder=3)
    sc_g = ax.scatter(pos_arr[:,0], pos_arr[:,1], c=scores, cmap="RdYlGn", s=100,
                      vmin=scores.min(), vmax=scores.max(), edgecolors="black", linewidths=0.5, zorder=5)
    if is_redundant.any():
        ax.scatter(pos_arr[is_redundant,0], pos_arr[is_redundant,1], s=260,
                   facecolors="none", edgecolors="#ff4444", linewidths=2.0, zorder=7)
    plt.colorbar(sc_g, ax=ax, pad=0.02, fraction=0.046, label="Contribution")
    for i, (x_p, y_p) in enumerate(positions):
        ax.annotate(f"{i}", (x_p, y_p), fontsize=6, ha="center", va="center", color="black", zorder=6)
    ax.set_title("Graphe réseau (épaisseur ∝ attention GAT)")

    # Couverture
    im = axes[1,1].imshow(coverage_grid.T, cmap="Blues", origin="lower", aspect="auto")
    plt.colorbar(im, ax=axes[1,1])
    axes[1,1].set_title(f"Couverture spatiale ({grid_res}×{grid_res} px)")

    # Barplot contribution vs unicité
    ax = axes[1,2]
    idx_sorted = np.argsort(scores)[::-1]
    bw = 0.35; x_pos = np.arange(n)
    ax.bar(x_pos - bw/2, scores[idx_sorted], bw, label="Contribution GAT", color="steelblue", alpha=0.8)
    ax.bar(x_pos + bw/2, unicite[idx_sorted], bw, label="Unicité", color="orange", alpha=0.8)
    ax.set_title("Contribution vs Unicité")
    ax.legend(fontsize=8)
    ax.set_xticks(x_pos[::3])
    ax.set_xticklabels([f"C{idx_sorted[i]}" for i in range(0, n, 3)], fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / f"gnn_network_analysis{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure → {out_dir}/gnn_network_analysis{suffix}.png")

    # Recommandations textuelles
    print("\n── Recommandations GNN ────────────────────────────────────────────")
    combined = scores - redundancy_score
    for ci in np.argsort(combined)[:5]:
        print(f"  Candidat suppression : C{ci:2d} @ {positions[ci]} | "
              f"contrib={scores[ci]:.3f} | redondance={redundancy_score[ci]:.3f}")
    gaps = np.argwhere(coverage_grid == 0)
    print(f"\n  Zones lacunaires : {len(gaps)} cellules non couvertes")

    return scores, redundancy_score, coverage_grid


# ══════════════════════════════════════════════════════════════════════════════
#  ÉVALUATION INDUCTIVE (nouveaux capteurs)
# ══════════════════════════════════════════════════════════════════════════════

def inductive_eval(sage_model, graph_dict, new_positions, args):
    """
    Évalue des capteurs hypothétiques avec GraphSAGE pré-entraîné.
    Les nouveaux nœuds sont connectés aux k plus proches voisins existants.
    """
    from scipy.spatial import KDTree

    print("\n── Évaluation Inductive (nouveaux capteurs) ───────────────────────")
    out_dir = Path(args.output_dir)

    existing_pos = graph_dict["positions"]
    n_existing = len(existing_pos)
    all_positions = existing_pos + new_positions

    # Features pour les nouveaux nœuds (8 dim, inconnus remplis à 0)
    in_dim = graph_dict["x"].shape[1]
    new_feat_list = []
    for x_p, y_p in new_positions:
        f = [x_p / NX, y_p / NY, 0.5, 0.0]  # pos, corr_max=0.5, degree=0
        f += [0.0] * (in_dim - 4)             # var_T, var_S, grad, dist_bord inconnus
        # Distance au bord (celle-ci on peut la calculer)
        if in_dim >= 8:
            f[7] = min(x_p, NX-1-x_p, y_p, NY-1-y_p) / (max(NX, NY) / 2)
        new_feat_list.append(f)
    new_feat = torch.tensor(new_feat_list, dtype=torch.float)
    x_ext = torch.cat([graph_dict["x"], new_feat], dim=0)

    # Connexion des nouveaux nœuds aux k plus proches existants
    pos_arr = np.array(all_positions, dtype=np.float32)
    tree = KDTree(pos_arr[:n_existing])
    new_src, new_dst = [], []
    for i, (x_p, y_p) in enumerate(new_positions):
        _, idxs = tree.query([x_p, y_p], k=min(4, n_existing))
        for j in idxs:
            new_src += [n_existing + i, j]
            new_dst += [j, n_existing + i]

    edge_ext = torch.tensor(
        [graph_dict["edge_index"][0].tolist() + new_src,
         graph_dict["edge_index"][1].tolist() + new_dst], dtype=torch.long)

    sage_model.eval()
    with torch.no_grad():
        scores_all = sage_model(x_ext.to(DEVICE), edge_ext.to(DEVICE))
    scores_new = scores_all[n_existing:].cpu().numpy()

    print(f"  Scores prédits pour {len(new_positions)} nouveaux capteurs :")
    for i, (pos, sc) in enumerate(zip(new_positions, scores_new)):
        print(f"    Nouveau capteur @ {pos} → score = {sc:.4f}")

    # Visualisation
    fig, ax = plt.subplots(figsize=(8, 6))
    ex_arr = np.array(existing_pos)
    ax.scatter(ex_arr[:,0], ex_arr[:,1], c="steelblue", s=80, label="Existants", zorder=5)
    new_arr = np.array(new_positions)
    sc = ax.scatter(new_arr[:,0], new_arr[:,1], c=scores_new, cmap="RdYlGn", s=200,
                    marker="*", edgecolors="black", linewidths=1, label="Nouveaux (score)", zorder=6)
    plt.colorbar(sc, ax=ax, label="Score contribution")
    ax.set_xlim(0, NX); ax.set_ylim(0, NY)
    ax.set_title("Évaluation inductive de nouveaux capteurs")
    ax.legend(); ax.grid(True, alpha=0.2)
    fig.savefig(out_dir / "gnn_inductive_eval.png", dpi=150)
    plt.close()
    print(f"  ✓ Figure → {out_dir}/gnn_inductive_eval.png")


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Brique 2 — GNN")
    p.add_argument("--train",          action="store_true")
    p.add_argument("--analyze",        action="store_true")
    p.add_argument("--inductive",      action="store_true")
    p.add_argument("--seed_ocean",     type=int,   default=42)
    p.add_argument("--seed_buoys",     type=int,   default=7)
    p.add_argument("--new_positions",  type=str, default="[(10,20),(80,150),(130,40)]")
    p.add_argument("--corr_threshold", type=float, default=0.5)
    p.add_argument("--k_nearest",      type=int,   default=4)
    p.add_argument("--gnn_epochs",     type=int,   default=200)
    p.add_argument("--output_dir",     type=str,   default="outputs")
    p.add_argument("--n_buoys",        type=int,   default=N_BUOYS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not (args.train or args.analyze or args.inductive):
        print("Usage: python 02_gnn.py --train --analyze [--inductive]")
        sys.exit(0)

    set_global_seed(args.seed_ocean)

    print("═" * 60)
    print(" Brique 2 — GNN : Structure du Réseau d'Observation")
    print("═" * 60)

    print(f"\n[1/3] Nature run (seed={args.seed_ocean}, nt=500)...")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=500, seed=args.seed_ocean)

    rng = np.random.default_rng(args.seed_buoys)
    positions = [(int(rng.integers(0, NX)), int(rng.integers(0, NY))) for _ in range(args.n_buoys)]

    print(f"\n[2/3] Corrélation spatiale...")
    corr_matrix = build_spatial_correlation(T, S, positions, n_timestamps=300)

    print(f"\n[3/3] Graphe (seuil={args.corr_threshold}, k={args.k_nearest})...")
    graph_dict = build_graph(positions, corr_matrix,
                             corr_threshold=args.corr_threshold, k_nearest=args.k_nearest,
                             T=T, S=S)
    print(f"      Nœuds : {len(positions)} | Arêtes : {graph_dict['edge_index'].shape[1]}")
    targets = compute_proxy_targets(positions, corr_matrix)

    model_gat = None
    model_sage = None

    if args.train:
        model_gat = train_gnn(args, graph_dict, targets)
        model_sage = train_sage(args, graph_dict, targets)

    if args.analyze:
        if model_gat is None:
            model_gat = OceanNetworkGAT(in_dim=graph_dict["x"].shape[1]).to(DEVICE)
            ckpt_path = Path(args.output_dir) / "gnn_best.pt"
            if ckpt_path.exists():
                model_gat.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        analyze_network(model_gat, graph_dict, targets, args, T=T)

    if args.inductive:
        if model_sage is None:
            model_sage = GraphSAGEInductive(in_dim=graph_dict["x"].shape[1]).to(DEVICE)
            sage_path = Path(args.output_dir) / "sage_best.pt"
            if sage_path.exists():
                model_sage.load_state_dict(torch.load(sage_path, map_location=DEVICE, weights_only=True))
                print(f"  SAGE chargé depuis {sage_path}")
            else:
                print("  [WARN] Pas de checkpoint SAGE — entraînement automatique")
                model_sage = train_sage(args, graph_dict, targets)
        try:
            new_positions = ast.literal_eval(args.new_positions)
        except Exception:
            new_positions = [(10, 20), (80, 150), (130, 40)]
        inductive_eval(model_sage, graph_dict, new_positions, args)

    print("\n  ✓ Brique 2 terminée.")
