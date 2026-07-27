"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BRIQUE 2 — Graph Neural Network pour la Structure du Réseau         ║
║                                                                              ║
║  Pipeline :                                                                  ║
║    1. Construction du graphe depuis les positions des capteurs               ║
║       et la matrice de corrélation spatiale du nature run                    ║
║    2. Graph Attention Network (GAT) : apprend l'importance relative          ║
║       de chaque voisin → poids d'attention = proxy de redondance             ║
║    3. GraphSAGE en mode inductif : évalue des capteurs hypothétiques         ║
║       (gliders, Argo) non présents à l'entraînement                         ║
║    4. Analyse : détection redondance, lacunes, ranking des capteurs          ║
║                                                                              ║
║  Usage :                                                                     ║
║    python 02_gnn.py --build_graph                                            ║
║    python 02_gnn.py --train                                                  ║
║    python 02_gnn.py --analyze                                                ║
║    python 02_gnn.py --inductive --new_positions "[(10,20),(80,150)]"        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dépendances : pip install torch-geometric
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
from data.loader import load_ocean, add_data_args
from data.dataset import BuoySampler

# ── Import PyTorch Geometric ───────────────────────────────────────────────────
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[WARN] torch-geometric non disponible — implémentation manuelle utilisée.")
    print("       pip install torch-geometric  pour activer les GATConv/SAGEConv natifs")


# ══════════════════════════════════════════════════════════════════════════════
#  IMPLÉMENTATION MANUELLE (fallback si torch_geometric absent)
#  Message Passing + Attention simplifiés
# ══════════════════════════════════════════════════════════════════════════════

class ManualGATLayer(nn.Module):
    """
    Graph Attention Layer manuel (single-head) si PyG indisponible.
    Identique à Veličković et al. (2018) pour 1 tête.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W  = nn.Linear(in_dim, out_dim, bias=False)
        self.a  = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, h, edge_index):
        """
        h          : (N, in_dim)  features nodaux
        edge_index : (2, E)       arêtes (src, dst)
        """
        Wh = self.W(h)                        # (N, out_dim)
        src, dst = edge_index[0], edge_index[1]

        # Calcul des coefficients d'attention
        e = torch.cat([Wh[src], Wh[dst]], dim=-1)  # (E, 2*out_dim)
        alpha = F.leaky_relu(self.a(e), 0.2).squeeze(-1)  # (E,)

        # Softmax par nœud destination
        alpha_exp = torch.exp(alpha - alpha.max())
        alpha_sum = torch.zeros(h.size(0), device=h.device)
        alpha_sum.scatter_add_(0, dst, alpha_exp)
        alpha_norm = alpha_exp / (alpha_sum[dst] + 1e-9)   # (E,)

        # Agrégation
        out = torch.zeros_like(Wh)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(Wh[src]),
                         alpha_norm.unsqueeze(-1) * Wh[src])
        return F.elu(out), alpha_norm


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DU GRAPHE
# ══════════════════════════════════════════════════════════════════════════════

def build_spatial_correlation(T, S, positions, n_timestamps=200):
    """
    Estime la matrice de corrélation spatiale entre capteurs à partir du nature run.

    Rho[i,j] = corrélation de Pearson entre les séries temporelles
                (T+S normalisé) aux positions i et j.

    Cette corrélation naturelle du système guide la construction des arêtes :
    deux capteurs très corrélés sont potentiellement redondants.
    """
    n = len(positions)
    t_idx = np.random.choice(len(T), min(n_timestamps, len(T)), replace=False)

    # `T` peut être soit un champ 2D+T (nt, nx, ny) — mode legacy —, soit le
    # tenseur multi-canaux (nt, n_ch, nx, ny). Dans le second cas on empile
    # TOUS les canaux : les courants apportent une information de connectivité
    # dynamique que T/S seuls ne portent pas, et c'est précisément ce que le
    # GNN doit exploiter pour juger la redondance entre stations.
    if T.ndim == 4:
        fields = T
        series = np.zeros((n, fields.shape[1] * len(t_idx)))
        for k, (x, y) in enumerate(positions):
            cols = []
            for c in range(fields.shape[1]):
                ts = fields[:, c, x, y]
                cols.append((ts[t_idx] - ts.mean()) / (ts.std() + 1e-9))
            series[k] = np.concatenate(cols)
    else:
        series = np.zeros((n, len(t_idx)))
        for k, (x, y) in enumerate(positions):
            ts_T = (T[t_idx, x, y] - T[:, x, y].mean()) / (T[:, x, y].std() + 1e-9)
            ts_S = (S[t_idx, x, y] - S[:, x, y].mean()) / (S[:, x, y].std() + 1e-9)
            series[k] = 0.6 * ts_T + 0.4 * ts_S

    corr_matrix = np.corrcoef(series)   # (n, n)
    return corr_matrix


def build_graph(positions, corr_matrix,
                corr_threshold=0.5,
                k_nearest=4):
    """
    Construit le graphe du réseau d'observation.

    Stratégie d'arête combinée :
        (a) Seuil de corrélation : |rho| > threshold → arête
        (b) k plus proches voisins géographiques : garantit la connexité

    Features nodaux (x_nodes) :
        [lon_norm, lat_norm, variance_T_norm, variance_S_norm,
         degré_géo_norm, corrélation_max_avec_voisin]

    Retourne un dict compatible torch_geometric.Data et le fallback manuel.
    """
    n = len(positions)
    pos_arr = np.array(positions, dtype=np.float32)

    # ── (a) Arêtes par seuil de corrélation ──────────────────────────────────
    src_list, dst_list, edge_attr_list = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > corr_threshold:
                src_list += [i, j]
                dst_list += [j, i]
                edge_attr_list += [corr_matrix[i, j], corr_matrix[i, j]]

    # ── (b) k-NN géographique ────────────────────────────────────────────────
    from scipy.spatial import KDTree
    tree = KDTree(pos_arr)
    for i in range(n):
        dists, idxs = tree.query(pos_arr[i], k=k_nearest + 1)
        for j in idxs[1:]:   # exclure self
            if (i, j) not in set(zip(src_list, dst_list)):
                dist_norm = dists[list(idxs).index(j)] / (NX + NY)
                src_list += [i, j]
                dst_list += [j, i]
                edge_attr_list += [max(corr_matrix[i, j], 0.1),
                                   max(corr_matrix[i, j], 0.1)]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(-1)

    # ── Features nodaux ───────────────────────────────────────────────────────
    # Position normalisée
    x_norm = pos_arr[:, 0:1] / NX
    y_norm = pos_arr[:, 1:2] / NY
    # Corrélation max avec voisin (proxy de redondance)
    corr_max = np.array([corr_matrix[i, :].copy() for i in range(n)])
    np.fill_diagonal(corr_max, 0)
    corr_max_vals = corr_max.max(axis=1, keepdims=True)
    # Degré du nœud normalisé
    degree = np.bincount(src_list, minlength=n).reshape(-1, 1).astype(np.float32)
    degree_norm = degree / (degree.max() + 1e-9)

    node_features = np.hstack([x_norm, y_norm, corr_max_vals, degree_norm])
    x_nodes = torch.tensor(node_features, dtype=torch.float)

    graph_dict = {
        "x": x_nodes,                      # (N, 4)
        "edge_index": edge_index,           # (2, E)
        "edge_attr": edge_attr,             # (E, 1)
        "positions": positions,
        "corr_matrix": corr_matrix,
    }
    return graph_dict


# ══════════════════════════════════════════════════════════════════════════════
#  MODÈLE — GAT + GraphSAGE
# ══════════════════════════════════════════════════════════════════════════════

class OceanNetworkGAT(nn.Module):
    """
    Graph Attention Network pour l'analyse du réseau d'observation.

    Architecture :
        GAT layer 1 (4→32, 4 têtes) → GAT layer 2 (128→64, 1 tête)
        → MLP → score par nœud (observabilité prédite)

    Tâche supervisée :
        Pour chaque nœud (capteur), prédire sa "contribution locale"
        = amélioration de reconstruction RMSE apportée (target = LOO score
          normalisé, produit par la Brique 1 ou une proxy rapide).

    Les poids d'attention alpha_{ij} sont le signal principal d'analyse :
        alpha_{ij} élevé → capteur j fortement influencé par i
        → redondance potentielle si la corrélation est élevée
    """
    def __init__(self, in_dim=4, hidden_dim=32, out_dim=1, n_heads=4):
        super().__init__()
        if PYG_AVAILABLE:
            self.gat1 = GATConv(in_dim, hidden_dim, heads=n_heads, dropout=0.1)
            self.gat2 = GATConv(hidden_dim * n_heads, hidden_dim, heads=1, concat=False)
        else:
            self.gat1 = ManualGATLayer(in_dim, hidden_dim * n_heads)
            self.gat2 = ManualGATLayer(hidden_dim * n_heads, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, out_dim),
        )
        self._attention_weights = None   # stocké pour l'analyse post-hoc

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        if PYG_AVAILABLE:
            h, (ei, alpha1) = self.gat1(x, edge_index,
                                         return_attention_weights=True)
            h = F.elu(h)
            h, (_, alpha2) = self.gat2(h, edge_index,
                                        return_attention_weights=True)
            h = F.elu(h)
            self._attention_weights = alpha2.detach()
        else:
            h, alpha1 = self.gat1(x, edge_index)
            h, alpha2 = self.gat2(h, edge_index)
            self._attention_weights = alpha2.detach()

        node_scores = self.head(h).squeeze(-1)    # (N,)
        if return_attention:
            return node_scores, alpha2
        return node_scores


class GraphSAGEInductive(nn.Module):
    """
    GraphSAGE en mode inductif pour évaluer de nouveaux capteurs.

    Contrairement au GAT transductif (qui ne voit que les nœuds d'entraînement),
    GraphSAGE apprend des fonctions d'agrégation généralisables :
    on peut insérer un nouveau nœud (glider, Argo) dans le graphe
    et obtenir immédiatement son embedding sans ré-entraînement.

    Usage OED :
        → Simuler l'ajout d'une nouvelle bouée/glider
        → Prédire sa contribution marginale sans LOO exhaustif
    """
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=1):
        super().__init__()
        if PYG_AVAILABLE:
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        else:
            # Agrégation mean manuelle
            self.conv1 = ManualGATLayer(in_dim, hidden_dim)
            self.conv2 = ManualGATLayer(hidden_dim, hidden_dim // 2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, out_dim),
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
#  GÉNÉRATION DES TARGETS (proxy rapide sans Brique 1)
# ══════════════════════════════════════════════════════════════════════════════

def compute_proxy_targets(positions, corr_matrix):
    """
    Target de supervision rapide (sans charger la Brique 1) :
        contribution_i = 1 - mean(|corr(i, j)|) pour j ≠ i

    Interprétation : un capteur très corrélé à tous ses voisins
    a une faible contribution marginale → il est redondant.
    Ce proxy est cohérent avec la définition théorique OED
    (réduction d'entropie marginale en cas gaussien).
    """
    n = len(positions)
    targets = np.zeros(n)
    for i in range(n):
        off_diag = np.delete(corr_matrix[i], i)
        targets[i] = 1.0 - np.mean(np.abs(off_diag))
    # Normalisation [0, 1]
    targets = (targets - targets.min()) / (targets.max() - targets.min() + 1e-9)
    return torch.tensor(targets, dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT GNN
# ══════════════════════════════════════════════════════════════════════════════

def train_gnn(args, graph_dict, targets):
    """
    Entraîne le GAT sur la tâche de scoring nodal.

    Stratégie semi-supervisée :
        On entraîne sur un sous-ensemble de nœuds (masque train)
        et on évalue sur le reste → simule l'évaluation de nouveaux capteurs.
    """
    print("\n── Entraînement GAT ───────────────────────────────────────────────")
    model = OceanNetworkGAT(in_dim=graph_dict["x"].shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x          = graph_dict["x"].to(DEVICE)
    edge_index = graph_dict["edge_index"].to(DEVICE)
    y          = targets.to(DEVICE)

    n = x.shape[0]
    # Masque train/test (80/20 aléatoire sur les nœuds)
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:int(0.8 * n)]] = True
    test_mask  = ~train_mask

    best_loss = np.inf
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.gnn_epochs + 1):
        model.train()
        scores = model(x, edge_index)
        loss = F.mse_loss(scores[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                scores_eval = model(x, edge_index)
                test_loss = F.mse_loss(scores_eval[test_mask], y[test_mask])
            print(f"  Époque {epoch:3d} | Train MSE={loss.item():.4f} | "
                  f"Test MSE={test_loss.item():.4f}")
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                torch.save(model.state_dict(), out_dir / "gnn_best.pt")

    print(f"  ✓ Checkpoint → {out_dir}/gnn_best.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSE DU RÉSEAU
# ══════════════════════════════════════════════════════════════════════════════

def analyze_network(model, graph_dict, targets, args, T=None, label=""):
    """
    Produit le diagnostic complet du réseau :
        - Scores nodaux prédits (contribution de chaque capteur)
        - Poids d'attention GAT (redondance inter-capteurs)
        - Identification des zones lacunaires (coarse grid non couvert)
        - Recommandations : capteurs à retirer / zones à couvrir

    T      : nature run (NT, NX, NY) — si fourni, SST moyenne en fond sur
             les cartes contribution / redondance / graphe réseau.
    label  : suffixe pour le nom de fichier (ex: "rl_optimal", "random")
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    x          = graph_dict["x"].to(DEVICE)
    edge_index = graph_dict["edge_index"].to(DEVICE)
    positions  = graph_dict["positions"]
    pos_arr    = np.array(positions)
    corr_matrix = graph_dict["corr_matrix"]

    with torch.no_grad():
        scores, attention = model(x, edge_index, return_attention=True)
    scores = scores.cpu().numpy()

    # ── Matrice d'attention ────────────────────────────────────────────────────
    # Pour chaque arête, le poids d'attention indique l'influence du nœud source
    # La matrice attention_matrix[i,j] = mean des poids sur l'arête (i→j)
    n = len(positions)
    attention_matrix = np.zeros((n, n))
    ei = graph_dict["edge_index"].numpy()
    if PYG_AVAILABLE:
        a_vals = attention.cpu().squeeze().numpy()
    else:
        a_vals = attention.cpu().numpy()

    if a_vals.ndim > 1:
        a_vals = a_vals.mean(axis=-1)

    for k, (s, d) in enumerate(zip(ei[0], ei[1])):
        if k < len(a_vals):
            attention_matrix[s, d] = max(attention_matrix[s, d], float(a_vals[k]))

    # ── Score de redondance par nœud (corrélation pairwise moyenne) ───────────
    # L'attention peut être creuse → utiliser la corrélation directe entre capteurs
    # redondance_i = mean |corr(i,j)| pour j ≠ i  (voisins dans le graphe)
    # unicite_i    = 1 - redondance_i
    # Un capteur très corrélé avec ses voisins est redondant.
    # Un capteur faiblement corrélé = information unique → forte unicité.
    n = len(positions)
    corr_mat = corr_matrix  # (n, n) corrélation déjà calculée
    redundancy_score = np.zeros(n)
    for i in range(n):
        row = np.abs(corr_mat[i, :])
        row[i] = 0.0                      # exclure la corrélation avec soi-même
        neighbors = np.where(row > 0)[0]  # voisins avec corrélation non nulle
        if len(neighbors) > 0:
            redundancy_score[i] = row[neighbors].mean()
        else:
            # Nœud isolé : pas de voisins → unicité maximale par défaut
            redundancy_score[i] = 0.0

    # Normaliser [0, 1] pour comparaison avec les scores de contribution
    r_min, r_max = redundancy_score.min(), redundancy_score.max()
    if r_max > r_min:
        redundancy_score = (redundancy_score - r_min) / (r_max - r_min)
    else:
        # Tous les capteurs également redondants → unicité uniforme à 0.5
        redundancy_score = np.full(n, 0.5)

    # ── Couverture spatiale (grille grossière) ─────────────────────────────────
    # Résolution de la grille de couverture, adaptée au domaine.
    # L'ancien grid_res=16 fixe donnait 3x3 cellules sur une grille 48x48 :
    # beaucoup trop grossier pour localiser une lacune.
    grid_res = max(4, min(NX, NY) // 8)
    # ceil() sans "+1" : avec NX=48 et grid_res=16, "NX//16 + 1" créait une
    # 4e ligne fantôme (pixels 48-63) hors du domaine, d'où des zones
    # lacunaires signalées à des coordonnées inexistantes.
    coverage_grid = np.zeros((-(-NX // grid_res), -(-NY // grid_res)))
    for (x_p, y_p) in positions:
        coverage_grid[x_p // grid_res, y_p // grid_res] += 1

    # ── Fond SST (optionnel) ──────────────────────────────────────────────────
    # SST moyenne temporelle — image de fond cohérente pour tous les panneaux
    from matplotlib.colors import LinearSegmentedColormap
    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    sst_bg     = T.mean(axis=0) if T is not None else None   # (NX, NY)
    sst_vmin   = sst_bg.min()   if sst_bg is not None else 0
    sst_vmax   = sst_bg.max()   if sst_bg is not None else 1

    def _bg(ax):
        """Affiche la SST moyenne en fond + cadre."""
        if sst_bg is not None:
            ax.imshow(sst_bg.T, cmap=ocean_cmap, origin="lower", aspect="auto",
                      vmin=sst_vmin, vmax=sst_vmax, alpha=0.45,
                      extent=[0, NX, 0, NY])
        ax.set_xlim(0, NX); ax.set_ylim(0, NY)

    # ── Seuil de redondance ────────────────────────────────────────────────────
    # Un capteur est "redondant" si son unicité est dans le quartile inférieur
    # (très corrélé avec ses voisins → apport marginal faible)
    unicite = 1 - redundancy_score
    redondance_thr = np.percentile(unicite, 25)   # 25% les moins uniques
    is_redundant   = unicite < redondance_thr      # (n,) bool

    # ── Visualisation ─────────────────────────────────────────────────────────
    suffix = f"_{label}" if label else ""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Brique 2 — GNN : Analyse Structurelle du Réseau"
                 + (f"  [{label}]" if label else ""),
                 fontsize=14, fontweight="bold")

    def _scatter_on_sst(ax, pos, colors, cmap, vmin, vmax,
                        title, cbar_label, size=130, mark_redundant=False):
        """Scatter coloré sur fond SST. Cercles rouges sur les bouées redondantes."""
        _bg(ax)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=colors,
                        cmap=cmap, s=size, vmin=vmin, vmax=vmax,
                        zorder=5, edgecolors="white", linewidths=0.8)
        # Cercle rouge sur les bouées redondantes
        if mark_redundant and is_redundant.any():
            ax.scatter(pos[is_redundant, 0], pos[is_redundant, 1],
                       s=size * 2.2, facecolors="none",
                       edgecolors="#ff4444", linewidths=2.0, zorder=7,
                       label=f"Redondant ({is_redundant.sum()})")
            ax.legend(fontsize=7, loc="upper right",
                      framealpha=0.7, facecolor="#111")
        # Colorbar du score
        cb_score = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, location="right")
        cb_score.set_label(cbar_label, fontsize=8)
        # Colorbar SST
        if sst_bg is not None:
            import matplotlib.cm as _cm
            import matplotlib.colors as _mcolors
            sm_sst = _cm.ScalarMappable(
                cmap=ocean_cmap,
                norm=_mcolors.Normalize(vmin=sst_vmin, vmax=sst_vmax))
            sm_sst.set_array([])
            cb_sst = plt.colorbar(sm_sst, ax=ax, pad=0.13, fraction=0.03,
                                  location="right", shrink=0.55)
            cb_sst.set_label("SST moy. (°C)", fontsize=7, color="#555555")
            cb_sst.ax.tick_params(labelsize=6, color="#888888", labelcolor="#555555")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
        return sc

    # 1. Scores de contribution + fond SST + marque redondants
    _scatter_on_sst(axes[0, 0], pos_arr,
                    colors=scores, cmap="RdYlGn",
                    vmin=scores.min(), vmax=scores.max(),
                    title="Score de contribution GAT\n(vert = fort | ○ rouge = redondant)",
                    cbar_label="Contribution [0→1]",
                    mark_redundant=True)

    # 2. Score d'unicité (1 - redondance) + fond SST + marque redondants
    _scatter_on_sst(axes[0, 1], pos_arr,
                    colors=unicite, cmap="RdYlGn",
                    vmin=0, vmax=1,
                    title="Score d'unicité (1 − redondance)\n(vert = unique | ○ rouge = redondant)",
                    cbar_label="Unicité [0→1]",
                    mark_redundant=True)

    # 3. Matrice de corrélation
    ax = axes[0, 2]
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Matrice de corrélation spatiale\nSeuil arêtes = {args.corr_threshold}")
    ax.set_xlabel("Index capteur"); ax.set_ylabel("Index capteur")

    # 4. Graphe du réseau + fond SST + marquage redondants
    ax = axes[1, 0]
    _bg(ax)
    for k, (s, d) in enumerate(zip(ei[0], ei[1])):
        if s < d:
            alpha_val = float(attention_matrix[s, d])
            ax.plot([pos_arr[s, 0], pos_arr[d, 0]],
                    [pos_arr[s, 1], pos_arr[d, 1]],
                    color="white", alpha=min(alpha_val * 5, 0.8), linewidth=1.5, zorder=3)
    sc_g = ax.scatter(pos_arr[:, 0], pos_arr[:, 1],
               c=scores, cmap="RdYlGn", s=100,
               vmin=scores.min(), vmax=scores.max(),
               edgecolors="black", linewidths=0.5, zorder=5)
    # Cercles rouges sur redondants
    if is_redundant.any():
        ax.scatter(pos_arr[is_redundant, 0], pos_arr[is_redundant, 1],
                   s=260, facecolors="none", edgecolors="#ff4444",
                   linewidths=2.0, zorder=7, label=f"Redondant ({is_redundant.sum()})")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7, facecolor="#111")
    plt.colorbar(sc_g, ax=ax, pad=0.02, fraction=0.046, label="Contribution")
    for i, (x_p, y_p) in enumerate(positions):
        ax.annotate(f"{i}", (x_p, y_p), fontsize=6, ha="center", va="center",
                    color="black", zorder=6)
    ax.set_title("Graphe du réseau\n(épaisseur arête ∝ attention GAT | ○ rouge = redondant)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # 5. Couverture spatiale
    ax = axes[1, 1]
    im = ax.imshow(coverage_grid.T, cmap="Blues", origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Couverture spatiale (grille {grid_res}×{grid_res} px)\n"
                 f"(blanc = zone non couverte → candidat à l'ajout)")
    ax.set_xlabel(f"x / {grid_res}"); ax.set_ylabel(f"y / {grid_res}")

    # 6. Recommandations : barplot contribution vs redondance
    ax = axes[1, 2]
    idx_sorted = np.argsort(scores)[::-1]
    bar_width = 0.35
    x_pos = np.arange(n)
    ax.bar(x_pos - bar_width/2, scores[idx_sorted],
           bar_width, label="Contribution GAT", color="steelblue", alpha=0.8)
    ax.bar(x_pos + bar_width/2, 1 - redundancy_score[idx_sorted],
           bar_width, label="Unicité (1 - redondance)", color="orange", alpha=0.8)
    ax.set_xlabel("Capteurs (triés par contribution)")
    ax.set_ylabel("Score [0, 1]")
    ax.set_title("Contribution vs Unicité par capteur\n(orange > bleu → candidat à supprimer)")
    ax.legend(fontsize=8)
    ax.set_xticks(x_pos[::3])
    ax.set_xticklabels([f"C{idx_sorted[i]}" for i in range(0, n, 3)], fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "gnn_network_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure → {out_dir}/gnn_network_analysis.png")

    # ── Recommandations textuelles ─────────────────────────────────────────────
    print("\n── Recommandations GNN ────────────────────────────────────────────")
    # Candidats à la suppression : faible contribution + forte redondance
    combined_score = scores - redundancy_score
    candidates_remove = np.argsort(combined_score)[:5]
    print(f"  Capteurs candidats à la SUPPRESSION (redondants) :")
    for ci in candidates_remove:
        print(f"    C{ci:2d} @ {positions[ci]} | "
              f"contribution={scores[ci]:.3f} | redondance={redundancy_score[ci]:.3f}")

    # Zones lacunaires
    gaps = np.argwhere(coverage_grid == 0)
    print(f"\n  Zones lacunaires ({len(gaps)} cellules de grille non couvertes) :")
    if len(gaps) > 0:
        for gx, gy in gaps[:5]:
            print(f"    Cellule grille ({gx}, {gy}) → "
                  f"pixel centre (~{gx*grid_res+grid_res//2}, ~{gy*grid_res+grid_res//2})")
    if len(gaps) > 5:
        print(f"    ... et {len(gaps) - 5} autres zones")

    return scores, redundancy_score, coverage_grid


def inductive_eval(model, graph_dict, new_positions, corr_matrix_orig, args):
    """
    Évalue des capteurs hypothétiques (gliders, Argo) non vus à l'entraînement.

    Processus :
        1. Ajouter les nouveaux nœuds au graphe (features géographiques)
        2. Connecter aux voisins existants par kNN
        3. Passer le graphe étendu dans GraphSAGE
        4. Lire les scores des nouveaux nœuds → contribution prédite

    C'est l'avantage fondamental du mode inductif : pas de ré-entraînement.
    """
    print("\n── Évaluation Inductive (nouveaux capteurs) ───────────────────────")
    out_dir = Path(args.output_dir)

    existing_pos = graph_dict["positions"]
    n_existing   = len(existing_pos)
    all_positions = existing_pos + new_positions
    n_all = len(all_positions)

    # Nouvelles features nodales (positions inconnues)
    new_features = []
    for (x_p, y_p) in new_positions:
        new_features.append([x_p / NX, y_p / NY, 0.5, 0.0])  # corr_max=0.5, degree=0 (inco.)
    new_feat_tensor = torch.tensor(new_features, dtype=torch.float)
    x_extended = torch.cat([graph_dict["x"], new_feat_tensor], dim=0)

    # Connexion des nouveaux nœuds aux k plus proches voisins existants
    from scipy.spatial import KDTree
    pos_arr = np.array(all_positions, dtype=np.float32)
    tree = KDTree(pos_arr[:n_existing])
    new_edges_src, new_edges_dst = [], []
    for i, (x_p, y_p) in enumerate(new_positions):
        _, idxs = tree.query([x_p, y_p], k=min(4, n_existing))
        for j in idxs:
            new_edges_src += [n_existing + i, j]
            new_edges_dst += [j, n_existing + i]

    edge_ext = torch.tensor(
        [graph_dict["edge_index"][0].tolist() + new_edges_src,
         graph_dict["edge_index"][1].tolist() + new_edges_dst],
        dtype=torch.long)

    # Évaluation avec GraphSAGE
    sage_model = GraphSAGEInductive(in_dim=x_extended.shape[1]).to(DEVICE)
    # Note : en production, charger les poids SAGE pré-entraînés
    # Ici, on illustre le pipeline avec un modèle non entraîné
    sage_model.eval()
    with torch.no_grad():
        scores_all = sage_model(x_extended.to(DEVICE), edge_ext.to(DEVICE))
    scores_new = scores_all[n_existing:].cpu().numpy()

    print(f"  Scores prédits pour {len(new_positions)} nouveaux capteurs :")
    for i, (pos, sc) in enumerate(zip(new_positions, scores_new)):
        print(f"    Nouveau capteur @ {pos} → score = {sc:.4f}")

    # Visualisation
    fig, ax = plt.subplots(figsize=(8, 6))
    ex_arr = np.array(existing_pos)
    ax.scatter(ex_arr[:, 0], ex_arr[:, 1],
               c="steelblue", s=80, label="Capteurs existants", zorder=5)
    new_arr = np.array(new_positions)
    sc = ax.scatter(new_arr[:, 0], new_arr[:, 1],
                    c=scores_new, cmap="RdYlGn", s=200,
                    marker="*", edgecolors="black", linewidths=1,
                    label="Nouveaux capteurs (score)", zorder=6)
    plt.colorbar(sc, ax=ax, label="Score de contribution prédit")
    ax.set_xlim(0, NX); ax.set_ylim(0, NY)
    ax.set_title("Évaluation inductive de nouveaux capteurs\n(étoile = glider/Argo hypothétique)")
    ax.legend()
    ax.grid(True, alpha=0.2)
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
    p.add_argument("--report",         action="store_true",
                   help="Produit un rapport .txt avec les métriques clés")
    p.add_argument("--seed_ocean",     type=int,   default=42)
    p.add_argument("--seed_buoys",     type=int,   default=7)
    p.add_argument("--new_positions",  type=str, default="[(10,20),(80,150),(130,40)]")
    p.add_argument("--corr_threshold", type=float, default=0.5)
    p.add_argument("--k_nearest",      type=int,   default=4)
    p.add_argument("--gnn_epochs",     type=int,   default=200)
    p.add_argument("--output_dir",     type=str,   default="outputs")
    p.add_argument("--n_buoys",        type=int,   default=N_BUOYS)
    add_data_args(p)
    return p.parse_args()


if __name__ == "__main__":
    from datetime import datetime
    args = parse_args()

    if not (args.train or args.analyze or args.inductive):
        print("Usage: python 02_gnn.py --train --analyze [--inductive] [--report]")
        sys.exit(0)

    print("═" * 60)
    print(" Brique 2 — GNN : Structure du Réseau d'Observation")
    print("═" * 60)

    print("\n[1/3] Chargement du champ océanique...")
    fields, channels, sea_mask, data_info = load_ocean(args)
    T, S = fields[:, 0], fields[:, 1]
    print(f"      {data_info['source']} | {fields.shape} | canaux={channels}")

    positions = BuoySampler(NX, NY, args.n_buoys, sea_mask=sea_mask,
                            min_dist=MIN_BUOY_DIST,
                            rng=args.seed_buoys).positions
    print(f"      {args.n_buoys} capteurs (seed_buoys={args.seed_buoys}, "
          f"dist_min={MIN_BUOY_DIST} px)")

    print("\n[2/3] Calcul de la matrice de corrélation spatiale...")
    corr_matrix = build_spatial_correlation(fields, None, positions,
                                            n_timestamps=300)

    print(f"\n[3/3] Construction du graphe (seuil={args.corr_threshold}, k={args.k_nearest})...")
    graph_dict = build_graph(positions, corr_matrix,
                             corr_threshold=args.corr_threshold,
                             k_nearest=args.k_nearest)
    n_edges = graph_dict["edge_index"].shape[1]
    print(f"      Nœuds : {len(positions)} | Arêtes : {n_edges}")
    targets = compute_proxy_targets(positions, corr_matrix)

    model = None
    scores_out = redund_out = None
    if args.train:
        model = train_gnn(args, graph_dict, targets)

    if args.analyze:
        if model is None:
            model = OceanNetworkGAT(in_dim=graph_dict["x"].shape[1]).to(DEVICE)
            ckpt_path = Path(args.output_dir) / "gnn_best.pt"
            if ckpt_path.exists():
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE,
                                                  weights_only=True))
                print(f"  Modèle chargé depuis {ckpt_path}")
        scores_out, redund_out, _ = analyze_network(
            model, graph_dict, targets, args, T=T)

    if args.inductive:
        if model is None:
            model = OceanNetworkGAT(in_dim=graph_dict["x"].shape[1]).to(DEVICE)
        try:
            new_positions = ast.literal_eval(args.new_positions)
        except Exception:
            new_positions = [(10, 20), (80, 150), (130, 40)]
        inductive_eval(model, graph_dict, new_positions, corr_matrix, args)

    if args.report:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output_dir)
        unicite = (1 - redund_out) if redund_out is not None else None
        is_redond = (unicite < np.percentile(unicite, 25)) if unicite is not None else None
        lines = [
            "=" * 68,
            "  Brique 2 — GNN — Rapport",
            f"  Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 68, "",
            "── REPRODUCTIBILITÉ ─────────────────────────────────────────────────",
            f"  seed_ocean     : {args.seed_ocean}",
            f"  seed_buoys     : {args.seed_buoys}",
            f"  n_buoys        : {args.n_buoys}",
            f"  corr_threshold : {args.corr_threshold}",
            f"  k_nearest      : {args.k_nearest}",
            "",
            "── POSITIONS DES BOUÉES ─────────────────────────────────────────────",
        ] + [f"  B{i:02d} : ({px:4d}, {py:4d})" for i, (px, py) in enumerate(positions)] + [
            "",
            "── GRAPHE ───────────────────────────────────────────────────────────",
            f"  Nœuds   : {len(positions)}",
            f"  Arêtes  : {n_edges}",
        ]
        if scores_out is not None:
            lines += [
                f"  Score contribution moy : {scores_out.mean():.3f} ± {scores_out.std():.3f}",
                f"  Redondance moyenne     : {redund_out.mean():.3f}",
                f"  Capteurs redondants    : {int(is_redond.sum())}  (unicité Q25)",
                f"  IDs redondants         : {[int(i) for i in np.where(is_redond)[0]]}",
            ]
        lines += ["", "── FICHIERS PRODUITS ────────────────────────────────────────────────"]
        for f in sorted(out.iterdir()):
            if f.suffix in {".pt", ".png"}:
                lines.append(f"  {f.name:<44} {f.stat().st_size//1024:>5} KB")
        lines += ["", "=" * 68]
        rpt = out / f"rapport_gnn_{ts}.txt"
        rpt.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  Rapport GNN → {rpt}")

    print("\n  ✓ Brique 2 terminée.")
