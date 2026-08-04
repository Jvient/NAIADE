"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         BRICK 2 -- Graph Neural Network for network structure                ║
║                                                                              ║
║  Pipeline:                                                                   ║
║    1. Build the graph from sensor positions and the spatial                  ║
║       correlation matrix of the nature run                                   ║
║    2. Graph Attention Network (GAT): learns the relative importance          ║
║       of each neighbour -> attention weights = redundancy proxy              ║
║    3. GraphSAGE in inductive mode: scores hypothetical sensors               ║
║       (gliders, Argo floats) absent from training                            ║
║    4. Analysis: redundancy detection, gaps, sensor ranking                   ║
║                                                                              ║
║  Usage:                                                                      ║
║    python 02_gnn.py --build_graph                                            ║
║    python 02_gnn.py --train                                                  ║
║    python 02_gnn.py --analyze                                                ║
║    python 02_gnn.py --inductive --new_positions "[(10,20),(80,150)]"        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dependencies: pip install torch-geometric
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
from data.dataset import (SyntheticOceanGenerator, sensor_series,
                          local_variance_map, mesoscale_anomaly,
                          sample_separated_positions)

# -- PyTorch Geometric import -------------------------------------------------
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("[WARN] torch-geometric unavailable -- using the hand-written fallback.")
    print("       pip install torch-geometric  to enable native GATConv/SAGEConv")


# ══════════════════════════════════════════════════════════════════════════════
#  HAND-WRITTEN FALLBACK (used when torch_geometric is absent)
#  Simplified message passing + attention
# ══════════════════════════════════════════════════════════════════════════════

class ManualGATLayer(nn.Module):
    """
    Hand-written single-head graph attention layer, used when PyG is absent.
    Identical to Velickovic et al. (2018) for one head.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W  = nn.Linear(in_dim, out_dim, bias=False)
        self.a  = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, h, edge_index):
        """
        h          : (N, in_dim)  node features
        edge_index : (2, E)       edges (src, dst)
        """
        Wh = self.W(h)                        # (N, out_dim)
        src, dst = edge_index[0], edge_index[1]

        # Attention coefficients
        e = torch.cat([Wh[src], Wh[dst]], dim=-1)  # (E, 2*out_dim)
        alpha = F.leaky_relu(self.a(e), 0.2).squeeze(-1)  # (E,)

        # Softmax over destination nodes
        alpha_exp = torch.exp(alpha - alpha.max())
        alpha_sum = torch.zeros(h.size(0), device=h.device)
        alpha_sum.scatter_add_(0, dst, alpha_exp)
        alpha_norm = alpha_exp / (alpha_sum[dst] + 1e-9)   # (E,)

        # Aggregation
        out = torch.zeros_like(Wh)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(Wh[src]),
                         alpha_norm.unsqueeze(-1) * Wh[src])
        return F.elu(out), alpha_norm


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_spatial_correlation(T, S, positions, n_timestamps=200,
                              deseason=DESEASON_ANALYSIS):
    """
    Inter-sensor correlation matrix, estimated on the nature run.

    Rho[i,j] = Pearson correlation between the standardised (0.6*T + 0.4*S)
    series at positions i and j.

    `deseason=True` (default) removes the domain mean at every step before
    correlating. This is indispensable with the v3 nature run: the seasonal
    cycle is a near-uniform mode, and keeping it puts 40% of buoy pairs above
    |rho| = 0.5 regardless of their separation. The graph becomes a
    near-clique and redundancy loses all meaning. Once removed, mean |rho|
    drops from 0.45 to 0.17 and depends only on mesoscale structure -- which
    is what the network actually has to resolve.
    """
    t_idx = np.sort(np.random.choice(len(T), min(n_timestamps, len(T)),
                                     replace=False))
    series = sensor_series(T, S, positions, deseason=deseason, t_idx=t_idx)
    corr_matrix = np.corrcoef(series)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)
    return corr_matrix


def build_graph(positions, corr_matrix,
                corr_threshold=GNN_CORR_THRESHOLD,
                k_nearest=4,
                T=None, S=None):
    """
    Build the observing-network graph.

    Combined edge strategy:
        (a) correlation threshold: |rho| > threshold -> edge
        (b) k geographic nearest neighbours: guarantees connectivity

    Node features (x_nodes):
        [x_norm, y_norm, max_correlation_with_neighbour, degree_norm,
         local_SST_variance, local_SSS_variance]

    Returns a dict compatible with both torch_geometric.Data and the fallback.
    """
    n = len(positions)
    pos_arr = np.array(positions, dtype=np.float32)

    # -- (a) Edges from the correlation threshold ------------------------------
    src_list, dst_list, edge_attr_list = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > corr_threshold:
                src_list += [i, j]
                dst_list += [j, i]
                edge_attr_list += [corr_matrix[i, j], corr_matrix[i, j]]

    # -- (b) Geographic k-NN ---------------------------------------------------
    from scipy.spatial import KDTree
    tree = KDTree(pos_arr)
    for i in range(n):
        dists, idxs = tree.query(pos_arr[i], k=k_nearest + 1)
        for j in idxs[1:]:   # exclude self
            if (i, j) not in set(zip(src_list, dst_list)):
                dist_norm = dists[list(idxs).index(j)] / (NX + NY)
                src_list += [i, j]
                dst_list += [j, i]
                edge_attr_list += [max(corr_matrix[i, j], 0.1),
                                   max(corr_matrix[i, j], 0.1)]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float).unsqueeze(-1)

    # -- Node features ---------------------------------------------------------
    # Normalised position
    x_norm = pos_arr[:, 0:1] / NX
    y_norm = pos_arr[:, 1:2] / NY
    # Max correlation with a neighbour (redundancy proxy)
    corr_max = np.array([corr_matrix[i, :].copy() for i in range(n)])
    np.fill_diagonal(corr_max, 0)
    corr_max_vals = corr_max.max(axis=1, keepdims=True)
    # Normalised node degree
    degree = np.bincount(src_list, minlength=n).reshape(-1, 1).astype(np.float32)
    degree_norm = degree / (degree.max() + 1e-9)

    # Local field variability seen by each sensor.
    # The original docstring advertised variance_T / variance_S among the
    # features, but they were not there: the GNN only saw network geometry,
    # never the ocean. They are added here, standardised separately
    # (var_T ~ 3 degC^2 against var_S ~ 0.03 psu^2: without standardisation
    # salinity vanishes numerically).
    if T is not None and S is not None:
        _, vT, vS = local_variance_map(T, S, positions)
        zT = ((vT - vT.mean()) / (vT.std() + 1e-9)).reshape(-1, 1)
        zS = ((vS - vS.mean()) / (vS.std() + 1e-9)).reshape(-1, 1)
    else:
        zT = np.zeros((n, 1), dtype=np.float32)
        zS = np.zeros((n, 1), dtype=np.float32)

    node_features = np.hstack([x_norm, y_norm, corr_max_vals, degree_norm,
                               zT, zS]).astype(np.float32)
    x_nodes = torch.tensor(node_features, dtype=torch.float)

    graph_dict = {
        "x": x_nodes,                      # (N, 6)
        "feature_names": ["x_norm", "y_norm", "corr_max", "degree",
                          "var_SST", "var_SSS"],
        "edge_index": edge_index,           # (2, E)
        "edge_attr": edge_attr,             # (E, 1)
        "positions": positions,
        "corr_matrix": corr_matrix,
    }
    return graph_dict


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL -- GAT + GraphSAGE
# ══════════════════════════════════════════════════════════════════════════════

class OceanNetworkGAT(nn.Module):
    """
    Graph Attention Network for observing-network analysis.

    Architecture:
        GAT layer 1 (in -> 32, 4 heads) -> GAT layer 2 (128 -> 32, 1 head)
        -> MLP -> per-node score (predicted observability)

    Supervised task:
        For each node (sensor), predict its "local contribution" = the
        reconstruction RMSE improvement it brings (target = normalised LOO
        score, produced by Brick 1 or by a fast proxy).

    The attention weights alpha_{ij} are the main analysis signal:
        high alpha_{ij} -> sensor j strongly influenced by i
        -> potential redundancy when the correlation is also high
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
        self._attention_weights = None   # stored for post-hoc analysis

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
    GraphSAGE in inductive mode, to score new sensors.

    Unlike the transductive GAT (which only sees training nodes), GraphSAGE
    learns generalisable aggregation functions: a new node (glider, Argo
    float) can be inserted into the graph and get its embedding immediately,
    without retraining.

    OED use:
        -> simulate adding a new buoy or glider
        -> predict its marginal contribution without an exhaustive LOO sweep
    """
    def __init__(self, in_dim=4, hidden_dim=64, out_dim=1):
        super().__init__()
        if PYG_AVAILABLE:
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        else:
            # Hand-written mean aggregation
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
#  TARGET GENERATION (fast proxy, no dependency on Brick 1)
# ══════════════════════════════════════════════════════════════════════════════

def compute_proxy_targets(positions, corr_matrix):
    """
    Fast supervision target (no need to load Brick 1):
        contribution_i = 1 - mean(|corr(i, j)|) over j != i

    Reading: a sensor strongly correlated with all its neighbours has a low
    marginal contribution -- it is redundant. This proxy is consistent with
    the theoretical OED definition (marginal entropy reduction in the
    Gaussian case).
    """
    n = len(positions)
    targets = np.zeros(n)
    for i in range(n):
        off_diag = np.delete(corr_matrix[i], i)
        targets[i] = 1.0 - np.mean(np.abs(off_diag))
    # Normalise to [0, 1]
    targets = (targets - targets.min()) / (targets.max() - targets.min() + 1e-9)
    return torch.tensor(targets, dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════════════
#  GNN TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_gnn(args, graph_dict, targets):
    """
    Train the GAT on the node-scoring task.

    Semi-supervised strategy: train on a subset of nodes (train mask) and
    evaluate on the rest -> mimics scoring sensors never seen before.
    """
    print("\n-- GAT training ---------------------------------------------------")
    model = OceanNetworkGAT(in_dim=graph_dict["x"].shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x          = graph_dict["x"].to(DEVICE)
    edge_index = graph_dict["edge_index"].to(DEVICE)
    y          = targets.to(DEVICE)

    n = x.shape[0]
    # Train/test mask (random 80/20 split over nodes)
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
            print(f"  Epoch {epoch:3d} | Train MSE={loss.item():.4f} | "
                  f"Test MSE={test_loss.item():.4f}")
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                torch.save(model.state_dict(), out_dir / "gnn_best.pt")

    print(f"  [ok] Checkpoint -> {out_dir}/gnn_best.pt")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_network(model, graph_dict, targets, args, T=None, label=""):
    """
    Produce the full network diagnostic:
        - predicted node scores (contribution of each sensor)
        - GAT attention weights (inter-sensor redundancy)
        - identification of gap zones (uncovered coarse-grid cells)
        - recommendations: sensors to remove / zones to cover

    T      : nature run (NT, NX, NY) -- if provided, the time-mean SST is used
             as a background for the contribution / redundancy / graph maps.
    label  : filename suffix (e.g. "rl_optimal", "random")
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

    # -- Attention matrix -------------------------------------------------------
    # For each edge the attention weight gives the influence of the source
    # node; attention_matrix[i,j] = mean weight over the edge (i -> j)
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

    # -- Per-node redundancy score (mean pairwise correlation) ------------------
    # Attention can be sparse -> use the direct inter-sensor correlation
    # redundancy_i = mean |corr(i,j)| over j != i  (graph neighbours)
    # uniqueness_i = 1 - redundancy_i
    # A sensor strongly correlated with its neighbours is redundant.
    # A weakly correlated sensor carries unique information -> high uniqueness.
    n = len(positions)
    corr_mat = corr_matrix  # (n, n) correlation, already computed
    redundancy_score = np.zeros(n)
    for i in range(n):
        row = np.abs(corr_mat[i, :])
        row[i] = 0.0                      # exclude self-correlation
        neighbors = np.where(row > 0)[0]  # neighbours with non-zero correlation
        if len(neighbors) > 0:
            redundancy_score[i] = row[neighbors].mean()
        else:
            # Isolated node: no neighbours -> maximum uniqueness by default
            redundancy_score[i] = 0.0

    # Normalise to [0, 1] for comparison with the contribution scores
    r_min, r_max = redundancy_score.min(), redundancy_score.max()
    if r_max > r_min:
        redundancy_score = (redundancy_score - r_min) / (r_max - r_min)
    else:
        # All sensors equally redundant -> uniform uniqueness at 0.5
        redundancy_score = np.full(n, 0.5)

    # -- Spatial coverage (coarse grid) -----------------------------------------
    grid_res = 16
    coverage_grid = np.zeros((NX // grid_res + 1, NY // grid_res + 1))
    for (x_p, y_p) in positions:
        coverage_grid[x_p // grid_res, y_p // grid_res] += 1

    # -- SST background (optional) ----------------------------------------------
    # Time-mean SST -- consistent background image for every panel
    from matplotlib.colors import LinearSegmentedColormap
    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    sst_bg     = T.mean(axis=0) if T is not None else None   # (NX, NY)
    sst_vmin   = sst_bg.min()   if sst_bg is not None else 0
    sst_vmax   = sst_bg.max()   if sst_bg is not None else 1

    def _bg(ax):
        """Draw the mean SST as background + frame."""
        if sst_bg is not None:
            ax.imshow(sst_bg.T, cmap=ocean_cmap, origin="lower", aspect="auto",
                      vmin=sst_vmin, vmax=sst_vmax, alpha=0.45,
                      extent=[0, NX, 0, NY])
        ax.set_xlim(0, NX); ax.set_ylim(0, NY)

    # -- Redundancy threshold ---------------------------------------------------
    # A sensor is "redundant" when its uniqueness falls in the lower quartile
    # (strongly correlated with its neighbours -> low marginal contribution)
    uniqueness = 1 - redundancy_score
    redundancy_thr = np.percentile(uniqueness, 25)   # the 25% least unique
    is_redundant   = uniqueness < redundancy_thr      # (n,) bool

    # -- Visualisation ----------------------------------------------------------
    suffix = f"_{label}" if label else ""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Brick 2 -- GNN: network structure analysis"
                 + (f"  [{label}]" if label else ""),
                 fontsize=14, fontweight="bold")

    def _scatter_on_sst(ax, pos, colors, cmap, vmin, vmax,
                        title, cbar_label, size=130, mark_redundant=False):
        """Coloured scatter over the SST background. Red rings mark redundant buoys."""
        _bg(ax)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=colors,
                        cmap=cmap, s=size, vmin=vmin, vmax=vmax,
                        zorder=5, edgecolors="white", linewidths=0.8)
        # Red ring on redundant buoys
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
            cb_sst.set_label("mean SST (degC)", fontsize=7, color="#555555")
            cb_sst.ax.tick_params(labelsize=6, color="#888888", labelcolor="#555555")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x (pixels)"); ax.set_ylabel("y (pixels)")
        return sc

    # 1. Contribution scores + SST background + redundant markers
    _scatter_on_sst(axes[0, 0], pos_arr,
                    colors=scores, cmap="RdYlGn",
                    vmin=scores.min(), vmax=scores.max(),
                    title="GAT contribution score\n(green = high | red ring = redundant)",
                    cbar_label="Contribution [0-1]",
                    mark_redundant=True)

    # 2. Uniqueness score (1 - redundancy) + SST background + redundancy rings
    _scatter_on_sst(axes[0, 1], pos_arr,
                    colors=uniqueness, cmap="RdYlGn",
                    vmin=0, vmax=1,
                    title="Uniqueness score (1 - redundancy)\n(green = unique | red ring = redundant)",
                    cbar_label="Uniqueness [0-1]",
                    mark_redundant=True)

    # 3. Correlation matrix
    ax = axes[0, 2]
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Spatial correlation matrix\nEdge threshold = {args.corr_threshold}")
    ax.set_xlabel("Sensor index"); ax.set_ylabel("Sensor index")

    # 4. Network graph + SST background + redundancy rings
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
    # Red circles mark redundant sensors
    if is_redundant.any():
        ax.scatter(pos_arr[is_redundant, 0], pos_arr[is_redundant, 1],
                   s=260, facecolors="none", edgecolors="#ff4444",
                   linewidths=2.0, zorder=7, label=f"Redondant ({is_redundant.sum()})")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7, facecolor="#111")
    plt.colorbar(sc_g, ax=ax, pad=0.02, fraction=0.046, label="Contribution")
    for i, (x_p, y_p) in enumerate(positions):
        ax.annotate(f"{i}", (x_p, y_p), fontsize=6, ha="center", va="center",
                    color="black", zorder=6)
    ax.set_title("Network graph\n(edge width ~ GAT attention | red ring = redundant)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # 5. Spatial coverage
    ax = axes[1, 1]
    im = ax.imshow(coverage_grid.T, cmap="Blues", origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Spatial coverage ({grid_res}x{grid_res} px grid)\n"
                 f"(white = uncovered zone -> candidate for a new sensor)")
    ax.set_xlabel(f"x / {grid_res}"); ax.set_ylabel(f"y / {grid_res}")

    # 6. Recommendations: contribution vs redundancy bar plot
    ax = axes[1, 2]
    idx_sorted = np.argsort(scores)[::-1]
    bar_width = 0.35
    x_pos = np.arange(n)
    ax.bar(x_pos - bar_width/2, scores[idx_sorted],
           bar_width, label="Contribution GAT", color="steelblue", alpha=0.8)
    ax.bar(x_pos + bar_width/2, 1 - redundancy_score[idx_sorted],
           bar_width, label="Uniqueness (1 - redundancy)", color="orange", alpha=0.8)
    ax.set_xlabel("Sensors (sorted by contribution)")
    ax.set_ylabel("Score [0, 1]")
    ax.set_title("Contribution vs uniqueness per sensor\n(orange > blue -> removal candidate)")
    ax.legend(fontsize=8)
    ax.set_xticks(x_pos[::3])
    ax.set_xticklabels([f"C{idx_sorted[i]}" for i in range(0, n, 3)], fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "gnn_network_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [ok] Figure -> {out_dir}/gnn_network_analysis.png")

    # -- Text recommendations ---------------------------------------------------
    print("\n-- GNN recommendations --------------------------------------------")
    # Removal candidates: low contribution + high redundancy
    combined_score = scores - redundancy_score
    candidates_remove = np.argsort(combined_score)[:5]
    print(f"  Sensors that are REMOVAL candidates (redundant):")
    for ci in candidates_remove:
        print(f"    C{ci:2d} @ {positions[ci]} | "
              f"contribution={scores[ci]:.3f} | redundancy={redundancy_score[ci]:.3f}")

    # Gap zones
    gaps = np.argwhere(coverage_grid == 0)
    print(f"\n  Gap zones ({len(gaps)} uncovered grid cells):")
    if len(gaps) > 0:
        for gx, gy in gaps[:5]:
            print(f"    Grid cell ({gx}, {gy}) -> "
                  f"pixel centre (~{gx*grid_res+grid_res//2}, ~{gy*grid_res+grid_res//2})")
    if len(gaps) > 5:
        print(f"    ... and {len(gaps) - 5} more zones")

    return scores, redundancy_score, coverage_grid


def inductive_eval(model, graph_dict, new_positions, corr_matrix_orig, args,
                   T=None, S=None):
    """
    Score hypothetical sensors (gliders, Argo floats) unseen during training.

    Procedure:
        1. add the new nodes to the graph (geographic features)
        2. connect them to existing neighbours by kNN
        3. push the extended graph through GraphSAGE
        4. read the new nodes' scores -> predicted contribution

    This is the fundamental advantage of the inductive mode: no retraining.
    """
    print("\n-- Inductive evaluation (new sensors) -----------------------------")
    out_dir = Path(args.output_dir)

    existing_pos = graph_dict["positions"]
    n_existing   = len(existing_pos)
    all_positions = existing_pos + new_positions
    n_all = len(all_positions)

    # New node features -- SAME dimension as build_graph, otherwise the
    # concatenation breaks as soon as the feature set changes.
    n_feat = graph_dict["x"].shape[1]
    if T is not None and S is not None:
        # local variability is known even for a hypothetical sensor: that is
        # precisely what allows predicting its value before deployment
        _, vT_new, vS_new = local_variance_map(T, S, new_positions)
        _, vT_ref, vS_ref = local_variance_map(T, S, list(existing_pos))
        zT_new = (vT_new - vT_ref.mean()) / (vT_ref.std() + 1e-9)
        zS_new = (vS_new - vS_ref.mean()) / (vS_ref.std() + 1e-9)
    else:
        zT_new = np.zeros(len(new_positions), dtype=np.float32)
        zS_new = np.zeros(len(new_positions), dtype=np.float32)

    new_features = []
    for k, (x_p, y_p) in enumerate(new_positions):
        row = [x_p / NX, y_p / NY, 0.5, 0.0, float(zT_new[k]), float(zS_new[k])]
        new_features.append(row[:n_feat] + [0.0] * max(0, n_feat - len(row)))
    new_feat_tensor = torch.tensor(new_features, dtype=torch.float)
    x_extended = torch.cat([graph_dict["x"], new_feat_tensor], dim=0)

    # Connect the new nodes to their k nearest existing neighbours
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

    # Evaluation with GraphSAGE
    sage_model = GraphSAGEInductive(in_dim=x_extended.shape[1]).to(DEVICE)
    # Note: in production, load pre-trained SAGE weights here
    # Here the pipeline is illustrated with an untrained model
    sage_model.eval()
    with torch.no_grad():
        scores_all = sage_model(x_extended.to(DEVICE), edge_ext.to(DEVICE))
    scores_new = scores_all[n_existing:].cpu().numpy()

    print(f"  Predicted scores for {len(new_positions)} new sensors:")
    for i, (pos, sc) in enumerate(zip(new_positions, scores_new)):
        print(f"    New sensor @ {pos} -> score = {sc:.4f}")

    # Visualisation
    fig, ax = plt.subplots(figsize=(8, 6))
    ex_arr = np.array(existing_pos)
    ax.scatter(ex_arr[:, 0], ex_arr[:, 1],
               c="steelblue", s=80, label="Existing sensors", zorder=5)
    new_arr = np.array(new_positions)
    sc = ax.scatter(new_arr[:, 0], new_arr[:, 1],
                    c=scores_new, cmap="RdYlGn", s=200,
                    marker="*", edgecolors="black", linewidths=1,
                    label="New sensors (score)", zorder=6)
    plt.colorbar(sc, ax=ax, label="Predicted contribution score")
    ax.set_xlim(0, NX); ax.set_ylim(0, NY)
    ax.set_title("Inductive evaluation of new sensors\n(star = hypothetical glider / Argo float)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.savefig(out_dir / "gnn_inductive_eval.png", dpi=150)
    plt.close()
    print(f"  [ok] Figure -> {out_dir}/gnn_inductive_eval.png")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Brick 2 -- GNN")
    p.add_argument("--train",          action="store_true")
    p.add_argument("--analyze",        action="store_true")
    p.add_argument("--inductive",      action="store_true")
    p.add_argument("--report",         action="store_true",
                   help="Write a .txt report with the key metrics")
    p.add_argument("--seed_ocean",     type=int,   default=42)
    p.add_argument("--seed_buoys",     type=int,   default=7)
    p.add_argument("--new_positions",  type=str, default="[(10,20),(80,150),(130,40)]")
    p.add_argument("--nt",             type=int,   default=500,
                   help="Nature run length (days)")
    p.add_argument("--corr_threshold", type=float, default=GNN_CORR_THRESHOLD)
    p.add_argument("--deseason",       type=int,   default=int(DESEASON_ANALYSIS),
                   help="1 = remove the seasonal cycle before correlating")
    p.add_argument("--k_nearest",      type=int,   default=4)
    p.add_argument("--gnn_epochs",     type=int,   default=200)
    p.add_argument("--output_dir",     type=str,   default="outputs")
    p.add_argument("--n_buoys",        type=int,   default=N_BUOYS)
    return p.parse_args()


if __name__ == "__main__":
    from datetime import datetime
    args = parse_args()

    if not (args.train or args.analyze or args.inductive):
        print("Usage: python 02_gnn.py --train --analyze [--inductive] [--report]")
        sys.exit(0)

    print("═" * 60)
    print(" Brick 2 -- GNN: observing-network structure")
    print("═" * 60)

    print(f"\n[1/3] Nature run generation (seed_ocean={args.seed_ocean}, nt={args.nt})...")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=args.nt, seed=args.seed_ocean)

    rng = np.random.default_rng(args.seed_buoys)
    if args.nt < 365:
        print(f"  [WARNING] nt={args.nt} < 365: incomplete seasonal "
              f"cycle, biased statistics.")

    positions = [(int(rng.integers(0, NX)), int(rng.integers(0, NY)))
                 for _ in range(args.n_buoys)]
    print(f"      {args.n_buoys} sensors (seed_buoys={args.seed_buoys}, "
          f"separation >= {MIN_BUOY_SEP_KM:.0f} km)")

    print("\n[2/3] Computing the spatial correlation matrix...")
    corr_matrix = build_spatial_correlation(T, S, positions,
                                            n_timestamps=min(300, args.nt),
                                            deseason=bool(args.deseason))
    _off = corr_matrix[~np.eye(len(positions), dtype=bool)]
    print(f"      mean |rho| = {np.abs(_off).mean():.3f} | "
          f"pairs above threshold = {(np.abs(_off) > args.corr_threshold).mean():.1%}"
          f"  (deseason={bool(args.deseason)})")

    print(f"\n[3/3] Building the graph (threshold={args.corr_threshold}, k={args.k_nearest})...")
    graph_dict = build_graph(positions, corr_matrix,
                             corr_threshold=args.corr_threshold,
                             k_nearest=args.k_nearest,
                             T=T, S=S)
    n_edges = graph_dict["edge_index"].shape[1]
    print(f"      Nodes : {len(positions)} | Edges : {n_edges}")
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
                print(f"  Model loaded from {ckpt_path}")
        scores_out, redund_out, _ = analyze_network(
            model, graph_dict, targets, args, T=T)

    if args.inductive:
        if model is None:
            model = OceanNetworkGAT(in_dim=graph_dict["x"].shape[1]).to(DEVICE)
        try:
            new_positions = ast.literal_eval(args.new_positions)
        except Exception:
            new_positions = [(10, 20), (80, 150), (130, 40)]
        inductive_eval(model, graph_dict, new_positions, corr_matrix, args,
                       T=T, S=S)

    if args.report:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output_dir)
        uniqueness = (1 - redund_out) if redund_out is not None else None
        is_redundant = (uniqueness < np.percentile(uniqueness, 25)) if uniqueness is not None else None
        lines = [
            "=" * 68,
            "  Brick 2 -- GNN -- Report",
            f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 68, "",
            "-- REPRODUCIBILITY --------------------------------------------------",
            f"  seed_ocean     : {args.seed_ocean}",
            f"  seed_buoys     : {args.seed_buoys}",
            f"  n_buoys        : {args.n_buoys}",
            f"  corr_threshold : {args.corr_threshold}",
            f"  k_nearest      : {args.k_nearest}",
            "",
            "-- BUOY POSITIONS ---------------------------------------------------",
        ] + [f"  B{i:02d} : ({px:4d}, {py:4d})" for i, (px, py) in enumerate(positions)] + [
            "",
            "-- GRAPH ------------------------------------------------------------",
            f"  Nodes   : {len(positions)}",
            f"  Edges   : {n_edges}",
        ]
        if scores_out is not None:
            lines += [
                f"  Mean contribution      : {scores_out.mean():.3f} +/- {scores_out.std():.3f}",
                f"  Mean redundancy        : {redund_out.mean():.3f}",
                f"  Redundant sensors      : {int(is_redundant.sum())}  (uniqueness Q25)",
                f"  Redundant IDs          : {[int(i) for i in np.where(is_redundant)[0]]}",
            ]
        lines += ["", "-- FILES PRODUCED ---------------------------------------------------"]
        for f in sorted(out.iterdir()):
            if f.suffix in {".pt", ".png"}:
                lines.append(f"  {f.name:<44} {f.stat().st_size//1024:>5} KB")
        lines += ["", "=" * 68]
        rpt = out / f"report_gnn_{ts}.txt"
        rpt.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  GNN report -> {rpt}")

    print("\n  [ok] Brick 2 done.")
