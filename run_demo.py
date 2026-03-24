"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OED-IA pour SNO Marins — Orchestrateur                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  --mode individual : briques indépendantes (AE → GNN → RL)                 ║
║  --mode pipeline   : RL → GNN → AE (réseau optimisé)                       ║
║                                                                             ║
║  Usage :                                                                    ║
║    python run_demo.py --mode individual                                     ║
║    python run_demo.py --mode pipeline --seed_ocean 42 --seed_buoys 7       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, argparse, time, json, importlib.util, types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from config import DEVICE, NX, NY, NT, N_BUOYS, set_global_seed, make_output_dir
try:
    from dataset import SyntheticOceanGenerator, build_datasets
except ModuleNotFoundError:
    from data.dataset import SyntheticOceanGenerator, build_datasets


# ══════════════════════════════════════════════════════════════════════════════
#  Arguments
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="OED-IA Orchestrateur")
    p.add_argument("--mode",        choices=["individual", "pipeline"], default="individual")
    p.add_argument("--seed_ocean",  type=int, default=42)
    p.add_argument("--seed_buoys",  type=int, default=7)
    p.add_argument("--nt",          type=int, default=200)
    p.add_argument("--n_buoys",     type=int, default=None)
    p.add_argument("--eval_light",  action="store_true",
                   help="Pipeline : évalue aussi la config légère (N★/2) avec GNN + AE")
    p.add_argument("--rl_method",   choices=["pareto", "efficiency", "scalarized"],
                   default="pareto", help="Méthode de sélection N★ pour le RL")
    # AE
    p.add_argument("--ae_epochs",   type=int, default=5)
    p.add_argument("--ae_base_ch",  type=int, default=16)
    # GNN
    p.add_argument("--gnn_epochs",  type=int, default=30)
    # RL
    p.add_argument("--rl_steps",    type=int, default=2000)
    p.add_argument("--rl_grid_x",   type=int, default=8)
    p.add_argument("--rl_grid_y",   type=int, default=12)
    p.add_argument("--rl_n_min",    type=int, default=5)
    p.add_argument("--rl_n_max",    type=int, default=20)
    p.add_argument("--rl_episode_len", type=int, default=20)
    p.add_argument("--gif_frames",  type=int, default=40)
    p.add_argument("--output_dir",  type=str, default="outputs")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_brick(filename):
    spec = importlib.util.spec_from_file_location(filename.stem, filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_report(path, sections):
    lines = []
    for s in sections:
        lines.extend(s if isinstance(s, list) else [s])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Rapport → {path}")


def plot_ocean_overview(T, S, positions, out_dir, seed_ocean=42, seed_buoys=7):
    """
    Figure d'illustration : SST et SSS à 3 instants + réseau de bouées.
    Layout 2×3 : ligne 1 = SST, ligne 2 = SSS ; colonnes = t=0, t=mid, t=fin.
    Bouées affichées sur chaque panneau.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    sal_cmap = LinearSegmentedColormap.from_list("sal",
        ["#003c30","#01665e","#35978f","#80cdc1","#f5f5f5",
         "#dfc27d","#bf812d","#8c510a","#543005"], N=256)
    BG = "#0a1628"

    nt = len(T)
    snaps = [0, nt // 2, nt - 1]
    obs = np.array(positions)
    vT = (T.min(), T.max())
    vS = (S.min(), S.max())

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.22,
                           left=0.04, right=0.97, top=0.90, bottom=0.04)

    for col, t in enumerate(snaps):
        for row, (field, cmap, vmin, vmax, var_name, unit) in enumerate([
            (T, ocean_cmap, *vT, "SST", "°C"),
            (S, sal_cmap,   *vS, "SSS", "psu"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(field[t].T, cmap=cmap, origin="lower", aspect="auto",
                           vmin=vmin, vmax=vmax, interpolation="bilinear")
            # Bouées
            ax.scatter(obs[:, 0], obs[:, 1], c="white", s=28,
                       edgecolors="black", linewidths=0.6, zorder=6)
            ax.scatter(obs[:, 0], obs[:, 1], c="#ffd93d", s=8, zorder=7)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("#050d1a")
            for sp in ax.spines.values():
                sp.set_edgecolor("#1a3a5c")
            ax.set_title(f"{var_name}  t={t}", color="white", fontsize=10,
                         fontweight="bold", pad=5)
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label(unit, color="white", fontsize=7)
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)

    fig.text(0.5, 0.96,
             f"Nature Run — {len(positions)} bouées  "
             f"(seed_ocean={seed_ocean}  seed_buoys={seed_buoys}  nt={nt})",
             ha="center", color="white", fontsize=13, fontweight="bold")

    out_path = Path(out_dir) / "ocean_overview.png"
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Figure océan + bouées → {out_path}")


def _train_ae_quick(b1, T, S, args, ae_ns):
    """Entraînement AE minimal."""
    train_ds, val_ds = build_datasets(T, S, split=0.8,
                                      n_obs_min=ae_ns.n_obs_min,
                                      n_obs_max=ae_ns.n_obs_max,
                                      augment_train=True)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    model  = b1.ObservabilityAE(
        base_ch=ae_ns.base_ch, latent_ch=ae_ns.latent_ch,
        dropout_p=ae_ns.dropout_p, cond_dim=ae_ns.cond_dim).to(DEVICE)
    optim  = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit   = b1.AELoss(w_unobs=ae_ns.w_unobs, lambda_grad=ae_ns.lambda_grad,
                        huber_delta=ae_ns.huber_delta)

    best_loss = np.inf
    t0 = time.time()
    model.train()
    for ep in range(ae_ns.epochs):
        ep_loss = 0.0
        for x, y, mask in loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            pred, z, aux = model(x)
            loss, *_ = crit(pred, y, mask, aux_preds=aux)
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)
        best_loss = min(best_loss, ep_loss)
        print(f"    ep {ep+1}/{ae_ns.epochs} | Loss={ep_loss:.4f}")

    # RMSE validation MC
    model.eval()
    val_ld = DataLoader(val_ds, batch_size=8, shuffle=False)
    rmses = []
    with torch.no_grad():
        for x, y, mask in val_ld:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            preds = torch.stack([model(x)[0] for _ in range(ae_ns.n_mc_val)])
            pm = preds.mean(0)
            for b in range(x.shape[0]):
                rmses.append(float(torch.sqrt(((pm[b] - y[b])**2 * (1 - mask[b])).mean()).item()))

    val_rmse = float(np.mean(rmses))
    elapsed  = round(time.time() - t0, 1)
    norm = {"T_mean": float(T.mean()), "T_std": float(T.std()),
            "S_mean": float(S.mean()), "S_std": float(S.std())}
    torch.save({"model_state": model.state_dict(), "args": vars(ae_ns), "norm": norm},
               Path(ae_ns.output_dir) / "ae_best.pt")
    return model, norm, best_loss, val_rmse, elapsed


# ══════════════════════════════════════════════════════════════════════════════
#  Rapport
# ══════════════════════════════════════════════════════════════════════════════

SEP = "─" * 68

def _report_header(mode, args, T, positions, ts):
    return [
        "=" * 68,
        "  OED-IA SNO Marins — Rapport",
        f"  Mode     : {mode}",
        f"  Généré le: {ts}",
        "=" * 68, "",
        "── REPRODUCTIBILITÉ ─────────────────────────────────────────────────",
        f"  seed_ocean  : {args.seed_ocean}",
        f"  seed_buoys  : {args.seed_buoys}",
        f"  nt          : {args.nt}  pas de temps",
        f"  n_buoys     : {len(positions)}  capteurs",
        "",
        "── NATURE RUN ───────────────────────────────────────────────────────",
        f"  SST : [{T.min():.2f}, {T.max():.2f}] °C",
        "",
        "── POSITIONS (pixel x, y) ──────────────────────────────────────────",
    ] + [f"  B{i:02d} : ({px:4d}, {py:4d})" for i, (px, py) in enumerate(positions)]

def _report_ae(m):
    return ["", "── BRIQUE 1 — AE-UNet MC-Dropout ────────────────────────────────────",
            f"  Loss train (best)   : {m['ae_best_loss']:.4f}",
            f"  RMSE_val (normalisé): {m['ae_rmse_val']:.4f}",
            f"  RMSE_val physique   : {m['ae_rmse_phys']:.3f} °C",
            f"  Temps               : {m['ae_time']} s"]

def _report_gnn(m):
    lines = ["", "── BRIQUE 2 — GNN ───────────────────────────────────────────────────",
             f"  Arêtes graphe          : {m['gnn_edges']}",
             f"  Score contrib. moy±std : {m['gnn_score_mean']:.3f} ± {m['gnn_score_std']:.3f}",
             f"  Redondance moyenne     : {m['gnn_redond_mean']:.3f}",
             f"  Capteurs redondants    : {m['gnn_n_redondant']}  (unicité Q25)",
             f"  Temps                  : {m['gnn_time']} s"]
    if m.get("gnn_redundant_ids"):
        lines.append(f"  IDs redondants         : {m['gnn_redundant_ids']}")
    return lines

def _report_rl(m):
    method = m.get("rl_method", "pareto").upper()
    return ["", f"── BRIQUE 3 — RL [{method}] ──────────────────────────────────────────",
            f"  N★ optimal           : {m['rl_n_star']} capteurs",
            f"  Score info N★        : {m['rl_info_star']:.3f}",
            f"  Score info max       : {m['rl_info_max']:.3f}",
            f"  Config légère N      : {m['rl_n_light']} capteurs",
            f"  Perte info dense→lég : {m['rl_perte_pct']:.1f} %",
            f"  Temps                : {m['rl_time']} s"]

def _report_footer(mode, total, args, metrics, out_dir):
    lines = ["", "── RÉSUMÉ ───────────────────────────────────────────────────────────",
             f"  Mode        : {mode}",
             f"  Temps total : {round(total, 1)} s", "",
             "── FICHIERS PRODUITS ────────────────────────────────────────────────"]
    for f in sorted(Path(out_dir).iterdir()):
        if f.suffix in {".pt", ".png", ".gif", ".txt"}:
            lines.append(f"  {f.name:<46} {f.stat().st_size // 1024:>5} KB")
    lines += ["", json.dumps({"seed_ocean": args.seed_ocean, "seed_buoys": args.seed_buoys,
                              "nt": args.nt, "mode": mode}, indent=2), "=" * 68]
    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  MODE INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════

def _run_individual(args, T, S, positions, b1, b2, b3,
                    ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0):

    print(f"\n{SEP}\n  MODE INDIVIDUAL — 3 briques indépendantes\n{SEP}")

    # Brique 1 — AE
    print(f"\n{SEP}\n  BRIQUE 1 — AE-UNet MC-Dropout\n{SEP}")
    model_ae, norm, best_loss, val_rmse, ae_time = _train_ae_quick(b1, T, S, args, ae_ns)
    ae_fig_ns = types.SimpleNamespace(**vars(ae_ns), figures=True)
    ae_fig_ns.output_dir = str(out)
    model_ae.eval()
    b1.plot_network_evaluation(model_ae, T, S, norm, ae_fig_ns, positions=positions, n_samples=ae_ns.n_mc)
    b1.plot_uncertainty_maps(model_ae, T, S, norm, ae_fig_ns, n_samples=ae_ns.n_mc)
    m_ae = {"ae_best_loss": float(best_loss), "ae_rmse_val": val_rmse,
            "ae_rmse_phys": val_rmse * float(T.std()), "ae_time": ae_time}
    metrics.update(m_ae); report_sections += _report_ae(m_ae)
    print(f"  ✓ AE RMSE_val={val_rmse:.4f} ({val_rmse*T.std():.3f} °C) [{ae_time}s]")

    # Brique 2 — GNN
    print(f"\n{SEP}\n  BRIQUE 2 — GNN Structure\n{SEP}")
    t0_gnn = time.time()
    corr  = b2.build_spatial_correlation(T, S, positions, n_timestamps=min(80, args.nt))
    graph = b2.build_graph(positions, corr, corr_threshold=0.5, k_nearest=4, T=T, S=S)
    tgts  = b2.compute_proxy_targets(positions, corr)
    print(f"  Graphe : {len(positions)} nœuds, {graph['edge_index'].shape[1]} arêtes")
    model_gnn = b2.train_gnn(gnn_ns, graph, tgts)
    scores_gnn, redund, _ = b2.analyze_network(model_gnn, graph, tgts, gnn_ns, T=T)
    gnn_time = round(time.time() - t0_gnn, 1)
    unicite  = 1 - redund
    is_redond = unicite < np.percentile(unicite, 25)
    m_gnn = {"gnn_edges": int(graph['edge_index'].shape[1]),
             "gnn_score_mean": float(scores_gnn.mean()), "gnn_score_std": float(scores_gnn.std()),
             "gnn_redond_mean": float(redund.mean()), "gnn_n_redondant": int(is_redond.sum()),
             "gnn_redundant_ids": [int(i) for i in np.where(is_redond)[0]], "gnn_time": gnn_time}
    metrics.update(m_gnn); report_sections += _report_gnn(m_gnn)
    print(f"  ✓ GNN {m_gnn['gnn_n_redondant']} redondants [{gnn_time}s]")

    # Brique 3 — RL
    print(f"\n{SEP}\n  BRIQUE 3 — RL [{rl_ns.rl_method.upper()}]\n{SEP}")
    t0_rl = time.time()
    env   = b3.OceanNetworkEnv(T, S, grid_x=rl_ns.grid_x, grid_y=rl_ns.grid_y,
                                n_min=rl_ns.n_min, n_max=rl_ns.n_max, episode_len=rl_ns.episode_len)
    policy, _ = b3.train_ppo(rl_ns, env)
    pts, n_star = b3.run_rl_method(env, policy, rl_ns)
    b3.visualize_two_configs(env, n_star, policy, rl_ns)
    b3.save_rl_gif(env, policy, rl_ns, n_frames=rl_ns.gif_frames)
    rl_time = round(time.time() - t0_rl, 1)
    info_vals = np.array([p["info_mean"] for p in pts])
    n_vals    = np.array([p["n_buoys"] for p in pts])
    n_star    = int(np.clip(n_star, max(2, env.n_min), env.n_max))
    nl = b3._n_light(n_star)
    info_star  = float(info_vals[np.argmin(np.abs(n_vals - n_star))])
    info_light = float(info_vals[np.argmin(np.abs(n_vals - nl))]) if len(n_vals) > 0 else 0.0
    perte_pct  = max(0, (info_star - info_light) / max(info_star, 1e-3) * 100)
    m_rl = {"rl_n_star": n_star, "rl_info_star": info_star, "rl_info_max": float(info_vals.max()),
            "rl_n_light": nl, "rl_info_light": info_light, "rl_perte_pct": perte_pct,
            "rl_time": rl_time, "rl_method": rl_ns.rl_method}
    metrics.update(m_rl); report_sections += _report_rl(m_rl)

    total = time.time() - t0
    report_sections += _report_footer("individual", total, args, metrics, str(out))
    write_report(out / f"rapport_individual_{ts}.txt", report_sections)
    _print_summary("individual", args, m_ae, m_gnn, m_rl, total)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE PIPELINE : RL → GNN → AE
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(args, T, S, init_pos, b1, b2, b3,
                  ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0):

    print(f"\n{SEP}\n  MODE PIPELINE : RL → GNN → AE\n{SEP}")

    # Étape 1 : RL
    print(f"\n{SEP}\n  ÉTAPE 1/3 — RL [{rl_ns.rl_method.upper()}]\n{SEP}")
    t0_rl = time.time()
    env    = b3.OceanNetworkEnv(T, S, grid_x=rl_ns.grid_x, grid_y=rl_ns.grid_y,
                                 n_min=rl_ns.n_min, n_max=rl_ns.n_max, episode_len=rl_ns.episode_len)
    policy, _ = b3.train_ppo(rl_ns, env)
    pts, n_star = b3.run_rl_method(env, policy, rl_ns)

    # Extraction des positions du meilleur checkpoint
    best_ckpt = torch.load(Path(rl_ns.output_dir) / "rl_best.pt", map_location="cpu", weights_only=False)
    best_mask = best_ckpt["active_mask"]
    active_idx   = np.where(best_mask > 0.5)[0] if len(best_mask) > 0 else np.array([], dtype=int)
    rl_positions = [env.candidate_positions[i] for i in active_idx] if len(active_idx) > 0 else []

    # Si scalarized, le best_mask peut être vide → utiliser la politique directement
    if len(rl_positions) == 0:
        active_idx, info_retained = b3._run_policy_config(env, policy, int(n_star))
        rl_positions = [env.candidate_positions[i] for i in active_idx]
        best_mask = env.active_mask.copy()
    else:
        env.active_mask = best_mask.copy()
        info_retained = float(env._compute_info_reward())
    print(f"  Réseau RL : {len(rl_positions)} bouées | info={info_retained:.3f}")

    b3.visualize_two_configs(env, n_star, policy, rl_ns, best_mask=best_mask)
    b3.mark_retained_config_on_pareto(len(rl_positions), info_retained, rl_ns.output_dir)
    b3.save_rl_gif(env, policy, rl_ns, n_frames=rl_ns.gif_frames)
    rl_time = round(time.time() - t0_rl, 1)

    info_vals = np.array([p["info_mean"] for p in pts])
    n_vals    = np.array([p["n_buoys"] for p in pts])
    n_star = int(np.clip(n_star, max(2, env.n_min), env.n_max))
    nl = b3._n_light(n_star)
    info_star  = float(info_vals[np.argmin(np.abs(n_vals - n_star))]) if len(n_vals) > 0 else info_retained
    info_light = float(info_vals[np.argmin(np.abs(n_vals - nl))]) if len(n_vals) > 0 else 0.0
    perte_pct  = max(0, (info_star - info_light) / max(info_star, 1e-3) * 100)

    # Complétion si trop peu de positions pour le GNN
    GNN_MIN = 5
    if len(rl_positions) < GNN_MIN:
        print(f"  [INFO] Réseau RL ({len(rl_positions)}) < {GNN_MIN} — complétion aléatoire")
        extra_pool = [p for p in env.candidate_positions if p not in rl_positions]
        extra = list(np.random.default_rng(args.seed_buoys).choice(
            len(extra_pool), GNN_MIN - len(rl_positions), replace=False))
        rl_positions += [extra_pool[e] for e in extra]

    m_rl = {"rl_n_star": n_star, "rl_info_star": info_star, "rl_info_max": float(info_vals.max()) if len(info_vals) else info_star,
            "rl_n_light": nl, "rl_info_light": info_light, "rl_perte_pct": perte_pct,
            "rl_time": rl_time, "rl_method": rl_ns.rl_method}
    metrics.update(m_rl); report_sections += _report_rl(m_rl)
    report_sections += ["", "── POSITIONS OPTIMALES RL ───────────────────────────────────────────"]
    report_sections += [f"  R{i:02d} : ({px:4d}, {py:4d})" for i, (px, py) in enumerate(rl_positions)]

    # Étape 2 : GNN sur réseau RL
    print(f"\n{SEP}\n  ÉTAPE 2/3 — GNN\n{SEP}")
    t0_gnn = time.time()
    corr  = b2.build_spatial_correlation(T, S, rl_positions, n_timestamps=min(80, args.nt))
    graph = b2.build_graph(rl_positions, corr, corr_threshold=0.5, k_nearest=4, T=T, S=S)
    tgts  = b2.compute_proxy_targets(rl_positions, corr)
    print(f"  Graphe : {len(rl_positions)} nœuds, {graph['edge_index'].shape[1]} arêtes")
    model_gnn = b2.train_gnn(gnn_ns, graph, tgts)
    scores_gnn, redund, _ = b2.analyze_network(model_gnn, graph, tgts, gnn_ns, T=T, label="rl_optimal")
    gnn_time = round(time.time() - t0_gnn, 1)
    unicite = 1 - redund
    is_redond = unicite < np.percentile(unicite, 25)
    m_gnn = {"gnn_edges": int(graph['edge_index'].shape[1]),
             "gnn_score_mean": float(scores_gnn.mean()), "gnn_score_std": float(scores_gnn.std()),
             "gnn_redond_mean": float(redund.mean()), "gnn_n_redondant": int(is_redond.sum()),
             "gnn_redundant_ids": [int(i) for i in np.where(is_redond)[0]], "gnn_time": gnn_time}
    metrics.update(m_gnn); report_sections += _report_gnn(m_gnn)

    # Étape 3 : AE sur réseau RL
    print(f"\n{SEP}\n  ÉTAPE 3/3 — AE\n{SEP}")
    model_ae, norm, best_loss, val_rmse, ae_time = _train_ae_quick(b1, T, S, args, ae_ns)
    ae_fig_ns = types.SimpleNamespace(**vars(ae_ns), figures=True)
    ae_fig_ns.output_dir = str(out)
    model_ae.eval()
    b1.plot_network_evaluation(model_ae, T, S, norm, ae_fig_ns, positions=rl_positions, n_samples=ae_ns.n_mc)
    b1.plot_uncertainty_maps(model_ae, T, S, norm, ae_fig_ns, n_samples=ae_ns.n_mc)
    m_ae = {"ae_best_loss": float(best_loss), "ae_rmse_val": val_rmse,
            "ae_rmse_phys": val_rmse * float(T.std()), "ae_time": ae_time}
    metrics.update(m_ae); report_sections += _report_ae(m_ae)

    # ── Optionnel : évaluation config légère (N★/2) avec GNN + AE ────────
    if getattr(args, 'eval_light', False):
        print(f"\n{SEP}\n  BONUS — Config légère (N★/2) évaluée par GNN + AE\n{SEP}")
        n_light_eval = max(2, n_star // 2)
        if n_light_eval >= n_star:
            n_light_eval = max(2, n_star - max(3, n_star // 3))
        # Générer la config légère avec la politique entraînée
        env.active_mask[:] = 0.0
        env.active_mask[np.random.choice(env.K, min(n_light_eval, env.K), replace=False)] = 1.0
        obs_l = env._get_obs()
        policy.eval()
        with torch.no_grad():
            for _ in range(env.ep_len):
                obs_t = torch.tensor(obs_l, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                act_l, _, _, _ = policy.get_action(obs_t, deterministic=True)
                obs_l, _, done_l, _ = env.step(act_l.item())
                if done_l:
                    break
        light_idx = np.where(env.active_mask > 0.5)[0]
        light_positions = [env.candidate_positions[i] for i in light_idx]
        light_info = float(env._compute_info_reward())
        print(f"  Config légère : {len(light_positions)} bouées | info={light_info:.3f}")

        if len(light_positions) >= GNN_MIN:
            # GNN sur config légère
            corr_l  = b2.build_spatial_correlation(T, S, light_positions, n_timestamps=min(80, args.nt))
            graph_l = b2.build_graph(light_positions, corr_l, corr_threshold=0.5, k_nearest=4, T=T, S=S)
            tgts_l  = b2.compute_proxy_targets(light_positions, corr_l)
            print(f"  Graphe léger : {len(light_positions)} nœuds, {graph_l['edge_index'].shape[1]} arêtes")
            model_gnn_l = b2.train_gnn(gnn_ns, graph_l, tgts_l)
            b2.analyze_network(model_gnn_l, graph_l, tgts_l, gnn_ns, T=T, label="rl_light")

            # AE sur config légère
            b1.plot_network_evaluation(model_ae, T, S, norm, ae_fig_ns,
                                       positions=light_positions, n_samples=ae_ns.n_mc)
            print(f"  ✓ Config légère évaluée : {len(light_positions)} bouées")

            report_sections += [
                "", "── CONFIG LÉGÈRE (N★/2) ─────────────────────────────────────────────",
                f"  N bouées        : {len(light_positions)}",
                f"  Score info      : {light_info:.3f}",
            ]
        else:
            print(f"  [SKIP] Config légère trop petite ({len(light_positions)} < {GNN_MIN})")

    total = time.time() - t0
    report_sections += _report_footer("pipeline", total, args, metrics, str(out))
    write_report(out / f"rapport_pipeline_{ts}.txt", report_sections)
    _print_summary("pipeline", args, m_ae, m_gnn, m_rl, total)


# ══════════════════════════════════════════════════════════════════════════════
#  Résumé console
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(mode, args, m_ae, m_gnn, m_rl, total):
    print(f"\n{'='*68}")
    print(f"  ✓ {mode} terminé ({total:.0f}s)")
    print(f"  seed_ocean={args.seed_ocean}  seed_buoys={args.seed_buoys}")
    print(f"  AE  RMSE_val={m_ae['ae_rmse_val']:.4f} ({m_ae['ae_rmse_phys']:.3f} °C)")
    print(f"  GNN {m_gnn['gnn_n_redondant']} redondants | score={m_gnn['gnn_score_mean']:.3f}")
    print(f"  RL  [{m_rl.get('rl_method','pareto').upper()}] N★={m_rl['rl_n_star']} | "
          f"info={m_rl['rl_info_star']:.3f} | perte {m_rl['rl_perte_pct']:.1f}%")
    print(f"{'='*68}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    t0     = time.time()
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_buoys = args.n_buoys or N_BUOYS

    # Répertoire de sortie horodaté avec seeds
    out = make_output_dir(base=args.output_dir, seed_ocean=args.seed_ocean,
                          seed_buoys=args.seed_buoys, mode=args.mode)

    # Seed globale
    set_global_seed(args.seed_ocean)

    print("=" * 68)
    print(f"  OED-IA SNO Marins | mode={args.mode}")
    print(f"  seed_ocean={args.seed_ocean}  seed_buoys={args.seed_buoys}  nt={args.nt}")
    print("=" * 68)

    # Nature run commun
    print(f"\n{SEP}\n  Nature Run (seed={args.seed_ocean}, nt={args.nt})\n{SEP}")
    gen  = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=args.nt, seed=args.seed_ocean)
    print(f"  T : {T.shape}  [{T.min():.2f}, {T.max():.2f}] °C")

    rng      = np.random.default_rng(args.seed_buoys)
    init_pos = [(int(rng.integers(0, NX)), int(rng.integers(0, NY))) for _ in range(n_buoys)]
    print(f"  Réseau initial : {n_buoys} bouées (seed_buoys={args.seed_buoys})")

    # Figure d'illustration océan + bouées
    plot_ocean_overview(T, S, init_pos, str(out),
                        seed_ocean=args.seed_ocean, seed_buoys=args.seed_buoys)

    metrics = {"positions": init_pos}

    # Chargement des briques
    brick_dir = Path(__file__).parent
    b1 = load_brick(brick_dir / "01_autoencoder.py")
    b2 = load_brick(brick_dir / "02_gnn.py")
    b3 = load_brick(brick_dir / "03_rl.py")

    # Namespaces
    ae_ns = types.SimpleNamespace(
        epochs=args.ae_epochs, batch_size=8, lr=3e-4,
        base_ch=args.ae_base_ch, latent_ch=32, cond_dim=16, dropout_p=0.15,
        w_unobs=4.0, lambda_grad=0.5, huber_delta=0.5,
        n_obs_min=10, n_obs_max=60, n_mc_val=3, n_mc=20,
        output_dir=str(out), checkpoint=str(out / "ae_best.pt"),
        seed_ocean=args.seed_ocean, seed_buoys=args.seed_buoys)

    gnn_ns = types.SimpleNamespace(
        gnn_epochs=args.gnn_epochs, output_dir=str(out), corr_threshold=0.5)

    rl_ns = types.SimpleNamespace(
        rl_steps=args.rl_steps, buffer_size=512, lr=3e-4, output_dir=str(out),
        grid_x=args.rl_grid_x, grid_y=args.rl_grid_y,
        n_min=args.rl_n_min, n_max=args.rl_n_max,
        episode_len=args.rl_episode_len, w_info=1.0, w_budget=0.5,
        gif_frames=args.gif_frames, rl_method=args.rl_method)

    report_sections = _report_header(args.mode, args, T, init_pos, ts)

    if args.mode == "individual":
        _run_individual(args, T, S, init_pos, b1, b2, b3,
                        ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0)
    else:
        _run_pipeline(args, T, S, init_pos, b1, b2, b3,
                      ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0)


if __name__ == "__main__":
    main()
