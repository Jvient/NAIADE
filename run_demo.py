"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   NAIADE -- OED-AI for marine SNOs -- Orchestrator                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Two execution modes:                                                        ║
║                                                                              ║
║  --mode individual   (default)                                               ║
║    Each brick runs independently on the same nature run.                     ║
║    Useful to compare brick outputs side by side.                             ║
║    Order: AE -> GNN -> RL                                                    ║
║                                                                              ║
║  --mode pipeline                                                             ║
║    1. RL proposes an optimal network (N* buoys)                              ║
║    2. GNN scores that network (structure + redundancy)                       ║
║    3. AE scores that network (gap zones + 3 proposed buoys)                  ║
║    -> Bricks 2 and 3 work on the SAME RL network.                            ║
║                                                                              ║
║  In both cases: .txt report + reproducibility JSON.                          ║
║                                                                              ║
║  Usage:                                                                      ║
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
from config import *
from data.dataset import animate_nature_run
from data.dataset import (SyntheticOceanGenerator, build_datasets,
                          mesoscale_anomaly, plot_nature_run,
                          sample_separated_positions)


# ══════════════════════════════════════════════════════════════════════════════
#  Arguments
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NAIADE orchestrator (individual|pipeline)")
    p.add_argument("--mode",        choices=["individual", "pipeline"],
                   default="individual",
                   help="individual: independent bricks | pipeline: RL -> GNN -> AE")
    p.add_argument("--seed_ocean",  type=int, default=42,
                   help="Nature run seed (reproducibility)")
    p.add_argument("--seed_buoys",  type=int, default=7,
                   help="Seed of the initial buoy network")
    p.add_argument("--nt",          type=int, default=365,
                   help="Nature run length in days (>= 365 recommended: below "
                        "that the seasonal cycle is not fully sampled)")
    p.add_argument("--n_buoys",     type=int, default=None,
                   help="Number of buoys (default = config.N_BUOYS)")
    # AE
    p.add_argument("--ae_epochs",   type=int, default=5)
    p.add_argument("--ae_base_ch",  type=int, default=16)
    p.add_argument("--n_proposed",  type=int, default=3,
                   help="number of new buoys the AE proposes from the gap map, "
                        "and therefore the number the GNN then scores")
    p.add_argument("--gap_margin_px", type=float, default=None,
                   help="keep proposals this far from the domain edge")
    p.add_argument("--gap_min_sep_px", type=float, default=None,
                   help="keep proposals this far from existing sensors")
    # GNN
    p.add_argument("--gnn_epochs",  type=int, default=30)
    p.add_argument("--gnn_corr_threshold", type=float, default=GNN_CORR_THRESHOLD,
                   help="|rho| threshold for creating an edge (mesoscale anomalies)")
    # RL
    p.add_argument("--rl_steps",    type=int, default=2000)
    p.add_argument("--rl_grid_x",   type=int, default=8)
    p.add_argument("--rl_grid_y",   type=int, default=12)
    p.add_argument("--rl_n_min",    type=int, default=5)
    p.add_argument("--rl_n_max",    type=int, default=20)
    p.add_argument("--rl_info_mode", type=str, default="evf",
                   choices=["evf", "coverage", "legacy"],
                   help="RL information score")
    p.add_argument("--rl_min_sep", type=int, default=MIN_SEP_CELLS,
                   help="Minimum buoy separation (grid cells)")
    p.add_argument("--rl_influence_km", type=float, default=INFLUENCE_RADIUS_KM,
                   help="Sensor influence radius (km)")
    p.add_argument("--rl_episode_len", type=int, default=20,
                   help="Actions per RL episode (default 20, same as standalone)")
    p.add_argument("--no_inductive", action="store_true",
                   help="pipeline: skip the inductive scoring of new candidates")
    p.add_argument("--n_inductive", type=int, default=3,
                   help="pipeline: number of candidate positions to score")
    p.add_argument("--inductive_min_sep", type=float, default=40.0,
                   help="pipeline: minimum spacing between scored candidates, in pixels")
    p.add_argument("--gif_frames",  type=int, default=40)
    p.add_argument("--no_cost_compare", action="store_true",
                   help="pipeline: skip the information vs cost network comparison")
    p.add_argument("--cost_info_tol", type=float, default=0.10,
                   help="pipeline: acceptable information loss for the cost-aware "
                        "network, as a fraction (default 0.10)")
    p.add_argument("--cost_n_lambda", type=int, default=8,
                   help="pipeline: number of lambda values swept")
    p.add_argument("--cost_compare_ref", choices=["greedy", "rl"], default="greedy",
                   help="pipeline: baseline for the comparison. greedy (default) "
                        "puts both networks under the same optimiser and isolates "
                        "the effect of the cost term; rl compares against the "
                        "network the agent actually retained")
    p.add_argument("--output_dir",  type=str, default="outputs")
    p.add_argument("--ocean_gif", action="store_true",
                   help="also write an animated GIF of the nature run")
    p.add_argument("--ocean_gif_every", type=int, default=5,
                   help="one frame every N days (default 5)")
    p.add_argument("--ocean_gif_var", type=str, default="T,GRADT,S,GRADS",
                   help="fields to animate, comma separated: T, S, SSH, ZETA, "
                        "GRADT, GRADS")
    p.add_argument("--ocean_gif_fps", type=int, default=8)
    p.add_argument("--no_nature_fig", action="store_true",
                   help="Skip the nature run diagnostic figure")
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
    """Write the text report from a list of sections (str or list[str])."""
    lines = []
    for s in sections:
        if isinstance(s, list):
            lines.extend(s)
        else:
            lines.append(s)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report -> {path}")


def _train_ae_quick(b1, T, S, args, args1):
    """Minimal AE training (for pipeline or individual mode)."""
    train_ds, val_ds = build_datasets(T, S, split=0.8,
                                      n_obs_min=args1.n_obs_min,
                                      n_obs_max=args1.n_obs_max,
                                      augment_train=True)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    model  = b1.ObservabilityVAE(
        base_ch=args1.base_ch, latent_ch=args1.latent_ch,
        dropout_p=args1.dropout_p, cond_dim=args1.cond_dim).to(DEVICE)
    optim  = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit   = b1.VAELoss(w_unobs=args1.w_unobs, lambda_grad=args1.lambda_grad,
                         huber_delta=args1.huber_delta, beta_max=args1.beta_max)
    best_loss = np.inf
    t0 = time.time()
    model.train()
    for ep in range(args1.epochs):
        ep_loss = 0.0
        for x, y, mask in loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            pred, mu, lv, aux = model(x)
            loss, *_ = crit(pred, y, mask, mu, lv, beta=0.1, aux_preds=aux)
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)
        best_loss = min(best_loss, ep_loss)
        print(f"    ep {ep+1}/{args1.epochs} | Loss={ep_loss:.4f}")
    # RMSE val MC
    model.eval()
    val_ld = DataLoader(val_ds, batch_size=8, shuffle=False)
    rmses, rmses_T, rmses_S = [], [], []
    with torch.no_grad():
        for x, y, mask in val_ld:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            preds = torch.stack([model(x)[0] for _ in range(args1.n_mc_val)])
            pm = preds.mean(0)
            for b in range(x.shape[0]):
                sq = (pm[b] - y[b])**2
                w  = (1 - mask[b])
                rmses.append(float(torch.sqrt((sq*w).mean()).item()))
                rmses_T.append(float(torch.sqrt((sq[0:1]*w).mean()).item()))
                rmses_S.append(float(torch.sqrt((sq[1:2]*w).mean()).item()))
    val_rmse = float(np.mean(rmses))
    # The two channels are normalised by very different standard deviations
    # (~2.6 degC and ~0.18 psu): an aggregate RMSE converts back to physical
    # units for neither variable. Report them separately.
    rmse_T_phys = float(np.mean(rmses_T)) * train_ds.T_std
    rmse_S_phys = float(np.mean(rmses_S)) * train_ds.S_std
    elapsed  = round(time.time() - t0, 1)
    norm = {"T_mean": float(T.mean()), "T_std": float(T.std()),
            "S_mean": float(S.mean()), "S_std": float(S.std())}
    torch.save({"model_state": model.state_dict(), "args": vars(args1),
                "norm": norm,
                "ocean": {"seed_ocean": args.seed_ocean, "nt": args.nt}},
               Path(args.output_dir) / "vae_best.pt")
    return model, norm, best_loss, val_rmse, elapsed, rmse_T_phys, rmse_S_phys


# ══════════════════════════════════════════════════════════════════════════════
#  Shared report sections
# ══════════════════════════════════════════════════════════════════════════════

SEP = "─" * 68

def _compare_info_vs_cost(b3, env, rl_positions, best_mask, args, out):
    """
    Build the cost-aware counterpart of the retained network, at equal size,
    and draw the two side by side.

    Returns a metrics dict, or None if the comparison could not be made.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx_rl = np.where(best_mask > 0.5)[0].astype(int)
    n_star = len(idx_rl)
    if n_star < 3:
        print("  [skip] retained network too small to compare")
        return None

    info_rl = b3._config_info(env, idx_rl)
    cost_rl, co2_rl, km_rl = env.network_cost(idx_rl)
    print(f"  RL retained       : N={n_star}  info={info_rl:.3f}  "
          f"cost={cost_rl:.0f} kEUR/yr  {co2_rl:.0f} tCO2/yr")

    # Reference for the comparison. Both networks must come from the SAME
    # optimiser, otherwise the difference mixes the effect of the objective
    # with the quality of the search, and the figure proves nothing.
    if getattr(args, "cost_compare_ref", "greedy") == "rl":
        idx_info = idx_rl
    else:
        seq0 = b3._greedy_weighted(env, n_star, 0.0)
        if not seq0:
            print("  [skip] greedy reference failed")
            return None
        idx_info = np.asarray(seq0[-1], dtype=int)

    info_a = b3._config_info(env, idx_info)
    cost_a, co2_a, km_a = env.network_cost(idx_info)

    # lambda scale: cost of one buoy against the information it typically buys
    lam_ref = info_a / max(cost_a, 1e-6)
    lambdas = np.geomspace(0.05 * lam_ref, 5.0 * lam_ref, args.cost_n_lambda)

    print(f"  Information only  : N={len(idx_info)}  info={info_a:.3f}  "
          f"cost={cost_a:.0f} kEUR/yr  {co2_a:.0f} tCO2/yr  tour={km_a:.0f} km")
    print(f"  Sweeping {len(lambdas)} lambda values at fixed N={n_star}...")

    best = None
    for lam in lambdas:
        seqs = b3._greedy_weighted(env, n_star, float(lam))
        if not seqs:
            continue
        idx_b = np.asarray(seqs[-1], dtype=int)
        if len(idx_b) != n_star:
            continue
        info_b = b3._config_info(env, idx_b)
        cost_b, co2_b, km_b = env.network_cost(idx_b)
        loss = (info_a - info_b) / (info_a + 1e-9)
        saving = (cost_a - cost_b) / (cost_a + 1e-9)
        if loss <= args.cost_info_tol and (best is None or saving > best["saving"]):
            best = {"idx": idx_b, "info": info_b, "cost": cost_b, "co2": co2_b,
                    "km": km_b, "loss": loss, "saving": saving, "lam": float(lam)}

    if best is None:
        print(f"  [skip] no cost-aware network within {args.cost_info_tol:.0%} "
              f"information loss")
        return None

    idx_cost = best["idx"]
    print(f"  Cost-aware network: N={len(idx_cost)}  info={best['info']:.3f}  "
          f"cost={best['cost']:.0f} kEUR/yr  {best['co2']:.0f} tCO2/yr  "
          f"tour={best['km']:.0f} km")
    print(f"  -> cost {-best['saving']:+.0%}, information {-best['loss']:+.0%} "
          f"(lambda={best['lam']:.3g})")

    set_a, set_b = set(idx_info.tolist()), set(idx_cost.tolist())
    shared = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    moved = len(only_a)
    print(f"  {len(shared)}/{n_star} positions in common, {moved} moved")

    # ---- figure -----------------------------------------------------------
    try:
        var_field = env.field_stats.reshape(env.grid_x, env.grid_y)
    except Exception:
        var_field = None
    P = np.asarray(env.candidate_positions, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), facecolor="white")
    panels = [
        (axes[0], idx_info, only_a, "#c0392b",
         f"Information only\nN={n_star}  info={info_a:.3f}  "
         f"{cost_a:.0f} kEUR/yr  {co2_a:.0f} tCO2/yr"),
        (axes[1], idx_cost, only_b, "#1e8449",
         f"Information and cost\nN={len(idx_cost)}  info={best['info']:.3f}  "
         f"{best['cost']:.0f} kEUR/yr  {best['co2']:.0f} tCO2/yr"),
    ]
    for ax, idx, uniq, col, title in panels:
        if var_field is not None:
            ax.imshow(np.asarray(var_field).T, origin="lower", cmap="YlOrBr",
                      alpha=0.5, extent=[0, NX, 0, NY], aspect="auto")
        ax.scatter(P[:, 0], P[:, 1], s=6, c="0.75", zorder=2,
                   label="candidate positions")
        sh = np.array([env.candidate_positions[i] for i in shared], dtype=float)
        if len(sh):
            ax.scatter(sh[:, 0], sh[:, 1], s=95, c="0.35", edgecolors="black",
                       linewidths=0.7, zorder=4, label="kept in both")
        un = np.array([env.candidate_positions[i] for i in uniq], dtype=float)
        if len(un):
            ax.scatter(un[:, 0], un[:, 1], s=140, c=col, edgecolors="black",
                       linewidths=0.9, marker="D", zorder=5,
                       label="specific to this network")
        ax.set_title(title, fontsize=11, fontweight="bold", color="#002060")
        ax.set_xlim(0, NX); ax.set_ylim(0, NY)
        ax.set_xlabel("x (pixels)", fontsize=9)
        ax.set_ylabel("y (pixels)", fontsize=9)
        ax.grid(True, alpha=0.25, color="0.7")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.suptitle("Same size, different answer: adding cost and carbon moves "
                 f"{moved} of the {n_star} buoys",
                 fontsize=13, fontweight="bold", color="#002060", y=1.00)
    fig.text(0.5, -0.02,
             f"operating cost {-best['saving']:+.0%}, CO2 "
             f"{-(co2_a - best['co2']) / (co2_a + 1e-9):+.0%}, information "
             f"{-best['loss']:+.0%}   |   "
             f"maintenance tour {km_a:.0f} km against {best['km']:.0f} km",
             ha="center", fontsize=10, color="0.30")
    fig.tight_layout()
    out_path = Path(out) / "rl_info_vs_cost_networks.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Figure -> {out_path}")

    return {"cmp_n": n_star,
            "cmp_info_a": info_a, "cmp_cost_a": cost_a, "cmp_co2_a": co2_a,
            "cmp_info_b": best["info"], "cmp_cost_b": best["cost"],
            "cmp_co2_b": best["co2"],
            "cmp_info_loss_pct": best["loss"] * 100,
            "cmp_cost_saving_pct": best["saving"] * 100,
            "cmp_moved": moved, "cmp_shared": len(shared),
            "cmp_lambda": best["lam"],
            "cmp_info_rl": info_rl, "cmp_cost_rl": cost_rl, "cmp_co2_rl": co2_rl,
            "cmp_positions_cost": [tuple(int(v) for v in env.candidate_positions[i])
                                   for i in idx_cost]}


def _report_header(mode, args, T, positions, ts, ocean_diag=None):
    _D = ocean_diag or {}
    return [
        "=" * 68,
        "  NAIADE -- OED-AI for marine SNOs -- metrics report",
        f"  Mode     : {mode}",
        f"  Generated : {ts}",
        "=" * 68, "",
        "-- REPRODUCIBILITY --------------------------------------------------",
        f"  seed_ocean  : {args.seed_ocean}",
        f"  seed_buoys  : {args.seed_buoys}",
        f"  nt          : {args.nt}  time steps",
        f"  n_buoys     : {len(positions)}  sensors",
        "",
        "-- NATURE RUN -------------------------------------------------------",
        f"  Domain      : {NX*DX_KM:.0f} x {NY*DX_KM:.0f} km  (dx = {DX_KM:.0f} km)",
        f"  SST         : [{T.min():.2f}, {T.max():.2f}] degC   sigma = {T.std():.2f} degC",
        f"  Obs noise   : {OBS_NOISE_T} degC (SST) / {OBS_NOISE_S} psu (SSS)",
        f"  L_decorr    : {_D.get('L_decorr_SST_km', float('nan')):.0f} km  "
        f"(reference sensor spacing)",
        f"  tau mesoscl : {_D.get('tau_SST_mesoscale_days', float('nan')):.0f} days  "
        f"(reference sampling frequency)",
        "",
        "-- BUOY POSITIONS (pixel x, y) --------------------------------------",
    ] + [f"  B{i:02d} : ({px:4d}, {py:4d})"
         for i, (px, py) in enumerate(positions)]


def _report_ae(m):
    return [
        "", "-- BRICK 1 -- AE-UNet MC-Dropout ------------------------------------",
        f"  Train loss (best)   : {m['ae_best_loss']:.4f}",
        f"  RMSE_val (normalised): {m['ae_rmse_val']:.4f}",
        f"  RMSE_val SST        : {m['ae_rmse_T_degC']:.3f} degC",
        f"  RMSE_val SSS        : {m['ae_rmse_S_psu']:.4f} psu",
        f"  Time                : {m['ae_time']} s",
    ]


def _report_gnn(m):
    lines = [
        "", "-- BRICK 2 -- GNN ---------------------------------------------------",
        f"  Graph edges            : {m['gnn_edges']}",
        f"  Contribution mean/std  : {m['gnn_score_mean']:.3f} +/- {m['gnn_score_std']:.3f}",
        f"  Mean redundancy        : {m['gnn_redond_mean']:.3f}",
        f"  Redundant sensors      : {m['gnn_n_redondant']}  (uniqueness Q25)",
        f"  Time                   : {m['gnn_time']} s",
    ]
    if m.get("gnn_redundant_ids"):
        lines.append(f"  Redundant IDs          : {m['gnn_redundant_ids']}")
    return lines


def _report_rl(m):
    return [
        "", "-- BRICK 3 -- RL ----------------------------------------------------",
        f"  N* (elbow point)     : {m['rl_n_star']} sensors",
        f"  Info score at N*     : {m['rl_info_star']:.3f}",
        f"  Max info score       : {m['rl_info_max']:.3f}",
        f"  Light config N       : {m['rl_n_light']} sensors",
        f"  Light config info    : {m['rl_info_light']:.3f}",
        f"  Info loss dense->light: {m['rl_perte_pct']:.1f} %",
        f"  Time                 : {m['rl_time']} s",
    ]


def _report_footer(mode, total, args, metrics, out_dir):
    lines = [
        "", "-- SUMMARY ----------------------------------------------------------",
        f"  Mode        : {mode}",
        f"  Total time  : {round(total, 1)} s",
        "",
        "-- FILES PRODUCED ---------------------------------------------------",
    ]
    for f in sorted(Path(out_dir).iterdir()):
        if f.suffix in {".pt", ".png", ".gif", ".txt"}:
            lines.append(f"  {f.name:<46} {f.stat().st_size // 1024:>5} KB")
    lines += [
        "",
        "-- REPRODUCIBILITY JSON ---------------------------------------------",
        json.dumps({"seed_ocean": args.seed_ocean,
                    "seed_buoys": args.seed_buoys,
                    "nt": args.nt,
                    "n_buoys": len(metrics["positions"]),
                    "mode": mode}, indent=2),
        "=" * 68,
    ]
    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    t0     = time.time()
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out    = Path(args.output_dir); out.mkdir(exist_ok=True)
    n_buoys = args.n_buoys or N_BUOYS

    print("=" * 68)
    print(f"  NAIADE  |  mode={args.mode}")
    print(f"  seed_ocean={args.seed_ocean}  seed_buoys={args.seed_buoys}  nt={args.nt}")
    print("=" * 68)

    # -- Shared nature run ------------------------------------------------------
    print(f"\n{SEP}\n  Nature Run  (seed={args.seed_ocean}, nt={args.nt})\n{SEP}")
    gen  = SyntheticOceanGenerator()
    run  = gen.generate_full(nt=args.nt, seed=args.seed_ocean)
    T, S = run["T"], run["S"]
    print(f"  T : {T.shape}  [{T.min():.2f}, {T.max():.2f}] degC  sigma={T.std():.2f}")
    print(f"  S : {S.shape}  [{S.min():.2f}, {S.max():.2f}] psu  sigma={S.std():.3f}")
    diag = gen.diagnostics()
    print(f"  Scales      : L_decorr={diag['L_decorr_SST_km']:.0f} km  "
          f"tau_mesoscale={diag['tau_SST_mesoscale_days']:.0f} d  "
          f"corr T-S={diag['corr_TS_global']:+.2f}")
    if not args.no_nature_fig:
        plot_nature_run(run, out_path=str(out / "ocean_nature_run.png"))
    if args.ocean_gif:
        animate_nature_run(run, out_path=str(out / "ocean_nature_run.gif"),
                           every=args.ocean_gif_every, var=args.ocean_gif_var,
                           fps=args.ocean_gif_fps)
    metrics_ocean = diag

    rng      = np.random.default_rng(args.seed_buoys)
    init_pos = sample_separated_positions(NX, NY, n_buoys, rng=rng)
    if args.nt < 365:
        print(f"  [WARNING] nt={args.nt} < 365: the seasonal cycle is not "
              f"sampled over a full period.\n"
              f"            T-S correlations and variability statistics "
              f"will be biased.")
    print(f"  Initial network : {n_buoys} buoys  (seed_buoys={args.seed_buoys})")

    metrics = {"positions": init_pos, "ocean": metrics_ocean}

    # -- Load the bricks --------------------------------------------------------
    brick_dir = Path(__file__).parent
    b1 = load_brick(brick_dir / "01_autoencoder.py")
    b2 = load_brick(brick_dir / "02_gnn.py")
    b3 = load_brick(brick_dir / "03_rl.py")

    # Shared AE namespace
    ae_ns = types.SimpleNamespace(
        train=True, score=False, figures=False,
        epochs=args.ae_epochs, batch_size=8, lr=3e-4,
        base_ch=args.ae_base_ch, latent_ch=32, cond_dim=16, dropout_p=0.15,
        w_unobs=4.0, lambda_grad=0.5, lambda_spec=0.0, lambda_ts=0.0,
        huber_delta=0.5, beta_max=0.0, n_obs_min=10, n_obs_max=60,
        n_mc_val=3, n_mc=20, output_dir=str(out),
        n_proposed=args.n_proposed,
        gap_margin_px=args.gap_margin_px,
        gap_min_sep_px=args.gap_min_sep_px,
        checkpoint=str(out / "vae_best.pt"))

    # GNN namespace
    gnn_ns = types.SimpleNamespace(
        gnn_epochs=args.gnn_epochs, output_dir=str(out),
        corr_threshold=args.gnn_corr_threshold, k_nearest=4,
        deseason=1, n_buoys=n_buoys,
        seed_ocean=args.seed_ocean, seed_buoys=args.seed_buoys)

    # RL namespace -- parameters identical to standalone mode
    # (buffer_size and episode_len must be consistent across both modes)
    rl_ns = types.SimpleNamespace(
        rl_steps=args.rl_steps, buffer_size=512, lr=3e-4,
        output_dir=str(out),
        grid_x=args.rl_grid_x, grid_y=args.rl_grid_y,
        n_min=args.rl_n_min, n_max=args.rl_n_max,
        episode_len=args.rl_episode_len, w_info=1.0, w_budget=0.5,
        info_mode=args.rl_info_mode, influence_km=args.rl_influence_km,
        n_random=15,
        gif_frames=args.gif_frames,
        seed_ocean=args.seed_ocean, seed_buoys=args.seed_buoys)

    report_sections = _report_header(args.mode, args, T, init_pos, ts,
                                     ocean_diag=metrics_ocean)

    # ══════════════════════════════════════════════════════════════════════════
    if args.mode == "individual":
        _run_individual(args, T, S, init_pos, b1, b2, b3,
                        ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0)
    else:
        _run_pipeline(args, T, S, init_pos, b1, b2, b3,
                      ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0)


# ══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL MODE: AE -> GNN -> RL  (each on the initial network)
# ══════════════════════════════════════════════════════════════════════════════

def _run_individual(args, T, S, positions, b1, b2, b3,
                    ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0):

    print(f"\n{SEP}\n  INDIVIDUAL MODE -- 3 independent bricks\n{SEP}")

    # -- Brick 1 -- AE ----------------------------------------------------------
    print(f"\n{SEP}\n  BRICK 1 -- AE-UNet MC-Dropout\n{SEP}")
    (model_ae, norm, best_loss, val_rmse, ae_time,
     rmse_T_phys, rmse_S_phys) = _train_ae_quick(b1, T, S, args, ae_ns)

    ae_fig_ns = types.SimpleNamespace(**{k: v for k, v in vars(ae_ns).items()
                                         if k != "figures"}, figures=True)
    ae_fig_ns.output_dir = str(out)
    print("  AE figures...")
    model_ae.eval()
    b1.plot_network_evaluation(model_ae, T, S, norm, ae_fig_ns,
                                positions=positions, n_samples=ae_ns.n_mc)
    b1.plot_uncertainty_maps(model_ae, T, S, norm, ae_fig_ns, n_samples=ae_ns.n_mc)

    m_ae = {"ae_best_loss": float(best_loss), "ae_rmse_val": val_rmse,
            "ae_rmse_T_degC": rmse_T_phys, "ae_rmse_S_psu": rmse_S_phys,
            "ae_time": ae_time}
    metrics.update(m_ae)
    report_sections += _report_ae(m_ae)
    print(f"  ✓ AE  RMSE_val={val_rmse:.4f}  "
          f"({rmse_T_phys:.3f} °C | {rmse_S_phys:.4f} psu)  [{ae_time}s]")
    print(f"\n{SEP}\n  BRICK 2 -- GNN network structure\n{SEP}")
    t0_gnn = time.time()
    corr   = b2.build_spatial_correlation(T, S, positions, n_timestamps=min(600, args.nt))
    graph  = b2.build_graph(positions, corr, corr_threshold=0.5, k_nearest=4)
    tgts   = b2.compute_proxy_targets(positions, corr)
    print(f"  Graph : {len(positions)} nodes, {graph['edge_index'].shape[1]} edges")
    model_gnn = b2.train_gnn(gnn_ns, graph, tgts)
    scores_gnn, redund, _ = b2.analyze_network(model_gnn, graph, tgts, gnn_ns, T=T)
    gnn_time = round(time.time() - t0_gnn, 1)
    unicite  = 1 - redund
    is_redond = unicite < np.percentile(unicite, 25)
    m_gnn = {"gnn_edges": int(graph['edge_index'].shape[1]),
             "gnn_score_mean": float(scores_gnn.mean()),
             "gnn_score_std":  float(scores_gnn.std()),
             "gnn_redond_mean": float(redund.mean()),
             "gnn_n_redondant": int(is_redond.sum()),
             "gnn_redundant_ids": [int(i) for i in np.where(is_redond)[0]],
             "gnn_time": gnn_time}
    metrics.update(m_gnn)
    report_sections += _report_gnn(m_gnn)
    print(f"  [ok] GNN  {m_gnn['gnn_n_redondant']} redundant  [{gnn_time}s]")

    # -- Brick 3 -- RL ----------------------------------------------------------
    print(f"\n{SEP}\n  BRICK 3 -- RL optimisation\n{SEP}")
    t0_rl = time.time()
    env   = b3.OceanNetworkEnv(T, S, grid_x=rl_ns.grid_x, grid_y=rl_ns.grid_y,
                                n_min=rl_ns.n_min, n_max=rl_ns.n_max,
                                episode_len=rl_ns.episode_len)
    policy = b3.train_ppo(rl_ns, env)
    pareto_pts, pareto_mask, n_star = b3.compute_pareto_front(
        env, policy, rl_ns, n_random=15)
    b3.visualize_two_configs(env, pareto_pts, n_star, policy, rl_ns)
    print("  Progression GIF...")
    b3.save_rl_gif(env, policy, rl_ns, n_frames=rl_ns.gif_frames)
    rl_time = round(time.time() - t0_rl, 1)
    info_vals = np.array([p["info_mean"] for p in pareto_pts])
    n_vals    = np.array([p["n_buoys"]   for p in pareto_pts])
    n_star    = int(np.clip(n_star, env.n_min, env.n_max))   # clamp hors-plage
    n_light   = max(env.n_min, n_star // 2)
    info_light = float(info_vals[np.argmin(np.abs(n_vals - n_light))])
    info_star  = float(info_vals[np.argmin(np.abs(n_vals - n_star))])
    perte_pct  = (info_star - info_light) / (info_star + 1e-9) * 100
    m_rl = {"rl_n_star": int(n_star), "rl_info_star": info_star,
            "rl_info_max": float(info_vals.max()),
            "rl_n_light": int(n_light), "rl_info_light": info_light,
            "rl_perte_pct": perte_pct, "rl_time": rl_time}
    metrics.update(m_rl)
    report_sections += _report_rl(m_rl)
    print(f"  [ok] RL   N*={n_star}  [{rl_time}s]")

    # -- Report -----------------------------------------------------------------
    total = time.time() - t0
    report_sections += _report_footer("individual", total, args, metrics, str(out))
    write_report(out / f"report_individual_{ts}.txt", report_sections)
    _print_summary("individual", args, m_ae, m_gnn, m_rl, total)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE MODE: RL -> optimal positions -> GNN + AE score that network
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(args, T, S, init_pos, b1, b2, b3,
                  ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0):

    print(f"\n{SEP}\n  PIPELINE MODE : RL -> GNN -> AE\n{SEP}")

    # -- Step 1: RL proposes the optimal network ---------------------------------
    print(f"\n{SEP}\n  STEP 1/3 -- RL: search for the optimal network\n{SEP}")
    t0_rl = time.time()
    env    = b3.OceanNetworkEnv(T, S, grid_x=rl_ns.grid_x, grid_y=rl_ns.grid_y,
                                 n_min=rl_ns.n_min, n_max=rl_ns.n_max,
                                 episode_len=rl_ns.episode_len,
                                 info_mode=rl_ns.info_mode,
                                 influence_km=rl_ns.influence_km,
                                 min_sep=args.rl_min_sep)
    policy = b3.train_ppo(rl_ns, env)
    pareto_pts, pareto_mask, n_star = b3.compute_pareto_front(
        env, policy, rl_ns, n_random=15)

    # -- Extract BEFORE the figures: best_mask from the checkpoint --------------
    ckpt_path = Path(rl_ns.output_dir) / "rl_best.pt"
    best_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_mask = best_ckpt["active_mask"]
    active_idx   = np.where(best_mask > 0.5)[0]
    rl_positions = [env.candidate_positions[i] for i in active_idx]
    env.active_mask = best_mask.copy()
    info_retained   = float(env._compute_info_reward())
    print(f"  RL network (best checkpoint): {len(rl_positions)} active buoys  info={info_retained:.3f}")

    # RL figures: two_configs shows the retained config, Pareto annotated
    b3.visualize_two_configs(env, pareto_pts, n_star, policy, rl_ns, best_mask=best_mask)
    b3.mark_retained_config_on_pareto(len(rl_positions), info_retained, rl_ns.output_dir)
    print("  Progression GIF...")
    b3.save_rl_gif(env, policy, rl_ns, n_frames=rl_ns.gif_frames)
    rl_time = round(time.time() - t0_rl, 1)

    info_vals = np.array([p["info_mean"] for p in pareto_pts])
    n_vals    = np.array([p["n_buoys"]   for p in pareto_pts])
    n_star = int(np.clip(n_star, env.n_min, env.n_max))
    n_light   = max(env.n_min, n_star // 2)
    info_light = float(info_vals[np.argmin(np.abs(n_vals - n_light))])
    info_star  = float(info_vals[np.argmin(np.abs(n_vals - n_star))])
    perte_pct  = (info_star - info_light) / (info_star + 1e-9) * 100

    # Guarantee at least 5 positions for the GNN to be valid
    # (correlation matrix, k_nearest=4 requires >= 5 nodes)
    GNN_MIN = 5
    if len(rl_positions) < GNN_MIN:
        print(f"  [INFO] RL network ({len(rl_positions)} pos) < {GNN_MIN} -- "
              f"randomly padded up to {GNN_MIN}")
        all_cands = list(env.candidate_positions)
        extra_pool = [p for p in all_cands if p not in rl_positions]
        extra = list(np.random.default_rng(args.seed_buoys).choice(
            len(extra_pool), GNN_MIN - len(rl_positions), replace=False))
        rl_positions += [extra_pool[e] for e in extra]

    m_rl = {"rl_n_star": int(n_star), "rl_info_star": info_star,
            "rl_info_max": float(info_vals.max()),
            "rl_n_light": int(n_light), "rl_info_light": info_light,
            "rl_perte_pct": perte_pct, "rl_time": rl_time}
    metrics.update(m_rl)
    report_sections += _report_rl(m_rl)
    print(f"  [ok] RL  N*={n_star}  {len(rl_positions)} positions extracted  [{rl_time}s]")

    # Update the positions in the report
    report_sections += [
        "", "-- RL OPTIMAL POSITIONS (pixel x, y) --------------------------------",
    ] + [f"  R{i:02d} : ({px:4d}, {py:4d})"
         for i, (px, py) in enumerate(rl_positions)]
    metrics["rl_positions"] = rl_positions

    # -- Step 1b: same network with cost and carbon in the objective ------------
    if not args.no_cost_compare:
        print(f"\n{SEP}\n  STEP 1b/3 -- RL: information alone against information "
              f"and cost\n{SEP}")
        m_cmp = _compare_info_vs_cost(b3, env, rl_positions, best_mask, args, out)
        if m_cmp:
            metrics.update(m_cmp)
            report_sections += [
                "",
                "-- INFORMATION-ONLY vs COST-AWARE NETWORK (equal size) --------------",
                f"  N buoys                : {m_cmp['cmp_n']}",
                f"  info   (info only)     : {m_cmp['cmp_info_a']:.4f}",
                f"  info   (with cost)     : {m_cmp['cmp_info_b']:.4f}"
                f"   ({-m_cmp['cmp_info_loss_pct']:+.1f} %)",
                f"  cost   (info only)     : {m_cmp['cmp_cost_a']:.0f} kEUR/yr",
                f"  cost   (with cost)     : {m_cmp['cmp_cost_b']:.0f} kEUR/yr"
                f"   ({-m_cmp['cmp_cost_saving_pct']:+.1f} %)",
                f"  CO2    (info only)     : {m_cmp['cmp_co2_a']:.1f} t/yr",
                f"  CO2    (with cost)     : {m_cmp['cmp_co2_b']:.1f} t/yr",
                f"  positions in common    : {m_cmp['cmp_shared']}",
                f"  positions moved        : {m_cmp['cmp_moved']}",
                f"  lambda retained        : {m_cmp['cmp_lambda']:.4g}",
                f"  (RL retained network   : info={m_cmp['cmp_info_rl']:.4f}  "
                f"cost={m_cmp['cmp_cost_rl']:.0f} kEUR/yr  "
                f"{m_cmp['cmp_co2_rl']:.1f} tCO2/yr)",
            ] + ["", "-- COST-AWARE POSITIONS (pixel x, y) --------------------------------"] \
              + [f"  C{i:02d} : ({px:4d}, {py:4d})"
                 for i, (px, py) in enumerate(m_cmp["cmp_positions_cost"])]

    # -- Step 2: GNN scores the RL network --------------------------------------
    print(f"\n{SEP}\n  STEP 2/3 -- GNN: scoring the RL network\n{SEP}")
    t0_gnn = time.time()
    corr   = b2.build_spatial_correlation(T, S, rl_positions,
                                           n_timestamps=min(600, args.nt))
    graph  = b2.build_graph(rl_positions, corr,
                             corr_threshold=gnn_ns.corr_threshold,
                             k_nearest=gnn_ns.k_nearest, T=T, S=S)
    tgts   = b2.compute_proxy_targets(rl_positions, corr)
    print(f"  Graph : {len(rl_positions)} nodes, {graph['edge_index'].shape[1]} edges")
    model_gnn = b2.train_gnn(gnn_ns, graph, tgts)
    scores_gnn, redund, _ = b2.analyze_network(
        model_gnn, graph, tgts, gnn_ns, T=T, label="rl_optimal")
    gnn_time = round(time.time() - t0_gnn, 1)
    unicite  = 1 - redund
    is_redond = unicite < np.percentile(unicite, 25)
    m_gnn = {"gnn_edges": int(graph['edge_index'].shape[1]),
             "gnn_score_mean": float(scores_gnn.mean()),
             "gnn_score_std":  float(scores_gnn.std()),
             "gnn_redond_mean": float(redund.mean()),
             "gnn_n_redondant": int(is_redond.sum()),
             "gnn_redundant_ids": [int(i) for i in np.where(is_redond)[0]],
             "gnn_time": gnn_time}
    metrics.update(m_gnn)
    report_sections += _report_gnn(m_gnn)
    print(f"  [ok] GNN  {m_gnn['gnn_n_redondant']} redundant  [{gnn_time}s]")

    # -- Step 3: AE scores the RL network ---------------------------------------
    print(f"\n{SEP}\n  STEP 3/3 -- AE: gap zones + proposed buoys\n{SEP}")
    (model_ae, norm, best_loss, val_rmse, ae_time,
     rmse_T_phys, rmse_S_phys) = _train_ae_quick(b1, T, S, args, ae_ns)

    ae_fig_ns = types.SimpleNamespace(**{k: v for k, v in vars(ae_ns).items()
                                         if k != "figures"}, figures=True)
    ae_fig_ns.output_dir = str(out)
    print("  AE figures on the RL network...")
    model_ae.eval()
    loo_delta, gap_map, _ae_pos, proposed_arr = b1.plot_network_evaluation(
        model_ae, T, S, norm, ae_fig_ns,
        positions=rl_positions, n_samples=ae_ns.n_mc)
    b1.plot_uncertainty_maps(model_ae, T, S, norm, ae_fig_ns, n_samples=ae_ns.n_mc)

    # -- Step 3b: the GNN scores the buoys the AE proposed ----------------------
    if not args.no_inductive:
        print(f"\n{SEP}\n  STEP 3b/3 -- GNN: scoring the buoys proposed by the AE"
              f"\n{SEP}")
        new_positions = [(int(px), int(py)) for px, py in np.asarray(proposed_arr)]
        # drop anything that coincides with a buoy already in the network
        retained = {(int(px), int(py)) for px, py in rl_positions}
        new_positions = [p for p in new_positions if p not in retained]

        if not new_positions:
            print("  [skip] the AE proposed no position outside the network")
        else:
            print(f"  {len(new_positions)} positions proposed by the AE gap map:")
            ret_arr = np.asarray(rl_positions, dtype=float)
            for i, (px, py) in enumerate(new_positions):
                d = float(np.sqrt(((ret_arr - np.array([px, py], dtype=float)) ** 2)
                                  .sum(1)).min())
                print(f"    P{i+1} @ ({px}, {py})  nearest existing buoy at {d:.0f} px")

            b2.inductive_eval(model_gnn, graph, new_positions, corr, gnn_ns,
                              T=T, S=S)
            metrics["gnn_inductive_positions"] = new_positions
            report_sections += [
                "",
                "-- GNN SCORES FOR THE AE-PROPOSED BUOYS (pixel x, y) ----------------",
            ] + [f"  P{i+1} : ({px:4d}, {py:4d})"
                 for i, (px, py) in enumerate(new_positions)]

    m_ae = {"ae_best_loss": float(best_loss), "ae_rmse_val": val_rmse,
            "ae_rmse_T_degC": rmse_T_phys, "ae_rmse_S_psu": rmse_S_phys,
            "ae_time": ae_time}
    metrics.update(m_ae)
    report_sections += _report_ae(m_ae)
    print(f"  ✓ AE  RMSE_val={val_rmse:.4f}  "
          f"({rmse_T_phys:.3f} °C | {rmse_S_phys:.4f} psu)  [{ae_time}s]")

    # -- Report -----------------------------------------------------------------
    total = time.time() - t0
    report_sections += _report_footer("pipeline", total, args, metrics, str(out))
    write_report(out / f"report_pipeline_{ts}.txt", report_sections)
    _print_summary("pipeline", args, m_ae, m_gnn, m_rl, total)


# ══════════════════════════════════════════════════════════════════════════════
#  Console summary
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(mode, args, m_ae, m_gnn, m_rl, total):
    print(f"\n{'='*68}")
    print(f"  [ok] Pipeline {mode}  ({total:.0f}s)")
    print(f"  seed_ocean={args.seed_ocean}  seed_buoys={args.seed_buoys}")
    print(f"  AE  RMSE_val={m_ae['ae_rmse_val']:.4f}  "
          f"({m_ae['ae_rmse_T_degC']:.3f} °C | {m_ae['ae_rmse_S_psu']:.4f} psu)")
    print(f"  GNN {m_gnn['gnn_n_redondant']} redundant | "
          f"score moy={m_gnn['gnn_score_mean']:.3f}")
    print(f"  RL  N*={m_rl['rl_n_star']} | info={m_rl['rl_info_star']:.3f} | "
          f"light N={m_rl['rl_n_light']} (loss {m_rl['rl_perte_pct']:.1f}%)")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
