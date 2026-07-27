"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   OED-IA pour SNO Marins — Orchestrateur                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Deux modes d'exécution :                                                   ║
║                                                                             ║
║  --mode individual   (défaut)                                               ║
║    Chaque brique est lancée indépendamment sur le même nature run.          ║
║    Utile pour comparer les sorties brique par brique.                       ║
║    Ordre : AE → GNN → RL                                                    ║
║                                                                             ║
║  --mode pipeline                                                            ║
║    1. RL propose un réseau optimal (N★ bouées)                              ║
║    2. GNN évalue ce réseau (structure + redondance)                         ║
║    3. AE évalue ce réseau (zones lacunaires + 3 bouées proposées)           ║
║    → Les briques 2 et 3 travaillent sur le MÊME réseau RL.                 ║
║                                                                             ║
║  Dans les deux cas : rapport .txt + JSON de reproductibilité.               ║
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
from config import (DEVICE, NX, NY, NT, N_BUOYS, N_CHANNELS,
                    VAE_IN_CH, VAE_OUT_CH, OBSERVED_VARS, MIN_BUOY_DIST)
from data.loader import load_ocean, add_data_args
from data.dataset import build_datasets, BuoySampler


# ══════════════════════════════════════════════════════════════════════════════
#  Arguments
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="OED-IA Orchestrateur (individual|pipeline)")
    p.add_argument("--mode",        choices=["individual", "pipeline"],
                   default="individual",
                   help="individual : briques indépendantes | pipeline : RL→GNN→AE")
    p.add_argument("--seed_ocean",  type=int, default=42,
                   help="Seed du nature run (reproductibilité)")
    p.add_argument("--seed_buoys",  type=int, default=7,
                   help="Seed du réseau initial de bouées")
    p.add_argument("--nt",          type=int, default=200,
                   help="Pas de temps du nature run")
    p.add_argument("--n_buoys",     type=int, default=None,
                   help="Nombre de bouées (défaut = config.N_BUOYS)")
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
    p.add_argument("--rl_episode_len", type=int, default=20,
                   help="Actions par épisode RL (défaut=20, identique au standalone)")
    p.add_argument("--gif_frames",  type=int, default=40)
    p.add_argument("--output_dir",  type=str, default="outputs")
    add_data_args(p)
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
    """Écrit le rapport texte depuis une liste de sections (str ou list[str])."""
    lines = []
    for s in sections:
        if isinstance(s, list):
            lines.extend(s)
        else:
            lines.append(s)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Rapport → {path}")


def _train_ae_quick(b1, fields, channels, sea_mask, args, args1):
    """Entraînement AE minimal (pour le pipeline ou mode individual)."""
    train_ds, val_ds = build_datasets(fields, channels, split=0.8,
                                      sea_mask=sea_mask,
                                      observed_vars=OBSERVED_VARS,
                                      n_obs_min=args1.n_obs_min,
                                      n_obs_max=args1.n_obs_max,
                                      warn_snr=False)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    model  = b1.ObservabilityVAE(
        in_ch=VAE_IN_CH, out_ch=VAE_OUT_CH,
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
        for x, y, mask, sea in loader:
            x, y, mask, sea = (x.to(DEVICE), y.to(DEVICE),
                               mask.to(DEVICE), sea.to(DEVICE))
            pred, mu, lv, aux = model(x)
            loss, *_ = crit(pred, y, mask, mu, lv, beta=0.1, aux_preds=aux,
                            sea=sea)
            optim.zero_grad(); loss.backward(); optim.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)
        best_loss = min(best_loss, ep_loss)
        print(f"    ep {ep+1}/{args1.epochs} | Loss={ep_loss:.4f}")
    # RMSE val MC
    model.eval()
    val_ld = DataLoader(val_ds, batch_size=8, shuffle=False)
    rmses  = []
    ch_sq = np.zeros(VAE_OUT_CH); ch_n = 0.0
    with torch.no_grad():
        for x, y, mask, sea in val_ld:
            x, y, mask, sea = (x.to(DEVICE), y.to(DEVICE),
                               mask.to(DEVICE), sea.to(DEVICE))
            preds = torch.stack([model(x)[0] for _ in range(args1.n_mc_val)])
            pm = preds.mean(0)
            w_all = (1 - mask) * sea
            ch_sq += ((pm - y)**2 * w_all).sum(dim=(0, 2, 3)).cpu().numpy()
            ch_n  += float(w_all.sum().item())
            for b in range(x.shape[0]):
                sq, wb = (pm[b] - y[b])**2, w_all[b]
                rmses.append(float(torch.sqrt(
                    (sq*wb).sum() / wb.sum().clamp_min(1.0)).item()))
    val_rmse = float(np.mean(rmses))
    # RMSE par canal en unités physiques — seule forme interprétable dès
    # qu'on mélange des °C, des PSU et des m/s.
    rmse_phys = {c: float(np.sqrt(ch_sq[i] / max(ch_n, 1.0)) * train_ds.std[i])
                 for i, c in enumerate(channels)}
    elapsed  = round(time.time() - t0, 1)
    norm = {"T_mean": float(train_ds.mean[0]), "T_std": float(train_ds.std[0]),
            "S_mean": float(train_ds.mean[1]), "S_std": float(train_ds.std[1])}
    torch.save({"model_state": model.state_dict(), "args": vars(args1),
                "norm": norm, "channels": channels, "rmse_phys": rmse_phys,
                # `stats` est requis par 04_baselines.py pour dénormaliser
                "stats": {"mean": train_ds.mean, "std": train_ds.std}},
               Path(args.output_dir) / "vae_best.pt")
    return model, norm, best_loss, val_rmse, elapsed, rmse_phys


# ══════════════════════════════════════════════════════════════════════════════
#  Rapport commun
# ══════════════════════════════════════════════════════════════════════════════

SEP = "─" * 68

def _report_header(mode, args, T, positions, ts, data_info=None):
    di = data_info or {}
    glorys = str(di.get("source", "")).startswith("GLORYS")

    lines = [
        "=" * 68,
        "  OED-IA SNO Marins — Rapport de métriques",
        f"  Mode     : {mode}",
        f"  Généré le: {ts}",
        "=" * 68, "",
        "── DONNÉES ──────────────────────────────────────────────────────────",
        f"  source      : {di.get('source', 'inconnue')}",
    ]

    if glorys:
        lo = di.get("lon_range", [0, 0]); la = di.get("lat_range", [0, 0])
        lines += [
            f"  période     : {di.get('date_start')} → {di.get('date_end')}"
            f"   ({di.get('n_times')} dates)",
            f"  fenêtre     : lon [{lo[0]:.4f}, {lo[1]:.4f}]   "
            f"lat [{la[0]:.4f}, {la[1]:.4f}]",
            f"  grille      : {NX} × {NY} px   ({di.get('dx_km')} km/px)",
            f"  100 % mer   : {di.get('full_sea')}   "
            f"(fraction {di.get('sea_fraction')})",
            f"  niveaux     : {di.get('depths_m')} m",
            f"  canaux      : {di.get('channels')}",
            f"  observés    : {di.get('observed_vars')}",
            f"  désaisonn.  : {di.get('seasonal_removed')}",
        ]

    ch0 = (di.get("channels") or ["canal 0"])[0]
    lines += [
        "",
        "── REPRODUCTIBILITÉ ─────────────────────────────────────────────────",
        f"  seed_ocean  : {args.seed_ocean}",
        f"  seed_buoys  : {args.seed_buoys}",
        f"  nt          : {args.nt}  pas de temps",
        f"  n_buoys     : {len(positions)}  capteurs",
        "",
        "── CHAMP ────────────────────────────────────────────────────────────",
        f"  {ch0} : [{T.min():.2f}, {T.max():.2f}]"
        + ("   (anomalie désaisonnalisée)" if di.get("seasonal_removed")
           else "   (valeurs brutes)"),
        "",
    ]

    # Coordonnées géographiques en plus des pixels : une position en pixels
    # n'est ni déployable sur le terrain ni lisible en présentation.
    if glorys and di.get("lon_range"):
        lo = di["lon_range"]; la = di["lat_range"]
        lon_ax = np.linspace(lo[0], lo[1], NX)
        lat_ax = np.linspace(la[0], la[1], NY)
        lines += [
            "── POSITIONS DES BOUÉES ─────────────────────────────────────────────",
            "        pixel (x, y)      longitude    latitude",
        ]
        for k, (px, py) in enumerate(positions):
            lines.append(f"  B{k:02d} : ({px:4d}, {py:4d})   "
                         f"{lon_ax[px]:10.4f}  {lat_ax[py]:10.4f}")
    else:
        lines += ["── POSITIONS DES BOUÉES (pixel x, y) ───────────────────────────────"]
        lines += [f"  B{k:02d} : ({px:4d}, {py:4d})"
                  for k, (px, py) in enumerate(positions)]
    return lines


def _report_ae(m):
    return [
        "", "── BRIQUE 1 — AE-UNet MC-Dropout ────────────────────────────────────",
        f"  Loss train (best)   : {m['ae_best_loss']:.4f}",
        f"  RMSE_val (normalisé): {m['ae_rmse_val']:.4f}",
        "  RMSE_val par canal (unités physiques) :",
        *[f"      {c:<12} {v:8.4f} "
          f"{ {'thetao':'°C','so':'PSU','uo':'m/s','vo':'m/s'}.get(c.rsplit('_z',1)[0],'') }"
          for c, v in m["ae_rmse_by_channel"].items()],
        f"  Temps               : {m['ae_time']} s",
    ]


def _report_gnn(m):
    lines = [
        "", "── BRIQUE 2 — GNN ───────────────────────────────────────────────────",
        f"  Arêtes graphe          : {m['gnn_edges']}",
        f"  Score contrib. moy±std : {m['gnn_score_mean']:.3f} ± {m['gnn_score_std']:.3f}",
        f"  Redondance moyenne     : {m['gnn_redond_mean']:.3f}",
        f"  Capteurs redondants    : {m['gnn_n_redondant']}  (unicité Q25)",
        f"  Temps                  : {m['gnn_time']} s",
    ]
    if m.get("gnn_redundant_ids"):
        lines.append(f"  IDs redondants         : {m['gnn_redundant_ids']}")
    return lines


def _report_rl(m):
    return [
        "", "── BRIQUE 3 — RL ────────────────────────────────────────────────────",
        f"  N★ (point de coude)  : {m['rl_n_star']} capteurs",
        f"  Score info N★        : {m['rl_info_star']:.3f}",
        f"  Score info max       : {m['rl_info_max']:.3f}",
        f"  Config légère N      : {m['rl_n_light']} capteurs",
        f"  Score info légère    : {m['rl_info_light']:.3f}",
        f"  Perte info dense→lég : {m['rl_perte_pct']:.1f} %",
        f"  Temps                : {m['rl_time']} s",
    ]


def _report_footer(mode, total, args, metrics, out_dir):
    lines = [
        "", "── RÉSUMÉ ───────────────────────────────────────────────────────────",
        f"  Mode        : {mode}",
        f"  Temps total : {round(total, 1)} s",
        "",
        "── FICHIERS PRODUITS ────────────────────────────────────────────────",
    ]
    for f in sorted(Path(out_dir).iterdir()):
        if f.suffix in {".pt", ".png", ".gif", ".txt"}:
            lines.append(f"  {f.name:<46} {f.stat().st_size // 1024:>5} KB")
    lines += [
        "",
        "── JSON REPRODUCTIBILITÉ ────────────────────────────────────────────",
        json.dumps({"seed_ocean": args.seed_ocean,
                    "seed_buoys": args.seed_buoys,
                    "nt": args.nt,
                    "n_buoys": len(metrics["positions"]),
                    "mode": mode}, indent=2),
        "=" * 68,
    ]
    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  Exécution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    t0     = time.time()
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out    = Path(args.output_dir); out.mkdir(exist_ok=True)
    n_buoys = args.n_buoys or N_BUOYS

    print("=" * 68)
    print(f"  OED-IA SNO Marins  |  mode={args.mode}")
    print(f"  seed_ocean={args.seed_ocean}  seed_buoys={args.seed_buoys}  nt={args.nt}")
    print("=" * 68)

    # ── Nature run commun ──────────────────────────────────────────────────────
    print(f"\n{SEP}\n  Champ océanique\n{SEP}")
    fields, channels, sea_mask, data_info = load_ocean(args)
    T, S = fields[:, 0], fields[:, 1]
    print(f"  source  : {data_info['source']}")
    print(f"  champ   : {fields.shape}  |  canaux : {channels}")
    if data_info.get("lon_range"):
        print(f"  fenêtre : lon [{data_info['lon_range'][0]:.3f}, "
              f"{data_info['lon_range'][1]:.3f}]  lat "
              f"[{data_info['lat_range'][0]:.3f}, {data_info['lat_range'][1]:.3f}]"
              f"  ({data_info['dx_km']} km/px)")
        print(f"  période : {data_info['date_start']} → {data_info['date_end']}"
              f"  |  100 % mer : {data_info['full_sea']}")

    init_pos = BuoySampler(NX, NY, n_buoys, sea_mask=sea_mask,
                           min_dist=MIN_BUOY_DIST, rng=args.seed_buoys).positions
    print(f"  Réseau initial : {n_buoys} bouées  (seed_buoys={args.seed_buoys}, "
          f"dist_min={MIN_BUOY_DIST} px)")

    metrics = {"positions": init_pos, "data_info": data_info}

    # ── Chargement des briques ─────────────────────────────────────────────────
    brick_dir = Path(__file__).parent
    b1 = load_brick(brick_dir / "01_autoencoder.py")
    b2 = load_brick(brick_dir / "02_gnn.py")
    b3 = load_brick(brick_dir / "03_rl.py")

    # Namespace commun AE
    ae_ns = types.SimpleNamespace(
        train=True, score=False, figures=False,
        epochs=args.ae_epochs, batch_size=8, lr=3e-4,
        base_ch=args.ae_base_ch, latent_ch=32, cond_dim=16, dropout_p=0.15,
        w_unobs=4.0, lambda_grad=0.5, lambda_spec=0.0, lambda_ts=0.0,
        huber_delta=0.5, beta_max=0.0, n_obs_min=10, n_obs_max=60,
        n_mc_val=3, n_mc=20, output_dir=str(out),
        checkpoint=str(out / "vae_best.pt"))

    # Namespace GNN
    gnn_ns = types.SimpleNamespace(
        gnn_epochs=args.gnn_epochs, output_dir=str(out), corr_threshold=0.5)

    # Namespace RL — paramètres identiques au mode standalone
    # (buffer_size et episode_len doivent être cohérents entre les deux modes)
    rl_ns = types.SimpleNamespace(
        rl_steps=args.rl_steps, buffer_size=512, lr=3e-4,
        output_dir=str(out),
        grid_x=args.rl_grid_x, grid_y=args.rl_grid_y,
        n_min=args.rl_n_min, n_max=args.rl_n_max,
        episode_len=args.rl_episode_len, w_info=1.0, w_budget=0.5,
        gif_frames=args.gif_frames)

    report_sections = _report_header(args.mode, args, T, init_pos, ts, data_info)

    # ══════════════════════════════════════════════════════════════════════════
    if args.mode == "individual":
        _run_individual(args, fields, channels, sea_mask, data_info, init_pos, b1, b2, b3,
                        ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0)
    else:
        _run_pipeline(args, fields, channels, sea_mask, data_info, init_pos, b1, b2, b3,
                      ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE INDIVIDUAL : AE → GNN → RL  (chacun sur le réseau initial)
# ══════════════════════════════════════════════════════════════════════════════

def _run_individual(args, fields, channels, sea_mask, data_info, positions, b1, b2, b3,
                    ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0):

    T, S = fields[:, 0], fields[:, 1]   # thetao_z0, so_z0

    print(f"\n{SEP}\n  MODE INDIVIDUAL — 3 briques indépendantes\n{SEP}")

    # ── Brique 1 — AE ─────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  BRIQUE 1 — AE-UNet MC-Dropout\n{SEP}")
    model_ae, norm, best_loss, val_rmse, ae_time, rmse_phys = _train_ae_quick(b1, fields, channels, sea_mask, args, ae_ns)

    ae_fig_ns = types.SimpleNamespace(**{k: v for k, v in vars(ae_ns).items()
                                         if k != "figures"}, figures=True)
    ae_fig_ns.output_dir = str(out)
    ae_fig_ns.sea_mask = sea_mask
    print("  Figures AE...")
    model_ae.eval()
    b1.plot_network_evaluation(model_ae, T, S, norm, ae_fig_ns,
                                positions=positions, n_samples=ae_ns.n_mc)
    b1.plot_uncertainty_maps(model_ae, T, S, norm, ae_fig_ns, n_samples=ae_ns.n_mc)

    m_ae = {"ae_best_loss": float(best_loss), "ae_rmse_val": val_rmse,
            "ae_rmse_by_channel": rmse_phys, "ae_time": ae_time}
    metrics.update(m_ae)
    report_sections += _report_ae(m_ae)
    # val_rmse est en unités normalisées, agrégé sur tous les canaux : le
    # multiplier par T.std() n'a plus de sens dès qu'on mélange °C, PSU et m/s.
    print(f"  ✓ AE  RMSE_val={val_rmse:.4f} (normalisé)  |  "
          f"thetao {rmse_phys['thetao_z0']:.4f} °C, "
          f"so {rmse_phys['so_z0']:.4f} PSU  [{ae_time}s]")
    print(f"\n{SEP}\n  BRIQUE 2 — GNN Structure du Réseau\n{SEP}")
    t0_gnn = time.time()
    corr   = b2.build_spatial_correlation(fields, None, positions, n_timestamps=min(80, args.nt))
    graph  = b2.build_graph(positions, corr, corr_threshold=0.5, k_nearest=4)
    tgts   = b2.compute_proxy_targets(positions, corr)
    print(f"  Graphe : {len(positions)} nœuds, {graph['edge_index'].shape[1]} arêtes")
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
    print(f"  ✓ GNN  {m_gnn['gnn_n_redondant']} redondants  [{gnn_time}s]")

    # ── Brique 3 — RL ─────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  BRIQUE 3 — RL Optimisation\n{SEP}")
    t0_rl = time.time()
    env   = b3.OceanNetworkEnv(fields, sea_mask=sea_mask,
                             dx_km=data_info.get('dx_km'),
                             grid_x=rl_ns.grid_x, grid_y=rl_ns.grid_y,
                                n_min=rl_ns.n_min, n_max=rl_ns.n_max,
                                episode_len=rl_ns.episode_len)
    policy = b3.train_ppo(rl_ns, env)
    pareto_pts, pareto_mask, n_star = b3.compute_pareto_front(env, policy, rl_ns)
    b3.visualize_two_configs(env, pareto_pts, n_star, policy, rl_ns)
    print("  GIF progression...")
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
    print(f"  ✓ RL   N★={n_star}  [{rl_time}s]")

    # ── Rapport ───────────────────────────────────────────────────────────────
    total = time.time() - t0
    report_sections += _report_footer("individual", total, args, metrics, str(out))
    write_report(out / f"rapport_individual_{ts}.txt", report_sections)
    _print_summary("individual", args, m_ae, m_gnn, m_rl, total)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE PIPELINE : RL → positions optimales → GNN + AE évaluent ce réseau
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(args, fields, channels, sea_mask, data_info, init_pos, b1, b2, b3,
                  ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0):

    T, S = fields[:, 0], fields[:, 1]   # thetao_z0, so_z0

    print(f"\n{SEP}\n  MODE PIPELINE : RL → GNN → AE\n{SEP}")

    # ── Étape 1 : RL propose le réseau optimal ─────────────────────────────────
    print(f"\n{SEP}\n  ÉTAPE 1/3 — RL : recherche du réseau optimal\n{SEP}")
    t0_rl = time.time()
    env    = b3.OceanNetworkEnv(fields, sea_mask=sea_mask,
                             dx_km=data_info.get('dx_km'),
                             grid_x=rl_ns.grid_x, grid_y=rl_ns.grid_y,
                                 n_min=rl_ns.n_min, n_max=rl_ns.n_max,
                                 episode_len=rl_ns.episode_len)
    policy = b3.train_ppo(rl_ns, env)
    pareto_pts, pareto_mask, n_star = b3.compute_pareto_front(env, policy, rl_ns)

    # ── Extraction AVANT les figures — best_mask du checkpoint ───────────────
    ckpt_path = Path(rl_ns.output_dir) / "rl_best.pt"
    best_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_mask = best_ckpt["active_mask"]
    active_idx   = np.where(best_mask > 0.5)[0]
    rl_positions = [env.candidate_positions[i] for i in active_idx]
    env.active_mask = best_mask.copy()
    info_retained   = float(env._compute_info_reward())
    print(f"  Réseau RL (best checkpoint) : {len(rl_positions)} bouées actives  info={info_retained:.3f}")

    # Figures RL : two_configs montre la config retenue (★), pareto annoté
    b3.visualize_two_configs(env, pareto_pts, n_star, policy, rl_ns, best_mask=best_mask)
    b3.mark_retained_config_on_pareto(len(rl_positions), info_retained, rl_ns.output_dir)
    print("  GIF progression...")
    b3.save_rl_gif(env, policy, rl_ns, n_frames=rl_ns.gif_frames)
    rl_time = round(time.time() - t0_rl, 1)

    info_vals = np.array([p["info_mean"] for p in pareto_pts])
    n_vals    = np.array([p["n_buoys"]   for p in pareto_pts])
    n_star = int(np.clip(n_star, env.n_min, env.n_max))
    n_light   = max(env.n_min, n_star // 2)
    info_light = float(info_vals[np.argmin(np.abs(n_vals - n_light))])
    info_star  = float(info_vals[np.argmin(np.abs(n_vals - n_star))])
    perte_pct  = (info_star - info_light) / (info_star + 1e-9) * 100

    # Garantir un minimum de 5 positions pour que le GNN soit valide
    # (matrice de corrélation, k_nearest=4 exige ≥ 5 nœuds)
    GNN_MIN = 5
    if len(rl_positions) < GNN_MIN:
        print(f"  [INFO] Réseau RL ({len(rl_positions)} pos) < {GNN_MIN} — "
              f"complétion aléatoire jusqu'à {GNN_MIN}")
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
    print(f"  ✓ RL  N★={n_star}  {len(rl_positions)} positions extraites  [{rl_time}s]")

    # Mettre à jour les positions dans le rapport
    report_sections += [
        "", "── POSITIONS OPTIMALES RL (pixel x, y) ─────────────────────────────",
    ] + [f"  R{i:02d} : ({px:4d}, {py:4d})"
         for i, (px, py) in enumerate(rl_positions)]
    metrics["rl_positions"] = rl_positions

    # ── Étape 2 : GNN évalue le réseau RL ─────────────────────────────────────
    print(f"\n{SEP}\n  ÉTAPE 2/3 — GNN : évaluation réseau RL\n{SEP}")
    t0_gnn = time.time()
    corr   = b2.build_spatial_correlation(fields, None, rl_positions,
                                           n_timestamps=min(80, args.nt))
    graph  = b2.build_graph(rl_positions, corr, corr_threshold=0.5, k_nearest=4)
    tgts   = b2.compute_proxy_targets(rl_positions, corr)
    print(f"  Graphe : {len(rl_positions)} nœuds, {graph['edge_index'].shape[1]} arêtes")
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
    print(f"  ✓ GNN  {m_gnn['gnn_n_redondant']} redondants  [{gnn_time}s]")

    # ── Étape 3 : AE évalue le réseau RL ──────────────────────────────────────
    print(f"\n{SEP}\n  ÉTAPE 3/3 — AE : zones lacunaires + bouées proposées\n{SEP}")
    model_ae, norm, best_loss, val_rmse, ae_time, rmse_phys = _train_ae_quick(b1, fields, channels, sea_mask, args, ae_ns)

    ae_fig_ns = types.SimpleNamespace(**{k: v for k, v in vars(ae_ns).items()
                                         if k != "figures"}, figures=True)
    ae_fig_ns.output_dir = str(out)
    ae_fig_ns.sea_mask = sea_mask
    print("  Figures AE sur réseau RL...")
    model_ae.eval()
    b1.plot_network_evaluation(model_ae, T, S, norm, ae_fig_ns,
                                positions=rl_positions, n_samples=ae_ns.n_mc)
    b1.plot_uncertainty_maps(model_ae, T, S, norm, ae_fig_ns, n_samples=ae_ns.n_mc)

    m_ae = {"ae_best_loss": float(best_loss), "ae_rmse_val": val_rmse,
            "ae_rmse_by_channel": rmse_phys, "ae_time": ae_time}
    metrics.update(m_ae)
    report_sections += _report_ae(m_ae)
    # val_rmse est en unités normalisées, agrégé sur tous les canaux : le
    # multiplier par T.std() n'a plus de sens dès qu'on mélange °C, PSU et m/s.
    print(f"  ✓ AE  RMSE_val={val_rmse:.4f} (normalisé)  |  "
          f"thetao {rmse_phys['thetao_z0']:.4f} °C, "
          f"so {rmse_phys['so_z0']:.4f} PSU  [{ae_time}s]")

    # ── Rapport ───────────────────────────────────────────────────────────────
    total = time.time() - t0
    report_sections += _report_footer("pipeline", total, args, metrics, str(out))
    write_report(out / f"rapport_pipeline_{ts}.txt", report_sections)
    _print_summary("pipeline", args, m_ae, m_gnn, m_rl, total)


# ══════════════════════════════════════════════════════════════════════════════
#  Résumé console
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(mode, args, m_ae, m_gnn, m_rl, total):
    print(f"\n{'='*68}")
    print(f"  ✓ Pipeline {mode}  ({total:.0f}s)")
    print(f"  seed_ocean={args.seed_ocean}  seed_buoys={args.seed_buoys}")
    _ru = {"thetao": "°C", "so": "PSU", "uo": "m/s", "vo": "m/s"}
    print(f"  AE  RMSE_val={m_ae['ae_rmse_val']:.4f} (normalisé) | par canal :")
    for c, v in m_ae["ae_rmse_by_channel"].items():
        print(f"        {c:<12} {v:8.4f} {_ru.get(c.rsplit('_z', 1)[0], '')}")
    print(f"  GNN {m_gnn['gnn_n_redondant']} redondants | "
          f"score moy={m_gnn['gnn_score_mean']:.3f}")
    print(f"  RL  N★={m_rl['rl_n_star']} | info={m_rl['rl_info_star']:.3f} | "
          f"légère N={m_rl['rl_n_light']} (perte {m_rl['rl_perte_pct']:.1f}%)")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
