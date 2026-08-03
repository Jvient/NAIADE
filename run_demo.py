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
from config import *
from data.dataset import (SyntheticOceanGenerator, build_datasets,
                          mesoscale_anomaly, plot_nature_run,
                          sample_separated_positions)


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
    p.add_argument("--nt",          type=int, default=365,
                   help="Longueur du nature run en jours (>= 365 conseille : "
                        "en deca le cycle saisonnier n est pas echantillonne)")
    p.add_argument("--n_buoys",     type=int, default=None,
                   help="Nombre de bouées (défaut = config.N_BUOYS)")
    # AE
    p.add_argument("--ae_epochs",   type=int, default=5)
    p.add_argument("--ae_base_ch",  type=int, default=16)
    # GNN
    p.add_argument("--gnn_epochs",  type=int, default=30)
    p.add_argument("--gnn_corr_threshold", type=float, default=GNN_CORR_THRESHOLD,
                   help="Seuil |rho| pour creer une arete (anomalies mesoechelle)")
    # RL
    p.add_argument("--rl_steps",    type=int, default=2000)
    p.add_argument("--rl_grid_x",   type=int, default=8)
    p.add_argument("--rl_grid_y",   type=int, default=12)
    p.add_argument("--rl_n_min",    type=int, default=5)
    p.add_argument("--rl_n_max",    type=int, default=20)
    p.add_argument("--rl_info_mode", type=str, default="evf",
                   choices=["evf", "coverage", "legacy"],
                   help="Score d information du RL")
    p.add_argument("--rl_min_sep", type=int, default=MIN_SEP_CELLS,
                   help="Separation mini entre bouees (cases de grille)")
    p.add_argument("--rl_influence_km", type=float, default=INFLUENCE_RADIUS_KM,
                   help="Rayon d influence d un capteur (km)")
    p.add_argument("--rl_episode_len", type=int, default=20,
                   help="Actions par épisode RL (défaut=20, identique au standalone)")
    p.add_argument("--gif_frames",  type=int, default=40)
    p.add_argument("--output_dir",  type=str, default="outputs")
    p.add_argument("--no_nature_fig", action="store_true",
                   help="Ne pas produire la figure diagnostique du nature run")
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


def _train_ae_quick(b1, T, S, args, args1):
    """Entraînement AE minimal (pour le pipeline ou mode individual)."""
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
    # Les deux canaux sont normalises par des ecarts-types differents
    # (~2.6 °C et ~0.18 psu) : un RMSE agrege ne se reconvertit en unite
    # physique pour aucune des deux variables. On les separe.
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
#  Rapport commun
# ══════════════════════════════════════════════════════════════════════════════

SEP = "─" * 68

def _report_header(mode, args, T, positions, ts, ocean_diag=None):
    _D = ocean_diag or {}
    return [
        "=" * 68,
        "  OED-IA SNO Marins — Rapport de métriques",
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
        f"  Domaine     : {NX*DX_KM:.0f} x {NY*DX_KM:.0f} km  (dx = {DX_KM:.0f} km)",
        f"  SST         : [{T.min():.2f}, {T.max():.2f}] °C   sigma = {T.std():.2f} °C",
        f"  Bruit obs   : {OBS_NOISE_T} °C (SST) / {OBS_NOISE_S} psu (SSS)",
        f"  L_decorr    : {_D.get('L_decorr_SST_km', float('nan')):.0f} km  "
        f"(espacement de reference des capteurs)",
        f"  tau mesoech : {_D.get('tau_SST_mesoech_j', float('nan')):.0f} j  "
        f"(frequence de reference d echantillonnage)",
        "",
        "── POSITIONS DES BOUÉES (pixel x, y) ───────────────────────────────",
    ] + [f"  B{i:02d} : ({px:4d}, {py:4d})"
         for i, (px, py) in enumerate(positions)]


def _report_ae(m):
    return [
        "", "── BRIQUE 1 — AE-UNet MC-Dropout ────────────────────────────────────",
        f"  Loss train (best)   : {m['ae_best_loss']:.4f}",
        f"  RMSE_val (normalisé): {m['ae_rmse_val']:.4f}",
        f"  RMSE_val SST        : {m['ae_rmse_T_degC']:.3f} °C",
        f"  RMSE_val SSS        : {m['ae_rmse_S_psu']:.4f} psu",
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
    print(f"\n{SEP}\n  Nature Run  (seed={args.seed_ocean}, nt={args.nt})\n{SEP}")
    gen  = SyntheticOceanGenerator()
    run  = gen.generate_full(nt=args.nt, seed=args.seed_ocean)
    T, S = run["T"], run["S"]
    print(f"  T : {T.shape}  [{T.min():.2f}, {T.max():.2f}] °C  sigma={T.std():.2f}")
    print(f"  S : {S.shape}  [{S.min():.2f}, {S.max():.2f}] psu  sigma={S.std():.3f}")
    diag = gen.diagnostics()
    print(f"  Echelles    : L_decorr={diag['L_decorr_SST_km']:.0f} km  "
          f"tau_mesoech={diag['tau_SST_mesoech_j']:.0f} j  "
          f"corr T-S={diag['corr_TS_globale']:+.2f}")
    if not args.no_nature_fig:
        plot_nature_run(run, out_path=str(out / "ocean_nature_run.png"))
    metrics_ocean = diag

    rng      = np.random.default_rng(args.seed_buoys)
    init_pos = sample_separated_positions(NX, NY, n_buoys, rng=rng)
    if args.nt < 365:
        print(f"  [ATTENTION] nt={args.nt} < 365 : le cycle saisonnier n est pas "
              f"echantillonne sur un cycle complet.\n"
              f"              Les correlations T-S et les statistiques de "
              f"variabilite seront biaisees.")
    print(f"  Réseau initial : {n_buoys} bouées  (seed_buoys={args.seed_buoys})")

    metrics = {"positions": init_pos, "ocean": metrics_ocean}

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
        gnn_epochs=args.gnn_epochs, output_dir=str(out),
        corr_threshold=args.gnn_corr_threshold, k_nearest=4,
        deseason=1, n_buoys=n_buoys,
        seed_ocean=args.seed_ocean, seed_buoys=args.seed_buoys)

    # Namespace RL — paramètres identiques au mode standalone
    # (buffer_size et episode_len doivent être cohérents entre les deux modes)
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
#  MODE INDIVIDUAL : AE → GNN → RL  (chacun sur le réseau initial)
# ══════════════════════════════════════════════════════════════════════════════

def _run_individual(args, T, S, positions, b1, b2, b3,
                    ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0):

    print(f"\n{SEP}\n  MODE INDIVIDUAL — 3 briques indépendantes\n{SEP}")

    # ── Brique 1 — AE ─────────────────────────────────────────────────────────
    print(f"\n{SEP}\n  BRIQUE 1 — AE-UNet MC-Dropout\n{SEP}")
    (model_ae, norm, best_loss, val_rmse, ae_time,
     rmse_T_phys, rmse_S_phys) = _train_ae_quick(b1, T, S, args, ae_ns)

    ae_fig_ns = types.SimpleNamespace(**{k: v for k, v in vars(ae_ns).items()
                                         if k != "figures"}, figures=True)
    ae_fig_ns.output_dir = str(out)
    print("  Figures AE...")
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
    print(f"\n{SEP}\n  BRIQUE 2 — GNN Structure du Réseau\n{SEP}")
    t0_gnn = time.time()
    corr   = b2.build_spatial_correlation(T, S, positions, n_timestamps=min(80, args.nt))
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
    env   = b3.OceanNetworkEnv(T, S, grid_x=rl_ns.grid_x, grid_y=rl_ns.grid_y,
                                n_min=rl_ns.n_min, n_max=rl_ns.n_max,
                                episode_len=rl_ns.episode_len)
    policy = b3.train_ppo(rl_ns, env)
    pareto_pts, pareto_mask, n_star = b3.compute_pareto_front(
        env, policy, rl_ns, n_random=15)
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

def _run_pipeline(args, T, S, init_pos, b1, b2, b3,
                  ae_ns, gnn_ns, rl_ns, metrics, report_sections, out, ts, t0):

    print(f"\n{SEP}\n  MODE PIPELINE : RL → GNN → AE\n{SEP}")

    # ── Étape 1 : RL propose le réseau optimal ─────────────────────────────────
    print(f"\n{SEP}\n  ÉTAPE 1/3 — RL : recherche du réseau optimal\n{SEP}")
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
    corr   = b2.build_spatial_correlation(T, S, rl_positions,
                                           n_timestamps=min(80, args.nt))
    graph  = b2.build_graph(rl_positions, corr,
                             corr_threshold=gnn_ns.corr_threshold,
                             k_nearest=gnn_ns.k_nearest, T=T, S=S)
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
    (model_ae, norm, best_loss, val_rmse, ae_time,
     rmse_T_phys, rmse_S_phys) = _train_ae_quick(b1, T, S, args, ae_ns)

    ae_fig_ns = types.SimpleNamespace(**{k: v for k, v in vars(ae_ns).items()
                                         if k != "figures"}, figures=True)
    ae_fig_ns.output_dir = str(out)
    print("  Figures AE sur réseau RL...")
    model_ae.eval()
    b1.plot_network_evaluation(model_ae, T, S, norm, ae_fig_ns,
                                positions=rl_positions, n_samples=ae_ns.n_mc)
    b1.plot_uncertainty_maps(model_ae, T, S, norm, ae_fig_ns, n_samples=ae_ns.n_mc)

    m_ae = {"ae_best_loss": float(best_loss), "ae_rmse_val": val_rmse,
            "ae_rmse_T_degC": rmse_T_phys, "ae_rmse_S_psu": rmse_S_phys,
            "ae_time": ae_time}
    metrics.update(m_ae)
    report_sections += _report_ae(m_ae)
    print(f"  ✓ AE  RMSE_val={val_rmse:.4f}  "
          f"({rmse_T_phys:.3f} °C | {rmse_S_phys:.4f} psu)  [{ae_time}s]")

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
    print(f"  AE  RMSE_val={m_ae['ae_rmse_val']:.4f}  "
          f"({m_ae['ae_rmse_T_degC']:.3f} °C | {m_ae['ae_rmse_S_psu']:.4f} psu)")
    print(f"  GNN {m_gnn['gnn_n_redondant']} redondants | "
          f"score moy={m_gnn['gnn_score_mean']:.3f}")
    print(f"  RL  N★={m_rl['rl_n_star']} | info={m_rl['rl_info_star']:.3f} | "
          f"légère N={m_rl['rl_n_light']} (perte {m_rl['rl_perte_pct']:.1f}%)")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
