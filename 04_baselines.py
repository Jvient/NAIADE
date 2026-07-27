r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NAIADE — Brique 4 : BASELINES de placement de capteurs                     ║
║                                                                              ║
║  Sans point de comparaison, un réseau proposé par RL n'est pas évaluable :   ║
║  « info = 0.09 » ne dit rien tant qu'on ignore ce que donnerait un tirage    ║
║  aléatoire. Ce module implémente quatre méthodes de référence et les compare ║
║  au RL sur une métrique INDÉPENDANTE.                                        ║
║                                                                              ║
║  ── Le point méthodologique central ────────────────────────────────────────  ║
║  Évaluer les baselines avec la récompense du MDP ferait gagner le RL par     ║
║  construction : c'est la fonction qu'il a explicitement maximisée. On évalue ║
║  donc tout le monde avec la RMSE de reconstruction de l'autoencodeur sur     ║
║  les pixels NON OBSERVÉS — une métrique qu'aucune méthode n'a optimisée      ║
║  directement. La récompense du MDP reste calculée à titre indicatif, pour    ║
║  montrer justement l'écart entre les deux points de vue.                     ║
║                                                                              ║
║  ── Méthodes ──────────────────────────────────────────────────────────────  ║
║  random    Tirage uniforme en mer. Le plancher. Répété n_repeat fois pour    ║
║            obtenir une moyenne et un écart-type — un RL qui ne bat pas       ║
║            random ± 1σ n'a rien appris.                                      ║
║  variance  Les n pixels de plus forte variance temporelle. Piège classique : ║
║            sans contrainte d'espacement, tous les capteurs s'agglutinent     ║
║            dans la zone la plus active. On impose donc min_dist.             ║
║  eof_qr    Placement optimal par pivots QR sur la base EOF                   ║
║            (Manohar, Brunton, Kutz & Brunton 2018, IEEE Control Syst. Mag.). ║
║            C'est LA référence du domaine pour le placement parcimonieux :    ║
║            si NAIADE ne la bat pas, la salle le demandera.                   ║
║  coverage  Échantillonnage du point le plus éloigné (farthest-point).        ║
║            Baseline purement géométrique, souvent étonnamment solide.        ║
║  rl        Réseau issu de la brique 3, chargé depuis rl_best.pt.             ║
║                                                                              ║
║  Usage :                                                                      ║
║      python 04_baselines.py --checkpoint outputs/vae_best.pt \                ║
║             --rl_checkpoint outputs/rl_best.pt --output_dir outputs           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from data.loader import load_ocean, add_data_args
from data.dataset import BuoySampler

UNITS = {"thetao": "°C", "so": "PSU", "uo": "m/s", "vo": "m/s"}


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def _sea_indices(sea_mask):
    xs, ys = np.where(sea_mask)
    return np.stack([xs, ys], axis=1)


def _greedy_with_spacing(order, idx, n, min_dist):
    """
    Parcourt les candidats dans l'ordre `order` et retient les n premiers
    respectant une distance minimale.

    Sans cette contrainte, les méthodes gloutonnes basées sur un score local
    (variance notamment) sélectionnent n pixels voisins dans le même tourbillon :
    le réseau est formellement « optimal » et pratiquement inutilisable.
    """
    chosen = []
    for i in order:
        p = idx[i]
        if min_dist <= 0 or all((p[0]-q[0])**2 + (p[1]-q[1])**2 >= min_dist**2
                                for q in chosen):
            chosen.append(p)
            if len(chosen) == n:
                break
    if len(chosen) < n:      # contrainte trop forte : on complète sans espacement
        for i in order:
            p = idx[i]
            if not any((p == q).all() for q in chosen):
                chosen.append(p)
                if len(chosen) == n:
                    break
    return [(int(p[0]), int(p[1])) for p in chosen]


# ══════════════════════════════════════════════════════════════════════════════
#  Méthodes de placement
# ══════════════════════════════════════════════════════════════════════════════

def place_random(fields, sea_mask, n, rng=None, min_dist=0):
    """Tirage uniforme parmi les pixels océaniques."""
    return BuoySampler(fields.shape[2], fields.shape[3], n,
                       sea_mask=sea_mask, min_dist=min_dist, rng=rng).positions


def place_variance(fields, sea_mask, n, min_dist=MIN_BUOY_DIST,
                   observed_only=True):
    """
    Les n pixels de plus forte variance temporelle.

    La variance est calculée sur les canaux NORMALISÉS puis moyennée : sinon
    la température (O(0.1) en °C²) écrase totalement les courants (O(1e-4)
    en m²/s²) et le critère se réduit à « où la SST bouge le plus ».
    """
    F = fields
    if observed_only:
        keep = [i for i in range(F.shape[1])
                if _chan_var(i) in OBSERVED_VARS] or list(range(F.shape[1]))
        F = F[:, keep]
    sd = F.std(axis=(0, 2, 3), keepdims=True) + 1e-9
    var_map = ((F / sd).var(axis=0)).mean(axis=0)      # (nx, ny)

    idx = _sea_indices(sea_mask)
    scores = var_map[idx[:, 0], idx[:, 1]]
    order = np.argsort(-scores)
    return _greedy_with_spacing(order, idx, n, min_dist)


_CHANNELS_CACHE = []


def _chan_var(i):
    if i < len(_CHANNELS_CACHE):
        return _CHANNELS_CACHE[i].rsplit("_z", 1)[0]
    return ""


def place_eof_qr(fields, sea_mask, n, n_modes=None):
    """
    Placement par pivots QR sur la base EOF — Manohar et al. (2018).

    Principe
    --------
    1. Matrice d'états X : (n_pixels_mer, nt · n_canaux_observés), centrée.
    2. SVD → U, base spatiale (EOF/POD).
    3. QR avec pivotage de colonnes sur Uᵣᵀ : les r premiers pivots sont les
       lignes (donc les pixels) qui rendent la sous-matrice la mieux
       conditionnée — c'est-à-dire les capteurs qui reconstruisent le mieux
       le champ dans le sous-espace dominant.

    Sur-échantillonnage : quand n > r, on pivote sur UᵣUᵣᵀ, comme dans
    l'article. C'est la référence standard du domaine — la baseline à battre.
    """
    from scipy.linalg import qr

    idx = _sea_indices(sea_mask)
    keep = [i for i in range(fields.shape[1])
            if _chan_var(i) in OBSERVED_VARS] or list(range(fields.shape[1]))
    F = fields[:, keep]
    sd = F.std(axis=(0, 2, 3), keepdims=True) + 1e-9
    Fn = F / sd

    # (n_pixels, nt * n_ch) : chaque ligne = signature temporelle d'un pixel
    X = np.concatenate([Fn[:, c][:, idx[:, 0], idx[:, 1]].T
                        for c in range(Fn.shape[1])], axis=1)
    X = X - X.mean(axis=1, keepdims=True)

    r_max = min(X.shape) - 1
    r = min(n_modes or n, r_max)
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    Ur = U[:, :r]

    if n <= r:
        _, _, piv = qr(Ur.T, pivoting=True, mode="economic")
    else:
        _, _, piv = qr(Ur @ Ur.T, pivoting=True, mode="economic")

    sel = piv[:n]
    return [(int(idx[i, 0]), int(idx[i, 1])) for i in sel]


def place_coverage(fields, sea_mask, n, rng=None):
    """
    Échantillonnage du point le plus éloigné (farthest-point sampling).

    Baseline purement géométrique : aucune information sur la physique, juste
    une couverture spatiale maximale. Elle est fréquemment difficile à battre,
    ce qui en fait un test exigeant — si le RL ne la dépasse pas, c'est qu'il
    n'exploite pas la dynamique du champ.
    """
    rng = np.random.default_rng(rng)
    idx = _sea_indices(sea_mask).astype(np.float32)
    start = rng.integers(len(idx))
    chosen = [start]
    d = np.linalg.norm(idx - idx[start], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(idx - idx[k], axis=1))
    return [(int(idx[i, 0]), int(idx[i, 1])) for i in chosen]


def load_rl_positions(rl_checkpoint, fields, sea_mask, args):
    """Recharge le réseau proposé par la brique 3."""
    ck = torch.load(rl_checkpoint, map_location="cpu", weights_only=False)
    mask = np.asarray(ck["active_mask"])

    sx, sy = NX / args.grid_x, NY / args.grid_y
    cands = []
    for gx in range(args.grid_x):
        for gy in range(args.grid_y):
            px = min(int(gx * sx + sx / 2), NX - 1)
            py = min(int(gy * sy + sy / 2), NY - 1)
            if sea_mask[px, py]:
                cands.append((px, py))

    if len(mask) != len(cands):
        raise ValueError(
            f"\n  active_mask de taille {len(mask)} mais {len(cands)} candidats "
            f"reconstruits avec grid_x={args.grid_x}, grid_y={args.grid_y}."
            f"\n  → relancer avec les MÊMES --grid_x / --grid_y que la brique 3.")
    return [cands[i] for i in np.where(mask > 0.5)[0]]


# ══════════════════════════════════════════════════════════════════════════════
#  Évaluation — métrique indépendante
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_positions(model, fields_n, positions, sea_mask, channels, std,
                       obs_idx, noise_norm, t_indices, n_mc=8):
    """
    RMSE de reconstruction sur les pixels NON OBSERVÉS et EN MER.

    `fields_n` doit déjà être normalisé avec les statistiques du modèle.
    Renvoie un dict {canal: rmse_physique} + la RMSE agrégée normalisée.
    """
    mask = np.zeros((NX, NY), dtype=np.float32)
    for (x, y) in positions:
        mask[x, y] = 1.0
    sea_f = sea_mask.astype(np.float32)
    w = torch.from_numpy((1.0 - mask) * sea_f).to(DEVICE)

    m_t = torch.from_numpy(mask[None]).to(DEVICE)
    sq = torch.zeros(len(channels), device=DEVICE)
    denom = 0.0

    for t in t_indices:
        y_true = torch.from_numpy(fields_n[t]).to(DEVICE)
        obs = y_true[obs_idx]
        noise = torch.randn_like(obs) * torch.from_numpy(
            noise_norm[obs_idx][:, None, None]).to(DEVICE)
        x_in = torch.cat([(obs + noise) * m_t, m_t], dim=0)[None]

        preds = torch.stack([model(x_in)[0] for _ in range(n_mc)]).mean(0)[0]
        sq += ((preds - y_true) ** 2 * w).sum(dim=(1, 2))
        denom += float(w.sum().item())

    rmse_n = (sq / max(denom, 1.0)).sqrt().cpu().numpy()
    return ({c: float(rmse_n[i] * std[i]) for i, c in enumerate(channels)},
            float(np.mean(rmse_n)))


# ══════════════════════════════════════════════════════════════════════════════
#  Comparaison
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(args):
    print("=" * 70)
    print("  Brique 4 — Baselines de placement de capteurs")
    print("=" * 70)

    # ── Données ──────────────────────────────────────────────────────────────
    fields, channels, sea_mask, data_info = load_ocean(args)
    global _CHANNELS_CACHE
    _CHANNELS_CACHE = list(channels)
    print(f"\n  {data_info['source']} | {fields.shape} | canaux={channels}")

    # ── Modèle AE (métrique d'évaluation) ────────────────────────────────────
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b1", Path(__file__).parent / "01_autoencoder.py")
    b1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b1)

    model = b1.ObservabilityVAE(
        in_ch=VAE_IN_CH, out_ch=VAE_OUT_CH,
        base_ch=ck["args"]["base_ch"], latent_ch=ck["args"]["latent_ch"],
        dropout_p=ck["args"].get("dropout_p", 0.1),
        cond_dim=ck["args"].get("cond_dim", 32)).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()

    if "stats" not in ck:
        raise KeyError("Le checkpoint AE ne contient pas 'stats' — "
                       "réentraîner la brique 1 après la migration.")
    mean = np.asarray(ck["stats"]["mean"], dtype=np.float32)
    std = np.asarray(ck["stats"]["std"], dtype=np.float32)
    fields_n = (fields - mean[None, :, None, None]) / std[None, :, None, None]

    obs_idx = [i for i, c in enumerate(channels)
               if c.rsplit("_z", 1)[0] in OBSERVED_VARS]
    phys_noise = np.array([OBS_NOISE.get(c.rsplit("_z", 1)[0], 0.0)
                           for c in channels], dtype=np.float32)
    noise_norm = (phys_noise / std).astype(np.float32)

    rng_eval = np.random.default_rng(args.seed_eval)
    t_idx = rng_eval.choice(len(fields), min(args.n_eval, len(fields)),
                            replace=False)
    print(f"  Évaluation : {len(t_idx)} dates × {args.n_mc} passes MC-Dropout")
    print(f"  Métrique   : RMSE de reconstruction AE sur pixels non observés")
    print(f"               (indépendante de la récompense du MDP)\n")

    # ── Réseau RL, s'il existe ───────────────────────────────────────────────
    rl_positions = None
    if args.rl_checkpoint and Path(args.rl_checkpoint).exists():
        try:
            rl_positions = load_rl_positions(args.rl_checkpoint, fields,
                                             sea_mask, args)
            print(f"  Réseau RL chargé : {len(rl_positions)} bouées")
        except Exception as e:
            print(f"  ⚠ Réseau RL non chargé : {e}")

    n_list = sorted(set(args.n_sensors))
    if rl_positions and len(rl_positions) not in n_list:
        n_list = sorted(set(n_list + [len(rl_positions)]))

    results = {}

    def _eval(pos):
        return evaluate_positions(model, fields_n, pos, sea_mask, channels,
                                  std, obs_idx, noise_norm, t_idx, args.n_mc)

    def _eval_repeated(pos, k):
        """
        Évalue k fois le MÊME réseau.

        Les positions sont déterministes, mais l'évaluation ne l'est pas :
        MC-Dropout et bruit d'observation sont tirés à chaque passe. Sans
        cette dispersion, on compare des méthodes déterministes à un chiffre
        unique et on conclut sur des écarts qui ne sortent pas du bruit —
        typiquement, un « le RL gagne de 1 % » qui ne veut rien dire.
        """
        rs, aggs = [], []
        for _ in range(k):
            r, a = _eval(pos)
            rs.append(r); aggs.append(a)
        return ({c: float(np.mean([r[c] for r in rs])) for c in channels},
                float(np.mean(aggs)), float(np.std(aggs)))

    for n in n_list:
        print(f"  ── N = {n} capteurs " + "─" * 40)
        entry = {}

        # random : moyenne ± écart-type sur n_repeat tirages
        rs, aggs = [], []
        for k in range(args.n_repeat):
            pos = place_random(fields, sea_mask, n, rng=1000 + k,
                               min_dist=MIN_BUOY_DIST)
            r, a = _eval(pos)
            rs.append(r); aggs.append(a)
        entry["random"] = {
            "rmse": {c: float(np.mean([r[c] for r in rs])) for c in channels},
            "rmse_std": {c: float(np.std([r[c] for r in rs])) for c in channels},
            "agg": float(np.mean(aggs)), "agg_std": float(np.std(aggs)),
            "n": n}
        print(f"    random    agg={entry['random']['agg']:.4f} "
              f"± {entry['random']['agg_std']:.4f}  (n={args.n_repeat} tirages)")

        for name, fn in [("variance", place_variance),
                         ("eof_qr", place_eof_qr),
                         ("coverage", place_coverage)]:
            try:
                pos = fn(fields, sea_mask, n)
                r, a, sd = _eval_repeated(pos, args.n_eval_repeat)
                entry[name] = {"rmse": r, "agg": a, "agg_std": sd, "n": n,
                               "positions": [list(p) for p in pos]}
                print(f"    {name:<9} agg={a:.4f} ± {sd:.4f}")
            except Exception as e:
                print(f"    {name:<9} échec : {e}")

        if rl_positions and n == len(rl_positions):
            r, a, sd = _eval_repeated(rl_positions, args.n_eval_repeat)
            entry["rl"] = {"rmse": r, "agg": a, "agg_std": sd, "n": n,
                           "positions": [list(p) for p in rl_positions]}
            print(f"    rl        agg={a:.4f} ± {sd:.4f}")

        results[str(n)] = entry

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    _verdict(results, channels, rl_positions)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "baselines_comparison.json", "w") as f:
        json.dump({"results": results, "channels": list(channels),
                   "data_info": data_info,
                   "metric": "AE reconstruction RMSE on unobserved sea pixels",
                   "n_eval": len(t_idx), "n_mc": args.n_mc}, f, indent=2)
    print(f"\n  JSON → {out}/baselines_comparison.json")

    _plot(results, channels, out, data_info)
    return results


def _verdict(results, channels, rl_positions):
    """Compare le RL aux baselines et tranche explicitement."""
    for n, entry in results.items():
        if "rl" not in entry:
            continue
        rl = entry["rl"]["agg"]
        rnd, rnd_sd = entry["random"]["agg"], entry["random"]["agg_std"]
        others = {k: v["agg"] for k, v in entry.items()
                  if k not in ("random", "rl")}

        print(f"\n  N = {n} capteurs")
        print(f"    RL vs aléatoire : {rl:.4f} vs {rnd:.4f} ± {rnd_sd:.4f}")
        if rl < rnd - rnd_sd:
            gain = 100 * (rnd - rl) / rnd
            print(f"      ✓ le RL bat l'aléatoire de {gain:.1f} % "
                  f"(au-delà de 1σ)")
        elif rl < rnd:
            print("      ~ le RL est meilleur mais DANS le bruit du tirage "
                  "aléatoire — non concluant")
        else:
            print("      ✗ le RL ne bat PAS un tirage aléatoire")

        if others:
            best = min(others, key=others.get)
            b_val = others[best]
            b_sd = entry[best].get("agg_std", 0.0)
            rl_sd = entry["rl"].get("agg_std", 0.0)
            # Incertitude combinée des deux mesures
            comb = float(np.sqrt(b_sd**2 + rl_sd**2))
            print(f"    Meilleure baseline : {best} "
                  f"({b_val:.4f} ± {b_sd:.4f})")
            diff = b_val - rl
            if diff > comb:
                print(f"      ✓ le RL la dépasse de {100*diff/b_val:.1f} % "
                      f"(écart {diff:.4f} > incertitude combinée {comb:.4f})")
            elif diff > 0:
                print(f"      ~ le RL est devant de {100*diff/b_val:.1f} % mais "
                      f"l'écart ({diff:.4f}) est INFÉRIEUR à l'incertitude "
                      f"combinée ({comb:.4f})")
                print(f"        → statistiquement non concluant : ne pas "
                      f"présenter cet écart comme un gain")
            else:
                print(f"      ✗ {best} reste meilleure que le RL de "
                      f"{100*(-diff)/rl:.1f} % — "
                      f"c'est le résultat à expliquer, pas à cacher")

    if not any("rl" in e for e in results.values()):
        print("\n  Aucun réseau RL évalué : lancer la brique 3 d'abord,")
        print("  ou vérifier --grid_x / --grid_y.")


def _plot(results, channels, out, data_info):
    methods = ["random", "variance", "eof_qr", "coverage", "rl"]
    labels = {"random": "Aléatoire", "variance": "Variance max",
              "eof_qr": "EOF + pivots QR", "coverage": "Couverture",
              "rl": "NAIADE (RL)"}
    cols = {"random": "#888780", "variance": "#EF9F27", "eof_qr": "#378ADD",
            "coverage": "#1D9E75", "rl": "#D4537E"}
    BG = "#0a1628"

    ns = sorted(int(k) for k in results)
    n_ch = len(channels)
    fig, axes = plt.subplots(1, n_ch + 1, figsize=(5 * (n_ch + 1), 4.6),
                             facecolor=BG)

    # Panneau 1 : RMSE agrégée vs N
    ax = axes[0]
    for m in methods:
        xs = [n for n in ns if m in results[str(n)]]
        if not xs:
            continue
        ys = [results[str(n)][m]["agg"] for n in xs]
        sd = [results[str(n)][m].get("agg_std", 0.0) for n in xs]
        if any(v > 0 for v in sd):
            ax.fill_between(xs, np.array(ys) - np.array(sd),
                            np.array(ys) + np.array(sd),
                            color=cols[m], alpha=0.20)
        style = "o-" if m != "rl" else "*-"
        ax.plot(xs, ys, style, color=cols[m], label=labels[m],
                lw=2, ms=12 if m == "rl" else 6)
    ax.set_title("RMSE agrégée (normalisée)", color="white", fontweight="bold")
    ax.set_xlabel("Nombre de capteurs", color="white")
    ax.set_ylabel("RMSE non observé", color="white")
    ax.legend(fontsize=8, facecolor=BG, labelcolor="white", framealpha=0.3)
    ax.tick_params(colors="white"); ax.set_facecolor("#050d1a")
    ax.grid(alpha=0.2)

    # Un panneau par canal, en unités physiques
    for k, c in enumerate(channels):
        ax = axes[k + 1]
        var = c.rsplit("_z", 1)[0]
        for m in methods:
            xs = [n for n in ns if m in results[str(n)]]
            if not xs:
                continue
            ys = [results[str(n)][m]["rmse"][c] for n in xs]
            style = "o-" if m != "rl" else "*-"
            ax.plot(xs, ys, style, color=cols[m], lw=2,
                    ms=12 if m == "rl" else 6)
        ax.set_title(f"{c}  [{UNITS.get(var, '')}]", color="white",
                     fontweight="bold")
        ax.set_xlabel("Nombre de capteurs", color="white")
        ax.tick_params(colors="white"); ax.set_facecolor("#050d1a")
        ax.grid(alpha=0.2)

    fig.suptitle("Placement de capteurs — NAIADE vs baselines\n"
                 "métrique : RMSE de reconstruction AE sur pixels non observés",
                 color="white", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "baselines_comparison.png", dpi=140,
                facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Figure → {out}/baselines_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Baselines de placement de capteurs")
    p.add_argument("--checkpoint", type=str, default="outputs/vae_best.pt")
    p.add_argument("--rl_checkpoint", type=str, default="outputs/rl_best.pt")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--n_sensors", type=int, nargs="+",
                   default=[5, 10, 20, 30, 40])
    p.add_argument("--n_repeat", type=int, default=10,
                   help="Tirages aléatoires pour la moyenne ± écart-type")
    p.add_argument("--n_eval", type=int, default=20, help="Dates d'évaluation")
    p.add_argument("--n_eval_repeat", type=int, default=4,
                   help="Répétitions d'évaluation des méthodes déterministes, "
                        "pour obtenir une barre d'erreur")
    p.add_argument("--n_mc", type=int, default=8)
    p.add_argument("--seed_eval", type=int, default=123)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--nt", type=int, default=None)
    p.add_argument("--seed_ocean", type=int, default=42)
    add_data_args(p)
    return p.parse_args()


if __name__ == "__main__":
    run_comparison(parse_args())
