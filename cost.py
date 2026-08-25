#!/usr/bin/env python3
"""
cost_compare_patch.py
=====================

Adds a step 1b to `run_demo.py --mode pipeline`: a side-by-side map of the
network the RL agent retained on information alone, against the network you
get at the *same size* once operating cost and carbon enter the objective.

Why same size. At a fixed number of buoys the cost still varies by a factor
1.3 to 1.6, because the maintenance tour from the port depends on how spread
out the network is. Holding N constant therefore isolates exactly the effect
you want to show: the geometry changes, not the budget line. Comparing two
networks of different sizes would confound the two.

How the second network is built. Greedy on the scalarised criterion
`info - lambda * cost`, which is already in `03_rl.py` as `_greedy_weighted`.
Lambda is calibrated so that the cost term is worth roughly as much as the
information term at the retained size, then the sweep keeps the value that
gives the best cost saving for an information loss under a tolerance you set
(default 10 %).

The figure, `rl_info_vs_cost_networks.png`, shows both networks over the local
variance field, with the shared positions in grey, the information-only ones
in red and the cost-aware ones in green, plus a bar showing information, cost
and CO2 for each.

Usage, from the root of your NAIADE clone:

    python cost_compare_patch.py            # apply, keeps a .bak
    python cost_compare_patch.py --revert   # undo
    python cost_compare_patch.py --dry-run

New options on run_demo.py:

    --no_cost_compare        skip step 1b
    --cost_info_tol F        max acceptable information loss, 0-1 (default 0.10)
    --cost_n_lambda N        number of lambda values swept (default 8)

Apply after `pipeline_inductive_patch.py` if you use both; they touch
different anchors and do not conflict.
"""

import argparse
import shutil
import sys
from pathlib import Path

TARGET = "run_demo.py"
MARKER = "STEP 1b/3"

# ---------------------------------------------------------------- CLI options
ANCHOR_ARGS = '''    p.add_argument("--output_dir",  type=str, default="outputs")'''

NEW_ARGS = '''    p.add_argument("--no_cost_compare", action="store_true",
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
    p.add_argument("--output_dir",  type=str, default="outputs")'''

# ---------------------------------------------------------------- helper fn
ANCHOR_HELPER = '''def _report_header(mode, args, T, positions, ts, ocean_diag=None):'''

NEW_HELPER = '''def _compare_info_vs_cost(b3, env, rl_positions, best_mask, args, out):
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
         f"Information only\\nN={n_star}  info={info_a:.3f}  "
         f"{cost_a:.0f} kEUR/yr  {co2_a:.0f} tCO2/yr"),
        (axes[1], idx_cost, only_b, "#1e8449",
         f"Information and cost\\nN={len(idx_cost)}  info={best['info']:.3f}  "
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


def _report_header(mode, args, T, positions, ts, ocean_diag=None):'''

# ---------------------------------------------------------------- the step
ANCHOR_STEP = '''    # -- Step 2: GNN scores the RL network --------------------------------------'''

NEW_STEP = '''    # -- Step 1b: same network with cost and carbon in the objective ------------
    if not args.no_cost_compare:
        print(f"\\n{SEP}\\n  STEP 1b/3 -- RL: information alone against information "
              f"and cost\\n{SEP}")
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
            ] + ["", "-- COST-AWARE POSITIONS (pixel x, y) --------------------------------"] \\
              + [f"  C{i:02d} : ({px:4d}, {py:4d})"
                 for i, (px, py) in enumerate(m_cmp["cmp_positions_cost"])]

    # -- Step 2: GNN scores the RL network --------------------------------------'''


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    path = root / TARGET
    bak = root / (TARGET + ".costcmp.bak")

    if not path.exists():
        sys.exit(f"error: {path} not found")

    if a.revert:
        if bak.exists():
            shutil.copy(bak, path); bak.unlink(); print(f"reverted {TARGET}")
        else:
            print("nothing to revert")
        return

    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print("already patched, nothing to do")
        return

    for anchor, name in ((ANCHOR_ARGS, "output_dir argument"),
                         (ANCHOR_HELPER, "_report_header definition"),
                         (ANCHOR_STEP, "pipeline step 2 header")):
        if src.count(anchor) != 1:
            sys.exit(f"error: the {name} anchor was not found exactly once in "
                     f"{TARGET}. Patch by hand.")

    out = (src.replace(ANCHOR_ARGS, NEW_ARGS)
              .replace(ANCHOR_HELPER, NEW_HELPER)
              .replace(ANCHOR_STEP, NEW_STEP))

    if a.dry_run:
        print("would add 3 CLI options, one helper and step 1b")
        return

    shutil.copy(path, bak)
    path.write_text(out, encoding="utf-8")
    print(f"patched {TARGET}  (backup -> {bak.name})")
    print("\nNew figure: rl_info_vs_cost_networks.png")


if __name__ == "__main__":
    main()
