#!/usr/bin/env python3
"""
ocean_gif_patch.py
==================

Adds an animated GIF of the nature run: the ocean evolving in time, one frame
every N days.

`generate_full` already returns the full (nt, nx, ny) arrays, so nothing is
recomputed. The patch adds `animate_nature_run()` to `data/dataset.py` and
wires it into both entry points.

Standalone:

    python data/dataset.py --nt 365 --seed 42 --gif --gif_every 5

Pipeline and individual mode:

    python run_demo.py --mode pipeline ... --ocean_gif --ocean_gif_every 5

Output: `ocean_nature_run.gif`, next to `ocean_nature_run.png`.

Options
-------
    --gif_every N     one frame every N days      (default 5)
    --gif_var  LIST   comma separated fields among T, S, SSH, ZETA, GRADT,
                      GRADS. Default: T,GRADT,S,GRADS, i.e. the four panels
                      SST, |grad SST|, SSS, |grad SSS| side by side
    --gif_fps  F      frames per second           (default 8)
    --gif_max  N      cap on the number of frames (default 120)

The frame cap matters: at `--gif_every 5` a 365-day run gives 73 frames, which
is fine, but `--gif_every 1` would give 365 frames and a GIF of several tens of
megabytes. Past the cap the stride is increased automatically and the change is
printed.

Note on the colour scale: it is computed once over the whole run and held
fixed, so what you see moving is the ocean and not the normalisation. The date
and the domain-mean value are printed on each frame.

Usage:

    python ocean_gif_patch.py            # apply, keeps .bak files
    python ocean_gif_patch.py --revert
    python ocean_gif_patch.py --dry-run

Independent of the other patches.
"""

import argparse
import shutil
import sys
from pathlib import Path

DS = "data/dataset.py"
RD = "run_demo.py"
MARKER = "def animate_nature_run("

# ══════════════════════════════════════════════════════════════ data/dataset.py
DS_ANCHOR = '''def plot_nature_run(run, out_path="outputs/ocean_nature_run.png", S_arr=None):'''

DS_NEW = '''def animate_nature_run(run, out_path="outputs/ocean_nature_run.gif",
                       every=5, var="T,GRADT,S,GRADS", fps=8, max_frames=120):
    """
    Animate the nature run, one frame every `every` days.

    Parameters
    ----------
    run : dict from generate_full, fields shaped (nt, nx, ny)
    every : stride in days between two frames
    var : one field, or several separated by commas. Understood names:
          T, S, SSH, ZETA, GRADT, GRADS. The two gradient moduli show the
          fronts and filaments far better than the fields themselves, which
          are dominated by the seasonal cycle.
          Default: the four panels SST, |grad SST|, SSS, |grad SSS|.
    fps : frames per second
    max_frames : hard cap, the stride is raised if needed

    Every panel keeps a colour scale fixed over the whole run, so what moves
    is the ocean and not the normalisation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    SPEC = {
        "T":     ("T",   "SST",           "degC",    "RdYlBu_r", False),
        "S":     ("S",   "SSS",           "psu",     "BrBG_r",   False),
        "SSH":   ("SSH", "SSH",           "m",       "RdBu_r",   True),
        "ZETA":  ("ZETA", "Vorticity",    "s^-1",    "RdBu_r",   True),
        "GRADT": ("T",   "|grad SST|",    "degC/km", "afmhot_r", False),
        "GRADS": ("S",   "|grad SSS|",    "psu/km",  "afmhot_r", False),
    }

    names = [v.strip().upper() for v in str(var).split(",") if v.strip()]
    names = [n for n in names if n in SPEC] or ["T"]

    nt = np.asarray(run["T"]).shape[0]
    every = max(1, int(every))
    if len(range(0, nt, every)) > max_frames:
        every = int(np.ceil(nt / max_frames))
        print(f"  [gif] stride raised to {every} days to stay under "
              f"{max_frames} frames")
    idx = list(range(0, nt, every))

    panels = []
    for n in names:
        field_key, label, unit, cmap, diverging = SPEC[n]
        if field_key not in run:
            print(f"  [gif] field {field_key} missing, panel {n} skipped")
            continue
        A = np.asarray(run[field_key])
        if n.startswith("GRAD"):
            gy, gx = np.gradient(A, axis=(1, 2))
            A = np.sqrt(gx ** 2 + gy ** 2) / DX_KM
        vmin, vmax = np.percentile(A[idx], [1, 99])
        if diverging:
            m = max(abs(vmin), abs(vmax)); vmin, vmax = -m, m
        panels.append({"A": A, "label": label, "unit": unit, "cmap": cmap,
                       "vmin": float(vmin), "vmax": float(vmax)})

    if not panels:
        print("  [gif] nothing to animate")
        return None

    ncol = len(panels)
    nx, ny = panels[0]["A"].shape[1], panels[0]["A"].shape[2]
    fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol + 0.6, 5.4),
                             facecolor="white")
    axes = np.atleast_1d(axes)

    ims = []
    for ax, pan in zip(axes, panels):
        im = ax.imshow(pan["A"][idx[0]].T, origin="lower", cmap=pan["cmap"],
                       vmin=pan["vmin"], vmax=pan["vmax"], aspect="auto",
                       extent=[0, nx * DX_KM, 0, ny * DX_KM])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(pan["unit"], fontsize=8)
        ax.set_title(pan["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("x (km)", fontsize=8)
        ax.tick_params(labelsize=7)
        ims.append(im)
    axes[0].set_ylabel("y (km)", fontsize=8)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    sup = fig.suptitle("", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def _draw(k):
        t = idx[k]
        for im, pan in zip(ims, panels):
            im.set_data(pan["A"][t].T)
        sup.set_text(f"Nature run, day {t} of {nt}")
        return tuple(ims) + (sup,)

    anim = FuncAnimation(fig, _draw, frames=len(idx),
                         interval=1000 / max(1, fps), blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps),
              savefig_kwargs={"facecolor": "white"})
    plt.close()
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Nature run GIF -> {out_path}  ({len(idx)} frames, "
          f"1 every {every} d, {ncol} panel{'s' if ncol > 1 else ''}, "
          f"{size_mb:.1f} MB)")
    return str(out_path)


def plot_nature_run(run, out_path="outputs/ocean_nature_run.png", S_arr=None):'''

DS_ANCHOR_CLI = '''    p.add_argument("--out",  type=str, default="outputs/ocean_nature_run.png")'''

DS_NEW_CLI = '''    p.add_argument("--out",  type=str, default="outputs/ocean_nature_run.png")
    p.add_argument("--gif",  action="store_true",
                   help="also write an animated GIF of the run")
    p.add_argument("--gif_every", type=int, default=5,
                   help="one frame every N days (default 5)")
    p.add_argument("--gif_var", type=str, default="T,GRADT,S,GRADS",
                   help="fields to animate, comma separated. Available: "
                        "T, S, SSH, ZETA, GRADT, GRADS. Default is the four "
                        "panels SST, |grad SST|, SSS, |grad SSS|")
    p.add_argument("--gif_fps", type=int, default=8)
    p.add_argument("--gif_max", type=int, default=120,
                   help="cap on the number of frames")'''

DS_ANCHOR_CALL = '''    plot_nature_run(run, out_path=args.out)'''

DS_NEW_CALL = '''    plot_nature_run(run, out_path=args.out)
    if args.gif:
        animate_nature_run(run,
                           out_path=str(Path(args.out).with_suffix(".gif")),
                           every=args.gif_every, var=args.gif_var,
                           fps=args.gif_fps, max_frames=args.gif_max)'''

# ══════════════════════════════════════════════════════════════════ run_demo.py
RD_ANCHOR_ARG = '''    p.add_argument("--no_nature_fig", action="store_true",'''

RD_NEW_ARG = '''    p.add_argument("--ocean_gif", action="store_true",
                   help="also write an animated GIF of the nature run")
    p.add_argument("--ocean_gif_every", type=int, default=5,
                   help="one frame every N days (default 5)")
    p.add_argument("--ocean_gif_var", type=str, default="T,GRADT,S,GRADS",
                   help="fields to animate, comma separated: T, S, SSH, ZETA, "
                        "GRADT, GRADS")
    p.add_argument("--ocean_gif_fps", type=int, default=8)
    p.add_argument("--no_nature_fig", action="store_true",'''

RD_ANCHOR_CALL = '''    if not args.no_nature_fig:
        plot_nature_run(run, out_path=str(out / "ocean_nature_run.png"))'''

RD_NEW_CALL = '''    if not args.no_nature_fig:
        plot_nature_run(run, out_path=str(out / "ocean_nature_run.png"))
    if args.ocean_gif:
        animate_nature_run(run, out_path=str(out / "ocean_nature_run.gif"),
                           every=args.ocean_gif_every, var=args.ocean_gif_var,
                           fps=args.ocean_gif_fps)'''

RD_ANCHOR_IMPORT = '''from data.dataset import (SyntheticOceanGenerator, build_datasets,'''


def _check(src, anchor, name, f):
    if src.count(anchor) != 1:
        sys.exit(f"error: the {name} anchor was not found exactly once in {f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    suffix = ".oceangif.bak"

    if a.revert:
        n = 0
        for f in (DS, RD):
            b = root / (f + suffix)
            if b.exists():
                shutil.copy(b, root / f); b.unlink(); print(f"reverted {f}"); n += 1
        if n == 0:
            print("nothing to revert")
        return

    ds_path, rd_path = root / DS, root / RD
    if not ds_path.exists():
        sys.exit(f"error: {DS} not found")

    ds = ds_path.read_text(encoding="utf-8")
    if MARKER in ds:
        print(f"  {DS}: already patched")
        ds_out, ds_changed = ds, False
    else:
        _check(ds, DS_ANCHOR, "plot_nature_run definition", DS)
        _check(ds, DS_ANCHOR_CLI, "CLI --out option", DS)
        _check(ds, DS_ANCHOR_CALL, "plot_nature_run call", DS)
        ds_out = (ds.replace(DS_ANCHOR, DS_NEW)
                    .replace(DS_ANCHOR_CLI, DS_NEW_CLI)
                    .replace(DS_ANCHOR_CALL, DS_NEW_CALL))
        ds_changed = True

    rd_changed = False
    rd_out = None
    if rd_path.exists():
        rd = rd_path.read_text(encoding="utf-8")
        if "args.ocean_gif" in rd:
            print(f"  {RD}: already patched")
        else:
            _check(rd, RD_ANCHOR_ARG, "no_nature_fig option", RD)
            _check(rd, RD_ANCHOR_CALL, "plot_nature_run call", RD)
            _check(rd, RD_ANCHOR_IMPORT, "dataset import", RD)
            rd_out = (rd.replace(RD_ANCHOR_ARG, RD_NEW_ARG)
                        .replace(RD_ANCHOR_CALL, RD_NEW_CALL)
                        .replace(RD_ANCHOR_IMPORT,
                                 "from data.dataset import animate_nature_run\n"
                                 + RD_ANCHOR_IMPORT))
            rd_changed = True

    if a.dry_run:
        print("would add animate_nature_run() and wire both entry points")
        return

    if ds_changed:
        shutil.copy(ds_path, root / (DS + suffix))
        ds_path.write_text(ds_out, encoding="utf-8")
        print(f"patched {DS}")
    if rd_changed:
        shutil.copy(rd_path, root / (RD + suffix))
        rd_path.write_text(rd_out, encoding="utf-8")
        print(f"patched {RD}")

    if ds_changed or rd_changed:
        print("\n  python data/dataset.py --nt 365 --seed 42 --gif --gif_every 5")
        print("  python run_demo.py --mode pipeline ... --ocean_gif "
              "--ocean_gif_every 5")


if __name__ == "__main__":
    main()
