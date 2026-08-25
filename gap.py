#!/usr/bin/env python3
"""
gapmap_patch.py
===============

Fixes where the autoencoder proposes new buoys.

Two problems with the current gap map in `01_autoencoder.py`.

**Proposals land in the corners.** The distance term is normalised by its own
global maximum:

    dist_to_sensor_n = dist_to_sensor / dist_to_sensor.max()
    gap_map = combined_sigma * dist_to_sensor_n

so the further from any sensor, the better, without limit. The farthest points
of a rectangular domain are always its corners, and they win almost every time.
Physically that is wrong: past the decorrelation scale a sensor constrains
nothing at all, so 200 km away is not twice as useful as 90 km away. The patch
**saturates the distance term at the influence radius**, which removes the
corner bias at its root rather than papering over it.

**Proposals land next to an existing buoy.** Nothing forbade it, and the MC
uncertainty often spikes right at the observation points, especially on a
short training run, so the argmax was drawn to those very pixels. The patch
adds an eligibility mask: at least `--gap_min_sep_px` from any existing sensor,
at least `--gap_margin_px` from the domain edge, and the same separation
between the proposals themselves.

Defaults, derived from the physics rather than picked by hand:

    influence radius   90 km / 5 km per pixel = 18 px   (config.INFLUENCE_RADIUS_KM)
    --gap_min_sep_px   18    one influence radius from any existing buoy
    --gap_margin_px     9    half a radius from the edge, so that a new buoy
                             does not spend half its footprint outside the domain
    --n_proposed        3    also accepted by run_demo.py, so the
                             pipeline and the standalone brick agree

If nothing is eligible the constraints are relaxed in steps rather than
failing, and the relaxation is printed.

Usage, from the root of your NAIADE clone:

    python gapmap_patch.py            # apply, keeps a .bak
    python gapmap_patch.py --revert
    python gapmap_patch.py --dry-run

Independent of the other patches; order does not matter.
"""

import argparse
import shutil
import sys
from pathlib import Path

TARGET = "01_autoencoder.py"
RD = "run_demo.py"
MARKER = "SATURATES at the influence radius"

# run_demo builds its own AE namespace, so the CLI flag on 01_autoencoder.py
# never reaches the pipeline. These two edits wire it through.
RD_ANCHOR_ARG = '''    p.add_argument("--ae_base_ch",  type=int, default=16)'''
RD_NEW_ARG = '''    p.add_argument("--ae_base_ch",  type=int, default=16)
    p.add_argument("--n_proposed",  type=int, default=3,
                   help="number of new buoys the AE proposes from the gap map, "
                        "and therefore the number the GNN then scores")
    p.add_argument("--gap_margin_px", type=float, default=None,
                   help="keep proposals this far from the domain edge")
    p.add_argument("--gap_min_sep_px", type=float, default=None,
                   help="keep proposals this far from existing sensors")'''

RD_ANCHOR_NS = '''        n_mc_val=3, n_mc=20, output_dir=str(out),'''
RD_NEW_NS = '''        n_mc_val=3, n_mc=20, output_dir=str(out),
        n_proposed=args.n_proposed,
        gap_margin_px=args.gap_margin_px,
        gap_min_sep_px=args.gap_min_sep_px,'''

# ---------------------------------------------------------------------------
ANCHOR = '''    # Coverage score: a gap is high sigma AND far from any sensor
    gap_map = combined_sigma * dist_to_sensor_n   # in [0, 1]
    gap_threshold = np.percentile(gap_map, 80)
    gap_binary = (gap_map > gap_threshold).astype(float)

    # -- 3 proposed buoys, maximising gap coverage ------------------------------
    # Greedy algorithm: at each step place the buoy at the maximum of the
    # residual gap_map, then update the distance to the nearest sensor.
    from scipy.ndimage import distance_transform_edt as _edt
    proposed_positions = []
    gap_residual = gap_map.copy()
    mask_augmented = mask_np.copy()
    for _ in range(3):
        flat_idx = np.argmax(gap_residual)
        px, py   = np.unravel_index(flat_idx, gap_residual.shape)  # px in [0,NX), py in [0,NY)
        proposed_positions.append((int(px), int(py)))
        mask_augmented[px, py] = 1.0
        dist_new = _edt(1 - mask_augmented) / (dist_to_sensor.max() + 1e-9)
        gap_residual = combined_sigma * dist_new
    proposed_arr = np.array(proposed_positions)  # (3, 2) -- (x, y) in pixels'''

NEW = '''    # Coverage score: a gap is high sigma AND far from any sensor.
    #
    # The distance term SATURATES at the influence radius. Past that scale a
    # sensor constrains nothing, so being 200 km from the nearest buoy is not
    # twice as valuable as being 90 km away. Dividing by the global maximum,
    # as an earlier version did, made the value grow without bound and the
    # corners of the domain won almost every time.
    infl_km = float(getattr(args, "gap_influence_km", INFLUENCE_RADIUS_KM))
    L_px    = max(1.0, infl_km / DX_KM)
    dist_sat = np.minimum(dist_to_sensor, L_px) / L_px

    gap_map = combined_sigma * dist_sat            # in [0, 1]
    gap_threshold = np.percentile(gap_map, 80)
    gap_binary = (gap_map > gap_threshold).astype(float)

    # -- Proposed buoys, maximising gap coverage --------------------------------
    # Greedy on the gap map, but only over positions that are actually
    # deployable: away from the edge, away from existing sensors, and away
    # from each other. Without this the argmax is drawn to the uncertainty
    # spikes that sit right on the observation points.
    # None is a valid value on the CLI and means "use the physical default",
    # so it has to be handled explicitly: getattr's fallback only fires when
    # the attribute is absent, not when it is present and None.
    n_prop = getattr(args, "n_proposed", None)
    n_prop = 3 if n_prop is None else int(n_prop)
    _m = getattr(args, "gap_margin_px", None)
    _s = getattr(args, "gap_min_sep_px", None)
    margin  = int(round(0.5 * L_px if _m is None else float(_m)))
    min_sep = float(L_px if _s is None else float(_s))

    gx, gy = np.meshgrid(np.arange(gap_map.shape[0]),
                         np.arange(gap_map.shape[1]), indexing="ij")

    def _eligible(margin_, min_sep_, taken):
        ok = np.zeros_like(gap_map, dtype=bool)
        m = max(0, int(margin_))
        ok[m:gap_map.shape[0] - m or None, m:gap_map.shape[1] - m or None] = True
        ok &= (dist_to_sensor >= min_sep_)
        for (tx, ty) in taken:
            ok &= ((gx - tx) ** 2 + (gy - ty) ** 2) >= min_sep_ ** 2
        return ok

    proposed_positions = []
    relaxed = False
    for _ in range(n_prop):
        ok = _eligible(margin, min_sep, proposed_positions)
        if not ok.any():
            # relax rather than fail: half the separation, then drop the
            # margin, then give up on the constraints entirely
            for m_, s_ in ((margin, 0.5 * min_sep), (0, 0.5 * min_sep), (0, 0.0)):
                ok = _eligible(m_, s_, proposed_positions)
                if ok.any():
                    relaxed = True
                    break
        if not ok.any():
            break
        scored = np.where(ok, gap_map, -np.inf)
        px, py = np.unravel_index(np.argmax(scored), scored.shape)
        proposed_positions.append((int(px), int(py)))

    proposed_arr = np.array(proposed_positions)    # (n, 2) -- (x, y) in pixels

    if len(proposed_positions):
        _d = [float(dist_to_sensor[px, py]) for px, py in proposed_positions]
        print(f"  Gap map: saturation {L_px:.0f} px, margin {margin} px, "
              f"min separation {min_sep:.0f} px"
              + ("  [relaxed, domain too crowded]" if relaxed else ""))
        print("  Distance from each proposal to the nearest existing buoy: "
              + ", ".join(f"{d:.0f} px" for d in _d))'''

# a matching CLI block, so the values can be tuned without editing the file
ANCHOR_TITLE = '''         f"Gap zones + 3 proposed buoys\\n"'''

NEW_TITLE = '''         f"Gap zones + {len(proposed_arr)} proposed "
         f"buoy{'s' if len(proposed_arr) != 1 else ''}\\n"'''

ANCHOR_ARGS = '''    p.add_argument("--n_mc",'''

NEW_ARGS = '''    p.add_argument("--n_proposed", type=int, default=3,
                   help="number of new buoys proposed from the gap map")
    p.add_argument("--gap_influence_km", type=float, default=INFLUENCE_RADIUS_KM,
                   help="scale at which the gap map distance term saturates")
    p.add_argument("--gap_margin_px", type=float, default=None,
                   help="keep proposals this far from the domain edge "
                        "(default: half an influence radius)")
    p.add_argument("--gap_min_sep_px", type=float, default=None,
                   help="keep proposals this far from existing sensors and from "
                        "each other (default: one influence radius)")
    p.add_argument("--n_mc",'''


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    path = root / TARGET
    bak = root / (TARGET + ".gapmap.bak")

    if not path.exists():
        sys.exit(f"error: {path} not found")

    if a.revert:
        n = 0
        for f in (TARGET, RD):
            b = root / (f + ".gapmap.bak")
            if b.exists():
                shutil.copy(b, root / f); b.unlink(); print(f"reverted {f}"); n += 1
        if n == 0:
            print("nothing to revert")
        return

    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print("already patched, nothing to do")
        return

    if src.count(ANCHOR) != 1:
        sys.exit(f"error: the gap map block was not found exactly once in "
                 f"{TARGET}. Patch by hand.")
    out = src.replace(ANCHOR, NEW)

    if out.count(ANCHOR_TITLE) == 1:
        out = out.replace(ANCHOR_TITLE, NEW_TITLE)
    else:
        print("  note: the figure title still says '3 proposed buoys'")

    if out.count(ANCHOR_ARGS) == 1:
        out = out.replace(ANCHOR_ARGS, NEW_ARGS)
    else:
        print("  note: could not add the CLI options, the defaults still apply")

    if a.dry_run:
        print("would saturate the distance term and constrain the proposals")
        return

    shutil.copy(path, bak)
    path.write_text(out, encoding="utf-8")
    print(f"patched {TARGET}  (backup -> {bak.name})")

    rd_path, rd_bak = root / RD, root / (RD + ".gapmap.bak")
    if rd_path.exists():
        rd = rd_path.read_text(encoding="utf-8")
        if "n_proposed=args.n_proposed" in rd:
            print(f"  {RD}: already wired")
        elif rd.count(RD_ANCHOR_ARG) == 1 and rd.count(RD_ANCHOR_NS) == 1:
            shutil.copy(rd_path, rd_bak)
            rd_path.write_text(rd.replace(RD_ANCHOR_ARG, RD_NEW_ARG)
                                 .replace(RD_ANCHOR_NS, RD_NEW_NS),
                               encoding="utf-8")
            print(f"patched {RD}  (--n_proposed now reaches the pipeline)")
        else:
            print(f"  note: could not wire {RD}, set n_proposed there by hand")
    print("\nProposals now keep one influence radius (18 px, 90 km) from any")
    print("existing buoy and half a radius from the domain edge.")


if __name__ == "__main__":
    main()
