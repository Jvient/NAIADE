# Maintenance-aware observing network design

An extension to NAIADE that replaces the `cost ∝ number of buoys` proxy with an
explicit model of keeping buoys alive at sea — and, crucially, **connects that
cost back to information** instead of bolting it on as a second Pareto axis.

```
budget → affordable campaigns → visit interval per buoy
       → data availability    → effective observation error
       → variance explained by the network
```

An unmaintained buoy is not a free buoy. It is a buoy that breaks down and
stops carrying information.

---

## Contents

1. [The mechanism](#1-the-mechanism)
2. [What is established](#2-what-is-established)
3. [What did not work](#3-what-did-not-work)
4. [Modules and commands](#4-modules-and-commands)
5. [Domain presets](#5-domain-presets)
6. [Criterion calibration](#6-criterion-calibration)
7. [Multi-year scenarios and maintenance policies](#7-multi-year-scenarios-and-maintenance-policies)
8. [Sequential evaluation](#8-sequential-evaluation)
9. [Realistic observation masks](#9-realistic-observation-masks)
10. [Known limitations](#10-known-limitations)
11. [Roadmap](#11-roadmap)

---

## 1. The mechanism

### Availability

Failure (or loss, or vandalism) is drawn from an exponential distribution with
mean `MTBF` and repaired only at the next ship visit. Over a visit interval
`Δ`, availability is

```
a(Δ) = (1 − e^(−Δ/MTBF)) / (Δ/MTBF)
```

With the `pirata` profile (MTBF = 420 d): **0.28 / 0.67 / 0.81** for 0, 1 and 2
annual campaigns. Monotone, saturating — diminishing returns on maintenance
come out of the model rather than being imposed on it.

### How availability enters the information criterion

Let `m_i ~ Bernoulli(a_i)` be the presence indicator and `z_i = m_i y_i` the
observation actually received. Then

```
Cov(z_i, z_j) = a_i a_j C_ij     (i ≠ j)
Var(z_i)      = a_i (C_ii + R_ii)
Cov(z_i, x_c) = a_i C_ic
```

so the BLUE built on `z` is the nominal BLUE with inflated noise:

```
R_eff_i = R_ii / a_i  +  C_ii (1 − a_i) / a_i
```

A **diagonal update**, therefore free. The second term dominates: at `a = 0.5` a
buoy carries noise of the order of the signal variance itself.

Validity, measured against explicit gap sequences (`scenario.py`): the
approximation holds to within 5 % as long as the budget affords at least one
annual campaign, and breaks down badly (+47 %) under chronic underfunding,
where the network decays without ever recovering. A stationary intermittency
model cannot represent a drift.

### Cost and campaign planning

Annual cost = amortised hardware + days at sea + mobilisation per leg +
consumables. The route is a nearest-neighbour tour refined by 2-opt from port,
split into legs that respect ship endurance. Stations whose round trip alone
exceeds endurance are declared **out of reach** and excluded — on a
2560 × 3840 km domain, 14 of 25 buoys are unreachable by a coastal vessel and 0
by an offshore one.

Campaigns are **nested**: campaign *k* visits every buoy with service level
≥ *k*. When a campaign does not fit the budget, buoys are dropped iteratively by
worst *relevance / detour saving* ratio.

---

## 2. What is established

### The operational rule — the headline result

> **Do not deploy what you cannot service at least once a year.**

A network designed while ignoring maintenance loses **50 % to 96 %** of the
information delivered by a network designed with that single rule. Five ocean
seeds, two domains (800 × 1200 km and 2560 × 3840 km), two ship profiles,
paired comparison, 5/5 seeds at every constrained budget level.

| Budget (`large`, pirata) | integrated | naive control | gain |
|---:|---:|---:|---:|
| 1090 k€ | 0.0463 (N=9, avail. 0.54) | 0.0290 (N=30, avail. 0.28) | **+52.6 % ± 15.8** |
| 1530 k€ | 0.0621 (N=13, avail. 0.61) | 0.0467 (N=30, avail. 0.33) | **+31.3 % ± 11.6** |
| 2400 k€ | 0.0721 | 0.0672 | +4.0 % ± 6.4 (n.s.) |
| 4360 k€ | 0.0940 | 0.0936 | +2.6 % ± 2.6 (n.s.) |

The mechanism is direct: **nine maintained buoys beat thirty abandoned ones.**
The effect switches off exactly where theory says it should — once the budget
maintains everyone, design no longer matters.

### The tooling

A campaign planner producing **ship trajectories** and a cost breakdown
(hardware / sea days / consumables) plus CO₂, not an abstract scalar. Budget
return curves: information per k€ falls by a factor of 3 across the swept
range, which reframes the question put to a funder — not "how much information"
but "how much information per additional k€".

### A finding about Brick 2

Brick 2's current training target (`1 − mean correlation`) is **uncorrelated
with marginal sensor contribution**: Spearman **−0.02 ± 0.08** over three seeds.
A GNN trained on the true target reaches **+0.52 ± 0.06**, against +0.22 for the
local-variance proxy. If Brick 2 is presented as ranking sensor relevance, it
does not.

### Adaptive maintenance beats a fixed plan

Deciding what to repair from the **observed** state of the network, rather than
applying a fixed annual plan, is worth **+19 %** of integrated information at
constant sea time (34 days/year either way). The fixed plan wastes days visiting
live buoys.

---

## 3. What did not work

Reported because it constrains what can be claimed.

**Reinforcement learning for static placement.** PPO reaches roughly **half**
the information of a greedy baseline at every budget level, and below the naive
control. Mean episode reward stays negative throughout training; the best
episode plateaus at 18 000 steps out of 100 000. The problem as posed —
one-shot placement under a fixed budget — is deterministic combinatorics, and
greedy is the right tool.

**Placement optimisation against a sensible operator.** Against a control that
ignores effective information but refuses to deploy what it cannot service once
a year, the integrated design gains only **+0.5 % to +6.8 %**, with at least one
losing seed at every level and standard deviations of the same order. No level
is conclusive. The value lies in the rule, not in the optimiser. The +50–96 %
figures above are measured against a control that degenerates at low budget — it
buys 26 moorings and has nothing left to charter a ship.

**The GNN as a decision input.** Despite doubling the Spearman score, feeding
GNN relevance into the planner changes the decision by nothing measurable, and
conditioning relevance on the surviving buoys does not help either. The exact
leave-one-out costs 27 evaluations per departure — nothing. The GNN would only
pay off on much larger networks or inside a learning loop.

**A learned recurrent reconstruction as an estimator.** The nature run advects
**passive** tracers: `_velocity(t)` never depends on T or S. The system is
linear with a time-varying operator, so a Kalman filter using the true operator
is optimal by construction — and that operator is *computable* (`ceiling.py`).
No learned model can beat it, and there is nothing to approximate that is not
already available exactly.

---

## 4. Modules and commands

| Module | Role |
|---|---|
| `maintenance.py` | Upkeep model: availability, costs, routing, budget-constrained planning |
| `campaign.py` | Budget-constrained greedy design, campaign demo, figures |
| `priority.py` | GNN predicting true marginal contribution (Brick 2 → Brick 3 coupling) |
| `scenario.py` | Multi-year failure scenarios, real gap masks, saw-tooth curves |
| `policy.py` | Adaptive maintenance policies plus an oracle ceiling |
| `kalman.py` | Sequential EOF/LIM evaluator with temporal memory |
| `ceiling.py` | Exact time-varying propagator — the computable upper bound |
| `maint_masks.py` | Maintenance-driven observation masks for Brick 1 |
| `diag_priority.py`, `diag_ar1.py` | Diagnostics (see §6) |
| `test_replication.py`, `test_priority.py`, `compare_masks.py` | Multi-seed validation harnesses |

### Core commands

```bash
# Campaign demo: optimal network and ship route per budget level
NAIADE_DOMAIN=large python 03_rl.py --maintenance pirata --campaign \
    --influence_fit --evf_cv 1 --n_max 30

# Main result, replicated across seeds, three control arms
NAIADE_DOMAIN=large python test_replication.py --seeds 42 43 44 45 46 \
    --maintenance pirata --n_max 30

# Multi-year scenarios and saw-tooth curves
NAIADE_DOMAIN=large python scenario.py --maintenance pirata --n_max 30

# Adaptive maintenance policies and the margin left for an agent
NAIADE_DOMAIN=large python policy.py --maintenance pirata --n_max 30 --gnn

# Sequential evaluator instead of the instantaneous criterion
NAIADE_DOMAIN=large python policy.py --maintenance pirata --n_max 30 \
    --evaluator kalman

# Computable ceiling for a learned reconstruction
NAIADE_DOMAIN=large python ceiling.py --maintenance pirata --n_max 30

# Realistic vs random observation masks for Brick 1
NAIADE_DOMAIN=large python compare_masks.py --maintenance pirata \
    --n_max 30 --epochs 30 --n_draws 48 --mix_random 0.3
```

### Reinforcement-learning options (Brick 3)

| Option | Effect |
|---|---|
| `--maintenance off\|regional\|pirata` | `off` preserves the historical behaviour exactly |
| `--budget_keur B` | Hard constraint: exact capex ceiling in the action mask, penalty on sea-day overrun |
| `--reward_mode ratio` | Change in information / upkeep cost. Telescopes over an episode to `E(final) − E(initial)` |
| `--reward_mode penalized` | `information − w_cost × cost` |
| `--maint_refine` | Full 2-opt during training (~3× slower) |

Two scalars (cost/budget, mean availability) are appended to the observation
vector when maintenance is active — without them the policy is blind to the
constraint it is asked to respect. **Consequence: checkpoints are not portable
between the two modes.** The code detects the mismatch and skips it cleanly.

---

## 5. Domain presets

Enlarging the domain is not just a matter of increasing `NX`/`NY`: every length
scale must follow, or you get three-pixel eddies. A factor `s = dx/5 km` is
applied to jet width, eddy radii, deformation radius and minimum buoy
separation; eddy count follows cell count, preserving areal density.

| | grid | domain | eddies | ΔT north–south | `N_BUOYS` |
|---|---|---|---|---:|---:|
| `demo` | 160×240 @ 5 km | 800 × 1200 km | 22 (35–75 km) | 9.0 °C | 30 |
| `large` | 320×480 @ 8 km | 2560 × 3840 km | 88 (56–120 km) | 16.1 °C | 60 |
| `basin` | 384×512 @ 12 km | 4608 × 6144 km | 113 (84–180 km) | 20.4 °C | 68 |

Select with `NAIADE_DOMAIN=large python …`, or by editing `DOMAIN` in
`config.py`. The environment variable was the only clean route given the
pervasive `from config import *`. The EVF evaluation stride follows the domain,
keeping criterion cost roughly constant (~600 cells).

**Changing domain invalidates Brick 1 checkpoints** (input size) and every
diagnosed statistic.

---

## 6. Criterion calibration

**Read this section before trusting any number produced by this code.** Several
errors corrected during development were calibration errors that produced
plausible, wrong results without raising any exception.

### The influence radius must be fitted, not derived

`INFLUENCE_RADIUS_KM` is scaled with the domain, and **that is an unreliable
fallback**. On `demo` it matches the diagnostic (90 configured vs 95 measured);
on `large` it gives 144 km where the run diagnoses **416–432**.

Neither is right. `--influence_auto` reads `L_decorr_SST_km`, a **zonal**
decorrelation measured on a **single snapshot**. The jet is zonally coherent by
construction, so that measurement captures the jet rather than the mesoscale.
And the kernel is isotropic — calibrating it on one direction makes it wrong in
the other.

**Use `--influence_fit`**, which fits `L` on the empirical correlation of all
candidate pairs, all directions, all time steps, using exactly the functional
form the model employs: 52 km on `demo` (vs 90 configured), 110 km on `large`
(vs 144 configured, 416 diagnosed zonally). Out-of-sample EVF improves by 52 %.

```bash
python 03_rl.py --influence_fit --evf_cv 1 ...
```

### The kernel remains a crude approximation

The fit reports its residual RMS on correlation: **0.17 to 0.19**. No kernel
family does better — Gaussian 0.171, exponential 0.170, Matérn 3/2 and 5/2
0.170 — because the empirical correlation **is not a decreasing function of
distance**. It reaches −0.69: gyre lobes and jet meanders create anticorrelated
structures at distance that no monotone isotropic kernel can represent.

**Practical consequence: differences below a few percent are not
interpretable.** Always work with `--evf_cv 1`.

### Do not lower the shrinkage

Relying more on the empirical covariance makes things far worse. Measured
out-of-sample on `demo` (20 buoys):

| shrinkage | 0.0 | 0.2 | 0.5 | 0.7 | **0.9** | 0.95 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| out-of-sample EVF | −0.63 | −0.07 | 0.12 | 0.18 | **0.19** | 0.18 | 0.16 |

Negative EVF means the network predicts worse than climatology. The empirical
covariance is estimated in dimension 768 from about fifteen independent
realisations — it is noise. The 0.9 default is the optimum. `--evf_shrink` now
actually reaches the environment (it was parsed but never passed through).

### Check that the lever exists before comparing

`diag_priority.py` verifies that the maintenance priority has any grip at all at
a given budget. If the budget comfortably funds every campaign, or funds none,
the iterative drop never triggers and **any** priority yields the same plan.
Three relevance sources once produced results identical to the last digit; that
was not "no gain", it was no measurement.

```bash
NAIADE_DOMAIN=large python diag_priority.py --maintenance pirata \
    --budgets 600 900 1200 1530 2000 2400 --sizes 11 16 25 30
```

### Watch for saturation

`ceiling.py` warns when explained variance exceeds 0.90. In that regime the
metric crushes differences and the two readings diverge wildly: +3.8 % in
explained variance, −82 % in residual error, for the same quantity. Adding modes
does not help — the saturation is physical (see §11).

---

## 7. Multi-year scenarios and maintenance policies

`scenario.py` simulates explicit failure trajectories over several years:
exponential failures, repairs on the date the **ship actually reaches** each
buoy (derived from the planned route, leg by leg), and the criterion evaluated
at every time step with the real mask — no averaging.

The output is a **saw-tooth curve**: information erodes as buoys die, and
recovers at each campaign. Flat means the budget is sufficient; a downward slope
across years means the network is doomed. Readable in three seconds by an
operations manager.

`policy.py` compares adaptive policies on common random draws (all policies
replay identical lifetime sequences, so differences come from decisions, not
luck):

| policy | vs fixed plan (`large`) |
|---|---:|
| fixed plan | +0.0 % |
| oldest first | +9.9 % |
| dead first | +17.1 % |
| cheapest first | +18.4 % |
| by contribution | +19.4 % |
| oracle | +19.6 % |

**Margin left for a learned agent: +0.2 %.** The simple heuristic already
reaches the oracle. Under the Kalman evaluator, *cheapest first* even beats the
oracle — with temporal memory, doing many cheap repairs beats doing few relevant
ones.

The operational rule: **repair what is dead and close to port. Relevance
ordering changes nothing.**

Caveat: the oracle is myopic over one campaign, so it only tests *ordering
within* a campaign. It cannot express "skip this repair to save budget" or
"shift a campaign date". If RL has a chance here, that is where it is.

---

## 8. Sequential evaluation

The instantaneous criterion says a dead buoy stops informing immediately, and a
repaired buoy informs fully on day one. Both are wrong. `kalman.py` provides an
EOF-space Kalman filter in which information persists:

```
        day 99  day 101  day 110  day 140  day 300
kalman   0.753   0.707    0.587    0.400    0.121
static   0.753   0.000    0.000    0.000    0.000
```
*(total network extinction on day 100, never repaired)*

Two properties keep it cheap and exact. In the linear-Gaussian case the error
covariance `P` depends **only on the mask sequence**, not on observed values — so
`P` is propagated alone, without simulating any measurement. And buoys are
regressed onto the EOFs rather than interpolated, the residual variance becoming
a representativeness error added to `R`.

**Bonus: no Gaussian kernel needed.** The EOFs carry the true spatial structure,
anticorrelations included.

### Why the propagator is a full matrix, not an AR(1) per mode

The first version used `A = diag(a_i)` and was wrong by a factor of 10 on
decorrelation time (`diag_ar1.py`, `large`):

```
mode        0     1     2     3     7
empirical  90 d  53 d  25 d  28 d  21 d
AR(1)      90+   90+   90+   90+   90+     ratio ×3.3 (capped by max lag)
LIM        88 d  53 d  25 d  27 d  20 d    ratio ×1.0
```

Fitted at lag 1, an AR(1) reads an autocorrelation of 0.999 on a spatially smooth
field and extrapolates years of memory. And `a^lag` is positive by construction,
so it cannot represent the **negative** autocorrelations observed on most modes —
the signature of a structure propagating through and re-exciting the EOF in
reverse. A full matrix admits complex eigenvalues, hence rotation and
propagation, at identical cost.

Judge propagators on **implied decorrelation** `[A^lag C0]_ii / C0_ii`, never on
one-day forecast skill: the AR(1) scores 0.97 there while being wrong by a factor
of 10 on memory.

---

## 9. Realistic observation masks

Brick 1 trains on `_random_mask`: 5 to 60 pixels drawn uniformly,
**independently at every time step**. That is not an observing network, it is a
scatter that teleports daily.

```
Shared positions over 6 consecutive days
  random mask (current)      0 %
  maintenance mask          50 %

Gap persistence (maintenance mask, large domain)
  N= 9 | avail. 0.73 | mean outage 133 d | max 362 d
  N=26 | avail. 0.43 | mean outage 550 d | max 973 d
  N=29 | avail. 0.31 | mean outage 736 d | max 992 d
```

Independent gaps average out; a buoy dead for 550 days digs a structural hole.

Measured with `compare_masks.py` (cross-evaluation on a frozen validation set,
`large`, 30 epochs, 48 draws, 30 % mixing):

| trained on | eval random | eval maintenance |
|---|---:|---:|
| random | 0.2690 | 0.2861 |
| maintenance | 0.2713 | **0.2661** |

**The old protocol overstates Brick 1 by 6 %**, and realistic training recovers
**7 %** in the regime that matters, at no generalisation cost.

Draw diversity is critical: with only 8 network draws, the model memorises those
positions and collapses by 36 % outside its own regime while showing the *best*
training loss. Use `--n_draws 48` and `--mix_random 0.3`; the random mask acts as
data augmentation. Validation stays maintenance-pure.

---

## 10. Known limitations

**Maintenance parameters are invented.** MTBF 420 days, 30 k€/ship-day, 1 day on
station: plausible orders of magnitude, not SNO figures. The MTBF alone drives
the entire availability curve, therefore the whole coupling. It is the first
number to have validated by operations, and the first thing anyone will ask
about. Everything lives in `MAINT_PROFILES` (`maintenance.py`).

**The greedy is myopic.** Discrepancies of a few percent sit at its noise level,
in both directions. A swap-based local search would settle them; it is not
implemented.

**Priority inside the RL loop is a proxy** (local variability), to avoid the
circularity availability → information → priority → availability. The true
leave-one-out is used only for the final replanning, outside the loop.

**Budget masking covers capex only** — an exact, instant ceiling. Sea-day
overruns remain a soft penalty; evaluating them per candidate would cost K
campaign plans per step.

**Single ocean realisation** for everything except `test_replication.py` and
`test_priority.py`.

**Routing is heuristic.** Nearest neighbour plus 2-opt is not optimal, nested
campaigns are one policy among many, and the greedy drop has no guarantee.

---

## 11. Roadmap

### The perfect-model pitfall

`ceiling.py` computes the exact ceiling by capturing `A_t = Eᵀ M_t E` during a
replay of the nature run: `_departure_stencil` is temporarily wrapped so the
generator's trajectory is reproduced identically while the operator is captured
in passing.

The result exposes a deeper problem. Free forecast error, with no observations
at all:

```
  1 day    1 % of variance
  5 days   6 %
 10 days  13 %
 30 days  35 %
```

A model that loses only 13 % over ten days barely needs observations. The filter
reconstructs the state dynamically and 27 buoys lock down the rest — hence
saturation at any mode count. The root cause is structural: our "model" is a LIM
fitted **on the nature run itself**, i.e. a near-perfect-model OSSE.

Any DA-based network evaluation in this setting will conclude that any network
will do. The stage-1 saw-tooth curves rest on the instantaneous criterion and are
unaffected, but future DA-based evaluation needs a realistic model-error
inflation — a modelling choice that determines everything downstream.

### Next step: a chaotic ocean

Moving to a nature run with state-dependent dynamics fixes all three limits at
once: the Gaussian kernel that cannot represent anticorrelations, the linearity
that makes the filter optimal by construction, and above all the excessive
predictability.

Candidates, all free runs (no assimilation), which is the point — a reanalysis
such as GLORYS already contains the imprint of the very network under
evaluation, and smoothing does not remove it, because the imprint is localised
where the observations are rather than confined to small scales.

- **NATL60-CJM165 / eNATL60-BLB002** via the MEOM ocean data challenges, which
  distribute daily resamplings at 1/20° and 1/8° — the subsetting work is already
  done. The Gulf Stream box (65°W–55°W, 33°N–43°N) has extreme mesoscale
  heterogeneity, so placement actually has something to bite on, and results
  become comparable to the community mapping benchmark.
- eNATL60-BLB002 additionally covers the tropical and equatorial Atlantic, for a
  PIRATA-like configuration.

Three adaptations to expect: those datasets distribute **SSH**, not SST/SSS;
NATL60 covers **one year only**, so multi-year maintenance scenarios must recycle
it; and the influence radius must be re-fitted from scratch, on a field that will
be strongly **anisotropic** — one more argument for the Kalman evaluator, which
needs no kernel.

First test before porting anything: fit a LIM on the target box and look at
variance lost at ten days. If it is 40–50 % rather than 13 %, every question
closed above as "not applicable" reopens — the recurrent autoencoder, RL for
maintenance, and the value of dynamic conditioning.
