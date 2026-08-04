# NAIADE

**Optimal Experimental Design for marine observing networks, with AI.**

NAIADE is a Python/PyTorch framework for designing, scoring and evolving ocean
observing networks.

Everything runs in an OSSE framework: a synthetic ocean plays the role of
ground truth, so any network configuration can be evaluated against a known
answer.

```
                     ┌──────────────────────────────┐
                     │   Synthetic ocean (nature run)│
                     │   SST · SSS · SSH · 2D+T      │
                     └───────────────┬──────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
   ┌───────────┐              ┌────────────┐             ┌──────────────┐
   │  Brick 1  │              │  Brick 2   │             │   Brick 3    │
   │ Autoencoder│             │    GNN     │             │      RL      │
   │ observability            │ network    │             │ network      │
   │ & gap maps │             │ structure  │             │ optimisation │
   └───────────┘              └────────────┘             └──────────────┘
   Where is the network       Which sensors are          How many sensors,
   blind?                     redundant?                 where, at what cost?
```

---

## Table of contents

0. [Gallery](#0-gallery)
1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [The synthetic ocean](#3-the-synthetic-ocean)
4. [Brick 1 — Observability autoencoder](#4-brick-1--observability-autoencoder)
5. [Brick 2 — Graph neural network](#5-brick-2--graph-neural-network)
6. [Brick 3 — Reinforcement learning](#6-brick-3--reinforcement-learning)
7. [Orchestrator](#7-orchestrator)
8. [Configuration reference](#8-configuration-reference)
9. [Output files](#9-output-files)
10. [Known limitations](#10-known-limitations)
11. [Citing NAIADE](#11-citing-naiade)
12. [References](#12-references)

---

## 0. Gallery

**[→ Full result gallery with reproducible commands](docs/GALLERY.md)**

Every figure below comes from `--seed_ocean 42 --seed_buoys 7 --nt 365`.

![Nature run](docs/figures/01_nature_run.png)

*The synthetic ocean. Fronts and filaments are not drawn — they emerge from the
competition between geostrophic stirring and restoring towards climatology.*

![Pareto front](docs/figures/09_rl_pareto_front_info_vs_N.png)

*Information vs number of buoys. **N★ = 23, explaining 63.7 % of mesoscale
variance — 23 optimised buoys are worth 43 randomly placed ones.***

| annual budget | N | explained variance | tCO₂/yr |
|---|---|---|---|
| 500 k€ | 5 | 0.276 | 238 |
| 900 k€ | 15 | 0.525 | 379 |
| 1400 k€ | 28 | 0.687 | 553 |

*Budget-constrained optimal networks, from the information/cost Pareto front.*

---

## 1. Install

```bash
git clone https://github.com/Jvient/NAIADE
cd NAIADE
pip install torch numpy scipy matplotlib
pip install torch-geometric        # optional — Brick 2 falls back to a
                                   # hand-written GAT if it is missing
```

Repository layout:

```
NAIADE/
├── config.py              all physical and methodological parameters
├── data/
│   └── dataset.py         ocean generator + PyTorch datasets + shared utilities
├── 01_autoencoder.py      Brick 1
├── 02_gnn.py              Brick 2
├── 03_rl.py               Brick 3
├── run_demo.py            orchestrator (individual | pipeline)
└── outputs/               everything the code writes
```

Everything runs on CPU. A GPU is used automatically if available
(`config.DEVICE`).

---

## 2. Quick start

Generate the ocean and look at it:

```bash
python data/dataset.py --nt 365 --seed 42
```

Run the whole framework end to end:

```bash
python run_demo.py --mode pipeline --nt 365 --seed_ocean 42 --seed_buoys 7
```

Two orchestration modes:

| mode | what it does |
|---|---|
| `individual` | the three bricks run independently on the same ocean and the same initial network — use it to compare bricks |
| `pipeline` | RL proposes an optimal network → the GNN scores its structure → the autoencoder maps its blind spots |

---

## 3. The synthetic ocean

### 3.1 What it is

`data/dataset.py` produces a 2D+T *nature run*: a physically consistent surface
ocean over a zonal channel, in real units.

| property | value |
|---|---|
| domain | 800 × 1200 km (160 × 240 points, Δx = 5 km) |
| central latitude | 42°N (β-plane) |
| time step | 1 day |
| variables | SST (°C), SSS (psu), SSH (m), u, v (m/s), ζ (s⁻¹), σ₀ (kg/m³) |

### 3.2 How it works

The ocean is **not** a sum of analytical patterns painted into the temperature
field. It is a small dynamical model.

1. **Geostrophic streamfunction** ψ(x, y, t) = background double gyre +
   meandering zonal jet + mesoscale eddies + an unresolved perturbation with a
   k⁻³ spectrum and Ornstein–Uhlenbeck decorrelation in time. Velocity follows
   as u = −∂ψ/∂y, v = +∂ψ/∂x, so the flow is non-divergent by construction, and
   sea surface height is SSH = f₀ψ/g.

2. **Tracers are advected** by that flow:

   ```
   ∂C/∂t + u·∇C = −(C − C_clim(y,t))/τ + κ∇²C
   ```

   with a semi-Lagrangian scheme (Catmull-Rom interpolation) and implicit
   restoring.

3. **Fronts, filaments and sharp gradients are not drawn — they emerge** from
   the competition between stirring by the flow and restoring towards
   climatology. That is what gives the field its realistic texture.

4. **Eddies live in ψ, not in SST.** They are advected by the large-scale flow
   plus westward β-drift, they are born preferentially along the jet
   (baroclinic instability) and they decay.

5. **SST and SSS have different restoring timescales** — 40 days for air-sea
   heat flux, 150 days for freshwater flux. Their decorrelation timescales
   therefore differ, which is precisely the information that justifies sizing a
   network variable by variable.

A spin-up of 150 days is run and discarded so that filaments already exist at
t = 0.

### 3.3 Diagnostics (seed 42, nt = 365)

| quantity | value | why it matters |
|---|---|---|
| σ(SST) | 2.59 °C | |
| σ(SSS) | 0.177 psu | 15× smaller than SST — never mix them without standardising |
| spatial decorrelation length | 90 km | reference sensor spacing |
| mesoscale decorrelation time | 12 days | reference sampling frequency |
| SST decorrelation time (total) | 52 days | dominated by the seasonal cycle |
| seasonal SST range | 4.1 °C | |
| T–S correlation | +0.77 | warm & salty, subtropical density compensation |
| Rossby number (p99) | 0.39 | |
| radial spectral slope | −2.95 (mesoscale) / −2.7 (submesoscale) | between QG and SQG |

Run `python data/dataset.py` to print these for your own seed.

### 3.4 Command

```bash
python data/dataset.py [--nt 1000] [--seed 42] [--out outputs/ocean_nature_run.png]
```

Produces a 16-panel diagnostic figure: SST snapshots, temporal variability,
SSH with geostrophic streamlines, relative vorticity ζ/f, |∇SST| showing
fronts and filaments, SSS, radial spectrum with k⁻² and k⁻³ references, spatial
and temporal autocorrelations, T–S diagram with σ₀ isopycnals, time series,
distributions, T–S correlation map and a sample buoy network.

### 3.5 Python API

```python
from data.dataset import SyntheticOceanGenerator

gen  = SyntheticOceanGenerator()
T, S = gen.generate_dataset(nt=365, seed=42)      # (nt, nx, ny) float32

run  = gen.generate_full(nt=365, seed=42)         # dict: T, S, SSH, U, V, ZETA, SIGMA0
diag = gen.diagnostics()                          # decorrelation scales, EKE, ...
```

`generate_dataset(nt, seed)` is fully deterministic: the same seed gives the
same ocean, every time.

> **Use `--nt 365` or more.** Below a full year the seasonal cycle is not
> sampled over a complete period and the correlation statistics are biased. All
> entry points warn you when `nt < 365`.

---

## 4. Brick 1 — Observability autoencoder

### 4.1 Purpose

Reconstruct the full SST/SSS field from a sparse set of observations. The
reconstruction error tells you how much information the network carries, and
*where* it is blind.

Architecture: U-Net with MC-Dropout for uncertainty (dropout stays on at
inference, N forward passes → predictive variance), skip connections gated on
local observation density (ObsGate), FiLM conditioning on the number of
observations, GroupNorm, Huber loss and deep supervision. Training uses a
stochastic observation mask so the model is robust to any network geometry.

### 4.2 Commands

```bash
# train
python 01_autoencoder.py --train --nt 365 --epochs 100

# figures from an existing checkpoint
python 01_autoencoder.py --figures --nt 365 --checkpoint outputs/vae_best.pt

# leave-one-out contribution of each sensor
python 01_autoencoder.py --score --nt 365

# everything plus a text report
python 01_autoencoder.py --train --figures --score --report --nt 365
```

### 4.3 Parameters

| flag | default | meaning |
|---|---|---|
| `--nt` | `config.NT` (1000) | nature run length in days |
| `--seed_ocean` | 42 | ocean seed — controls the nature run |
| `--seed_buoys` | 7 | reference network seed |
| `--epochs` | 100 | training epochs |
| `--batch_size` | 16 | |
| `--lr` | 3e-4 | AdamW, cosine schedule with warm-up |
| `--base_ch` | 32 | U-Net width — the main cost/quality knob |
| `--latent_ch` | 64 | bottleneck depth |
| `--dropout_p` | 0.1 | MC-Dropout rate, active at inference too |
| `--w_unobs` | 4.0 | weight of unobserved pixels in the loss |
| `--lambda_grad` | 0.5 | gradient-matching term (keeps fronts sharp) |
| `--huber_delta` | 0.5 | Huber transition point |
| `--n_obs_min` / `--n_obs_max` | 10 / 80 | random mask size range during training |
| `--n_mc_val` | 15 | MC passes for validation RMSE |
| `--n_mc` | 60 | MC passes for the figures |
| `--checkpoint` | `outputs/vae_best.pt` | |
| `--output_dir` | `outputs` | |

### 4.4 Reading the output

Validation RMSE is reported **per variable in physical units** — `2.10 °C` and
`0.167 psu`, not a single aggregate. The two channels are normalised by very
different standard deviations, so an aggregate number converts to neither.

`vae_network_evaluation.png` shows, for a given network: true field with sensors
coloured by their leave-one-out contribution, reconstruction, MC uncertainty
map, gap map (high σ × far from any sensor) with three greedily proposed new
buoys, and a bar chart ranking sensors from indispensable to redundant.

### 4.5 Fast example

```bash
python 01_autoencoder.py --train --nt 365 --epochs 10 --base_ch 8 \
    --latent_ch 16 --batch_size 4 --n_mc_val 3
```

---

## 5. Brick 2 — Graph neural network

### 5.1 Purpose

Model the observing network as a graph — nodes are sensors, edges encode
spatial correlation — and learn which nodes carry unique information and which
are redundant. Attention weights are the redundancy signal. A GraphSAGE branch
runs inductively, so a hypothetical glider or Argo float can be scored without
retraining.

### 5.2 The seasonal cycle trap

Correlations are computed on **de-seasonalised** anomalies by default. The
seasonal cycle is a near-uniform mode: keep it and two buoys 1000 km apart
correlate strongly simply because they both see summer arrive.

| nt = 365, threshold 0.35 | mean \|ρ\| | edges / 435 pairs | density |
|---|---|---|---|
| raw | 0.446 | 318 | 73 % |
| de-seasonalised | 0.169 | 36 | 8.3 % |

At 73 % density the graph is a near-clique and redundancy is meaningless.
`--deseason 0` restores the old behaviour if you want to see it.

Node features: normalised position, maximum correlation with any neighbour,
degree, and local SST/SSS variance (standardised separately — var(SST) ≈ 3 °C²
against var(SSS) ≈ 0.03 psu², so an unstandardised mix erases salinity).

### 5.3 Commands

```bash
# train and analyse
python 02_gnn.py --train --analyze --nt 365

# add inductive evaluation of three hypothetical sensors
python 02_gnn.py --train --analyze --inductive \
    --new_positions "[(20,40),(90,160),(140,60)]" --nt 365

# with a text report
python 02_gnn.py --train --analyze --inductive --report --nt 365
```

### 5.4 Parameters

| flag | default | meaning |
|---|---|---|
| `--nt` | 500 | nature run length in days |
| `--seed_ocean` / `--seed_buoys` | 42 / 7 | |
| `--n_buoys` | `config.N_BUOYS` (30) | network size |
| `--corr_threshold` | `config.GNN_CORR_THRESHOLD` (0.35) | \|ρ\| above which an edge is created |
| `--k_nearest` | 4 | geographic k-NN edges, guarantee connectivity |
| `--deseason` | 1 | remove the domain mean before correlating |
| `--gnn_epochs` | 200 | |
| `--new_positions` | `"[(10,20),(80,150),(130,40)]"` | pixel coordinates to score inductively |
| `--output_dir` | `outputs` | |

`--corr_threshold` is calibrated for de-seasonalised anomalies. If you set
`--deseason 0`, raise it to around 0.6 or the graph saturates.

### 5.5 Fast example

```bash
python 02_gnn.py --train --analyze --nt 365 --gnn_epochs 50
```

---

## 6. Brick 3 — Reinforcement learning

### 6.1 Purpose

Search directly for the best network under constraints. A PPO agent toggles
candidate positions on a coarse grid; the reward is the marginal information
gain minus a budget penalty. Two Pareto fronts come out of it: information
versus number of buoys, and information versus operating cost and carbon
footprint.

### 6.2 The information criterion

Default `--info_mode evf`: **explained variance** by optimal linear estimation
(BLUE / optimal interpolation), the standard OSSE criterion.

```
EVF = Σ_c  C_cO (C_OO + R)⁻¹ C_Oc  /  Σ_c C_cc
```

The observation vector holds **both SST and SSS** at every buoy — 2n
observations — each normalised by its own standard deviation and given its own
instrumental noise, so salinity actually counts. The criterion is increasing,
saturating and submodular, which guarantees diminishing returns and a
well-defined elbow.

**The covariance is not empirical, and that matters.** Mesoscale decorrelation
time is ~12 days, so one year of nature run holds only about 30 independent
realisations, against 2n = 40 parameters as soon as you have 20 buoys. The raw
sample covariance overfits massively: measured out of sample, explained
variance goes *negative* (−0.49 at N = 20, while the in-sample score claimed
0.62). The covariance is therefore shrunk towards a parametric model
σ(x)·exp(−d²/2L²) built from the nature run's own diagnostics, exactly as
operational optimal interpolation does. `EVF_SHRINKAGE = 0.9`.

Two other modes exist: `coverage` (fast geometric coverage kernel, useful when
you need many reward evaluations) and `legacy` (the historical formula, kept
for comparison only — it is not monotone in N).

> **Reporting a number?** Add `--evf_cv 1`. Statistics are then estimated on
> the first half of the series and the score is measured on the second. It is
> markedly lower than the analytical score — 0.12 against 0.37 at N = 20 — but
> it is the defensible figure. Keep the analytical mode for optimisation, it is
> smoother.

### 6.3 The separation constraint

Two buoys cannot occupy adjacent cells of the candidate grid. This is a **hard**
constraint, enforced by masking the actor's logits, not a reward penalty.

```
n_feasible_max = ceil(grid_x / min_sep) × ceil(grid_y / min_sep)
```

| grid | candidates | feasible maximum |
|---|---|---|
| 16 × 24 | 384 | 96 |
| 8 × 12 | 96 | 24 |

`n_max` is clipped to that ceiling automatically, with an explicit message.
`MIN_SEP_DIAGONAL = False` in `config.py` switches from Chebyshev (diagonals
forbidden) to Manhattan (only the four direct neighbours forbidden).

Note the units: the constraint is expressed in **grid cells**, not kilometres.
On a 16 × 24 grid a cell is 50 km, so `--min_sep 2` means 100 km. For a 50 km
effective separation, double the grid resolution to 32 × 48.

### 6.4 Commands

```bash
# train the policy
python 03_rl.py --train --nt 365 --rl_steps 50000

# Pareto front: information vs number of buoys
python 03_rl.py --pareto --nt 365 --report

# Pareto front: information vs cost and carbon
python 03_rl.py --multiobj --nt 365 --report

# everything in one pass
python 03_rl.py --train --pareto --multiobj --gif --report \
    --nt 365 --rl_steps 50000
```

`--pareto` and `--multiobj` reload `outputs/rl_best.pt` if it exists, so they
can run without `--train`. Grid parameters must match between training and
fronts, otherwise the checkpoint will not load.

### 6.5 Parameters

**MDP and constraints**

| flag | default | meaning |
|---|---|---|
| `--grid_x` / `--grid_y` | 16 / 24 | candidate grid — K = grid_x × grid_y actions |
| `--n_min` / `--n_max` | 10 / 40 | allowed range of active buoys |
| `--min_sep` | `config.MIN_SEP_CELLS` (2) | minimum separation in grid cells |
| `--episode_len` | 20 | toggles per episode |

**Information criterion**

| flag | default | meaning |
|---|---|---|
| `--info_mode` | `evf` | `evf` \| `coverage` \| `legacy` |
| `--influence_km` | `config.INFLUENCE_RADIUS_KM` (90) | sensor influence radius |
| `--evf_shrink` | `config.EVF_SHRINKAGE` (0.9) | shrinkage towards the parametric covariance |
| `--evf_cv` | 0 | 1 = score validated out of sample |

**PPO**

| flag | default | meaning |
|---|---|---|
| `--rl_steps` | 50000 | environment steps |
| `--buffer_size` | 512 | rollout buffer |
| `--lr` | 3e-4 | |
| `--w_info` / `--w_budget` | 1.0 / 0.5 | reward weights |

**Pareto fronts**

| flag | default | meaning |
|---|---|---|
| `--n_random` | 25 | random configurations drawn per N |
| `--gif_frames` | 80 | frames in the progression GIF |

### 6.6 Reading the fronts

`rl_pareto_front.png` — three panels: the cloud of evaluated configurations
coloured by source (random baseline, PPO policy, greedy reference with its
1 − 1/e submodular guarantee), the non-dominated set, the upper envelope and
N★; the marginal gain per added buoy; and N★ as a function of λ, the marginal
cost of a buoy, obtained by sweeping `max_N [info(N) − λ·N]`. That last panel
is the one that answers "what is the best compromise".

Typical output (nt = 365, grid 16 × 24, `--min_sep 2`):

```
N★ = 21 buoys  —  21 optimised buoys are worth 38 randomly placed ones
```

`rl_pareto_cost.png` — information versus operating cost, where cost is

```
cost = N · COST_BUOY_FIXED + tour_length · COST_SHIP_PER_KM · N_CAMPAIGNS_YEAR
```

The maintenance tour starts from a port, visits every buoy by nearest neighbour
and returns. Cost is therefore **not** proportional to N: at fixed N it varies
by a factor 1.3 to 1.6 depending on how spread out the network is. That is what
makes the two objectives genuinely antagonistic. The brick prints a directly
usable table:

```
 budget |  N |  info | actual cost | tCO2/yr
  500 k€ |  5 | 0.271 |      441 k€ |     211
  700 k€ | 10 | 0.418 |      683 k€ |     313
  900 k€ | 14 | 0.499 |      888 k€ |     400
 1100 k€ | 22 | 0.615 |     1094 k€ |     461
 1400 k€ | 29 | 0.688 |     1383 k€ |     575
```

### 6.7 Fast example

```bash
python 03_rl.py --train --pareto --multiobj --nt 365 --rl_steps 5000 \
    --grid_x 8 --grid_y 12 --n_min 5 --n_max 20 --n_random 10
```

---

## 7. Orchestrator

```bash
# three bricks independently, same ocean, same initial network
python run_demo.py --mode individual --nt 365

# RL → GNN → AE on the network RL proposes
python run_demo.py --mode pipeline --nt 365 --seed_ocean 42 --seed_buoys 7
```

`run_demo.py` writes a timestamped text report gathering every metric, plus a
reproducibility JSON block, plus the nature-run diagnostic figure. The report
header carries the decorrelation length and mesoscale timescale, which are the
reference spacing and sampling frequency the whole design rests on.

Main flags: `--mode`, `--nt`, `--seed_ocean`, `--seed_buoys`, `--n_buoys`,
`--ae_epochs`, `--ae_base_ch`, `--gnn_epochs`, `--gnn_corr_threshold`,
`--rl_steps`, `--rl_grid_x`, `--rl_grid_y`, `--rl_n_min`, `--rl_n_max`,
`--rl_info_mode`, `--rl_min_sep`, `--rl_influence_km`, `--gif_frames`,
`--output_dir`, `--no_nature_fig`.

Quick smoke test:

```bash
python run_demo.py --mode pipeline --nt 365 --ae_epochs 1 --ae_base_ch 8 \
    --gnn_epochs 15 --rl_steps 1000 --gif_frames 5 --no_nature_fig
```

---

## 8. Configuration reference

Everything lives in `config.py`.

**Domain and numerics**

| name | default | |
|---|---|---|
| `NX`, `NY` | 160, 240 | grid points, zonal × meridional |
| `NT` | 1000 | default nature run length in days |
| `DX_KM` | 5.0 | resolution |
| `LAT0` | 42.0 | central latitude |
| `N_SUBSTEPS` | 2 | advection substeps per output step |
| `SPINUP_DAYS` | 150 | discarded spin-up |
| `KAPPA` | 25.0 | diffusivity (m²/s) — sets the dissipation scale |

**Circulation**

| name | default | |
|---|---|---|
| `U_GYRE`, `U_JET` | 0.08, 0.55 | m/s |
| `JET_WIDTH_KM`, `JET_LAT_FRAC` | 40.0, 0.55 | |
| `N_EDDIES` | 22 | simultaneous eddies |
| `EDDY_V_MAX`, `EDDY_R_KM`, `EDDY_LIFE_DAYS` | 0.25, (35, 80), (60, 180) | |
| `RD_KM` | 25.0 | Rossby radius → β-drift |

**Tracers**

| name | default | |
|---|---|---|
| `SST_MEAN`, `SST_GRADIENT`, `SST_SEASONAL_AMP` | 15.0, 9.0, 2.5 | °C |
| `TAU_T_DAYS` | 40.0 | thermal restoring |
| `SSS_MEAN`, `SSS_GRADIENT`, `SSS_PLUME_AMP` | 35.0, 1.30, 0.75 | psu |
| `TAU_S_DAYS` | 150.0 | haline restoring — much slower, no feedback |
| `TS_CORRELATION` | 0.7 | share of the S climatology aligned with T |

**Observation and analysis**

| name | default | |
|---|---|---|
| `N_BUOYS` | 30 | |
| `OBS_NOISE_T`, `OBS_NOISE_S` | 0.05 °C, 0.02 psu | per-variable instrumental noise |
| `DESEASON_ANALYSIS` | True | remove the domain mean before correlating |
| `INFLUENCE_RADIUS_KM` | 90.0 | sensor influence radius |
| `GNN_CORR_THRESHOLD` | 0.35 | |
| `EVF_SHRINKAGE` | 0.9 | |
| `MIN_SEP_CELLS`, `MIN_SEP_DIAGONAL` | 2, True | separation constraint |
| `MIN_BUOY_SEP_KM` | 50.0 | equivalent for pixel-drawn networks |

**Cost model**

| name | default | |
|---|---|---|
| `PORT_XY_FRAC` | (0.04, 0.03) | port position, domain fraction |
| `COST_BUOY_FIXED` | 12.0 | k€/yr per buoy |
| `COST_SHIP_PER_KM` | 0.090 | k€/km of research vessel |
| `N_CAMPAIGNS_YEAR` | 2 | |
| `CO2_SHIP_PER_KM` | 0.050 | tCO2/km |

> Cost parameters are indicative orders of magnitude, not sourced figures.
> Calibrate them against the real costs of your target SNO — that is why they
> are isolated in the config.

---

## 9. Output files

Everything lands in `--output_dir` (default `outputs/`).

| file | produced by |
|---|---|
| `ocean_nature_run.png` | `data/dataset.py`, `run_demo.py` |
| `vae_best.pt`, `vae_training_curves.png` | `01 --train` |
| `vae_network_evaluation.png`, `vae_uncertainty_density.png` | `01 --figures` |
| `vae_loo_scores.json` | `01 --score` |
| `gnn_best.pt`, `gnn_network_analysis.png` | `02 --train --analyze` |
| `gnn_inductive_eval.png` | `02 --inductive` |
| `rl_best.pt`, `rl_training_curves.png`, `rl_optimal_network.png`, `rl_progression.gif` | `03 --train` |
| `rl_pareto_front.png`, `rl_two_configs.png` | `03 --pareto` |
| `rl_pareto_cost.png` | `03 --multiobj` |
| `rapport_*.txt` | any `--report`, and `run_demo.py` |

---

## 10. Known limitations

**Reproducibility covers the ocean and the network, not the training.**
`--seed_ocean` and `--seed_buoys` fully determine the nature run and the
reference network. PyTorch seeds are not fixed, so two training runs will
differ slightly.

**Nature run length.** Below `--nt 365` the seasonal cycle is not sampled over
a full period and correlation statistics are biased. Even at 365 days the
mesoscale holds only ~30 independent realisations, which is why the EVF
covariance has to be shrunk.

**Cost model.** The nearest-neighbour tour overestimates the optimal route by
roughly 10–25 % on a Euclidean TSP. If transit cost dominates in your case, a
2-opt pass would tighten it.

**Old checkpoints.** GNN node features went from 4 to 6 dimensions; a
`gnn_best.pt` produced by an earlier version will not reload. Regenerate it.

**Single-domain OSSE.** The nature run is a mid-latitude zonal channel. MOOSE
and PIRATA sit in very different regimes; transposing means recalibrating
`LAT0`, the gradients and the eddy statistics, or plugging in a real model
output through the same interface.


---

## 11. Citing NAIADE

If you use this framework, please cite both the software and the proposal it
implements.

```bibtex
@software{naiade,
  author  = {JM VIENT},
  title   = {NAIADE: AI methods for Optimal Experimental Design
             of marine observing networks},
  year    = {2026},
  url     = {https://github.com/Jvient/NAIADE}
}

```

---

## 12. References

### 12.1 What each design choice implements

This table maps the non-obvious implementation choices to the work that
justifies them. It is the part worth reading if you want to check the method
rather than the code.

| Where | Choice | Reference |
|---|---|---|
| Brick 3 — information criterion | Explained variance by optimal linear estimation, with error maps used to design the array. This is exactly the Gauss–Markov objective-analysis framework introduced to oceanography for MODE-73. | Bretherton, Davis & Fandry (1976) |
| Brick 3 — covariance regularisation | Shrinkage of the sample covariance towards a structured target when the effective sample size is small. | Ledoit & Wolf (2004) |
| Brick 3 — greedy reference | The criterion is monotone submodular, so greedy selection is within 1 − 1/e of the optimum. | Nemhauser, Wolsey & Fisher (1978) |
| Brick 3 — sensor placement framing | Near-optimal sensor placement in Gaussian processes; mutual-information criterion and submodularity in a spatial-monitoring setting. | Krause, Singh & Guestrin (2008) |
| Brick 3 — policy | PPO, discrete action space, clipped surrogate objective. | Schulman et al. (2017) |
| Brick 3 — elbow detection | Maximum distance to the chord on a concave saturating curve. | Satopää et al. (2011) |
| Brick 3 — mooring array design precedent | Model-based assessment and design of a tropical mooring array — the closest published analogue to what Brick 3 does for PIRATA. | Oke & Schiller (2007) |
| Brick 2 — attention as redundancy | Graph attention networks; attention weights read as neighbour influence. | Veličković et al. (2018) |
| Brick 2 — inductive scoring | GraphSAGE: aggregation functions that generalise to unseen nodes, so a hypothetical glider can be scored without retraining. | Hamilton, Ying & Leskovec (2017) |
| Brick 1 — uncertainty | MC-Dropout: dropout kept active at inference, N forward passes give an approximate posterior. | Gal & Ghahramani (2016) |
| Brick 1 — backbone | U-Net encoder–decoder with skip connections. | Ronneberger, Fischer & Brox (2015) |
| Brick 1 — conditioning on N_obs | FiLM: feature-wise linear modulation by a conditioning vector. | Perez et al. (2018) |
| Brick 1 — bottleneck attention | CBAM: sequential channel and spatial attention. | Woo et al. (2018) |
| Ocean — advection scheme | Semi-Lagrangian integration with iterated midpoint departure points. | Staniforth & Côté (1991) |
| Ocean — interpolation | Catmull–Rom cubic, chosen over bilinear to preserve filaments. | Catmull & Rom (1974) |
| Ocean — k⁻³ spectrum | Geostrophic turbulence: enstrophy cascade and the k⁻³ mesoscale slope. | Charney (1971) |
| Ocean — k⁻² submesoscale slope | Surface quasi-geostrophic dynamics, the shallower surface-tracer slope. | Held et al. (1995) |
| Ocean — eddy propagation | Westward β-drift and observed eddy lifetimes and radii. | Chelton, Schlax & Samelson (2011) |
| Ocean — density | EOS-80 one-atmosphere equation of state, used for σ₀ and the T–S diagram isopycnals. | Millero & Poisson (1981) |
| Framework — OSSE context | Requirements for an integrated in situ observing system derived from coordinated OSSEs. | Gasparin et al. (2019) |

### 12.2 Bibliography

**Foundations — OED and observing-network design**

- Krause, A., Singh, A., & Guestrin, C. (2008). Near-optimal sensor placements in Gaussian processes: theory, efficient algorithms and empirical studies. *Journal of Machine Learning Research*, 9, 235–284.
- Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions — I. *Mathematical Programming*, 14(1), 265–294. doi:10.1007/BF01588971
- Wikle, C. K., & Royle, J. A. (1999). Space-time dynamic design of environmental monitoring networks. *Journal of the American Statistical Association*, 94(445), 1–11.
- Huan, X., & Marzouk, Y. M. (2013). Simulation-based optimal Bayesian experimental design for nonlinear systems. *Journal of Computational Physics*, 232(1), 288–317.
- Chaloner, K., & Verdinelli, I. (1995). Bayesian experimental design: a review. *Statistical Science*, 10(3), 273–304.
- Ryan, K. J. (2003). Estimating expected information gains for experimental designs with application to the random fatigue-limit model. *Journal of Computational and Graphical Statistics*, 12(3), 585–603.

**OSSE and oceanographic array design**

- Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976). A technique for objective analysis and design of oceanographic experiments applied to MODE-73. *Deep-Sea Research and Oceanographic Abstracts*, 23(7), 559–582. doi:10.1016/0011-7471(76)90001-2
- Oke, P. R., & Schiller, A. (2007). A model-based assessment and design of a tropical Indian Ocean mooring array. *Journal of Climate*, 20(13), 3269–3283.
- Gasparin, F., et al. (2019). Requirements for an integrated in situ Atlantic Ocean observing system from coordinated observing system simulation experiments. *Frontiers in Marine Science*, 6, 83.
- Heimbach, P., et al. (2019). Putting it all together: adding value to the global ocean and climate observing systems with complete self-consistent ocean state and parameter estimates. *Frontiers in Marine Science*, 6, 55.
- Sakov, P., & Sandery, P. A. (2017). An adaptive quality control procedure for data assimilation. *Tellus A*, 69(1), 1318031.

**Ocean dynamics and numerics (synthetic ocean)**

- Charney, J. G. (1971). Geostrophic turbulence. *Journal of the Atmospheric Sciences*, 28(6), 1087–1095.
- Held, I. M., Pierrehumbert, R. T., Garner, S. T., & Swanson, K. L. (1995). Surface quasi-geostrophic dynamics. *Journal of Fluid Mechanics*, 282, 1–20.
- Chelton, D. B., Schlax, M. G., & Samelson, R. M. (2011). Global observations of nonlinear mesoscale eddies. *Progress in Oceanography*, 91(2), 167–216.
- Staniforth, A., & Côté, J. (1991). Semi-Lagrangian integration schemes for atmospheric models — a review. *Monthly Weather Review*, 119(9), 2206–2223.
- Catmull, E., & Rom, R. (1974). A class of local interpolating splines. In *Computer Aided Geometric Design*, Academic Press, 317–326.
- Millero, F. J., & Poisson, A. (1981). International one-atmosphere equation of state of seawater. *Deep-Sea Research Part A*, 28(6), 625–629. doi:10.1016/0198-0149(81)90122-9

**Statistics and machine learning methods**

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *ICML 2016*, PMLR 48, 1050–1059.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI 2015*, LNCS 9351, 234–241.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: visual reasoning with a general conditioning layer. *AAAI 2018*.
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: convolutional block attention module. *ECCV 2018*.
- Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a "kneedle" in a haystack: detecting knee points in system behavior. *ICDCS Workshops 2011*.
- Poole, B., et al. (2019). On variational bounds of mutual information. *ICML 2019*, PMLR 97, 5171–5180.

**Autoencoders and representation learning for geophysical data**

- Shi, X., et al. (2015). Convolutional LSTM network: a machine learning approach for precipitation nowcasting. *NeurIPS*, 28.
- Manucharyan, G. E., et al. (2021). A deep learning approach to spatiotemporal sea surface temperature variability. *Journal of Physical Oceanography*, 51(6), 1809–1824.
- Lguensat, R., et al. (2018). The analog data assimilation. *Monthly Weather Review*, 145(10), 4093–4107.
- Fablet, R., et al. (2021). Learning variational data assimilation models and solvers. *JAMES*, 13(10), e2021MS002572.
- Grooms, I., et al. (2023). Hybrid ensemble-variational algorithms for data assimilation: tutorial and review. *Frontiers in Applied Mathematics and Statistics*, 9.

**Graph neural networks for spatial and climate systems**

- Veličković, P., et al. (2018). Graph attention networks. *ICLR 2018*.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs (GraphSAGE). *NeurIPS*, 30.
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
- Cachay, S. R., et al. (2021). The world as a graph: improving El Niño forecasts with graph neural networks. *NeurIPS 2021 Workshop Climate Change AI*.
- Lam, R., et al. (2023). GraphCast: learning skillful medium-range global weather forecasting. *Science*, 382(6677), 1416–1421.
- Rossi, E., et al. (2020). Temporal graph networks for deep learning on dynamic graphs. *ICML 2020 Workshop GRL+*.

**Reinforcement learning for physical-system optimisation**

- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Haarnoja, T., et al. (2018). Soft actor-critic: off-policy maximum entropy deep reinforcement learning. *ICML 2018*.
- Duffield, S., et al. (2022). Deep reinforcement learning for adaptive ocean observation. *Environmental Data Science*, 1, e13.
- Petersen, M. N., et al. (2022). Autonomous ocean sampling with a multi-agent reinforcement learning approach. *Ocean Science*, 18, 1653–1669.
- Mankowitz, D. J., et al. (2023). Faster sorting algorithms discovered using deep reinforcement learning. *Nature*, 618, 257–263.

**Multi-objective optimisation and carbon footprint**

- Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
- Hernandez-Lobato, J. M., et al. (2016). Predictive entropy search for multi-objective Bayesian optimization. *NeurIPS*, 29.
- Racault, M.-F., et al. (2023). Towards sustainable ocean observation: carbon footprint benchmarking. *Frontiers in Marine Science*, 10, 1101993.

**Target networks**

- Coppola, L., et al. (2019). A posteriori quality control of the MOOSE-GE cruises. *Frontiers in Marine Science*, 6, 233.
- Bourlès, B., et al. (2019). PIRATA: a sustained observing system for tropical Atlantic climate research and forecasting. *BAMS*, 100(4), 655–686.
- Testor, P., et al. (2018). OceanGliders: a component of the integrated GOOS. *Frontiers in Marine Science*, 6, 422.
