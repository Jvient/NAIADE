# NAIADE

**Optimal Experimental Design for marine observing networks, with AI.**

> This README was written with the help of an AI assistant, then reviewed and
> corrected by the author. The code, the results and the scientific choices are
> the author's.

NAIADE is a Python/PyTorch framework for designing, scoring and evolving ocean
observing networks. Everything runs as an OSSE: a synthetic ocean plays the
role of ground truth, so any network configuration can be evaluated against a
known answer.

Three components share one ocean and one scoring rule, so their answers can be
put side by side. Each also runs on its own.

```
                  synthetic ocean, the nature run
                  SST, SSS, SSH in 2D and time
                               |
     ┌─────────────────────────┼─────────────────────────┐
     ▼                         ▼                         ▼
 01_autoencoder.py        02_gnn.py                  03_rl.py
 observability            network structure          optimisation
 and gap maps             and redundancy             under constraints

 Where is the network     Which sensors are          How many, where,
 blind?                   redundant?                 at what cost?
```

The files are numbered in the order they were written, not in the order they
run. **The pipeline runs RL first**, then the GNN, then the autoencoder, so
that every diagnostic describes the same network. This README refers to the
components by name rather than by number, to avoid the confusion.

This is a proof of concept. The ocean is synthetic and there is no data
assimilation in the loop. Read [Known limitations](#10-known-limitations)
before quoting any number from it.

---

## Table of contents

0. [Gallery](#0-gallery)
1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [The synthetic ocean](#3-the-synthetic-ocean)
4. [Autoencoder, observability](#4-autoencoder-observability)
5. [Graph network, structure and redundancy](#5-graph-network-structure-and-redundancy)
6. [Reinforcement learning, optimisation](#6-reinforcement-learning-optimisation)
7. [Orchestrator](#7-orchestrator)
8. [Configuration reference](#8-configuration-reference)
9. [Output files](#9-output-files)
10. [Known limitations](#10-known-limitations)
11. [Citing NAIADE](#11-citing-naiade)
12. [References](#12-references)

---

## 0. Gallery

Figures live in [`figures/`](figures/), with a note on what each one shows.
They come from a pipeline run at `--seed_ocean 42 --seed_buoys 7`. A run writes
to `outputs/`, which is git-ignored; the gallery is refreshed by copying across
what is worth keeping.

![Nature run](figures/ocean_nature_run.png)

*The synthetic ocean. Fronts and filaments are not drawn, they emerge from the
competition between geostrophic stirring and restoring towards climatology.*

![Pareto front](figures/rl_pareto_front.png)

*Information against number of buoys. On the reference run, N★ = 23 explaining
63.7 % of the mesoscale variance, and those 23 optimised buoys carry as much
information as 41 placed at random.*

![Information against cost](figures/rl_info_vs_cost_networks.png)

*The same network size, optimised on information alone and on information plus
cost. The geometry changes: cost is not proportional to the number of buoys.*

> **The numbers in this README come from one run** and move with `--nt`, the
> seed and the training budget. Regenerate rather than quote.

---

## 1. Install

```bash
git clone https://github.com/Jvient/NAIADE
cd NAIADE
pip install -r requirements.txt
```

PyTorch Geometric is optional; without it `02_gnn.py` falls back to a
hand-written attention layer that gives the same results, more slowly. `torch`
itself is optional for the nature run and the OED core, which are pure numpy
and stay importable without it.

```
NAIADE/
├── config.py              all physical and methodological parameters
├── data/
│   └── dataset.py         ocean generator, PyTorch datasets, shared utilities
├── 01_autoencoder.py      observability and gap maps
├── 02_gnn.py              structure, redundancy, inductive scoring
├── 03_rl.py               optimisation under constraints
├── run_demo.py            orchestrator, individual and pipeline modes
├── figures/               reference gallery, committed
└── outputs/               everything a run writes, git-ignored
```

Everything runs on CPU. A GPU is used automatically when available
(`config.DEVICE`).

---

## 2. Quick start

Generate the ocean and look at it, with an animation:

```bash
python data/dataset.py --nt 1500 --seed 42 --gif --gif_every 10
```

The whole framework, end to end:

```bash
python run_demo.py --mode pipeline \
  --seed_ocean 42 --seed_buoys 7 --nt 1500 \
  --rl_grid_x 16 --rl_grid_y 24 --rl_n_max 20 --rl_steps 50000 \
  --ae_epochs 200 --ae_base_ch 32 --gnn_epochs 500 \
  --cost_compare_ref rl \
  --ocean_gif --ocean_gif_var T,GRADT,S --ocean_gif_every 10
```

A first look, a couple of minutes:

```bash
python run_demo.py --mode pipeline --nt 90 --rl_steps 400 \
  --ae_epochs 1 --ae_base_ch 8 --gnn_epochs 40 --ocean_gif
```

| mode | what it does |
|---|---|
| `individual` | the three components run independently on the same ocean and the same initial network, to compare them |
| `pipeline` | RL proposes a network, the same size is compared with and without cost, the GNN scores its structure, the autoencoder maps its blind spots and proposes additions, the GNN scores those additions |

The pipeline is the interesting one: the buoy positions are fixed once, by the
agent, and everything after is scored on that same network.

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

   with a semi-Lagrangian scheme (Catmull–Rom interpolation) and implicit
   restoring.

3. **Fronts, filaments and sharp gradients are not drawn, they emerge** from
   the competition between stirring by the flow and restoring towards
   climatology. That is what gives the field its texture.

4. **Eddies live in ψ, not in SST.** They are advected by the large-scale flow
   plus westward β-drift, born preferentially along the jet, and they decay.

5. **SST and SSS have different restoring timescales**, 40 days for air-sea
   heat flux against 150 days for freshwater flux. Their decorrelation
   timescales therefore differ, which is precisely the information that
   justifies sizing a network variable by variable.

A spin-up of 150 days is run and discarded, so filaments already exist at t = 0.

### 3.3 Diagnostics (seed 42, nt = 365)

| quantity | value | why it matters |
|---|---|---|
| σ(SST) | 2.60 °C | |
| σ(SSS) | 0.177 psu | 15× smaller than SST, never mix them without standardising |
| spatial decorrelation length | 95 km | reference sensor spacing |
| mesoscale decorrelation time | 12 days | reference sampling frequency |
| SST decorrelation time, total | ~52 days | dominated by the seasonal cycle |
| T–S correlation | +0.77 | warm and salty, subtropical density compensation |
| Rossby number, p99 | 0.39 | |
| radial spectral slope | −2.95 mesoscale, −2.7 submesoscale | between QG and SQG |

Run `python data/dataset.py` to print these for your own seed and length.

### 3.4 Command

```bash
python data/dataset.py [--nt 1500] [--seed 42] [--out outputs/ocean_nature_run.png]
                       [--gif] [--gif_every 10] [--gif_var T,GRADT,S,GRADS]
                       [--gif_fps 8] [--gif_max 120]
```

The static figure has 16 panels: SST snapshots, temporal variability, SSH with
geostrophic streamlines, relative vorticity, |∇SST| showing fronts and
filaments, SSS, radial spectrum with k⁻² and k⁻³ references, spatial and
temporal autocorrelations, T–S diagram with σ₀ isopycnals, time series,
distributions, T–S correlation map and a sample buoy network.

`--gif` adds an animation, one frame every `--gif_every` days, on a square
grid. `--gif_var` takes a comma-separated list among `T`, `S`, `SSH`, `ZETA`,
`GRADT`, `GRADS`. The two gradient moduli show the fronts far better than the
fields themselves, which are dominated by the seasonal cycle. Frames are capped
at `--gif_max`; past that the stride is raised automatically and reported.

### 3.5 Python API

```python
from data.dataset import SyntheticOceanGenerator

gen  = SyntheticOceanGenerator()
T, S = gen.generate_dataset(nt=1500, seed=42)     # (nt, nx, ny) float32
run  = gen.generate_full(nt=1500, seed=42)        # dict: T, S, SSH, U, V, ZETA, SIGMA0
diag = gen.diagnostics()                          # decorrelation scales, EKE, ...
```

`generate_dataset(nt, seed)` is fully deterministic: same seed, same ocean.

> **Length matters more than you would think.** What counts is not `nt` but the
> number of independent mesoscale realisations, `nt / 12 days`. One year gives
> about 30 of them against `2n` covariance parameters, which is why `--nt 1500`
> (≈125 realisations) is the recommended setting. Below 365 days the seasonal
> cycle is not even sampled over a full period; every entry point warns you.

---

## 4. Autoencoder, observability

`01_autoencoder.py`

### 4.1 Purpose

Reconstruct the full SST/SSS field from a sparse set of observations. The
reconstruction error says how much information the network carries, and *where*
it is blind.

U-Net backbone with MC-Dropout for uncertainty (dropout stays on at inference,
N forward passes give a predictive variance), skip connections gated on local
observation density, FiLM conditioning on the number of observations,
GroupNorm, Huber loss and deep supervision.

The training mask is stochastic: the number and position of hidden pixels
change at every step, from 10 to 80 sensors. One trained model therefore scores
any network geometry without retraining. The loss is measured on the hidden
pixels only, so the model cannot win by copying its input.

### 4.2 Commands

```bash
python 01_autoencoder.py --train --nt 1500 --epochs 200
python 01_autoencoder.py --figures --nt 1500 --checkpoint outputs/vae_best.pt
python 01_autoencoder.py --score --nt 1500
python 01_autoencoder.py --train --figures --score --report --nt 1500
```

### 4.3 Parameters

| flag | default | meaning |
|---|---|---|
| `--nt` | `config.NT` | nature run length in days |
| `--seed_ocean` / `--seed_buoys` | 42 / 7 | ocean and reference network |
| `--epochs` | 100 | |
| `--batch_size` | 16 | |
| `--lr` | 3e-4 | AdamW, cosine schedule with warm-up |
| `--base_ch` | 32 | U-Net width, the main cost/quality knob |
| `--latent_ch` | 64 | bottleneck depth |
| `--dropout_p` | 0.1 | MC-Dropout rate, active at inference too |
| `--w_unobs` | 4.0 | weight of unobserved pixels in the loss |
| `--lambda_grad` | 0.5 | gradient matching, keeps fronts sharp |
| `--huber_delta` | 0.5 | |
| `--n_obs_min` / `--n_obs_max` | 10 / 80 | random mask size range |
| `--n_mc_val` / `--n_mc` | 15 / 60 | MC passes for validation / figures |
| `--n_proposed` | 3 | new buoys proposed from the gap map |
| `--gap_influence_km` | `INFLUENCE_RADIUS_KM` | where the gap map distance term saturates |
| `--gap_margin_px` | half an influence radius | keeps proposals off the domain edge |
| `--gap_min_sep_px` | one influence radius | keeps proposals away from existing sensors and from each other |

### 4.4 Reading the output

Validation RMSE is reported **per variable in physical units**, for instance
`2.10 °C` and `0.167 psu`, not a single aggregate. The two channels are
normalised by very different standard deviations, so an aggregate converts to
neither.

`vae_network_evaluation.png` shows, for a given network: true field with
sensors coloured by their leave-one-out contribution, reconstruction, MC
uncertainty map, gap map with the proposed new buoys, and a bar chart ranking
sensors from indispensable to redundant.

**A negative leave-one-out score means removing that sensor improves the
reconstruction**: it was contributing redundancy and noise, nothing else.

**On the gap map.** The distance term saturates at the influence radius. Past
that scale a sensor constrains nothing, so being 200 km from the nearest buoy
is not twice as valuable as being 90 km away. Normalising by the global maximum
instead, as an earlier version did, made the corners of the domain win almost
every time. Proposals are also kept off the edges and at least one influence
radius from any existing sensor. If the domain is too crowded the constraints
are relaxed in steps, and the relaxation is printed.

### 4.5 Fast example

```bash
python 01_autoencoder.py --train --nt 365 --epochs 10 --base_ch 8 \
    --latent_ch 16 --batch_size 4 --n_mc_val 3
```

---

## 5. Graph network, structure and redundancy

`02_gnn.py`

### 5.1 Purpose

Model the network as a graph: nodes are sensors, edges encode correlation.
Learn which nodes carry unique information and which are redundant; attention
weights are the redundancy signal. A GraphSAGE branch runs inductively, so a
hypothetical mooring, glider or float can be scored without retraining.

### 5.2 The seasonal cycle trap

Correlations are computed on **de-seasonalised** anomalies by default. The
seasonal cycle is a near-uniform mode: keep it and two buoys 1000 km apart
correlate strongly simply because they both see summer arrive.

| nt = 365, threshold 0.35 | mean \|ρ\| | edges / 435 pairs | density |
|---|---|---|---|
| raw | 0.446 | 318 | 73 % |
| de-seasonalised | 0.169 | 36 | 8.3 % |

At 73 % density the graph is a near-clique and redundancy is meaningless.
`--deseason 0` restores the old behaviour, in which case raise
`--corr_threshold` to around 0.6 or the graph saturates.

Node features: normalised position, maximum correlation with any neighbour,
degree, and local SST/SSS variance, standardised separately since var(SST) ≈
3 °C² against var(SSS) ≈ 0.03 psu² and an unstandardised mix erases salinity.

### 5.3 Inductive scoring is trained, not random

The GraphSAGE head is **trained** on the existing network before it scores
anything, with 20 % of nodes held out. Generalising to nodes that did not exist
at training time is the whole point, so the held-out MSE is the only honest
indication of what a prediction on a brand new position is worth. It is printed
and written onto the figure.

If that MSE sits close to the target variance, the model has no skill on unseen
nodes and the colours should not be over-read.

### 5.4 Commands

```bash
python 02_gnn.py --train --analyze --nt 1500
python 02_gnn.py --train --analyze --inductive \
    --new_positions "[(20,40),(90,160),(140,60)]" --nt 1500
python 02_gnn.py --train --analyze --inductive --report --nt 1500
```

In pipeline mode the positions are not given by hand: the GNN scores whatever
the autoencoder proposed.

### 5.5 Parameters

| flag | default | meaning |
|---|---|---|
| `--nt` | 500 | nature run length in days |
| `--seed_ocean` / `--seed_buoys` | 42 / 7 | |
| `--n_buoys` | `config.N_BUOYS` (30) | network size |
| `--corr_threshold` | `config.GNN_CORR_THRESHOLD` (0.35) | \|ρ\| above which an edge is created |
| `--k_nearest` | 4 | geographic k-NN edges, guarantee connectivity |
| `--deseason` | 1 | remove the domain mean before correlating |
| `--gnn_epochs` | 200 | |
| `--sage_epochs` | falls back to `--gnn_epochs` | inductive head |
| `--new_positions` | three hard-coded pixels | standalone mode only |

---

## 6. Reinforcement learning, optimisation

`03_rl.py`

### 6.1 Purpose

Search directly for the best network under constraints. A PPO agent toggles
candidate positions on a coarse grid; the reward is the marginal information
gain minus a budget penalty. Two Pareto fronts come out: information against
number of buoys, and information against operating cost and carbon.

### 6.2 The information criterion

Default `--info_mode evf`: **explained variance** by optimal linear estimation
(BLUE, optimal interpolation), the standard OSSE criterion.

```
EVF = Σ_c  C_cO (C_OO + R)⁻¹ C_Oc  /  Σ_c C_cc
```

The observation vector holds **both SST and SSS** at every buoy, 2n
observations, each normalised by its own standard deviation and given its own
instrumental noise, so salinity actually counts. The criterion is increasing,
saturating and submodular, which guarantees diminishing returns and a
well-defined elbow.

**The covariance is not empirical, and that matters.** Mesoscale decorrelation
is ~12 days, so one year of nature run holds only about 30 independent
realisations against 2n = 40 parameters as soon as you have 20 buoys. The raw
sample covariance overfits massively: measured out of sample, explained
variance goes *negative* (−0.49 at N = 20, while the in-sample score claimed
0.62). The covariance is therefore shrunk towards a parametric model
σ(x)·exp(−d²/2L²) built from the nature run's own diagnostics, exactly as
operational optimal interpolation does. `EVF_SHRINKAGE = 0.9`.

`coverage` (fast geometric kernel) and `legacy` (historical formula, not
monotone in N) also exist, for comparison only.

> **Reporting a number?** Use `--evf_cv 1`. Statistics are then estimated on the
> first half of the series and the score measured on the second. It is markedly
> lower than the analytical score, and it is the defensible figure. Keep the
> analytical mode for optimisation, it is smoother. One more reason to run long:
> at `--nt 1500` the two halves are 750 days each.

### 6.3 The separation constraint, and the size ceiling

Two buoys cannot occupy adjacent cells of the candidate grid. This is a **hard**
constraint, enforced by masking the actor's logits, not a reward penalty.

```
n_feasible_max = ceil(grid_x / min_sep) × ceil(grid_y / min_sep)
```

| grid | candidates | feasible maximum |
|---|---|---|
| 16 × 24 | 384 | 96 |
| 8 × 12 | 96 | 24 |

`--n_max` is **also a hard cap**: once the network holds `n_max` buoys every
activation is masked, and the Pareto sweep is clamped to `[n_min, n_max]`.

> **If N★ comes out equal to `--n_max`, the cap is binding** and the elbow is
> your constraint rather than a property of the data. Raise `n_max` and look
> again. The cap bounds the search, it should not pre-decide the answer.

> **`min_sep` counts grid cells, not kilometres.** On a 16 × 24 grid over the
> 800 × 1200 km domain a cell is 50 km, so `--min_sep 2` is 100 km, matching
> the 90 km influence radius. Double the grid to 32 × 48 without touching it
> and the effective separation halves to 50 km, well inside the influence
> radius, and buoys start clustering. Scale it: `--min_sep 4`, which also keeps
> `n_feasible_max` at 96.

Worth knowing before enlarging the grid: at 50 km spacing you already sample
the 95 km decorrelation better than Nyquist. Going finer quadruples the action
space and the greedy cost for very little information.

### 6.4 Commands

```bash
python 03_rl.py --train --nt 1500 --rl_steps 50000
python 03_rl.py --pareto --nt 1500 --report
python 03_rl.py --multiobj --nt 1500 --report
python 03_rl.py --train --pareto --multiobj --gif --report --nt 1500 --rl_steps 50000
```

`--pareto` and `--multiobj` reload `outputs/rl_best.pt` if it exists, so they
run without `--train`. Grid parameters must match between training and fronts,
or the checkpoint will not load.

`rl_optimal_network.png` and `rl_pareto_cost.png` are produced **only** in
standalone mode; the pipeline does not call `--multiobj`.

### 6.5 Parameters

**MDP and constraints**

| flag | default | meaning |
|---|---|---|
| `--grid_x` / `--grid_y` | 16 / 24 | candidate grid, K = grid_x × grid_y actions |
| `--n_min` / `--n_max` | 10 / 40 | hard bounds on active buoys |
| `--min_sep` | `config.MIN_SEP_CELLS` (2) | minimum separation, in grid cells |
| `--episode_len` | 20 | toggles per episode |

**Information criterion**

| flag | default | meaning |
|---|---|---|
| `--info_mode` | `evf` | `evf` \| `coverage` \| `legacy` |
| `--influence_km` | `config.INFLUENCE_RADIUS_KM` (90) | |
| `--evf_shrink` | `config.EVF_SHRINKAGE` (0.9) | |
| `--evf_cv` | 0 | 1 = score validated out of sample |

**PPO**

| flag | default | meaning |
|---|---|---|
| `--rl_steps` | 50000 | environment steps |
| `--buffer_size` | 512 | rollout buffer |
| `--lr` | 3e-4 | |
| `--w_info` / `--w_budget` | 1.0 / 0.5 | reward weights |

### 6.6 Reading the fronts

`rl_pareto_front.png`, three panels: the cloud of evaluated configurations by
source (random baseline, PPO policy, greedy reference with its 1 − 1/e
submodular guarantee), the non-dominated set and N★; the marginal gain per
added buoy; and N★ as a function of λ, the marginal cost of a buoy, from
sweeping `max_N [info(N) − λ·N]`. That last panel answers "what is the best
compromise".

The gap between the learned policy and the greedy reference is itself a
convergence check. Greedy is hard to beat on a static submodular objective; the
agent earns its place when the objective stops being submodular, which is where
logistics, maintenance routes and multi-year decisions live.

`rl_pareto_cost.png`, information against operating cost, where

```
cost = N · COST_BUOY_FIXED + tour_length · COST_SHIP_PER_KM · N_CAMPAIGNS_YEAR
```

The maintenance tour starts from a port, visits every buoy by nearest neighbour
and returns. Cost is therefore **not** proportional to N: at fixed N it varies
by a factor 1.3 to 1.6 depending on how spread out the network is. That is what
makes the two objectives genuinely antagonistic, and the non-domination test
worth doing.

`rl_info_vs_cost_networks.png`, pipeline only, makes the same point on a map:
the same number of buoys, optimised on information alone against information
plus cost, side by side. By default both networks come from the same optimiser,
so the difference is the objective and not the quality of the search.
`--cost_compare_ref rl` compares against the agent's own network instead, which
is what you want if the agent is well trained, and what you should avoid if it
is not: a weak agent will be beaten by the cost-aware greedy on *both* axes and
the figure will prove the wrong thing.

---

## 7. Orchestrator

```bash
python run_demo.py --mode individual --nt 1500
python run_demo.py --mode pipeline --nt 1500 --seed_ocean 42 --seed_buoys 7
```

`run_demo.py` writes a timestamped `report_*.txt` gathering every metric, plus
a reproducibility block and the nature-run diagnostic figure. The header
carries the decorrelation length and mesoscale timescale, which are the
reference spacing and sampling frequency the whole design rests on.

Pipeline stages:

```
1.  RL          proposes a network under cost and separation constraints
1b. compare     the same size, with and without cost in the objective
2.  GNN         redundancy, coverage, structure
3.  AE          reconstruction, uncertainty, where to add sensors
3b. GNN         scores the positions the AE just proposed
```

Beyond the components' own flags:

| flag | default | meaning |
|---|---|---|
| `--no_inductive` | off | skip stage 3b |
| `--n_inductive` | 3 | candidates scored when the AE proposes none |
| `--inductive_min_sep` | 40 px | spacing between scored candidates |
| `--n_proposed` | 3 | buoys the AE proposes, and therefore what the GNN scores |
| `--gap_margin_px`, `--gap_min_sep_px` | see §4.3 | passed through to the AE |
| `--no_cost_compare` | off | skip stage 1b |
| `--cost_info_tol` | 0.10 | acceptable information loss for the cost-aware network |
| `--cost_n_lambda` | 8 | lambda values swept |
| `--cost_compare_ref` | `greedy` | `greedy` isolates the cost term, `rl` compares against the agent's network |
| `--ocean_gif` | off | animate the nature run |
| `--ocean_gif_every` | 5 | one frame every N days |
| `--ocean_gif_var` | `T,GRADT,S,GRADS` | comma-separated field list |
| `--ocean_gif_fps` | 8 | |

Plus `--mode`, `--nt`, `--seed_ocean`, `--seed_buoys`, `--n_buoys`,
`--ae_epochs`, `--ae_base_ch`, `--gnn_epochs`, `--gnn_corr_threshold`,
`--rl_steps`, `--rl_grid_x`, `--rl_grid_y`, `--rl_n_min`, `--rl_n_max`,
`--rl_info_mode`, `--rl_min_sep`, `--rl_influence_km`, `--rl_episode_len`,
`--gif_frames`, `--output_dir`, `--no_nature_fig`.

Smoke test:

```bash
python run_demo.py --mode pipeline --nt 90 --ae_epochs 1 --ae_base_ch 8 \
    --gnn_epochs 20 --rl_steps 400 --gif_frames 5
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
| `KAPPA` | 25.0 | diffusivity (m²/s), sets the dissipation scale |

**Circulation**

| name | default | |
|---|---|---|
| `U_GYRE`, `U_JET` | 0.08, 0.55 | m/s |
| `JET_WIDTH_KM`, `JET_LAT_FRAC` | 40.0, 0.55 | |
| `N_EDDIES` | 22 | simultaneous eddies |
| `EDDY_V_MAX`, `EDDY_R_KM`, `EDDY_LIFE_DAYS` | 0.25, (35, 80), (60, 180) | |
| `RD_KM` | 25.0 | Rossby radius, sets β-drift |

**Tracers**

| name | default | |
|---|---|---|
| `SST_MEAN`, `SST_GRADIENT`, `SST_SEASONAL_AMP` | 15.0, 9.0, 2.5 | °C |
| `TAU_T_DAYS` | 40.0 | thermal restoring |
| `SSS_MEAN`, `SSS_GRADIENT`, `SSS_PLUME_AMP` | 35.0, 1.30, 0.75 | psu |
| `TAU_S_DAYS` | 150.0 | haline restoring, much slower, no feedback |
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
> Calibrate them against the real costs of your target network. That is why
> they are isolated in the config.

---

## 9. Output files

Everything lands in `--output_dir`, default `outputs/`.

| file | produced by |
|---|---|
| `ocean_nature_run.png` | `data/dataset.py`, `run_demo.py` |
| `ocean_nature_run.gif` | `--gif` / `--ocean_gif` |
| `vae_best.pt`, `vae_training_curves.png` | `01 --train` |
| `vae_network_evaluation.png`, `vae_uncertainty_density.png` | `01 --figures`, pipeline |
| `vae_loo_scores.json` | `01 --score` |
| `gnn_best.pt`, `gnn_network_analysis.png` | `02 --train --analyze`, pipeline |
| `sage_best.pt`, `gnn_inductive_eval.png` | `02 --inductive`, pipeline stage 3b |
| `rl_best.pt`, `rl_training_curves.png`, `rl_progression.gif` | `03 --train`, pipeline |
| `rl_optimal_network.png` | `03 --train` standalone |
| `rl_pareto_front.png`, `rl_two_configs.png` | `03 --pareto`, pipeline |
| `rl_pareto_front_pipeline.png` | pipeline, the front with the retained configuration marked |
| `rl_info_vs_cost_networks.png` | pipeline stage 1b |
| `rl_pareto_cost.png` | `03 --multiobj` standalone only |
| `report_*.txt` | any `--report`, and `run_demo.py` |

Checkpoints, GIFs and the whole of `outputs/` are git-ignored.

---

## 10. Known limitations

**No assimilation in the loop.** Information is measured by optimal linear
estimation, so what is quantified is how reconstructable the analysed field is,
not how much forecast error goes down. Whether one is an acceptable surrogate
for the other is an open question, not a settled one. This is the limitation
that matters most.

**Single-domain OSSE.** A mid-latitude zonal channel. Other regimes sit in very
different dynamics; transposing means recalibrating `LAT0`, the gradients and
the eddy statistics, or plugging a real model output through the same
interface.

**Surface only.** SST, SSS, SSH. No vertical structure, no biogeochemistry.

**Reproducibility covers the ocean and the network, not the training.**
`--seed_ocean` and `--seed_buoys` fully determine the nature run and the
reference network. PyTorch seeds are not fixed, so two training runs differ
slightly.

**Nature run length.** Below `--nt 365` the seasonal cycle is not sampled over a
full period and correlation statistics are biased. Even at 365 days the
mesoscale holds only ~30 independent realisations, which is why the EVF
covariance has to be shrunk. See §3.5.

**Cost model.** The nearest-neighbour tour overestimates the optimal route by
roughly 10 to 25 % on a Euclidean TSP. If transit cost dominates in your case, a
2-opt pass would tighten it.

**Old checkpoints.** GNN node features went from 4 to 6 dimensions; a
`gnn_best.pt` from an earlier version will not reload. Regenerate it.

---

## 11. Citing NAIADE

```bibtex
@software{naiade,
  author  = {Vient, Jean-Marie},
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
| RL, information criterion | Explained variance by optimal linear estimation, with error maps used to design the array. The Gauss–Markov objective-analysis framework introduced to oceanography for MODE-73. | Bretherton, Davis & Fandry (1976) |
| RL, covariance regularisation | Shrinkage of the sample covariance towards a structured target when the effective sample size is small. | Ledoit & Wolf (2004) |
| RL, greedy reference | The criterion is monotone submodular, so greedy selection is within 1 − 1/e of the optimum. | Nemhauser, Wolsey & Fisher (1978) |
| RL, sensor placement framing | Near-optimal sensor placement in Gaussian processes; mutual information and submodularity in spatial monitoring. | Krause, Singh & Guestrin (2008) |
| RL, policy | PPO, discrete action space, clipped surrogate objective. | Schulman et al. (2017) |
| RL, elbow detection | Maximum distance to the chord on a concave saturating curve. | Satopää et al. (2011) |
| RL, mooring array precedent | Model-based assessment and design of a tropical mooring array, the closest published analogue. | Oke & Schiller (2007) |
| GNN, attention as redundancy | Graph attention networks; attention weights read as neighbour influence. | Veličković et al. (2018) |
| GNN, inductive scoring | GraphSAGE: aggregation functions that generalise to unseen nodes. | Hamilton, Ying & Leskovec (2017) |
| AE, uncertainty | MC-Dropout: dropout kept active at inference, N passes give an approximate posterior. | Gal & Ghahramani (2016) |
| AE, backbone | U-Net encoder–decoder with skip connections. | Ronneberger, Fischer & Brox (2015) |
| AE, conditioning on N_obs | FiLM: feature-wise linear modulation by a conditioning vector. | Perez et al. (2018) |
| AE, bottleneck attention | CBAM: sequential channel and spatial attention. | Woo et al. (2018) |
| Ocean, advection | Semi-Lagrangian integration with iterated midpoint departure points. | Staniforth & Côté (1991) |
| Ocean, interpolation | Catmull–Rom cubic, chosen over bilinear to preserve filaments. | Catmull & Rom (1974) |
| Ocean, k⁻³ spectrum | Geostrophic turbulence: enstrophy cascade and the mesoscale slope. | Charney (1971) |
| Ocean, k⁻² submesoscale slope | Surface quasi-geostrophic dynamics. | Held et al. (1995) |
| Ocean, eddy propagation | Westward β-drift, observed eddy lifetimes and radii. | Chelton, Schlax & Samelson (2011) |
| Ocean, density | EOS-80 one-atmosphere equation of state, for σ₀ and the T–S isopycnals. | Millero & Poisson (1981) |
| Framework, OSSE context | Requirements for an integrated in situ observing system from coordinated OSSEs. | Gasparin et al. (2019) |

### 12.2 Bibliography

**Foundations, OED and observing-network design**

- Krause, A., Singh, A., & Guestrin, C. (2008). Near-optimal sensor placements in Gaussian processes. *JMLR*, 9, 235–284.
- Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions I. *Mathematical Programming*, 14(1), 265–294.
- Wikle, C. K., & Royle, J. A. (1999). Space-time dynamic design of environmental monitoring networks. *JASA*, 94(445), 1–11.
- Huan, X., & Marzouk, Y. M. (2013). Simulation-based optimal Bayesian experimental design for nonlinear systems. *JCP*, 232(1), 288–317.
- Chaloner, K., & Verdinelli, I. (1995). Bayesian experimental design: a review. *Statistical Science*, 10(3), 273–304.
- Ryan, K. J. (2003). Estimating expected information gains for experimental designs. *JCGS*, 12(3), 585–603.

**OSSE and oceanographic array design**

- Bretherton, F. P., Davis, R. E., & Fandry, C. B. (1976). A technique for objective analysis and design of oceanographic experiments applied to MODE-73. *Deep-Sea Research*, 23(7), 559–582.
- Oke, P. R., & Schiller, A. (2007). A model-based assessment and design of a tropical Indian Ocean mooring array. *Journal of Climate*, 20(13), 3269–3283.
- Gasparin, F., et al. (2019). Requirements for an integrated in situ Atlantic Ocean observing system from coordinated OSSEs. *Frontiers in Marine Science*, 6, 83.
- Heimbach, P., et al. (2019). Putting it all together. *Frontiers in Marine Science*, 6, 55.
- Sakov, P., & Sandery, P. A. (2017). An adaptive quality control procedure for data assimilation. *Tellus A*, 69(1), 1318031.

**Ocean dynamics and numerics**

- Charney, J. G. (1971). Geostrophic turbulence. *JAS*, 28(6), 1087–1095.
- Held, I. M., et al. (1995). Surface quasi-geostrophic dynamics. *JFM*, 282, 1–20.
- Chelton, D. B., Schlax, M. G., & Samelson, R. M. (2011). Global observations of nonlinear mesoscale eddies. *Progress in Oceanography*, 91(2), 167–216.
- Staniforth, A., & Côté, J. (1991). Semi-Lagrangian integration schemes for atmospheric models. *MWR*, 119(9), 2206–2223.
- Catmull, E., & Rom, R. (1974). A class of local interpolating splines. In *Computer Aided Geometric Design*, 317–326.
- Millero, F. J., & Poisson, A. (1981). International one-atmosphere equation of state of seawater. *Deep-Sea Research A*, 28(6), 625–629.

**Statistics and machine learning**

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *JMVA*, 88(2), 365–411.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*, PMLR 48, 1050–1059.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net. *MICCAI*, LNCS 9351, 234–241.
- Perez, E., et al. (2018). FiLM: visual reasoning with a general conditioning layer. *AAAI*.
- Woo, S., et al. (2018). CBAM: convolutional block attention module. *ECCV*.
- Satopää, V., et al. (2011). Finding a "kneedle" in a haystack. *ICDCS Workshops*.
- Poole, B., et al. (2019). On variational bounds of mutual information. *ICML*, PMLR 97, 5171–5180.

**Autoencoders and representation learning for geophysical data**

- Shi, X., et al. (2015). Convolutional LSTM network. *NeurIPS*, 28.
- Manucharyan, G. E., et al. (2021). A deep learning approach to spatiotemporal sea surface temperature variability. *JPO*, 51(6), 1809–1824.
- Lguensat, R., et al. (2018). The analog data assimilation. *MWR*, 145(10), 4093–4107.
- Fablet, R., et al. (2021). Learning variational data assimilation models and solvers. *JAMES*, 13(10), e2021MS002572.
- Grooms, I., et al. (2023). Hybrid ensemble-variational algorithms for data assimilation. *Frontiers in Applied Mathematics and Statistics*, 9.

**Graph neural networks for spatial and climate systems**

- Veličković, P., et al. (2018). Graph attention networks. *ICLR*.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS*, 30.
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
- Cachay, S. R., et al. (2021). The world as a graph: improving El Niño forecasts with GNNs. *NeurIPS Workshop Climate Change AI*.
- Lam, R., et al. (2023). GraphCast. *Science*, 382(6677), 1416–1421.
- Rossi, E., et al. (2020). Temporal graph networks for deep learning on dynamic graphs. *ICML Workshop GRL+*.

**Reinforcement learning for physical-system optimisation**

- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Haarnoja, T., et al. (2018). Soft actor-critic. *ICML*.
- Duffield, S., et al. (2022). Deep reinforcement learning for adaptive ocean observation. *Environmental Data Science*, 1, e13.
- Petersen, M. N., et al. (2022). Autonomous ocean sampling with a multi-agent reinforcement learning approach. *Ocean Science*, 18, 1653–1669.

**Multi-objective optimisation and carbon footprint**

- Deb, K., et al. (2002). NSGA-II. *IEEE TEC*, 6(2), 182–197.
- Hernandez-Lobato, J. M., et al. (2016). Predictive entropy search for multi-objective Bayesian optimization. *NeurIPS*, 29.
- Racault, M.-F., et al. (2023). Towards sustainable ocean observation: carbon footprint benchmarking. *Frontiers in Marine Science*, 10, 1101993.

**Target networks**

- Coppola, L., et al. (2019). A posteriori quality control of the MOOSE-GE cruises. *Frontiers in Marine Science*, 6, 233.
- Bourlès, B., et al. (2019). PIRATA: a sustained observing system for tropical Atlantic climate research and forecasting. *BAMS*, 100(4), 655–686.
- Testor, P., et al. (2018). OceanGliders: a component of the integrated GOOS. *Frontiers in Marine Science*, 6, 422.

---

## Contact

Jean-Marie Vient, Shom, Brest. jean-marie.vient@shom.fr
