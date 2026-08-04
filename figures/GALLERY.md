# NAIADE — Result gallery

Every figure below was produced by the commands shown, from a clean checkout,
with **`--seed_ocean 42 --seed_buoys 7 --nt 365`**. Copy a command, run it, and
you get the same ocean and the same reference network.

Reports and raw scores for this run are in [`reports/`](reports/).

> **Reproducibility scope.** The seeds fully determine the nature run and the
> reference buoy network. PyTorch seeds are *not* fixed, so training curves and
> learned quantities (AE reconstruction, GNN attention, PPO policy) will differ
> slightly between runs. Everything derived from the ocean and the network
> geometry — decorrelation scales, correlation matrix, greedy Pareto front,
> cost table — is bit-for-bit reproducible.

> **These figures are a showcase, not a benchmark.** They were produced on CPU
> under a compute budget. The autoencoder in particular ran **4 epochs at
> `base_ch 16`** instead of the nominal 100 epochs at `base_ch 32`. Its RMSE is
> therefore far from what the architecture reaches — treat figures 2–4 as
> illustrations of the *output structure*, not of achievable accuracy. Nominal
> commands are given at the end of each section.

Approximate wall-clock times are for a CPU container; they will vary a lot with
hardware.

---

## 1. The synthetic ocean

```bash
python data/dataset.py --nt 365 --seed 42 --out docs/figures/01_nature_run.png
```
*~30 s*

![Nature run](figures/01_nature_run.png)

Sixteen diagnostic panels. What to look at:

- **Row 1** — SST at three dates plus temporal variability. The meandering
  front, the eddies and the filaments are *not drawn*: they emerge from the
  competition between stirring by the geostrophic flow and restoring towards
  climatology.
- **Row 2** — SSH with geostrophic streamlines, relative vorticity ζ/f, |∇SST|
  which makes the filaments explicit, and SSS.
- **Row 3** — radial spectrum against k⁻² and k⁻³ references, spatial and
  temporal autocorrelations, and the T–S diagram with σ₀ isopycnals. The
  temporal autocorrelation panel shows both the raw curve and the de-seasonalised
  one: the gap between them is the entire reason Bricks 2 and 3 remove the
  domain mean before analysing anything.
- **Row 4** — time series, distributions, the spatially varying T–S correlation
  map, and a sample buoy network.

Diagnostics printed by this run:

| quantity | value |
|---|---|
| σ(SST) | 2.60 °C |
| σ(SSS) | 0.177 psu |
| spatial decorrelation length | 95 km |
| mesoscale decorrelation time | 12 days |
| SST decorrelation time, total | 53 days |
| seasonal SST range | 4.13 °C |
| T–S correlation | +0.774 |
| Rossby number (p99) | 0.390 |
| RMS speed | 0.248 m/s |

The 95 km and 12 days are the two numbers everything downstream rests on: they
are the reference sensor spacing and sampling frequency.

---

## 2. Brick 1 — Observability autoencoder

```bash
python 01_autoencoder.py --train --nt 365 --seed_ocean 42 --seed_buoys 7 \
    --epochs 4 --base_ch 16 --latent_ch 32 --cond_dim 16 --batch_size 8 \
    --n_mc_val 4 --output_dir docs/figures
```
*~5 min*

![AE training curves](figures/02_ae_training_curves.png)

Five panels: total loss, validation RMSE on unobserved pixels, then **RMSE in
physical units, separately for SST and SSS**. That separation matters — the two
channels are normalised by 2.60 °C and 0.177 psu, so a single aggregate RMSE
converts to neither. Final values for this short run: **0.935 °C** and
**0.0746 psu**.

```bash
python 01_autoencoder.py --figures --score --report --nt 365 \
    --seed_ocean 42 --seed_buoys 7 --base_ch 16 --n_mc 30 --n_mc_val 5 \
    --checkpoint docs/figures/vae_best.pt --output_dir docs/figures
```
*~8 min (MC-Dropout with 30 passes is the cost)*

![AE network evaluation](figures/03_ae_network_evaluation.png)

For the 30-buoy reference network: ground truth with sensors coloured by their
leave-one-out contribution, reconstruction, MC uncertainty, and the gap map
(high σ **and** far from any sensor) with three greedily proposed new buoys. The
bottom bar chart ranks sensors from indispensable to redundant — a negative
delta means removing that sensor *improves* the reconstruction, i.e. it is pure
redundancy plus noise.

![AE uncertainty vs density](figures/04_ae_uncertainty_density.png)

How predictive uncertainty responds to network density, at 40, 20 and 8
observations. This is the panel that turns "we have N buoys" into "we can
reconstruct the field to within X".

Nominal run:

```bash
python 01_autoencoder.py --train --nt 365 --epochs 100 --base_ch 32
python 01_autoencoder.py --figures --score --report --nt 365
```

---

## 3. Brick 2 — Graph neural network

```bash
python 02_gnn.py --train --analyze --inductive --report --nt 365 \
    --seed_ocean 42 --seed_buoys 7 --gnn_epochs 200 --output_dir docs/figures
```
*~2 min*

![GNN network analysis](figures/05_gnn_network_analysis.png)

Network as a graph: 30 nodes, 168 edges, mean |ρ| = 0.162 with **6.0 % of pairs
above the 0.35 threshold**. Compare with the same computation without
de-seasonalising, where 73 % of pairs pass and the graph collapses into a
near-clique — the seasonal cycle is a global mode and it drowns the mesoscale
structure the network is actually meant to sample.

Panels: contribution scores over the SST field with redundant sensors flagged,
the correlation matrix, the attention-derived redundancy, spatial coverage, and
a contribution-versus-redundancy bar plot. Removal candidates from this run:

```
C14 @ (76, 121) | contribution=-0.009 | redundancy=0.973
C29 @ (70, 123) | contribution= 0.108 | redundancy=0.966
C13 @ (158,106) | contribution= 0.439 | redundancy=1.000
```

![GNN inductive evaluation](figures/06_gnn_inductive_eval.png)

GraphSAGE scoring three hypothetical sensor positions **without retraining**.
This is the answer to "what would a new glider line bring us" before anything is
deployed.

---

## 4. Brick 3 — Reinforcement learning

```bash
python 03_rl.py --train --pareto --gif --report --nt 365 --seed_ocean 42 \
    --rl_steps 20000 --grid_x 16 --grid_y 24 --n_min 10 --n_max 40 \
    --min_sep 2 --n_random 12 --gif_frames 40 --output_dir docs/figures
```
*~3 min*

![RL training curves](figures/07_rl_training_curves.png)

PPO on 384 candidate positions, feasible maximum 96 buoys under the two-cell
separation constraint.

![RL optimal network](figures/08_rl_optimal_network.png)

Final configuration over the local-variance field. Note that no two active
positions are adjacent — the constraint is enforced by masking the actor's
logits, not by a reward penalty, so it holds exactly rather than on average.

### Pareto front — information vs number of buoys

![Pareto front info vs N](figures/09_rl_pareto_front_info_vs_N.png)

The headline result:

> **N★ = 23 buoys, 63.7 % of mesoscale variance explained.
> 23 optimised buoys are worth 43 randomly placed ones.**

Left panel — every evaluated configuration, coloured by how it was generated:
random draws (the baseline, i.e. what an unoptimised network is worth), the PPO
policy, and greedy selection (the high reference, guaranteed within 1 − 1/e of
the optimum by submodularity). Non-dominated configurations are circled. After
20 000 steps the policy sits clearly above random and close to greedy; the
remaining gap is a useful convergence metric in its own right.

Middle panel — marginal gain per added buoy, with a 20 %-of-initial-gain
threshold.

Right panel — N★ as a function of λ, the marginal cost of one buoy, from
sweeping `max_N [info(N) − λ·N]`. This is the panel that answers "what is the
best compromise": a more expensive buoy pushes the optimum towards a lighter
network.

![Two configurations](figures/10_rl_two_configs.png)

Dense versus light configuration, side by side, with the information loss
quantified.

### Pareto front — information vs cost and carbon

```bash
python 03_rl.py --multiobj --report --nt 365 --seed_ocean 42 \
    --grid_x 16 --grid_y 24 --n_min 10 --n_max 40 --min_sep 2 \
    --n_random 8 --output_dir docs/figures
```
*~2.5 min*

![Pareto front cost and carbon](figures/11_rl_pareto_cost_carbon.png)

Cost includes the maintenance tour from a port, so it is **not** proportional to
N: at fixed N it varies by a factor 1.3–1.6 depending on how spread out the
network is. That is what makes the two objectives genuinely antagonistic and the
non-domination test meaningful.

Directly usable output:

| annual budget | N | explained variance | actual cost | tCO₂/yr |
|---|---|---|---|---|
| 500 k€ | 5 | 0.276 | 488 k€ | 238 |
| 700 k€ | 9 | 0.408 | 699 k€ | 329 |
| 900 k€ | 15 | 0.525 | 862 k€ | 379 |
| 1100 k€ | 19 | 0.586 | 1042 k€ | 452 |
| 1400 k€ | 28 | 0.687 | 1332 k€ | 553 |

Cost parameters (`COST_BUOY_FIXED`, `COST_SHIP_PER_KM`, `N_CAMPAIGNS_YEAR`,
`CO2_SHIP_PER_KM`) are indicative orders of magnitude, not sourced figures.
Calibrate them against the real costs of your target SNO — they are isolated in
`config.py` for exactly that reason.

![RL progression](figures/12_rl_progression.gif)

The agent reconfiguring the network, with the cumulative reward curve alongside.

---

## 5. Full pipeline

```bash
python run_demo.py --mode pipeline --nt 365 --seed_ocean 42 --seed_buoys 7
```

RL proposes an optimal network → the GNN scores its structure → the autoencoder
maps its blind spots. Writes a timestamped report with every metric plus a
reproducibility JSON block.

```bash
python run_demo.py --mode individual --nt 365 --seed_ocean 42 --seed_buoys 7
```

The three bricks independently on the same ocean and the same initial network,
for brick-by-brick comparison.

---

## 6. Reproducing the whole gallery

```bash
mkdir -p docs/figures docs/reports

python data/dataset.py --nt 365 --seed 42 --out docs/figures/01_nature_run.png

python 01_autoencoder.py --train --nt 365 --seed_ocean 42 --seed_buoys 7 \
    --epochs 4 --base_ch 16 --latent_ch 32 --cond_dim 16 --batch_size 8 \
    --n_mc_val 4 --output_dir docs/figures
python 01_autoencoder.py --figures --score --report --nt 365 \
    --seed_ocean 42 --seed_buoys 7 --base_ch 16 --n_mc 30 --n_mc_val 5 \
    --checkpoint docs/figures/vae_best.pt --output_dir docs/figures

python 02_gnn.py --train --analyze --inductive --report --nt 365 \
    --seed_ocean 42 --seed_buoys 7 --gnn_epochs 200 --output_dir docs/figures

python 03_rl.py --train --pareto --gif --report --nt 365 --seed_ocean 42 \
    --rl_steps 20000 --grid_x 16 --grid_y 24 --n_min 10 --n_max 40 \
    --min_sep 2 --n_random 12 --gif_frames 40 --output_dir docs/figures
python 03_rl.py --multiobj --report --nt 365 --seed_ocean 42 \
    --grid_x 16 --grid_y 24 --n_min 10 --n_max 40 --min_sep 2 \
    --n_random 8 --output_dir docs/figures
```

Total: roughly 20 minutes on CPU. Figures here were then renamed with a numeric
prefix and downscaled to 2000 px so the repository stays light; the `.pt`
checkpoints were removed.

Grid parameters must match between `--train` and the Pareto fronts, otherwise
the checkpoint will not reload.
