# Reference gallery

Figures from a pipeline run, kept in the repository so the README has something
stable to point at. A run writes to `outputs/`, which is git-ignored; refresh
the gallery by copying across what you want to keep.

| File | What it shows |
|---|---|
| `ocean_nature_run.png` | the synthetic ocean and its diagnostics |
| `vae_network_evaluation.png` | reconstruction, uncertainty, gap map, sensor ranking |
| `vae_uncertainty_density.png` | uncertainty against network density |
| `gnn_network_analysis.png` | redundancy, correlation, coverage |
| `gnn_inductive_eval.png` | the AE-proposed positions, scored by the GNN |
| `rl_pareto_front.png` | information against number of buoys |
| `rl_pareto_front_pipeline.png` | the same front, retained configuration marked |
| `rl_two_configs.png` | dense against light network |
| `rl_info_vs_cost_networks.png` | the same size, with and without cost in the objective |
| `rl_training_curves.png` | PPO diagnostics |

Two figures only exist in standalone mode, from `03_rl.py --multiobj`:
`rl_optimal_network.png` and `rl_pareto_cost.png`. The AE training curves
likewise come from `01_autoencoder.py --train --figures`.

Animations stay in `outputs/`: `ocean_nature_run.gif` and `rl_progression.gif`
are git-ignored, they are too heavy for the repository.
