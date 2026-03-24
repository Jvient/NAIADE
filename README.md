# OED-IA SNO Marins (NAIADE) 
Documentation générée par IA (Opus 3) 
Optimal Experimental Design par Intelligence Artificielle — Application aux réseaux d'observation océanographiques.

---

## Table des matières

1. [Structure du projet](#structure-du-projet)
2. [Dépendances](#dépendances)
3. [Lancement rapide](#lancement-rapide)
4. [Orchestrateur — run_demo.py](#orchestrateur)
5. [Brique 1 — Autoencoder](#brique-1--autoencoder)
6. [Brique 2 — GNN](#brique-2--gnn)
7. [Brique 3 — RL](#brique-3--rl)
8. [Données synthétiques](#données-synthétiques)
9. [Reproductibilité](#reproductibilité)
10. [Fichiers produits](#fichiers-produits)

---

## Structure du projet

```
config.py              Configuration partagée (domaine, physique, seeds)
dataset.py             Générateur d'océan synthétique + Dataset PyTorch
01_autoencoder.py      Brique 1 — AE-UNet MC-Dropout (reconstruction + incertitude)
02_gnn.py              Brique 2 — GAT + GraphSAGE (structure réseau + inductif)
03_rl.py               Brique 3 — PPO (optimisation placement capteurs)
run_demo.py            Orchestrateur (individual | pipeline)
outputs/               Répertoire de sortie (créé automatiquement)
```

Les sorties sont archivées dans des sous-dossiers horodatés, par exemple `outputs/pipeline_20260319_143022_so42_sb7/`.

---

## Dépendances

```bash
pip install torch numpy scipy matplotlib
```

Optionnel (pour les `GATConv`/`SAGEConv` natifs, sinon fallback manuel) :

```bash
pip install torch-geometric
```

---

## Lancement rapide

```bash
# Pipeline complet (RL → GNN → AE) avec paramètres démo
python run_demo.py --mode pipeline

# Pipeline + évaluation config légère (N★/2) par GNN + AE
python run_demo.py --mode pipeline --eval_light

# Briques indépendantes
python run_demo.py --mode individual
```

Run réel (individual, environ 30-60 min selon GPU) :

```bash
python run_demo.py --mode individual \
  --seed_ocean 42 --seed_buoys 7 \
  --nt 800 --n_buoys 30 \
  --ae_epochs 80 --ae_base_ch 32 \
  --gnn_epochs 200 \
  --rl_steps 50000 --rl_grid_x 16 --rl_grid_y 24 \
  --rl_n_min 10 --rl_n_max 40
```

Run réel (pipeline avec comparaison dense/légère, environ 25-50 min selon GPU) :

```bash
python run_demo.py --mode pipeline --eval_light \
  --seed_ocean 42 --seed_buoys 7 \
  --nt 800 --n_buoys 30 \
  --ae_epochs 60 --ae_base_ch 32 \
  --gnn_epochs 200 \
  --rl_steps 50000 --rl_grid_x 16 --rl_grid_y 24 \
  --rl_n_min 10 --rl_n_max 40
```

Briques standalone (entraînement complet) :

```bash
python 01_autoencoder.py --train --figures
python 02_gnn.py --train --analyze --inductive
python 03_rl.py --train --evaluate --rl_method pareto
python 03_rl.py --train --evaluate --rl_method efficiency
python 03_rl.py --train --evaluate --rl_method scalarized
```

Générer uniquement la figure du nature run :

```bash
python dataset.py --nt 500 --seed 42
```

---

## Orchestrateur

**`run_demo.py`**

Deux modes d'exécution. Les trois briques partagent le même nature run et le même réseau initial de bouées (reproductibilité via seeds).

- **`individual`** : AE, GNN et RL tournent chacun indépendamment sur le réseau initial.
- **`pipeline`** : RL optimise le réseau, puis GNN l'analyse, puis AE l'évalue.

```bash
python run_demo.py --mode pipeline --seed_ocean 42 --seed_buoys 7
python run_demo.py --mode individual --nt 300 --ae_epochs 10
```

### Arguments

| Argument | Description | Défaut |
|---|---|---|
| `--mode` | `individual` ou `pipeline` | `individual` |
| `--seed_ocean` | Seed du nature run (champs SST/SSS) | `42` |
| `--seed_buoys` | Seed du réseau initial de bouées | `7` |
| `--nt` | Nombre de pas de temps du nature run | `200` |
| `--n_buoys` | Nombre de bouées du réseau initial | `30` |
| `--eval_light` | Active l'évaluation de la config légère (N★/2), mode pipeline uniquement | désactivé |
| `--ae_epochs` | Époques d'entraînement de l'AE | `5` |
| `--ae_base_ch` | Canaux de base de l'AE (16 = démo, 32 = complet) | `16` |
| `--gnn_epochs` | Époques d'entraînement GAT et GraphSAGE | `30` |
| `--rl_steps` | Nombre de steps PPO (2000 = démo, 50000+ = convergence) | `2000` |
| `--rl_grid_x` | Résolution X de la grille candidate RL | `8` |
| `--rl_grid_y` | Résolution Y de la grille candidate RL | `12` |
| `--rl_n_min` | Nombre minimum de bouées actives | `5` |
| `--rl_n_max` | Nombre maximum de bouées actives | `20` |
| `--rl_episode_len` | Actions par épisode RL | `20` |
| `--rl_method` | Méthode de sélection N★ : `pareto`, `efficiency` ou `scalarized` | `pareto` |
| `--gif_frames` | Nombre de frames du GIF de progression RL | `40` |
| `--output_dir` | Répertoire de sortie | `outputs` |

---

## Brique 1 — Autoencoder

**`01_autoencoder.py`**

AE-UNet déterministe avec incertitude par MC-Dropout. Reconstruit les champs SST/SSS à partir d'observations partielles. Produit des cartes d'incertitude, des scores LOO et des propositions de bouées.

```bash
# Entraînement complet
python 01_autoencoder.py --train --epochs 100

# Entraînement + figures diagnostiques
python 01_autoencoder.py --train --figures

# Figures seules depuis un checkpoint
python 01_autoencoder.py --figures --checkpoint outputs/ae_best.pt

# Scoring Leave-One-Out
python 01_autoencoder.py --score --checkpoint outputs/ae_best.pt

# Tout à la fois
python 01_autoencoder.py --train --figures --score
```

### Arguments

| Argument | Description | Défaut |
|---|---|---|
| `--train` | Lance l'entraînement | — |
| `--score` | Calcule les scores LOO par capteur | — |
| `--figures` | Génère les figures diagnostiques | — |
| `--seed_ocean` | Seed du nature run | `42` |
| `--seed_buoys` | Seed des positions de bouées | `7` |
| `--checkpoint` | Chemin du checkpoint à charger | `outputs/ae_best.pt` |
| `--output_dir` | Répertoire de sortie | `outputs` |
| `--epochs` | Nombre d'époques | `100` |
| `--batch_size` | Taille de batch | `16` |
| `--lr` | Learning rate (AdamW) | `3e-4` |
| `--base_ch` | Canaux de base de l'UNet (niveaux : 2x, 4x, 8x, 16x) | `32` |
| `--latent_ch` | Dimension du bottleneck latent | `64` |
| `--cond_dim` | Dimension de l'embedding FiLM | `32` |
| `--dropout_p` | Probabilité de MC-Dropout | `0.1` |
| `--w_unobs` | Poids de la loss sur les pixels non observés | `4.0` |
| `--lambda_grad` | Poids de la loss de gradient spatial | `0.5` |
| `--huber_delta` | Seuil de la loss Huber | `0.5` |
| `--n_obs_min` | Nombre minimum d'observations par échantillon | `10` |
| `--n_obs_max` | Nombre maximum d'observations par échantillon | `80` |
| `--n_mc_val` | Passes MC-Dropout pour la validation | `15` |
| `--n_mc` | Passes MC-Dropout pour les figures | `60` |

---

## Brique 2 — GNN

**`02_gnn.py`**

Graph Attention Network pour analyser la structure du réseau de capteurs. Détecte les redondances, identifie les lacunes, évalue de nouveaux capteurs de manière inductive via GraphSAGE.

```bash
# Entraînement GAT + GraphSAGE + analyse complète
python 02_gnn.py --train --analyze

# Avec évaluation inductive de capteurs hypothétiques
python 02_gnn.py --train --analyze --inductive

# Positions personnalisées pour l'évaluation inductive
python 02_gnn.py --train --inductive --new_positions "[(50,100),(120,200)]"

# Analyse seule depuis checkpoints existants
python 02_gnn.py --analyze

# Évaluation inductive seule
python 02_gnn.py --inductive --new_positions "[(30,60),(90,180),(140,50)]"
```

### Arguments

| Argument | Description | Défaut |
|---|---|---|
| `--train` | Entraîne le GAT et le GraphSAGE | — |
| `--analyze` | Analyse complète du réseau (contribution, redondance, couverture) | — |
| `--inductive` | Évalue des capteurs hypothétiques via GraphSAGE | — |
| `--seed_ocean` | Seed du nature run | `42` |
| `--seed_buoys` | Seed des positions de bouées | `7` |
| `--new_positions` | Positions (x,y) des capteurs hypothétiques | `"[(10,20),(80,150),(130,40)]"` |
| `--corr_threshold` | Seuil de corrélation pour créer une arête | `0.5` |
| `--k_nearest` | Nombre de plus proches voisins géographiques | `4` |
| `--gnn_epochs` | Époques d'entraînement | `200` |
| `--output_dir` | Répertoire de sortie | `outputs` |
| `--n_buoys` | Nombre de bouées du réseau | `30` |

---

## Brique 3 — RL

**`03_rl.py`**

Reinforcement Learning (PPO) pour optimiser le placement des capteurs sur une grille candidate. Produit un front de Pareto information vs nombre de capteurs et compare config dense vs légère.

```bash
# Entraînement PPO + évaluation Pareto
python 03_rl.py --train --evaluate

# Méthode efficacité
python 03_rl.py --train --evaluate --rl_method efficiency

# Méthode scalarisée (sweep lambda)
python 03_rl.py --train --evaluate --rl_method scalarized

# Évaluation seule depuis checkpoint
python 03_rl.py --evaluate --rl_method efficiency

# GIF de progression seul
python 03_rl.py --gif

# Tout
python 03_rl.py --train --evaluate --gif --rl_method pareto
```

### Méthodes de sélection du N★ optimal

- **`pareto`** (défaut) : sweep info vs N avec politique + random, coude Kneedle, front de Pareto.
- **`efficiency`** : eta(N) = info(N) / (1 + log(N)), N★ = argmax(eta). Le log(N) pénalise doucement les grands réseaux.
- **`scalarized`** : entraîne 4 PPO avec lambda croissant comme coût marginal par capteur, le meilleur est choisi par eta.

### Arguments

| Argument | Description | Défaut |
|---|---|---|
| `--train` | Lance l'entraînement PPO | — |
| `--evaluate` | Évalue le réseau avec la méthode choisie | — |
| `--gif` | Génère le GIF animé de progression | — |
| `--rl_method` | `pareto`, `efficiency` ou `scalarized` | `pareto` |
| `--seed_ocean` | Seed du nature run | `42` |
| `--seed_buoys` | Seed (réservé pour usage futur) | `7` |
| `--checkpoint` | Chemin du checkpoint RL | `outputs/rl_best.pt` |
| `--output_dir` | Répertoire de sortie | `outputs` |
| `--rl_steps` | Nombre total de steps PPO (2000 = test, 50000+ = convergence) | `50000` |
| `--buffer_size` | Taille du rollout buffer PPO | `512` |
| `--lr` | Learning rate (Adam) | `3e-4` |
| `--grid_x` | Résolution X de la grille candidate | `16` |
| `--grid_y` | Résolution Y de la grille candidate | `24` |
| `--n_min` | Nombre minimum de bouées actives | `10` |
| `--n_max` | Nombre maximum de bouées actives | `40` |
| `--episode_len` | Actions par épisode | `20` |
| `--w_info` | Poids de la récompense d'information | `1.0` |
| `--w_budget` | Poids de la pénalité de budget | `0.5` |
| `--gif_frames` | Nombre de frames du GIF | `80` |

---

## Données synthétiques

**`dataset.py`**

Générateur de nature run 2D+T avec structures physiques réalistes : double gyre, front zonal à méandres, tourbillons méso-échelle, cycle saisonnier, turbulence spectrale k⁻³, SSS couplée à SST.

```bash
python dataset.py --nt 500 --seed 42
```

| Argument | Description | Défaut |
|---|---|---|
| `--nt` | Nombre de pas de temps | `1000` |
| `--seed` | Seed numpy | `42` |
| `--out` | Chemin de la figure de sortie | `outputs/ocean_nature_run.png` |

---

## Reproductibilité

La fonction `set_global_seed(seed)` dans `config.py` fixe numpy, PyTorch CPU, PyTorch GPU et active le mode déterministe cuDNN.

Deux seeds contrôlent l'ensemble du pipeline :

- **`seed_ocean`** : contrôle le nature run (champs SST/SSS). Même seed = mêmes données océaniques.
- **`seed_buoys`** : contrôle les positions initiales des bouées. Même seed = même réseau de départ.

Pour reproduire exactement un run :

```bash
python run_demo.py --mode pipeline --seed_ocean 42 --seed_buoys 7
```

La seed globale est fixée au démarrage de chaque brique. Tous les tirages aléatoires (initialisations réseau, DataLoader, masques, splits train/test, resets RL) sont reproductibles.

---

## Fichiers produits

Les sorties sont archivées dans des sous-dossiers horodatés, identifiables par mode, date et seeds :

```
outputs/pipeline_20260319_143022_so42_sb7/
outputs/individual_20260319_150100_so42_sb7/
```

### Checkpoints

| Fichier | Description |
|---|---|
| `ae_best.pt` | Meilleur modèle AE (RMSE val minimale) |
| `gnn_best.pt` | Meilleur modèle GAT |
| `sage_best.pt` | Meilleur modèle GraphSAGE |
| `rl_best.pt` | Meilleure politique PPO + active_mask |

### Figures — Autoencoder

| Fichier | Description |
|---|---|
| `ae_training_curves.png` | Courbes loss / RMSE / deep supervision |
| `ae_network_evaluation.png` | Reconstruction, sigma MC, LOO, zones lacunaires, bouées D-optimal |
| `ae_uncertainty_density.png` | Incertitude comparée Dense/Moyen/Clairsemé |

### Figures — GNN

| Fichier | Description |
|---|---|
| `gnn_network_analysis.png` | Contribution, unicité, corrélation, graphe, couverture, barplot |
| `gnn_network_analysis_rl_optimal.png` | Idem sur réseau RL (mode pipeline) |
| `gnn_network_analysis_rl_light.png` | Idem sur config légère (si `--eval_light`) |
| `gnn_inductive_eval.png` | Scores des capteurs hypothétiques (si `--inductive`) |

### Figures — RL

| Fichier | Description |
|---|---|
| `rl_training_curves.png` | Récompense, N actifs, info, reward cumulée |
| `rl_final_config.png` | Configuration optimale trouvée |
| `rl_two_configs.png` | Comparaison dense (N★) vs légère (N★/2) |
| `rl_progression.gif` | Animation de l'apprentissage RL |
| `rl_pareto_front.png` | Front info vs N + gain marginal (méthode pareto) |
| `rl_efficiency.png` | Info, eta(N), info vs coût log (méthode efficiency) |
| `rl_scalarized.png` | N par lambda, info vs N, eta par lambda (méthode scalarized) |

### Rapports

| Fichier | Description |
|---|---|
| `rapport_pipeline_YYYYMMDD.txt` | Rapport complet (mode pipeline) |
| `rapport_individual_YYYYMMDD.txt` | Rapport complet (mode individual) |
| `ae_loo_scores.json` | Scores LOO par capteur |
