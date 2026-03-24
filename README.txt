================================================================================
  OED-IA SNO Marins — README
  Optimal Experimental Design par Intelligence Artificielle
  Application aux réseaux d'observation océanographiques
================================================================================


TABLE DES MATIÈRES
──────────────────
  1. Structure du projet
  2. Dépendances
  3. Lancement rapide
  4. Orchestrateur (run_demo.py)
  5. Brique 1 — Autoencoder (01_autoencoder.py)
  6. Brique 2 — GNN (02_gnn.py)
  7. Brique 3 — RL (03_rl.py)
  8. Données synthétiques (dataset.py)
  9. Reproductibilité (seeds)
  10. Fichiers produits


================================================================================
  1. STRUCTURE DU PROJET
================================================================================

  config.py              Configuration partagée (domaine, physique, seeds)
  dataset.py             Générateur d'océan synthétique + Dataset PyTorch
  01_autoencoder.py      Brique 1 — AE-UNet MC-Dropout (reconstruction + incertitude)
  02_gnn.py              Brique 2 — GAT + GraphSAGE (structure réseau + inductif)
  03_rl.py               Brique 3 — PPO (optimisation placement capteurs)
  run_demo.py            Orchestrateur (individual | pipeline)
  outputs/               Répertoire de sortie (créé automatiquement)
                         Sous-dossiers archivés par run :
                         outputs/pipeline_20260319_143022_so42_sb7/


================================================================================
  2. DÉPENDANCES
================================================================================

  pip install torch numpy scipy matplotlib

  Optionnel (pour les GATConv/SAGEConv natifs, sinon fallback manuel) :
  pip install torch-geometric


================================================================================
  3. LANCEMENT RAPIDE
================================================================================

  # Pipeline complet (RL → GNN → AE) avec paramètres démo :
  python run_demo.py --mode pipeline

  # Pipeline + évaluation config légère (N★/2) par GNN + AE :
  python run_demo.py --mode pipeline --eval_light

  # Briques indépendantes :
  python run_demo.py --mode individual

  # Run réel — individual (≈30-60 min selon GPU) :
  python run_demo.py --mode individual --seed_ocean 42 --seed_buoys 7 --nt 800 --n_buoys 30 --ae_epochs 80 --ae_base_ch 32 --gnn_epochs 200 --rl_steps 50000 --rl_grid_x 16 --rl_grid_y 24 --rl_n_min 10 --rl_n_max 40

  # Run réel — pipeline avec comparaison dense/légère (≈25-50 min selon GPU) :
  python run_demo.py --mode pipeline --eval_light --seed_ocean 42 --seed_buoys 7 --nt 800 --n_buoys 30 --ae_epochs 60 --ae_base_ch 32 --gnn_epochs 200 --rl_steps 50000 --rl_grid_x 16 --rl_grid_y 24 --rl_n_min 10 --rl_n_max 40

  # Briques standalone (entraînement complet) :
  python 01_autoencoder.py --train --figures
  python 02_gnn.py --train --analyze --inductive
  python 03_rl.py --train --evaluate --rl_method pareto
  python 03_rl.py --train --evaluate --rl_method efficiency
  python 03_rl.py --train --evaluate --rl_method scalarized

  # Générer uniquement la figure du nature run :
  python dataset.py --nt 500 --seed 42


================================================================================
  4. ORCHESTRATEUR — run_demo.py
================================================================================

  Deux modes d'exécution. Les trois briques partagent le même nature run
  et le même réseau initial de bouées (reproductibilité via seeds).

  Mode individual : AE → GNN → RL, chacun sur le réseau initial.
  Mode pipeline   : RL optimise le réseau → GNN l'analyse → AE l'évalue.

  USAGE
  ─────
  python run_demo.py --mode pipeline --seed_ocean 42 --seed_buoys 7
  python run_demo.py --mode individual --nt 300 --ae_epochs 10

  ARGUMENTS
  ─────────
  --mode MODE            Mode d'exécution.
                         individual : chaque brique tourne indépendamment
                                      sur le réseau initial de bouées.
                         pipeline   : RL → GNN → AE, le réseau optimisé par
                                      le RL est transmis aux deux autres.
                         Défaut : individual

  --seed_ocean SEED      Seed du nature run (océan synthétique).
                         Contrôle les champs SST/SSS générés.
                         Défaut : 42

  --seed_buoys SEED      Seed du réseau initial de bouées.
                         Contrôle les positions aléatoires des capteurs.
                         Défaut : 7

  --nt NT                Nombre de pas de temps du nature run.
                         Plus grand = océan plus riche mais plus lent.
                         Défaut : 200

  --n_buoys N            Nombre de bouées du réseau initial.
                         Si omis, utilise N_BUOYS du config.py (30).
                         Défaut : None (→ 30)

  --eval_light           Active l'évaluation de la config légère (N★/2)
                         par GNN + AE en plus de la config dense.
                         Uniquement en mode pipeline. Produit des figures
                         supplémentaires suffixées "_rl_light".
                         Défaut : désactivé

  --ae_epochs N          Époques d'entraînement de l'AE (mode démo).
                         5 est rapide pour tester, 50-100 pour de vrais résultats.
                         Défaut : 5

  --ae_base_ch N         Nombre de canaux de base de l'AE.
                         16 pour la démo, 32 pour l'entraînement complet.
                         Défaut : 16

  --gnn_epochs N         Époques d'entraînement du GAT et GraphSAGE.
                         Défaut : 30

  --rl_steps N           Nombre de steps PPO.
                         2000 pour la démo, 50000+ pour convergence réelle.
                         Défaut : 2000

  --rl_grid_x GX         Résolution X de la grille candidate RL.
                         GX × GY = nombre de positions candidates.
                         Défaut : 8

  --rl_grid_y GY         Résolution Y de la grille candidate RL.
                         Défaut : 12

  --rl_n_min N           Nombre minimum de bouées actives autorisé.
                         Pénalité budget si en dessous.
                         Défaut : 5

  --rl_n_max N           Nombre maximum de bouées actives autorisé.
                         Pénalité budget si au-dessus.
                         Défaut : 20

  --rl_episode_len N     Actions par épisode RL.
                         Défaut : 20

  --rl_method METHOD     Méthode de sélection du N★ optimal.
                         pareto     : front de Pareto info vs N, coude Kneedle
                         efficiency : η = info/(1+log N), N★ = argmax(η)
                         scalarized : 4 PPO avec λ croissant, best par η
                         Défaut : pareto

  --gif_frames N         Nombre de frames du GIF de progression RL.
                         Défaut : 40

  --output_dir DIR       Répertoire de sortie pour tous les fichiers.
                         Défaut : outputs


================================================================================
  5. BRIQUE 1 — AUTOENCODER (01_autoencoder.py)
================================================================================

  AE-UNet déterministe avec incertitude par MC-Dropout.
  Reconstruit les champs SST/SSS à partir d'observations partielles.
  Produit : cartes d'incertitude, scores LOO, proposition de bouées.

  USAGE
  ─────
  # Entraînement complet :
  python 01_autoencoder.py --train --epochs 100

  # Entraînement + figures diagnostiques :
  python 01_autoencoder.py --train --figures

  # Figures seules (depuis un checkpoint existant) :
  python 01_autoencoder.py --figures --checkpoint outputs/ae_best.pt

  # Scoring Leave-One-Out :
  python 01_autoencoder.py --score --checkpoint outputs/ae_best.pt

  # Tout à la fois :
  python 01_autoencoder.py --train --figures --score

  ARGUMENTS
  ─────────
  --train                Lance l'entraînement de l'AE.

  --score                Calcule les scores Leave-One-Out par capteur.
                         Nécessite un checkpoint.

  --figures              Génère les deux figures diagnostiques :
                         - ae_network_evaluation.png (reconstruction, σ MC,
                           zones lacunaires, LOO, 3 bouées proposées)
                         - ae_uncertainty_density.png (incertitude vs densité)
                         Nécessite un checkpoint.

  --seed_ocean SEED      Seed du nature run.
                         Défaut : 42

  --seed_buoys SEED      Seed des positions de bouées (pour figures/score).
                         Défaut : 7

  --checkpoint PATH      Chemin du checkpoint à charger pour --score et --figures.
                         Défaut : outputs/ae_best.pt

  --output_dir DIR       Répertoire de sortie.
                         Défaut : outputs

  --epochs N             Nombre d'époques d'entraînement.
                         Défaut : 100

  --batch_size N         Taille de batch.
                         Défaut : 16

  --lr LR                Learning rate (AdamW).
                         Défaut : 3e-4

  --base_ch N            Canaux de base de l'UNet. Les niveaux suivants sont
                         2×, 4×, 8×, 16× ce nombre.
                         16 = léger (~200K params), 32 = standard (~800K params).
                         Défaut : 32

  --latent_ch N          Dimension du bottleneck latent.
                         Défaut : 64

  --cond_dim N           Dimension de l'embedding FiLM (conditionnement N_obs).
                         Défaut : 32

  --dropout_p P          Probabilité de MC-Dropout (actif aussi à l'inférence).
                         Plus haut = plus de régularisation + incertitude plus large.
                         Défaut : 0.1

  --w_unobs W            Poids de la loss sur les pixels NON observés.
                         >1 force le modèle à mieux interpoler les lacunes.
                         Défaut : 4.0

  --lambda_grad L        Poids de la loss de gradient spatial.
                         Pénalise les discontinuités dans la reconstruction.
                         Défaut : 0.5

  --huber_delta D        Seuil δ de la loss Huber.
                         Plus petit = plus robuste aux outliers, moins sensible.
                         Défaut : 0.5

  --n_obs_min N          Nombre minimum d'observations par échantillon (training).
                         Le masque stochastique tire entre n_obs_min et n_obs_max.
                         Défaut : 10

  --n_obs_max N          Nombre maximum d'observations par échantillon.
                         Défaut : 80

  --n_mc_val N           Passes MC-Dropout pour la validation.
                         Plus haut = RMSE val plus stable mais plus lent.
                         Défaut : 15

  --n_mc N               Passes MC-Dropout pour les figures (incertitude).
                         60-80 donne des cartes σ lisses.
                         Défaut : 60


================================================================================
  6. BRIQUE 2 — GNN (02_gnn.py)
================================================================================

  Graph Attention Network pour analyser la structure du réseau de capteurs.
  Détecte les redondances, identifie les lacunes, évalue de nouveaux capteurs
  de manière inductive via GraphSAGE.

  USAGE
  ─────
  # Entraînement GAT + GraphSAGE + analyse complète :
  python 02_gnn.py --train --analyze

  # Avec évaluation inductive de capteurs hypothétiques :
  python 02_gnn.py --train --analyze --inductive

  # Positions personnalisées pour l'évaluation inductive :
  python 02_gnn.py --train --inductive --new_positions "[(50,100),(120,200)]"

  # Analyse seule (depuis checkpoints existants) :
  python 02_gnn.py --analyze

  # Évaluation inductive seule :
  python 02_gnn.py --inductive --new_positions "[(30,60),(90,180),(140,50)]"

  ARGUMENTS
  ─────────
  --train                Entraîne le GAT et le GraphSAGE.
                         Produit gnn_best.pt et sage_best.pt.

  --analyze              Analyse complète du réseau :
                         - Scores de contribution par capteur
                         - Matrice de corrélation et d'attention
                         - Détection de redondance (unicité Q25)
                         - Couverture spatiale
                         Produit gnn_network_analysis.png.

  --inductive            Évalue des capteurs hypothétiques (gliders, Argo)
                         non vus à l'entraînement via GraphSAGE.
                         Charge sage_best.pt ou entraîne si absent.
                         Produit gnn_inductive_eval.png.

  --seed_ocean SEED      Seed du nature run.
                         Défaut : 42

  --seed_buoys SEED      Seed des positions de bouées.
                         Défaut : 7

  --new_positions STR    Positions (x,y) des capteurs hypothétiques pour
                         l'évaluation inductive. Format Python list de tuples.
                         Défaut : "[(10,20),(80,150),(130,40)]"

  --corr_threshold TH    Seuil de corrélation pour créer une arête.
                         |ρ| > seuil → arête entre deux capteurs.
                         Plus bas = graphe plus connecté.
                         Défaut : 0.5

  --k_nearest K          Nombre de plus proches voisins géographiques.
                         Garantit la connexité du graphe.
                         Défaut : 4

  --gnn_epochs N         Époques d'entraînement du GAT et du GraphSAGE.
                         Défaut : 200

  --output_dir DIR       Répertoire de sortie.
                         Défaut : outputs

  --n_buoys N            Nombre de bouées du réseau.
                         Défaut : 30 (config.N_BUOYS)


================================================================================
  7. BRIQUE 3 — RL (03_rl.py)
================================================================================

  Reinforcement Learning (PPO) pour optimiser le placement des capteurs
  sur une grille candidate. Produit un front de Pareto information vs
  nombre de capteurs et compare config dense vs légère.

  USAGE
  ─────
  # Entraînement PPO + évaluation Pareto (par défaut) :
  python 03_rl.py --train --evaluate

  # Entraînement + méthode efficacité :
  python 03_rl.py --train --evaluate --rl_method efficiency

  # Entraînement + méthode scalarisée (sweep λ) :
  python 03_rl.py --train --evaluate --rl_method scalarized

  # Évaluation seule (depuis checkpoint) :
  python 03_rl.py --evaluate --rl_method efficiency

  # GIF de progression seul :
  python 03_rl.py --gif

  # Tout :
  python 03_rl.py --train --evaluate --gif --rl_method pareto

  ARGUMENTS
  ─────────
  --train                Lance l'entraînement PPO.
                         Produit rl_best.pt, rl_training_curves.png,
                         rl_final_config.png, rl_progression.gif.

  --evaluate             Évalue le réseau avec la méthode choisie (--rl_method).
                         Produit la figure spécifique à la méthode +
                         rl_two_configs.png (dense vs légère).

  --gif                  Génère le GIF animé de progression RL.
                         Produit rl_progression.gif.

  --rl_method METHOD     Méthode de sélection du N★ optimal :

                         pareto (défaut)
                           Sweep info vs N avec politique + random.
                           Kneedle pour le coude. Front de Pareto.
                           Figure : rl_pareto_front.png

                         efficiency
                           η(N) = info(N) / (1 + log(N))
                           N★ = argmax(η). Score unique, pas de sweep.
                           Le log(N) pénalise doucement les grands réseaux.
                           Figure : rl_efficiency.png (3 panneaux)

                         scalarized
                           Entraîne 4 PPO avec λ ∈ {0.001, 0.005, 0.01, 0.02}
                           comme coût marginal par ajout de capteur.
                           Chaque λ → un compromis N/info différent.
                           Le meilleur est choisi par η.
                           Figure : rl_scalarized.png (3 panneaux)
                         Défaut : pareto

  --seed_ocean SEED      Seed du nature run.
                         Défaut : 42

  --seed_buoys SEED      Seed (réservé pour usage futur).
                         Défaut : 7

  --checkpoint PATH      Chemin du checkpoint RL.
                         Défaut : outputs/rl_best.pt

  --output_dir DIR       Répertoire de sortie.
                         Défaut : outputs

  --rl_steps N           Nombre total de steps d'interaction avec l'environnement.
                         La politique est mise à jour tous les buffer_size steps.
                         2000 pour test, 50000-100000 pour convergence.
                         Défaut : 50000

  --buffer_size N        Taille du rollout buffer PPO.
                         Nombre de transitions collectées avant chaque mise à jour.
                         Défaut : 512

  --lr LR                Learning rate (Adam).
                         Défaut : 3e-4

  --grid_x GX            Résolution X de la grille candidate.
                         L'espace d'action a K = grid_x × grid_y positions.
                         Défaut : 16

  --grid_y GY            Résolution Y de la grille candidate.
                         Défaut : 24

  --n_min N              Nombre minimum de bouées actives.
                         En dessous, pénalité de budget proportionnelle.
                         Défaut : 10

  --n_max N              Nombre maximum de bouées actives.
                         Au-dessus, pénalité de budget proportionnelle.
                         Défaut : 40

  --episode_len N        Nombre d'actions par épisode.
                         L'agent fait N toggles puis l'épisode se termine.
                         Défaut : 20

  --w_info W             Poids de la récompense d'information.
                         Défaut : 1.0

  --w_budget W           Poids de la pénalité de budget.
                         Plus haut = l'agent respecte mieux [n_min, n_max].
                         Défaut : 0.5

  --gif_frames N         Nombre de frames du GIF de progression.
                         Défaut : 80


================================================================================
  8. DONNÉES SYNTHÉTIQUES — dataset.py
================================================================================

  Générateur de nature run 2D+T avec structures physiques réalistes :
  double gyre, front zonal à méandres, tourbillons méso-échelle,
  cycle saisonnier, turbulence spectrale k⁻³, SSS couplée à SST.

  USAGE
  ─────
  # Générer la figure diagnostique du nature run :
  python dataset.py --nt 500 --seed 42

  ARGUMENTS
  ─────────
  --nt NT                Nombre de pas de temps.
                         Défaut : 1000 (config.NT)

  --seed SEED            Seed numpy pour la reproductibilité.
                         Défaut : 42

  --out PATH             Chemin de la figure de sortie.
                         Défaut : outputs/ocean_nature_run.png


================================================================================
  9. REPRODUCTIBILITÉ (SEEDS)
================================================================================

  La fonction set_global_seed(seed) dans config.py fixe :
    - numpy (np.random.seed)
    - PyTorch CPU (torch.manual_seed)
    - PyTorch GPU (torch.cuda.manual_seed_all)
    - cuDNN déterministe (torch.backends.cudnn.deterministic = True)

  Deux seeds contrôlent l'ensemble du pipeline :

    seed_ocean    Contrôle le nature run (champs SST/SSS).
                  Même seed → mêmes données océaniques.

    seed_buoys    Contrôle les positions initiales des bouées.
                  Même seed → même réseau de départ.

  Pour reproduire exactement un run :
    python run_demo.py --mode pipeline --seed_ocean 42 --seed_buoys 7

  La seed globale est fixée au démarrage de chaque brique et de
  l'orchestrateur. Tous les tirages aléatoires (initialisations réseau,
  DataLoader, masques, splits train/test, resets RL) sont reproductibles.


================================================================================
  10. FICHIERS PRODUITS
================================================================================

  Les sorties sont archivées dans des sous-dossiers horodatés :
    outputs/pipeline_20260319_143022_so42_sb7/
    outputs/individual_20260319_150100_so42_sb7/

  Chaque run crée son propre répertoire, identifiable par mode + date + seeds.

  CHECKPOINTS
    ae_best.pt                    Meilleur modèle AE (RMSE val minimale)
    gnn_best.pt                   Meilleur modèle GAT
    sage_best.pt                  Meilleur modèle GraphSAGE
    rl_best.pt                    Meilleure politique PPO + active_mask

  FIGURES — BRIQUE 1 (AE)
    ae_training_curves.png        Courbes loss / RMSE / deep supervision
    ae_network_evaluation.png     Reconstruction, σ MC, LOO, zones lacunaires,
                                  bouées D-optimal (★) — minimisation trace(Σ)
    ae_uncertainty_density.png    Incertitude comparée Dense/Moyen/Clairsemé

  FIGURES — BRIQUE 2 (GNN)
    gnn_network_analysis.png      6 panneaux : contribution, unicité (attention GAT),
                                  corrélation, graphe réseau, couverture, barplot
    gnn_network_analysis_rl_optimal.png   Idem sur réseau RL (mode pipeline)
    gnn_network_analysis_rl_light.png     Idem sur config légère (si --eval_light)
    gnn_inductive_eval.png        Scores des capteurs hypothétiques (si --inductive)

  FIGURES — BRIQUE 3 (RL)
    rl_training_curves.png        Courbes récompense, N actifs, info, reward cumulée
    rl_final_config.png           Configuration optimale trouvée
    rl_two_configs.png            Comparaison dense (N★) vs légère (N★/2)
    rl_progression.gif            Animation de l'apprentissage RL
    rl_pareto_front.png           [pareto] Front info vs N + gain marginal
    rl_efficiency.png             [efficiency] Info, η(N), info vs coût log
    rl_scalarized.png             [scalarized] N par λ, info vs N, η par λ
    rl_training_curves_lam=*.png  [scalarized] Courbes pour chaque λ

  RAPPORTS
    rapport_pipeline_YYYYMMDD.txt    Rapport complet (mode pipeline)
    rapport_individual_YYYYMMDD.txt  Rapport complet (mode individual)
    ae_loo_scores.json               Scores LOO par capteur (si --score)

================================================================================
