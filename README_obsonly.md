# NAIADE — extension obs-only

Quatre modules additifs. Ils ne modifient **aucun** fichier existant : `01_autoencoder.py`
est importé et réutilisé tel quel (`ObservabilityAE` est instancié avec `in_ch=4, out_ch=4`).

| fichier | rôle |
|---|---|
| `obs_operator.py` | Brique 0 — prélèvement typé par plateforme dans un nature run |
| `obsonly.py` | Brique 1b — AE hétéroscédastique, NLL aux capteurs tenus à l'écart |
| `gnn_lobo.py` | Brique 2b — GNN de nœud masqué sur le graphe d'observations (option A) |
| `validate_obsonly.py` | l'expérience décisive : corrélation de rang contre la vérité |

## Enchaînement

```bash
# 0. prélever les observations dans le nature run (le run n'est plus relu ensuite)
python - <<'EOF'
import numpy as np
from dataset import SyntheticOceanGenerator           # ou dataset_glorys
from obs_operator import ObsNetwork
gen = SyntheticOceanGenerator(); T, S = gen.generate_dataset(nt=800)
T = (T - T.mean()) / T.std(); S = (S - S.mean()) / S.std()
net = ObsNetwork(nx=T.shape[1], ny=T.shape[2], nt=len(T),
                 rng=np.random.default_rng(7))
net.add_moorings(n=20).add_argo(n=15).add_drifters(n=10, u=0.15, v=0.5)
net.add_glider(waypoints=[(20,30),(120,200)], n_repeat=6)
net.sample(T, S).save("outputs/obs_synth.npz")
np.savez("outputs/_truth.npz", T=T, S=S)      # scellé jusqu'à la validation
EOF

# 1. AE obs-only
python obsonly.py --train --obs outputs/obs_synth.npz --epochs 80
python obsonly.py --lobo  --obs outputs/obs_synth.npz --ckpt outputs/ae_obsonly.pt

# 2. GNN obs-only (option A)
python gnn_lobo.py --train --obs outputs/obs_synth.npz --epochs 200
python gnn_lobo.py --lobo  --obs outputs/obs_synth.npz --ckpt outputs/gnn_lobo.pt

# 3. l'expérience décisive
python validate_obsonly.py
```

## Ce qui a changé dans la supervision

```
AVANT (01_autoencoder.py, AELoss)
    L = w_obs·Huber(pred, VÉRITÉ)|obs + 4.0·Huber(pred, VÉRITÉ)|non-obs
        + 0.5·Huber(grad pred, grad VÉRITÉ)
    -> 80 % du gradient vient du nature run

APRÈS (obsonly.py, HeldOutNLL)
    L = NLL(mu, sigma² ; OBSERVATION)|capteurs tenus à l'écart
        + lambda_tv·TV(mu)
    -> le nature run n'apparaît plus dans la boucle
```

Trois points non négociables :

1. **Sortie hétéroscédastique.** L'utilité obs-only est une variance prédictive :
   elle doit être *calibrée*. MC-Dropout donne une dispersion, pas une variance
   calibrée. La NLL sur held-out donne les deux.
2. **Masquage par groupe, pas par pixel.** Un glider entier est retiré d'un coup ;
   sinon les points voisins du même transect fuient dans l'entrée.
3. **`w_in = 0` par défaut.** Mettre la NLL aussi sur les points d'entrée
   encourage la copie identité.

## Diagnostics à surveiller

`z_std` (écart-type du résidu standardisé) doit valoir 1.0, `coverage_95` doit
valoir 0.95, l'histogramme PIT doit être plat à 0.1. Tous se calculent sur
observations seules — donc ils resteront disponibles sur un SNO réel.

Comportement attendu et normal : `z_std` dérive au-dessus de 1 en cours
d'entraînement. Les capteurs de validation ne sont **jamais** en entrée, ils sont
donc plus difficiles que les held-out d'entraînement — le modèle devient
sur-confiant sur eux. C'est pour ça que le checkpoint est sélectionné sur le
**CRPS**, pas sur la RMSE.

## Les deux scores du GNN ne disent pas la même chose

| | `skill` élevé | `skill` bas |
|---|---|---|
| **`delta` élevé** | pivot d'un groupe redondant | irremplaçable |
| **`delta` bas** | redondant → candidat au retrait | isolé → lacune du réseau autour |

La case bas-bas est celle qu'on rate toujours avec un score unique : un capteur
isolé est à la fois mal prédit *et* peu utile aux autres. Ce n'est pas un capteur
précieux, c'est le signe qu'il manque des capteurs autour de lui.

## Nouveautés du graphe par rapport à `02_gnn.py`

- **Corrélation à lag optimal** au lieu du lag 0. L'information est advectée ;
  deux capteurs alignés sur le courant sont liés à un décalage temporel, invisible
  au lag 0. Sur le test à `U_MEAN`-like, 302 arêtes sur 458 ressortent à lag non nul,
  lag médian 6 jours.
- **Arêtes dirigées** amont → aval, le lag devient attribut d'arête, et le message
  de *j* vers *i* lit la valeur de *j* à `t − lag_ij`.
- **Features nodales obs-only** : variance de la propre série du capteur par bande,
  temps de décorrélation, type de plateforme, taux de retour, période — plus aucune
  lecture de voisinage 5×5 du nature run.
- **Cible non circulaire** : la mesure réelle du nœud masqué, pas une fonction de
  la matrice de corrélation qui a servi à bâtir le graphe.

## Réalisme du prélèvement

`obs_operator.py` applique trois choses qui rendent le test honnête :

- **erreur de représentativité** — lecture décalée de ±3 jours (Gasparin et al. 2023, §2.2) ;
- **erreur instrumentale** par variable (`OBS_NOISE_T`, `OBS_NOISE_S`) ;
- **manquants non aléatoires** — hasard de panne modulé par la variance locale
  (`hazard_var_amp`), avec maintenance périodique. Les vraies bouées meurent
  davantage là où c'est énergétique ; un estimateur qui ne survit pas à ça ne
  survivra pas à PIRATA.

Sans ces trois éléments le problème redevient trop facile, exactement comme le
twin identique de la version actuelle.

## Limites connues

- `GraphBatcher` applique à chaque nœud le lag **médian** de ses arêtes sortantes,
  alors qu'un nœud peut être source d'arêtes de lags différents. Approximation
  acceptable en option A ; exacte en option B (gather par arête).
- La référence OI de `validate_obsonly.py` n'est définie que pour les capteurs
  **fixes** — les plateformes mobiles changent de position, leur contribution
  marginale n'a pas de définition statique. C'est une limite du protocole de
  validation, pas de l'estimateur.
- `dataset.py` (générateur d'océan synthétique) est absent de la branche `glo12` :
  `git checkout main -- dataset.py`.

---

## Démarrer sur une branche partie de `main`

```bash
git checkout main && git checkout -b obs-ops
# copier les 5 modules de l'extension à la racine, puis :
python preflight_obsonly.py        # inspecte, ne modifie rien
```

`preflight_obsonly.py` vérifie fichiers, dépendances, symboles de `config.py`
et — surtout — instancie réellement `ObservabilityAE(in_ch=4, out_ch=4)` pour
confirmer que la greffe 4 canaux tient sur l'architecture de la branche cible.

### Trois écarts `main` / `glo12` qui touchent l'extension

| | `main` | `glo12` | effet |
|---|---|---|---|
| `OBS_NOISE_T` / `_S` | absents | présents | repli sur `OBS_NOISE_STD` (T) et `0.4·OBS_NOISE_STD` (S) — **arbitraire, à recaler** |
| `DX_KM`, `INFLUENCE_RADIUS_KM` | absents | présents | replis 5 km/px et 90 km → `L = 18 px`, valeur calibrée POUR main |
| masque océan | aucun | `GlorysData.ocean` | laisser `ocean=None` dans `ObsNetwork` |

L'import de `config` dans `obs_operator.py` se fait **nom par nom**, pas en bloc :
un `from config import (NX, NY, OBS_NOISE_T, ...)` global échouerait entièrement
dès qu'un seul nom manque, et les vrais `NX`/`NY` seraient silencieusement
remplacés par les valeurs par défaut.

### Piège des dériveurs

Ne pas passer `config.U_MEAN` / `V_MEAN` à `add_drifters()` : ces constantes sont
dans les unités internes du générateur, pas en pixels par pas de temps. L'erreur
est silencieuse — les dériveurs partent simplement au mauvais endroit à la
mauvaise vitesse, et la structure du graphe s'en trouve faussée.

```python
from obs_operator import estimate_advection
u, v, corr = estimate_advection(T)      # mesuré sur le champ, sans ambiguïté
net.add_drifters(n=10, u=u, v=v)
```

---

## Divergence d'API entre branches

`naiade_compat.py` résout à l'exécution les symboles qui ont divergé entre
`main` et `glo12`, au lieu d'imposer un renommage sur la branche cible.

| symbole | résolution |
|---|---|
| classe de l'autoencodeur | liste de noms connus, puis heuristique sur la signature (`in_ch` + `out_ch`), puis `--ae_class <Nom>` |
| fonction de seed | `set_global_seed`, `set_seed`, `seed_everything`, `setup_seed`… puis repli interne (numpy + torch + cudnn) |
| `DEVICE` | `config.DEVICE`, sinon déduit de `torch.cuda.is_available()` |

Aucun de ces trois éléments ne justifie de bloquer : un nom de fonction de seed
n'est pas une contrainte d'architecture. Seule la **signature** de la classe AE
en est une — il lui faut `in_ch` / `out_ch` paramétrables pour passer à 4 canaux.

Si la résolution automatique échoue, le message d'erreur liste les classes
réellement présentes dans le module :

```bash
python obsonly.py --train --ae_class ObservabilityAutoencoder ...
```

`preflight_obsonly.py` affiche systématiquement les classes `nn.Module` trouvées
dans `01_autoencoder.py`, même quand tout va bien : c'est ce qui permet de
diagnostiquer un renommage en un coup d'œil.

---

## Cas concret : `main` / branche `obs-ops`

Sur `main` la classe de l'autoencodeur s'appelle **`ObservabilityVAE`**. C'est un
nom historique : les v1–v3 étaient de vrais VAE, la v4 a retiré la
reparamétrisation (le `forward` renvoie `logvar = torch.zeros_like(z)` par pure
compatibilité d'API) mais la classe n'a jamais été renommée. Elle est résolue
automatiquement.

L'heuristique de repli ne suffisait pas ici : `ResDoubleConv`, `Down` et
`FiLMUp` ont eux aussi `in_ch`/`out_ch`, ce qui donnait quatre candidates et
donc aucune décision. Le critère exige maintenant en plus un paramètre de
capacité (`latent_ch` ou `base_ch`) **et** les méthodes `encode`/`decode`.

### MC-Dropout toujours actif

`MCDropout2d.forward` force `training=True`, donc `model.eval()` ne désactive
pas le dropout — c'est voulu, c'est le principe de MC-Dropout. Conséquence pour
l'extension : une passe unique n'est qu'un tirage de dropout parmi d'autres, et
`n_mc_val=1` mesurait le bruit de tirage plutôt que la calibration du modèle.
Le défaut passe à 8.

### Ce qui n'est PAS touché

`VAELoss` reste en place et n'est pas utilisée par l'extension : `obsonly.py`
a sa propre `HeldOutNLL`. Le `forward` à quatre valeurs de retour
(`pred, z, logvar_nul, aux`) n'est jamais appelé non plus — la greffe passe par
`encode()` et `decode()` directement, donc la différence d'arité est sans effet.

### `set_global_seed`

Absente de `config.py` sur `main`. `train()` y sème explicitement via
`generate_dataset(seed=args.seed_ocean)`, ce qui suffit pour la Brique 1 mais
pas pour l'extension (masquage par capteur, splits). Le repli de
`naiade_compat` sème numpy, torch et cudnn.
