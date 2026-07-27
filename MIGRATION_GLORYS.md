# NAIADE — GLORYS12, golfe de Gascogne : migration terminée

Le pipeline complet (`run_demo.py --mode pipeline` et `--mode individual`)
tourne sur GLORYS12 exactement comme sur l'océan synthétique.

**Configuration** — fenêtre 100 % océanique, 4 variables en surface,
grille 48 × 48.

```
GLORYS_LON_RANGE = (-9.75, -5.50)     # boîte « B carrée »
GLORYS_LAT_RANGE = (44.25, 48.25)
GLORYS_DEPTHS    = (0,)               # ~0.494 m
GLORYS_GRID_MULT = 16                 # 52×49 bruts → 48×48
OBSERVED_VARS    = ("thetao", "so")   # VAE : in_ch=3, out_ch=4
```

---

## 1. Mise en route

```bash
pip install xarray netcdf4          # + dask si >500 journaliers
mkdir -p data/raw/glorys_gascogne data/cache

# Vérifier la fenêtre sur les fichiers réels
python -m data.glorys --probe data/raw/glorys_gascogne \
       --lon -9.75 -5.50 --lat 44.25 48.25 --depths 0 \
       --require_full_sea --cache data/cache --plot outputs/quicklook.png

# Reporter NX / NY affichés dans config.py, puis
python run_demo.py --mode pipeline
```

Bascule ponctuelle vers le synthétique, sans éditer `config.py` :
```bash
python run_demo.py --mode pipeline --data_source synthetic
```

Si `--require_full_sea` échoue (cellules peu profondes du plateau armoricain
masquées par GLORYS), chercher une fenêtre sur les données réelles :
```bash
python -m data.glorys --find-box data/raw/glorys_gascogne --plot outputs/boxes.png
```

---

## 2. Fichiers

| Fichier | Statut |
|---|---|
| `data/glorys.py` | **nouveau** — lecture NetCDF, masque terre, fenêtre sans terre, rognage grille |
| `data/loader.py` | **nouveau** — `load_ocean(args)`, point d'entrée unique |
| `data/dataset.py` | **remplacé** — dataset multi-canaux, normalisation par canal |
| `config.py` | **remplacé** |
| `01_autoencoder.py` | patché |
| `02_gnn.py` | patché |
| `03_rl.py` | patché |
| `run_demo.py` | patché |

---

## 3. Ce qui a changé dans les briques

**Brique 1 — AE.** `in_ch=3` (thetao, so, masque) → `out_ch=4` (thetao, so,
uo, vo). Le déséquilibre est le cœur du dispositif : le réseau observe T/S et
doit reconstruire les courants. Loss et RMSE pondérés par le masque mer.
RMSE reportée **par canal en unités physiques** — une RMSE agrégée sur des
°C, des PSU et des m/s n'a aucune interprétation.

**Brique 2 — GNN.** La corrélation inter-stations utilise les 4 canaux et non
plus `0.6·T + 0.4·S`. Les courants portent une information de connectivité
dynamique que T/S seuls n'ont pas, et c'est exactement ce dont le GNN a besoin
pour juger la redondance.

**Brique 3 — RL.** Environnement multi-canaux, candidats filtrés en mer
(`env.K` devient dynamique), variance calculée sur les canaux **normalisés** —
sans quoi la variance en °C écrase celle des courants et le RL n'optimise que
pour la température.

**Orchestrateur.** Rapport avec provenance GLORYS complète (période, fenêtre,
résolution, niveaux, canaux) et positions de bouées **en longitude/latitude**
en plus des pixels.

---

## 4. Quatre bugs pré-existants corrigés au passage

Ils n'étaient pas dus à la migration, mais la grille 48 × 48 et la série
courte les ont rendus visibles. Ils auraient produit des chiffres faux ou des
plantages sur les vraies données.

1. **Balayage Pareto hors budget.** `range(n_min-5, n_max+10)` explorait des
   configurations que le MDP pénalise explicitement ; le point de coude
   pouvait tomber sous `n_min`, donnant une recommandation inapplicable.
   Ramené à `[n_min, n_max]`.

2. **Étiquettes « Dense » / « Légère » inversées.** Les deux configurations
   sont re-simulées par la politique, qui modifie librement le nombre de
   bouées : un run affichait « Dense N=4 » et « Légère N=6 ». Les étiquettes
   sont désormais échangées si l'ordre s'inverse, avec un avertissement.

3. **Grille de couverture GNN.** `NX // 16 + 1` créait une ligne fantôme quand
   NX est multiple de 16 : des zones lacunaires étaient signalées au pixel 56
   sur un domaine de 48. Résolution rendue adaptative au domaine.

4. **Instants de figures codés en dur** (`t=50`, `t=150`) : plantage dès que
   la série compte moins de 151 dates — quasi certain avec quelques mois de
   GLORYS. Dérivés de la longueur réelle.

---

## 5. Points de vigilance

**Rapport signal/bruit sur la salinité.** Le dataset alerte si le bruit
capteur approche l'écart-type du signal. Après désaisonnalisation, l'anomalie
de SSS peut tomber à quelques centièmes de PSU, soit l'ordre de grandeur de
`OBS_NOISE["so"] = 0.02`. Si l'alerte se déclenche sur les vraies données :
soit revoir le bruit, soit reconnaître que la salinité n'est pas observable
dans ce cadre. Les deux sont présentables, mais il faut le savoir avant la
visio.

**Cycle saisonnier retiré par défaut.** Sans cela, la récompense RL pondérée
par la variance placerait les bouées là où l'amplitude saisonnière est
maximale — un signal qui se prédit sans observation. Prévoir une slide de
sensibilité avec/sans : la question sera posée.

**Split chronologique.** Deux dates GLORYS voisines sont très corrélées ; le
découpage est sans mélange. Si les chiffres paraissent trop beaux, c'est le
premier point à vérifier.

---

## 6. Reste à faire avant la présentation

- [ ] Lancer sur les vrais fichiers CMEMS et confirmer **100.00 % mer**.
- [ ] Axes des figures en **degrés lon/lat** (encore en indices pixel).
- [ ] Titres de figures : « SST » alors que le champ est une **anomalie**.
- [ ] **Baseline de comparaison** — placement aléatoire en mer + placement par
      variance/EOF. Sans elle, impossible de démontrer l'apport du RL, et
      c'est l'objection la plus probable de la salle.
- [ ] Résultat à mettre en avant : **un réseau T/S contraint-il les
      courants ?** C'est ce que mesure la RMSE sur `uo_z0` / `vo_z0`.

---

## 7. Brique 4 — baselines (`04_baselines.py`)

```bash
python 04_baselines.py --checkpoint outputs/vae_best.pt \
       --rl_checkpoint outputs/rl_best.pt \
       --grid_x 16 --grid_y 16 --n_sensors 5 10 20 30 40
```

⚠ Passer les **mêmes** `--grid_x` / `--grid_y` qu'à la brique 3, sinon les
positions candidates ne peuvent pas être reconstruites (erreur explicite).

### Métrique
La comparaison utilise la **RMSE de reconstruction de l'autoencodeur sur les
pixels non observés**, et non la récompense du MDP. Évaluer avec la récompense
ferait gagner le RL par construction : c'est la fonction qu'il a maximisée.

### Méthodes
| | Description |
|---|---|
| `random` | Tirage uniforme en mer, moyenne ± σ sur `n_repeat` tirages |
| `variance` | Pixels de plus forte variance temporelle, avec espacement minimal |
| `eof_qr` | Pivots QR sur base EOF — Manohar, Brunton, Kutz & Brunton (2018) |
| `coverage` | Farthest-point sampling, purement géométrique |
| `rl` | Réseau de la brique 3, rechargé depuis `rl_best.pt` |

### Barres d'erreur — indispensable
Les méthodes déterministes sont évaluées `n_eval_repeat` fois : les positions
ne changent pas, mais MC-Dropout et le bruit d'observation, si. Sans cette
dispersion on conclut sur des écarts qui ne sortent pas du bruit.

**Démonstration concrète** : sur un run de validation, la première version
annonçait « le RL dépasse eof_qr de 1.4 % ». Avec les répétitions, eof_qr
s'est révélée **meilleure de 2.4 %** — le gain était entièrement du bruit
d'évaluation. C'est exactement le type de chiffre à ne pas mettre en slide.

### Résultats de validation (données synthétiques 4 ans, AE partiellement entraîné)

RMSE par canal, N = 21 capteurs :

| méthode | thetao (°C) | so (PSU) | uo (m/s) | vo (m/s) |
|---|---|---|---|---|
| random | 0.2018 | 0.0300 | 0.0029 | 0.0045 |
| variance | 0.6731 | 0.1011 | 0.0078 | 0.0106 |
| **eof_qr** | **0.1779** | **0.0260** | **0.0026** | 0.0041 |
| coverage | 0.1961 | 0.0288 | 0.0028 | 0.0045 |
| rl | 0.1790 | 0.0261 | 0.0026 | 0.0043 |

Trois enseignements :

1. **`variance` est catastrophique** (3× pire que tout le reste). Concentrer
   les capteurs là où le champ varie le plus produit un réseau redondant qui
   n'observe qu'une seule structure. C'est un contre-exemple pédagogique
   utile — l'intuition naïve est franchement mauvaise.
2. **`eof_qr` et le RL sont au coude à coude**, tous deux nettement devant
   l'aléatoire. Attendu : QR sur base EOF est optimal au sens de la
   reconstruction linéaire, exactement ce que mesure la métrique.
3. **`coverage` domine à grand N** (N=40). La couverture géométrique suffit
   quand les capteurs sont nombreux ; c'est à budget serré que le placement
   informé compte.

⚠ Ces chiffres viennent de données synthétiques et d'un AE partiellement
entraîné. Ils valident la **mécanique**, pas la performance de NAIADE.

### Si eof_qr reste devant sur les vraies données
C'est un résultat défendable, à condition de l'assumer. QR/EOF optimise la
reconstruction linéaire d'un champ **stationnaire**, sans contrainte de budget
ni de coût carbone, et ne fournit ni incertitude ni analyse de redondance.
Le RL optimise sous contraintes opérationnelles. Positionner NAIADE comme un
outil de **pré-screening multi-contraintes**, pas comme un concurrent frontal
de QR sur son propre terrain — et le dire avant qu'on le demande.
