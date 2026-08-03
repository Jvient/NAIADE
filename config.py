import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
#  Domaine
# =============================================================================
NX = 160                 # points zonaux   (axe 0, x = est)
NY = 240                 # points méridiens (axe 1, y = nord)
NT = 1000                # pas de temps enregistrés

DX_KM        = 5.0       # résolution -> domaine 800 x 1200 km (submésoéchelle)
LAT0         = 42.0      # latitude centrale (°N) -> f0, beta
DT_DAYS      = 1.0       # pas de sortie (1 jour)
N_SUBSTEPS   = 2         # sous-pas d'advection par pas de sortie (dt = 12 h)
SPINUP_DAYS  = 150       # spin-up jeté : laisse le temps aux filaments de se former

# =============================================================================
#  Circulation  (fonction de courant géostrophique)
# =============================================================================
U_GYRE        = 0.08     # m/s  — vitesse du double gyre de fond
U_JET         = 0.55     # m/s  — vitesse max du jet zonal
JET_WIDTH_KM  = 40.0     # km   — demi-largeur du jet
JET_LAT_FRAC  = 0.55     # position moyenne de l'axe du jet (fraction de Ly)

N_EDDIES       = 22      # tourbillons simultanés
EDDY_V_MAX     = 0.25    # m/s  — vitesse orbitale de référence
EDDY_R_KM      = (35.0, 80.0)     # rayon (km)
EDDY_LIFE_DAYS = (60.0, 180.0)    # durée de vie
RD_KM          = 25.0    # rayon de déformation de Rossby -> dérive beta

PERT_TAU_DAYS = 12.0     # temps de décorrélation de la perturbation non résolue
PERT_AMP      = 3.5e3    # m2/s — amplitude de cette perturbation sur psi

KAPPA = 25.0             # m2/s — diffusivité (contrôle l'échelle de dissipation)

# =============================================================================
#  Traceurs de surface
# =============================================================================
SST_MEAN         = 15.0  # °C  — température moyenne du domaine
SST_GRADIENT     = 9.0   # °C  — écart nord-sud imposé par la climatologie
SST_SEASONAL_AMP = 2.5   # °C  — demi-amplitude du cycle saisonnier
TAU_T_DAYS       = 40.0  # j   — rappel thermique (flux air-mer / couche mélangée)

SSS_MEAN      = 35.0     # psu
SSS_GRADIENT  = 1.30     # psu — maximum subtropical (évaporation)
SSS_PLUME_AMP = 0.75     # psu — panache dessalé, casse la dégénérescence T-S
TAU_S_DAYS    = 150.0    # j   — rappel halin : bien plus lent (pas de rétroaction)

TS_CORRELATION   = 0.7   # part de la clim. de S alignée sur celle de T (>0 :
                         # chaud & salé, compensation de densité subtropicale)
SEASON_PHASE_DAYS = 60.0 # décalage du cycle saisonnier (jour du minimum)

# Écarts-types indicatifs (diagnostic a posteriori, plus imposés a priori)
SST_STD = 3.0
SSS_STD = 0.20
NOISE_STD = 0.01

# =============================================================================
#  Réseau d'observation
# =============================================================================
N_BUOYS = 30

# Bruit instrumental, PAR VARIABLE et en unités physiques.
# Un seul OBS_NOISE_STD n'a plus de sens : sigma(SST) ~ 2.6 °C et
# sigma(SSS) ~ 0.18 psu, un bruit unique de 0.05 vaut 2 % du signal en T
# mais 28 % en S. Valeurs typiques d'une bouée/thermosalinographe.
OBS_NOISE_T = 0.05       # °C
OBS_NOISE_S = 0.02       # psu
OBS_NOISE_STD = OBS_NOISE_T   # alias hérité

# =============================================================================
#  Analyse OED — échelles issues du nature run
# =============================================================================
# Retirer la moyenne de domaine avant d'analyser les corrélations entre
# capteurs. Le cycle saisonnier est un mode global : sans cette étape,
# toutes les bouées corrèlent entre elles et la redondance est illisible.
DESEASON_ANALYSIS = True

# Rayon d'influence d'un capteur = échelle de décorrélation spatiale du
# nature run (diagnostiquée à ~90 km). Sert de noyau de couverture au RL.
INFLUENCE_RADIUS_KM = 90.0
RL_INFO_GAIN = 20.0      # met delta_info à l'échelle de la pénalité budget
GNN_CORR_THRESHOLD = 0.35  # seuil adapté aux anomalies mésoéchelle

# Régularisation de la covariance du critère d'information du RL.
# Avec tau ~ 12 j, un an de nature run ne contient que ~30 réalisations
# indépendantes de la mésoéchelle, pour 2n paramètres à estimer. La covariance
# empirique brute sur-apprend massivement (variance expliquée hors échantillon
# NÉGATIVE). On la contracte vers un modèle paramétrique sigma(x)·rho(d/L),
# comme le fait l'interpolation optimale opérationnelle.
# 0 = covariance empirique pure, 1 = modèle paramétrique pur.
EVF_SHRINKAGE = 0.9

# =============================================================================
#  Modèle de coût opérationnel  (2e objectif du front de Pareto)
# =============================================================================
# Le nombre de bouées est un proxy grossier du coût : deux réseaux de même
# taille n'ont pas le même coût si l'un est compact près du port et l'autre
# dispersé au large. C'est ce qui rend l'arbitrage information/coût réellement
# antagoniste — et donc le front non trivial.
PORT_XY_FRAC      = (0.04, 0.03)   # position du port (fraction du domaine)
COST_BUOY_FIXED   = 12.0           # k€/an et par bouée (amortissement, capteurs)
COST_SHIP_PER_KM  = 0.090          # k€/km de navire océanographique
N_CAMPAIGNS_YEAR  = 2              # campagnes de maintenance par an
CO2_SHIP_PER_KM   = 0.050          # tCO2/km

# =============================================================================
#  Contrainte de séparation minimale entre bouées
# =============================================================================
# Deux bouées ne peuvent pas occuper des cases adjacentes de la grille
# candidate : il faut au moins une case vide entre elles.
#   MIN_SEP_CELLS = 2  ->  distance de Tchebychev >= 2 cases
#                          (interdit aussi les diagonales)
#   MIN_SEP_DIAGONAL = False -> ne contraint que les 4 voisins (distance de
#                          Manhattan), les diagonales redeviennent permises.
# Conséquence : le nombre maximal de bouées est plafonné à
#   ceil(grid_x / MIN_SEP_CELLS) * ceil(grid_y / MIN_SEP_CELLS)
MIN_SEP_CELLS    = 2
MIN_SEP_DIAGONAL = True

# Équivalent en distance physique, pour les réseaux tirés directement en
# pixels (briques AE et GNN, réseau initial de run_demo).
MIN_BUOY_SEP_KM  = 50.0

# --- alias hérités (ne sont plus utilisés par le générateur) ---
DT = DT_DAYS
U_MEAN = U_GYRE
V_MEAN = 0.5 * U_GYRE
