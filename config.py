import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
#  Domain  --  préréglages auto-cohérents
# =============================================================================
# Agrandir le domaine ne se réduit pas à augmenter NX/NY : toutes les échelles
# de longueur doivent suivre, sinon on obtient un océan sous-résolu (tourbillons
# de 3 pixels) ou une soupe de tourbillons minuscules dans un bassin immense.
# Chaque préréglage définit (nx, ny, dx) et un facteur d'échelle s = dx / 5 km
# appliqué a la largeur du jet, aux rayons de tourbillon, au rayon de
# deformation et au rayon d'influence des capteurs. Le nombre de tourbillons
# suit le nombre de mailles, ce qui conserve leur densite surfacique.
#
#   demo   :  800 x 1200 km  -- vitrine, ~35 s de nature run
#   large  : 2560 x 3840 km  -- echelle sous-bassin, ~4x le calcul
#   basin  : 4608 x 6144 km  -- echelle bassin (comparable a la boite PIRATA)
#
# Selection : editer DOMAIN ci-dessous, ou exporter NAIADE_DOMAIN=large.
# ATTENTION : changer de domaine invalide les checkpoints AE (taille d'entree)
# et les statistiques diagnostiquees (L_decorr, sigma).
DOMAIN_PRESETS = {
    "demo":  dict(nx=160, ny=240, dx_km=5.0),
    "large": dict(nx=320, ny=480, dx_km=8.0),
    "basin": dict(nx=384, ny=512, dx_km=12.0),
}
DOMAIN = os.environ.get("NAIADE_DOMAIN", "demo")
if DOMAIN not in DOMAIN_PRESETS:
    raise ValueError(f"NAIADE_DOMAIN={DOMAIN!r} inconnu. "
                     f"Choix : {list(DOMAIN_PRESETS)}")
_D = DOMAIN_PRESETS[DOMAIN]

# Surcharges par variable d'environnement : indispensables pour les donnees
# reelles, dont la grille est imposee par le fichier. Elles doivent etre
# posees AVANT le premier import de config (le projet utilise
# `from config import *` partout, une affectation apres coup ne se propagerait
# pas). `run_real.py` s'en charge en lisant l'entete NetCDF d'abord.
NX = int(os.environ.get("NAIADE_NX", _D["nx"]))
NY = int(os.environ.get("NAIADE_NY", _D["ny"]))
NT = 1000                # recorded time steps

DX_KM        = float(os.environ.get("NAIADE_DX_KM", _D["dx_km"]))
_S           = DX_KM / 5.0     # facteur d'echelle des longueurs physiques
_CELLS       = (NX * NY) / (160 * 240)

LAT0         = 42.0      # central latitude (deg N) -> f0, beta
DT_DAYS      = 1.0       # output time step (1 day)
N_SUBSTEPS   = 2         # advection substeps per output step (dt = 12 h)
SPINUP_DAYS  = 150       # discarded spin-up: lets filaments develop before t = 0

# =============================================================================
#  Circulation  (geostrophic streamfunction)
# =============================================================================
U_GYRE        = 0.08     # m/s  -- background double-gyre velocity
U_JET         = 0.55     # m/s  -- peak zonal jet velocity
JET_WIDTH_KM  = 40.0 * _S     # km   -- jet half-width
JET_LAT_FRAC  = 0.55     # mean jet axis position (fraction of Ly)

N_EDDIES       = int(round(22 * _CELLS))   # densite surfacique conservee
EDDY_V_MAX     = 0.25    # m/s  -- reference orbital velocity
EDDY_R_KM      = (35.0 * _S, 75.0 * _S)     # radius (km)
EDDY_LIFE_DAYS = (60.0, 180.0)    # lifetime
RD_KM          = 25.0 * _S    # Rossby deformation radius -> beta drift

PERT_TAU_DAYS = 12.0     # decorrelation time of the unresolved perturbation
PERT_AMP      = 3.5e3    # m2/s -- amplitude of that perturbation on psi

KAPPA = 25.0             # m2/s -- diffusivity (sets the dissipation scale)

# =============================================================================
#  Surface tracers
# =============================================================================
# Contraste climatologique nord-sud. Il croit avec l'extension meridienne du
# domaine, mais en racine : garder le gradient par kilometre constant donnerait
# 29 degC sur un bassin, garder le contraste total donnerait un ocean plat.
# La racine maintient le rapport anomalie mesoscale / gradient dans une plage
# realiste. A rediagnostiquer si le domaine change (generate_full().diagnostics).
_GRAD_SCALE      = ((NY * DX_KM) / 1200.0) ** 0.5
SST_MEAN         = 15.0  # degC -- domain mean temperature
SST_GRADIENT     = 9.0 * _GRAD_SCALE   # degC -- north-south contrast
SST_SEASONAL_AMP = 2.5   # degC -- seasonal cycle half-amplitude
TAU_T_DAYS       = 40.0  # days -- thermal restoring (air-sea flux / mixed layer)

SSS_MEAN      = 35.0     # psu
SSS_GRADIENT  = 1.30 * _GRAD_SCALE   # psu -- subtropical maximum
SSS_PLUME_AMP = 0.75     # psu -- fresh plume, breaks the T-S degeneracy

TAU_S_DAYS    = 150.0    # days -- haline restoring: much slower, no feedback

TS_CORRELATION   = 0.7   # share of the S climatology aligned with that of T
                         # (> 0: warm & salty, subtropical density compensation)
SEASON_PHASE_DAYS = 60.0 # seasonal cycle offset (day of the minimum)

# Indicative standard deviations (diagnosed a posteriori, no longer imposed)
SST_STD = 3.0
SSS_STD = 0.20
NOISE_STD = 0.01

# =============================================================================
#  Observing network
# =============================================================================
N_BUOYS = int(round(30 * _CELLS ** 0.5))   # densite lineaire conservee

# Instrumental noise, PER VARIABLE and in physical units.
# A single OBS_NOISE_STD no longer makes sense: sigma(SST) ~ 2.6 degC and
# sigma(SSS) ~ 0.18 psu, so a common value of 0.05 is 2% of the signal in T
# but 28% in S. Values are typical of a buoy / thermosalinograph.
OBS_NOISE_T = 0.05       # degC
OBS_NOISE_S = 0.02       # psu
OBS_NOISE_STD = OBS_NOISE_T   # legacy alias

# =============================================================================
#  OED analysis -- scales derived from the nature run
# =============================================================================
# Remove the domain mean before analysing inter-sensor correlations. The
# seasonal cycle is a global mode: without this step every buoy correlates
# with every other one and redundancy becomes unreadable.
DESEASON_ANALYSIS = True

# Sous-echantillonnage de la grille d'evaluation du critere EVF. Il croit avec
# le domaine pour garder un nombre de cellules d'evaluation a peu pres constant
# (donc un cout de calcul du critere a peu pres constant).
EVAL_STRIDE = int(round(8 * _CELLS ** 0.5))

# Rayon d'influence d'un capteur = longueur de decorrelation spatiale du nature
# run. Valeur de REPLI seulement : elle est mise a l'echelle du domaine, ce qui
# n'est PAS fiable. Sur le prereglage "demo" la valeur configuree (90 km) colle
# au diagnostic (95 km) ; sur "large" elle donne 144 km alors que le run en
# diagnostique 424, parce que la longueur de decorrelation ne suit pas la taille
# des tourbillons des lors que le contraste climatologique change lui aussi.
#
# Un rayon mal specifie rend le modele de covariance parametrique de l'EVF
# faux, donc le classement des configurations ininterpretable. Utiliser
# --influence_auto (ou diagnose_influence_km) pour le lire sur le run plutot
# que sur cette constante ; un avertissement est emis si l'ecart depasse 50 %.
INFLUENCE_RADIUS_KM = 90.0 * _S
INFLUENCE_AUTO_TOL  = 0.5   # ecart relatif au-dela duquel on alerte
RL_INFO_GAIN = 20.0      # scales delta_info against the budget penalty
GNN_CORR_THRESHOLD = 0.35  # threshold suited to mesoscale anomalies

# Covariance regularisation for the RL information criterion.
# With tau ~ 12 days, one year of nature run holds only ~30 independent
# mesoscale realisations against 2n parameters to estimate. The raw sample
# covariance overfits badly (explained variance goes NEGATIVE out of sample).
# It is shrunk towards a parametric model sigma(x) * rho(d/L), exactly as
# operational optimal interpolation does.
# 0 = pure sample covariance, 1 = pure parametric model.
EVF_SHRINKAGE = 0.9

# =============================================================================
#  Operating cost model  (2nd Pareto objective)
# =============================================================================
# Buoy count is a crude proxy for cost: two networks of equal size do not cost
# the same if one is compact near the port and the other spread offshore. That
# is what makes the information/cost trade-off genuinely antagonistic.
PORT_XY_FRAC      = (0.04, 0.03)   # port position (domain fraction)
COST_BUOY_FIXED   = 12.0           # k€/year per buoy (amortisation, sensors)
COST_SHIP_PER_KM  = 0.090          # k€/km of research vessel
N_CAMPAIGNS_YEAR  = 2              # maintenance campaigns per year
CO2_SHIP_PER_KM   = 0.050          # tCO2/km

# =============================================================================
#  Maintien en condition opérationnelle  (modèle explicite, cf. maintenance.py)
# =============================================================================
# Le bloc ci-dessus reste le proxy historique. Le modèle de maintien, lui,
# planifie de vraies campagnes sous contrainte de budget et rétroagit sur
# l'information via la DISPONIBILITÉ des bouées :
#
#   budget -> campagnes finançables -> intervalle de visite -> disponibilité
#          -> variance d'erreur effective -> variance expliquée par le réseau
#
# Profils disponibles : "regional" (défaut, cohérent avec le domaine
# synthétique) et "pirata" (mouillages hauturiers, navire hauturier,
# MTBF dégradé par les déprédations). Voir MAINT_PROFILES dans maintenance.py.
MAINT_PROFILE     = "regional"
MAINT_ENABLED     = False          # activé par --maintenance en ligne de commande

# Budget annuel de référence pour normaliser le ratio information / coût.
# Le ratio publié est l'EVF pour 100 k€/an ; COST_REF sert d'échelle interne
# pour que la récompense reste d'ordre 1.
COST_REF_KEUR     = 600.0

# Niveaux de budget balayés par la démo de campagne (k€/an).
# None = calibration automatique sur le budget minimum viable du reseau
# (capex + une campagne annuelle complete), cf. campaign.auto_budget_levels.
# Des valeurs en dur ne sont transposables qu'a taille de reseau constante.
BUDGET_LEVELS_KEUR = None

# Gain appliqué à la variation du ratio info/coût en mode de récompense
# "ratio" (l'équivalent de RL_INFO_GAIN pour ce mode).
RL_RATIO_GAIN     = 5.0

# =============================================================================
#  Minimum separation constraint between buoys
# =============================================================================
# Two buoys may not occupy adjacent cells of the candidate grid: at least one
# empty cell must separate them.
#   MIN_SEP_CELLS = 2  ->  Chebyshev distance >= 2 cells (diagonals forbidden)
#   MIN_SEP_DIAGONAL = False -> only the 4 direct neighbours are constrained
#                          (Manhattan distance), diagonals become legal again.
# Consequence: the maximum buoy count is capped at
#   ceil(grid_x / MIN_SEP_CELLS) * ceil(grid_y / MIN_SEP_CELLS)
MIN_SEP_CELLS    = 2
MIN_SEP_DIAGONAL = True

# Physical-distance equivalent, for networks drawn directly in pixels
# (AE and GNN bricks, initial network of run_demo).
MIN_BUOY_SEP_KM  = 50.0 * _S

# --- legacy aliases (no longer used by the generator) ---
DT = DT_DAYS
U_MEAN = U_GYRE
V_MEAN = 0.5 * U_GYRE
