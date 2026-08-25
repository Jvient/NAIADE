try:                                    # torch is only needed by the NN bricks;
    import torch                        # the nature run and the OED core are
    HAS_TORCH = True                   # pure numpy and must stay importable
except ModuleNotFoundError:             # without it (CI, headless OED runs).
    torch = None
    HAS_TORCH = False

DEVICE = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

# =============================================================================
#  Domain
# =============================================================================
NX = 160                 # zonal grid points   (axis 0, x = east)
NY = 240                 # meridional points   (axis 1, y = north)
NT = 1000                # recorded time steps

DX_KM        = 5.0       # resolution -> 800 x 1200 km domain (submesoscale-permitting)
LAT0         = 42.0      # central latitude (deg N) -> f0, beta
DT_DAYS      = 1.0       # output time step (1 day)
N_SUBSTEPS   = 2         # advection substeps per output step (dt = 12 h)
SPINUP_DAYS  = 150       # discarded spin-up: lets filaments develop before t = 0

# =============================================================================
#  Circulation  (geostrophic streamfunction)
# =============================================================================
U_GYRE        = 0.08     # m/s  -- background double-gyre velocity
U_JET         = 0.55     # m/s  -- peak zonal jet velocity
JET_WIDTH_KM  = 40.0     # km   -- jet half-width
JET_LAT_FRAC  = 0.55     # mean jet axis position (fraction of Ly)

N_EDDIES       = 22      # simultaneous eddies
EDDY_V_MAX     = 0.25    # m/s  -- reference orbital velocity
EDDY_R_KM      = (35.0, 75.0)     # radius (km)
EDDY_LIFE_DAYS = (60.0, 180.0)    # lifetime
RD_KM          = 25.0    # Rossby deformation radius -> beta drift

PERT_TAU_DAYS = 12.0     # decorrelation time of the unresolved perturbation
PERT_AMP      = 3.5e3    # m2/s -- amplitude of that perturbation on psi

KAPPA = 25.0             # m2/s -- diffusivity (sets the dissipation scale)

# =============================================================================
#  Surface tracers
# =============================================================================
SST_MEAN         = 15.0  # degC -- domain mean temperature
SST_GRADIENT     = 9.0   # degC -- north-south contrast imposed by climatology
SST_SEASONAL_AMP = 2.5   # degC -- seasonal cycle half-amplitude
TAU_T_DAYS       = 40.0  # days -- thermal restoring (air-sea flux / mixed layer)

SSS_MEAN      = 35.0     # psu
SSS_GRADIENT  = 1.30     # psu -- subtropical maximum (evaporation)
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
N_BUOYS = 15

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

# Sensor influence radius = spatial decorrelation scale of the nature run
# (diagnosed at ~90 km). Used as the coverage kernel by the RL brick.
INFLUENCE_RADIUS_KM = 90.0
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
MIN_BUOY_SEP_KM  = 50.0

# --- legacy aliases (no longer used by the generator) ---
DT = DT_DAYS
U_MEAN = U_GYRE
V_MEAN = 0.5 * U_GYRE
