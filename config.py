"""
NAIADE  configuration globale.

Configuration courante : GLORYS12, golfe de Gascogne, fenêtre 100 % océanique,
4 variables en SURFACE = 4 canaux.

 Les briques font `from config import *`, donc NX/NY/N_CHANNELS sont liés
   À L'IMPORT. Les modifier après import n'a aucun effet : c'est ici, et
   seulement ici, qu'on change de domaine.

Vérification de la configuration :
    python -m data.glorys --probe data/raw/glorys_gascogne \
        --lon -9.75 -5.50 --lat 44.25 48.25 --depths 0 --require_full_sea
"""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 
#  Source de données : "synthetic" | "glorys"
# 
DATA_SOURCE = "glorys"

#  GLORYS12 (CMEMS GLOBAL_MULTIYEAR_PHY_001_030) 
GLORYS_DIR       = "/mnt/data/jmv/glory/expoed"
GLORYS_CACHE     = "data/cache"

GLORYS_VARIABLES = ("thetao", "so", "uo", "vo")
GLORYS_DEPTHS    = (0,)            # surface seule (~0.494 m)

# Fenêtre océanique  golfe de Gascogne, boîte  B carrée .
#   Contraintes côtières qui la déterminent :
#     Estaca de Bares   43.79°N    plancher latitude
#     Ouessant           5.14°W    plafond longitude à haute latitude
#     Galice             9.30°W    sans effet ici (fenêtre au nord de 44°N)
#   Marge minimale au trait de côte : ~30 km.
#
#   Variantes si besoin :
#     A large           lon (-9.75, -2.75)  lat (44.25, 47.00)   85  34
#     C marge maximale  lon (-9.50, -4.50)  lat (44.50, 47.00)   61  31
#   A est très allongée : 34 px en latitude ne survivent pas aux quatre
#   sous-échantillonnages du VAE (3417842). Préférer B.
GLORYS_LON_RANGE = (-9.75, -5.50)
GLORYS_LAT_RANGE = (44.25, 48.25)

GLORYS_COARSEN   = 1               # 2  1/6°, utile pour prototyper vite
GLORYS_GRID_MULT = 16              # rogner NX/NY à un multiple de 16
GLORYS_SEASONAL  = True            # retirer le cycle saisonnier (cf. note)
GLORYS_FULL_SEA  = True            # échouer si la fenêtre contient de la terre

# Variables réellement mesurées par les capteurs.
#   Une bouée de surface mesure T et S ; les courants exigent un ADCP, bien
#   plus coûteux. Les canaux non observés restent des CIBLES à reconstruire :
#   c'est tout l'intérêt de garder uo/vo. La question devient  un réseau T/S
#   suffit-il à contraindre le champ de courant de surface ? , nettement plus
#   parlant qu'une simple reconstruction de SST.
#   Mettre OBSERVED_VARS = GLORYS_VARIABLES pour supposer des mouillages
#   complets (T/S + ADCP).
OBSERVED_VARS = ("thetao", "so")

# NOTE  GLORYS_SEASONAL
#   Le cycle annuel de SST domine largement la variance sur la Gascogne. Or la
#   récompense RL est pondérée par la variance locale : sans désaisonnalisation,
#   l'optimiseur place les bouées là où l'amplitude saisonnière est maximale,
#   c'est-à-dire sur le plateau. Résultat convaincant visuellement mais vide :
#   un cycle annuel se prédit sans observation. Le retirer recentre l'étude sur
#   la variabilité méso-échelle, qui est le vrai enjeu d'observabilité.

# NOTE  GLORYS_DEPTHS = (0,)
#   Surface seule. Les niveaux 0 (0.494 m) et 1 (1.541 m) sont distants d'un
#   mètre, donc toujours dans la couche de mélange : leur corrélation dépasse
#   0.99 et le second ne ferait que doubler le coût de calcul.
#   Pour une dimension verticale réellement informative, prendre plus tard
#   (0, k) avec k autour de 2050 m :  un réseau de surface contraint-il la
#   subsurface ?  est une vraie question d'observabilité, contrairement à
#    z = 0.5 m contraint-il z = 1.5 m ? .

# 
#  Domaine
# 
if DATA_SOURCE == "glorys":
    # Boîte B : 52  49 px bruts à 1/12°, rognés à 48  48 par GLORYS_GRID_MULT.
    # x  longitude, y  latitude
    NX = 48
    NY = 48
    NT = 1000                       # borne haute, tronquée au nb de fichiers

    N_CHANNELS = len(GLORYS_VARIABLES) * len(GLORYS_DEPTHS)   # = 4 (cibles)
    N_OBS_CH   = len(OBSERVED_VARS)    * len(GLORYS_DEPTHS)   # = 2 (entrées)

    # Le VAE ingère les canaux OBSERVÉS + le masque, et reconstruit TOUS les
    # canaux. in_ch  out_ch : c'est voulu, pas une erreur.
    #   entrées : thetao_z0, so_z0, masque          3
    #   sorties : thetao_z0, so_z0, uo_z0, vo_z0    4
    VAE_IN_CH  = N_OBS_CH + 1       # = 3
    VAE_OUT_CH = N_CHANNELS         # = 4

else:
    NX, NY, NT = 160, 240, 1000
    N_CHANNELS = N_OBS_CH = 2
    VAE_IN_CH, VAE_OUT_CH = 3, 2

    SST_MEAN, SST_STD = 15.0, 5.0
    SSS_MEAN, SSS_STD = 35.0, 0.8

# 
#  Dynamique (générateur synthétique uniquement)
# 
DT, U_MEAN, V_MEAN = 1.0, 0.05, 0.02
NOISE_STD = 0.01
TS_CORRELATION = 0.7

# 
#  Réseau d'observation
# 
N_BUOYS = 30

# Bruit d'observation en unités PHYSIQUES, par variable.
OBS_NOISE = {
    "thetao": 0.01,     # °C    capteur de bouée
    "so":     0.02,     # PSU
    "uo":     0.02,     # m/s   ADCP
    "vo":     0.02,     # m/s
}
OBS_NOISE_STD = 0.02    # legacy, mode synthétique

# Distance minimale entre capteurs (pixels). À 1/12°, 1 px  7 km.
# Attention : sur une grille 4848, MIN_BUOY_DIST=4 avec N_BUOYS=30 est déjà
# contraignant. Réduire à 3 si l'échantillonnage échoue.
MIN_BUOY_DIST = 3

