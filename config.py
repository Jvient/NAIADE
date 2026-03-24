"""Configuration partagée OED-IA SNO Marins."""

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Domaine spatial
NX = 160
NY = 240
NT = 1000

# Dynamique océanique
DT = 1.0
U_MEAN = 0.05
V_MEAN = 0.02
NOISE_STD = 0.01

# Physique SST / SSS
SST_MEAN = 15.0
SST_STD = 5.0
SSS_MEAN = 35.0
SSS_STD = 0.8
TS_CORRELATION = 0.7

# Bouées
N_BUOYS = 30
OBS_NOISE_STD = 0.05


def set_global_seed(seed: int):
    """Fixe toutes les sources d'aléa pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_dir(base="outputs", seed_ocean=42, seed_buoys=7, mode="run"):
    """
    Crée un répertoire de sortie horodaté avec seeds pour archivage.
    Format : outputs/run_YYYYMMDD_HHMMSS_so42_sb7/
    Retourne le Path créé.
    """
    from datetime import datetime
    from pathlib import Path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{mode}_{ts}_so{seed_ocean}_sb{seed_buoys}"
    out = Path(base) / name
    out.mkdir(parents=True, exist_ok=True)
    return out
