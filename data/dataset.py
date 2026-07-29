"""
NAIADE — Dataset PyTorch multi-canaux.

Généralise l'ancien couple (T, S) à un tenseur `fields` (nt, n_ch, nx, ny)
regroupant plusieurs variables et plusieurs niveaux verticaux.

Points clés
-----------
1. NORMALISATION PAR CANAL — obligatoire dès qu'on mélange des °C (O(10)),
   des PSU (O(35)) et des m/s (O(0.1)). Une normalisation globale écraserait
   totalement le signal des courants.

2. MASQUE D'OBSERVATION PAR VARIABLE — une bouée de surface mesure T et S,
   mais pas nécessairement les courants (il faut un ADCP). `observed_vars`
   permet de déclarer ce que le réseau observe réellement ; les canaux non
   observés restent des cibles à reconstruire, jamais des entrées.

3. MASQUE TERRE/MER — conservé pour rester compatible avec un domaine côtier,
   même si la configuration courante est 100 % océanique.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from config import *


# =============================================================================
#  Dataset multi-canaux
# =============================================================================

class OceanFieldDataset(Dataset):
    """
    Paramètres
    ----------
    fields : (nt, n_ch, nx, ny) float32
    channels : list[str]
        Noms des canaux, ex. ['thetao_z0', 'thetao_z1', 'so_z0', ...].
    observed_vars : tuple[str] | None
        Variables réellement mesurées par les capteurs. None = toutes.
        Ex. ("thetao", "so") pour une bouée T/S sans courantomètre.
    n_obs_min, n_obs_max : int
        Bornes du nombre de points d'observation tirés à chaque échantillon.
    noise_std : dict[str, float] | float
        Bruit d'observation en unités PHYSIQUES, par variable.
    sea_mask : (nx, ny) bool | None
    stats : dict | None
        Statistiques (mean, std) par canal, à réutiliser depuis le split
        d'entraînement. Ne JAMAIS recalculer sur la validation.

    Retour de __getitem__
    ---------------------
    x    : (n_obs_ch + 1, nx, ny) — canaux observés masqués, puis le masque
    y    : (n_ch, nx, ny)         — tous les canaux (cibles)
    mask : (1, nx, ny)            — masque d'observation
    sea  : (1, nx, ny)            — masque océanique
    """

    def __init__(self, fields, channels,
                 observed_vars=None,
                 n_obs_min=5, n_obs_max=60,
                 noise_std=None,
                 sea_mask=None,
                 stats=None,
                 normalize=True,
                 warn_snr=True):
        self.fields = np.ascontiguousarray(fields, dtype=np.float32)
        self.channels = list(channels)
        self.n_ch = self.fields.shape[1]
        self.nx, self.ny = self.fields.shape[2], self.fields.shape[3]
        self.n_obs_min, self.n_obs_max = n_obs_min, n_obs_max

        if len(self.channels) != self.n_ch:
            raise ValueError(f"{len(self.channels)} noms pour {self.n_ch} canaux.")

        # ── Canaux observés ──────────────────────────────────────────────────
        self.observed_vars = tuple(observed_vars) if observed_vars else None
        if self.observed_vars is None:
            self.obs_idx = list(range(self.n_ch))
        else:
            self.obs_idx = [i for i, c in enumerate(self.channels)
                            if c.rsplit("_z", 1)[0] in self.observed_vars]
            if not self.obs_idx:
                raise ValueError(f"observed_vars={self.observed_vars} ne "
                                 f"correspond à aucun canal de {self.channels}")
        self.n_obs_ch = len(self.obs_idx)

        # ── Masque terre / mer ───────────────────────────────────────────────
        if sea_mask is None:
            self.sea_mask = np.ones((self.nx, self.ny), dtype=bool)
        else:
            self.sea_mask = np.asarray(sea_mask, dtype=bool)
            if self.sea_mask.shape != (self.nx, self.ny):
                raise ValueError(f"sea_mask {self.sea_mask.shape} ≠ "
                                 f"champ ({self.nx}, {self.ny})")
        self.has_land = not self.sea_mask.all()
        self._sea_flat = np.where(self.sea_mask.ravel())[0]
        self._sea_f32 = self.sea_mask.astype(np.float32)

        if len(self._sea_flat) < self.n_obs_max:
            raise ValueError(f"n_obs_max={self.n_obs_max} > "
                             f"{len(self._sea_flat)} pixels disponibles.")

        # ── Normalisation PAR CANAL ──────────────────────────────────────────
        if stats is not None:
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.asarray(stats["std"], dtype=np.float32)
        elif normalize:
            sm = self.sea_mask
            self.mean = np.array([self.fields[:, c][:, sm].mean()
                                  for c in range(self.n_ch)], dtype=np.float32)
            self.std = np.array([self.fields[:, c][:, sm].std()
                                 for c in range(self.n_ch)], dtype=np.float32)
        else:
            self.mean = np.zeros(self.n_ch, dtype=np.float32)
            self.std = np.ones(self.n_ch, dtype=np.float32)

        degenerate = [self.channels[c] for c in range(self.n_ch)
                      if self.std[c] < 1e-10]
        if degenerate:
            raise ValueError(f"Canaux de variance nulle : {degenerate}. "
                             f"Les retirer de GLORYS_VARIABLES.")

        self.fields = ((self.fields - self.mean[None, :, None, None])
                       / self.std[None, :, None, None])

        # ── Bruit d'observation, converti en unités normalisées ──────────────
        if noise_std is None:
            noise_std = OBS_NOISE
        if np.isscalar(noise_std):
            phys = np.full(self.n_ch, float(noise_std), dtype=np.float32)
        else:
            phys = np.array([float(noise_std.get(c.rsplit("_z", 1)[0], 0.0))
                             for c in self.channels], dtype=np.float32)
        self.noise_norm = (phys / self.std).astype(np.float32)

        # ── Diagnostic rapport signal / bruit ────────────────────────────────
        # Après désaisonnalisation, l'écart-type d'un canal chute fortement
        # (l'anomalie est bien plus faible que le champ brut) alors que le
        # bruit capteur, lui, ne change pas. Un ratio proche de 1 signifie
        # que le capteur ne distingue plus le signal : toute conclusion
        # d'observabilité sur ce canal serait dominée par le bruit.
        self.snr = {self.channels[i]: float(self.std[i] / max(phys[i], 1e-12))
                    for i in self.obs_idx}
        self.low_snr = [c for c, r in self.snr.items() if r < 3.0]
        if self.low_snr and warn_snr:
            print(f"  ⚠ [dataset] rapport signal/bruit faible sur {self.low_snr}")
            for c in self.low_snr:
                print(f"      {c:<12} SNR = {self.snr[c]:.2f}  "
                      f"(σ_signal={self.std[self.channels.index(c)]:.4f}, "
                      f"σ_bruit={phys[self.channels.index(c)]:.4f})")
            print("      → soit le bruit capteur est surestimé, soit la")
            print("        désaisonnalisation a retiré l'essentiel du signal.")

    # ── Statistiques à propager au split de validation ───────────────────────
    def get_stats(self) -> dict:
        return {"mean": self.mean.copy(), "std": self.std.copy(),
                "channels": list(self.channels)}

    def denormalize(self, arr, channel=None):
        """Repasse en unités physiques. `arr` : (..., n_ch, nx, ny) ou (nx, ny)."""
        if channel is not None:
            i = self.channels.index(channel) if isinstance(channel, str) else channel
            return arr * self.std[i] + self.mean[i]
        return arr * self.std[None, :, None, None] + self.mean[None, :, None, None]

    # ── Protocole Dataset ────────────────────────────────────────────────────
    def __len__(self):
        return len(self.fields)

    def _random_mask(self, n_obs):
        flat = np.zeros(self.nx * self.ny, dtype=np.float32)
        flat[np.random.choice(self._sea_flat, n_obs, replace=False)] = 1.0
        return flat.reshape(self.nx, self.ny)

    def __getitem__(self, t):
        n_obs = np.random.randint(self.n_obs_min, self.n_obs_max + 1)
        mask = self._random_mask(n_obs)
        y = self.fields[t]                                   # (n_ch, nx, ny)

        obs = y[self.obs_idx]                                # canaux mesurés
        noise = (np.random.randn(*obs.shape).astype(np.float32)
                 * self.noise_norm[self.obs_idx][:, None, None])
        x = np.concatenate([(obs + noise) * mask[None], mask[None]], axis=0)

        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(mask[None]),
                torch.from_numpy(self._sea_f32[None]))


def build_datasets(fields, channels, split=0.8, sea_mask=None, **kwargs):
    """
    Découpe temporelle train/val.

    Le découpage est CHRONOLOGIQUE et sans mélange : deux dates GLORYS voisines
    sont fortement corrélées, un split aléatoire ferait fuiter l'information et
    surestimerait nettement la performance.

    Les statistiques de normalisation sont calculées sur le train puis
    IMPOSÉES au split de validation.
    """
    n_tr = int(len(fields) * split)
    train = OceanFieldDataset(fields[:n_tr], channels, sea_mask=sea_mask, **kwargs)
    val = OceanFieldDataset(fields[n_tr:], channels, sea_mask=sea_mask,
                            stats=train.get_stats(), **kwargs)
    return train, val


# =============================================================================
#  Échantillonneur de capteurs
# =============================================================================

class BuoySampler:
    """Positions de capteurs, en mer et avec espacement minimal optionnel."""

    def __init__(self, nx, ny, n_buoys=N_BUOYS, sea_mask=None,
                 min_dist=0, rng=None):
        self.nx, self.ny = nx, ny
        self.n_buoys = n_buoys
        self.sea_mask = (np.ones((nx, ny), bool) if sea_mask is None
                         else np.asarray(sea_mask, dtype=bool))
        self.min_dist = min_dist
        self.rng = np.random.default_rng(rng)
        self.positions = self._random_positions()

    def _random_positions(self):
        idx = np.argwhere(self.sea_mask)
        if len(idx) < self.n_buoys:
            raise ValueError(f"{len(idx)} pixels mer pour {self.n_buoys} bouées.")
        if self.min_dist <= 0:
            sel = self.rng.choice(len(idx), self.n_buoys, replace=False)
            return [(int(idx[i, 0]), int(idx[i, 1])) for i in sel]

        chosen = []
        for i in self.rng.permutation(len(idx)):
            p = idx[i]
            if all((p[0]-q[0])**2 + (p[1]-q[1])**2 >= self.min_dist**2
                   for q in chosen):
                chosen.append(p)
                if len(chosen) == self.n_buoys:
                    break
        if len(chosen) < self.n_buoys:
            raise ValueError(f"min_dist={self.min_dist} trop grand : "
                             f"{len(chosen)}/{self.n_buoys} positions trouvées.")
        return [(int(p[0]), int(p[1])) for p in chosen]

    def set_positions(self, positions):
        self.positions = list(positions)
        self.n_buoys = len(self.positions)

    def build_mask(self):
        m = np.zeros((self.nx, self.ny), dtype=np.float32)
        for (i, j) in self.positions:
            m[i, j] = 1.0
        return m


# =============================================================================
#  Métriques masquées
# =============================================================================

def masked_rmse(pred, target, weight):
    """
    RMSE pondérée par `weight` (broadcastable).

        masked_rmse(y_hat, y, sea * (1 - obs_mask))   # non observé, en mer
    """
    w = weight.expand_as(pred) if weight.shape != pred.shape else weight
    return torch.sqrt(((pred - target) ** 2 * w).sum() / w.sum().clamp_min(1.0))


def per_channel_rmse(pred, target, weight, channels, std=None):
    """
    RMSE par canal, optionnellement reconvertie en unités physiques.

    Indispensable ici : une RMSE agrégée sur 8 canaux hétérogènes n'a aucune
    interprétation physique. C'est le chiffre à mettre dans les slides.
    """
    out = {}
    w = weight
    for i, name in enumerate(channels):
        p, t = pred[:, i:i+1], target[:, i:i+1]
        r = torch.sqrt(((p - t) ** 2 * w).sum() / w.sum().clamp_min(1.0))
        r = float(r)
        out[name] = r * float(std[i]) if std is not None else r
    return out


# =============================================================================
#  Rétro-compatibilité — ancien mode 2 canaux (T, S)
# =============================================================================

class OceanOEDDataset(OceanFieldDataset):
    """Ancienne interface (T, S) — conservée pour le mode synthétique."""

    def __init__(self, T, S, sea_mask=None, **kwargs):
        fields = np.stack([T, S], axis=1).astype(np.float32)
        kwargs.pop("augment", None)     # incompatible avec un trait de côte
        super().__init__(fields, ["thetao_z0", "so_z0"],
                         sea_mask=sea_mask, **kwargs)

    @property
    def T_mean(self): return float(self.mean[0])

    @property
    def T_std(self): return float(self.std[0])

    @property
    def S_mean(self): return float(self.mean[1])

    @property
    def S_std(self): return float(self.std[1])
