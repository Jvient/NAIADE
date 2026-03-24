"""
Données synthétiques et échantillonnage bouées.
Version 2 — Dynamique océanique enrichie :
  - Double gyre de fond (circulation méso-échelle)
  - Tourbillons méso-échelle évolutifs (naissance, dérive, décroissance)
  - Front thermique zonal avec méandres temporels
  - Cycle saisonnier SST
  - Turbulence géostrophique spectrale k^-3
  - Relation T-S avec compensation de densité et variabilité propre
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset

# Support dataset.py au niveau racine ou dans data/
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *


# =============================================================================
#  Générateur d'océan physiquement enrichi
# =============================================================================

class SyntheticOceanGenerator:
    """
    Génère un nature run 2D+T de SST et SSS avec des structures physiques
    réalistes pour l'entraînement et l'évaluation des méthodes OED.

    Composantes du champ :
        1. Double gyre de fond  — circulation large-échelle permanente
        2. Front thermique zonal avec méandres temporels
        3. Tourbillons méso-échelle — eddies évoluant en position et amplitude
        4. Cycle saisonnier   — modulation sinusoïdale de SST
        5. Turbulence spectrale k^-3 — bruit petite-échelle cohérent
        6. SSS couplée à SST via compensation de densité + bruit propre
    """

    def __init__(self, nx=NX, ny=NY,
                 n_eddies=5,
                 T_season=365.0,
                 eddy_lifetime=200,
                 front_meander_amp=15.0,
                 front_meander_period=120.0,
                 noise_std=NOISE_STD):
        self.nx = nx
        self.ny = ny
        self.n_eddies = n_eddies
        self.T_season = T_season
        self.eddy_lifetime = eddy_lifetime
        self.front_meander_amp = front_meander_amp
        self.front_meander_period = front_meander_period
        self.noise_std = noise_std

        # Grilles normalisées [0, 1]
        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        self._double_gyre = self._build_double_gyre()
        self._eddies = self._init_eddies()

    # -------------------------------------------------------------------------
    def _build_double_gyre(self):
        """Fonction de courant double-gyre -> proxy SST grande echelle."""
        psi   = np.sin(2 * np.pi * self.X) * np.sin(np.pi * self.Y)
        gyre  = -np.gradient(psi, axis=1)
        gyre -= gyre.mean()
        gyre /= gyre.std() + 1e-9
        return gyre

    # -------------------------------------------------------------------------
    def _init_eddies(self):
        """Initialise N tourbillons avec proprietes aleatoires."""
        eddies = []
        for _ in range(self.n_eddies):
            sign = np.random.choice([-1, 1])
            eddies.append({
                "cx":       np.random.uniform(0.1, 0.9),
                "cy":       np.random.uniform(0.15, 0.85),
                "vx":       np.random.uniform(-1.2e-3, 1.2e-3),
                "vy":       np.random.uniform(-6e-4, 6e-4),
                "amp":      sign * np.random.uniform(0.7, 1.5),
                "radius":   np.random.uniform(0.05, 0.13),
                "phase":    np.random.uniform(0, 2 * np.pi),
                "omega":    sign * np.random.uniform(0.02, 0.05),
                "age":      np.random.randint(0, self.eddy_lifetime),
                "lifetime": int(self.eddy_lifetime * np.random.uniform(0.6, 1.5)),
            })
        return eddies

    # -------------------------------------------------------------------------
    def _eddy_field(self, t):
        """Champ cumule de tous les tourbillons a l instant t."""
        field = np.zeros((self.nx, self.ny))
        for ed in self._eddies:
            age = ed["age"]
            lt  = ed["lifetime"]

            if age >= lt:
                ed["cx"]  = np.random.uniform(0.1, 0.9)
                ed["cy"]  = np.random.uniform(0.15, 0.85)
                ed["amp"] = np.random.choice([-1, 1]) * np.random.uniform(0.7, 1.5)
                ed["radius"] = np.random.uniform(0.05, 0.13)
                ed["age"] = 0
                age = 0

            frac = age / lt
            if frac < 0.15:
                envelope = np.sin(np.pi * frac / 0.3) ** 2
            elif frac > 0.80:
                envelope = np.sin(np.pi * (1 - frac) / 0.4) ** 2
            else:
                envelope = 1.0

            cx = (ed["cx"] + ed["vx"] * age) % 1.0
            cy = (ed["cy"] + ed["vy"] * age) % 1.0

            dx = self.X - cx
            dy = self.Y - cy
            dx = dx - np.round(dx)
            dy = dy - np.round(dy)

            theta = ed["omega"] * age + ed["phase"]
            dx_r  = dx * np.cos(theta) - dy * np.sin(theta)
            dy_r  = dx * np.sin(theta) + dy * np.cos(theta)

            dist2 = (dx_r ** 2 + 0.65 * dy_r ** 2) / (ed["radius"] ** 2)
            field += envelope * ed["amp"] * np.exp(-dist2)
            ed["age"] += 1

        return field

    # -------------------------------------------------------------------------
    def _front_field(self, t):
        """Front zonal avec meandre sinusoidal temporel."""
        amp   = self.front_meander_amp / self.ny
        omega = 2 * np.pi / self.front_meander_period
        y_front = 0.5 + amp * np.sin(2 * np.pi * self.X / 0.5 + omega * t)
        return np.tanh((self.Y - y_front) / 0.06)

    # -------------------------------------------------------------------------
    def _spectral_noise(self, alpha=3.0):
        """Bruit colore k^-alpha normalise."""
        kx = fftfreq(self.nx); ky = fftfreq(self.ny)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(KX**2 + KY**2); K[0, 0] = 1e-6
        noise = np.random.randn(self.nx, self.ny) + 1j * np.random.randn(self.nx, self.ny)
        field = np.real(ifft2(noise * np.sqrt(K**(-alpha))))
        field -= field.mean()
        return field / (field.std() + 1e-9)

    # -------------------------------------------------------------------------
    def _seasonal_forcing(self, t):
        """Cycle saisonnier : modulation sinusoidale + gradient meridional."""
        phase = 2 * np.pi * t / self.T_season
        return np.sin(phase) * (0.6 + 0.4 * (1 - self.Y))

    # -------------------------------------------------------------------------
    def generate_dataset(self, nt=NT, seed=None):
        """
        Genere le nature run complet.

        Composition SST :
            T = w_gyre*gyre + w_front*front(t) + w_eddy*eddies(t)
              + w_season*seasonal(t) + w_noise*turbulence(t)

        SSS couplee a SST avec compensation de densite :
            S = -alpha*T + halocline(y) + bruit_propre
        """
        if seed is not None:
            np.random.seed(seed)

        w_gyre, w_front, w_eddy, w_season, w_noise = 0.25, 0.30, 0.35, 0.20, 0.10

        halocline = np.tanh(3*(self.Y - 0.3)) - np.tanh(3*(self.Y - 0.7))
        halocline /= halocline.std() + 1e-9

        T_series, S_series = [], []

        for t in range(nt):
            T_norm = (w_gyre   * self._double_gyre
                    + w_front  * self._front_field(t)
                    + w_eddy   * self._eddy_field(t)
                    + w_season * self._seasonal_forcing(t)
                    + w_noise  * self._spectral_noise(alpha=3.0) * self.noise_std)

            T_norm = T_norm / (T_norm.std() + 1e-9)

            S_norm = (-TS_CORRELATION * T_norm
                    + 0.3 * halocline
                    + np.sqrt(1 - TS_CORRELATION**2) * self._spectral_noise(alpha=2.5) * 0.15)
            S_norm = S_norm / (S_norm.std() + 1e-9)

            T_series.append((SST_MEAN + SST_STD * T_norm).astype(np.float32))
            S_series.append((SSS_MEAN + SSS_STD * S_norm).astype(np.float32))

        return np.stack(T_series), np.stack(S_series)


# =============================================================================
#  Echantillonneur de bouees
# =============================================================================

class BuoySampler:
    def __init__(self, nx, ny, n_buoys=N_BUOYS):
        self.nx, self.ny = nx, ny
        self.n_buoys = n_buoys
        self.positions = self._random_positions()

    def _random_positions(self):
        xs = np.random.randint(0, self.nx, self.n_buoys)
        ys = np.random.randint(0, self.ny, self.n_buoys)
        return list(zip(xs.tolist(), ys.tolist()))

    def set_positions(self, positions):
        self.positions = positions
        self.n_buoys = len(positions)

    def build_mask(self):
        mask = np.zeros((self.nx, self.ny))
        for (i, j) in self.positions:
            mask[i, j] = 1.0
        return mask

    def sample(self, T, S):
        nt = T.shape[0]
        input_fields = []
        for t in range(nt):
            field = np.zeros((3, self.nx, self.ny))
            mask  = np.zeros((self.nx, self.ny))
            for (x, y) in self.positions:
                field[0, x, y] = T[t, x, y] + np.random.normal(0, OBS_NOISE_STD)
                field[1, x, y] = S[t, x, y] + np.random.normal(0, OBS_NOISE_STD)
                mask[x, y] = 1.0
            field[2] = mask
            input_fields.append(field)
        return np.stack(input_fields)


# =============================================================================
#  Dataset PyTorch — masque stochastique
# =============================================================================

class OceanOEDDataset(Dataset):
    def __init__(self, T, S, n_obs_min=5, n_obs_max=60,
                 noise_std=OBS_NOISE_STD, normalize=True, augment=False):
        self.T = T.astype(np.float32)
        self.S = S.astype(np.float32)
        self.nx, self.ny = T.shape[1], T.shape[2]
        self.n_obs_min = n_obs_min
        self.n_obs_max = n_obs_max
        self.noise_std = noise_std
        self.augment   = augment   # flip spatial (×4 la diversité)

        if normalize:
            self.T_mean = float(self.T.mean()); self.T_std = float(self.T.std())
            self.S_mean = float(self.S.mean()); self.S_std = float(self.S.std())
            self.T = (self.T - self.T_mean) / self.T_std
            self.S = (self.S - self.S_mean) / self.S_std
        else:
            self.T_mean = self.T_std = self.S_mean = self.S_std = None

    def __len__(self):
        return len(self.T)

    def _random_mask(self, n_obs):
        flat = np.zeros(self.nx * self.ny, dtype=np.float32)
        flat[np.random.choice(self.nx * self.ny, n_obs, replace=False)] = 1.0
        return flat.reshape(self.nx, self.ny)

    def __getitem__(self, t):
        n_obs = np.random.randint(self.n_obs_min, self.n_obs_max + 1)
        mask  = self._random_mask(n_obs)
        T_t, S_t = self.T[t], self.S[t]

        # Augmentation spatiale (flip horizontal / vertical aléatoire)
        # Les champs T et S sont physiquement cohérents sous flip
        # (invariance statistique approx. pour les structures méso-échelle)
        if self.augment:
            if np.random.rand() > 0.5:
                T_t = T_t[::-1].copy();  S_t = S_t[::-1].copy()
                mask = mask[::-1].copy()
            if np.random.rand() > 0.5:
                T_t = T_t[:, ::-1].copy(); S_t = S_t[:, ::-1].copy()
                mask = mask[:, ::-1].copy()

        ns = self.noise_std / (self.T_std if self.T_std else 1.0)
        noise = np.random.randn(*T_t.shape).astype(np.float32) * ns
        x = np.stack([T_t*mask + noise*mask, S_t*mask + noise*mask, mask])
        y = np.stack([T_t, S_t])
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(mask[None]))


def build_datasets(T, S, split=0.8, augment_train=False, **kwargs):
    n = len(T); n_tr = int(n * split)
    return (OceanOEDDataset(T[:n_tr], S[:n_tr], augment=augment_train, **kwargs),
            OceanOEDDataset(T[n_tr:], S[n_tr:], augment=False,         **kwargs))


# =============================================================================
#  Figure Nature Run
# =============================================================================

def plot_nature_run(T_arr, S_arr, out_path="outputs/ocean_nature_run.png"):
    """Genere la figure diagnostique complete du nature run."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

    nt = len(T_arr); NX_, NY_ = T_arr.shape[1], T_arr.shape[2]
    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    sal_cmap = LinearSegmentedColormap.from_list("sal",
        ["#003c30","#01665e","#35978f","#80cdc1","#f5f5f5",
         "#dfc27d","#bf812d","#8c510a","#543005"], N=256)

    fig = plt.figure(figsize=(20, 14), facecolor="#0a1628")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.40, wspace=0.35,
                            left=0.06, right=0.96, top=0.92, bottom=0.06)

    def styled(ax, title, im, label):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
        ax.set_title(title, color="white", fontsize=9, pad=5, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(label, color="white", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)

    vTmin, vTmax = T_arr.min(), T_arr.max()
    vSmin, vSmax = S_arr.min(), S_arr.max()
    snaps = [0, nt//4, nt//2]

    for col, t in enumerate(snaps):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(T_arr[t].T, cmap=ocean_cmap, origin="lower",
                       aspect="auto", vmin=vTmin, vmax=vTmax)
        ax.text(0.02, 0.95, f"t={t}", transform=ax.transAxes,
                color="white", fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="#0a1628", alpha=0.7))
        styled(ax, f"SST — t = {t}", im, "°C")

    ax = fig.add_subplot(gs[0, 3])
    im  = ax.imshow(T_arr.std(axis=0).T, cmap="plasma", origin="lower", aspect="auto")
    styled(ax, "Variabilite SST (sigma temporelle)", im, "°C")

    ax = fig.add_subplot(gs[1, 0])
    im  = ax.imshow(S_arr[0].T, cmap=sal_cmap, origin="lower",
                    aspect="auto", vmin=vSmin, vmax=vSmax)
    styled(ax, "SSS — t = 0", im, "psu")

    ax = fig.add_subplot(gs[1, 1])
    gx = np.gradient(T_arr[0], axis=0); gy = np.gradient(T_arr[0], axis=1)
    im  = ax.imshow(np.sqrt(gx**2+gy**2).T, cmap="hot", origin="lower", aspect="auto")
    styled(ax, "|grad SST| — t=0  (fronts/eddies)", im, "degC/px")

    ax = fig.add_subplot(gs[1, 2])
    T_a = T_arr - T_arr.mean(axis=0); S_a = S_arr - S_arr.mean(axis=0)
    corr = (T_a*S_a).mean(0) / (T_arr.std(0)*S_arr.std(0) + 1e-9)
    im   = ax.imshow(corr.T, cmap="RdBu_r", origin="lower", aspect="auto",
                     vmin=-1, vmax=1)
    styled(ax, "Correlation T-S temporelle", im, "rho")

    ax = fig.add_subplot(gs[1, 3])
    anom = T_arr[nt//2] - T_arr.mean(0); lim = anom.std()*2.5
    im   = ax.imshow(anom.T, cmap="RdBu_r", origin="lower", aspect="auto",
                     vmin=-lim, vmax=lim)
    styled(ax, f"Anomalie SST t={nt//2}", im, "degC")

    # Spectre radial
    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor("#050d1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
    fft_T = np.abs(np.fft.fft2(T_arr[0]-T_arr[0].mean()))**2
    FX, FY = np.meshgrid(fftfreq(NX_), fftfreq(NY_), indexing="ij")
    Kr = np.sqrt(FX**2+FY**2).ravel(); Pr = fft_T.ravel()
    kb = np.linspace(0.01, 0.48, 35); kc = 0.5*(kb[:-1]+kb[1:])
    Pb = [Pr[(Kr>=kb[i])&(Kr<kb[i+1])].mean() for i in range(len(kb)-1)]
    ax.loglog(kc, Pb, color="#6baed6", lw=2)
    kr = np.array([0.02,0.4]); ax.loglog(kr, 2e-4*kr**(-3), "w--", lw=1, alpha=0.6, label="k^-3")
    ax.set_title("Spectre radial SST", color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("k", color="white", fontsize=7)
    ax.tick_params(colors="white", labelsize=6)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#0a1628")
    ax.grid(True, alpha=0.2, color="white")

    # Series temporelles
    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor("#050d1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
    for (x,y,c,lbl) in [(NX_//5,NY_//4,"#ff6b6b","Pt A"),
                         (NX_//2,NY_//2,"#ffd93d","Pt B"),
                         (4*NX_//5,3*NY_//4,"#6bcb77","Pt C")]:
        ax.plot(T_arr[:,x,y], color=c, lw=1.2, alpha=0.9, label=lbl)
    ax.set_title("Series SST — 3 points", color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("Temps (pas)", color="white", fontsize=7)
    ax.tick_params(colors="white", labelsize=6)
    ax.legend(fontsize=7, labelcolor="white", facecolor="#0a1628")
    ax.grid(True, alpha=0.2, color="white")

    # Distributions
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor("#050d1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a4a7a")
    ax.hist(T_arr.ravel(), bins=60, color="#fc8d59", alpha=0.75, density=True, label="SST")
    ax2t = ax.twinx(); ax2t.set_facecolor("#050d1a")
    ax2t.hist(S_arr.ravel(), bins=60, color="#6baed6", alpha=0.6, density=True, label="SSS")
    ax.set_title("Distributions SST / SSS", color="white", fontsize=9, fontweight="bold")
    ax.tick_params(colors="white", labelsize=6); ax2t.tick_params(colors="#6baed6", labelsize=6)
    ax.legend(fontsize=7, labelcolor="white", facecolor="#0a1628", loc="upper left")
    ax2t.legend(fontsize=7, labelcolor="white", facecolor="#0a1628", loc="upper right")
    ax.grid(True, alpha=0.2, color="white")

    # Carte bouees
    ax = fig.add_subplot(gs[2, 3])
    im = ax.imshow(T_arr[0].T, cmap=ocean_cmap, origin="lower",
                   aspect="auto", vmin=vTmin, vmax=vTmax)
    np.random.seed(99)
    bx = np.random.randint(0, NX_, N_BUOYS); by = np.random.randint(0, NY_, N_BUOYS)
    ax.scatter(bx, by, c="white", s=35, edgecolors="black", linewidths=0.7, zorder=5)
    ax.scatter(bx, by, c="#ffd93d", s=10, zorder=6)
    styled(ax, f"SST + bouees (N={N_BUOYS})", im, "degC")

    fig.text(0.5, 0.965,
             "Nature Run 2D+T  —  Double Gyre + Eddies + Front  —  SST & SSS",
             ha="center", color="white", fontsize=13, fontweight="bold")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="#0a1628", bbox_inches="tight")
    plt.close()
    print(f"  Nature run figure -> {out_path}")


# =============================================================================
#  CLI
# =============================================================================
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--nt",   type=int, default=NT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out",  type=str, default="outputs/ocean_nature_run.png")
    args = p.parse_args()
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=args.nt, seed=args.seed)
    print(f"T: {T.shape}  [{T.min():.1f}, {T.max():.1f}] degC")
    print(f"S: {S.shape}  [{S.min():.2f}, {S.max():.2f}] psu")
    plot_nature_run(T, S, out_path=args.out)
