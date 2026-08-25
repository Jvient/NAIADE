"""
2D+T surface nature run -- SST / SSS / SSH -- and buoy sampling.

Version 3 -- a dynamical model rather than a superposition of patterns
=====================================================================
Version 2 built SST by adding analytical patterns (gyre + Gaussians + noise)
redrawn at every time step. The result had no small-scale temporal coherence,
no filamentation, and no link between the current and the tracer.

Here the ocean is produced by an actual small model:

    1.  Geostrophic streamfunction psi(x,y,t)
            psi = double gyre + meandering jet + eddies + perturbation
        hence u = -dpsi/dy, v = +dpsi/dx  (non-divergent by construction)
        and sea surface height SSH = f0 * psi / g.

    2.  SST and SSS tracers ADVECTED by that flow:
            dC/dt + u.grad(C) = -(C - C_clim(y,t))/tau + kappa * lap(C)
        semi-Lagrangian scheme (Catmull-Rom interpolation) + implicit restoring.

    3.  Fronts, filaments and gradients are not drawn: they EMERGE from the
        competition between stirring by the flow and restoring towards
        climatology. That is what gives the field its realistic texture.

    4.  Eddies live in psi (not in SST), are advected by the large-scale flow
        plus westward beta drift, are born preferentially along the jet
        (baroclinic instability), and decay.

    5.  SST and SSS have DIFFERENT restoring timescales (heat flux ~40 days
        against freshwater flux ~150 days). Their temporal decorrelation
        scales therefore differ -- which is precisely the information that
        justifies sizing a network variable by variable.

All quantities are dimensional (km, days, m/s, degC, psu, m) -- see config.py.

Unchanged API:
    SyntheticOceanGenerator().generate_dataset(nt, seed) -> (T, S)
New:
    .generate_full(nt, seed) -> full dict (T, S, SSH, U, V, ZETA, SIGMA0)
    .diagnostics()           -> decorrelation scales, EKE, T-S correlation
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python data/dataset.py`
from config import *

if HAS_TORCH:
    from torch.utils.data import Dataset
else:                                   # numpy-only fallback: the generator and
    class Dataset:                      # its diagnostics stay usable without torch
        pass


# =============================================================================
#  Physical constants
# =============================================================================
OMEGA_EARTH = 7.2921e-5      # rad/s
R_EARTH     = 6.371e6        # m
G_GRAV      = 9.81           # m/s2
DAY         = 86400.0        # s


def sigma0(T, S):
    """Surface potential density (EOS-80, p = 0) minus 1000 kg/m3."""
    T = np.asarray(T, dtype=np.float64); S = np.asarray(S, dtype=np.float64)
    rho_w = (999.842594 + 6.793952e-2*T - 9.095290e-3*T**2
             + 1.001685e-4*T**3 - 1.120083e-6*T**4 + 6.536332e-9*T**5)
    A = (0.824493 - 4.0899e-3*T + 7.6438e-5*T**2
         - 8.2467e-7*T**3 + 5.3875e-9*T**4)
    B = -5.72466e-3 + 1.0227e-4*T - 1.6546e-6*T**2
    C = 4.8314e-4
    return rho_w + A*S + B*S**1.5 + C*S**2 - 1000.0


# =============================================================================
#  Catmull-Rom bicubic interpolation (periodic in x, clamped in y)
# =============================================================================

def _cr_weights(f):
    f2 = f * f; f3 = f2 * f
    return (-0.5*f3 + f2 - 0.5*f,
             1.5*f3 - 2.5*f2 + 1.0,
            -1.5*f3 + 2.0*f2 + 0.5*f,
             0.5*f3 - 0.5*f2)


class _Stencil:
    """
    Precomputed Catmull-Rom bicubic interpolation stencil.

    Indices and weights depend only on the departure points, not on the field
    being interpolated: they are built once per time step and applied to SST,
    SSS, u and v. This is what makes the semi-Lagrangian scheme affordable in
    pure numpy (periodic in x, clamped in y).
    """
    __slots__ = ("idx", "w", "shape")

    def __init__(self, xi, yi, nx, ny):
        i0 = np.floor(xi).astype(np.int32); fx = (xi - i0).astype(np.float32)
        j0 = np.floor(yi).astype(np.int32); fy = (yi - j0).astype(np.float32)
        wx = _cr_weights(fx); wy = _cr_weights(fy)
        idx = np.empty((16,) + xi.shape, dtype=np.int32)
        w   = np.empty((16,) + xi.shape, dtype=np.float32)
        for a in range(4):
            ia = (i0 + (a - 1)) % nx
            for b in range(4):
                jb = np.clip(j0 + (b - 1), 0, ny - 1)
                k = 4*a + b
                idx[k] = ia * ny + jb
                w[k]   = wx[a] * wy[b]
        self.idx, self.w, self.shape = idx, w, xi.shape

    def apply(self, F):
        return np.einsum("kij,kij->ij", self.w, F.ravel()[self.idx])


def _interp_bicubic(F, xi, yi):
    """One-off interpolation (occasional use / backward compatibility)."""
    return _Stencil(xi, yi, *F.shape).apply(F)


def _laplacian(F):
    """5-point Laplacian: periodic in x, Neumann (zero flux) in y."""
    lap = np.roll(F, 1, axis=0) + np.roll(F, -1, axis=0) - 2.0 * F
    Fy = np.empty((F.shape[0], F.shape[1] + 2), dtype=F.dtype)
    Fy[:, 1:-1] = F; Fy[:, 0] = F[:, 0]; Fy[:, -1] = F[:, -1]
    lap += Fy[:, 2:] + Fy[:, :-2] - 2.0 * F
    return lap


def _grad(F, dx, dy):
    """Centred gradient: periodic in x, one-sided at the y boundaries."""
    dFdx = (np.roll(F, -1, axis=0) - np.roll(F, 1, axis=0)) / (2 * dx)
    dFdy = np.empty_like(F)
    dFdy[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * dy)
    dFdy[:, 0]    = (F[:, 1] - F[:, 0]) / dy
    dFdy[:, -1]   = (F[:, -1] - F[:, -2]) / dy
    return dFdx, dFdy


# =============================================================================
#  Generator
# =============================================================================

class SyntheticOceanGenerator:
    """
    OSSE nature run: SST / SSS / SSH over a zonal channel of Lx x Ly.

    `generate_dataset(nt, seed)` is fully deterministic: the seed controls the
    eddy population, the jet meanders and the stochastic noise. Two calls with
    the same seed give the same ocean.
    """

    def __init__(self, nx=NX, ny=NY, dx_km=DX_KM, lat0=LAT0,
                 n_eddies=N_EDDIES, seed=None, **legacy):
        self.nx, self.ny = nx, ny
        self.dx = self.dy = dx_km * 1e3                 # m
        self.Lx = nx * self.dx
        self.Ly = ny * self.dy
        self.lat0 = lat0
        self.n_eddies = n_eddies

        # Coriolis parameters on a beta plane
        phi      = np.deg2rad(lat0)
        self.f0  = 2 * OMEGA_EARTH * np.sin(phi)
        self.beta = 2 * OMEGA_EARTH * np.cos(phi) / R_EARTH

        # Metric grids (X zonal = axis 0, Y meridional = axis 1)
        self.xg = (np.arange(nx) + 0.5) * self.dx
        self.yg = (np.arange(ny) + 0.5) * self.dy
        self.X, self.Y = np.meshgrid(self.xg, self.yg, indexing="ij")
        self.Yf = self.Y / self.Ly                      # y normalised to [0, 1]

        # Taper near the walls: psi -> const at the boundaries, hence v -> 0
        Lw = 5 * self.dx
        self.wall = np.tanh(self.Y / Lw) * np.tanh((self.Ly - self.Y) / Lw)

        # Index grid (arrival points of the semi-Lagrangian scheme)
        self._ix = np.repeat(np.arange(nx, dtype=np.float64)[:, None], ny, axis=1)
        self._iy = np.repeat(np.arange(ny, dtype=np.float64)[None, :], nx, axis=0)

        self._seed(seed)

    # -------------------------------------------------------------------------
    #  Stochastic initialisation
    # -------------------------------------------------------------------------
    def _seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self._init_jet()
        self._init_eddies()
        self.psi_pert = np.zeros((self.nx, self.ny))

    def _init_jet(self):
        """Jet meanders: 3 integer zonal modes (domain is periodic in x)."""
        self.jet_modes = []
        for m, amp_km, per_d in zip((1, 2, 3), (55.0, 32.0, 18.0), (140., 85., 45.)):
            self.jet_modes.append({
                "m":     m,
                "amp":   amp_km * 1e3 * self.rng.uniform(0.7, 1.3),
                "omega": 2 * np.pi / (per_d * self.rng.uniform(0.8, 1.25)),
                "phase": self.rng.uniform(0, 2 * np.pi),
            })

    def _spawn_eddy(self, t=0.0, newborn=True):
        """Eddy birth: preferentially along the jet."""
        cx = self.rng.uniform(0, self.Lx)
        if self.rng.random() < 0.50:                     # meander pinch-off
            side = self.rng.choice([-1.0, 1.0])
            cy = self._jet_axis_1d(cx, t) + side * self.rng.uniform(0.6, 2.2) * EDDY_R_KM[1]*1e3
        else:
            cy = self.rng.uniform(0.08, 0.92) * self.Ly
        cy = float(np.clip(cy, 0.10 * self.Ly, 0.90 * self.Ly))
        life = self.rng.uniform(*EDDY_LIFE_DAYS)
        R    = self.rng.uniform(*EDDY_R_KM) * 1e3
        sign = self.rng.choice([-1.0, 1.0])              # cyclone / anticyclone
        return {
            "cx": cx, "cy": cy, "R": R, "sign": sign,
            "V":  EDDY_V_MAX * self.rng.uniform(0.55, 1.25),
            "t0": t if newborn else t - self.rng.uniform(0, life),
            "life": life,
        }

    def _init_eddies(self):
        self.eddies = [self._spawn_eddy(0.0, newborn=False)
                       for _ in range(self.n_eddies)]

    # -------------------------------------------------------------------------
    #  Streamfunction
    # -------------------------------------------------------------------------
    def _jet_axis_1d(self, x, t):
        """Latitude of the jet axis at a point x (m) and time t (days)."""
        y = JET_LAT_FRAC * self.Ly
        for md in self.jet_modes:
            y += md["amp"] * np.sin(2*np.pi*md["m"]*x/self.Lx
                                    - md["omega"]*t + md["phase"])
        return y

    def _psi_jet(self, t):
        yj = self._jet_axis_1d(self.X, t)
        L  = JET_WIDTH_KM * 1e3
        return -U_JET * L * np.tanh((self.Y - yj) / L)

    def _psi_gyre(self):
        psi0 = U_GYRE * self.Ly / np.pi
        return psi0 * np.sin(2*np.pi*self.X/self.Lx) * np.sin(np.pi*self.Yf)

    def _psi_eddies(self, t):
        psi = np.zeros((self.nx, self.ny))
        for ed in self.eddies:
            age  = t - ed["t0"]
            frac = age / ed["life"]
            env  = np.sin(np.pi * np.clip(frac, 0, 1)) ** 0.6   # naissance/mort douces
            if env <= 1e-3:
                continue
            A  = ed["sign"] * ed["V"] * ed["R"] * np.sqrt(np.e) * env
            dx = self.X - ed["cx"]
            dx -= self.Lx * np.round(dx / self.Lx)               # periodic in x
            dy = self.Y - ed["cy"]
            psi += A * np.exp(-(dx**2 + dy**2) / (2 * ed["R"]**2))
        return psi * self.wall

    def _step_eddies(self, t, dt_d, u, v):
        """Eddy advection by the flow + westward beta drift."""
        c_beta = -self.beta * (RD_KM * 1e3) ** 2                 # m/s (< 0 = westward)
        for k, ed in enumerate(self.eddies):
            if t - ed["t0"] > ed["life"]:
                self.eddies[k] = self._spawn_eddy(t, newborn=True)
                continue
            i = int(ed["cx"] / self.dx) % self.nx
            j = int(np.clip(ed["cy"] / self.dy, 0, self.ny - 1))
            ed["cx"] = (ed["cx"] + (0.5*u[i, j] + c_beta) * dt_d * DAY) % self.Lx
            ed["cy"] = float(np.clip(ed["cy"] + 0.5*v[i, j] * dt_d * DAY,
                                     0.06 * self.Ly, 0.94 * self.Ly))

    def _step_pert(self, dt_d):
        """Unresolved mesoscale perturbation: Ornstein-Uhlenbeck in time,
        k^-3 spectrum in space (2D geostrophic turbulence)."""
        a = np.exp(-dt_d / PERT_TAU_DAYS)
        b = np.sqrt(1 - a*a)
        self.psi_pert = a * self.psi_pert + b * self._colored_field(3.0)

    def _colored_field(self, alpha):
        """Random field with a k^-alpha spectrum, unit variance, filtered
        below the 4-grid-cell scale."""
        kx = fftfreq(self.nx); ky = fftfreq(self.ny)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(KX**2 + KY**2); K[0, 0] = 1e-9
        amp = K ** (-alpha / 2.0)
        amp[0, 0] = 0.0
        amp *= np.exp(-(K / 0.25) ** 4)                # sub-grid cutoff
        z = (self.rng.standard_normal((self.nx, self.ny))
             + 1j * self.rng.standard_normal((self.nx, self.ny)))
        f = np.real(ifft2(z * amp))
        f -= f.mean()
        return f / (f.std() + 1e-12)

    def _velocity(self, t):
        psi = (self._psi_gyre() + self._psi_jet(t) + self._psi_eddies(t)
               + PERT_AMP * self.psi_pert * self.wall)
        dpx, dpy = _grad(psi, self.dx, self.dy)
        return psi, -dpy, dpx                          # psi, u, v

    # -------------------------------------------------------------------------
    #  Restoring climatologies
    # -------------------------------------------------------------------------
    def _T_clim(self, t):
        """Meridional gradient (warm to the south) + seasonal cycle damped northward."""
        merid = SST_GRADIENT * (0.5 - self.Yf)
        seas  = SST_SEASONAL_AMP * np.sin(2*np.pi*(t - SEASON_PHASE_DAYS)/365.25)
        return SST_MEAN + merid + seas * (1.0 + 0.35*(0.5 - self.Yf))

    def _S_clim(self, t):
        """
        Subtropical salinity maximum (evaporation) aligned with the T gradient
        -> positive T-S correlation, consistent with density compensation.
        A coastal fresh plume (north-west) breaks the degeneracy: the T-S
        correlation becomes spatially variable, as in the real ocean.
        """
        aligned = SSS_GRADIENT * (0.5 - self.Yf)
        plume   = -SSS_PLUME_AMP * np.exp(
            -((self.X - 0.18*self.Lx)**2 / (2*(0.11*self.Lx)**2)
              + (self.Y - 0.80*self.Ly)**2 / (2*(0.09*self.Ly)**2)))
        rho = np.clip(TS_CORRELATION, -1.0, 1.0)
        seas = 0.10 * SSS_GRADIENT * np.sin(2*np.pi*(t - SEASON_PHASE_DAYS - 60)/365.25)
        return SSS_MEAN + rho*aligned + np.sqrt(1 - rho**2)*plume + seas

    # -------------------------------------------------------------------------
    #  Integration
    # -------------------------------------------------------------------------
    def _departure_stencil(self, u, v, dt_s):
        """
        Departure-point stencil for one semi-Lagrangian step.
        Since the velocity field is frozen over the output step, this stencil
        is identical for every substep and every tracer: it is built once.
        """
        ix, iy = self._ix, self._iy
        cx = u * (dt_s / self.dx)
        cy = v * (dt_s / self.dy)
        # midpoint iteration (second-order trajectory)
        mid = _Stencil(ix - 0.5*cx,
                       np.clip(iy - 0.5*cy, 0, self.ny - 1), self.nx, self.ny)
        cx = mid.apply(u) * (dt_s / self.dx)
        cy = mid.apply(v) * (dt_s / self.dy)
        return _Stencil(ix - cx, np.clip(iy - cy, 0, self.ny - 1),
                        self.nx, self.ny)

    def _integrate(self, nt, spinup, record=True, stride=1, light=False):
        dt_out = DT_DAYS
        nsub   = max(1, int(N_SUBSTEPS))
        dt_s   = dt_out * DAY / nsub
        kdt    = KAPPA * dt_s / self.dx**2               # diffusion number
        rT     = dt_s / (TAU_T_DAYS * DAY)
        rS     = dt_s / (TAU_S_DAYS * DAY)

        T_out, S_out, H_out, U_out, V_out = [], [], [], [], []
        t = -spinup * dt_out

        for n in range(spinup + nt):
            self._step_pert(dt_out)
            psi, u, v = self._velocity(t)
            stencil = self._departure_stencil(u, v, dt_s)

            for k in range(nsub):
                ts = t + (k + 0.5) * dt_out / nsub
                Tc, Sc = self._T_clim(ts), self._S_clim(ts)
                self.T = stencil.apply(self.T)
                self.S = stencil.apply(self.S)
                self.T += kdt * _laplacian(self.T)
                self.S += kdt * _laplacian(self.S)
                self.T = (self.T + rT * Tc) / (1 + rT)
                self.S = (self.S + rS * Sc) / (1 + rS)

            self._step_eddies(t, dt_out, u, v)
            t += dt_out

            if record and n >= spinup:
                sl = (slice(None, None, stride), slice(None, None, stride))
                T_out.append(self.T[sl].copy()); S_out.append(self.S[sl].copy())
                if light:            # long runs: T/S only, 5x less memory
                    continue
                H_out.append(self.f0 * psi / G_GRAV)
                U_out.append(u.copy()); V_out.append(v.copy())

        if not record:
            return None
        if light:
            return (np.stack(T_out).astype(np.float32),
                    np.stack(S_out).astype(np.float32), None, None, None)
        return (np.stack(T_out).astype(np.float32),
                np.stack(S_out).astype(np.float32),
                np.stack(H_out).astype(np.float32),
                np.stack(U_out).astype(np.float32),
                np.stack(V_out).astype(np.float32))

    # -------------------------------------------------------------------------
    #  Public API
    # -------------------------------------------------------------------------
    def generate_full(self, nt=NT, seed=None, spinup_days=None):
        """
        Full nature run. Returns a dict:
            T (degC), S (psu), SSH (m), U, V (m/s), ZETA (s-1), SIGMA0 (kg/m3)
        all shaped (nt, nx, ny), plus the domain metadata.
        """
        self._seed(seed)
        spinup = int(SPINUP_DAYS if spinup_days is None else spinup_days)

        # Initial state = climatology + a small coherent perturbation
        self.T = self._T_clim(-spinup * DT_DAYS) + 0.25 * self._colored_field(3.0)
        self.S = self._S_clim(-spinup * DT_DAYS) + 0.03 * self._colored_field(3.0)

        T, S, H, U, V = self._integrate(nt, spinup)

        dvdx = (np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)) / (2*self.dx)
        dudy = np.empty_like(U)
        dudy[:, :, 1:-1] = (U[:, :, 2:] - U[:, :, :-2]) / (2*self.dy)
        dudy[:, :, 0] = dudy[:, :, 1]; dudy[:, :, -1] = dudy[:, :, -2]
        ZETA = (dvdx - dudy).astype(np.float32)

        self.last_run = {
            "T": T, "S": S, "SSH": H, "U": U, "V": V, "ZETA": ZETA,
            "SIGMA0": sigma0(T, S).astype(np.float32),
            "f0": self.f0, "beta": self.beta, "dx_m": self.dx,
            "Lx_km": self.Lx/1e3, "Ly_km": self.Ly/1e3,
            "dt_days": DT_DAYS, "lat0": self.lat0, "seed": seed,
        }
        return self.last_run

    def generate_light(self, nt=NT, seed=None, stride=1, spinup_days=None):
        """
        Multi-year nature run without blowing up memory.

        Same dynamics as `generate_full`, but only T and S are recorded, and
        only every `stride`-th grid point. A 13-year run on a 192x288 domain
        costs 5.2 GB with `generate_full` and 130 MB here at stride 4 -- which
        is all the OED bricks need, since the mesoscale decorrelation scale
        (~100 km) is 20x the grid spacing.

        Returns (T, S) shaped (nt, ceil(nx/stride), ceil(ny/stride)).
        """
        self._seed(seed)
        spinup = int(SPINUP_DAYS if spinup_days is None else spinup_days)
        self.T = self._T_clim(-spinup * DT_DAYS) + 0.25 * self._colored_field(3.0)
        self.S = self._S_clim(-spinup * DT_DAYS) + 0.03 * self._colored_field(3.0)
        T, S, _, _, _ = self._integrate(nt, spinup, stride=stride, light=True)
        self.stride = stride
        return T, S

    def generate_dataset(self, nt=NT, seed=None):
        """Pipeline compatibility: returns (SST, SSS) shaped (nt, nx, ny)."""
        run = self.generate_full(nt=nt, seed=seed)
        return run["T"], run["S"]

    # -------------------------------------------------------------------------
    def diagnostics(self, run=None):
        """Characteristic scales of the nature run (useful for sizing the
        spacing and sampling frequency of a network)."""
        r = run or getattr(self, "last_run", None)
        if r is None:
            raise RuntimeError("Call generate_full() first.")
        T, S, U, V = r["T"], r["S"], r["U"], r["V"]
        dxkm = r["dx_m"] / 1e3

        def decorr_time(C, deseason=False):
            A = C - C.mean(axis=0, keepdims=True)
            if deseason:                       # remove the large-scale signal
                A = A - A.mean(axis=(1, 2), keepdims=True)
            A = A / (A.std(axis=0, keepdims=True) + 1e-9)
            nlag = min(90, len(C)//3)
            ac = np.array([(A[:len(A)-l] * A[l:]).mean() for l in range(nlag)])
            below = np.where(ac < 1/np.e)[0]
            return float(below[0] * r["dt_days"]) if len(below) else float(nlag)

        def decorr_len(C):
            """Zonal decorrelation scale of the anomalies (the zonal mean is
            removed, otherwise the large-scale structure dominates)."""
            F0 = C[len(C)//2]
            A = F0 - F0.mean(axis=0, keepdims=True)
            Fh = np.fft.rfft(A, axis=0)
            ac = np.fft.irfft((Fh*np.conj(Fh)).real, axis=0).mean(axis=1)
            ac /= ac[0] + 1e-12
            below = np.where(ac[:A.shape[0]//2] < 1/np.e)[0]
            return float(below[0] * dxkm) if len(below) else float(A.shape[0]//2*dxkm)

        eke = 0.5*(U.var(axis=0) + V.var(axis=0))
        Ta = T - T.mean(0); Sa = S - S.mean(0)
        rho_map = (Ta*Sa).mean(0) / (T.std(0)*S.std(0) + 1e-9)
        return {
            "SST_mean": float(T.mean()), "SST_std": float(T.std()),
            "SSS_mean": float(S.mean()), "SSS_std": float(S.std()),
            "SST_seasonal_range_degC": float(np.ptp(T.mean(axis=(1, 2)))),
            "tau_decorr_SST_days": decorr_time(T),
            "tau_decorr_SSS_days": decorr_time(S),
            "tau_SST_mesoscale_days": decorr_time(T, deseason=True),
            "tau_SSS_mesoscale_days": decorr_time(S, deseason=True),
            "L_decorr_SST_km":  decorr_len(T),
            "L_decorr_SSS_km":  decorr_len(S),
            "speed_rms_m_s":  float(np.sqrt((U**2 + V**2).mean())),
            "speed_p99_m_s":  float(np.percentile(np.sqrt(U**2 + V**2), 99)),
            "EKE_mean_m2_s2":    float(eke.mean()),
            "Rossby_p99":       float(np.percentile(np.abs(r["ZETA"]), 99) / abs(r["f0"])),
            "corr_TS_global":  float(np.corrcoef(T.ravel(), S.ravel())[0, 1]),
            "corr_TS_anomaly_median": float(np.median(rho_map)),
        }


# =============================================================================
#  Analysis utilities shared by the three bricks
# =============================================================================

def mesoscale_anomaly(F):
    """
    Remove the domain mean at every time step.

    Why this is indispensable here: the seasonal cycle is a near-uniform mode
    over the whole domain. Keep it and two buoys 1000 km apart show a
    correlation above 0.8 simply because both see summer arrive -- the GNN
    graph becomes a near-clique and the RL variance map flattens out. Once it
    is removed only mesoscale variability is left, which is what the network
    actually has to sample.
    """
    F = np.asarray(F)
    return F - F.mean(axis=(1, 2), keepdims=True)


def local_variance_map(T, S, positions, half_win=2, deseason=DESEASON_ANALYSIS,
                       w_T=0.6, w_S=0.4):
    """
    Local mesoscale variance around each position, for SST and SSS.

    The two variances are standardised SEPARATELY before being combined:
    var(SST) ~ 3 degC^2 against var(SSS) ~ 0.03 psu^2, so a direct mix of
    0.6*var_T + 0.4*var_S drops the salinity contribution below 0.1% of the
    mixture variance -- that is, to nothing at all.

    Returns (mix, vT_raw, vS_raw); mix is standardised to mean 0, std 1.
    """
    Tw, Sw = (mesoscale_anomaly(T), mesoscale_anomaly(S)) if deseason else (T, S)
    nx, ny = T.shape[1], T.shape[2]
    vT, vS = [], []
    for (px, py) in positions:
        x0, x1 = max(0, px - half_win), min(nx, px + half_win + 1)
        y0, y1 = max(0, py - half_win), min(ny, py + half_win + 1)
        vT.append(float(Tw[:, x0:x1, y0:y1].var()))
        vS.append(float(Sw[:, x0:x1, y0:y1].var()))
    vT = np.array(vT, dtype=np.float32); vS = np.array(vS, dtype=np.float32)
    zT = (vT - vT.mean()) / (vT.std() + 1e-9)
    zS = (vS - vS.mean()) / (vS.std() + 1e-9)
    mix = w_T * zT + w_S * zS
    mix = (mix - mix.mean()) / (mix.std() + 1e-9)
    return mix.astype(np.float32), vT, vS


def sensor_series(T, S, positions, deseason=DESEASON_ANALYSIS,
                  w_T=0.6, w_S=0.4, t_idx=None):
    """
    Standardised time series seen by each sensor (T/S mixture).
    Common basis for the GNN correlation matrix.
    """
    Tw, Sw = (mesoscale_anomaly(T), mesoscale_anomaly(S)) if deseason else (T, S)
    if t_idx is None:
        t_idx = np.arange(len(T))
    out = np.zeros((len(positions), len(t_idx)), dtype=np.float32)
    for k, (x, y) in enumerate(positions):
        a = Tw[:, x, y]; b = Sw[:, x, y]
        a = (a[t_idx] - a.mean()) / (a.std() + 1e-9)
        b = (b[t_idx] - b.mean()) / (b.std() + 1e-9)
        out[k] = w_T * a + w_S * b
    return out


def sample_separated_positions(nx, ny, n, min_sep_km=MIN_BUOY_SEP_KM,
                               rng=None, dx_km=DX_KM, max_tries=200):
    """
    Draw n pixel positions separated by at least `min_sep_km`.

    Two buoys that are too close are redundant by construction (they see the
    same mesoscale structure) and, on the RL candidate grid, they are
    forbidden. The reference networks of the AE and GNN bricks must obey the
    same constraint, otherwise feasible networks are compared against
    infeasible ones.

    Rejection sampling, then progressive relaxation of the constraint if the
    domain cannot host n positions (with an explicit warning).
    """
    rng = rng or np.random.default_rng()
    sep = float(min_sep_km) / float(dx_km)          # en pixels
    for attempt in range(6):
        pts = []
        for _ in range(max_tries * n):
            if len(pts) >= n:
                break
            c = (int(rng.integers(0, nx)), int(rng.integers(0, ny)))
            if all((c[0]-q[0])**2 + (c[1]-q[1])**2 >= sep*sep for q in pts):
                pts.append(c)
        if len(pts) >= n:
            return pts[:n]
        sep *= 0.8                                   # domain too constrained
    print(f"  [WARNING] {min_sep_km} km separation unreachable for {n} buoys: "
          f"relaxed to {sep*dx_km:.0f} km")
    while len(pts) < n:
        pts.append((int(rng.integers(0, nx)), int(rng.integers(0, ny))))
    return pts[:n]


# =============================================================================
#  Buoy sampler
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
                field[0, x, y] = T[t, x, y] + np.random.normal(0, OBS_NOISE_T)
                field[1, x, y] = S[t, x, y] + np.random.normal(0, OBS_NOISE_S)
                mask[x, y] = 1.0
            field[2] = mask
            input_fields.append(field)
        return np.stack(input_fields)


# =============================================================================
#  PyTorch dataset -- stochastic observation mask
# =============================================================================

class OceanOEDDataset(Dataset):
    def __init__(self, T, S, n_obs_min=5, n_obs_max=60,
                 noise_std=None, normalize=True, augment=False):
        self.T = T.astype(np.float32)
        self.S = S.astype(np.float32)
        self.nx, self.ny = T.shape[1], T.shape[2]
        self.n_obs_min = n_obs_min
        self.n_obs_max = n_obs_max
        # Per-variable instrumental noise, in physical units.
        # (previously a single scalar divided by T_std was applied to BOTH
        #  channels: the SSS channel received noise calibrated on temperature,
        #  unrelated to its own dynamic range)
        if noise_std is None:
            self.noise_T, self.noise_S = OBS_NOISE_T, OBS_NOISE_S
        elif np.isscalar(noise_std):
            self.noise_T = self.noise_S = float(noise_std)
        else:
            self.noise_T, self.noise_S = map(float, noise_std)
        self.noise_std = self.noise_T          # legacy alias
        self.augment   = augment

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

        # Augmentation: only the zonal flip is legitimate (the channel is
        # periodic in x). A meridional flip would reverse the north-south
        # gradient imposed by the climatological forcing -> physically
        # inconsistent field.
        if self.augment and np.random.rand() > 0.5:
            T_t = T_t[::-1].copy(); S_t = S_t[::-1].copy()
            mask = mask[::-1].copy()

        ns_T = self.noise_T / (self.T_std if self.T_std else 1.0)
        ns_S = self.noise_S / (self.S_std if self.S_std else 1.0)
        nT = np.random.randn(*T_t.shape).astype(np.float32) * ns_T
        nS = np.random.randn(*S_t.shape).astype(np.float32) * ns_S
        x = np.stack([(T_t + nT)*mask, (S_t + nS)*mask, mask])
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

def animate_nature_run(run, out_path="outputs/ocean_nature_run.gif",
                       every=5, var="T,GRADT,S,GRADS", fps=8, max_frames=120):
    """
    Animate the nature run, one frame every `every` days.

    Parameters
    ----------
    run : dict from generate_full, fields shaped (nt, nx, ny)
    every : stride in days between two frames
    var : one field, or several separated by commas. Understood names:
          T, S, SSH, ZETA, GRADT, GRADS. The two gradient moduli show the
          fronts and filaments far better than the fields themselves, which
          are dominated by the seasonal cycle.
          Default: the four panels SST, |grad SST|, SSS, |grad SSS|.
    fps : frames per second
    max_frames : hard cap, the stride is raised if needed

    Every panel keeps a colour scale fixed over the whole run, so what moves
    is the ocean and not the normalisation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    SPEC = {
        "T":     ("T",   "SST",           "degC",    "RdYlBu_r", False),
        "S":     ("S",   "SSS",           "psu",     "BrBG_r",   False),
        "SSH":   ("SSH", "SSH",           "m",       "RdBu_r",   True),
        "ZETA":  ("ZETA", "Vorticity",    "s^-1",    "RdBu_r",   True),
        "GRADT": ("T",   "|grad SST|",    "degC/km", "afmhot_r", False),
        "GRADS": ("S",   "|grad SSS|",    "psu/km",  "afmhot_r", False),
    }

    names = [v.strip().upper() for v in str(var).split(",") if v.strip()]
    names = [n for n in names if n in SPEC] or ["T"]

    nt = np.asarray(run["T"]).shape[0]
    every = max(1, int(every))
    if len(range(0, nt, every)) > max_frames:
        every = int(np.ceil(nt / max_frames))
        print(f"  [gif] stride raised to {every} days to stay under "
              f"{max_frames} frames")
    idx = list(range(0, nt, every))

    panels = []
    for n in names:
        field_key, label, unit, cmap, diverging = SPEC[n]
        if field_key not in run:
            print(f"  [gif] field {field_key} missing, panel {n} skipped")
            continue
        A = np.asarray(run[field_key])
        if n.startswith("GRAD"):
            gy, gx = np.gradient(A, axis=(1, 2))
            A = np.sqrt(gx ** 2 + gy ** 2) / DX_KM
        vmin, vmax = np.percentile(A[idx], [1, 99])
        if diverging:
            m = max(abs(vmin), abs(vmax)); vmin, vmax = -m, m
        panels.append({"A": A, "label": label, "unit": unit, "cmap": cmap,
                       "vmin": float(vmin), "vmax": float(vmax)})

    if not panels:
        print("  [gif] nothing to animate")
        return None

    ncol = len(panels)
    nx, ny = panels[0]["A"].shape[1], panels[0]["A"].shape[2]
    fig, axes = plt.subplots(1, ncol, figsize=(3.1 * ncol + 0.6, 5.4),
                             facecolor="white")
    axes = np.atleast_1d(axes)

    ims = []
    for ax, pan in zip(axes, panels):
        im = ax.imshow(pan["A"][idx[0]].T, origin="lower", cmap=pan["cmap"],
                       vmin=pan["vmin"], vmax=pan["vmax"], aspect="auto",
                       extent=[0, nx * DX_KM, 0, ny * DX_KM])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(pan["unit"], fontsize=8)
        ax.set_title(pan["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("x (km)", fontsize=8)
        ax.tick_params(labelsize=7)
        ims.append(im)
    axes[0].set_ylabel("y (km)", fontsize=8)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    sup = fig.suptitle("", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def _draw(k):
        t = idx[k]
        for im, pan in zip(ims, panels):
            im.set_data(pan["A"][t].T)
        sup.set_text(f"Nature run, day {t} of {nt}")
        return tuple(ims) + (sup,)

    anim = FuncAnimation(fig, _draw, frames=len(idx),
                         interval=1000 / max(1, fps), blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps),
              savefig_kwargs={"facecolor": "white"})
    plt.close()
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Nature run GIF -> {out_path}  ({len(idx)} frames, "
          f"1 every {every} d, {ncol} panel{'s' if ncol > 1 else ''}, "
          f"{size_mb:.1f} MB)")
    return str(out_path)


def plot_nature_run(run, out_path="outputs/ocean_nature_run.png", S_arr=None):
    """
    Diagnostic figure for the nature run.
    Accepts either the dict returned by generate_full(), or the legacy
    signature plot_nature_run(T_arr, S_arr, out_path).
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

    if not isinstance(run, dict):                      # backward compatibility
        run = {"T": run, "S": S_arr, "dx_m": DX_KM*1e3, "dt_days": DT_DAYS}

    T_arr, S_arr = run["T"], run["S"]
    nt, NX_, NY_ = T_arr.shape
    dxkm = run.get("dx_m", DX_KM*1e3) / 1e3
    ext  = [0, NX_*dxkm, 0, NY_*dxkm]

    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    sal_cmap = LinearSegmentedColormap.from_list("sal",
        ["#003c30","#01665e","#35978f","#80cdc1","#f5f5f5",
         "#dfc27d","#bf812d","#8c510a","#543005"], N=256)

    BG, PANEL, EDGE = "white", "white", "#bbbbbb"
    fig = plt.figure(figsize=(21, 17), facecolor=BG)
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.34, wspace=0.30,
                            left=0.05, right=0.96, top=0.93, bottom=0.05)

    def frame(ax, title):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(EDGE)
        ax.set_title(title, color="black", fontsize=9.5, pad=6, fontweight="bold")
        ax.tick_params(colors="black", labelsize=6)

    def styled(ax, title, im, label):
        ax.set_xticks([]); ax.set_yticks([])
        frame(ax, title)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(label, color="black", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="black", labelcolor="black", labelsize=6)

    def show(ax, F, **kw):
        return ax.imshow(F.T, origin="lower", aspect="auto", extent=ext, **kw)

    vT = (np.percentile(T_arr, 0.5), np.percentile(T_arr, 99.5))
    vS = (np.percentile(S_arr, 0.5), np.percentile(S_arr, 99.5))
    snaps = [0, nt//3, 2*nt//3]

    # -- Row 1: SST at three times + variability --------------------------------
    for col, t in enumerate(snaps):
        ax = fig.add_subplot(gs[0, col])
        im = show(ax, T_arr[t], cmap=ocean_cmap, vmin=vT[0], vmax=vT[1])
        styled(ax, f"SST -- day {t}", im, "°C")

    ax = fig.add_subplot(gs[0, 3])
    im = show(ax, T_arr.std(axis=0), cmap="plasma")
    styled(ax, "SST variability (temporal sigma)", im, "°C")

    # -- Row 2: dynamics --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    if "SSH" in run:
        H = run["SSH"][snaps[1]]
        im = show(ax, H, cmap="RdYlBu_r")
        xs = np.linspace(0, NX_*dxkm, NX_); ys = np.linspace(0, NY_*dxkm, NY_)
        ax.contour(xs, ys, H.T, levels=14, colors="#101820", linewidths=0.5, alpha=0.55)
        styled(ax, f"SSH + geostrophic streamlines -- day {snaps[1]}", im, "m")
    else:
        im = show(ax, S_arr[0], cmap=sal_cmap, vmin=vS[0], vmax=vS[1])
        styled(ax, "SSS -- day 0", im, "psu")

    ax = fig.add_subplot(gs[1, 1])
    if "ZETA" in run:
        Ro = run["ZETA"][snaps[1]] / run["f0"]
        lim = np.percentile(np.abs(Ro), 99)
        im = show(ax, Ro, cmap="RdBu_r", vmin=-lim, vmax=lim)
        styled(ax, "Relative vorticity zeta/f (Rossby number)", im, "ζ/f")
    else:
        im = show(ax, T_arr.mean(0), cmap=ocean_cmap)
        styled(ax, "Mean SST", im, "°C")

    ax = fig.add_subplot(gs[1, 2])
    gx, gy = np.gradient(T_arr[snaps[1]], dxkm*1e3*1e-2, dxkm*1e3*1e-2)  # °C/100km
    gmag = np.sqrt(gx**2 + gy**2)
    im = show(ax, gmag, cmap="hot", vmin=0, vmax=np.percentile(gmag, 99))
    styled(ax, f"|grad SST| -- day {snaps[1]}  (fronts & filaments)", im, "°C/100 km")

    ax = fig.add_subplot(gs[1, 3])
    im = show(ax, S_arr[snaps[1]], cmap=sal_cmap, vmin=vS[0], vmax=vS[1])
    styled(ax, f"SSS -- day {snaps[1]}", im, "psu")

    # -- Row 3: statistical diagnostics -----------------------------------------
    ax = fig.add_subplot(gs[2, 0]); frame(ax, "SST radial spectrum")
    A = T_arr[snaps[1]] - T_arr[snaps[1]].mean()
    P = np.abs(fft2(A))**2
    FX, FY = np.meshgrid(fftfreq(NX_, dxkm), fftfreq(NY_, dxkm), indexing="ij")
    Kr = np.sqrt(FX**2 + FY**2).ravel(); Pr = P.ravel()
    kb = np.logspace(np.log10(1/(NX_*dxkm/2)), np.log10(0.5/dxkm), 26)
    kc, Pb = [], []
    for i in range(len(kb)-1):
        m = (Kr >= kb[i]) & (Kr < kb[i+1])
        if m.sum() > 3:
            kc.append(0.5*(kb[i]+kb[i+1])); Pb.append(Pr[m].mean())
    kc = np.array(kc); Pb = np.array(Pb)
    ax.loglog(kc, Pb, color="#6baed6", lw=2.2)
    kk = np.array([kc[3], kc[-4]])
    for sl, st, c in [(-3, "k$^{-3}$ (QG)", "white"), (-2, "k$^{-2}$ (SQG)", "#ffd93d")]:
        ax.loglog(kk, Pb[3]*(kk/kk[0])**sl, "--", lw=1.1, color=c, alpha=0.75, label=st)
    for L, lbl in [(RD_KM, "$R_d$"), (2*dxkm, "2Δx")]:
        ax.axvline(1/L, color="#ff6b6b", lw=0.8, ls=":", alpha=0.8)
        ax.text(1/L, Pb.max(), f" {lbl}", color="#ff6b6b", fontsize=6, va="top")
    ax.set_xlabel("k (cycles/km)", color="black", fontsize=7)
    ax.legend(fontsize=7, labelcolor="black", facecolor=BG, edgecolor=EDGE)
    ax.grid(True, alpha=0.3, color="0.7", which="both")

    ax = fig.add_subplot(gs[2, 1]); frame(ax, "Spatial autocorrelation (zonal)")
    for F, c, lbl in [(T_arr[snaps[1]], "#fc8d59", "SST"), (S_arr[snaps[1]], "#6baed6", "SSS")]:
        Aa = F - F.mean(axis=0, keepdims=True)
        Fh = np.fft.rfft(Aa, axis=0)
        ac = np.fft.irfft((Fh*np.conj(Fh)).real, axis=0).mean(axis=1)
        ac /= ac[0] + 1e-12
        lags = np.arange(NX_//2) * dxkm
        ax.plot(lags, ac[:NX_//2], color=c, lw=1.8, label=lbl)
    ax.axhline(1/np.e, color="black", ls="--", lw=0.8, alpha=0.6)
    ax.text(lags[-1], 1/np.e, " 1/e", color="black", fontsize=6, va="bottom", ha="right")
    ax.axvline(RD_KM, color="#ff6b6b", ls=":", lw=0.9)
    ax.set_xlabel("distance (km)", color="black", fontsize=7)
    ax.legend(fontsize=7, labelcolor="black", facecolor=BG, edgecolor=EDGE)
    ax.grid(True, alpha=0.3, color="0.7")

    ax = fig.add_subplot(gs[2, 2]); frame(ax, "Temporal autocorrelation (anomalies)")
    nlag = min(90, nt//3)
    for F, c, lbl in [(T_arr, "#fc8d59", "SST"), (S_arr, "#6baed6", "SSS")]:
        for des, ls, sfx in [(False, "-", ""), (True, "--", " (deseasoned)")]:
            Aa = F - F.mean(axis=0, keepdims=True)
            if des:
                Aa = Aa - Aa.mean(axis=(1, 2), keepdims=True)
            Aa = Aa / (Aa.std(axis=0, keepdims=True) + 1e-9)
            ac = np.array([(Aa[:nt-l]*Aa[l:]).mean() for l in range(nlag)])
            ax.plot(np.arange(nlag)*run.get("dt_days", 1.0), ac,
                    color=c, lw=1.6, ls=ls, alpha=1.0 if not des else 0.8,
                    label=lbl + sfx)
    ax.axhline(1/np.e, color="black", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("lag (days)", color="black", fontsize=7)
    ax.legend(fontsize=6, labelcolor="black", facecolor=BG, edgecolor=EDGE)
    ax.grid(True, alpha=0.3, color="0.7")

    ax = fig.add_subplot(gs[2, 3]); frame(ax, "T-S diagram + sigma$_0$ isopycnals")
    sub = (slice(None, None, max(1, nt//60)), slice(None, 4), slice(None, 4))
    tt = T_arr[::max(1, nt//60), ::4, ::4].ravel()
    ss = S_arr[::max(1, nt//60), ::4, ::4].ravel()
    Tg = np.linspace(tt.min()-0.4, tt.max()+0.4, 120)
    Sg = np.linspace(ss.min()-0.04, ss.max()+0.04, 120)
    TT, SS = np.meshgrid(Tg, Sg, indexing="ij")
    ax.hist2d(ss, tt, bins=110, cmap="YlOrRd", cmin=1,
              range=[[Sg[0], Sg[-1]], [Tg[0], Tg[-1]]])
    cs = ax.contour(Sg, Tg, sigma0(TT, SS), levels=12, colors="#9fc4e8",
                    linewidths=0.7, alpha=0.85)
    ax.clabel(cs, fontsize=5.5, colors="#cfe3f5", fmt="%.1f")
    ax.set_xlabel("SSS (psu)", color="black", fontsize=7)
    ax.set_ylabel("SST (°C)", color="black", fontsize=7)

    # -- Row 4: time series, distributions, correlation, network ----------------
    ax = fig.add_subplot(gs[3, 0]); frame(ax, "SST time series -- 3 points + domain mean")
    days = np.arange(nt) * run.get("dt_days", 1.0)
    for (x, y, c, lbl) in [(NX_//5, NY_//4, "#ff6b6b", "south"),
                           (NX_//2, NY_//2, "#ffd93d", "jet"),
                           (4*NX_//5, 4*NY_//5, "#6bcb77", "north")]:
        ax.plot(days, T_arr[:, x, y], color=c, lw=1.0, alpha=0.9, label=lbl)
    ax.plot(days, T_arr.mean(axis=(1, 2)), color="black", lw=2.0, label="domain mean")
    ax.set_xlabel("days", color="black", fontsize=7)
    ax.legend(fontsize=6.5, labelcolor="black", facecolor=BG, edgecolor=EDGE, ncol=2)
    ax.grid(True, alpha=0.3, color="0.7")

    ax = fig.add_subplot(gs[3, 1]); frame(ax, "SST / SSS distributions")
    ax.hist(T_arr.ravel(), bins=70, color="#fc8d59", alpha=0.75, density=True, label="SST")
    ax2t = ax.twinx(); ax2t.set_facecolor(PANEL)
    ax2t.hist(S_arr.ravel(), bins=70, color="#6baed6", alpha=0.55, density=True, label="SSS")
    ax2t.tick_params(colors="#6baed6", labelsize=6)
    ax.set_xlabel("°C  /  psu", color="black", fontsize=7)
    ax.legend(fontsize=7, labelcolor="black", facecolor=BG, edgecolor=EDGE, loc="upper left")
    ax2t.legend(fontsize=7, labelcolor="black", facecolor=BG, edgecolor=EDGE, loc="upper right")
    ax.grid(True, alpha=0.3, color="0.7")

    ax = fig.add_subplot(gs[3, 2])
    Ta = T_arr - T_arr.mean(0); Sa = S_arr - S_arr.mean(0)
    corr = (Ta*Sa).mean(0) / (T_arr.std(0)*S_arr.std(0) + 1e-9)
    im = show(ax, corr, cmap="RdBu_r", vmin=-1, vmax=1)
    styled(ax, "T-S correlation of anomalies\n(the fresh plume decouples T and S in the north)", im, "ρ")

    ax = fig.add_subplot(gs[3, 3])
    im = show(ax, T_arr[0], cmap=ocean_cmap, vmin=vT[0], vmax=vT[1])
    rng = np.random.default_rng(99)
    bx = rng.integers(0, NX_, N_BUOYS)*dxkm; by = rng.integers(0, NY_, N_BUOYS)*dxkm
    ax.scatter(bx, by, c="white", s=34, edgecolors="black", linewidths=0.7, zorder=5)
    ax.scatter(bx, by, c="#ffd93d", s=9, zorder=6)
    styled(ax, f"SST + buoy network (N={N_BUOYS})", im, "°C")

    fig.text(0.5, 0.975,
             "2D+T Nature Run -- tracers advected by a geostrophic flow "
             "(double gyre + meandering jet + eddies)",
             ha="center", color="black", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.955,
             f"Domain {NX_*dxkm:.0f} x {NY_*dxkm:.0f} km  |  dx = {dxkm:.0f} km  |  "
             f"{nt} days  |  lat {run.get('lat0', LAT0):.0f}N  |  "
             f"$R_d$ = {RD_KM:.0f} km",
             ha="center", color="#8fb3d9", fontsize=9.5)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  Nature run figure -> {out_path}")


# =============================================================================
#  CLI
# =============================================================================
if __name__ == "__main__":
    import argparse, time
    p = argparse.ArgumentParser()
    p.add_argument("--nt",   type=int, default=NT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out",  type=str, default="outputs/ocean_nature_run.png")
    p.add_argument("--gif",  action="store_true",
                   help="also write an animated GIF of the run")
    p.add_argument("--gif_every", type=int, default=5,
                   help="one frame every N days (default 5)")
    p.add_argument("--gif_var", type=str, default="T,GRADT,S,GRADS",
                   help="fields to animate, comma separated. Available: "
                        "T, S, SSH, ZETA, GRADT, GRADS. Default is the four "
                        "panels SST, |grad SST|, SSS, |grad SSS|")
    p.add_argument("--gif_fps", type=int, default=8)
    p.add_argument("--gif_max", type=int, default=120,
                   help="cap on the number of frames")
    args = p.parse_args()

    t0 = time.time()
    gen = SyntheticOceanGenerator()
    run = gen.generate_full(nt=args.nt, seed=args.seed)
    print(f"  Generated in {time.time()-t0:.1f} s")

    print(f"  SST : {run['T'].shape}  [{run['T'].min():.1f}, {run['T'].max():.1f}] °C")
    print(f"  SSS : {run['S'].shape}  [{run['S'].min():.2f}, {run['S'].max():.2f}] psu")
    print(f"  SSH : {run['SSH'].shape}  [{run['SSH'].min():.2f}, {run['SSH'].max():.2f}] m")
    print("\n  -- Diagnostics --")
    for k, v in gen.diagnostics().items():
        print(f"    {k:<22} {v: .4f}")
    plot_nature_run(run, out_path=args.out)
    if args.gif:
        animate_nature_run(run,
                           out_path=str(Path(args.out).with_suffix(".gif")),
                           every=args.gif_every, var=args.gif_var,
                           fps=args.gif_fps, max_frames=args.gif_max)
