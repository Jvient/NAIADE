"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      BRICK 4 - Differentiable sensor placement (Gumbel-Softmax / Concrete)   ║
║                                                                              ║
║  Reformulates network design as a CONTINUOUS optimisation problem instead    ║
║  of a sequential MDP. The observation operator H is a Bernoulli random       ║
║  field whose logits l are learned by gradient descent:                       ║
║                                                                              ║
║      min_l   -EVF(m)  +  alpha * relu( sum(p) - N_budget )                   ║
║                       +  lambda_sep * p^T Conflict p                         ║
║                                                                              ║
║      m ~ BinaryConcrete(l, tau)      (differentiable relaxation)             ║
║                                                                              ║
║  EVF is the SAME explained-variance / BLUE criterion as Brick 3, so the      ║
║  two optimisers are directly comparable on one Pareto front.                 ║
║                                                                              ║
║  After Chapron, Fablet & Stephan (2026), arXiv:2604.22511, adapted to:       ║
║      - a bivariate SST+SSS observation vector (one buoy = 2 observations)    ║
║      - a shrunk covariance (mesoscale decorrelation ~12 d, see Brick 3)      ║
║      - operational constraints absent from the paper: minimum separation     ║
║                                                                              ║
║  Usage:                                                                      ║
║    python 04_gumbel.py --train --n_budget 23                                 ║
║    python 04_gumbel.py --sweep --report                                      ║
║    python 04_gumbel.py --train --n_budget 23 --loss oi_mse --learn_L         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, argparse, importlib, json, time
from pathlib import Path

import numpy as np
import torch
from datetime import datetime
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data.dataset import SyntheticOceanGenerator, mesoscale_anomaly

# Brick 3 module: the file name starts with a digit, so import it explicitly.
_rl = importlib.import_module("03_rl")
OceanNetworkEnv = _rl.OceanNetworkEnv
_greedy_sequence = _rl._greedy_sequence
_policy_sequence = _rl._policy_sequence
ActorCritic = _rl.ActorCritic


# ══════════════════════════════════════════════════════════════════════════════
#  1.  DIFFERENTIABLE INFORMATION CRITERION
# ══════════════════════════════════════════════════════════════════════════════

class DiffEVF(nn.Module):
    """
    Explained-variance fraction, differentiable w.r.t. a SOFT observation mask.

    Brick 3 evaluates a *binary* network by sub-selecting the active rows:

        EVF = sum_c  C_cO (C_OO + R)^-1 C_Oc  /  sum_c C_cc

    Sub-selection is not differentiable. The equivalent smooth formulation
    weights the observation operator instead of slicing it. With W = diag(w),
    w = [m, m] (a buoy carries one SST *and* one SSS observation):

        G(m) = W C_OO W + R            C(m) = W C_OY
        EVF  = tr( G^-1 C C^T ) / var_total

    At m in {0,1} this is EXACTLY the Brick 3 value: for an inactive index i,
    row and column i of W C_OO W vanish, C_i = 0 and G_ii = R_i > 0, so the
    entry contributes nothing and G stays invertible. In between, a mask value
    of 0.5 behaves like an observation with inflated noise, which is precisely
    the relaxation we want the gradient to see.

    P = C_OY C_OY^T is precomputed once (2K x 2K), which removes the evaluation
    grid (2M ~ 1200 columns) from the inner loop entirely. It is then factored
    as P = Z Z^T with Z (2K, r) from a truncated eigendecomposition, so the
    trace collapses to a squared Frobenius norm after ONE triangular solve:

        EVF = || L^-1 W Z ||_F^2 / var_total ,        G = L L^T

    That is O(K^2 r) instead of O(K^3) per mask -- an order of magnitude on the
    inner loop, which matters because the mask expectation is a Monte-Carlo
    average over several draws at every gradient step.
    """

    def __init__(self, env, device="cpu", dtype=torch.float32, energy_tol=1e-6,
                 jitter=1e-9):
        super().__init__()
        C_OO = np.asarray(env._C_OO, dtype=np.float64)
        C_OY = np.asarray(env._C_OY, dtype=np.float64)
        P = C_OY @ C_OY.T                      # (2K, 2K), accumulated in float64

        # P is symmetric PSD: keep the leading modes carrying 1 - energy_tol
        # of its trace. The nature-run covariance is smooth, so r << 2K.
        ev, V = np.linalg.eigh(P)
        ev = np.clip(ev[::-1], 0.0, None); V = V[:, ::-1]
        keep = np.searchsorted(np.cumsum(ev) / ev.sum(), 1.0 - energy_tol) + 1
        r = int(min(keep, len(ev)))
        Z = V[:, :r] * np.sqrt(ev[:r])

        self.K = env.K
        self.rank = r
        self.energy_kept = float(np.cumsum(ev)[r - 1] / ev.sum())
        self.register_buffer("C_OO", torch.tensor(C_OO, dtype=dtype, device=device))
        self.register_buffer("Z", torch.tensor(Z, dtype=dtype, device=device))
        self.register_buffer("R", torch.tensor(np.asarray(env._R_diag),
                                               dtype=dtype, device=device))
        self.var_total = float(env._var_total)
        self.dtype_ = dtype
        self.device_ = device
        self.jitter = jitter

    def forward(self, m):
        """m: (K,) or (B, K) in [0, 1]  ->  EVF: () or (B,)"""
        single = (m.dim() == 1)
        if single:
            m = m[None]
        w = torch.cat([m, m], dim=-1)                       # (B, 2K)
        G = self.C_OO[None] * (w[:, :, None] * w[:, None, :]) \
            + torch.diag_embed((self.R + self.jitter)[None].expand(w.shape[0], -1))
        L = torch.linalg.cholesky(G)
        WZ = w[:, :, None] * self.Z[None]                   # (B, 2K, r)
        Y = torch.linalg.solve_triangular(L, WZ, upper=False)
        evf = (Y ** 2).sum(dim=(-2, -1)) / self.var_total
        return evf[0] if single else evf


class DiffOIMSE(nn.Module):
    """
    Alternative loss, closer to the letter of the paper: the reconstruction
    error of the ACTUAL nature-run snapshots by optimal interpolation, where
    the OI covariance is the parametric model

        C[(i,v),(j,w)] = sigma_v(i) sigma_w(j) exp(-d_ij^2 / 2 L^2) c_vw

    and L (the correlation length) is a LEARNED parameter, optimised jointly
    with the mask. This is the one place where 'learning theta' is meaningful:
    the loss is measured against the true field, not against the same modelled
    covariance that produced the gain, so maximising over L is not circular.

    Statistics are fitted on the first half of the record, the loss is reported
    on the second half.
    """

    def __init__(self, env, device="cpu", dtype=torch.float32, learn_L=True,
                 L_lo_px=None, L_hi_px=None, L0_px=None):
        super().__init__()
        st = env.eval_stride
        Ta = mesoscale_anomaly(env.T) / (env.T.std() + 1e-9)
        Sa = mesoscale_anomaly(env.S) / (env.S.std() + 1e-9)
        nt = len(Ta)
        yT = Ta[:, ::st, ::st].reshape(nt, -1)
        yS = Sa[:, ::st, ::st].reshape(nt, -1)
        oT = np.stack([Ta[:, x, y] for (x, y) in env.candidate_positions], 1)
        oS = np.stack([Sa[:, x, y] for (x, y) in env.candidate_positions], 1)
        Y = np.concatenate([yT, yS], 1); O = np.concatenate([oT, oS], 1)
        Y = Y - Y.mean(0); O = O - O.mean(0)
        h = nt // 2
        tr, va = slice(0, h), slice(h, nt)

        cnd = np.array(env.candidate_positions, dtype=np.float64)
        cell = env._eval_xy.astype(np.float64)
        d2_oo = ((cnd[:, None, 0] - cnd[None, :, 0]) ** 2
                 + (cnd[:, None, 1] - cnd[None, :, 1]) ** 2)
        d2_oc = ((cnd[:, None, 0] - cell[None, :, 0]) ** 2
                 + (cnd[:, None, 1] - cell[None, :, 1]) ** 2)

        self.K = env.K
        reg = lambda n, a: self.register_buffer(
            n, torch.tensor(np.asarray(a), dtype=dtype, device=device))
        reg("d2_oo", d2_oo); reg("d2_oc", d2_oc)
        reg("sT_o", oT[tr].std(0)); reg("sS_o", oS[tr].std(0))
        reg("sT_c", yT[tr].std(0)); reg("sS_c", yS[tr].std(0))
        reg("O_va", O[va]); reg("Y_va", Y[va])
        reg("R", np.asarray(env._R_diag))
        self.rTS = float(np.clip(np.mean([
            np.corrcoef(oT[tr, k], oS[tr, k])[0, 1] for k in range(env.K)]), -1, 1))
        self.var_total = float((np.asarray(Y[va]) ** 2).mean(0).sum())

        # Correlation length, BOUNDED.
        #
        # An unconstrained log_L runs away: measured 572 km against a 90 km
        # prior. A single Gaussian kernel cannot represent both the mesoscale
        # (~90 km) and the residual large-scale mode, so the optimiser trades
        # the former for the latter and flattens the kernel towards Roo -> 1,
        # which is degenerate: every candidate looks equally informative and
        # the placement stops being mesoscale-driven. Bounding L to a physically
        # defensible bracket around the diagnosed decorrelation scale keeps the
        # gradient honest. Widen it deliberately if you want to test the
        # runaway, do not leave it unbounded by accident.
        L0 = float(L0_px if L0_px else env.influence_px)
        self.L_lo = float(L_lo_px if L_lo_px else 0.25 * L0)
        self.L_hi = float(L_hi_px if L_hi_px else 3.00 * L0)
        L0 = float(np.clip(L0, self.L_lo + 1e-6, self.L_hi - 1e-6))
        z0 = float(np.log((L0 - self.L_lo) / (self.L_hi - L0)))
        self.raw_L = nn.Parameter(torch.tensor(z0, dtype=dtype, device=device),
                                  requires_grad=bool(learn_L))

        # Nugget (observation-error inflation).
        #
        # Without it this operator is unusable and quietly so: the instrumental
        # noise is ~3e-4 in normalised units, so G = W C_OO W + R is very close
        # to singular for a smooth Gaussian kernel, and inverting it amplifies
        # sampling noise until the out-of-sample reconstruction is WORSE than
        # climatology (measured explained variance ~ -0.01). This is the same
        # pathology Brick 3 handles with EVF_SHRINKAGE = 0.9; the parametric
        # operator here needs its own regulariser. A nugget is the standard OI
        # answer and is exactly the kind of parameter the reference paper means
        # by "jointly optimising the reconstruction parameters".
        self.nug_lo, self.nug_hi = 1e-4, 1.0
        n0 = 0.05
        w0 = float(np.log((n0 - self.nug_lo) / (self.nug_hi - n0)))
        self.raw_nug = nn.Parameter(torch.tensor(w0, dtype=dtype, device=device),
                                    requires_grad=True)

    def nugget(self):
        return self.nug_lo + (self.nug_hi - self.nug_lo) * torch.sigmoid(self.raw_nug)

    def L_px(self):
        """Correlation length in pixels, squashed into [L_lo, L_hi]."""
        return self.L_lo + (self.L_hi - self.L_lo) * torch.sigmoid(self.raw_L)

    def _blocks(self):
        L2 = 2.0 * self.L_px() ** 2
        Roo = torch.exp(-self.d2_oo / L2)
        Roc = torch.exp(-self.d2_oc / L2)
        b = lambda Rm, sa, sb, x: (sa[:, None] * sb[None, :]) * Rm * (self.rTS if x else 1.0)
        C_OO = torch.cat([
            torch.cat([b(Roo, self.sT_o, self.sT_o, 0), b(Roo, self.sT_o, self.sS_o, 1)], 1),
            torch.cat([b(Roo, self.sS_o, self.sT_o, 1), b(Roo, self.sS_o, self.sS_o, 0)], 1)], 0)
        C_OY = torch.cat([
            torch.cat([b(Roc, self.sT_o, self.sT_c, 0), b(Roc, self.sT_o, self.sS_c, 1)], 1),
            torch.cat([b(Roc, self.sS_o, self.sT_c, 1), b(Roc, self.sS_o, self.sS_c, 0)], 1)], 0)
        return C_OO, C_OY

    def forward(self, m):
        """Returns the explained variance measured OUT OF SAMPLE (higher = better)."""
        single = (m.dim() == 1)
        if single:
            m = m[None]
        C_OO, C_OY = self._blocks()
        w = torch.cat([m, m], dim=-1)
        W2 = w[:, :, None] * w[:, None, :]
        G = C_OO[None] * W2 + torch.diag_embed(
            (self.R + self.nugget())[None].expand(w.shape[0], -1))
        C = w[:, :, None] * C_OY[None]
        B = torch.linalg.solve(G, C)                       # (B, 2K, 2M)
        pred = torch.einsum("ti,bij->btj", self.O_va, B)
        resid = self.Y_va[None] - pred
        mse = (resid ** 2).mean(1).sum(-1)
        evf = 1.0 - mse / self.var_total
        return evf[0] if single else evf


# ══════════════════════════════════════════════════════════════════════════════
#  2.  LEARNED LOGIT FIELD
# ══════════════════════════════════════════════════════════════════════════════

class LogitField(nn.Module):
    """
    One learnable logit per candidate position. Sampling uses the Binary
    Concrete relaxation (Maddison 2016), i.e. the two-category Gumbel-Softmax:

        m = sigmoid( (l + logit(u)) / tau ),   u ~ U(0,1)

    tau -> 0 recovers a hard Bernoulli draw; tau large gives a smooth,
    exploratory mask. Annealing tau during training is what lets the optimiser
    explore first and commit to a discrete, budget-feasible design later.
    """

    def __init__(self, K, p_init=0.1, seed=0, device="cpu"):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        l0 = float(np.log(p_init / (1 - p_init)))
        self.logits = nn.Parameter(
            l0 + 0.05 * torch.randn(K, generator=g).to(device))
        # Dedicated RNG for the Gumbel draws. Using the GLOBAL torch RNG makes
        # the result depend on how many random numbers were consumed earlier in
        # the process: the same budget then scores differently in a standalone
        # --train and as the 5th entry of a --sweep. Since replayability from
        # (seed_ocean, seed_buoys) is the contract of this repo, the sampler
        # gets its own stream.
        self._gen = torch.Generator(
            device=self.logits.device).manual_seed(seed + 1)

    def probs(self):
        return torch.sigmoid(self.logits)

    def sample(self, n_mc, tau):
        u = torch.rand(n_mc, self.logits.shape[0], generator=self._gen,
                       device=self.logits.device).clamp(1e-6, 1 - 1e-6)
        noise = torch.log(u) - torch.log1p(-u)
        return torch.sigmoid((self.logits[None] + noise) / tau)


def harden(p, env, n):
    """
    Turn a probability field into a deployable network: take the n highest
    probabilities, skipping any candidate that violates the minimum separation.

    The paper makes the same point in its discussion - training is stochastic,
    deployment need not be. For moorings (PIRATA, MOOSE) this deterministic
    top-budget mask is the object of interest.
    """
    order = np.argsort(-np.asarray(p))
    sel = []
    for c in order:
        if len(sel) >= n:
            break
        if not sel or not env._conflict[c, sel].any():
            sel.append(int(c))
    return np.array(sel, dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
#  3.  OPTIMISATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def optimize_gumbel(env, scorer, n_budget, args, verbose=True):
    """
    Gradient-based placement at a fixed sensor budget.

    Schedule, following the paper:
      - warm-up: alpha = 0 for the first `warmup` fraction, the mask explores
        the whole domain before the budget starts to bite;
      - alpha then ramps up linearly until the expected count sits at N_budget;
      - tau is annealed geometrically from tau0 to tau1 so the sampled masks
        converge towards crisp binary patterns.
    """
    dev, _dt = scorer_device_dtype(scorer)
    field = LogitField(env.K, p_init=min(0.5, max(0.02, n_budget / env.K)),
                       seed=args.seed_buoys, device=dev).to(dev).to(_dt)
    conflict = torch.tensor(env._conflict, dtype=_dt, device=dev)

    params = list(field.parameters())
    extra = [p for p in scorer.parameters() if p.requires_grad]
    opt = torch.optim.Adam([{"params": params, "lr": args.lr}]
                           + ([{"params": extra, "lr": args.lr_L}] if extra else []))

    hist = {"iter": [], "evf": [], "n_exp": [], "tau": [], "sep": [], "L": [],
            "hard_it": [], "hard": []}
    n_it = args.iters
    # The quantity that matters is the score of the DEPLOYED network, not of
    # the relaxed mask. Harden and evaluate periodically with the exact Brick 3
    # criterion, and keep the best: the last iterate is not always the best one
    # once the temperature is low and the gradient noisy.
    best = {"evf": -np.inf, "p": None, "idx": None, "it": -1}
    eval_every = max(1, n_it // 40)

    # With --loss evf the scorer IS Brick 3's criterion, so the numpy version is
    # used (it is exact and cheap). With --loss oi_mse they differ and the
    # scorer's own value is the one that matters for selection.
    _own = (args.loss != "evf")

    def _hard_score(idx):
        if not _own:
            return float(env.explained_variance(idx))
        with torch.no_grad():
            mb = torch.zeros(env.K, device=dev, dtype=_dt)
            mb[torch.as_tensor(np.asarray(idx), dtype=torch.long, device=dev)] = 1.0
            return float(scorer(mb))
    for it in range(n_it):
        frac = it / max(1, n_it - 1)
        tau = args.tau0 * (args.tau1 / args.tau0) ** frac
        ramp = 0.0 if frac < args.warmup else min(
            1.0, (frac - args.warmup) / max(1e-6, args.ramp))
        alpha = args.alpha * ramp
        lam = args.lam_sep * ramp

        m = field.sample(args.n_mc, tau)
        evf = scorer(m).mean()
        p = field.probs()
        budget = torch.relu(p.sum() - n_budget)
        sep = (p @ conflict @ p) / 2.0

        # Binarisation pressure.
        #
        # Diagnostic behind this term: with only -EVF + relu(sum p - N), once
        # the expected count reaches N the budget gradient vanishes and NOTHING
        # pushes the field towards 0/1. Because EVF saturates with coverage,
        # spreading mass thinly over many sites gives a higher EXPECTED relaxed
        # score than committing to a few: the optimiser happily settles on a
        # flat field (max p ~ 0.7, most cells 0.1-0.2). Hardening then just
        # reads the noisy top of a flat map, which is why the raw gradient
        # solution sat below greedy.
        #
        # sum_i p_i(1-p_i) is the total Bernoulli variance: zero exactly when
        # every probability is 0 or 1. Ramped in with the budget so exploration
        # is untouched during warm-up.
        binz = (p * (1.0 - p)).sum()

        loss = -evf + alpha * budget + lam * sep + args.w_bin * ramp * binz
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if it % eval_every == 0 or it == n_it - 1:
            p_np = p.detach().cpu().numpy()
            idx_it = harden(p_np, env, n_budget)
            # Model selection MUST use the criterion being optimised. Scoring
            # the hardened mask with Brick 3's EVF while the gradient minimises
            # the OI reconstruction error picks the iterate that happens to
            # please a DIFFERENT operator: with --loss oi_mse the "best" network
            # was being found at iteration 200 out of 1000 and the remaining 800
            # steps were discarded, which is exactly backwards.
            v = _hard_score(idx_it) if len(idx_it) else 0.0
            hist["hard_it"].append(it); hist["hard"].append(v)
            if v > best["evf"]:
                best = {"evf": v, "p": p_np.copy(), "idx": idx_it, "it": it}

        if it % max(1, n_it // 60) == 0 or it == n_it - 1:
            hist["iter"].append(it)
            hist["evf"].append(float(evf.detach()))
            hist["n_exp"].append(float(p.sum().detach()))
            hist["tau"].append(tau)
            hist["sep"].append(float(sep.detach()))
            hist["L"].append(float(scorer.L_px().detach())
                             if hasattr(scorer, "L_px") else np.nan)
            if verbose and it % max(1, n_it // 6) == 0:
                extra_s = (f" | L={float(scorer.L_px().detach())*DX_KM:5.0f}km"
                           f" | nugget {float(scorer.nugget().detach()):.4f}"
                           if hasattr(scorer, "L_px") else "")
                print(f"    it {it:5d} | tau {tau:5.3f} | EVF(soft) {float(evf.detach()):.4f} "
                      f"| E[N] {float(p.sum().detach()):6.2f} "
                      f"| conflicts {float(sep.detach()):5.2f}"
                      + extra_s)

    p, idx = best["p"], best["idx"]
    evf_raw = float(env.explained_variance(idx))   # Brick 3 metric, pre-polish
    own_raw = float(best["evf"])                  # metric actually optimised
    n_swaps = 0
    if args.polish and _own:
        print(f"    [note] --polish operates on the Brick 3 criterion and is "
              f"skipped with --loss {args.loss} (it would optimise a different "
              f"objective than the gradient).")
    if args.polish and not _own:
        idx2 = local_swap_polish(env, idx, max_rounds=args.polish)
        n_swaps = int(len(set(map(int, idx2)) - set(map(int, idx))))
        idx = idx2
    if verbose:
        print(f"    best deployed network found at iteration {best['it']}"
              + (f" | polish moved {n_swaps} buoy(s)" if args.polish else ""))
    return {"p": p, "idx": idx, "hist": hist, "best_iter": best["it"],
            "evf_raw": evf_raw, "own_raw": own_raw, "n_swaps": n_swaps,
            "evf_hard": float(env.explained_variance(idx)),
            "L_px": float(scorer.L_px().detach())
                    if hasattr(scorer, "L_px") else float(env.influence_px)}


# ══════════════════════════════════════════════════════════════════════════════
#  4.  BASELINES  (all subject to the same minimum-separation constraint)
# ══════════════════════════════════════════════════════════════════════════════

def bl_random(env, n, n_draw=50, seed=0):
    rng = np.random.default_rng(seed)
    v = [env.explained_variance(env.sample_feasible(n, rng)) for _ in range(n_draw)]
    return np.array(v)


def _topn_feasible(score, env, n):
    return harden(score, env, n)


def bl_variance(env, n):
    return _topn_feasible(env.field_stats, env, n)


def bl_regular(env, n, seed=0):
    """Regular lattice with a random offset, thinned down to n points."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(30):
        idx = env.sample_feasible(env.n_feasible_max, rng)
        if len(idx) < n:
            continue
        pos = np.array([env.candidate_positions[i] for i in idx], float)
        # farthest-point thinning -> quasi-uniform coverage
        sel = [int(rng.integers(len(idx)))]
        d = np.linalg.norm(pos - pos[sel[0]], axis=1)
        while len(sel) < n:
            k = int(np.argmax(d)); sel.append(k)
            d = np.minimum(d, np.linalg.norm(pos - pos[k], axis=1))
        cand = idx[sel]
        v = env.explained_variance(cand)
        if best is None or v > best[0]:
            best = (v, cand)
    return best[1] if best else env.sample_feasible(n, rng)


def bl_pcaqr(env, n, cache={}):
    """
    PCA-QR / QDEIM sparse sensing (Manohar et al. 2018). Column-pivoted QR on
    the leading EOFs restricted to the candidate grid. This is the paper's key
    negative-result baseline: in a strongly non-stationary regime it can do
    WORSE than uniform random placement, because a fixed modal score map is a
    poor proxy for what the reconstruction actually needs.
    """
    if "X" not in cache:
        Ta = mesoscale_anomaly(env.T) / (env.T.std() + 1e-9)
        Sa = mesoscale_anomaly(env.S) / (env.S.std() + 1e-9)
        oT = np.stack([Ta[:, x, y] for (x, y) in env.candidate_positions], 1)
        oS = np.stack([Sa[:, x, y] for (x, y) in env.candidate_positions], 1)
        X = np.concatenate([oT - oT.mean(0), oS - oS.mean(0)], 0).T   # (K, 2nt)
        cache["X"] = X
        cache["U"] = np.linalg.svd(X, full_matrices=False)[0]
    U = cache["U"]
    r = min(n, U.shape[1])
    from scipy.linalg import qr
    _, _, piv = qr(U[:, :r].T, pivoting=True)
    sel = []
    for c in piv:
        if len(sel) >= n:
            break
        if not sel or not env._conflict[c, sel].any():
            sel.append(int(c))
    return np.array(sel, dtype=int)


def local_swap_polish(env, idx, max_rounds=3, verbose=False):
    """
    1-swap local search on the EXACT Brick 3 criterion: repeatedly try moving
    each deployed buoy to the best feasible free candidate, keep the move if it
    raises the explained variance, stop when a full round changes nothing.

    Why this belongs here rather than being a different method: the Gumbel
    field optimises an EXPECTATION over random masks, so its argmax is a good
    but noisy summary of where information sits. The polish is a deterministic
    refinement of that same solution, not a competing placement strategy -- it
    is the discrete counterpart of the last few gradient steps that the
    relaxation cannot take. Report both numbers so the gradient's own
    contribution stays legible.
    """
    idx = list(map(int, idx))
    cur = float(env.explained_variance(np.array(idx, dtype=int)))
    for rnd in range(max_rounds):
        improved = False
        for pos in range(len(idx)):
            rest = idx[:pos] + idx[pos + 1:]
            free = env.feasible_candidates(np.array(rest, dtype=int))
            best_c, best_v = idx[pos], cur
            for c in free:
                v = float(env.explained_variance(
                    np.array(rest + [int(c)], dtype=int)))
                if v > best_v + 1e-12:
                    best_v, best_c = v, int(c)
            if best_c != idx[pos]:
                idx = rest + [best_c]; cur = best_v; improved = True
        if verbose:
            print(f"      polish round {rnd+1}: EVF = {cur:.4f}")
        if not improved:
            break
    return np.array(sorted(idx), dtype=int)


def greedy_on_scorer(env, scorer, n_max, batch=64, device="cpu", verbose=True):
    """
    Greedy maximisation of an ARBITRARY differentiable scorer, evaluated in
    batches of candidate masks.

    This exists to close a gap that invalidates the obvious reading of the
    `oi_mse` results. Saying "the gradient design beats greedy under the learned
    OI operator" is a weak claim if that greedy was built to maximise a
    DIFFERENT criterion (Brick 3's EVF). Each optimiser trivially wins on its
    own metric; the only informative comparison is greedy-on-OI against
    gradient-on-OI, both scored by the OI operator.

    Cost is K x N scorer evaluations, hence the batching -- the scorer accepts
    a (B, K) stack of masks.
    """
    dt = scorer_device_dtype(scorer)[1]
    sel, out = [], {}
    for step in range(min(n_max, env.n_feasible_max)):
        cands = env.feasible_candidates(sel)
        if len(cands) == 0:
            break
        best_c, best_v = None, -np.inf
        for i in range(0, len(cands), batch):
            chunk = cands[i:i + batch]
            M = torch.zeros(len(chunk), env.K, dtype=dt, device=device)
            if sel:
                M[:, torch.as_tensor(sel, dtype=torch.long, device=device)] = 1.0
            M[torch.arange(len(chunk)), torch.as_tensor(
                np.asarray(chunk), dtype=torch.long, device=device)] = 1.0
            with torch.no_grad():
                v = scorer(M).cpu().numpy()
            k = int(np.argmax(v))
            if v[k] > best_v:
                best_v, best_c = float(v[k]), int(chunk[k])
        sel = sel + [best_c]
        out[len(sel)] = (np.array(sel, dtype=int), best_v)
        if verbose and (step + 1) % 5 == 0:
            print(f"      greedy-on-scorer: N={len(sel)}  score={best_v:.4f}")
    return out


def jaccard_chance(env, n, n_draws=400, seed=0):
    """
    Expected Jaccard between two INDEPENDENT feasible networks of size n.

    Without this, an overlap of 0.10 is uninterpretable: it could be near-total
    disagreement or near-chance agreement, and those mean different things. With
    a self-consistency ceiling J_self (same objective, different seeds) and this
    floor J_chance, the natural statistic is

        agreement retained = (J - J_chance) / (J_self - J_chance)

    i.e. what fraction of the recoverable agreement survives the change. It is
    scale-free and does not depend on n or on the size of the candidate grid,
    so it is the number to quote across budgets.
    """
    rng = np.random.default_rng(seed)
    js = []
    for _ in range(n_draws):
        a = set(map(int, env.sample_feasible(n, rng)))
        b = set(map(int, env.sample_feasible(n, rng)))
        js.append(len(a & b) / max(1, len(a | b)))
    return float(np.mean(js))


def bl_greedy_sequence(env, n_max, cache={}):
    """Nested greedy sequence: one pass gives an optimum-quality set for every N."""
    if "seq" not in cache:
        cache["seq"] = _greedy_sequence(env, n_max)
    return cache["seq"]


def equivalent_random_N(env, target_evf, n_max, n_draw=20, seed=0):
    """How many RANDOMLY placed buoys are needed to match a given score?"""
    for n in range(2, n_max + 1):
        if bl_random(env, n, n_draw, seed).mean() >= target_evf:
            return n
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  5.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _frame(ax, title="", xlab="", ylab=""):
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def plot_train(env, res, n_budget, others, out):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    var = env.field_stats.reshape(env.grid_x, env.grid_y)
    P = res["p"].reshape(env.grid_x, env.grid_y)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(P.T, origin="lower", cmap="magma", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046).set_label("p(sensor)", fontsize=8)
    _frame(ax, "Learned sampling probability", "candidate x", "candidate y")

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(var.T, origin="lower", cmap="viridis", aspect="auto")
    gi = np.array([i // env.grid_y for i in res["idx"]])
    gj = np.array([i % env.grid_y for i in res["idx"]])
    ax.scatter(gi, gj, s=45, c="red", edgecolors="white", linewidths=0.8, zorder=3)
    plt.colorbar(im, ax=ax, fraction=0.046).set_label("mesoscale variability", fontsize=8)
    _frame(ax, f"Hardened network (N={len(res['idx'])}), EVF={res['evf_hard']:.4f}",
           "candidate x", "candidate y")

    ax = fig.add_subplot(gs[0, 2])
    h = res["hist"]
    ax.plot(h["iter"], h["evf"], lw=1.4, color="#1b6ca8", label="EVF (soft mask)")
    ax.plot(h["hard_it"], h["hard"], lw=1.4, color="#c0392b",
            label="EVF (deployed network)")
    ax.axhline(res["evf_hard"], ls="--", lw=0.9, color="#c0392b", alpha=.6)
    ax.axvline(res.get("best_iter", -1), ls=":", lw=0.9, color="#555")
    ax.legend(fontsize=8); _frame(ax, "Convergence", "iteration", "explained variance")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(h["iter"], h["n_exp"], lw=1.4, color="#2d6a4f")
    ax.axhline(n_budget, ls="--", lw=1.0, color="k")
    _frame(ax, "Expected sensor count vs budget", "iteration", "E[N]")
    ax2 = ax.twinx(); ax2.plot(h["iter"], h["tau"], lw=1.0, color="#999", ls=":")
    ax2.set_ylabel("tau", fontsize=8, color="#777"); ax2.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(h["iter"], h["sep"], lw=1.4, color="#8e44ad")
    _frame(ax, "Expected separation violations", "iteration", "conflicting pairs")

    ax = fig.add_subplot(gs[1, 2])
    names = list(others.keys()); vals = [others[k] for k in names]
    cols = ["#c0392b" if k.startswith("Gumbel") else "#4a6fa5" for k in names]
    ax.barh(range(len(names)), vals, color=cols, height=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.22)
    _frame(ax, f"Explained variance at N={n_budget}", "EVF", "")

    fig.suptitle(f"NAIADE - Brick 4: differentiable placement (Gumbel-Softmax), "
                 f"N = {n_budget}", fontsize=12, weight="bold")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


def plot_sweep(sweep, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    ns = sweep["n_list"]
    ax.fill_between(ns, sweep["rand_lo"], sweep["rand_hi"], color="#bbb", alpha=.45,
                    label="random (±1σ)")
    ax.plot(ns, sweep["random"], color="#777", lw=1.4, label="random (mean)")
    for k, c, mk in [("greedy", "#2d6a4f", "s"), ("pcaqr", "#b8860b", "^"),
                     ("variance", "#8e44ad", "v"), ("regular", "#4a6fa5", "d"),
                     ("ppo", "#e67e22", "P"), ("gumbel", "#c0392b", "o")]:
        if k in sweep and sweep[k] is not None:
            ax.plot(ns, sweep[k], marker=mk, ms=4.5, lw=1.6, color=c,
                    label=k if k != "pcaqr" else "PCA-QR")
    ax.legend(fontsize=8, loc="lower right")
    _frame(ax, "Pareto front: information vs sensor budget",
           "number of buoys N", "explained variance")

    ax = axes[1]
    base = np.array(sweep["random"])
    for k, c, mk in [("greedy", "#2d6a4f", "s"), ("pcaqr", "#b8860b", "^"),
                     ("variance", "#8e44ad", "v"), ("regular", "#4a6fa5", "d"),
                     ("ppo", "#e67e22", "P"), ("gumbel", "#c0392b", "o")]:
        if k in sweep and sweep[k] is not None:
            ax.plot(ns, np.array(sweep[k]) - base, marker=mk, ms=4.5, lw=1.6,
                    color=c, label=k if k != "pcaqr" else "PCA-QR")
    ax.axhline(0, color="#777", lw=1.2)
    ax.legend(fontsize=8)
    _frame(ax, "Gain over uniform random placement", "number of buoys N",
           "Δ explained variance")
    fig.suptitle("NAIADE - Brick 4: Gumbel-Softmax vs PPO vs classical baselines",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Brick 4 - differentiable placement")
    p.add_argument("--train", action="store_true", help="single budget optimisation")
    p.add_argument("--sweep", action="store_true", help="Pareto sweep over N")
    p.add_argument("--report", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="verify the differentiable EVF against Brick 3")
    p.add_argument("--n_budget", type=int, default=23)
    p.add_argument("--n_list", type=str, default="5,10,15,20,25,30,35,40")
    p.add_argument("--loss", type=str, default="evf",
                   choices=["evf", "oi_mse", "vae"])
    p.add_argument("--learn_L", action="store_true",
                   help="jointly learn the OI correlation length (oi_mse only)")
    p.add_argument("--iters", type=int, default=700)
    p.add_argument("--n_mc", type=int, default=8, help="mask draws per iteration")
    p.add_argument("--tau0", type=float, default=1.0)
    p.add_argument("--tau1", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.05, help="budget penalty weight")
    p.add_argument("--lam_sep", type=float, default=0.02, help="separation penalty")
    p.add_argument("--w_bin", type=float, default=0.004,
                   help="binarisation penalty sum p(1-p); 0 disables it")
    p.add_argument("--polish", type=int, default=0,
                   help="rounds of 1-swap local search on the exact criterion")
    p.add_argument("--warmup", type=float, default=0.15)
    p.add_argument("--ramp", type=float, default=0.35)
    p.add_argument("--lr", type=float, default=0.10)
    p.add_argument("--lr_L", type=float, default=0.02)
    p.add_argument("--n_random", type=int, default=50)
    p.add_argument("--seed_ocean", type=int, default=42)
    p.add_argument("--seed_buoys", type=int, default=7)
    p.add_argument("--nt", type=int, default=NT)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--min_sep", type=int, default=MIN_SEP_CELLS)
    p.add_argument("--eval_stride", type=int, default=8)
    p.add_argument("--evf_cv", type=int, default=0)
    p.add_argument("--influence_km", type=float, default=INFLUENCE_RADIUS_KM)
    p.add_argument("--checkpoint", type=str, default="outputs/rl_best.pt",
                   help="optional PPO policy, added to the sweep for comparison")
    p.add_argument("--cache", action="store_true",
                   help="cache/reuse the nature run on disk")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument("--L_lo_km", type=float, default=None,
                   help="lower bound on the learned correlation length")
    p.add_argument("--L_hi_km", type=float, default=None,
                   help="upper bound on the learned correlation length")
    p.add_argument("--vae_ckpt", type=str, default="figures/vae_best.pt",
                   help="Brick 1 checkpoint, used as a third operator")
    p.add_argument("--vae_probe", action="store_true",
                   help="score vs budget on random networks: checks the "
                        "operator responds to observations at all")
    p.add_argument("--n_probe_draws", type=int, default=3)
    p.add_argument("--vae_raw", action="store_true",
                   help="score the VAE on raw normalised fields instead of the "
                        "domain-mean anomaly (reproduces the seasonal confound)")
    p.add_argument("--vae_times", type=int, default=24,
                   help="time steps the VAE operator scores on")
    p.add_argument("--l_scan", type=str, default=None,
                   help="comma-separated L values in km; decomposes the "
                        "operator effect against correlation length")
    p.add_argument("--ocean_seeds", type=str, default=None,
                   help="comma-separated seed_ocean values: repeats the "
                        "operator-effect measurement on independent oceans")
    p.add_argument("--greedy_batch", type=int, default=32)
    p.add_argument("--n_seeds", type=int, default=3,
                   help="replicates for the --cross noise-floor control")
    p.add_argument("--cross", action="store_true",
                   help="cross-score every design under BOTH reconstruction "
                        "operators (Brick 3 EVF and the learned OI)")
    p.add_argument("--f64", action="store_true",
                   help="float64 criterion (slower, safest on large K)")
    return p.parse_args()


class DiffVAE(nn.Module):
    """
    Brick 1's AE-UNet wrapped as a third reconstruction operator.

    Why this matters. DiffEVF and DiffOIMSE are both linear Gaussian
    reconstructors; the L-scan showed the design difference between them is not
    the kernel width, but a sceptic can still say both live in the same family.
    Brick 1's network is nonlinear, and it already takes the observation mask as
    an input channel with explicit ObsGate/FiLM conditioning -- so it is a
    structurally different operator, not a third kernel.

    Differentiability. The input is x = [T*m, S*m, m] with m the pixel mask, so
    the gradient reaches the mask through both the masked fields and the gate
    conditioning. Nothing in the wrapper detaches it. The network's own weights
    are frozen: this operator is a fixed reconstructor being interrogated, not a
    model being trained.

    Two things to keep honest:
      - dropout is forced OFF here. Brick 1 uses MC-dropout for uncertainty, but
        stochastic dropout inside the objective would add variance the mask
        optimiser cannot distinguish from a real signal.
      - the score is explained variance over UNOBSERVED pixels only, matching
        `_compute_rmse_mc` in Brick 1. Scoring observed pixels would reward
        putting buoys where the answer is already known.
    """

    def __init__(self, env, ckpt_path, device="cpu", dtype=torch.float32,
                 n_times=24, seed=0, deseason=True):
        super().__init__()
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "brick1_ae", Path(__file__).parent / "01_autoencoder.py")
        b1 = _ilu.module_from_spec(spec); spec.loader.exec_module(b1)

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        a = ck["args"]
        net = b1.ObservabilityVAE(base_ch=a["base_ch"], latent_ch=a["latent_ch"],
                                  dropout_p=a.get("dropout_p", 0.1),
                                  cond_dim=a.get("cond_dim", 32))
        net.load_state_dict(ck["model_state"])
        net.eval()
        for q in net.parameters():
            q.requires_grad_(False)
        for mod in net.modules():                     # freeze the MC-dropout
            if isinstance(mod, nn.Dropout2d) or mod.__class__.__name__ == "MCDropout2d":
                mod.p = 0.0
        self.net = net.to(device)

        nm = ck["norm"]
        Tn = (env.T - nm["T_mean"]) / nm["T_std"]
        Sn = (env.S - nm["S_mean"]) / nm["S_std"]

        # A held-out slice, evenly spaced so the seasonal cycle is covered.
        rng = np.random.default_rng(seed)
        idx = np.linspace(0, len(Tn) - 1, n_times).astype(int)
        self.register_buffer("Y", torch.tensor(
            np.stack([Tn[idx], Sn[idx]], 1), dtype=dtype, device=device))

        # The network is FED the normalised fields it was trained on -- that
        # distribution must not change. But it is SCORED on the domain-mean
        # anomaly, exactly like Brick 3's criterion.
        #
        # Why this is not optional: `mesoscale_anomaly` in Brick 1/3 removes the
        # domain mean at each time step precisely because the seasonal cycle is
        # a near-uniform mode. Score against it and every design reconstructs it
        # from almost any 23 observations, so explained variance sits around
        # 0.78 whatever you do -- measured spread across ALL designs including
        # random: 0.755 to 0.832, with a regular grid outranking greedy. That is
        # not an operator with an opinion about placement, it is an operator
        # measuring the seasonal cycle. Comparing it to EVF/OI, which score
        # deseasoned fields, would not be an operator comparison at all: the
        # three would be answering different questions.
        self.deseason = bool(deseason)

        pos = np.asarray(env.candidate_positions, dtype=np.int64)
        self.register_buffer("px", torch.tensor(pos[:, 0], device=device))
        self.register_buffer("py", torch.tensor(pos[:, 1], device=device))
        self.K, self.dtype_ = env.K, dtype
        self.device_ = device

    def _pixel_mask(self, m):
        """(B, K) candidate mask -> (B, 1, NX, NY) pixel mask, differentiably."""
        B = m.shape[0]
        M = torch.zeros(B, NX, NY, dtype=m.dtype, device=m.device)
        M = M.index_put((torch.arange(B, device=m.device)[:, None].expand(-1, self.K),
                         self.px[None].expand(B, -1),
                         self.py[None].expand(B, -1)), m, accumulate=True)
        return M.unsqueeze(1)

    def forward(self, m):
        single = (m.dim() == 1)
        if single:
            m = m[None]
        m = m.to(self.Y.dtype)
        Mp = self._pixel_mask(m)                       # (B,1,NX,NY)
        B, Tn = m.shape[0], self.Y.shape[0]
        out = []
        for b in range(B):                             # B is small; T is the cost
            mk = Mp[b:b+1].expand(Tn, -1, -1, -1)
            x = torch.cat([self.Y[:, 0:1] * mk, self.Y[:, 1:2] * mk, mk], dim=1)
            pred, _, _, _ = self.net(x)
            tgt = self.Y
            if self.deseason:
                pred = pred - pred.mean(dim=(2, 3), keepdim=True)
                tgt = tgt - tgt.mean(dim=(2, 3), keepdim=True)
            w = (1.0 - mk)                             # unobserved pixels only
            num = ((pred - tgt) ** 2 * w).sum()
            den = (tgt ** 2 * w).sum() + 1e-9
            out.append(1.0 - num / den)                # explained variance
        r = torch.stack(out)
        return r[0] if single else r


def design_geometry(env, idx):
    """
    Geometric signature of a network: how spread out is it, and how much of the
    domain does it actually cover?

    Motivation. At N = 8 the VAE operator scored a RANDOM network at 0.687 and
    greedy at 0.447 -- a 0.24 inversion, far too large to be noise, and the exact
    opposite of the EVF ranking (greedy 0.384, random 0.166). Saying "the designs
    differ" is not enough at that point; the interesting question is HOW. These
    three numbers answer it in a form anyone can check:

      nn_km       mean distance from each buoy to its nearest neighbour
      spread_km   mean pairwise distance (compactness of the whole network)
      cover       fraction of the domain within one decorrelation length of at
                  least one buoy

    A variance-seeking criterion piles buoys into the energetic band: small
    nn_km, low cover. A reconstructor that must fill the whole field rewards
    coverage instead. If that is what separates the operators, it shows up here.
    """
    pos = np.asarray([env.candidate_positions[i] for i in idx], dtype=float) * DX_KM
    if len(pos) < 2:
        return {"nn_km": np.nan, "spread_km": np.nan, "cover": np.nan}
    D = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nn = D.min(1).mean()
    Du = D[np.isfinite(D)]
    gx, gy = np.meshgrid(np.arange(NX) * DX_KM, np.arange(NY) * DX_KM,
                         indexing="ij")
    R = env.influence_px * DX_KM
    d2 = ((gx[..., None] - pos[:, 0]) ** 2 + (gy[..., None] - pos[:, 1]) ** 2)
    dmin = np.sqrt(d2.min(-1))
    cover = float((dmin <= R).mean())
    # Thresholded coverage saturates at low N (every design sat at 0.15-0.20
    # with R = 90 km and 8 buoys, telling us nothing). The mean distance from an
    # arbitrary point of the domain to its nearest buoy does not saturate and is
    # the statistic that actually separates a clustered design from a spread one.
    return {"nn_km": float(nn), "spread_km": float(Du.mean()), "cover": cover,
            "gap_km": float(dmin.mean())}


def vae_probe(env, scorer, Ns=(0, 2, 5, 10, 23, 40), n_draws=3, seed=0):
    """
    Does this operator respond to observations at all?

    A ranking is only meaningful if the operator's score moves when the network
    changes. Before comparing designs, check the score against BUDGET on random
    networks -- including N = 0. If the score at N = 0 is already close to the
    score at the working budget, the network is reconstructing from its learned
    prior and the placement is irrelevant to it; no amount of design optimisation
    will show up. If instead the curve rises steeply and then flattens, the
    operator is fine and the flatness at the working budget means something
    real: a strong nonlinear reconstructor genuinely reduces the value of where
    the buoys go.

    Those two readings look identical if you only ever score at one budget.
    """
    rng = np.random.default_rng(seed)
    dev, dt = scorer_device_dtype(scorer)
    out = []
    for N in Ns:
        vals = []
        for _ in range(1 if N == 0 else n_draws):
            m = torch.zeros(env.K, device=dev, dtype=dt)
            if N > 0:
                m[torch.as_tensor(np.asarray(env.sample_feasible(N, rng)),
                                  dtype=torch.long, device=dev)] = 1.0
            with torch.no_grad():
                vals.append(float(scorer(m)))
        out.append((N, float(np.mean(vals)), float(np.std(vals))))
    return out


def build_env_for_seed(args, seed_ocean, out_dir):
    """Nature run + environment for one ocean realisation, cached on disk."""
    cache_f = Path(out_dir) / f"nature_{seed_ocean}_{args.nt}.npz"
    if args.cache and cache_f.exists():
        z = np.load(cache_f); T, S = z["T"], z["S"]
    else:
        T, S = SyntheticOceanGenerator().generate_dataset(nt=args.nt,
                                                          seed=seed_ocean)
        if args.cache:
            np.savez_compressed(cache_f, T=T, S=S)
    return OceanNetworkEnv(T, S, grid_x=args.grid_x, grid_y=args.grid_y,
                           n_min=5, n_max=max(40, args.n_budget),
                           info_mode="evf", influence_km=args.influence_km,
                           eval_stride=args.eval_stride,
                           evf_cv=bool(args.evf_cv), min_sep=args.min_sep)


def ocean_seed_study(args, out_dir):
    """
    Repeat the operator-effect measurement across independent ocean
    realisations.

    Every number in this work so far came from a single nature run
    (seed_ocean = 42): the retained agreement, the self-consistency floor, the
    ranking of designs. One draw cannot distinguish "the reconstruction operator
    determines the network" from "this particular arrangement of 22 eddies
    happens to make two operators disagree". This is the control that separates
    them, and it is the first question any reader will ask.

    Per ocean it measures, at one budget: the gradient and greedy designs under
    both Gaussian operators, the cross-operator overlap for each, the
    within-objective self-consistency floor (the ceiling for the normalisation)
    and the chance level (the floor). The reported statistic is the retained
    agreement, which is scale-free and therefore comparable across oceans.
    """
    def _jac(a, b):
        A, B = set(map(int, a)), set(map(int, b))
        return len(A & B) / max(1, len(A | B))

    seeds = [int(x) for x in args.ocean_seeds.split(",")]
    n = args.n_budget
    dt = torch.float64 if args.f64 else torch.float32
    rows = []

    for si, sd in enumerate(seeds):
        t0 = time.time()
        print(f"\n=== ocean {si+1}/{len(seeds)}  (seed_ocean = {sd}) "
              + "=" * 24)
        env = build_env_for_seed(args, sd, out_dir)
        sc_evf = DiffEVF(env, device=args.device, dtype=dt).to(args.device)
        sc_oi = DiffOIMSE(env, device=args.device, dtype=dt,
                          learn_L=args.learn_L).to(args.device)

        def _run(loss_name, scr, seed_b):
            a = argparse.Namespace(**vars(args))
            a.loss = loss_name; a.polish = 0; a.seed_buoys = seed_b
            return optimize_gumbel(env, scr, n, a, verbose=False)["idx"]

        d_evf = _run("evf", sc_evf, args.seed_buoys)
        d_oi = _run("oi_mse", sc_oi, args.seed_buoys)
        g_evf = bl_greedy_sequence(env, n)[n - 1][0]
        g_oi = greedy_on_scorer(env, sc_oi, n, batch=args.greedy_batch,
                                device=args.device, verbose=False)[n][0]

        floors = {}
        for tag, ln, scr in (("EVF", "evf", sc_evf), ("OI", "oi_mse", sc_oi)):
            reps = [_run(ln, scr, args.seed_buoys + 100 * k)
                    for k in range(args.n_seeds)]
            js = [_jac(reps[i], reps[j]) for i in range(len(reps))
                  for j in range(i + 1, len(reps))]
            floors[tag] = float(np.mean(js)) if js else float("nan")

        chance = jaccard_chance(env, n, seed=args.seed_buoys)
        floor = float(np.nanmean(list(floors.values())))
        j_grad, j_greedy = _jac(d_evf, d_oi), _jac(g_evf, g_oi)
        # The retained-agreement ratio is only defined when the ceiling is
        # meaningfully above the floor. If the gradient reproduces itself no
        # better than chance, the denominator is noise and the ratio explodes
        # (a 227 % "retained agreement" was produced this way on a dry run).
        # Report NaN rather than a number that invites over-reading.
        keep = ((j_grad - chance) / (floor - chance)
                if floor > chance + 0.10 else float("nan"))
        keep_g = (j_greedy - chance) / max(1e-9, 1.0 - chance)

        rows.append(dict(
            seed=sd, chance=chance, floor=floor,
            floor_evf=floors["EVF"], floor_oi=floors["OI"],
            j_grad=j_grad, j_greedy=j_greedy, keep=keep, keep_g=keep_g,
            evf_grad=float(env.explained_variance(d_evf)),
            evf_greedy=float(env.explained_variance(g_evf)),
            secs=time.time() - t0))
        r = rows[-1]
        print(f"  chance {r['chance']:.2f} | floor {r['floor']:.2f} "
              f"(EVF {r['floor_evf']:.2f}, OI {r['floor_oi']:.2f})")
        print(f"  Jaccard across operators: gradient {r['j_grad']:.2f}, "
              f"greedy {r['j_greedy']:.2f}")
        kk = ("n/a (floor too low)" if not np.isfinite(r["keep"])
              else f"{100*r['keep']:.0f} %")
        print(f"  retained agreement: {kk} gradient, "
              f"{100*r['keep_g']:.0f} % greedy      ({r['secs']:.0f}s)")
        if r["floor_evf"] < 0.20 or r["floor_oi"] < 0.20:
            print("  [WARN] a self-consistency floor is below 0.20: the "
                  "gradient barely reproduces itself on this ocean, so its "
                  "retained agreement is not reliable evidence here.")

    return rows


def write_ocean_report(args, rows, path):
    def col(k):
        return np.array([r[k] for r in rows], dtype=float)
    L = []
    L.append("NAIADE - Brick 4: operator effect across ocean realisations")
    L.append("=" * 70)
    L.append(f"date            : {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"ocean seeds     : {args.ocean_seeds}   (nt={args.nt})")
    L.append(f"budget N        : {args.n_budget}")
    L.append(f"candidate grid  : {args.grid_x}x{args.grid_y} | "
             f"min_sep={args.min_sep}")
    L.append(f"schedule        : {args.iters} it, n_mc={args.n_mc}, "
             f"tau {args.tau0}->{args.tau1}, alpha={args.alpha}, "
             f"lam_sep={args.lam_sep}, w_bin={args.w_bin}")
    L.append(f"replicates      : {args.n_seeds} buoy seeds per objective")
    L.append(f"OI operator     : L {'learned' if args.learn_L else 'fixed'} "
             f"at prior {INFLUENCE_RADIUS_KM:.0f} km")
    L.append("")
    L.append("-- Per ocean " + "-" * 57)
    L.append(f"   {'seed':>5s} {'chance':>7s} {'floor':>6s} {'fl.EVF':>7s} "
             f"{'fl.OI':>6s} {'J.grad':>7s} {'J.grdy':>7s} "
             f"{'keep':>6s} {'keep_g':>7s}")
    for r in rows:
        L.append(f"   {r['seed']:>5d} {r['chance']:>7.2f} {r['floor']:>6.2f} "
                 f"{r['floor_evf']:>7.2f} {r['floor_oi']:>6.2f} "
                 f"{r['j_grad']:>7.2f} {r['j_greedy']:>7.2f} "
                 + (f" {100*r['keep']:>5.0f}%" if np.isfinite(r['keep'])
                    else f" {'n/a':>6s}")
                 + f" {100*r['keep_g']:>6.0f}%")
    L.append("")
    L.append("-- Mean +- sd across oceans " + "-" * 42)
    for k, lab in (("chance", "chance level"),
                   ("floor", "self-consistency floor"),
                   ("j_grad", "Jaccard across operators, gradient"),
                   ("j_greedy", "Jaccard across operators, greedy"),
                   ("evf_grad", "EVF of the gradient design"),
                   ("evf_greedy", "EVF of the greedy design")):
        v = col(k)
        L.append(f"   {lab:<38s} {v.mean():.3f} +- {v.std(ddof=1) if len(v)>1 else 0:.3f}")
    for k, lab in (("keep", "retained agreement, gradient"),
                   ("keep_g", "retained agreement, greedy")):
        v = 100 * col(k)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            L.append(f"   {lab:<38s} n/a (self-consistency floor too low)")
        else:
            L.append(f"   {lab:<38s} {v.mean():.0f} % +- "
                     f"{v.std(ddof=1) if len(v)>1 else 0:.0f} %"
                     + ("" if len(v) == len(rows)
                        else f"   [{len(v)}/{len(rows)} oceans usable]"))
    L.append("")
    L.append("-- Reading " + "-" * 58)
    kg = 100 * col("keep_g")
    fl = min(col("floor_evf").min(), col("floor_oi").min())
    if fl < 0.20:
        L.append("   A self-consistency floor fell below 0.20 on at least one")
        L.append("   ocean: the gradient does not reproduce itself there, so")
        L.append("   its retained agreement is not evidence. The greedy column")
        L.append("   is deterministic and remains readable.")
    sd_ = kg.std(ddof=1) if len(kg) > 1 else 0.0
    if kg.mean() < 40 and sd_ < 15:
        L.append("   The greedy retained agreement is low and consistent across")
        L.append("   oceans: the operator effect is a property of the setup,")
        L.append("   not of one realisation.")
    elif sd_ >= 15:
        L.append("   The retained agreement varies widely between oceans")
        L.append(f"   (sd {sd_:.0f} points). One realisation would have been")
        L.append("   misleading; report the spread, not a single figure.")
    L.append("")
    L.append("=" * 70)
    Path(path).write_text("\n".join(L), encoding="utf-8")


def scorer_device_dtype(scorer):
    """
    Device and floating dtype of any scorer, without knowing which one it is.

    The old code reached for `scorer.C_OO` or `scorer.d2_oo` by name, which tied
    every call site to the two Gaussian operators and broke the moment a third
    one (DiffVAE) was added. Picking the first FLOATING-point buffer also matters:
    DiffVAE registers integer index buffers, and inheriting int64 as the working
    dtype would silently corrupt the mask.
    """
    for b in scorer.buffers():
        if b.is_floating_point():
            return b.device, b.dtype
    for q in scorer.parameters():
        if q.is_floating_point():
            return q.device, q.dtype
    return torch.device("cpu"), torch.float32


def build_scorer(env, args):
    dt = torch.float64 if args.f64 else torch.float32
    if args.loss == "oi_mse":
        return DiffOIMSE(env, device=args.device, dtype=dt,
                         learn_L=args.learn_L,
                         L_lo_px=(args.L_lo_km / DX_KM) if args.L_lo_km else None,
                         L_hi_px=(args.L_hi_km / DX_KM) if args.L_hi_km else None
                         ).to(args.device)
    if args.loss == "vae":
        if not args.vae_ckpt or not Path(args.vae_ckpt).exists():
            raise SystemExit(
                "--loss vae needs --vae_ckpt pointing at a Brick 1 checkpoint "
                "(e.g. figures/vae_best.pt).")
        return DiffVAE(env, args.vae_ckpt, device=args.device, dtype=dt,
                       n_times=args.vae_times, seed=args.seed_buoys,
                       deseason=not args.vae_raw)
    if args.learn_L:
        print("  [WARN] --learn_L is ignored with --loss evf (it would be "
              "circular: L defines the metric). Use --loss oi_mse.")
    return DiffEVF(env, device=args.device, dtype=dt).to(args.device)


if __name__ == "__main__":
    from datetime import datetime
    args = parse_args()
    if not (args.train or args.sweep or args.check or args.cross
            or args.l_scan or args.vae_probe or args.ocean_seeds):
        print("Usage: python 04_gumbel.py "
              "[--train] [--sweep] [--cross] [--l_scan L1,L2,...] "
              "[--report] [--check]")
        sys.exit(0)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.ocean_seeds:
        stamp0 = datetime.now().strftime("%Y%m%d_%H%M%S")
        rows = ocean_seed_study(args, out_dir)
        rp = out_dir / f"report_oceans_{stamp0}.txt"
        write_ocean_report(args, rows, rp)
        print(f"\n  -> {rp}")
        print(Path(rp).read_text())
        sys.exit(0)
    torch.manual_seed(args.seed_buoys); np.random.seed(args.seed_buoys)

    print(f"\n[1/3] Nature run (seed_ocean={args.seed_ocean}, nt={args.nt})...")
    cache_f = out_dir / f"nature_{args.seed_ocean}_{args.nt}.npz"
    if args.cache and cache_f.exists():
        z = np.load(cache_f); T, S = z["T"], z["S"]
        print(f"  loaded from {cache_f}")
    else:
        T, S = SyntheticOceanGenerator().generate_dataset(nt=args.nt,
                                                          seed=args.seed_ocean)
        if args.cache:
            np.savez_compressed(cache_f, T=T, S=S)
            print(f"  cached -> {cache_f}")

    print("[2/3] Environment (shared with Brick 3)...")
    env = OceanNetworkEnv(T, S, grid_x=args.grid_x, grid_y=args.grid_y,
                          n_min=5, n_max=max(40, args.n_budget),
                          info_mode="evf", influence_km=args.influence_km,
                          eval_stride=args.eval_stride,
                          evf_cv=bool(args.evf_cv), min_sep=args.min_sep)
    print(f"  K = {env.K} candidates | feasible max = {env.n_feasible_max} buoys "
          f"| eval cells = {len(env._eval_xy)}")

    print("[3/3] Differentiable criterion...")
    scorer = build_scorer(env, args)

    # ---- consistency check: soft criterion == Brick 3 on binary masks --------
    if args.check or args.train or args.sweep:
        rng = np.random.default_rng(0)
        errs = []
        chk = DiffEVF(env, device=args.device).to(args.device)
        for n in (5, 12, 23, 35):
            idx = env.sample_feasible(n, rng)
            m = torch.zeros(env.K, device=args.device); m[idx] = 1.0
            a = float(chk(m)); b = float(env.explained_variance(idx))
            errs.append(abs(a - b))
            print(f"  N={n:3d}  torch {a:.6f}   numpy(Brick 3) {b:.6f}   "
                  f"|delta| {abs(a-b):.2e}")
        print(f"  low-rank factor: r = {chk.rank}/{2*env.K} "
              f"({100*chk.energy_kept:.4f}% of trace P)")
        print(f"  max |delta| = {max(errs):.2e}"
              + ("  OK" if max(errs) < 2e-4 else "  <-- CHECK"))
        if args.check and not (args.train or args.sweep):
            sys.exit(0)

    results = {}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------- single budget ------------------------------------------
    if args.vae_probe:
        print("\n=== Operator response probe: score vs budget, random networks ===")
        rows = vae_probe(env, scorer, n_draws=args.n_probe_draws,
                         seed=args.seed_buoys)
        for N, mu, sd in rows:
            print(f"   N = {N:3d}   score {mu:+.4f}  +-{sd:.4f}")
        v0 = rows[0][1]
        vw = [r[1] for r in rows if r[0] == args.n_budget]
        vw = vw[0] if vw else rows[-1][1]
        print(f"   gain from 0 to {args.n_budget} buoys: {vw - v0:+.4f}")
        if vw - v0 < 0.15:
            print("   -> the operator barely responds to observations. Any "
                  "ranking it produces is noise; do not use it as a comparison "
                  "point.")
        results["probe"] = rows

    if args.train:
        n = args.n_budget
        print(f"\n=== Gumbel-Softmax optimisation, N_budget = {n} "
              f"(loss = {args.loss}) ===")
        t0 = time.time()
        res = optimize_gumbel(env, scorer, n, args)
        dt = time.time() - t0
        print(f"  done in {dt:.1f}s | EVF(hard) = {res['evf_hard']:.4f}")

        r = bl_random(env, n, args.n_random, args.seed_buoys)
        seq = bl_greedy_sequence(env, max(40, n))
        g_idx = seq[min(n, len(seq)) - 1][0]
        others = {}
        if args.polish and res["evf_hard"] > res["evf_raw"] + 1e-9:
            others["Gumbel-Softmax + polish"] = res["evf_hard"]
        others["Gumbel-Softmax (gradient only)"] = res["evf_raw"]
        others.update({
            "greedy (submodular)": env.explained_variance(g_idx),
            "PCA-QR": env.explained_variance(bl_pcaqr(env, n)),
            "variance top-N": env.explained_variance(bl_variance(env, n)),
            "regular grid": env.explained_variance(bl_regular(env, n, args.seed_buoys)),
            "random (mean)": float(r.mean()),
        })
        for k, v in others.items():
            print(f"    {k:<30s} {v:.4f}")
        neq = equivalent_random_N(env, res["evf_hard"], env.n_feasible_max,
                                  20, args.seed_buoys)
        if neq:
            print(f"  -> {n} optimised buoys are worth ~{neq} randomly placed ones")
        if args.loss != "evf":
            # A mask optimised for an OI with correlation length L must not be
            # judged solely by an OI with a DIFFERENT L. `others` above is the
            # Brick 3 criterion (L fixed at INFLUENCE_RADIUS_KM); here is the
            # score under the operator the gradient actually optimised. If the
            # two disagree strongly, the design is chasing a reconstruction
            # regime that Brick 3 does not model -- read that as a finding, not
            # as a failure of the placement.
            with torch.no_grad():
                mb = torch.zeros(env.K, device=args.device,
                                 dtype=scorer_device_dtype(scorer)[1])
                mb[res["idx"]] = 1.0
                own = float(scorer(mb))
            if hasattr(scorer, "L_px"):
                desc = (f"OI operator (L = {res['L_px']*DX_KM:.0f} km, "
                        f"nugget = {float(scorer.nugget().detach()):.4f})")
            else:
                desc = f"{args.loss} operator"
            print(f"  -> score under its own {desc}: {own:.4f}")
            if own < 0.05:
                print("     [WARN] this operator barely beats climatology "
                      "out of sample - the mask is optimising a reconstruction "
                      "that does not work. Do not read the placement yet.")
            print(f"  -> score under the Brick 3 operator (L = "
                  f"{env.influence_px*DX_KM:.0f} km): {res['evf_hard']:.4f}")

            # 0.27 in isolation says nothing: the OI operator's absolute skill
            # depends on its own regularisation. What matters is the RANKING of
            # designs under one and the same reconstructor -- which is the whole
            # premise of the reference paper. Score the baselines with it too.
            def _own_of(idx):
                with torch.no_grad():
                    mb = torch.zeros(env.K, device=args.device,
                                     dtype=scorer_device_dtype(scorer)[1])
                    mb[np.asarray(idx)] = 1.0
                    return float(scorer(mb))
            rng_o = np.random.default_rng(args.seed_buoys)
            own_tbl = {
                "Gumbel-Softmax": own,
                "greedy (submodular)": _own_of(g_idx),
                "PCA-QR": _own_of(bl_pcaqr(env, n)),
                "variance top-N": _own_of(bl_variance(env, n)),
                "regular grid": _own_of(bl_regular(env, n, args.seed_buoys)),
                "random (mean)": float(np.mean([
                    _own_of(env.sample_feasible(n, rng_o)) for _ in range(10)])),
            }
            geo_sets = {"Gumbel-Softmax": res["idx"],
                        "greedy (submodular)": g_idx,
                        "PCA-QR": bl_pcaqr(env, n),
                        "variance top-N": bl_variance(env, n),
                        "regular grid": bl_regular(env, n, args.seed_buoys),
                        "random (mean)": env.sample_feasible(
                            n, np.random.default_rng(args.seed_buoys))}
            print(f"  -> ranking under that same {args.loss} operator, with the "
                  f"geometry of each design:")
            print(f"       {'design':<28s} {'score':>7s} {'nn_km':>7s} "
                  f"{'spread':>7s} {'gap_km':>7s} {'cover':>6s}")
            geo_tbl = {}
            for k, v in own_tbl.items():
                gsig = design_geometry(env, geo_sets[k]) if k in geo_sets else {}
                geo_tbl[k] = gsig
                print(f"       {k:<28s} {v:7.4f} "
                      f"{gsig.get('nn_km', float('nan')):7.0f} "
                      f"{gsig.get('spread_km', float('nan')):7.0f} "
                      f"{gsig.get('gap_km', float('nan')):7.0f} "
                      f"{gsig.get('cover', float('nan')):6.2f}")
            blk = {"own_table": own_tbl, "operator": args.loss,
                   "geometry": geo_tbl}
            if hasattr(scorer, "L_px"):
                blk["L_km"] = res["L_px"] * DX_KM
                blk["nugget"] = float(scorer.nugget().detach())
            results.setdefault("own", {}).update(blk)
            if args.learn_L and hasattr(scorer, "L_px"):
                lo, hi = scorer.L_lo * DX_KM, scorer.L_hi * DX_KM
                at_bound = (res["L_px"] * DX_KM > 0.98 * hi
                            or res["L_px"] * DX_KM < 1.02 * lo)
                print(f"  -> learned correlation length: {res['L_px']*DX_KM:.1f} km "
                      f"(prior {env.influence_px*DX_KM:.1f} km, "
                      f"bounds {lo:.0f}-{hi:.0f} km)"
                      + ("   [AT THE BOUND - widen with --L_hi_km/--L_lo_km "
                         "or use a two-scale kernel]" if at_bound else ""))

        plot_train(env, res, n, others, out_dir / "gumbel_train.png")
        np.save(out_dir / "gumbel_probs.npy", res["p"])
        results["train"] = {"n": n, "evf": res["evf_hard"], "others": others,
                            "n_equiv_random": neq, "L_km": res["L_px"] * DX_KM,
                            "seconds": dt,
                            "positions": [list(map(int, env.candidate_positions[i]))
                                          for i in res["idx"]]}

    # ---------------- sweep over N -------------------------------------------
    # ---------------- cross-scoring: design x reconstruction operator --------
    if args.cross:
        print("\n=== Cross-scoring: every design under BOTH operators ===")
        print("  Each optimiser wins on its own metric by construction. The")
        print("  question this answers is whether the DESIGNS themselves differ,")
        print("  or only the yardsticks.")
        n = args.n_budget
        dt_ = torch.float64 if args.f64 else torch.float32
        sc_evf = DiffEVF(env, device=args.device, dtype=dt_).to(args.device)
        sc_oi = DiffOIMSE(env, device=args.device, dtype=dt_,
                          learn_L=args.learn_L,
                          L_lo_px=(args.L_lo_km / DX_KM) if args.L_lo_km else None,
                          L_hi_px=(args.L_hi_km / DX_KM) if args.L_hi_km else None
                          ).to(args.device)

        def _sc(scr, idx):
            with torch.no_grad():
                mb = torch.zeros(env.K, device=args.device,
                                 dtype=next(iter(scr.buffers())).dtype)
                mb[np.asarray(idx)] = 1.0
                return float(scr(mb))

        designs = {}
        print("  [1/4] gradient design on the EVF objective...")
        a = argparse.Namespace(**vars(args)); a.loss = "evf"; a.polish = 0
        designs["gradient (EVF objective)"] = optimize_gumbel(
            env, sc_evf, n, a, verbose=False)["idx"]

        print("  [2/4] gradient design on the OI objective...")
        a = argparse.Namespace(**vars(args)); a.loss = "oi_mse"; a.polish = 0
        designs["gradient (OI objective)"] = optimize_gumbel(
            env, sc_oi, n, a, verbose=False)["idx"]

        print("  [3/4] greedy on the EVF objective...")
        designs["greedy (EVF objective)"] = bl_greedy_sequence(
            env, max(40, n))[min(n, 40) - 1][0]

        print("  [4/4] greedy on the OI objective (K x N scorer calls)...")
        g_oi = greedy_on_scorer(env, sc_oi, n, device=args.device)
        designs["greedy (OI objective)"] = g_oi[max(g_oi)][0]

        rng_c = np.random.default_rng(args.seed_buoys)
        designs["random"] = env.sample_feasible(n, rng_c)

        table = {k: (_sc(sc_evf, v), _sc(sc_oi, v)) for k, v in designs.items()}
        print(f"\n  {'design':<34s} {'Brick3 EVF':>12s} {'learned OI':>12s}")
        for k, (x, y) in table.items():
            print(f"  {k:<34s} {x:>12.4f} {y:>12.4f}")

        # ---- Is the low overlap an OPERATOR effect or optimiser noise? -----
        #
        # A Jaccard of 0.10 between the EVF-design and the OI-design is only
        # meaningful if the SAME objective, re-optimised, reproduces itself.
        # The polish diagnostic already showed the criterion has a flat optimum
        # (11 of 23 buoys relocated for +0.011 EVF), so many near-equivalent
        # designs exist and a stochastic optimiser will not return the same one
        # twice. Without this control the operator comparison is unfalsifiable.
        #
        # Two references are computed:
        #   - within-objective, across seeds: the noise floor of the optimiser
        #   - greedy-EVF vs greedy-OI: greedy is DETERMINISTIC, so their overlap
        #     isolates the operator effect with no optimiser noise at all
        def _jac(a, b):
            A, B = set(map(int, a)), set(map(int, b))
            return len(A & B) / max(1, len(A | B)), len(A & B)

        print("\n  [control] same objective, different seeds "
              f"({args.n_seeds} replicates)...")
        reps = {}
        for tag, loss_name, scr in (("EVF", "evf", sc_evf),
                                    ("OI", "oi_mse", sc_oi)):
            runs = []
            for k in range(args.n_seeds):
                a = argparse.Namespace(**vars(args))
                a.loss = loss_name; a.polish = 0; a.seed_buoys = args.seed_buoys + 100 * k
                runs.append(optimize_gumbel(env, scr, n, a, verbose=False)["idx"])
            js = [_jac(runs[i], runs[j])[0]
                  for i in range(len(runs)) for j in range(i + 1, len(runs))]
            reps[tag] = float(np.mean(js)) if js else float("nan")
            print(f"    gradient on {tag:<3s}: mean within-objective Jaccard "
                  f"{reps[tag]:.2f}")

        jac, shared = _jac(designs["gradient (EVF objective)"],
                           designs["gradient (OI objective)"])
        jac_g, shared_g = _jac(designs["greedy (EVF objective)"],
                               designs["greedy (OI objective)"])

        print(f"\n  across operators, gradient: {shared}/{n} sites shared "
              f"(Jaccard {jac:.2f})")
        print(f"  across operators, greedy  : {shared_g}/{n} sites shared "
              f"(Jaccard {jac_g:.2f})   [deterministic, no optimiser noise]")
        floor = np.nanmean([reps.get("EVF", np.nan), reps.get("OI", np.nan)])
        chance = jaccard_chance(env, n, seed=args.seed_buoys)
        print(f"  optimiser noise floor     : Jaccard {floor:.2f}"
              f"   (self-consistency ceiling)")
        print(f"  chance level              : Jaccard {chance:.2f}"
              f"   (two independent feasible networks)")
        # Each generator must be normalised by ITS OWN self-consistency
        # ceiling. Greedy is deterministic: re-running it returns the identical
        # network, so its ceiling is 1, not the gradient's noise floor. Using
        # the gradient floor for both inflated greedy's retained agreement from
        # 13 % to 58 % -- a factor of four, and in the direction that flatters
        # the conclusion.
        keep = (jac - chance) / max(1e-9, floor - chance)
        keep_g = (jac_g - chance) / max(1e-9, 1.0 - chance)
        print(f"  agreement retained across operators: {100*keep:.0f} % "
              f"(gradient, vs its own floor {floor:.2f}), "
              f"{100*keep_g:.0f} % (greedy, vs a ceiling of 1.0)")
        # The mean floor hides an asymmetry: at N = 8 the EVF gradient
        # reproduced itself at only 0.12 while OI managed 0.36. Judge against
        # the WEAKER of the two, otherwise a design that cannot even reproduce
        # itself is credited with a large margin it has not earned.
        floor_min = np.nanmin([reps.get("EVF", np.nan), reps.get("OI", np.nan)])
        if floor_min < 0.20:
            print(f"  [WARN] the gradient reproduces itself at only "
                  f"{floor_min:.2f} on one objective: its cross-operator "
                  f"overlap is not reliable evidence here. Read the "
                  f"deterministic greedy control instead.")
        if jac < floor_min - 0.05:
            print("  -> the operator moves the buoys further than reseeding "
                  "does: an OPERATOR effect.")
        elif jac_g < 0.5:
            print("  -> the gradient comparison is inside its own noise, but "
                  "the deterministic greedy pair confirms an operator effect.")
        else:
            print("  -> the gradient comparison is INSIDE its own noise floor. "
                  "Do not read it as an operator effect.")

        results["cross"] = {"table": table, "jaccard": jac,
                            "jaccard_greedy": jac_g, "jaccard_floor": float(floor),
                            "jaccard_chance": float(chance),
                            "retained": float(keep), "retained_greedy": float(keep_g),
                            "replicates": reps,
                            "L_km": float(sc_oi.L_px().detach()) * DX_KM,
                            "positions": {k: [list(map(int,
                                env.candidate_positions[i])) for i in v]
                                for k, v in designs.items()}}

    # ---------------- L-scan: decomposing the operator effect ----------------
    if args.l_scan:
        # The obvious objection to the cross-scoring result: the two operators
        # differ mostly in their correlation length (90 km vs ~57 km) and in
        # their regularisation, and both are linear Gaussian reconstructors. A
        # reviewer will ask whether "operator effect" is really just sensitivity
        # to one hyper-parameter. This scan answers it directly: re-optimise the
        # design against the OI operator with L FIXED at a range of values, and
        # watch the overlap with the EVF design as a function of L alone.
        #
        # If the overlap climbs back towards the noise floor as L -> 90 km, the
        # effect is L sensitivity and should be reported as such. If it stays
        # low at every L, something other than the correlation length is moving
        # the buoys.
        print("\n=== L-scan: is the operator effect just the correlation "
              "length? ===")
        n = args.n_budget
        Ls = [float(x) for x in args.l_scan.split(",")]
        dt_ = torch.float64 if args.f64 else torch.float32
        sc_evf = DiffEVF(env, device=args.device, dtype=dt_).to(args.device)
        a = argparse.Namespace(**vars(args)); a.loss = "evf"; a.polish = 0
        d_evf = set(map(int, optimize_gumbel(env, sc_evf, n, a,
                                             verbose=False)["idx"]))
        print(f"  reference: gradient design on Brick 3 EVF "
              f"(L = {env.influence_px*DX_KM:.0f} km)")
        rows = []
        for Lk in Ls:
            sc_l = DiffOIMSE(env, device=args.device, dtype=dt_, learn_L=False,
                             L0_px=Lk / DX_KM).to(args.device)
            a = argparse.Namespace(**vars(args)); a.loss = "oi_mse"; a.polish = 0
            d = set(map(int, optimize_gumbel(env, sc_l, n, a,
                                             verbose=False)["idx"]))
            j = len(d_evf & d) / max(1, len(d_evf | d))
            rows.append((Lk, j, len(d_evf & d)))
            print(f"    OI at L = {Lk:6.1f} km -> Jaccard vs EVF design "
                  f"{j:.2f}  ({len(d_evf & d)}/{n} shared)")
        chance = jaccard_chance(env, n, seed=args.seed_buoys)
        print(f"  chance level (two independent networks): Jaccard {chance:.2f}")
        results["l_scan"] = {"L_km": [r[0] for r in rows],
                             "jaccard": [r[1] for r in rows],
                             "jaccard_chance": float(chance)}
        js = [r[1] for r in rows]
        near = [r for r in rows if abs(r[0] - env.influence_px * DX_KM) < 1e-6]
        if near and near[0][1] > max(js) - 0.05 and max(js) - min(js) > 0.15:
            print("  -> overlap recovers as L approaches the Brick 3 value: the "
                  "effect is largely CORRELATION-LENGTH sensitivity.")
        elif max(js) - min(js) < 0.15:
            print("  -> overlap is flat in L: the correlation length is NOT "
                  "what moves the buoys.")
        else:
            print("  -> mixed; read the curve rather than a verdict.")

    if args.sweep:
        n_list = [int(x) for x in args.n_list.split(",")
                  if int(x) <= env.n_feasible_max]
        print(f"\n=== Pareto sweep over N = {n_list} ===")
        seq = bl_greedy_sequence(env, max(n_list))
        policy = None
        ck = Path(args.checkpoint)
        if ck.exists():
            try:
                policy = ActorCritic(env.obs_dim, env.K).to(args.device)
                policy.load_state_dict(torch.load(ck, map_location=args.device,
                                                  weights_only=False)["policy_state"])
                pseq = _policy_sequence(env, policy, max(n_list))
                print(f"  PPO checkpoint loaded ({ck})")
            except Exception as e:
                print(f"  [WARN] PPO checkpoint unusable: {e}")
                policy, pseq = None, []
        else:
            pseq = []

        sw = {"n_list": n_list, "gumbel": [], "greedy": [], "pcaqr": [],
              "variance": [], "regular": [], "random": [], "rand_lo": [],
              "rand_hi": [], "ppo": [] if pseq else None}
        for n in n_list:
            t0 = time.time()
            res = optimize_gumbel(env, scorer, n, args, verbose=False)
            r = bl_random(env, n, args.n_random, args.seed_buoys)
            sw["gumbel"].append(res["evf_hard"])
            sw.setdefault("gumbel_raw", []).append(res["evf_raw"])
            sw["greedy"].append(seq[min(n, len(seq)) - 1][1])
            sw["pcaqr"].append(env.explained_variance(bl_pcaqr(env, n)))
            sw["variance"].append(env.explained_variance(bl_variance(env, n)))
            sw["regular"].append(env.explained_variance(
                bl_regular(env, n, args.seed_buoys)))
            sw["random"].append(float(r.mean()))
            sw["rand_lo"].append(float(r.mean() - r.std()))
            sw["rand_hi"].append(float(r.mean() + r.std()))
            if pseq:
                sw["ppo"].append(pseq[min(n, len(pseq)) - 1][1])
            print(f"  N={n:3d} | GS {sw['gumbel'][-1]:.4f} | greedy "
                  f"{sw['greedy'][-1]:.4f} | PCA-QR {sw['pcaqr'][-1]:.4f} | "
                  f"random {sw['random'][-1]:.4f}"
                  + (f" | PPO {sw['ppo'][-1]:.4f}" if pseq else "")
                  + f"   ({time.time()-t0:.0f}s)")
        plot_sweep(sw, out_dir / "gumbel_pareto.png")
        results["sweep"] = sw

    # ---------------- report --------------------------------------------------
    if args.report:
        p = out_dir / f"report_gumbel_{stamp}.txt"
        with open(p, "w") as f:
            f.write("NAIADE - Brick 4: differentiable placement (Gumbel-Softmax)\n")
            f.write("=" * 70 + "\n")
            f.write(f"date            : {stamp}\n")
            f.write(f"nature run      : nt={args.nt}, seed={args.seed_ocean}\n")
            f.write(f"candidate grid  : {args.grid_x}x{args.grid_y} = {env.K}"
                    f" | min_sep={env.min_sep} | feasible max={env.n_feasible_max}\n")
            if args.l_scan and not (args.train or args.cross or args.sweep):
                f.write("criterion       : L-scan, oi_mse at fixed L vs "
                        "Brick 3 evf\n")
            elif args.cross:
                f.write("criterion       : cross-scoring, evf + oi_mse"
                        + (" (L learned on the OI operator only)"
                           if args.learn_L else "") + "\n")
            else:
                f.write(f"criterion       : {args.loss}"
                        f"{' (L learned)' if args.learn_L and args.loss == 'oi_mse' else ''}\n")
            f.write(f"schedule        : {args.iters} it, n_mc={args.n_mc}, "
                    f"tau {args.tau0}->{args.tau1}, alpha={args.alpha}, "
                    f"lam_sep={args.lam_sep}\n\n")
            if "train" in results:
                t = results["train"]
                f.write(f"-- Single budget N = {t['n']} --\n")
                for k, v in t["others"].items():
                    f.write(f"   {k:<32s} {v:.4f}\n")
                if t["n_equiv_random"]:
                    f.write(f"   equivalent random network : "
                            f"{t['n_equiv_random']} buoys\n")
                f.write(f"   wall time                 : {t['seconds']:.1f} s\n\n")
            if "own" in results and results["own"].get("own_table"):
                o = results["own"]
                f.write(f"-- Ranking under the {o.get('operator', '?').upper()} "
                        f"operator (the one actually optimised) --\n")
                if "L_km" in o:
                    f.write(f"   (L = {o['L_km']:.1f} km, "
                            f"nugget = {o.get('nugget', float('nan')):.4f})\n")
                for k, v in results["own"]["own_table"].items():
                    f.write(f"   {k:<32s} {v:.4f}\n")
                f.write("\n")
            if "cross" in results:
                f.write("-- Cross-scoring matrix: design x reconstruction operator --\n")
                c = results["cross"]
                f.write(f"   {'design':<34s} {'Brick3 EVF':>12s} {'learned OI':>12s}\n")
                for k, (a, b) in c["table"].items():
                    f.write(f"   {k:<34s} {a:>12.4f} {b:>12.4f}\n")
                f.write(f"\n   Jaccard, gradient across operators : "
                        f"{c.get('jaccard', float('nan')):.2f}\n")
                f.write(f"   Jaccard, greedy across operators   : "
                        f"{c.get('jaccard_greedy', float('nan')):.2f}"
                        f"   (deterministic)\n")
                f.write(f"   optimiser noise floor              : "
                        f"{c.get('jaccard_floor', float('nan')):.2f}\n")
                if c.get("replicates"):
                    for k_, v_ in c["replicates"].items():
                        f.write(f"     within-objective, {k_:<3s}          : "
                                f"{v_:.2f}\n")
                f.write(f"   chance level (independent networks): "
                        f"{c.get('jaccard_chance', float('nan')):.2f}\n")
                f.write(f"   agreement retained across operators: "
                        f"{100*c.get('retained', float('nan')):.0f} % gradient, "
                        f"{100*c.get('retained_greedy', float('nan')):.0f} % greedy\n")
                f.write(f"   learned OI correlation length      : "
                        f"{c.get('L_km', float('nan')):.1f} km\n")
                fl = c.get("jaccard_floor", float("nan"))
                jg = c.get("jaccard_greedy", float("nan"))
                mg = fl - c.get("jaccard", float("nan"))
                f.write(f"   verdict : "
                        + ("operator effect (margin "
                           f"{mg:.2f}, deterministic control {jg:.2f})"
                           if mg > 0.15 else
                           "INCONCLUSIVE - margin under 0.15, do not report as "
                           "an operator effect") + "\n")
                f.write("\n")
            if "l_scan" in results:
                f.write("-- L-scan: overlap with the EVF design vs OI "
                        "correlation length --\n")
                for Lk, j in zip(results["l_scan"]["L_km"],
                                 results["l_scan"]["jaccard"]):
                    f.write(f"   L = {Lk:6.1f} km    Jaccard {j:.2f}\n")
                f.write(f"   chance level      Jaccard "
                        f"{results['l_scan'].get('jaccard_chance', float('nan')):.2f}\n")
                f.write("\n")
            if "sweep" in results:
                s = results["sweep"]
                f.write("-- Pareto sweep --\n")
                f.write(f"{'N':>4} {'gumbel':>9} {'greedy':>9} {'PCA-QR':>9} "
                        f"{'variance':>9} {'regular':>9} {'random':>9}"
                        + (f" {'PPO':>9}" if s.get("ppo") else "") + "\n")
                for i, n in enumerate(s["n_list"]):
                    f.write(f"{n:>4} {s['gumbel'][i]:>9.4f} {s['greedy'][i]:>9.4f} "
                            f"{s['pcaqr'][i]:>9.4f} {s['variance'][i]:>9.4f} "
                            f"{s['regular'][i]:>9.4f} {s['random'][i]:>9.4f}"
                            + (f" {s['ppo'][i]:>9.4f}" if s.get("ppo") else "") + "\n")
        json.dump(results, open(out_dir / f"gumbel_{stamp}.json", "w"), indent=2,
                  default=float)
        print(f"  -> {p}")

    print("\nDone.")
