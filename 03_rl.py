"""
===========================================================================
  BRIQUE 3 -- Reinforcement Learning : Optimisation du Reseau
  
  3 methodes de selection du N* optimal :
    pareto      : front de Pareto info vs N (sweep + Kneedle)
    efficiency  : eta(N) = info(N) / (1+log(N)), score unique
    scalarized  : PPO avec cout marginal integre, sweep sur lambda
  
  Usage :
    python 03_rl.py --train --evaluate --rl_method pareto
    python 03_rl.py --train --evaluate --rl_method efficiency
    python 03_rl.py --train --evaluate --rl_method scalarized
===========================================================================
"""

import sys, argparse
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from config import *
try:
    from dataset import SyntheticOceanGenerator
except ModuleNotFoundError:
    from data.dataset import SyntheticOceanGenerator

try:
    from importlib.util import spec_from_file_location, module_from_spec
    _spec = spec_from_file_location("autoencoder", Path(__file__).parent / "01_autoencoder.py")
    _ae_mod = module_from_spec(_spec)
    _spec.loader.exec_module(_ae_mod)
    ObservabilityAE = _ae_mod.ObservabilityAE
    AE_AVAILABLE = True
except Exception:
    AE_AVAILABLE = False


# =========================================================================
#  ENVIRONNEMENT MDP
# =========================================================================

class OceanNetworkEnv:
    """
    Grille candidate GX*GY. Action = toggle d'une position.
    Supporte deux modes de recompense :
      - standard  : delta_info - budget_penalty
      - scalarized: delta_info - lambda * cout marginal
    """
    def __init__(self, T, S, grid_x=16, grid_y=24,
                 n_min=10, n_max=40, episode_len=20,
                 w_info=1.0, w_budget=0.5, marginal_cost=0.0):
        self.T = T.astype(np.float32)
        self.S = S.astype(np.float32)
        self.grid_x, self.grid_y = grid_x, grid_y
        self.K = grid_x * grid_y
        self.n_min, self.n_max = n_min, n_max
        self.ep_len = episode_len
        self.w_info, self.w_budget = w_info, w_budget
        self.marginal_cost = marginal_cost
        self.nt = len(T)

        sx, sy = NX / grid_x, NY / grid_y
        self.candidate_positions = [
            (min(int(gx*sx + sx/2), NX-1), min(int(gy*sy + sy/2), NY-1))
            for gx in range(grid_x) for gy in range(grid_y)
        ]
        self._precompute_field_stats()
        self.active_mask = None
        self.step_count = 0
        self.obs_dim = self.K + len(self.field_stats)

    def _precompute_field_stats(self):
        stats = []
        for (px, py) in self.candidate_positions:
            x0, x1 = max(0, px-2), min(NX, px+3)
            y0, y1 = max(0, py-2), min(NY, py+3)
            stats.append(0.6*float(self.T[:, x0:x1, y0:y1].var())
                         + 0.4*float(self.S[:, x0:x1, y0:y1].var()))
        stats = np.array(stats, dtype=np.float32)
        s_min, s_max = stats.min(), stats.max()
        self.field_stats = (stats - s_min) / (s_max - s_min + 1e-9)

    def reset(self):
        n_init = np.random.randint(self.n_min, self.n_max + 1)
        self.active_mask = np.zeros(self.K, dtype=np.float32)
        self.active_mask[np.random.choice(self.K, n_init, replace=False)] = 1.0
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.active_mask, self.field_stats])

    def _compute_info_reward(self):
        active_idx = np.where(self.active_mask > 0.5)[0]
        if len(active_idx) == 0:
            return 0.0
        coverage = float(self.field_stats[active_idx].sum()) / self.K
        if len(active_idx) > 1:
            pos = np.array([self.candidate_positions[i] for i in active_idx], dtype=np.float32)
            nn_d, _ = KDTree(pos).query(pos, k=2)
            spread = float(nn_d[:, 1].mean() / np.sqrt(NX**2 + NY**2))
        else:
            spread = 0.0
        return 0.7 * coverage + 0.3 * spread

    def step(self, action):
        assert 0 <= action < self.K
        prev_info = self._compute_info_reward()
        was_active = self.active_mask[action] > 0.5
        self.active_mask[action] = 0.0 if was_active else 1.0
        n_active = int(self.active_mask.sum())
        new_info = self._compute_info_reward()
        delta_info = new_info - prev_info

        if self.marginal_cost > 0:
            cost = self.marginal_cost if not was_active else -self.marginal_cost * 0.5
            reward = self.w_info * delta_info - cost
        else:
            penalty = 0.0
            if n_active < self.n_min:
                penalty = float(self.n_min - n_active) / self.n_min
            elif n_active > self.n_max:
                penalty = float(n_active - self.n_max) / self.n_max
            reward = self.w_info * delta_info - self.w_budget * penalty

        self.step_count += 1
        done = self.step_count >= self.ep_len
        return self._get_obs(), float(reward), done, {
            "n_active": n_active, "total_info": new_info, "delta_info": delta_info}


# =========================================================================
#  POLITIQUE PPO
# =========================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, obs, deterministic=False):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class RolloutBuffer:
    def __init__(self, sz, obs_dim):
        self.obs = np.zeros((sz, obs_dim), np.float32)
        self.actions = np.zeros(sz, np.int64)
        self.rewards = np.zeros(sz, np.float32)
        self.dones = np.zeros(sz, np.float32)
        self.log_probs = np.zeros(sz, np.float32)
        self.values = np.zeros(sz, np.float32)
        self.ptr = 0; self.size = sz

    def add(self, obs, a, r, d, lp, v):
        i = self.ptr
        self.obs[i]=obs; self.actions[i]=a; self.rewards[i]=r
        self.dones[i]=float(d); self.log_probs[i]=lp; self.values[i]=v
        self.ptr = (self.ptr+1) % self.size

    def compute_returns(self, last_v, gamma=0.99, lam=0.95):
        adv = np.zeros(self.size, np.float32); gae = 0.0
        for t in reversed(range(self.size)):
            nv = last_v if t==self.size-1 else self.values[t+1]
            nd = 0.0 if t==self.size-1 else self.dones[t+1]
            delta = self.rewards[t] + gamma*nv*(1-nd) - self.values[t]
            gae = delta + gamma*lam*(1-nd)*gae; adv[t] = gae
        return adv, adv + self.values

    def get_tensors(self, adv, ret, dev):
        return {"obs": torch.tensor(self.obs, device=dev),
                "actions": torch.tensor(self.actions, device=dev),
                "log_probs": torch.tensor(self.log_probs, device=dev),
                "advantages": torch.tensor(adv, device=dev),
                "returns": torch.tensor(ret, device=dev)}


# =========================================================================
#  ENTRAINEMENT PPO (commun)
# =========================================================================

def train_ppo(args, env, label=""):
    prefix = f" [{label}]" if label else ""
    print(f"  PPO{prefix} : {args.rl_steps} steps")

    policy = ActorCritic(env.obs_dim, env.K).to(DEVICE)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    buf = RolloutBuffer(args.buffer_size, env.obs_dim)
    clip_eps, vf_c, ent_c, n_ep, mb = 0.2, 0.5, 0.01, 4, 64
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    hist = {"episode_reward": [], "n_active": [], "info_score": []}
    ep_rews = deque(maxlen=20); best_rew = -np.inf
    obs = env.reset(); ep_r = 0.0

    for step in range(args.rl_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            act, lp, _, val = policy.get_action(obs_t)
        nobs, rew, done, info = env.step(act.item())
        buf.add(obs, act.item(), rew, done, lp.item(), val.item())
        ep_r += rew; obs = nobs
        if done:
            ep_rews.append(ep_r)
            hist["episode_reward"].append(ep_r)
            hist["n_active"].append(info["n_active"])
            hist["info_score"].append(info["total_info"])
            if ep_r > best_rew:
                best_rew = ep_r
                torch.save({"policy_state": policy.state_dict(),
                            "args": vars(args),
                            "active_mask": env.active_mask.copy()},
                           out_dir / "rl_best.pt")
            obs = env.reset(); ep_r = 0.0
        if (step+1) % args.buffer_size == 0:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                _, _, _, lv = policy.get_action(obs_t)
            adv, ret = buf.compute_returns(lv.item())
            adv = (adv-adv.mean())/(adv.std()+1e-8)
            batch = buf.get_tensors(adv, ret, DEVICE)
            idx = np.arange(args.buffer_size)
            for _ in range(n_ep):
                np.random.shuffle(idx)
                for s in range(0, args.buffer_size, mb):
                    m = idx[s:s+mb]
                    lo, va = policy(batch["obs"][m])
                    dist = torch.distributions.Categorical(logits=lo)
                    lp_ = dist.log_prob(batch["actions"][m])
                    ent = dist.entropy().mean()
                    ratio = torch.exp(lp_ - batch["log_probs"][m])
                    a = batch["advantages"][m]
                    loss = (-torch.min(ratio*a, torch.clamp(ratio,1-clip_eps,1+clip_eps)*a).mean()
                            + vf_c*F.mse_loss(va, batch["returns"][m]) - ent_c*ent)
                    optim.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5); optim.step()
            if len(ep_rews) > 0 and (step+1)%(args.buffer_size*5)==0:
                print(f"    Step {step+1:6d} | R={np.mean(ep_rews):+.3f} | Best={best_rew:+.3f}")
    print(f"  Best reward: {best_rew:.4f}")

    # Courbes
    sfx = f"_{label}" if label else ""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"PPO{prefix}", fontsize=14, fontweight="bold")
    axes[0,0].plot(hist["episode_reward"], alpha=0.4, color="steelblue")
    w = max(1, len(hist["episode_reward"])//20)
    if len(hist["episode_reward"]) >= w:
        sm = np.convolve(hist["episode_reward"], np.ones(w)/w, mode="valid")
        axes[0,0].plot(range(w-1, len(hist["episode_reward"])), sm, color="navy", lw=2)
    axes[0,0].set_title("Reward/ep"); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(hist["n_active"], color="orange", alpha=0.6)
    axes[0,1].axhline(env.n_min, color="red", ls="--"); axes[0,1].axhline(env.n_max, color="red", ls=":")
    axes[0,1].set_title("N actifs"); axes[0,1].grid(True, alpha=0.3)
    axes[1,0].plot(hist["info_score"], color="green", alpha=0.6)
    axes[1,0].set_title("Info score"); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(np.cumsum(hist["episode_reward"]), color="#9b59b6", alpha=0.8)
    axes[1,1].set_title("Reward cumulee"); axes[1,1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"rl_training_curves{sfx}.png", dpi=150); plt.close()
    return policy, hist


# =========================================================================
#  HELPERS
# =========================================================================

def _sweep_info(env, policy, n_range, n_trials=20):
    policy.eval(); points = []
    for nt_ in n_range:
        scores = []
        for trial in range(n_trials):
            env.active_mask[:] = 0.0
            env.active_mask[np.random.choice(env.K, min(nt_, env.K), replace=False)] = 1.0
            if trial < n_trials//2:
                obs = env._get_obs()
                with torch.no_grad():
                    for _ in range(env.ep_len):
                        ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        a, _, _, _ = policy.get_action(ot, deterministic=False)
                        obs, _, d, _ = env.step(a.item())
                        if d: break
            scores.append(env._compute_info_reward())
        points.append({"n_buoys": nt_, "info_mean": float(np.mean(scores)),
                        "info_std": float(np.std(scores))})
    return points

def _run_policy_config(env, policy, n_target):
    env.active_mask[:] = 0.0
    env.active_mask[np.random.choice(env.K, min(n_target, env.K), replace=False)] = 1.0
    obs = env._get_obs(); policy.eval()
    with torch.no_grad():
        for _ in range(env.ep_len):
            ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a, _, _, _ = policy.get_action(ot, deterministic=True)
            obs, _, d, _ = env.step(a.item())
            if d: break
    return np.where(env.active_mask > 0.5)[0], float(env._compute_info_reward())

def _n_light(n_star):
    nl = max(2, int(n_star)//2)
    if nl >= int(n_star): nl = max(2, int(n_star) - max(3, int(n_star)//3))
    return nl


# =========================================================================
#  METHODE 1 -- PARETO (Kneedle)
# =========================================================================

def compute_pareto(env, policy, args):
    print("\n-- Methode PARETO --")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_range = range(max(2, env.n_min-5), min(env.K, env.n_max+10))
    points = _sweep_info(env, policy, n_range)
    iv = np.array([p["info_mean"] for p in points])
    nv = np.array([p["n_buoys"] for p in points])
    ist = np.array([p["info_std"] for p in points])
    # Kneedle
    x0,y0 = float(nv[0]),float(iv[0]); x1,y1 = float(nv[-1]),float(iv[-1])
    nn_ = (nv-x0)/(x1-x0+1e-9); ii_ = (iv-y0)/(y1-y0+1e-9)
    dist = np.abs(ii_-nn_)/np.sqrt(2)
    conc = ii_ >= nn_-0.05
    if conc.any():
        cands = np.where(conc)[0]; elbow = cands[int(np.argmax(dist[cands]))]
    else:
        elbow = int(np.argmax(dist))
    if elbow <= 1: elbow = len(nv)//3
    elif elbow >= len(nv)-2: elbow = 2*len(nv)//3
    n_star = int(nv[elbow])
    # Pareto mask
    pmask = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        pmask[i] = not any((iv[j]>=iv[i] and nv[j]<=nv[i]) and (iv[j]>iv[i] or nv[j]<nv[i])
                           for j in range(len(points)) if j!=i)
    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PARETO -- Info vs N", fontsize=14, fontweight="bold")
    axes[0].fill_between(nv, iv-ist, iv+ist, alpha=0.2, color="steelblue")
    axes[0].plot(nv, iv, "o-", color="steelblue", ms=4, label="Info")
    axes[0].scatter(nv[pmask], iv[pmask], c=nv[pmask], cmap="plasma", s=120, zorder=5,
                    edgecolors="black", lw=0.8, label="Pareto")
    axes[0].axvline(n_star, color="red", lw=1.5, ls="--", label=f"N*={n_star}")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("N"); axes[0].set_ylabel("Info")
    mg = np.gradient(iv, nv)
    axes[1].bar(nv, mg, color=["#2ecc71" if g>0 else "#e74c3c" for g in mg], alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("N"); axes[1].set_ylabel("Gain marginal"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir/"rl_pareto_front.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"  N* = {n_star} (Kneedle)")
    return points, n_star


# =========================================================================
#  METHODE 2 -- EFFICIENCY eta(N) = info(N) / (1+log(N))
# =========================================================================

def compute_efficiency(env, policy, args):
    print("\n-- Methode EFFICIENCY --")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_range = range(max(2, env.n_min-5), min(env.K, env.n_max+10))
    points = _sweep_info(env, policy, n_range)
    iv = np.array([p["info_mean"] for p in points])
    nv = np.array([p["n_buoys"] for p in points])
    ist = np.array([p["info_std"] for p in points])
    eta = iv / (1.0 + np.log(nv.astype(float)))
    best = int(np.argmax(eta)); n_star = int(nv[best])
    # Figure 1x3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("EFFICIENCY -- eta = info / (1+log N)", fontsize=14, fontweight="bold")
    axes[0].fill_between(nv, iv-ist, iv+ist, alpha=0.2, color="steelblue")
    axes[0].plot(nv, iv, "o-", color="steelblue", ms=4)
    axes[0].axvline(n_star, color="red", lw=1.5, ls="--", label=f"N*={n_star}")
    axes[0].set_xlabel("N"); axes[0].set_ylabel("Info"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].plot(nv, eta, "s-", color="#e67e22", ms=5, lw=2)
    axes[1].scatter([n_star], [eta[best]], c="red", s=200, zorder=6, marker="*")
    axes[1].axvline(n_star, color="red", lw=1.5, ls="--")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("eta"); axes[1].set_title("Efficacite"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(nv, iv, "o-", color="steelblue", ms=3, label="Info")
    ax2 = axes[2].twinx()
    ax2.plot(nv, 1+np.log(nv.astype(float)), "^-", color="#e74c3c", ms=3, label="1+log(N)")
    axes[2].set_xlabel("N"); axes[2].set_ylabel("Info", color="steelblue")
    ax2.set_ylabel("Cout log", color="#e74c3c")
    l1,lb1 = axes[2].get_legend_handles_labels(); l2,lb2 = ax2.get_legend_handles_labels()
    axes[2].legend(l1+l2, lb1+lb2, fontsize=8); axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir/"rl_efficiency.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"  N* = {n_star} | eta* = {eta[best]:.4f}")
    return points, n_star


# =========================================================================
#  METHODE 3 -- SCALARIZED (sweep lambda)
# =========================================================================

def compute_scalarized(env_T, env_S, policy_std, args):
    print("\n-- Methode SCALARIZED (sweep lambda) --")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lambdas = [0.001, 0.005, 0.01, 0.02]
    results = []; steps_lam = max(1000, args.rl_steps // 4)
    for lam in lambdas:
        print(f"\n  lambda = {lam} ({steps_lam} steps)...")
        env_lam = OceanNetworkEnv(env_T, env_S, grid_x=args.grid_x, grid_y=args.grid_y,
                                   n_min=2, n_max=args.n_max+20,
                                   episode_len=args.episode_len, marginal_cost=lam)
        args_lam = argparse.Namespace(**vars(args)); args_lam.rl_steps = steps_lam
        pol_lam, _ = train_ppo(args_lam, env_lam, label=f"lam={lam}")
        idx, info = _run_policy_config(env_lam, pol_lam, env_lam.n_max)
        n_act = len(idx); eta = info / (1+np.log(max(2, n_act)))
        results.append({"lambda": lam, "n_active": n_act, "info": info, "eta": eta,
                         "policy": pol_lam, "active_idx": idx})
        print(f"    -> N={n_act} | info={info:.3f} | eta={eta:.4f}")
    best = max(results, key=lambda r: r["eta"]); n_star = best["n_active"]
    torch.save({"policy_state": best["policy"].state_dict(), "args": vars(args),
                "active_mask": np.zeros(0)}, out_dir/"rl_best.pt")
    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SCALARIZED -- Sweep lambda", fontsize=14, fontweight="bold")
    lams = [r["lambda"] for r in results]
    ns = [r["n_active"] for r in results]
    infos = [r["info"] for r in results]
    etas = [r["eta"] for r in results]
    axes[0].bar(range(len(lams)), ns, color=["#3498db","#2ecc71","#e67e22","#e74c3c"], alpha=0.8)
    axes[0].set_xticks(range(len(lams))); axes[0].set_xticklabels([f"l={l}" for l in lams])
    axes[0].set_ylabel("N capteurs"); axes[0].set_title("N par lambda"); axes[0].grid(True, alpha=0.3)
    sc = axes[1].scatter(ns, infos, c=lams, cmap="RdYlGn_r", s=200, zorder=5, edgecolors="black", lw=1.2)
    for r in results:
        axes[1].annotate(f"l={r['lambda']}", (r["n_active"], r["info"]),
                         textcoords="offset points", xytext=(8,5), fontsize=8)
    axes[1].scatter([best["n_active"]], [best["info"]], marker="*", c="red", s=400, zorder=6)
    plt.colorbar(sc, ax=axes[1], label="lambda")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("Info"); axes[1].set_title("Info vs N (*=best eta)"); axes[1].grid(True, alpha=0.3)
    colors = ["#e74c3c" if r is best else "#3498db" for r in results]
    axes[2].bar(range(len(lams)), etas, color=colors, alpha=0.8)
    axes[2].set_xticks(range(len(lams))); axes[2].set_xticklabels([f"l={l}" for l in lams])
    axes[2].set_ylabel("eta"); axes[2].set_title("eta = info/(1+log N)"); axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir/"rl_scalarized.png", dpi=150, bbox_inches="tight"); plt.close()
    points = [{"n_buoys": r["n_active"], "info_mean": r["info"], "info_std": 0.0} for r in results]
    print(f"\n  Best: lam={best['lambda']} -> N*={n_star} | eta={best['eta']:.4f}")
    return points, n_star


# =========================================================================
#  DISPATCH
# =========================================================================

def run_rl_method(env, policy, args):
    method = getattr(args, "rl_method", "pareto")
    if method == "efficiency":
        return compute_efficiency(env, policy, args)
    elif method == "scalarized":
        return compute_scalarized(env.T, env.S, policy, args)
    else:
        return compute_pareto(env, policy, args)


# =========================================================================
#  VISUALISATIONS
# =========================================================================

def visualize_two_configs(env, n_star, policy, args, best_mask=None):
    from matplotlib.colors import LinearSegmentedColormap
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    oc = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    BG = "#0a1628"; nl = _n_light(n_star)
    if best_mask is not None:
        env.active_mask = best_mask.copy()
        di = np.where(best_mask>0.5)[0]; dinf = float(env._compute_info_reward())
        dl, dn = "Dense (retenue)", "-> GNN & AE"
    else:
        di, dinf = _run_policy_config(env, policy, int(n_star))
        dl, dn = "Dense (N*)", f"N*={n_star}"
    li, linf = _run_policy_config(env, policy, nl)
    ap = np.array(env.candidate_positions); Tb = env.T[0]; vm,vM = float(env.T.min()),float(env.T.max())
    method = getattr(args, "rl_method", "pareto").upper()
    fig = plt.figure(figsize=(18,8), facecolor=BG)
    fig.suptitle(f"RL [{method}] -- Dense vs Legere", color="white", fontsize=13, fontweight="bold", y=0.99)
    for col,(idx,inf,lb,clr) in enumerate([(di,dinf,dl,"#6bcb77"),(li,linf,f"Legere (N~{nl})","#ffd93d")]):
        inact = np.setdiff1d(range(env.K), idx)
        ax = fig.add_axes([0.05+col*0.47, 0.10, 0.40, 0.80])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.imshow(Tb.T, cmap=oc, origin="lower", aspect="auto", vmin=vm, vmax=vM, alpha=0.5, extent=[0,NX,0,NY])
        ax.scatter(ap[inact,0], ap[inact,1], c="#1a3a5c", s=14, alpha=0.35)
        sc = ax.scatter(ap[idx,0], ap[idx,1], c=env.field_stats[idx], cmap="plasma",
                        s=90, vmin=0, vmax=1, edgecolors="white", lw=0.8, zorder=6)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label("Var", color="white", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
        ax.set_title(f"{lb}\nN={len(idx)} | Info={inf:.3f}", color=clr, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    lp = max(0, (dinf-linf)/max(dinf,1e-3)*100)
    fig.text(0.5, 0.02, f"Dense: N={len(di)} info={dinf:.3f} | Legere: N={len(li)} info={linf:.3f} | Perte: {lp:.1f}%",
             ha="center", color="#8ab4d4", fontsize=9)
    fig.savefig(out_dir/"rl_two_configs.png", dpi=150, facecolor=BG, bbox_inches="tight"); plt.close()
    print(f"  Dense: N={len(di)} info={dinf:.3f}")
    print(f"  Legere: N={len(li)} info={linf:.3f} (perte {lp:.1f}%)")

def visualize_final_config(env, active_mask, args):
    out_dir = Path(args.output_dir)
    ai = np.where(active_mask>0.5)[0]; ii = np.where(active_mask<=0.5)[0]
    ap = np.array(env.candidate_positions)
    method = getattr(args, "rl_method", "pareto").upper()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"RL [{method}] -- Config optimale", fontsize=13, fontweight="bold")
    axes[0].scatter(ap[ii,0], ap[ii,1], c="lightgray", s=30, alpha=0.4)
    sc = axes[0].scatter(ap[ai,0], ap[ai,1], c=env.field_stats[ai], cmap="YlOrRd",
                         s=120, edgecolors="black", lw=0.8, zorder=5)
    plt.colorbar(sc, ax=axes[0], label="Var locale")
    axes[0].set_xlim(0,NX); axes[0].set_ylim(0,NY)
    axes[0].set_title(f"Reseau ({len(ai)} bouees)"); axes[0].grid(True, alpha=0.2)
    axes[1].bar(range(len(ai)), np.sort(env.field_stats[ai])[::-1], color="steelblue")
    axes[1].set_title("Variance (decroissant)"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir/"rl_final_config.png", dpi=150); plt.close()

def save_rl_gif(env, policy, args, n_frames=80):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap
    out_dir = Path(args.output_dir)
    oc = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    BG = "#0a1628"; Tb = env.T[0]; vm,vM = float(env.T.min()),float(env.T.max())
    ca = np.array(env.candidate_positions, dtype=float)
    method = getattr(args, "rl_method", "pareto").upper()
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,5.5),facecolor=BG)
    for ax in (ax1,ax2,ax3):
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
    ax1.imshow(Tb.T, cmap=oc, origin="lower", aspect="auto", vmin=vm, vmax=vM, alpha=0.5, extent=[0,NX,0,NY])
    si = ax1.scatter([],[],c="#1a3a5c",s=14,alpha=0.4)
    sa = ax1.scatter([],[],c="white",s=60,edgecolors="black",lw=0.5,zorder=5)
    ax1.set_xlim(0,NX); ax1.set_ylim(0,NY); ax1.set_title("Actions",color="white",fontsize=9,fontweight="bold")
    ax2.imshow(Tb.T, cmap=oc, origin="lower", aspect="auto", vmin=vm, vmax=vM, alpha=0.5, extent=[0,NX,0,NY])
    si2 = ax2.scatter([],[],c="#1a3a5c",s=14,alpha=0.3)
    sa2 = ax2.scatter([],[],c="white",s=70,edgecolors="white",lw=0.6,zorder=5)
    ax2.set_xlim(0,NX); ax2.set_ylim(0,NY); ax2.set_title("Reseau",color="white",fontsize=9,fontweight="bold")
    ax3.set_xlim(0,n_frames); ax3.set_ylim(0,0.5)
    info_line, = ax3.plot([],[],color="#6bcb77",lw=2,label="Info score")
    ax3_n = ax3.twinx()
    ax3_n.set_ylim(0, env.n_max+5)
    n_line, = ax3_n.plot([],[],color="#ffd93d",lw=1.5,alpha=0.7,label="N actifs")
    ax3_n.tick_params(colors="#ffd93d",labelsize=6)
    vl = ax3.axvline(0,color="white",lw=0.5,alpha=0.3)
    ax3.set_title("Info & N actifs",color="white",fontsize=9,fontweight="bold")
    ax3.tick_params(colors="#6bcb77",labelsize=6)
    ax3.set_ylabel("Info", color="#6bcb77", fontsize=7)
    ax3_n.set_ylabel("N", color="#ffd93d", fontsize=7)
    txt = fig.text(0.5,0.97,"",ha="center",color="white",fontsize=10,fontweight="bold")
    obs = env.reset(); rx,ry_info,ry_n=[],[],[]; el=[]
    def update(f):
        nonlocal obs,el
        if f==0: obs=env.reset(); rx.clear(); ry_info.clear(); ry_n.clear()
        ot = torch.tensor(obs,dtype=torch.float32,device=DEVICE).unsqueeze(0)
        with torch.no_grad(): a,_,_,_ = policy.get_action(ot)
        obs,r,d,info = env.step(a.item())
        cur_info = env._compute_info_reward()
        if d: obs=env.reset()
        ai=np.where(env.active_mask>0.5)[0]; ii=np.where(env.active_mask<=0.5)[0]
        n_active = len(ai)
        si.set_offsets(ca[ii] if len(ii) else np.empty((0,2)))
        sa.set_offsets(ca[ai] if n_active else np.empty((0,2)))
        for ln in el: ln.remove()
        el=[]
        if n_active>1:
            pa=ca[ai]; tree=KDTree(pa)
            for i in range(len(pa)):
                _,idxs=tree.query(pa[i],k=min(3,len(pa)))
                for j in idxs[1:]:
                    ln,=ax2.plot([pa[i,0],pa[j,0]],[pa[i,1],pa[j,1]],color="#2e75b6",alpha=0.5,lw=1.2)
                    el.append(ln)
        si2.set_offsets(ca[ii] if len(ii) else np.empty((0,2)))
        if n_active: sa2.set_offsets(ca[ai]); sa2.set_color(plt.cm.YlOrRd(env.field_stats[ai]))
        else: sa2.set_offsets(np.empty((0,2)))
        rx.append(f); ry_info.append(cur_info); ry_n.append(n_active)
        info_line.set_data(rx,ry_info); n_line.set_data(rx,ry_n)
        vl.set_xdata([f,f])
        # Auto-scale Y
        if len(ry_info) > 1:
            ax3.set_ylim(0, max(ry_info)*1.3+0.01)
        txt.set_text(f"RL [{method}] | Step {f+1}/{n_frames} | N={n_active} | Info={cur_info:.3f}")
        return (si,sa,si2,sa2,info_line,n_line,vl,txt)
    anim = FuncAnimation(fig,update,frames=n_frames,interval=200,blit=False)
    anim.save(str(out_dir/"rl_progression.gif"), writer=PillowWriter(fps=6), dpi=110,
              savefig_kwargs={"facecolor":BG})
    plt.close(); print(f"  GIF -> {out_dir}/rl_progression.gif")

def mark_retained_config_on_pareto(n_ret, info_ret, out_dir):
    import matplotlib.image as mpimg
    out_dir = Path(out_dir)
    for src_name in ["rl_pareto_front.png","rl_efficiency.png","rl_scalarized.png"]:
        src = out_dir/src_name
        if src.exists():
            img = mpimg.imread(str(src))
            fig,ax = plt.subplots(figsize=(14,6),dpi=150)
            ax.imshow(img); ax.axis("off")
            fig.text(0.5,0.01,f"* Config retenue: N={n_ret} | info={info_ret:.3f}",
                     ha="center",color="#ffd93d",fontsize=10,fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3",facecolor="#0a1628",edgecolor="#ffd93d",alpha=0.9))
            fig.savefig(out_dir/f"{src.stem}_pipeline.png",dpi=150,bbox_inches="tight",facecolor="#0a1628")
            plt.close(); break


# =========================================================================
#  POINT D'ENTREE
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Brique 3 -- RL pour OED")
    p.add_argument("--train", action="store_true")
    p.add_argument("--evaluate", action="store_true", help="Evalue avec la methode choisie")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--rl_method", choices=["pareto","efficiency","scalarized"],
                   default="pareto", help="Methode de selection N*")
    p.add_argument("--seed_ocean", type=int, default=42)
    p.add_argument("--seed_buoys", type=int, default=7)
    p.add_argument("--checkpoint", type=str, default="outputs/rl_best.pt")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--rl_steps", type=int, default=50000)
    p.add_argument("--buffer_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--n_min", type=int, default=10)
    p.add_argument("--n_max", type=int, default=40)
    p.add_argument("--episode_len", type=int, default=20)
    p.add_argument("--w_info", type=float, default=1.0)
    p.add_argument("--w_budget", type=float, default=0.5)
    p.add_argument("--gif_frames", type=int, default=80)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.train and not args.evaluate and not args.gif:
        print("Usage: python 03_rl.py --train [--evaluate] [--gif] [--rl_method pareto|efficiency|scalarized]")
        sys.exit(0)
    set_global_seed(args.seed_ocean)
    print(f"\n  Methode : {args.rl_method.upper()}")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=NT, seed=args.seed_ocean)
    env = OceanNetworkEnv(T, S, grid_x=args.grid_x, grid_y=args.grid_y,
                          n_min=args.n_min, n_max=args.n_max,
                          episode_len=args.episode_len, w_info=args.w_info, w_budget=args.w_budget)
    print(f"  K={env.K} | Budget [{args.n_min}, {args.n_max}]")
    policy = None
    if args.train:
        policy, _ = train_ppo(args, env)
        ckpt = torch.load(Path(args.output_dir)/"rl_best.pt", map_location=DEVICE, weights_only=False)
        visualize_final_config(env, ckpt["active_mask"], args)
        save_rl_gif(env, policy, args, n_frames=args.gif_frames)
    if args.evaluate:
        if policy is None:
            policy = ActorCritic(env.obs_dim, env.K).to(DEVICE)
            cp = Path(args.output_dir)/"rl_best.pt"
            if cp.exists():
                policy.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=False)["policy_state"])
        pts, n_star = run_rl_method(env, policy, args)
        visualize_two_configs(env, n_star, policy, args)
    if args.gif and policy is None:
        policy = ActorCritic(env.obs_dim, env.K).to(DEVICE)
        cp = Path(args.output_dir)/"rl_best.pt"
        if cp.exists():
            policy.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=False)["policy_state"])
        save_rl_gif(env, policy, args, n_frames=args.gif_frames)
    print("\n  Brique 3 terminee.")
