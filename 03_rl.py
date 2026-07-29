"""

         BRIQUE 3  Reinforcement Learning : Optimisation du Réseau          
                                                                              
  Formalisation MDP :                                                         
    État    s_t : masque binaire actuel (grille grossière) + stats champ     
    Action  a_t : activer / désactiver une des K positions candidates        
    Récompense  : gain de reconstruction RMSE  pénalité budget bouées      
                                                                              
  Algorithme : PPO (Proximal Policy Optimization) implémenté en PyTorch pur  
  Multi-objectif : front de Pareto information vs nombre de capteurs         
                                                                              
  Usage :                                                                     
    python 03_rl.py --train                                                   
    python 03_rl.py --pareto           (front Pareto info/nb capteurs)       
    python 03_rl.py --train --pareto                                          

"""

import sys, argparse
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data.loader import load_ocean, add_data_args

# Renseignés à la construction de l'environnement (récompense AE)
_CH_NAMES = []
try:
    _OBSERVED = tuple(OBSERVED_VARS)
except NameError:
    _OBSERVED = ("thetao", "so")

#  Import optionnel de la Brique 1 pour la récompense dense 
try:
    from brique1_autoencoder import ObservabilityVAE
    AE_AVAILABLE = True
except ImportError:
    AE_AVAILABLE = False


# 
#  ENVIRONNEMENT MDP
# 

class OceanNetworkEnv:
    """
    Environnement RL pour l'optimisation du réseau d'observation.

    Grille candidate (coarse grid) :
        On discrétise l'espace en une grille GX  GY de positions candidates.
        L'espace d'action est donc de taille K = GX  GY (toggle par position).
        Le budget de bouées actives est contraint : [n_min, n_max].

    État s_t : (K + n_stats,) float32
        - K premiers éléments : masque binaire des positions actives
        - n_stats derniers : statistiques globales du champ nature run
          (variance locale agrégée, gradient moyen...)

    Récompense r_t :
        r_t = w_info * r_info  w_budget * r_budget

        r_info    = amélioration de la couverture pondérée variance
                    (proxy : couverture variance locale)
        r_budget  = pénalité si hors de la plage [n_min, n_max] bouées actives

    Épisode :
        T_ep actions consécutives  à la fin, évaluation de la configuration finale
    """

    def __init__(self, T, S=None,
                 grid_x=16, grid_y=24,    # résolution de la grille candidate
                 n_min=10, n_max=40,       # plage de bouées actives
                 episode_len=20,           # actions par épisode
                 w_info=1.0, w_budget=0.5,
                 ae_model=None,            # autoencoder optionnel (Brique 1)
                 ae_n_dates=8,             # dates évaluées par récompense AE
                 channels=None,            # noms de canaux (pour obs/cible)
                 sea_mask=None,            # masque océanique (nx, ny)
                 dx_km=None):              # résolution, pour un Pareto en km
        global _CH_NAMES, _OBSERVED
        # Noms de canaux : fournis explicitement, ou reconstruits par défaut.
        if channels is not None:
            _CH_NAMES = list(channels)
        elif not _CH_NAMES:
            _CH_NAMES = [f"c{i}_z0" for i in range(
                T.shape[1] if T.ndim == 4 else (1 if S is None else 2))]
        self._ae_n_dates = ae_n_dates
        # `T` accepte deux formes :
        #   - (nt, nx, ny)        : champ unique, mode legacy
        #   - (nt, n_ch, nx, ny)  : tenseur multi-canaux GLORYS
        if T.ndim == 4:
            self.fields = T.astype(np.float32)
        else:
            arrs = [T] if S is None else [T, S]
            self.fields = np.stack(arrs, axis=1).astype(np.float32)
        self.n_ch = self.fields.shape[1]
        self.sea_mask = (np.ones(self.fields.shape[2:], dtype=bool)
                         if sea_mask is None else np.asarray(sea_mask, bool))
        self.dx_km = dx_km
        # Rétro-compatibilité : les figures accèdent encore à env.T / env.S
        self.T = self.fields[:, 0]
        self.S = self.fields[:, min(1, self.n_ch - 1)]
        self.grid_x  = grid_x
        self.grid_y  = grid_y
        self.K       = grid_x * grid_y    # nb de positions candidates
        self.n_min   = n_min
        self.n_max   = n_max
        self.ep_len  = episode_len
        self.w_info, self.w_budget = w_info, w_budget
        self.ae_model = ae_model
        self.nt = len(self.fields)

        #  Récompense fondée sur l'autoencodeur 
        # Si un modèle AE est fourni, la récompense d'information devient
        # RMSE de reconstruction sur les pixels non observés  la MÊME
        # métrique que la baseline. C'est ce qui aligne l'objectif du RL sur
        # ce qu'on évalue réellement, au lieu du proxy  variance locale  qui
        # s'est révélé anti-corrélé à la reconstruction.
        self._ae_cfg = None
        self._ae_cache = {}          # {clé de masque: info}  mémoïsation
        if ae_model is not None:
            ae_model.eval()
            self._setup_ae_reward()

        # Positions physiques des K candidats (centre de chaque cellule)
        self.candidate_positions = []
        sx = NX / grid_x
        sy = NY / grid_y
        for gx in range(grid_x):
            for gy in range(grid_y):
                px = min(int(gx * sx + sx / 2), NX - 1)
                py = min(int(gy * sy + sy / 2), NY - 1)
                # Un candidat sur la terre serait une bouée impossible à
                # déployer. Sans terre dans la fenêtre, ce filtre est neutre.
                if self.sea_mask[px, py]:
                    self.candidate_positions.append((px, py))
        # K n'est plus grid_x*grid_y dès qu'un candidat est écarté : il
        # dimensionne l'espace d'action du PPO et doit rester cohérent.
        self.K = len(self.candidate_positions)
        if self.K < n_max:
            raise ValueError(f"{self.K} candidats en mer < n_max={n_max}.")

        # Statistiques globales du nature run (pré-calculées une fois)
        self._precompute_field_stats()

        # État courant
        self.active_mask = None    # (K,) binaire
        self.step_count  = 0
        self._cur_info   = None    # cache de l'info de l'état courant
        self.t_current   = 0
        self.obs_dim = self.K + len(self.field_stats)

    def _precompute_field_stats(self):
        """
        Variance locale et gradient moyen par cellule candidate.
        Ces statistiques encodent la "difficulté" de chaque zone à reconstruire.
        """
        # Normalisation PAR CANAL avant tout calcul de variance.
        # Sans elle, la variance en °C (O(0.1)) écrase totalement celle des
        # courants en m/s (O(0.001)) : le RL n'optimiserait que pour la
        # température, et les canaux uo/vo n'auraient aucun poids.
        mu = self.fields.mean(axis=(0, 2, 3), keepdims=True)
        sd = self.fields.std(axis=(0, 2, 3), keepdims=True) + 1e-9
        Fn = (self.fields - mu) / sd

        stats = []
        for (px, py) in self.candidate_positions:
            # Fenêtre locale 55 autour de la position
            x0, x1 = max(0, px-2), min(NX, px+3)
            y0, y1 = max(0, py-2), min(NY, py+3)
            win_sea = self.sea_mask[x0:x1, y0:y1]
            sub_f = Fn[:, :, x0:x1, y0:y1]
            if win_sea.all():
                v = sub_f.var(axis=(0, 2, 3))
            else:
                # Exclure la terre : sa valeur constante ferait chuter la
                # variance et sous-estimerait l'intérêt des zones côtières.
                v = sub_f[:, :, win_sea].var(axis=(0, 2))
            stats.append(float(v.mean()))

        stats = np.array(stats, dtype=np.float32)
        # Normalisation
        stats = (stats - stats.mean()) / (stats.std() + 1e-9)
        self.field_stats = stats      # (K,)  variance locale normalisée par candidat

    def reset(self):
        """Initialise un épisode : placement aléatoire de n_init bouées."""
        n_init = np.random.randint(self.n_min, self.n_max + 1)
        self.active_mask = np.zeros(self.K, dtype=np.float32)
        init_idx = np.random.choice(self.K, n_init, replace=False)
        self.active_mask[init_idx] = 1.0
        self.step_count = 0
        self.t_current  = np.random.randint(0, self.nt)
        self._cur_info  = None      # info de l'état courant, calculée à la demande
        return self._get_obs()

    def _get_obs(self):
        """Vecteur d'état : masque actif  statistiques du champ."""
        return np.concatenate([self.active_mask, self.field_stats])

    def _setup_ae_reward(self):
        """Pré-calcule ce qui ne dépend pas du masque : normalisation, indices."""
        F = self.fields
        mean = F.mean(axis=(0, 2, 3), keepdims=True)
        std = F.std(axis=(0, 2, 3), keepdims=True) + 1e-9
        fields_n = ((F - mean) / std).astype(np.float32)   # normalisé par canal

        obs_idx = [i for i in range(self.n_ch)
                   if _CH_NAMES[i].rsplit("_z", 1)[0] in _OBSERVED]
        self._ae_cfg = {
            "fields_n": fields_n,
            "obs_idx": obs_idx,
            "std": std.squeeze(),
            "sea_f": self.sea_mask.astype(np.float32),
            "device": next(self.ae_model.parameters()).device,
        }

    def _ae_info(self, active_idx):
        """
        Récompense d'information = RMSE de reconstruction AE sur les pixels
        non observés et en mer, moyennée sur un petit lot de dates.

        Mémoïsée par masque : dans un épisode, beaucoup d'états se répètent
        (toggle puis re-toggle), donc le cache évite des forward AE inutiles.
        Sans MC-Dropout ici : on veut un signal rapide, la variance de
        Dropout est du bruit qui ralentit l'apprentissage RL.
        """
        import torch
        cfg = self._ae_cfg

        # Masque binaire  clé de cache compacte
        key = tuple(sorted(active_idx.tolist()))
        if key in self._ae_cache:
            return self._ae_cache[key]

        mask = np.zeros((NX, NY), dtype=np.float32)
        for i in active_idx:
            px, py = self.candidate_positions[i]
            mask[px, py] = 1.0

        idx = np.random.choice(len(cfg["fields_n"]), self._ae_n_dates,
                               replace=False)
        dev = cfg["device"]
        mt = torch.from_numpy(mask[None]).to(dev)
        w = torch.from_numpy((1.0 - mask) * cfg["sea_f"]).to(dev)

        batch = torch.from_numpy(cfg["fields_n"][idx]).to(dev)
        obs = batch[:, cfg["obs_idx"]]
        nd = batch.shape[0]
        x = torch.cat([obs * mt[None], mt[None].expand(nd, -1, -1, -1)], dim=1)

        self.ae_model.eval()
        with torch.no_grad():
            pred = self.ae_model(x)[0]
            rmse = torch.sqrt(((pred - batch) ** 2 * w).sum()
                              / w.sum().clamp_min(1.0))
        # Récompense = RMSE, recentrée pour rester du même ordre que le proxy
        info = float(-rmse.item())
        self._ae_cache[key] = info
        return info

    def _compute_info_reward(self):
        """
        Qualité informative du réseau courant.

        Deux régimes :
          - AE branché   RMSE de reconstruction (métrique alignée sur
                          l'évaluation, cf. brique 4).
          - sinon        proxy variance-pondérée + espacement (rapide, mais
                          anti-corrélé à la reconstruction  à n'utiliser que
                          pour un prototypage sans AE).
        """
        active_idx = np.where(self.active_mask > 0.5)[0]
        if len(active_idx) == 0:
            return -1.0 if self.ae_model is not None else 0.0

        if self.ae_model is not None:
            return self._ae_info(active_idx)

        #  Proxy sans AE (ancien comportement) 
        coverage_score = float(self.field_stats[active_idx].mean())
        if len(active_idx) > 1:
            positions_active = np.array([self.candidate_positions[i]
                                         for i in active_idx], dtype=np.float32)
            tree = KDTree(positions_active)
            nn_dists, _ = tree.query(positions_active, k=2)
            mean_nn_dist = nn_dists[:, 1].mean()
            max_dist = np.sqrt(NX**2 + NY**2)
            spread_bonus = float(mean_nn_dist / max_dist)
            self.last_mean_nn_km = (float(mean_nn_dist * self.dx_km)
                                    if self.dx_km else None)
        else:
            spread_bonus = 0.0
        return 0.7 * coverage_score + 0.3 * spread_bonus

    def step(self, action):
        """
        Action : toggle de la position candidate `action` (0..K-1).
        
        Retourne : (obs, reward, done, info)
        """
        assert 0 <= action < self.K, f"Action invalide : {action}"
        # `prev_info` = info de l'état courant. On le mémorise d'un step à
        # l'autre (self._cur_info) : sans ça, avec la récompense AE, on
        # paierait DEUX forward AE par step au lieu d'un seul.
        if self._cur_info is None:
            self._cur_info = self._compute_info_reward()
        prev_info = self._cur_info
        prev_n_active = int(self.active_mask.sum())

        # Toggle
        was_active = self.active_mask[action] > 0.5
        self.active_mask[action] = 0.0 if was_active else 1.0
        n_active = int(self.active_mask.sum())

        # Information après action
        new_info = self._compute_info_reward()
        self._cur_info = new_info          # devient le prev_info du prochain step
        delta_info = new_info - prev_info

        # Pénalité budget (hors de la plage autorisée)
        budget_penalty = 0.0
        if n_active < self.n_min:
            budget_penalty = float(self.n_min - n_active) / self.n_min
        elif n_active > self.n_max:
            budget_penalty = float(n_active - self.n_max) / self.n_max

        reward = (self.w_info   * delta_info
                  - self.w_budget * budget_penalty)

        self.step_count += 1
        done = (self.step_count >= self.ep_len)

        info = {
            "n_active":       n_active,
            "delta_info":     delta_info,
            "budget_penalty": budget_penalty,
            "total_info":     new_info,
        }
        return self._get_obs(), float(reward), done, info


# 
#  POLITIQUE PPO  Actor-Critic
# 

class ActorCritic(nn.Module):
    """
    Réseau actor-critic partagé pour PPO.

    Architecture :
        Tronc commun MLP (obs_dim  256  256)
         Actor   logits (K actions)  distribution catégorielle
         Critic  valeur d'état V(s) (scalaire)

    L'entrée mélange deux types d'information :
        - masque binaire actif (sparse) : traité via embedding
        - statistiques continues du champ
    On les concatène et on passe dans le MLP commun.
    """
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        # Initialisation orthogonale (recommandée pour PPO)
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


# 
#  ROLLOUT BUFFER
# 

class RolloutBuffer:
    """
    Stocke les transitions (obs, action, reward, done, log_prob, value)
    pour un mini-batch PPO.
    """
    def __init__(self, buffer_size, obs_dim):
        self.obs       = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions   = np.zeros(buffer_size, dtype=np.int64)
        self.rewards   = np.zeros(buffer_size, dtype=np.float32)
        self.dones     = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values    = np.zeros(buffer_size, dtype=np.float32)
        self.ptr       = 0
        self.size      = buffer_size

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr = (self.ptr + 1) % self.size

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        """GAE-λ : advantage estimé par Generalized Advantage Estimation."""
        advantages = np.zeros(self.size, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_val = last_value
                next_done = 0.0
            else:
                next_val  = self.values[t + 1]
                next_done = self.dones[t + 1]
            delta = (self.rewards[t]
                     + gamma * next_val * (1 - next_done)
                     - self.values[t])
            gae = delta + gamma * lam * (1 - next_done) * gae
            advantages[t] = gae
        returns = advantages + self.values
        return advantages, returns

    def get_tensors(self, advantages, returns, device):
        return {
            "obs":       torch.tensor(self.obs,       device=device),
            "actions":   torch.tensor(self.actions,   device=device),
            "log_probs": torch.tensor(self.log_probs, device=device),
            "advantages":torch.tensor(advantages,     device=device),
            "returns":   torch.tensor(returns,        device=device),
        }


# 
#  ENTRAÎNEMENT PPO
# 

def train_ppo(args, env):
    """
    Boucle d'entraînement PPO complète.

    Hyperparamètres clés :
        clip_eps     : clipping du ratio de politique (0.2 standard)
        entropy_coef : encourage l'exploration (0.01)
        vf_coef      : pondération de la value loss (0.5)
        n_epochs_ppo : passes sur chaque mini-batch (4)
    """
    print("" * 60)
    print(" Brique 3  PPO : Optimisation du Réseau d'Observation")
    print("" * 60)

    obs_dim   = env.obs_dim
    n_actions = env.K
    policy    = ActorCritic(obs_dim, n_actions).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\n  Politique PPO  {n_params:,} paramètres")
    print(f"  Espace d'état  : {obs_dim} dim")
    print(f"  Espace d'action: {n_actions} positions candidates")

    buffer     = RolloutBuffer(args.buffer_size, obs_dim)
    clip_eps   = 0.2
    vf_coef    = 0.5
    ent_coef   = 0.01
    n_ppo_ep   = 4
    mini_batch = 64

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"episode_reward": [], "n_active": [], "info_score": []}
    ep_rewards = deque(maxlen=20)
    best_reward = -np.inf

    obs = env.reset()
    ep_reward = 0.0
    global_step = 0

    print(f"\n  Entraînement : {args.rl_steps} steps | buffer={args.buffer_size}")
    print("" * 60)

    for step in range(args.rl_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, entropy, value = policy.get_action(obs_t)
        a = action.item()
        lp = log_prob.item()
        v  = value.item()

        next_obs, reward, done, info = env.step(a)
        buffer.add(obs, a, reward, done, lp, v)
        ep_reward += reward
        global_step += 1
        obs = next_obs

        if done:
            ep_rewards.append(ep_reward)
            history["episode_reward"].append(ep_reward)
            history["n_active"].append(info["n_active"])
            history["info_score"].append(info["total_info"])
            

            if ep_reward > best_reward:
                best_reward = ep_reward
                torch.save({"policy_state": policy.state_dict(),
                            "args": vars(args),
                            "active_mask": env.active_mask.copy()},
                           out_dir / "rl_best.pt")

            obs = env.reset()
            ep_reward = 0.0

        #  Mise à jour PPO tous les buffer_size steps 
        if (step + 1) % args.buffer_size == 0:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                _, _, _, last_value = policy.get_action(obs_t)
                last_v = last_value.item()

            advantages, returns = buffer.compute_returns(last_v)
            # Normalisation des avantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            batch = buffer.get_tensors(advantages, returns, DEVICE)

            # PPO updates
            idx = np.arange(args.buffer_size)
            for _ in range(n_ppo_ep):
                np.random.shuffle(idx)
                for start in range(0, args.buffer_size, mini_batch):
                    end  = start + mini_batch
                    mb   = idx[start:end]
                    obs_mb = batch["obs"][mb]
                    act_mb = batch["actions"][mb]
                    lp_old = batch["log_probs"][mb]
                    adv_mb = batch["advantages"][mb]
                    ret_mb = batch["returns"][mb]

                    logits, values = policy(obs_mb)
                    dist    = torch.distributions.Categorical(logits=logits)
                    lp_new  = dist.log_prob(act_mb)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(lp_new - lp_old)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
                    actor_loss  = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values, ret_mb)
                    loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

            if len(ep_rewards) > 0 and (step + 1) % (args.buffer_size * 5) == 0:
                print(f"  Step {step+1:6d} | "
                      f"Mean reward (20 ep) = {np.mean(ep_rewards):+.3f} | "
                      f"Best = {best_reward:+.3f}")

    print(f"\n   Meilleure récompense : {best_reward:.4f}")
    print(f"   Checkpoint  {out_dir}/rl_best.pt")

    #  Courbes d'apprentissage 
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Brique 3  PPO : Courbes d'entraînement", fontsize=14, fontweight="bold")

    axes[0, 0].plot(history["episode_reward"], alpha=0.4, color="steelblue")
    # Moyenne glissante
    w = max(1, len(history["episode_reward"]) // 20)
    smooth = np.convolve(history["episode_reward"],
                         np.ones(w)/w, mode="valid")
    axes[0, 0].plot(range(w-1, len(history["episode_reward"])), smooth, color="navy", lw=2)
    axes[0, 0].set_title("Récompense par épisode"); axes[0, 0].set_xlabel("Épisode")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["n_active"], color="orange", alpha=0.6)
    axes[0, 1].axhline(env.n_min, color="red", linestyle="--", label=f"n_min={env.n_min}")
    axes[0, 1].axhline(env.n_max, color="red", linestyle=":",  label=f"n_max={env.n_max}")
    axes[0, 1].set_title("Nombre de capteurs actifs en fin d'épisode")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["info_score"], color="green", alpha=0.6)
    axes[1, 0].set_title("Score d'information (couverture pondérée variance)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["info_score"], color="#74c476", alpha=0.8)
    axes[1, 1].set_title("Score d information (couverture pondérée variance)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "rl_training_curves.png", dpi=150)
    plt.close()
    print(f"   Courbes  {out_dir}/rl_training_curves.png")

    return policy


# 
#  FRONT DE PARETO  Optimisation sous contrainte budgétaire
# 

def compute_pareto_front(env, policy, args):
    """
    Courbe rendement décroissant : score d'information vs nombre de capteurs.

    Pour chaque N dans [n_min-5, n_max+10] on tire 30 configurations aléatoires
    et on mesure le score d'information moyen. La courbe montre le point de
    rendement décroissant : au-delà d'un certain N, ajouter un capteur
    n'améliore plus significativement la couverture.

    Les points "non-dominés" (info élevée pour N bas) sont mis en évidence.
    """
    print("\n Courbe Information vs Nombre de capteurs ")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Le balayage doit rester DANS le budget sous lequel la politique a été
    # entraînée. L'ancienne plage [n_min-5, n_max+10] explorait des
    # configurations que le MDP pénalise explicitement (budget_penalty), et
    # le point de coude pouvait donc tomber sous n_min  recommandation
    # inapplicable et incohérente avec l'énoncé du problème.
    n_range = range(env.n_min, min(env.K, env.n_max) + 1)
    pareto_points = []

    print("  Sweep sur le nombre de bouées actives...")
    for n_target in n_range:
        info_scores = []
        for _ in range(30):
            mask = np.zeros(env.K, dtype=np.float32)
            idx  = np.random.choice(env.K, n_target, replace=False)
            mask[idx] = 1.0
            env.active_mask = mask.copy()
            info_scores.append(env._compute_info_reward())
        pareto_points.append({
            "n_buoys":   n_target,
            "info_mean": float(np.mean(info_scores)),
            "info_std":  float(np.std(info_scores)),
        })

    #  Points non-dominés : info élevée ET N bas 
    info_vals = np.array([p["info_mean"] for p in pareto_points])
    n_vals    = np.array([p["n_buoys"]   for p in pareto_points])

    pareto_mask = np.zeros(len(pareto_points), dtype=bool)
    for i in range(len(pareto_points)):
        dominated = False
        for j in range(len(pareto_points)):
            if j == i: continue
            # j domine i si : plus d'info avec moins (ou égal) de capteurs
            if info_vals[j] >= info_vals[i] and n_vals[j] <= n_vals[i]:
                if info_vals[j] > info_vals[i] or n_vals[j] < n_vals[i]:
                    dominated = True; break
        pareto_mask[i] = not dominated

    #  Figure 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Brique 3  Information vs Nombre de capteurs",
                 fontsize=14, fontweight="bold")

    # Panneau gauche : scatter info vs N avec enveloppe σ
    ax = axes[0]
    info_stds = np.array([p["info_std"] for p in pareto_points])
    ax.fill_between(n_vals, info_vals - info_stds, info_vals + info_stds,
                    alpha=0.2, color="steelblue", label="1σ (variabilité configs)")
    ax.plot(n_vals, info_vals, "o-", color="steelblue", alpha=0.7,
            markersize=4, label="Info score moyen")

    # Points Pareto en évidence
    pf_n    = n_vals[pareto_mask]
    pf_info = info_vals[pareto_mask]
    sc = ax.scatter(pf_n, pf_info, c=pf_n, cmap="plasma",
                    s=120, zorder=5, edgecolors="black", linewidths=0.8,
                    label="Configurations Pareto-optimales")
    plt.colorbar(sc, ax=ax, label="Nombre de bouées")

    # Annotation rendement décroissant : point du coude
    grad = np.gradient(info_vals, n_vals)
    elbow_idx = int(np.argmax(np.abs(np.gradient(grad, n_vals))))
    ax.axvline(n_vals[elbow_idx], color="red", lw=1.5, linestyle="--", alpha=0.7,
               label=f"Coude (N={n_vals[elbow_idx]})")
    ax.annotate(f"N={n_vals[elbow_idx]}", (n_vals[elbow_idx], info_vals[elbow_idx]),
                textcoords="offset points", xytext=(8, -15),
                fontsize=10, color="red", fontweight="bold")

    ax.set_xlabel("Nombre de capteurs actifs")
    ax.set_ylabel("Score d'information (couverture pondérée variance)")
    ax.set_title("Rendement décroissant\n(rouge = point de coude optimal)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panneau droit : gain marginal (dérivée)
    ax2 = axes[1]
    marginal_gain = np.gradient(info_vals, n_vals)
    colors_mg = ["#2ecc71" if g > 0 else "#e74c3c" for g in marginal_gain]
    ax2.bar(n_vals, marginal_gain, color=colors_mg, alpha=0.8, edgecolor="gray", lw=0.3)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(marginal_gain.max() * 0.05, color="orange", lw=1.5,
                linestyle="--", alpha=0.8, label="Seuil 5% du gain max")

    # Régions Pareto en vert
    for i, p in enumerate(pareto_points):
        if pareto_mask[i]:
            ax2.axvspan(p["n_buoys"] - 0.5, p["n_buoys"] + 0.5,
                        alpha=0.12, color="green")

    ax2.set_xlabel("Nombre de capteurs actifs")
    ax2.set_ylabel("Gain marginal d'information (ΔInfo / ΔN)")
    ax2.set_title("Gain marginal par capteur ajouté\n(vert = zones Pareto-optimales)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "rl_pareto_front.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure  {out_dir}/rl_pareto_front.png")

    print(f"\n Recommandations ")
    print(f"  Point de coude : N = {n_vals[elbow_idx]} capteurs "
          f"(info={info_vals[elbow_idx]:.3f})")
    print(f"  {pareto_mask.sum()} configurations Pareto-optimales :")
    for i, p in enumerate(pareto_points):
        if pareto_mask[i]:
            print(f"    n={p['n_buoys']:2d} | info={p['info_mean']:.3f} {p['info_std']:.3f}")

    return pareto_points, pareto_mask, n_vals[elbow_idx]


# 
#  DEUX CONFIGURATIONS RÉSEAU : DENSE (optimal) + LÉGÈRE (~50%)
# 

def mark_retained_config_on_pareto(n_retained, info_retained, out_dir):
    """
    Ajoute une étoile  sur le graphe rl_pareto_front.png pour montrer
    la configuration effectivement retenue (depuis le best checkpoint).
    Produit rl_pareto_front_pipeline.png pour ne pas écraser l'original.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    out_dir = Path(out_dir)
    src = out_dir / "rl_pareto_front.png"
    if not src.exists():
        return

    img = mpimg.imread(str(src))
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.imshow(img)
    ax.axis("off")

    # Annoter en overlay  position textuelle en bas de l'image
    fig.text(0.5, 0.01,
             f" Config retenue (best checkpoint) : N={n_retained}  |  "
             f"info={info_retained:.3f}",
             ha="center", color="#ffd93d", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a1628",
                       edgecolor="#ffd93d", alpha=0.9))

    out = out_dir / "rl_pareto_front_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a1628")
    plt.close()
    print(f"  Pareto annoté  {out}")


def visualize_two_configs(env, pareto_points, n_star, policy, args,
                          best_mask=None):
    """
    Compare deux configurations réseau :

    Config Dense  : best_mask du checkpoint (si fourni) ou simulation depuis N
                     c'est la configuration RETENUE transmise à GNN et AE
    Config Légère : N  N // 2, simulée par la politique

    best_mask : np.ndarray (K,) float32  active_mask du meilleur épisode RL.
                Quand fourni (mode pipeline), le panneau Dense montre exactement
                la configuration qui sera évaluée par GNN et AE.
    """
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib.colors import LinearSegmentedColormap
    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    BG = "#0a1628"

    n_light = max(env.n_min, int(n_star) // 2)

    def _run_config_policy(n_target):
        """Simule une configuration à n_target bouées avec la politique."""
        env.active_mask[:] = 0.0
        idx = np.random.choice(env.K, min(n_target, env.K), replace=False)
        env.active_mask[idx] = 1.0
        obs = env._get_obs()
        policy.eval()
        with torch.no_grad():
            for _ in range(env.ep_len):
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=DEVICE).unsqueeze(0)
                action, _, _, _ = policy.get_action(obs_t, deterministic=True)
                obs, _, done, _ = env.step(action.item())
                if done:
                    break
        active_idx = np.where(env.active_mask > 0.5)[0]
        return active_idx, float(env._compute_info_reward())

    # Config dense : best_mask si fourni, sinon simulation depuis N
    if best_mask is not None:
        env.active_mask = best_mask.copy()
        dense_idx  = np.where(best_mask > 0.5)[0]
        dense_info = float(env._compute_info_reward())
        dense_label = "Dense  (config retenue)"
        dense_note  = " configuration transmise au GNN & AE"
    else:
        dense_idx, dense_info = _run_config_policy(int(n_star))
        dense_label = "Dense  (N simulée)"
        dense_note  = f"N={n_star} (coude Pareto)"

    light_idx, light_info = _run_config_policy(n_light)
    light_label = f"Légère  (N  2  {n_light})"

    # La politique peut librement activer/désactiver des positions pendant la
    # simulation : rien ne garantit que la config partie de N finisse avec
    # plus de bouées que celle partie de N/2. Si l'ordre s'inverse, on
    # échange pour que les étiquettes restent vraies.
    if len(light_idx) > len(dense_idx):
        print(f"   la politique a inversé les tailles "
              f"({len(dense_idx)} vs {len(light_idx)})  étiquettes échangées")
        dense_idx, light_idx = light_idx, dense_idx
        dense_info, light_info = light_info, dense_info
        dense_label, light_label = "Dense  (simulée)", "Légère  (simulée)"
        dense_note = f"N={n_star} (coude Pareto)"

    T_bg    = env.T[0]
    vTmin, vTmax = float(env.T.min()), float(env.T.max())
    all_pos = np.array(env.candidate_positions)

    fig = plt.figure(figsize=(18, 8), facecolor=BG)
    title = ("Brique 3 RL  Config retenue (best checkpoint) vs Légère"
             if best_mask is not None
             else "Brique 3 RL  Dense (N) vs Légère (N2)")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=0.99)

    for col, (active_idx, info_score, label, note, col_c) in enumerate([
        (dense_idx,  dense_info,  dense_label, dense_note,  "#6bcb77"),
        (light_idx,  light_info,  light_label, f"N={len(light_idx)} capteurs", "#ffd93d"),
    ]):
        env.active_mask[:] = 0.0
        env.active_mask[active_idx] = 1.0
        inactive_idx = np.where(env.active_mask <= 0.5)[0]

        ax = fig.add_axes([0.05 + col*0.47, 0.10, 0.40, 0.80])
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")

        ax.imshow(T_bg.T, cmap=ocean_cmap, origin="lower", aspect="auto",
                  vmin=vTmin, vmax=vTmax, alpha=0.5, extent=[0, NX, 0, NY])
        ax.set_xlim(0, NX); ax.set_ylim(0, NY)
        ax.scatter(all_pos[inactive_idx, 0], all_pos[inactive_idx, 1],
                   c="#1a3a5c", s=14, alpha=0.35, zorder=2)
        sc = ax.scatter(all_pos[active_idx, 0], all_pos[active_idx, 1],
                        c=env.field_stats[active_idx], cmap="plasma",
                        s=90, vmin=0, vmax=1,
                        edgecolors="white", linewidths=0.8, zorder=6)
        cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label("Variance locale", color="white", fontsize=7)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
        ax.set_title(f"{label}\nN={len(active_idx)} bouées  |  Info={info_score:.3f}",
                     color=col_c, fontsize=11, fontweight="bold", pad=6)
        ax.text(0.02, 0.03, note, transform=ax.transAxes,
                color=col_c, fontsize=8, alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])

    loss_info = (dense_info - light_info) / (dense_info + 1e-9) * 100
    fig.text(0.5, 0.02,
             f"{dense_label}: N={len(dense_idx)} | info={dense_info:.3f}     "
             f"{light_label}: N={len(light_idx)} | info={light_info:.3f}     "
             f"Perte info: {loss_info:.1f}%  pour {len(dense_idx)-len(light_idx)} capteurs en moins",
             ha="center", color="#8ab4d4", fontsize=9)

    out = out_dir / "rl_two_configs.png"
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"\n   Deux configurations ")
    print(f"  Dense  : N={len(dense_idx)} bouées  info={dense_info:.3f}  [{dense_note}]")
    print(f"  Légère : N={len(light_idx)} bouées  info={light_info:.3f}")
    print(f"  Figure  {out}")




# 
#  VISUALISATION CONFIGURATION FINALE
# 

def visualize_final_config(env, active_mask, args, title="Configuration optimale RL"):
    """Visualise la configuration de réseau trouvée par l'agent RL."""
    out_dir = Path(args.output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Brique 3  {title}", fontsize=13, fontweight="bold")

    active_idx = np.where(active_mask > 0.5)[0]
    inactive_idx = np.where(active_mask <= 0.5)[0]
    all_pos = np.array(env.candidate_positions)

    ax = axes[0]
    ax.scatter(all_pos[inactive_idx, 0], all_pos[inactive_idx, 1],
               c="lightgray", s=30, alpha=0.4, label="Positions inactives")
    sc = ax.scatter(all_pos[active_idx, 0], all_pos[active_idx, 1],
                    c=env.field_stats[active_idx], cmap="YlOrRd",
                    s=120, edgecolors="black", linewidths=0.8, zorder=5,
                    label=f"Bouées actives ({len(active_idx)})")
    plt.colorbar(sc, ax=ax, label="Variance locale (importance OED)")
    ax.set_xlim(0, NX); ax.set_ylim(0, NY)
    ax.set_title(f"Réseau optimal  {len(active_idx)}/{env.K} positions actives")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")

    # Heatmap de la variance du champ
    ax2 = axes[1]
    variance_grid = env.field_stats.reshape(env.grid_x, env.grid_y)
    im = ax2.imshow(variance_grid.T, cmap="YlOrRd", origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax2, label="Variance locale normalisée")
    # Overlay des bouées
    for ai in active_idx:
        gx_idx = ai // env.grid_y
        gy_idx = ai % env.grid_y
        ax2.scatter(gx_idx, gy_idx, c="blue", s=60, marker="*", zorder=5)
    ax2.set_title("Grille de variance + positionnement RL\n(étoile bleue = bouée active)")
    ax2.set_xlabel(f"x (cellule grille {env.grid_x})")
    ax2.set_ylabel(f"y (cellule grille {env.grid_y})")

    fig.tight_layout()
    fig.savefig(out_dir / "rl_optimal_network.png", dpi=150)
    plt.close()
    print(f"   Configuration finale  {out_dir}/rl_optimal_network.png")


# 
#  POINT D'ENTRÉE
# 

# =============================================================================
#  GIF  Progression de l'agent RL
# =============================================================================

def save_rl_gif(env, policy, args, n_frames=80):
    """
    Genere un GIF animant la progression de l agent RL.

    Chaque frame = 1 step de l agent dans l environnement.
    On rejoue depuis un etat initial aleatoire et on laisse tourner
    l agent avec la politique entrainees (mode deterministe).

    Panneau gauche  : champ de variance locale (OED target) + bouees actives
    Panneau central : graphe du reseau (positions + aretes kNN)
    Panneau droit   : courbe de recompense cumulee en temps reel
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap
    from collections import deque
    from scipy.spatial import KDTree as _KDTree

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ocean_cmap = LinearSegmentedColormap.from_list("oc",
        ["#08306b","#2171b5","#6baed6","#c6dbef","#fff5eb",
         "#fdd49e","#fc8d59","#d7301f","#7f0000"], N=256)
    var_cmap = LinearSegmentedColormap.from_list("vc",
        ["#050d1a","#1a3a5c","#2e75b6","#ffd93d","#ff6b6b"], N=256)

    BG = "#0a1628"
    cands = np.array(env.candidate_positions)

    # Rejouer avec la politique
    obs = env.reset()
    all_states   = [env.active_mask.copy()]
    all_rewards  = [0.0]
    cum_rewards  = [0.0]
    all_infos    = [{"n_active": int(env.active_mask.sum()),
                     "total_info": env._compute_info_reward()}]

    policy.eval()
    with torch.no_grad():
        for _ in range(n_frames - 1):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action, _, _, _ = policy.get_action(obs_t, deterministic=False)
            obs, reward, done, info = env.step(action.item())
            all_states.append(env.active_mask.copy())
            all_rewards.append(reward)
            cum_rewards.append(cum_rewards[-1] + reward)
            all_infos.append(info)
            if done:
                obs = env.reset()

    # Variance de fond sur la grille candidate
    var_grid = env.field_stats.reshape(env.grid_x, env.grid_y)

    # Coordonnées pixel des positions candidates  doivent correspondre
    # à l'extent de l'imshow SST (0NX, 0NY) pour que les points
    # soient bien placés sur la carte.
    cands_px_x = cands[:, 0].astype(float)   #  [0, NX]
    cands_px_y = cands[:, 1].astype(float)   #  [0, NY]

    # Construction du GIF
    fig = plt.figure(figsize=(18, 7), facecolor=BG)
    ax1 = fig.add_axes([0.03, 0.10, 0.28, 0.78])    # carte SST + variance + bouees
    ax2 = fig.add_axes([0.36, 0.10, 0.28, 0.78])    # graphe reseau
    ax3 = fig.add_axes([0.70, 0.10, 0.27, 0.78])    # courbe recompense

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#050d1a")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        ax.tick_params(colors="#8ab4d4", labelsize=7)

    # Fond fixe ax1 : champ SST du nature run (coordonnées pixel NXNY)
    T_bg = env.T[0]
    vTmin, vTmax = float(env.T.min()), float(env.T.max())
    ax1.imshow(T_bg.T, cmap=ocean_cmap, origin="lower", aspect="auto",
               vmin=vTmin, vmax=vTmax, extent=[0, NX, 0, NY])
    ax1.set_xlim(0, NX); ax1.set_ylim(0, NY)
    ax1.set_title("Variance locale (OED target)", color="white",
                  fontsize=9, fontweight="bold", pad=5)
    ax1.set_xticks([]); ax1.set_yticks([])

    # Fond fixe ax2 : même étendue pixel pour cohérence
    ax2.set_xlim(0, NX); ax2.set_ylim(0, NY)
    ax2.imshow(T_bg.T, cmap=ocean_cmap, origin="lower", aspect="auto",
               vmin=vTmin, vmax=vTmax, alpha=0.3, extent=[0, NX, 0, NY])
    ax2.set_title("Graphe du reseau (kNN)", color="white",
                  fontsize=9, fontweight="bold", pad=5)
    ax2.set_xticks([]); ax2.set_yticks([])

    # Courbe recompense
    ax3.set_xlim(0, n_frames); ax3.set_ylim(min(cum_rewards)*1.1, max(cum_rewards)*1.1)
    ax3.set_title("Recompense cumulee", color="white", fontsize=9, fontweight="bold", pad=5)
    ax3.set_xlabel("Etape", color="#8ab4d4", fontsize=8)
    ax3.grid(True, alpha=0.15, color="white")
    reward_line, = ax3.plot([], [], color="#6bcb77", lw=2)
    step_vline   = ax3.axvline(0, color="#ffd93d", lw=1, alpha=0.7)

    # Elements dynamiques  les offsets utilisent les coordonnées pixel (0NX, 0NY)
    sc_inactive = ax1.scatter([], [], c="#1a3a5c", s=20, alpha=0.3, zorder=2)
    sc_active1  = ax1.scatter([], [], s=90, zorder=5, edgecolors="white",
                               linewidths=0.7)
    sc_inactive2 = ax2.scatter([], [], c="#1a3a5c", s=15, alpha=0.3, zorder=2)
    sc_active2   = ax2.scatter([], [], s=70, zorder=5, edgecolors="white",
                                linewidths=0.7)

    edge_lines = []   # lignes d aretes (recreees a chaque frame)

    txt_step = fig.text(0.5, 0.96, "", ha="center", color="white",
                         fontsize=11, fontweight="bold")
    txt_n    = ax1.text(0.02, 0.02, "", transform=ax1.transAxes,
                         color="#ffd93d", fontsize=9, va="bottom", fontweight="bold")
    txt_r    = ax3.text(0.98, 0.05, "", transform=ax3.transAxes,
                         color="#6bcb77", fontsize=9, ha="right", fontweight="bold")

    reward_x, reward_y = [], []

    def update(frame):
        nonlocal edge_lines
        mask    = all_states[frame]
        info    = all_infos[frame]
        cum_r   = cum_rewards[frame]
        n_active = int(mask.sum())

        active_idx   = np.where(mask > 0.5)[0]
        inactive_idx = np.where(mask <= 0.5)[0]

        #  Panneau 1 : carte SST + variance 
        if len(inactive_idx) > 0:
            sc_inactive.set_offsets(
                np.c_[cands_px_x[inactive_idx], cands_px_y[inactive_idx]])
        if len(active_idx) > 0:
            colors = plt.cm.plasma(env.field_stats[active_idx])
            sc_active1.set_offsets(
                np.c_[cands_px_x[active_idx], cands_px_y[active_idx]])
            sc_active1.set_color(colors)
        else:
            sc_active1.set_offsets(np.empty((0, 2)))

        #  Panneau 2 : graphe reseau 
        for ln in edge_lines: ln.remove()
        edge_lines = []

        if len(active_idx) > 1:
            pos_active = cands[active_idx].astype(float)
            tree_ = _KDTree(pos_active)
            for i in range(len(pos_active)):
                dists, idxs = tree_.query(pos_active[i], k=min(4, len(pos_active)))
                for j in idxs[1:]:
                    alpha_ = float(np.clip(1 - dists[list(idxs).index(j)] /
                                           (0.5*np.sqrt(NX**2+NY**2)), 0.05, 0.8))
                    ln_, = ax2.plot([pos_active[i,0], pos_active[j,0]],
                                    [pos_active[i,1], pos_active[j,1]],
                                    color="#2e75b6", alpha=alpha_, lw=1.2)
                    edge_lines.append(ln_)

        if len(inactive_idx) > 0:
            sc_inactive2.set_offsets(cands[inactive_idx])
        if len(active_idx) > 0:
            colors2 = plt.cm.YlOrRd(env.field_stats[active_idx])
            sc_active2.set_offsets(cands[active_idx])
            sc_active2.set_color(colors2)
        else:
            sc_active2.set_offsets(np.empty((0, 2)))

        #  Panneau 3 : courbe 
        reward_x.append(frame); reward_y.append(cum_r)
        reward_line.set_data(reward_x, reward_y)
        step_vline.set_xdata([frame, frame])

        #  Textes 
        eps = max(0.05, 1.0 - frame / n_frames)
        txt_step.set_text(
            f"Brique 3  RL  |  Etape {frame+1}/{n_frames}  "
            f"|  epsilon={eps:.2f}  |  N actives={n_active}")
        txt_step.set_color(plt.cm.cool(frame / n_frames))
        txt_n.set_text(f"Bouees: {n_active} [{env.n_min}-{env.n_max}]")
        txt_r.set_text(f"R_cum = {cum_r:+.3f}")

        return (sc_inactive, sc_active1, sc_inactive2, sc_active2,
                reward_line, step_vline, txt_step, txt_n, txt_r)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=True)
    gif_path = out_dir / "rl_progression.gif"
    writer = PillowWriter(fps=6, metadata={"title": "OED-IA RL SNO"})
    anim.save(str(gif_path), writer=writer, dpi=110,
              savefig_kwargs={"facecolor": BG})
    plt.close()
    print(f"  GIF RL -> {gif_path}")


# =============================================================================
#  ARGUMENTS + MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Brique 3  RL pour OED")
    p.add_argument("--train",        action="store_true", help="Lancer PPO")
    p.add_argument("--pareto",       action="store_true", help="Front de Pareto")
    p.add_argument("--gif",          action="store_true", help="Genere le GIF")
    p.add_argument("--report",       action="store_true",
                   help="Produit un rapport .txt avec les métriques clés")
    p.add_argument("--seed_ocean",   type=int, default=42)
    p.add_argument("--seed_buoys",   type=int, default=7)
    p.add_argument("--checkpoint",   type=str, default="outputs/rl_best.pt")
    p.add_argument("--output_dir",   type=str, default="outputs")
    p.add_argument("--rl_steps",     type=int, default=50000)
    p.add_argument("--buffer_size",  type=int, default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--grid_x",       type=int, default=16)
    p.add_argument("--grid_y",       type=int, default=24)
    p.add_argument("--n_min",        type=int, default=10)
    p.add_argument("--n_max",        type=int, default=40)
    p.add_argument("--episode_len",  type=int, default=20)
    p.add_argument("--w_info",       type=float, default=1.0)
    p.add_argument("--w_budget",     type=float, default=0.5)
    p.add_argument("--gif_frames",   type=int, default=80)
    p.add_argument("--ae_checkpoint", type=str, default=None,
                   help="Checkpoint AE (vae_best.pt) : active la récompense "
                        "fondée sur la reconstruction au lieu du proxy variance")
    p.add_argument("--ae_n_dates",   type=int, default=8,
                   help="Dates évaluées par calcul de récompense AE")
    add_data_args(p)
    return p.parse_args()


def _load_ae_for_reward(ckpt_path, channels):
    """Charge un autoencodeur entraîné pour servir de fonction de récompense."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "b1", Path(__file__).parent / "01_autoencoder.py")
    b1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b1)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = b1.ObservabilityVAE(
        in_ch=VAE_IN_CH, out_ch=VAE_OUT_CH,
        base_ch=ck["args"]["base_ch"], latent_ch=ck["args"]["latent_ch"],
        dropout_p=ck["args"].get("dropout_p", 0.1),
        cond_dim=ck["args"].get("cond_dim", 32)).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


if __name__ == "__main__":
    from datetime import datetime
    args = parse_args()

    if not args.train and not args.pareto and not args.gif:
        print("Usage: python 03_rl.py --train [--pareto] [--gif] [--report]")
        sys.exit(0)

    print("\n[1/2] Chargement du champ oceanique...")
    fields, channels, sea_mask, data_info = load_ocean(args)
    print(f"  {data_info['source']} | {fields.shape} | canaux={channels}")

    print("[2/2] Initialisation de l environnement MDP...")
    ae_model = None
    if args.ae_checkpoint:
        print(f"  Récompense AE : chargement de {args.ae_checkpoint}")
        ae_model = _load_ae_for_reward(args.ae_checkpoint, channels)
        print("   récompense = RMSE de reconstruction (alignée sur la brique 4)")
    else:
        print("  Récompense : proxy variance-pondérée ( anti-corrélé à la "
              "reconstruction ; passer --ae_checkpoint pour l'aligner)")

    env = OceanNetworkEnv(
        fields, channels=channels,
        grid_x=args.grid_x, grid_y=args.grid_y,
        n_min=args.n_min, n_max=args.n_max,
        episode_len=args.episode_len,
        w_info=args.w_info, w_budget=args.w_budget,
        sea_mask=sea_mask, dx_km=data_info.get("dx_km"),
        ae_model=ae_model, ae_n_dates=args.ae_n_dates)
    print(f"  K = {env.K} positions candidates en mer "
          f"({args.grid_x}x{args.grid_y} = {args.grid_x*args.grid_y} theoriques)")
    print(f"  Budget bouees : [{args.n_min}, {args.n_max}]")

    policy = None; pareto_data = {}
    if args.train:
        policy = train_ppo(args, env)
        ckpt = torch.load(Path(args.output_dir) / "rl_best.pt",
                          map_location=DEVICE, weights_only=False)
        visualize_final_config(env, ckpt["active_mask"], args)
        print("\n  Generation automatique du GIF post-entrainement...")
        save_rl_gif(env, policy, args, n_frames=args.gif_frames)

    if args.pareto:
        if policy is None:
            policy = ActorCritic(env.obs_dim, env.K).to(DEVICE)
            ckpt_path = Path(args.output_dir) / "rl_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                policy.load_state_dict(ckpt["policy_state"])
        pareto_points, pareto_mask, n_star = compute_pareto_front(env, policy, args)
        visualize_two_configs(env, pareto_points, n_star, policy, args)
        info_vals = np.array([p["info_mean"] for p in pareto_points])
        n_vals    = np.array([p["n_buoys"]   for p in pareto_points])
        n_light   = max(env.n_min, n_star // 2)
        pareto_data = {
            "n_star": int(n_star),
            "info_star": float(info_vals[np.argmin(np.abs(n_vals - n_star))]),
            "info_max":  float(info_vals.max()),
            "n_light":   int(n_light),
            "info_light": float(info_vals[np.argmin(np.abs(n_vals - n_light))]),
            "n_pareto_opt": int(pareto_mask.sum()),
        }

    if args.gif:
        print("\n  Generation du GIF de progression RL...")
        if policy is None:
            policy = ActorCritic(env.obs_dim, env.K).to(DEVICE)
            ckpt_path = Path(args.output_dir) / "rl_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                policy.load_state_dict(ckpt["policy_state"])
                print(f"  Politique chargee depuis {ckpt_path}")
        save_rl_gif(env, policy, args, n_frames=args.gif_frames)

    if args.report:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output_dir)
        perte_pct = 0.0
        if pareto_data:
            i_s = pareto_data.get("info_star", 0)
            i_l = pareto_data.get("info_light", 0)
            perte_pct = (i_s - i_l) / (i_s + 1e-9) * 100
        lines = [
            "=" * 68,
            "  Brique 3  RL  Rapport",
            f"  Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 68, "",
            " REPRODUCTIBILITÉ ",
            f"  seed_ocean    : {args.seed_ocean}",
            f"  seed_buoys    : {args.seed_buoys}",
            "",
            " PARAMÈTRES RL ",
            f"  rl_steps      : {args.rl_steps}",
            f"  grid          : {args.grid_x}{args.grid_y}  ({env.K} candidats)",
            f"  n_min / n_max : {args.n_min} / {args.n_max}",
            f"  w_info        : {args.w_info}",
        ]
        if pareto_data:
            lines += [
                "",
                " RÉSULTATS PARETO ",
                f"  N (coude)              : {pareto_data['n_star']} capteurs",
                f"  Score info N           : {pareto_data['info_star']:.3f}",
                f"  Score info maximum      : {pareto_data['info_max']:.3f}",
                f"  Config légère N         : {pareto_data['n_light']} capteurs",
                f"  Score info légère       : {pareto_data['info_light']:.3f}",
                f"  Perte info denselégère : {perte_pct:.1f} %",
                f"  Configs Pareto-optimales: {pareto_data['n_pareto_opt']}",
            ]
        lines += ["", " FICHIERS PRODUITS "]
        for f in sorted(out.iterdir()):
            if f.suffix in {".pt", ".png", ".gif"}:
                lines.append(f"  {f.name:<44} {f.stat().st_size//1024:>5} KB")
        lines += ["", "=" * 68]
        rpt = out / f"rapport_rl_{ts}.txt"
        rpt.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  Rapport RL  {rpt}")

    print("\n  Brique 3 terminee.")

