"""
04_maintenance_rl.py — Module 4 : MAINTENANCE OPTIMALE d'un reseau sur 13 ans.

Les modules 1-3 repondent a "ou placer N bouees ?" sur un an. Avec un nature
run pluriannuel la question devient sequentielle et bien plus proche du reel :

    QUI entretenir, QUAND, DANS QUEL ORDRE, sous budget pluriannuel ?

MDP
---
  epoque       : 1 campagne = 182 jours (N_CAMPAIGNS_YEAR = 2)
  etat         : positions actives, age de chaque bouee, saison, budget
  action       : intervenir sur une cellule candidate
                   - occupee   -> entretien (age remis a 0)
                   - libre     -> deploiement (si separation mini respectee)
                 ou "fin de campagne"
  transition   : vieillissement 182 j + tirage de perte (Weibull) + saut dans
                 le nature run
  recompense   : + variance resolue HORS ECHANTILLON moyennee sur le semestre
                 - cout de la campagne (amortissement + km navire, tournee 2-opt)

L'abandon d'une position n'a pas besoin d'action dediee : ne pas entretenir
suffit, la bouee vieillit puis meurt et libere la place. La politique doit donc
apprendre a differer l'entretien des bouees redondantes, a grouper les visites
sur une meme tournee, et a anticiper la saison ou une position devient critique.

Politique
---------
Scoreur par candidat (features locales + globales diffusees) -> logit, softmax
sur K+1 actions. Permutation-equivariant, donc transferable a un reseau de
taille differente. REINFORCE + baseline apprise, numpy pur.

Protocole
---------
Apprentissage sur les 9 premieres annees, EVALUATION SUR LES 4 DERNIERES,
jamais vues. C'est le seul moyen de distinguer une politique qui a compris la
structure spatio-temporelle d'une politique qui a memorise la sequence
d'eddies.

Usage
-----
    python 04_maintenance_rl.py --selftest              # ~1 min, valide le noyau
    python 04_maintenance_rl.py --demo                  # ~4 min, 2 ans, petit domaine
    python 04_maintenance_rl.py --train --evaluate      # run complet 13 ans
"""

import sys, json, argparse, time
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import N_CAMPAIGNS_YEAR
from oed_core import (build_nature_run, EOFBasis, SensorAgeing, AnalysisError,
                      Domain, DAYS_PER_YEAR)

EPOCH_DAYS = DAYS_PER_YEAR / N_CAMPAIGNS_YEAR      # 182.5 j


# =============================================================================
#  Environnement
# =============================================================================
class MaintenanceEnv:
    def __init__(self, basis, ageing, domain, cand_cells,
                 n_max=30, max_ops=6, w_cost=0.35, cost_ref=None,
                 n_probe=3, shortlist=40, period=None, rng=None):
        self.b, self.ag, self.dom = basis, ageing, domain
        self.ae = AnalysisError(basis, ageing)
        self.cand = np.asarray(cand_cells, int)
        self.K = len(self.cand)
        self.n_max, self.max_ops = n_max, max_ops
        self.w_cost = w_cost
        self.n_probe = n_probe
        self.shortlist = shortlist
        self.period = period or (0, len(basis.A))
        self.rng = rng or np.random.default_rng(0)

        # features statiques par candidat
        rows = basis.obs_rows(self.cand)
        lev = (basis.U[rows] ** 2) @ basis.lam                # levier EOF
        lev = lev[:self.K] + lev[self.K:]
        self.lev = (lev - lev.mean()) / (lev.std() + 1e-9)
        self.dport = domain.d_port[self.cand]
        self.dport_n = (self.dport - self.dport.mean()) / (self.dport.std() + 1e-9)
        self.n_epochs = int((self.period[1] - self.period[0]) // EPOCH_DAYS)

        # Unite de compte du cout : une campagne NOMINALE = entretien de
        # max_ops bouees d'un reseau plein. w_cost est alors lisible
        # directement en "fraction de variance resolue qu'une campagne doit
        # rapporter pour etre rentable".
        if cost_ref is None:
            probe = self.cand[np.linspace(0, self.K - 1, max_ops).astype(int)]
            cost_ref = domain.campaign_cost(probe, n_max)[0]
        self.cost_ref = float(max(cost_ref, 1e-6))

    # ---------------------------------------------------------------- reset
    def reset(self, n_init=None, epoch0=None):
        n_init = n_init or int(self.rng.integers(8, min(20, self.n_max)))
        pos, ages = [], []
        order = self.rng.permutation(self.K)
        for j in order:
            if len(pos) >= n_init:
                break
            if self.dom.feasible([self.cand[j]], [self.cand[p] for p in pos])[0]:
                pos.append(j); ages.append(float(self.rng.uniform(0, 400)))
        self.pos = list(pos)                                  # indices dans self.cand
        self.age = list(ages)
        self.epoch = 0
        self.epoch0 = epoch0 if epoch0 is not None else 0
        self.ops = 0
        self.serviced = []
        self.log = []
        return self._obs()

    # ------------------------------------------------------------- features
    def _obs(self):
        occ = np.zeros(self.K); age = np.zeros(self.K)
        for j, a in zip(self.pos, self.age):
            occ[j] = 1.0; age[j] = a / self.ag.tau
        act_cells = [self.cand[j] for j in self.pos]
        if act_cells:
            d = self.dom.dist(self.cand, act_cells)
            dmin = d.min(1)
            free_ok = dmin >= self.dom.min_sep_km
        else:
            dmin = np.full(self.K, 5 * self.dom.min_sep_km); free_ok = np.ones(self.K, bool)
        avail = np.where(occ > 0, 1.0,
                         (free_ok & (len(self.pos) < self.n_max)).astype(float))
        cand_age = np.full(self.K, np.nan)
        for j, a in zip(self.pos, self.age):
            cand_age[j] = a
        dg = self.ae.marginal_gain_fast([self.cand[p] for p in self.pos], self.age,
                                        self.cand, cand_age=cand_age)
        dg = dg / (dg.std() + 1e-12)
        if self.serviced:
            dtour = self.dom.dist(self.cand, self.serviced).min(1)
        else:
            dtour = np.full(self.K, self.dom.d_port.max())
        phase = 2 * np.pi * ((self.epoch0 + self.epoch * EPOCH_DAYS) % DAYS_PER_YEAR) / DAYS_PER_YEAR
        glob = np.array([len(self.pos) / self.n_max, self.ops / self.max_ops,
                         np.sin(phase), np.cos(phase),
                         self.epoch / max(self.n_epochs, 1)])
        F = np.stack([occ, age, self.lev, self.dport_n,
                      np.tanh(dmin / (3 * self.dom.min_sep_km)),
                      np.tanh(dtour / 500.0), avail, np.tanh(dg)], 1)  # (K, 8)
        F = np.concatenate([F, np.repeat(glob[None], self.K, 0)], 1)   # (K, 13)

        # PRE-SELECTION PAR LE CRITERE OED : la politique n'arbitre que sur les
        # `shortlist` meilleures options du moment (gain marginal de Kalman),
        # pas sur les ~200 cellules de la grille. C'est le couplage
        # criticite -> RL : l'agent ne cherche plus a l'aveugle, il tranche
        # entre des candidats deja qualifies, sous contrainte de cout et de
        # calendrier -- ce que le critere myope, lui, ne sait pas faire.
        ok = np.where(avail > 0)[0]
        if len(ok) > self.shortlist:
            ok = ok[np.argsort(-dg[ok])[:self.shortlist]]
        return dict(F=F[ok], avail=avail[ok], glob=glob, idx=ok)

    # ------------------------------------------------------------ dynamique
    def apply_op(self, j):
        """Entretien si occupee, deploiement sinon. Retourne True si l'op a eu lieu."""
        if self.ops >= self.max_ops:
            return False
        cell = self.cand[j]
        if j in self.pos:
            self.age[self.pos.index(j)] = 0.0
        else:
            act = [self.cand[p] for p in self.pos]
            if len(self.pos) >= self.n_max or not self.dom.feasible([cell], act)[0]:
                return False
            self.pos.append(j); self.age.append(0.0)
        self.serviced.append(cell)
        self.ops += 1
        return True

    def _semester_score(self):
        """Variance resolue OOS moyennee sur le semestre, ages croissants."""
        t0 = self.epoch0 + int(self.epoch * EPOCH_DAYS)
        cells = [self.cand[p] for p in self.pos]
        if not cells:
            return 0.0
        seg = EPOCH_DAYS / self.n_probe
        sc = []
        for q in range(self.n_probe):
            a = np.array(self.age) + (q + 0.5) * seg
            s = slice(min(t0 + int(q * seg), len(self.b.A) - 2),
                      min(t0 + int((q + 1) * seg), len(self.b.A) - 1))
            if s.stop <= s.start:
                continue
            sc.append(self.ae.resolved_oos(cells, a, s, rng=self.rng))
        return float(np.mean(sc)) if sc else 0.0

    def close_campaign(self):
        """Fin de campagne : score, cout, vieillissement, pertes."""
        info = self._semester_score()
        cost, co2, km = self.dom.campaign_cost(self.serviced, len(self.pos),
                                               n_new=max(0, self.ops - len(self.serviced) + 0))
        reward = info - self.w_cost * cost / self.cost_ref

        surv = self.ag.survival(np.array(self.age), EPOCH_DAYS) if self.pos else np.array([])
        keep = self.rng.random(len(surv)) < surv
        n_lost = int((~keep).sum())
        self.pos = [p for p, k in zip(self.pos, keep) if k]
        self.age = [a + EPOCH_DAYS for a, k in zip(self.age, keep) if k]

        self.log.append(dict(epoch=self.epoch, info=info, cost=cost, km=km, co2=co2,
                             n=len(self.pos) + n_lost, ops=self.ops, lost=n_lost))
        self.epoch += 1; self.ops = 0; self.serviced = []
        done = self.epoch >= self.n_epochs
        return reward, done, self._obs()


# =============================================================================
#  Politique : scoreur par candidat + baseline (numpy, backprop manuel)
# =============================================================================
class MLP:
    def __init__(self, sizes, rng, out_scale=0.1):
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.standard_normal((a, b)) * np.sqrt(2.0 / a))
            self.b.append(np.zeros(b))
        self.W[-1] *= out_scale
        self.m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(v) for v in self.b]
        self.v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(v) for v in self.b]
        self.t = 0

    def forward(self, X):
        self.h = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.h[-1] @ W + b
            self.h.append(np.tanh(z) if i < len(self.W) - 1 else z)
        return self.h[-1]

    def backward(self, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = self.h[i].T @ d; gb[i] = d.sum(0)
            if i:
                d = (d @ self.W[i].T) * (1 - self.h[i] ** 2)
        return gW + gb

    def step(self, grads, lr=3e-3, b1=0.9, b2=0.999, clip=5.0):
        self.t += 1
        params = self.W + self.b
        gn = np.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12
        sc = min(1.0, clip / gn)
        for i, (p, g) in enumerate(zip(params, grads)):
            g = g * sc
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t); vh = self.v[i] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)


class Policy:
    def __init__(self, n_feat, hidden=48, seed=0, stop_bias=-1.5):
        rng = np.random.default_rng(seed)
        self.actor = MLP([n_feat, hidden, hidden, 1], rng)
        self.critic = MLP([2 * n_feat + 1, hidden, 1], rng)
        self.stop_bias = stop_bias      # a priori : agir plutot que ne rien faire
        self.rng = rng

    def logits(self, obs):
        z = self.actor.forward(obs["F"]).ravel()
        z = np.where(obs["avail"] > 0, z, -1e9)
        return np.concatenate([z, [self.stop_bias]])   # derniere action = STOP

    def act(self, obs, greedy=False):
        z = self.logits(obs)
        z = z - z.max()
        p = np.exp(z); p /= p.sum()
        a = int(np.argmax(p)) if greedy else int(self.rng.choice(len(p), p=p))
        return a, p

    def value(self, obs):
        F = obs["F"]
        x = np.concatenate([F.mean(0), F.max(0), [F.shape[0] / 100.0]])[None]
        return float(self.critic.forward(x)[0, 0]), x


# =============================================================================
#  Entrainement REINFORCE avec baseline
# =============================================================================
def run_episode(env, pol, greedy=False, collect=True, epoch0=None, n_init=None):
    obs = env.reset(n_init=n_init, epoch0=epoch0)
    traj, rewards, done = [], [], False
    while not done:
        for _ in range(env.max_ops + 1):
            a, p = pol.act(obs, greedy)
            v, xv = pol.value(obs)
            if collect:
                traj.append(dict(obs=obs, a=a, p=p, v=v, xv=xv))
            if a == len(obs["idx"]):                        # fin de campagne
                break
            env.apply_op(int(obs["idx"][a]))
            obs = env._obs()
            if env.ops >= env.max_ops:
                break
        r, done, obs = env.close_campaign()
        rewards.append(r)
        if collect:
            traj[-1]["r"] = r
    return traj, rewards


def _snapshot(pol):
    return ([w.copy() for w in pol.actor.W], [b.copy() for b in pol.actor.b],
            pol.stop_bias)


def _restore(pol, snap):
    pol.actor.W = [w.copy() for w in snap[0]]
    pol.actor.b = [b.copy() for b in snap[1]]
    pol.stop_bias = snap[2]


def train(env, pol, n_episodes=600, gamma=0.98, lr=3e-3, ent=0.01,
          val_env=None, val_epoch0=None, eval_every=50, n_val=3, verbose=True):
    """
    REINFORCE + baseline, AVEC SELECTION DE MODELE SUR VALIDATION.

    Sans elle, la politique se degrade en fin d'entrainement (effondrement de
    l'entropie : elle apprend a ne plus intervenir du tout, ce qui est un
    optimum local peu couteux). On garde donc le meilleur jeu de poids mesure
    sur une fenetre de validation disjointe, jamais utilisee pour le gradient.
    """
    hist, best, best_snap = [], -np.inf, _snapshot(pol)
    for ep in range(n_episodes):
        e0 = int(env.rng.integers(0, max(1, env.period[1] - env.period[0]
                                         - int(env.n_epochs * EPOCH_DAYS)))) + env.period[0]
        traj, rewards = run_episode(env, pol, epoch0=e0)

        # retours a rebours, recompense placee sur l'action terminale de campagne
        R, returns = 0.0, []
        for st in reversed(traj):
            R = st.get("r", 0.0) + gamma * R
            returns.append(R)
        returns = np.array(returns[::-1])
        adv = returns - np.array([st["v"] for st in traj])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        gA = [np.zeros_like(w) for w in pol.actor.W + pol.actor.b]
        gC = [np.zeros_like(w) for w in pol.critic.W + pol.critic.b]
        gstop = 0.0
        for st, A, Rt in zip(traj, adv, returns):
            p, a = st["p"], st["a"]
            dz = -A * (np.eye(len(p))[a] - p)                 # -grad log pi * A
            dz += ent * p * (np.log(p + 1e-12) + 1.0)         # bonus d'entropie
            gstop += dz[-1]
            pol.actor.forward(st["obs"]["F"])
            for g, gi in zip(gA, pol.actor.backward(dz[:-1, None])):
                g += gi
            pol.critic.forward(st["xv"])
            for g, gi in zip(gC, pol.critic.backward(np.array([[st["v"] - Rt]]))):
                g += gi
        n = max(len(traj), 1)
        pol.actor.step([g / n for g in gA], lr=lr)
        pol.critic.step([g / n for g in gC], lr=lr)
        pol.stop_bias -= lr * gstop / n

        hist.append(dict(ep=ep, ret=float(np.sum(rewards)),
                         info=float(np.mean([l["info"] for l in env.log])),
                         cost=float(np.sum([l["cost"] for l in env.log]))))
        if val_env is not None and (ep + 1) % eval_every == 0:
            v = np.mean([np.sum(run_episode(val_env, pol, greedy=(i == 0),
                                            collect=False, epoch0=val_epoch0)[1])
                         for i in range(n_val)])
            hist[-1]["val"] = float(v)
            if v > best:
                best, best_snap = v, _snapshot(pol)

        if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
            w = hist[-max(1, n_episodes // 10):]
            print(f"  ep {ep+1:4d}  return {np.mean([h['ret'] for h in w]):8.2f}"
                  f"  info {np.mean([h['info'] for h in w]):.3f}"
                  f"  cout {np.mean([h['cost'] for h in w]):7.1f} kEUR"
                  + (f"  | val {hist[-1]['val']:.2f}" if "val" in hist[-1] else ""))
    if val_env is not None:
        _restore(pol, best_snap)
        if verbose:
            print(f"      meilleur modele retenu (validation = {best:.2f})")
    return hist


# =============================================================================
#  Baselines
# =============================================================================
def policy_baseline(env, kind, epoch0=None, every=2, seed=0):
    """none | periodic | reactive | greedy"""
    rng = np.random.default_rng(seed)
    obs = env.reset(epoch0=epoch0)
    rewards, done = [], False
    while not done:
        if kind == "periodic" and env.epoch % every == 0:
            order = np.argsort([-a for a in env.age])
            for j in [env.pos[i] for i in order[:env.max_ops]]:
                env.apply_op(j)
        elif kind == "reactive":
            # redeployer pour revenir a l'effectif nominal
            free = [j for j in range(env.K) if j not in env.pos]
            rng.shuffle(free)
            for j in free:
                if env.ops >= env.max_ops or len(env.pos) >= env.n_max * 0.7:
                    break
                env.apply_op(j)
        elif kind == "greedy":
            for _ in range(env.max_ops):
                cells = [env.cand[p] for p in env.pos]
                base = env.ae.resolved(cells, env.age)
                best, bj = 0.0, None
                for j in range(env.K):
                    if j in env.pos:
                        k = env.pos.index(j)
                        a2 = list(env.age); a2[k] = 0.0
                        g = env.ae.resolved(cells, a2) - base
                    else:
                        if len(env.pos) >= env.n_max or not env.dom.feasible(
                                [env.cand[j]], cells)[0]:
                            continue
                        g = env.ae.resolved(cells + [env.cand[j]],
                                            list(env.age) + [0.0]) - base
                    if g > best:
                        best, bj = g, j
                if bj is None:
                    break
                env.apply_op(bj)
        r, done, obs = env.close_campaign()
        rewards.append(r)
    return rewards, list(env.log)


# =============================================================================
#  Figure
# =============================================================================
def make_figure(hist, logs, res, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0a1628")
    for a in ax.ravel():
        a.set_facecolor("#050d1a"); a.tick_params(colors="white", labelsize=8)
        for sp in a.spines.values(): sp.set_edgecolor("#2a4a7a")
        a.grid(alpha=0.2, color="white")

    def ttl(a, t, xl, yl):
        a.set_title(t, color="white", fontsize=10, fontweight="bold")
        a.set_xlabel(xl, color="white", fontsize=8); a.set_ylabel(yl, color="white", fontsize=8)

    if hist:
        r = np.array([h["ret"] for h in hist]); w = max(1, len(r) // 25)
        ax[0, 0].plot(r, color="#2a4a7a", lw=0.6)
        ax[0, 0].plot(np.convolve(r, np.ones(w) / w, "valid"), color="#ffd93d", lw=2)
    ttl(ax[0, 0], "Apprentissage (retour par episode)", "episode", "retour")

    cols = {"none": "#888", "reactive": "#6baed6", "periodic": "#6bcb77",
            "greedy": "#fc8d59", "rl": "#ffd93d"}
    for k, lg in logs.items():
        ax[0, 1].plot([l["info"] for l in lg], "-o", ms=3, color=cols.get(k), label=k)
        ax[1, 0].plot(np.cumsum([l["cost"] for l in lg]), "-o", ms=3, color=cols.get(k), label=k)
    ttl(ax[0, 1], "Variance resolue OOS par campagne (periode de test)", "campagne", "resolue")
    ttl(ax[1, 0], "Cout cumule", "campagne", "k EUR")
    for a in (ax[0, 1], ax[1, 0]):
        a.legend(fontsize=7, labelcolor="white", facecolor="#0a1628")

    for k, v in res.items():
        c, i = v["cost"], v["info"]
        ax[1, 1].scatter(c, i, s=140, color=cols.get(k), edgecolors="white", zorder=5)
        ax[1, 1].annotate(k, (c, i), color="white", fontsize=9,
                          xytext=(6, 6), textcoords="offset points")
    ttl(ax[1, 1], "Front cout / information (test, moyenne sur les graines)",
        "cout total k EUR", "variance resolue moyenne")

    fig.suptitle("Module 4 — maintenance optimale sur horizon pluriannuel",
                 color="white", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, facecolor="#0a1628", bbox_inches="tight")
    plt.close()
    print(f"      figure -> {out_path}")


# =============================================================================
#  Selftest
# =============================================================================
def selftest():
    print("=" * 72); print("SELFTEST oed_core + env"); print("=" * 72)
    from tests.test_oed_maintenance import run_all
    return run_all()


# =============================================================================
#  Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true", help="petit run rapide")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--nx", type=int, default=192)
    ap.add_argument("--ny", type=int, default=288)
    ap.add_argument("--years", type=float, default=13.0)
    ap.add_argument("--train_years", type=float, default=9.0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--n_eddies", type=int, default=45)
    ap.add_argument("--pert_amp", type=float, default=6.0e3,
                    help="amplitude de la perturbation non resolue (chaos)")
    ap.add_argument("--cand_stride", type=int, default=3)
    ap.add_argument("--k_eof", type=int, default=80)
    ap.add_argument("--n_max", type=int, default=30)
    ap.add_argument("--max_ops", type=int, default=6)
    ap.add_argument("--shortlist", type=int, default=40,
                    help="candidats presentes a la politique (pre-selection OED)")
    ap.add_argument("--w_cost", type=float, default=0.35,
                    help="cout d'une campagne nominale, en unites de variance resolue")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=8, help="campagnes par episode")
    ap.add_argument("--n_eval", type=int, default=5,
                    help="reseaux initiaux d'evaluation (memes graines pour toutes les politiques)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="outputs")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.demo:
        args.nx, args.ny, args.years, args.train_years = 128, 192, 5.0, 3.0
        args.n_eddies, args.k_eof = 30, 50
        args.episodes = args.episodes or 400

    args.episodes = args.episodes or 600

    t0 = time.time()
    nt = int(args.years * DAYS_PER_YEAR)
    n_tr = int(args.train_years * DAYS_PER_YEAR)
    print(f"[1/5] nature run {args.nx}x{args.ny}, {args.years} ans, "
          f"stride {args.stride}, {args.n_eddies} eddies, pert {args.pert_amp:g}")
    nr = build_nature_run(nx=args.nx, ny=args.ny, nt=nt, stride=args.stride,
                          n_eddies=args.n_eddies, seed=args.seed,
                          pert_amp=args.pert_amp)
    T, S = nr["T"], nr["S"]
    print(f"      {T.shape}  ({'cache' if nr['cached'] else 'calcule'}) "
          f"{time.time()-t0:.0f}s")

    print(f"[2/5] base EOF sur les {args.train_years} premieres annees")
    basis = EOFBasis(T, S, slice(0, n_tr), k=args.k_eof, seed=args.seed)
    print(f"      k={basis.k}, variance expliquee {basis.var_explained:.3f}")

    ageing = SensorAgeing(basis.sig_T, basis.sig_S)
    dom = Domain(T.shape[1:], nr["dx_km"])
    nxc, nyc = T.shape[1:]
    gi, gj = np.meshgrid(np.arange(1, nxc, args.cand_stride),
                         np.arange(1, nyc, args.cand_stride), indexing="ij")
    cand = (gi * nyc + gj).ravel()
    print(f"[3/5] {len(cand)} candidats, separation mini {dom.min_sep_km:.0f} km")

    rng = np.random.default_rng(args.seed)
    mk = lambda per: MaintenanceEnv(basis, ageing, dom, cand, n_max=args.n_max,
                                    max_ops=args.max_ops, w_cost=args.w_cost,
                                    shortlist=args.shortlist, period=per,
                                    rng=np.random.default_rng(args.seed))
    env_tr = mk((0, n_tr)); env_tr.n_epochs = args.horizon
    env_te = mk((n_tr, len(T)))
    print(f"      train {env_tr.n_epochs} campagnes/episode | "
          f"test {env_te.n_epochs} campagnes (annees {args.train_years:.0f}-{args.years:.0f})")

    pol = Policy(n_feat=13, seed=args.seed)
    hist = []
    if args.train or args.demo:
        print(f"[4/5] REINFORCE, {args.episodes} episodes")
        n_val_days = int(1.0 * DAYS_PER_YEAR)
        env_va = mk((max(0, n_tr - n_val_days), n_tr))
        env_va.n_epochs = max(2, int(n_val_days // EPOCH_DAYS))
        hist = train(env_tr, pol, n_episodes=args.episodes,
                     val_env=env_va, val_epoch0=max(0, n_tr - n_val_days))

    print(f"[5/5] evaluation sur la periode de TEST (jamais vue), "
          f"{args.n_eval} reseaux initiaux")

    def evaluate(runner):
        """
        Meme graine -> MEME reseau initial et MEMES tirages de panne pour
        toutes les politiques. Sans cela on compare des trajectoires
        stochastiques differentes et l'ecart mesure est du bruit.
        """
        acc = []
        for sd in range(args.n_eval):
            env_te.rng = np.random.default_rng(1000 + sd)
            rw, log = runner()
            acc.append(dict(ret=float(np.sum(rw)),
                            info=float(np.mean([l["info"] for l in log])),
                            cost=float(np.sum([l["cost"] for l in log])),
                            n=float(np.mean([l["n"] for l in log])),
                            km=float(np.sum([l["km"] for l in log]))))
        return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}, log

    res, logs = {}, {}
    for kind in ["none", "reactive", "periodic", "greedy"]:
        res[kind], logs[kind] = evaluate(
            lambda k=kind: policy_baseline(env_te, k, epoch0=n_tr))
    res["rl"], logs["rl"] = evaluate(
        lambda: (run_episode(env_te, pol, greedy=True, collect=False,
                             epoch0=n_tr)[1], env_te.log))

    for v in res.values():
        v["eff"] = 100.0 * v["info"] / max(v["cost"], 1e-9)     # resolu / 100 k EUR
    print(f"\n{'politique':<12}{'retour':>9}{'var. res.':>11}{'cout kEUR':>11}"
          f"{'res./100kE':>12}{'N moyen':>9}{'km':>9}")
    print("-" * 73)
    for k, v in res.items():
        print(f"{k:<12}{v['ret']:>9.2f}{v['info']:>11.3f}{v['cost']:>11.0f}"
              f"{v['eff']:>12.4f}{v['n']:>9.1f}{v['km']:>9.0f}")
    print("\nglouton = optimum myope, mais ~%d evaluations de Pa par campagne ;"
          % (len(cand) * args.max_ops))
    print("la politique RL decide en une passe avant, et anticipe les campagnes suivantes.")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    make_figure(hist, logs, res, out / f"maintenance_{stamp}.png")
    fp = out / f"maintenance_{stamp}.json"
    json.dump(dict(args=vars(args), results=res, history=hist[-50:],
                   eof=dict(k=basis.k, var_explained=basis.var_explained)),
              open(fp, "w"), indent=2)
    print(f"\n-> {fp}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
