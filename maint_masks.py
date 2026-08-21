"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ETAGE 3b  —  MASQUES D'OBSERVATION ISSUS DE LA MAINTENANCE                  ║
║                                                                              ║
║  La brique 1 s'entraine sur `_random_mask` : 5 a 60 pixels tires             ║
║  uniformement dans le domaine, INDEPENDAMMENT a chaque pas de temps. Ce      ║
║  n'est pas un reseau d'observation, c'est un semis aleatoire qui se          ║
║  teleporte chaque jour.                                                      ║
║                                                                              ║
║  Deux consequences, et la seconde est la plus genante :                      ║
║                                                                              ║
║   - les positions changent en permanence, alors qu'un reseau reel est fixe   ║
║     et laisse des regions durablement non observees ;                        ║
║   - les lacunes sont independantes dans le temps, donc elles se compensent   ║
║     par moyennage. Les vraies lacunes sont PERSISTANTES : une bouee tombe    ║
║     en panne et le reste jusqu'a la campagne suivante, ce qui creuse un      ║
║     trou structurel pendant des mois.                                        ║
║                                                                              ║
║  Un autoencodeur entraine sur des lacunes independantes voit donc un regime  ║
║  plus favorable que le reel, et sa reconstruction est optimiste.             ║
║                                                                              ║
║  Ce module fournit un echantillonneur de masques calque sur le modele de     ║
║  maintien : positions fixes, pannes exponentielles, reparations aux dates    ║
║  reelles de passage du navire. Le reseau et le budget sont tirés au hasard   ║
║  d'une epoque a l'autre pour que le modele reste robuste a toute             ║
║  configuration -- ce qui etait la seule bonne propriete du masque aleatoire, ║
║  et qu'il faut conserver.                                                    ║
║                                                                              ║
║  Ce que ca ne fait PAS : rendre la brique 1 optimale. Sur un ocean lineaire  ║
║  a modele quasi parfait, un filtre de Kalman reste imbattable (ceiling.py).  ║
║  Ca la rend REALISTE, ce qui est une autre question et vaut par soi-meme.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from config import PORT_XY_FRAC, NX, NY, DX_KM
from data.dataset import OceanOEDDataset


class MaintenanceMaskSampler:
    """
    Genere des sequences de masques (positions fixes + pannes persistantes).

    Un "tirage" = un reseau, un budget, un plan de campagnes, et une
    trajectoire de pannes sur toute la periode. Le sampler en pre-calcule
    `n_draws` et pioche dedans, ce qui evite de replanifier a chaque item du
    DataLoader.
    """

    def __init__(self, env, n_days, n_draws=8, n_range=(8, 30),
                 budget_frac_range=(0.2, 1.2), seed=0, verbose=True,
                 greedy_frac=0.25):
        from campaign import greedy_under_budget, auto_budget_levels
        from scenario import visit_calendar, simulate_uptime

        self.nx, self.ny = env.T.shape[1], env.T.shape[2]
        rng = np.random.default_rng(seed)
        _, viable = auto_budget_levels(env, n_ref=n_range[1],
                                       fractions=(0.35,))
        self.draws = []
        for d in range(n_draws):
            n = int(rng.integers(*n_range))
            budget = viable * float(rng.uniform(*budget_frac_range))
            # Un reseau optimise pour ce budget, ou un tirage libre. Le
            # glouton coute une minute par tirage sur un grand domaine, donc
            # il ne sert qu'a une minorite : ce qui compte pour la diversite,
            # c'est le NOMBRE de configurations vues, pas leur optimalite.
            if rng.random() < greedy_frac:
                g = greedy_under_budget(env, budget, "effective",
                                        verbose=False, n_max=n)
                idx = g["idx"]
            else:
                idx = env.sample_feasible(n, rng=rng)
            if len(idx) < 3:
                continue
            ev = env.evaluate(idx, budget_keur=budget, refine=True,
                              with_plan=True)
            cal = visit_calendar(ev["plan"], env.maint.p)
            up = simulate_uptime(len(idx), cal, env.maint.p.mtbf_days,
                                 max(n_days / 365.0, 1.0), rng)
            pos = np.array([env.candidate_positions[i] for i in idx], int)
            self.draws.append((pos, up))
            if verbose:
                print(f"    tirage {d+1}/{n_draws} : N={len(idx)}, "
                      f"budget {budget:.0f} k€, dispo {up.mean():.2f}",
                      flush=True)
        if not self.draws:
            raise RuntimeError("Aucun tirage valide.")

    def mask(self, t, rng=None):
        rng = rng or np.random
        pos, up = self.draws[rng.randint(len(self.draws))
                             if hasattr(rng, "randint")
                             else int(rng.integers(len(self.draws)))]
        row = up[t % len(up)]
        m = np.zeros((self.nx, self.ny), dtype=np.float32)
        live = pos[row]
        if len(live):
            m[live[:, 0], live[:, 1]] = 1.0
        return m


class MaintenanceOEDDataset(OceanOEDDataset):
    """
    `OceanOEDDataset` dont le masque provient du modele de maintien.

    Tout le reste -- normalisation, bruit d'observation, augmentation -- est
    hérité, de sorte que la comparaison avec l'entrainement historique ne
    porte que sur le masque.
    """

    def __init__(self, T, S, sampler, mix_random=0.0, seed=0, **kwargs):
        super().__init__(T, S, **kwargs)
        self.sampler = sampler
        self.mix_random = float(mix_random)
        self._rng = np.random.default_rng(seed)

    def _random_mask(self, n_obs):        # n_obs ignore : c'est le plan qui decide
        # Melange optionnel des deux regimes. Le masque aleatoire est une
        # AUGMENTATION DE DONNEES : il expose le modele a des configurations
        # que la maintenance ne produit jamais. S'entrainer uniquement sur
        # quelques reseaux fixes fait memoriser leurs positions -- c'est ce
        # qu'on observe quand le modele "maintenance" s'effondre hors de son
        # regime tout en ayant la meilleure perte d'entrainement.
        if self.mix_random > 0 and self._rng.random() < self.mix_random:
            return super()._random_mask(n_obs)
        return self.sampler.mask(self._t_current)

    def __getitem__(self, t):
        self._t_current = t
        return super().__getitem__(t)


def build_maintenance_datasets(T, S, env, split=0.8, n_draws=8, seed=0,
                               augment_train=False, verbose=True,
                               mix_random=0.0, greedy_frac=0.25, **kwargs):
    """
    Equivalent de `build_datasets`, masques de maintenance.

    `mix_random` ne s'applique qu'a l'ENTRAINEMENT. La validation reste en
    maintenance pure, sinon on evaluerait sur un regime qui n'existe pas en
    operation.
    """
    n = len(T); n_tr = int(n * split)
    if verbose:
        print("  Echantillonneur de masques de maintenance :", flush=True)
    smp = MaintenanceMaskSampler(env, n_days=n, n_draws=n_draws, seed=seed,
                                 verbose=verbose, greedy_frac=greedy_frac)
    return (MaintenanceOEDDataset(T[:n_tr], S[:n_tr], smp,
                                  mix_random=mix_random, seed=seed,
                                  augment=augment_train, **kwargs),
            MaintenanceOEDDataset(T[n_tr:], S[n_tr:], smp, mix_random=0.0,
                                  seed=seed + 1, augment=False, **kwargs))


def mask_statistics(sampler, n_days):
    """
    Compare les deux regimes de lacunes sur les memes positions.

    La quantite qui compte est la PERSISTANCE : duree moyenne d'une
    interruption. Elle vaut un jour pour un masque aleatoire par
    construction, et plusieurs mois pour un reseau reellement entretenu.
    """
    out = []
    for pos, up in sampler.draws:
        n = up.shape[1]
        runs = []
        for j in range(n):
            c = 0
            for t in range(len(up)):
                if not up[t, j]:
                    c += 1
                elif c:
                    runs.append(c); c = 0
            if c:
                runs.append(c)
        out.append({"n_buoys": int(n),
                    "availability": float(up.mean()),
                    "mean_gap_days": float(np.mean(runs)) if runs else 0.0,
                    "max_gap_days": float(np.max(runs)) if runs else 0.0})
    return out
