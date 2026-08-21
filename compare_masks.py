"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  COMPARAISON DES DEUX REGIMES DE MASQUE POUR LA BRIQUE 1                     ║
║                                                                              ║
║  Deux entrainements identiques a une chose pres -- le masque d'observation : ║
║                                                                              ║
║      aleatoire    5 a 60 pixels tires uniformement, RETIRES CHAQUE JOUR      ║
║                   (regime historique de la brique 1)                         ║
║      maintenance  positions fixes, pannes exponentielles, reparations aux    ║
║                   dates reelles de passage du navire                         ║
║                                                                              ║
║  L'evaluation est CROISEE, sur un jeu de validation commun et FIGE : les     ║
║  memes champs, les memes masques, les memes bruits pour les deux modeles.    ║
║  Sans ce gel, on comparerait deux tirages differents et l'ecart mesure       ║
║  serait du bruit.                                                            ║
║                                                                              ║
║  La lecture qui compte est la colonne "masques de maintenance" : c'est le    ║
║  regime que la brique 1 rencontrera en operation. La colonne "aleatoire" ne  ║
║  sert qu'a verifier qu'on n'a pas simplement echange un biais contre un      ║
║  autre.                                                                      ║
║                                                                              ║
║  Attendu : le modele entraine sur des masques realistes fait MOINS BIEN en   ║
║  chiffre absolu que l'ancien evalue sur son propre regime -- les lacunes     ║
║  persistantes sont plus dures que des lacunes independantes qui se           ║
║  compensent par moyennage. Ce n'est pas une regression, c'est la fin d'une   ║
║  surestimation.                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

    NAIADE_DOMAIN=demo python compare_masks.py --epochs 12
"""

from __future__ import annotations

import argparse, importlib.util, json, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (DOMAIN, NX, NY, DX_KM, PORT_XY_FRAC, NT, DEVICE)
from data.dataset import SyntheticOceanGenerator, build_datasets
from maint_masks import build_maintenance_datasets, mask_statistics


def load_brick(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), Path(__file__).with_name(name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def freeze_validation(ds, n_batches, batch_size, seed=0):
    """
    Fige un jeu de validation : masques, bruits et champs tires une fois et
    reutilises tels quels pour les deux modeles. Les datasets tirent leur
    masque a chaque appel de __getitem__, donc sans gel les deux evaluations
    ne verraient pas les memes donnees.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    out = []
    for i, b in enumerate(dl):
        if i >= n_batches:
            break
        out.append(tuple(t.clone() for t in b))
    return out


@torch.no_grad()
def evaluate(model, batches):
    """RMSE de reconstruction, separee observe / non observe.
    C'est l'erreur sur les points NON OBSERVES qui mesure la reconstruction ;
    sur les points observes, le modele n'a qu'a recopier."""
    model.eval()
    se_u = n_u = se_o = n_o = 0.0
    for x, y, m in batches:
        x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
        out = model(x)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        pred = pred[:, :2]
        err2 = (pred - y) ** 2
        mo = m.expand_as(err2)
        se_o += float((err2 * mo).sum()); n_o += float(mo.sum())
        se_u += float((err2 * (1 - mo)).sum()); n_u += float((1 - mo).sum())
    return (float(np.sqrt(se_u / max(n_u, 1))),
            float(np.sqrt(se_o / max(n_o, 1))))


def train_model(b1, train_ds, args, tag):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = b1.ObservabilityVAE(base_ch=args.base_ch).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = b1.VAELoss(w_obs=1.0, w_unobs=args.w_unobs)
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    drop_last=True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for x, y, m in dl:
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            out = model(x)
            pred, mu, logvar = out[0], out[1], out[2]
            loss = crit(pred, y, m, mu, logvar, beta=args.beta)[0]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss)
        if (ep + 1) % max(args.epochs // 4, 1) == 0:
            print(f"    [{tag}] epoque {ep+1}/{args.epochs} "
                  f"loss {tot/max(len(dl),1):.4f}", flush=True)
    print(f"    [{tag}] entraine en {time.time()-t0:.0f}s", flush=True)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nt", type=int, default=NT)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument("--w_unobs", type=float, default=1.0)
    p.add_argument("--grid_x", type=int, default=16)
    p.add_argument("--grid_y", type=int, default=24)
    p.add_argument("--n_max", type=int, default=30)
    p.add_argument("--maintenance", type=str, default="pirata",
                   choices=["regional", "pirata"])
    p.add_argument("--n_draws", type=int, default=48,
                   help="Nombre de configurations de reseau vues a "
                        "l entrainement. Trop peu (8) et le modele memorise "
                        "ces positions au lieu d apprendre a reconstruire.")
    p.add_argument("--greedy_frac", type=float, default=0.25,
                   help="Part des tirages passant par le glouton (couteux)")
    p.add_argument("--mix_random", type=float, default=0.0,
                   help="Part des items d entrainement tires avec le masque "
                        "aleatoire. Le masque aleatoire agit comme une "
                        "augmentation de donnees ; la validation reste en "
                        "maintenance pure.")
    p.add_argument("--val_batches", type=int, default=24)
    p.add_argument("--out_dir", type=str, default="outputs")
    a = p.parse_args()

    b1 = load_brick("01_autoencoder.py")
    b3 = load_brick("03_rl.py")

    print(f"\n  Domaine {DOMAIN} {NX}x{NY} @ {DX_KM:.0f} km | "
          f"{a.epochs} epoques | profil {a.maintenance}")
    gen = SyntheticOceanGenerator()
    T, S = gen.generate_dataset(nt=a.nt, seed=a.seed)

    port = np.array([PORT_XY_FRAC[0] * NX, PORT_XY_FRAC[1] * NY]) * DX_KM
    maint = b3.MaintenanceModel(b3.get_params(a.maintenance), port)
    env = b3.OceanNetworkEnv(T, S, grid_x=a.grid_x, grid_y=a.grid_y,
                             n_min=8, n_max=a.n_max, maintenance=maint)

    tr_rand, va_rand = build_datasets(T, S)
    tr_mnt, va_mnt = build_maintenance_datasets(
        T, S, env, n_draws=a.n_draws, seed=a.seed,
        mix_random=a.mix_random, greedy_frac=a.greedy_frac)

    print("\n  Persistance des lacunes (tirages de maintenance)")
    for s in mask_statistics(tr_mnt.sampler, len(T)):
        print(f"    N={s['n_buoys']:>2} | dispo {s['availability']:.2f} | "
              f"interruption moyenne {s['mean_gap_days']:>5.0f} j | "
              f"max {s['max_gap_days']:>5.0f} j")
    print("    (masque aleatoire : 1 jour par construction, positions\n"
          "     redistribuees a chaque pas de temps)")

    val = {"aleatoire": freeze_validation(va_rand, a.val_batches,
                                          a.batch_size, seed=1),
           "maintenance": freeze_validation(va_mnt, a.val_batches,
                                            a.batch_size, seed=1)}

    print("\n  Entrainements")
    models = {"aleatoire": train_model(b1, tr_rand, a, "aleatoire"),
              "maintenance": train_model(b1, tr_mnt, a, "maintenance")}

    print(f"\n  RMSE sur les points NON OBSERVES (jeu de validation commun)")
    print(f"  {'entraine sur':>14} | {'eval aleatoire':>15} | "
          f"{'eval maintenance':>17}")
    print("  " + "-" * 54)
    res = {}
    for tag, mdl in models.items():
        row = {k: evaluate(mdl, v) for k, v in val.items()}
        res[tag] = {k: {"rmse_unobs": u, "rmse_obs": o}
                    for k, (u, o) in row.items()}
        print(f"  {tag:>14} | {row['aleatoire'][0]:>15.4f} | "
              f"{row['maintenance'][0]:>17.4f}")

    ra = res["aleatoire"]["maintenance"]["rmse_unobs"]
    rm = res["maintenance"]["maintenance"]["rmse_unobs"]
    gain = (ra - rm) / max(ra, 1e-9) * 100
    print(f"\n  En regime realiste (colonne de droite), l'entrainement sur\n"
          f"  masques de maintenance change la RMSE de {gain:+.1f} %.")
    if gain > 3:
        print("  -> L'entrainement realiste ameliore la reconstruction dans le\n"
              "     regime qui compte. A adopter comme defaut de la brique 1.")
    elif gain > -3:
        print("  -> Pas d'ecart significatif. Le masque realiste reste\n"
              "     preferable par honnetete du protocole, mais il ne faut pas\n"
              "     lui attribuer de gain de performance.")
    else:
        print("  -> L'entrainement realiste degrade la reconstruction : le\n"
              "     masque aleatoire agit comme une augmentation de donnees.\n"
              "     Envisager de melanger les deux regimes.")

    d = (res["aleatoire"]["aleatoire"]["rmse_unobs"]
         - res["aleatoire"]["maintenance"]["rmse_unobs"])
    print(f"\n  Surestimation de l'ancien protocole : le modele historique\n"
          f"  affiche {res['aleatoire']['aleatoire']['rmse_unobs']:.4f} sur son\n"
          f"  propre regime contre "
          f"{res['aleatoire']['maintenance']['rmse_unobs']:.4f} en regime\n"
          f"  realiste, soit un ecart de {abs(d):.4f} "
          f"({abs(d)/max(res['aleatoire']['aleatoire']['rmse_unobs'],1e-9)*100:.0f} %).")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "mask_regime_comparison.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  Resultats -> {out / 'mask_regime_comparison.json'}")


if __name__ == "__main__":
    main()
