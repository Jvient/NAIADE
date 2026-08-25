"""Vérifie quelles fonctionnalités sont présentes dans chaque module.
Plus fiable qu'un md5 : tolère les éditions locales.
    python check_state.py
"""
from pathlib import Path

CHECKS = {
    "00_make_obs.py": [
        ("mode GLORYS",            "_load_glorys"),
        ("PIRATA réel (17)",       "pirata_real"),
        ("conversion du bruit",    "unités normalisées"),
        ("advection mesurée",      "estimate_advection"),
    ],
    "obs_operator.py": [
        ("import config par nom",  "_cfg_get"),
        ("advection empirique",    "def estimate_advection"),
        ("manquants non aléat.",   "hazard_var_amp"),
        ("split par capteur",      "def split_sensors"),
    ],
    "obsonly.py": [
        ("NLL held-out",           "class HeldOutNLL"),
        ("recalibration variance", "fit_variance_scale"),
        ("augmentation zonale",    "flip_roll"),
        ("arrêt anticipé",         "patience"),
        ("mode sigma",             "lobo_sigma_scores"),
        ("test de monotonie",      "monotonicity_check"),
    ],
    "gnn_lobo.py": [
        ("corrélation à lag",      "lagged_correlation"),
        ("dropout de features",    "feat_dropout"),
        ("correctif CUDA",         "detach().cpu().numpy()"),
        ("arrêt anticipé",         "patience"),
    ],
    "validate_obsonly.py": [
        ("L mesurée",              "estimate_decorrelation_px"),
        ("RMSE hors échantillon",  "RMSE_oos"),
        ("table de sensibilité",   "shrinkage"),
    ],
    "naiade_compat.py": [
        ("ObservabilityVAE",       "ObservabilityVAE"),
        ("résolution du seed",     "resolve_seed_fn"),
    ],
    "pirata_real.py": [
        ("17 positions",           "PT075"),
        ("pas de rabattage",       "jamais rabattue"),
    ],
    "dataset_glorys.py": [
        ("GlorysData",             "class GlorysData"),
    ],
}

missing_files, missing_feat = [], []
for fn, checks in CHECKS.items():
    p = Path(fn)
    if not p.exists():
        print(f"[ABSENT] {fn}")
        missing_files.append(fn)
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    flags = [(lbl, tok in txt) for lbl, tok in checks]
    ok = sum(v for _, v in flags)
    print(f"[{'OK ' if ok == len(flags) else 'PART'}] {fn:22s} {ok}/{len(flags)}")
    for lbl, v in flags:
        if not v:
            print(f"          manque : {lbl}")
            missing_feat.append((fn, lbl))

print("-" * 60)
if missing_files:
    print(f"{len(missing_files)} fichier(s) à copier : {', '.join(missing_files)}")
if missing_feat:
    print(f"{len(missing_feat)} fonctionnalité(s) manquante(s) — recopiez le "
          "fichier complet plutôt que d'appliquer un patch")
if not missing_files and not missing_feat:
    print("Tout est à jour. Rien à appliquer.")
