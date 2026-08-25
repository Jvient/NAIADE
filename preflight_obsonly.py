"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PRÉFLIGHT — compatibilité de la branche courante avec l'extension obs-only  ║
║                                                                              ║
║  À lancer en PREMIER sur une nouvelle branche (ex. obs-ops partie de main).  ║
║  Ne modifie rien, n'entraîne rien : inspecte et rapporte.                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

    python preflight_obsonly.py
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

OK, WARN, BAD = "  OK  ", " ATTN ", " STOP "
_rows = []


def note(level, item, msg=""):
    _rows.append((level, item, msg))


# ── 1. fichiers ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
CORE = {"config.py": True, "01_autoencoder.py": True,
        "dataset.py": False, "dataset_glorys.py": False,
        "02_gnn.py": False, "03_rl.py": False}
for f, required in CORE.items():
    if (ROOT / f).exists():
        note(OK, f, "présent")
    elif required:
        note(BAD, f, "MANQUANT — indispensable")
    else:
        note(WARN, f, "absent")

if not (ROOT / "dataset.py").exists() and not (ROOT / "dataset_glorys.py").exists():
    note(BAD, "source de données",
         "ni dataset.py ni dataset_glorys.py : aucun nature run disponible")

# ── 2. dépendances ────────────────────────────────────────────────────────────
for mod, required in (("numpy", True), ("torch", True), ("scipy", False),
                      ("matplotlib", False), ("torch_geometric", False)):
    try:
        m = __import__(mod)
        note(OK, mod, getattr(m, "__version__", ""))
    except ImportError:
        note(BAD if required else WARN, mod,
             "manquant" + ("" if required else " (optionnel)"))

# ── 3. config.py ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
try:
    import config as cfg
    note(OK, "import config", "")
except Exception as e:
    note(BAD, "import config", str(e)[:60])
    cfg = None

if cfg is not None:
    for k in ("NX", "NY"):
        note(OK if hasattr(cfg, k) else BAD, f"config.{k}",
             "" if hasattr(cfg, k) else "requis")
    note(OK if hasattr(cfg, "DEVICE") else WARN, "config.DEVICE",
         "" if hasattr(cfg, "DEVICE") else "déduit de torch.cuda (compat)")

    from naiade_compat import _SEED_NAMES
    found = [n for n in _SEED_NAMES if callable(getattr(cfg, n, None))]
    if found:
        note(OK, "fonction de seed",
             f"config.{found[0]}()" + ("" if found[0] == "set_global_seed"
                                       else "  (alias, résolu par compat)"))
    else:
        cands = [n for n in dir(cfg)
                 if callable(getattr(cfg, n, None)) and "seed" in n.lower()]
        note(WARN, "fonction de seed",
             f"aucune connue — repli interne. Vu dans config : {cands}"
             if cands else "aucune — repli interne (numpy+torch)")

    # ajoutés sur glo12, absents de main : repli automatique mais à valider
    FALLBACK = {
        "OBS_NOISE_T": "repli sur OBS_NOISE_STD",
        "OBS_NOISE_S": "repli sur 0.4 * OBS_NOISE_STD — À RECALER",
        "DX_KM": "repli 5.0 km/px",
        "INFLUENCE_RADIUS_KM": "repli 90 km (validate_obsonly)",
        "EVF_SHRINKAGE": "repli 0.9 (validate_obsonly)",
    }
    for k, msg in FALLBACK.items():
        note(OK if hasattr(cfg, k) else WARN, f"config.{k}",
             "" if hasattr(cfg, k) else msg)

    if hasattr(cfg, "OBS_NOISE_STD") and not hasattr(cfg, "OBS_NOISE_T"):
        note(WARN, "bruit d'observation",
             f"OBS_NOISE_STD={cfg.OBS_NOISE_STD} unique pour T et S : "
             "~2 % du signal en T mais ~25 % en S. Scinder.")

# ── 4. API de l'autoencodeur ──────────────────────────────────────────────────
try:
    spec = importlib.util.spec_from_file_location(
        "ae_check", ROOT / "01_autoencoder.py")
    ae = importlib.util.module_from_spec(spec)
    sys.modules["ae_check"] = ae
    spec.loader.exec_module(ae)
    note(OK, "import 01_autoencoder", "")

    from naiade_compat import (find_ae_class, list_module_classes,
                               check_ae_signature)
    classes = list_module_classes(ae)
    note(OK, "classes nn.Module trouvées",
         ", ".join(n for n, _ in classes) or "aucune")
    try:
        AE_CLS = find_ae_class(ae, verbose=False)
        note(OK, "classe autoencodeur", AE_CLS.__name__)
    except AttributeError as e:
        AE_CLS = None
        note(BAD, "classe autoencodeur", str(e).replace(chr(10), " | ")[:150])

    if AE_CLS is not None:
        ae.ObservabilityAE = AE_CLS
        sig = inspect.signature(AE_CLS.__init__).parameters
        need = ["in_ch", "out_ch", "base_ch", "latent_ch",
                "dropout_p", "cond_dim"]
        miss = [k for k in need if k not in sig]
        note(OK if not miss else BAD, f"{AE_CLS.__name__}.__init__",
             "signature compatible" if not miss
             else f"paramètres manquants : {miss}")
        for a in ("encode", "decode"):
            note(OK if hasattr(AE_CLS, a) else BAD,
                 f"{AE_CLS.__name__}.{a}", "méthode")
        # cond_embed est créé dans __init__ : instancier pour le vérifier
        try:
            probe = AE_CLS(in_ch=4, out_ch=4, base_ch=4, latent_ch=4)
            note(OK if hasattr(probe, "cond_embed") else BAD,
                 f"{AE_CLS.__name__}.cond_embed",
                 "attribut d'instance, FiLM disponible")
            note(OK, "instanciation 4->4 canaux", "réussie")
            del probe
        except Exception as e:
            note(BAD, "instanciation 4->4 canaux",
                 f"{type(e).__name__}: {str(e)[:55]}")
        d = sig.get("in_ch")
        if d is not None and d.default not in (3, 4):
            note(WARN, "in_ch par défaut",
                 f"{d.default} — l'extension force 4 (T, S, mask_T, mask_S)")


    note(OK, "AELoss",
         ("présente — NON modifiée, obsonly.py utilise HeldOutNLL"
          if hasattr(ae, "AELoss") else "absente — sans effet, non utilisée"))
except Exception as e:
    note(BAD, "import 01_autoencoder", f"{type(e).__name__}: {str(e)[:70]}")

# ── 5. générateur synthétique ─────────────────────────────────────────────────
if (ROOT / "dataset.py").exists():
    try:
        spec = importlib.util.spec_from_file_location("ds", ROOT / "dataset.py")
        ds = importlib.util.module_from_spec(spec)
        sys.modules["ds"] = ds
        spec.loader.exec_module(ds)
        for k in ("SyntheticOceanGenerator", "build_datasets"):
            note(OK if hasattr(ds, k) else WARN, f"dataset.{k}", "")
        if hasattr(ds, "SyntheticOceanGenerator"):
            g = inspect.signature(ds.SyntheticOceanGenerator.generate_dataset)
            note(OK, "generate_dataset", f"signature {g}")
    except Exception as e:
        note(WARN, "import dataset", f"{type(e).__name__}: {str(e)[:60]}")

# ── 6. modules de l'extension ─────────────────────────────────────────────────
for f in ("naiade_compat.py", "obs_operator.py", "obsonly.py",
          "gnn_lobo.py", "validate_obsonly.py"):
    note(OK if (ROOT / f).exists() else BAD, f,
         "" if (ROOT / f).exists() else "à copier depuis l'extension")

# ── rapport ───────────────────────────────────────────────────────────────────
print("=" * 74)
print("  PRÉFLIGHT obs-only —", ROOT.resolve().name)
print("=" * 74)
for lvl, item, msg in _rows:
    print(f"[{lvl}] {item:<28s} {msg}")

n_bad = sum(1 for l, _, _ in _rows if l == BAD)
n_warn = sum(1 for l, _, _ in _rows if l == WARN)
print("-" * 74)
if n_bad:
    print(f"  {n_bad} bloquant(s), {n_warn} avertissement(s) — corriger avant de lancer.")
else:
    print(f"  Aucun bloquant, {n_warn} avertissement(s).")

print("""
  Rappels spécifiques à une branche partie de main
  ------------------------------------------------
  · OBS_NOISE_S : le repli 0.4*OBS_NOISE_STD est ARBITRAIRE. Le bruit
    instrumental d'un capteur de salinité n'a pas de raison de valoir 40 % de
    celui d'un capteur de température — à recaler sur les specs réelles.
  · Dériveurs : ne PAS passer config.U_MEAN / V_MEAN directement à
    add_drifters(). Ces constantes sont dans les unités internes du
    générateur, pas en pixels par pas de temps. Utiliser :
        from obs_operator import estimate_advection
        u, v, c = estimate_advection(T)
        net.add_drifters(n=10, u=u, v=v)
  · INFLUENCE_RADIUS_KM = 90 km avec DX_KM = 5 donne L = 18 px, valeur
    diagnostiquée sur le nature run synthétique de moyenne latitude. Elle est
    correcte pour main, FAUSSE en Atlantique tropical.
  · Pas de masque océan sur main : laisser ocean=None dans ObsNetwork.
""")
sys.exit(1 if n_bad else 0)
