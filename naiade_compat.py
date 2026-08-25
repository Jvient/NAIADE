"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  COUCHE DE COMPATIBILITÉ INTER-BRANCHES                                      ║
║                                                                              ║
║  L'extension obs-only se greffe sur du code qui a divergé entre main et      ║
║  glo12 (noms de classes, symboles de config.py). Plutôt que d'imposer un     ║
║  renommage sur la branche cible, on résout les symboles à l'exécution.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import inspect

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  DEVICE / SEED — présents sur glo12, pas garantis ailleurs
# ══════════════════════════════════════════════════════════════════════════════

_SEED_NAMES = ("set_global_seed", "set_seed", "seed_everything",
               "set_all_seeds", "fix_seed", "setup_seed")


def get_device():
    try:
        import config
        d = getattr(config, "DEVICE", None)
        if d:
            return d
    except Exception:
        pass
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_seed_fn(verbose=False):
    """Retourne la fonction de seed de config.py, quel que soit son nom.

    Si aucune n'existe, on en fournit une équivalente plutôt que d'échouer :
    la reproductibilité ne doit pas dépendre d'une convention de nommage.
    """
    try:
        import config
        for n in _SEED_NAMES:
            f = getattr(config, n, None)
            if callable(f):
                if verbose and n != "set_global_seed":
                    print(f"  [compat] seed : config.{n}() utilisée")
                return f
    except Exception:
        pass

    def _fallback(seed: int):
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if verbose:
        print("  [compat] seed : aucune fonction dans config.py, "
              "repli interne")
    return _fallback


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSE AUTOENCODEUR — nom variable selon la branche
# ══════════════════════════════════════════════════════════════════════════════

# ObservabilityVAE : nom historique sur main. Le code est un AE déterministe
# depuis la v4 (la reparamétrisation VAE a été retirée, forward renvoie
# logvar = zeros), mais la classe n'a jamais été renommée.
_AE_NAMES = ("ObservabilityAE", "ObservabilityVAE", "ObservabilityAutoencoder",
             "ObsAE", "ObservabilityUNet", "AEUNet", "UNetAE",
             "ObservabilityModel", "Autoencoder", "AE")

# in_ch + out_ch ne suffisent PAS à identifier l'autoencodeur : ResDoubleConv,
# Down et FiLMUp les ont aussi. On exige en plus un paramètre de capacité
# globale ET les méthodes encode/decode, que seuls les modèles complets ont.
_NEEDED_KWARGS = ("in_ch", "out_ch")
_CAPACITY_KWARGS = ("latent_ch", "base_ch")
_NEEDED_METHODS = ("encode", "decode")


def list_module_classes(mod):
    """Toutes les classes nn.Module définies dans le module, avec leur
    signature — pour diagnostiquer quand rien ne correspond."""
    import torch.nn as nn
    out = []
    for name, obj in vars(mod).items():
        if (inspect.isclass(obj) and issubclass(obj, nn.Module)
                and obj.__module__ == mod.__name__):
            try:
                kw = list(inspect.signature(obj.__init__).parameters)[1:]
            except (TypeError, ValueError):
                kw = ["?"]
            out.append((name, kw))
    return out


def find_ae_class(mod, override=None, verbose=True):
    """Résout la classe de l'autoencodeur.

    Ordre : nom explicite (--ae_class) > noms connus > toute classe nn.Module
    dont __init__ accepte in_ch et out_ch. En dernier recours, une erreur qui
    LISTE ce que le module contient réellement.
    """
    import torch.nn as nn

    if override:
        cls = getattr(mod, override, None)
        if cls is None:
            raise AttributeError(
                f"--ae_class {override} : absent de {mod.__name__}. "
                f"Classes disponibles : "
                f"{[n for n, _ in list_module_classes(mod)]}")
        return cls

    for n in _AE_NAMES:
        cls = getattr(mod, n, None)
        if inspect.isclass(cls) and issubclass(cls, nn.Module):
            if verbose and n != "ObservabilityAE":
                print(f"  [compat] autoencodeur : {n} (nom non standard)")
            return cls

    # heuristique : signature compatible avec la greffe 4 canaux, capacité
    # paramétrable, et méthodes encode/decode (exclut les blocs élémentaires)
    cands = [(n, kw) for n, kw in list_module_classes(mod)
             if all(k in kw for k in _NEEDED_KWARGS)
             and any(k in kw for k in _CAPACITY_KWARGS)
             and all(hasattr(getattr(mod, n), m) for m in _NEEDED_METHODS)]
    if len(cands) == 1:
        if verbose:
            print(f"  [compat] autoencodeur déduit par signature : {cands[0][0]}")
        return getattr(mod, cands[0][0])

    found = list_module_classes(mod)
    raise AttributeError(
        f"Aucune classe d'autoencodeur identifiable dans {mod.__name__}.\n"
        f"  Classes trouvées : {[n for n, _ in found]}\n"
        + (f"  Candidates plausibles : {[n for n, _ in cands]}\n"
           if cands else "")
        + "  Relancez avec --ae_class <NomDeLaClasse>.")


def check_ae_signature(cls):
    """Vérifie que la classe accepte les kwargs dont la greffe a besoin.
    Retourne (ok, manquants, defaults_utilisables)."""
    sig = inspect.signature(cls.__init__).parameters
    need = ("in_ch", "out_ch", "base_ch", "latent_ch", "dropout_p", "cond_dim")
    missing = [k for k in need if k not in sig]
    return (not missing), missing, {k: sig[k].default
                                    for k in need if k in sig}
