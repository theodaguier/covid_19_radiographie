"""Detection d'environnement et configuration backend.

Ce module centralise la configuration qui differe entre l'execution locale
(Mac Apple Silicon, eventuellement tensorflow-metal) et Google Colab (GPU T4/L4).
Les notebooks 06/07/08 importent ces helpers pour rester portables.

Usage typique en tete de notebook :

    from src.utils.env import setup_environment, get_paths
    info = setup_environment()      # configure GPU + mixed precision
    paths = get_paths()             # chemins resolus selon l'environnement
"""

from __future__ import annotations

import os
from pathlib import Path


def is_colab() -> bool:
    """Vrai si le code tourne dans Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def get_base_dir() -> Path:
    """Racine du projet, resolue selon l'environnement.

    - En local : remonte depuis ce fichier (src/utils/env.py -> racine).
    - Sur Colab : utilise la variable d'environnement COVID_BASE_DIR si definie
      (ex. un dossier Google Drive monte), sinon le repertoire courant.
    """
    env_override = os.environ.get("COVID_BASE_DIR")
    if env_override:
        return Path(env_override).resolve()
    if is_colab():
        return Path.cwd()
    return Path(__file__).resolve().parents[2]


def detect_gpu() -> list[str]:
    """Retourne la liste des GPU visibles par TensorFlow (vide si aucun)."""
    import tensorflow as tf

    return [d.name for d in tf.config.list_physical_devices("GPU")]


def setup_environment(mixed_precision: bool = True, verbose: bool = True) -> dict:
    """Configure TensorFlow pour l'environnement courant.

    - Active la croissance memoire GPU (evite l'allocation totale).
    - Active la precision mixte (mixed_float16) uniquement si un GPU est present
      (la couche softmax finale reste en float32, gere dans build_tl_model).
    - Choisit un batch size par defaut adapte au backend.

    Retourne un dict d'information (env, gpus, batch_size, mixed_precision).
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Doit etre appele avant l'initialisation du contexte ; on ignore sinon.
            pass

    use_mixed = bool(mixed_precision and gpus)
    if use_mixed:
        from tensorflow.keras import mixed_precision as mp

        mp.set_global_policy("mixed_float16")

    info = {
        "colab": is_colab(),
        "gpus": [g.name for g in gpus],
        "n_gpus": len(gpus),
        "mixed_precision": use_mixed,
        "default_batch_size": 32 if gpus else 8,
        "base_dir": str(get_base_dir()),
        "tf_version": tf.__version__,
    }

    if verbose:
        print("=== Configuration environnement ===")
        print(f"Colab            : {info['colab']}")
        print(f"TensorFlow       : {info['tf_version']}")
        print(f"GPU detecte(s)   : {info['gpus'] or 'AUCUN (CPU uniquement)'}")
        print(f"Mixed precision  : {info['mixed_precision']}")
        print(f"Batch size def.  : {info['default_batch_size']}")
        print(f"Base dir         : {info['base_dir']}")
        if not gpus:
            print(
                "ATTENTION : aucun GPU detecte. L'entrainement des modeles de "
                "Transfer Learning sera tres lent. Utilisez Google Colab (GPU)."
            )
    return info


def get_paths() -> dict:
    """Chemins standards du projet, resolus depuis la base dir."""
    base = get_base_dir()
    return {
        "base": base,
        "data": base / "data" / "COVID-19_Radiography_Dataset",
        "models": base / "models",
        "models_transfer": base / "models" / "transfer",
        "reports": base / "reports",
        "figures": base / "reports" / "figures",
        "splits": base / "reports" / "splits",
        "history": base / "reports" / "history",
        "metrics": base / "reports" / "metrics",
    }


def ensure_dirs() -> dict:
    """Cree les dossiers d'artefacts s'ils n'existent pas. Retourne les chemins."""
    paths = get_paths()
    for key in ("models_transfer", "figures", "splits", "history", "metrics"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths
