"""Runner headless : entraine les modeles de Transfer Learning de bout en bout.

Pour chaque backbone : protocole 2 phases -> evaluation sur le test set fige ->
sauvegarde des metriques (JSON), de l'historique (CSV) et des figures (confusion,
ROC/PR). Resilient : un echec sur un backbone n'interrompt pas les suivants.

Usage :
    python -m src.models.run_transfer_learning              # VGG16 + EfficientNetB0 + ResNet50
    python -m src.models.run_transfer_learning vgg16        # un seul
    python -m src.models.run_transfer_learning vgg16 resnet50

Pense pour tourner en arriere-plan ; logs en ligne par epoque (verbose=2).
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

from src.utils.env import setup_environment, ensure_dirs
from src.data import tf_pipeline as tp
from src.models import transfer_learning as tl
from src.models import evaluation as ev

DEFAULT_BACKBONES = ["vgg16", "efficientnetb0", "resnet50"]
DEFAULT_TRAINING = {
    # Le rapport 14.06.26 presente VGG16 avec 3 epoques + lr=1e-4,
    # puis 3 epoques de fine-tuning + lr=1e-5.
    "vgg16": {"epochs_phase1": 3, "epochs_phase2": 3, "lr_phase1": 1e-4, "lr_phase2": 1e-5},
}


def run_one(backbone: str, paths: dict, class_weights: dict, batch_size: int) -> dict | None:
    spec = tl.get_spec(backbone)
    bs = batch_size
    if spec.img_size == 299:        # InceptionV3 plus gourmand
        bs = max(8, batch_size // 2)

    train_ds = tp.make_dataset("train", spec.img_size, spec.preprocess_fn, bs, augment=True)
    val_ds = tp.make_dataset("val", spec.img_size, spec.preprocess_fn, bs)
    test_ds = tp.make_dataset("test", spec.img_size, spec.preprocess_fn, bs, shuffle=False)

    print(f"\n{'='*70}\n>>> {backbone} (img={spec.img_size}, batch={bs})\n{'='*70}", flush=True)
    t0 = time.time()
    training = {
        "epochs_phase1": 12,
        "epochs_phase2": 15,
        "lr_phase1": 1e-3,
        "lr_phase2": 1e-5,
        **DEFAULT_TRAINING.get(backbone, {}),
    }
    model, _, _ = tl.train_tl_model(
        backbone, train_ds, val_ds, class_weights, paths,
        verbose=2,
        **training,
    )
    print(f"[{backbone}] entrainement termine en {(time.time()-t0)/60:.1f} min", flush=True)

    # Evaluation sur le test set fige
    y_true = tp.get_labels("test")
    y_proba = ev.predict_keras(model, test_ds)
    y_pred = y_proba.argmax(axis=1)
    metrics = ev.evaluate_model(backbone, y_true, y_pred, y_proba, save_dir=paths["metrics"])
    ev.print_metrics_summary(metrics)
    ev.plot_confusion(metrics, save_path=paths["figures"] / f"confusion_{backbone}.png")
    ev.plot_roc_pr_ovr(y_true, y_proba, save_path=paths["figures"] / f"roc_pr_{backbone}.png", title=backbone)
    return metrics


def main(argv: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")

    backbones = argv or DEFAULT_BACKBONES
    info = setup_environment()
    paths = ensure_dirs()
    tp.build_split_csvs()
    class_weights = tp.get_class_weights()
    batch_size = 16  # robuste en CPU/16 Go

    print(f"Backbones a entrainer : {backbones}", flush=True)
    print(f"Poids de classe : {class_weights}", flush=True)

    results = {}
    for backbone in backbones:
        try:
            m = run_one(backbone, paths, class_weights, batch_size)
            if m:
                results[backbone] = m["f1_macro"]
        except Exception:
            print(f"\n!!! ECHEC sur {backbone} :", flush=True)
            traceback.print_exc()

    print(f"\n{'='*70}\nRECAP F1-macro : {results}\n{'='*70}", flush=True)
    print("Termine. Lancez le notebook 08 pour la comparaison + re-scoring hybride.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
