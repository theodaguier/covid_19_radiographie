"""Experience anti-biais : re-entrainement Transfer Learning sur images MASQUEES.

On applique le masque pulmonaire (pixels hors poumons a 0) sur train/val/test, puis on
re-entraine le backbone choisi selon le meme protocole 2 phases. Objectif : forcer le
modele a se baser uniquement sur les champs pulmonaires et verifier :
  (a) si les metriques restent acceptables ;
  (b) si l'attention Grad-CAM se concentre davantage dans les poumons.

Artefacts sauvegardes sous le nom `<backbone>_masked` pour ne pas ecraser les modeles
initiaux non masques.

Usage :
    PYTHONPATH=. python -m src.models.train_masked resnet50
    PYTHONPATH=. python -m src.models.train_masked efficientnetb0 --gradcam-n 12
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tensorflow.keras import optimizers

from src.utils.env import setup_environment, ensure_dirs
from src.data import tf_pipeline as tp
from src.models import transfer_learning as tl
from src.models import evaluation as ev
from src.visualization import interpretability as interp

DEFAULT_BACKBONE = "resnet50"


def _mask01(fp, size):
    m = np.array(Image.open(fp.replace("/images/", "/masks/")).convert("L").resize((size, size)))
    return (m > 127).astype(np.float32)


def quantify(model, spec, n=12, split="test", class_name="COVID"):
    """% de chaleur Grad-CAM dans les poumons sur n cas d'une classe donnee.

    Le score est calcule sur l'input masque, c'est-a-dire dans les memes conditions que
    l'entrainement correctif.
    """
    covid = tp.load_split_df(split)
    covid = covid[covid["class"] == class_name].head(n)
    fr, area = [], []
    for fp in covid["filepath"]:
        mask = _mask01(fp, spec.img_size)
        raw = interp.load_raw_rgb(fp, spec.img_size).astype("float32")
        raw_masked = raw * mask[..., None]                       # input masque (comme l'entrainement)
        b = spec.preprocess_fn(np.expand_dims(raw_masked, 0))
        hm, _ = interp.make_gradcam_heatmap(b, model, spec.last_conv_layer)
        hmu = np.array(Image.fromarray(np.uint8(255 * hm)).resize((spec.img_size, spec.img_size))).astype("float32") / 255.0
        fr.append((hmu * mask).sum() / (hmu.sum() + 1e-9))
        area.append(mask.mean())
    return float(np.mean(fr)), float(np.mean(area))


def save_gradcam_examples(model, spec, paths: dict, tag: str, n: int = 4, class_name: str = "COVID") -> None:
    """Sauvegarde quelques panneaux Grad-CAM du modele masque."""
    import matplotlib.pyplot as plt

    df = tp.load_split_df("test")
    df = df[df["class"] == class_name].head(n)
    out_dir = Path(paths["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, fp in enumerate(df["filepath"]):
        mask = _mask01(fp, spec.img_size)
        raw = interp.load_raw_rgb(fp, spec.img_size).astype("float32")
        raw_masked = raw * mask[..., None]
        batch = spec.preprocess_fn(np.expand_dims(raw_masked, 0))
        heatmap, pred_idx = interp.make_gradcam_heatmap(batch, model, spec.last_conv_layer)
        overlay = interp.overlay_heatmap(raw, heatmap)

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(raw.astype("uint8"))
        axes[0].set_title("Originale")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Masque poumons")
        axes[2].imshow(raw_masked.astype("uint8"))
        axes[2].set_title("Input masque")
        axes[3].imshow(overlay)
        axes[3].set_title(f"Grad-CAM - pred. {tp.CLASS_NAMES[pred_idx]}")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"gradcam_{tag}_{class_name}_{idx}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def run(backbone: str = DEFAULT_BACKBONE, batch_size: int = 16, gradcam_n: int = 12, verbose: int = 2) -> dict:
    """Lance l'experience masquee pour un backbone et retourne les metriques."""
    setup_environment()
    paths = ensure_dirs()
    tp.build_split_csvs()
    spec = tl.get_spec(backbone)
    tag = f"{backbone}_masked"
    bs = batch_size
    if spec.img_size == 299:
        bs = max(8, batch_size // 2)
    cw = tp.get_class_weights()

    tr = tp.make_dataset("train", spec.img_size, spec.preprocess_fn, bs, augment=True, apply_mask=True)
    va = tp.make_dataset("val", spec.img_size, spec.preprocess_fn, bs, apply_mask=True)
    te = tp.make_dataset("test", spec.img_size, spec.preprocess_fn, bs, shuffle=False, apply_mask=True)

    model, base = tl.build_tl_model(backbone)
    print(f"\n[{tag}] Phase 1 (gele), lr=1e-3", flush=True)
    model.compile(optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    t = time.time()
    model.fit(tr, validation_data=va, epochs=12, class_weight=cw,
              callbacks=tl.get_callbacks(tag, 1, paths), verbose=verbose)

    n_tr = tl.unfreeze_for_finetuning(base, spec)
    print(f"\n[{tag}] Phase 2 (fine-tuning, {n_tr} couches), lr=1e-5", flush=True)
    model.compile(optimizers.Adam(1e-5), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(tr, validation_data=va, epochs=15, class_weight=cw,
              callbacks=tl.get_callbacks(tag, 2, paths), verbose=verbose)
    print(f"[{tag}] entrainement: {(time.time()-t)/60:.1f} min", flush=True)

    # Evaluation sur le test MASQUE
    y_true = tp.get_labels("test")
    y_proba = ev.predict_keras(model, te)
    y_pred = y_proba.argmax(1)
    m = ev.evaluate_model(tag, y_true, y_pred, y_proba, save_dir=paths["metrics"])
    ev.print_metrics_summary(m)
    ev.plot_confusion(m, save_path=paths["figures"] / f"confusion_{tag}.png")
    ev.plot_roc_pr_ovr(y_true, y_proba, save_path=paths["figures"] / f"roc_pr_{tag}.png", title=tag)

    # Re-quantification du biais (Grad-CAM dans les poumons)
    frac, area = quantify(model, spec, n=gradcam_n)
    save_gradcam_examples(model, spec, paths, tag, n=min(4, gradcam_n))

    bias_summary = {
        "model": tag,
        "gradcam_in_lungs": frac,
        "lung_area_reference": area,
        "n_gradcam_cases": gradcam_n,
        "class_analyzed": "COVID",
        "interpretation": (
            "Plus le score Grad-CAM dans les poumons depasse la surface pulmonaire "
            "moyenne, plus l'attention est compatible avec une decision fondee sur "
            "les champs pulmonaires."
        ),
    }
    bias_path = Path(paths["metrics"]) / f"{tag}_bias_summary.json"
    with bias_path.open("w", encoding="utf-8") as f:
        json.dump(bias_summary, f, indent=2, ensure_ascii=False)

    print("\n=== BIAIS (apres masquage) ===", flush=True)
    print(f"Grad-CAM dans poumons : {100*frac:.0f}%  (hasard ~{100*area:.0f}%)", flush=True)
    baseline_path = Path(paths["metrics"]) / f"{backbone}_test_metrics.json"
    if baseline_path.exists():
        with baseline_path.open(encoding="utf-8") as f:
            baseline = json.load(f)
        print(
            f"F1-macro non masque = {baseline['f1_macro']:.3f} | "
            f"F1-macro masque = {m['f1_macro']:.3f}",
            flush=True,
        )
    else:
        print(f"F1-macro masque = {m['f1_macro']:.3f} (baseline non masque introuvable)", flush=True)
    print("RECAP_MASKED f1=%.3f recall_covid=%.3f gradcam_in_lungs=%.2f" %
          (m["f1_macro"], m["per_class"]["COVID"]["recall"], frac), flush=True)
    return m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-entraine un backbone TL sur images masquees.")
    parser.add_argument("backbone", nargs="?", default=DEFAULT_BACKBONE, choices=sorted(tl.BACKBONES))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradcam-n", type=int, default=12)
    parser.add_argument("--verbose", type=int, default=2, choices=(0, 1, 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.backbone, batch_size=args.batch_size, gradcam_n=args.gradcam_n, verbose=args.verbose)


if __name__ == "__main__":
    main()
