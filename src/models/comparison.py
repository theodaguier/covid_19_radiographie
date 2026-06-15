"""Comparaison des modeles sur le MEME test set fige.

Assemble une table maitresse a partir des JSON de metriques produits par
src.models.evaluation, trace les courbes d'apprentissage (diagnostic du
sur-apprentissage) et les matrices de confusion cote a cote.

Important : les modeles classiques (RF-HOG) et l'hybride doivent etre re-scores
sur reports/splits/test.csv AVANT comparaison (cf. notebook 08), car leurs
metriques d'origine viennent de splits differents et ne sont pas comparables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.tf_pipeline import CLASS_NAMES
from src.utils.env import get_paths


def load_all_metrics(metrics_dir: Path | None = None) -> list[dict]:
    """Charge tous les *_test_metrics.json du dossier reports/metrics."""
    metrics_dir = metrics_dir or get_paths()["metrics"]
    out = []
    for path in sorted(Path(metrics_dir).glob("*_test_metrics.json")):
        with path.open(encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def build_comparison_table(metrics_list: list[dict]) -> pd.DataFrame:
    """Table maitresse : une ligne par modele, colonnes = metriques cles."""
    rows = []
    for m in metrics_list:
        row = {
            "modele": m.get("model", "?"),
            "accuracy": m.get("accuracy"),
            "balanced_acc": m.get("balanced_accuracy"),
            "f1_macro": m.get("f1_macro"),
            "f1_weighted": m.get("f1_weighted"),
            "roc_auc": m.get("roc_auc_ovr_macro"),
            "pr_auc": m.get("pr_auc_ovr_macro"),
        }
        per_class = m.get("per_class", {})
        if "COVID" in per_class:
            row["rappel_COVID"] = per_class["COVID"]["recall"]
        if "Viral Pneumonia" in per_class:
            row["rappel_Viral"] = per_class["Viral Pneumonia"]["recall"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if "f1_macro" in df.columns:
        df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return df


def save_comparison_table(df: pd.DataFrame, save_path: Path | None = None) -> Path:
    save_path = Path(save_path) if save_path else get_paths()["metrics"] / "comparison_table.csv"
    df.to_csv(save_path, index=False)
    return save_path


def plot_learning_curves(backbone: str, history_dir: Path | None = None, save_path: Path | None = None):
    """Trace loss et accuracy train vs val (phases 1+2) pour diagnostiquer
    le sur-apprentissage (ecart train-val). Lit les CSV CSVLogger."""
    import matplotlib.pyplot as plt

    history_dir = history_dir or get_paths()["history"]
    frames = []
    offset = 0
    phase_starts = []
    for phase in (1, 2):
        csv = Path(history_dir) / f"{backbone}_phase{phase}.csv"
        if not csv.exists():
            continue
        h = pd.read_csv(csv)
        h["epoch_global"] = h["epoch"] + offset
        phase_starts.append(offset)
        offset += len(h)
        frames.append(h)
    if not frames:
        raise FileNotFoundError(f"Aucun historique pour {backbone} dans {history_dir}")
    hist = pd.concat(frames, ignore_index=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    ax_loss.plot(hist["epoch_global"], hist["loss"], label="train")
    ax_loss.plot(hist["epoch_global"], hist["val_loss"], label="val")
    ax_loss.set(xlabel="Epoque", ylabel="Loss", title=f"Loss - {backbone}")
    ax_acc.plot(hist["epoch_global"], hist["accuracy"], label="train")
    ax_acc.plot(hist["epoch_global"], hist["val_accuracy"], label="val")
    ax_acc.set(xlabel="Epoque", ylabel="Accuracy", title=f"Accuracy - {backbone}")
    for ax in (ax_loss, ax_acc):
        for ps in phase_starts[1:]:
            ax.axvline(ps - 0.5, color="grey", linestyle="--", alpha=0.6)
        ax.legend()
    fig.suptitle(f"Courbes d'apprentissage {backbone} (ligne pointillee = debut phase 2)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_confusion_grid(metrics_list: list[dict], save_path: Path | None = None):
    """Matrices de confusion normalisees cote a cote pour tous les modeles."""
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    n = len(metrics_list)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, m in zip(axes, metrics_list):
        cm = np.array(m["confusion_matrix"], dtype=float)
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(m.get("model", "?"))
        ax.set_xlabel("Predit"); ax.set_ylabel("Reel")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
