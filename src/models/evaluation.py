"""Evaluation unifiee des modeles sous fort desequilibre de classes.

Toutes les metriques sont calculees ici, une seule fois, pour que les modeles
classiques (RF-HOG), l'hybride (CNN+GB) et les modeles de Transfer Learning
soient strictement comparables.

Choix de la metrique (justification, cf. rapport) :
- Metrique PRINCIPALE : F1-macro. Avec 48% de Normal vs 6.4% de Viral Pneumonia,
  l'accuracy est trompeuse (un classifieur "toujours Normal" atteint 48% sans
  valeur clinique). Le macro pondere chaque classe egalement -> la performance
  sur les classes rares ne peut etre masquee.
- F1-weighted en complement (performance a la prevalence reelle).
- GARDE-FOU CLINIQUE : rappel (sensibilite) par classe, en particulier COVID.
  Un faux negatif (COVID manque) est bien plus couteux qu'un faux positif.
- Balanced accuracy (moyenne des rappels par classe).
- ROC-AUC OvR macro et surtout PR-AUC OvR (plus informatif sous desequilibre).
- Matrice de confusion (brute + normalisee par ligne).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.data.tf_pipeline import CLASS_NAMES, NUM_CLASSES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Calcule l'ensemble des metriques a partir des labels entiers.

    Parametres
    ----------
    y_true, y_pred : labels entiers (0..K-1).
    y_proba : probabilites (N, K) optionnelles, requises pour ROC-AUC et PR-AUC.
    class_names : noms des classes (defaut : CLASS_NAMES du pipeline).
    """
    class_names = class_names or CLASS_NAMES
    labels = list(range(len(class_names)))

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        y_bin = label_binarize(y_true, classes=labels)
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None
        metrics["pr_auc_ovr_macro"] = float(
            average_precision_score(y_bin, y_proba, average="macro")
        )
        metrics["pr_auc_per_class"] = {
            name: float(average_precision_score(y_bin[:, i], y_proba[:, i]))
            for i, name in enumerate(class_names)
        }

    return metrics


def predict_keras(model, dataset) -> np.ndarray:
    """Probabilites (N, K) pour un modele Keras sur un tf.data NON melange."""
    proba = model.predict(dataset, verbose=0)
    return np.asarray(proba)


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
    save_dir: Path | None = None,
) -> dict:
    """Evalue un modele et persiste les metriques en JSON.

    Retourne le dict de metriques, augmente du nom du modele.
    """
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names)
    metrics["model"] = name

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{name}_test_metrics.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def print_metrics_summary(metrics: dict) -> None:
    """Affichage lisible des metriques principales."""
    print(f"\n=== {metrics.get('model', 'modele')} ===")
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"F1-macro  (cle)    : {metrics['f1_macro']:.4f}")
    print(f"F1-weighted        : {metrics['f1_weighted']:.4f}")
    if metrics.get("roc_auc_ovr_macro") is not None:
        print(f"ROC-AUC OvR macro  : {metrics['roc_auc_ovr_macro']:.4f}")
    if metrics.get("pr_auc_ovr_macro") is not None:
        print(f"PR-AUC OvR macro   : {metrics['pr_auc_ovr_macro']:.4f}")
    print("Rappel par classe (sensibilite) :")
    for name, vals in metrics["per_class"].items():
        flag = "  <- garde-fou clinique" if name == "COVID" else ""
        print(f"  {name:<16} rappel={vals['recall']:.3f}  f1={vals['f1']:.3f}{flag}")


# --------------------------------------------------------------------------- #
# Figures (matrice de confusion, ROC, PR) sauvegardees dans reports/figures
# --------------------------------------------------------------------------- #
def plot_confusion(metrics: dict, save_path: Path | None = None, normalize: bool = True):
    """Trace la matrice de confusion (normalisee par ligne par defaut)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.array(metrics["confusion_matrix"], dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else ".0f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=True,
    )
    ax.set_xlabel("Predit")
    ax.set_ylabel("Reel")
    title = metrics.get("model", "modele")
    ax.set_title(f"Matrice de confusion {'(normalisee)' if normalize else ''} - {title}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_roc_pr_ovr(y_true, y_proba, save_path: Path | None = None, title: str = ""):
    """Trace les courbes ROC et Precision-Recall One-vs-Rest par classe."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    labels = list(range(NUM_CLASSES))
    y_bin = label_binarize(y_true, classes=labels)
    y_proba = np.asarray(y_proba)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax_roc.plot(fpr, tpr, label=name)
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ax_pr.plot(rec, prec, label=name)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set(xlabel="Taux faux positifs", ylabel="Taux vrais positifs", title=f"ROC OvR {title}")
    ax_roc.legend()
    ax_pr.set(xlabel="Rappel", ylabel="Precision", title=f"Precision-Rappel OvR {title}")
    ax_pr.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
