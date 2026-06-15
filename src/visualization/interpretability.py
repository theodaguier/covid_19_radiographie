"""Interpretabilite des modeles de Transfer Learning : Grad-CAM + SHAP.

Grad-CAM (Gradient-weighted Class Activation Mapping) :
  Met en evidence les regions de l'image qui contribuent le plus a la prediction
  d'une classe. Applique aux 3 modeles, par classe et sur des cas mal classes.

SHAP (GradientExplainer) :
  Attributions par pixel basees sur les gradients esperes. Applique uniquement au
  meilleur modele (selon F1-macro) car couteux sur images. Usage qualitatif.

Le modele TL etant construit "a plat" (cf. transfer_learning.build_tl_model), la
derniere couche convolutive du backbone est accessible via le sous-modele
`<backbone>_backbone`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.tf_pipeline import CLASS_NAMES


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def _find_backbone_and_layer(model, last_conv_layer: str | None):
    """Localise le sous-modele backbone et sa derniere couche conv.

    Retourne (grad_model_input, conv_output_tensor) en reconstruisant un modele
    qui expose a la fois la sortie conv et la sortie finale.
    """
    # Le backbone est imbrique comme un sous-modele Functional (il possede .layers).
    backbone = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(getattr(layer, "layers", [])) > 1:
            backbone = layer
            break
    if backbone is None:
        raise ValueError("Backbone (sous-modele) introuvable dans le modele.")

    if last_conv_layer is None:
        # Derniere couche a sortie 4D dans le backbone.
        for layer in reversed(backbone.layers):
            if len(layer.output.shape) == 4:
                last_conv_layer = layer.name
                break
    conv_layer = backbone.get_layer(last_conv_layer)
    return backbone, conv_layer


def make_gradcam_heatmap(img_array: np.ndarray, model, last_conv_layer: str | None = None,
                         pred_index: int | None = None) -> tuple[np.ndarray, int]:
    """Calcule la heatmap Grad-CAM pour une image (deja preprocessee).

    img_array : tenseur (1, H, W, 3) deja passe par preprocess_input.
    Retourne (heatmap [0,1] de taille HxW conv, indice de classe predite).

    On scinde le modele en deux :
      - conv_model       : input -> activation de la derniere couche conv du backbone
      - classifier_model : activation conv -> prediction finale (tete)
    puis on calcule le gradient de la classe cible vis-a-vis de l'activation conv.
    """
    backbone, conv_layer = _find_backbone_and_layer(model, last_conv_layer)

    # 1) input du modele -> sortie conv du backbone
    conv_model = tf.keras.Model(backbone.input, conv_layer.output)

    # 2) sortie conv -> prediction : on rejoue les couches situees apres le backbone
    classifier_input = tf.keras.Input(shape=conv_layer.output.shape[1:])
    x = classifier_input
    started = False
    for layer in model.layers:
        if layer is backbone:
            started = True
            continue
        if started:
            x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-9)
    return heatmap.numpy(), pred_index


def overlay_heatmap(orig_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superpose la heatmap (colormap jet) sur l'image originale [0,1] ou [0,255]."""
    import matplotlib.cm as cm
    from PIL import Image

    if orig_img.max() > 1.5:
        orig = orig_img.astype(np.uint8)
    else:
        orig = (orig_img * 255).astype(np.uint8)
    if orig.ndim == 2:
        orig = np.stack([orig] * 3, axis=-1)

    h, w = orig.shape[:2]
    hm = np.uint8(255 * heatmap)
    hm = np.array(Image.fromarray(hm).resize((w, h)))
    jet = cm.get_cmap("jet")(hm)[..., :3]
    jet = np.uint8(jet * 255)
    return np.uint8(jet * alpha + orig * (1 - alpha))


def gradcam_panel_for_image(model, raw_img_rgb, preprocess_fn, last_conv_layer, save_path=None,
                            true_label=None):
    """Trace original | heatmap | overlay pour une image RGB brute [0,255]."""
    import matplotlib.pyplot as plt

    batch = preprocess_fn(np.expand_dims(raw_img_rgb.astype("float32"), 0))
    heatmap, pred_idx = make_gradcam_heatmap(batch, model, last_conv_layer)
    overlay = overlay_heatmap(raw_img_rgb, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw_img_rgb.astype("uint8")); axes[0].set_title("Originale")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Heatmap Grad-CAM")
    axes[2].imshow(overlay); axes[2].set_title("Superposition")
    pred_name = CLASS_NAMES[pred_idx]
    sub = f"Predit : {pred_name}"
    if true_label is not None:
        sub += f"  |  Reel : {true_label}"
    fig.suptitle(sub)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig, pred_name


def load_raw_rgb(filepath: str, img_size: int) -> np.ndarray:
    """Charge un PNG niveaux de gris -> RGB [0,255] redimensionne (non preprocesse)."""
    raw = tf.io.read_file(filepath)
    img = tf.image.decode_png(raw, channels=1)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize(img, [img_size, img_size])
    return img.numpy()


# --------------------------------------------------------------------------- #
# SHAP (GradientExplainer) - meilleur modele uniquement
# --------------------------------------------------------------------------- #
def shap_gradient_explain(model, background_batch: np.ndarray, sample_batch: np.ndarray,
                          save_path: Path | None = None):
    """Calcule et trace les attributions SHAP pour quelques images.

    background_batch : ~50-100 images d'arriere-plan (deja preprocessees).
    sample_batch     : ~8-12 images a expliquer (deja preprocessees).

    GradientExplainer est choisi plutot que DeepExplainer car robuste sur les
    modeles fonctionnels Keras modernes (EfficientNet/Inception).
    """
    import shap

    explainer = shap.GradientExplainer(model, background_batch)
    shap_values = explainer.shap_values(sample_batch)
    shap.image_plot(shap_values, sample_batch, show=save_path is None)
    if save_path:
        import matplotlib.pyplot as plt

        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return shap_values
