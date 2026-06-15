"""Transfer Learning : construction et fine-tuning de backbones ImageNet.

Backbones principaux du rapport :
  - VGG16         (baseline TL classique) -> 224, block5_conv3
  - EfficientNetB0 (compound scaling)  -> 224, top_conv
  - ResNet50       (connexions residuelles) -> 224, conv5_block3_out

Backbone optionnel conserve dans le code :
  - InceptionV3    (modules multi-echelle)   -> 299, mixed10

Protocole de fine-tuning en 2 phases :
  Phase 1 (extraction)   : backbone gele, on entraine seulement la tete (lr=1e-3).
  Phase 2 (fine-tuning)  : on degele le dernier bloc du backbone MAIS on garde
                           toutes les couches BatchNormalization gelees (critique),
                           on RECOMPILE (obligatoire apres changement de trainable)
                           avec un lr faible (1e-5).

Le modele est construit "a plat" (fonctionnel, couches exposees) pour que Grad-CAM
puisse cibler directement la derniere couche convolutive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import (
    EfficientNetB0,
    InceptionV3,
    ResNet50,
    VGG16,
)
from tensorflow.keras.applications import efficientnet, inception_v3, resnet, vgg16


def _cached_weights_or_imagenet(filename: str):
    """Utilise le cache Keras local si present, sinon laisse Keras telecharger ImageNet."""
    path = Path.home() / ".keras" / "models" / filename
    return str(path) if path.exists() else "imagenet"


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    img_size: int
    preprocess_fn: callable
    last_conv_layer: str
    builder: callable
    # Prefixe(s) de nom de couche a partir desquels degeler en phase 2.
    finetune_from_prefixes: tuple
    head_units: int = 256
    dropout_before_dense: float | None = None
    dropout_after_dense: float | None = None
    use_head_batchnorm: bool = False


BACKBONES = {
    "vgg16": BackboneSpec(
        name="vgg16",
        img_size=224,
        preprocess_fn=vgg16.preprocess_input,
        last_conv_layer="block5_conv3",
        builder=lambda shape: VGG16(
            include_top=False,
            weights=_cached_weights_or_imagenet("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"),
            input_shape=shape,
        ),
        finetune_from_prefixes=("block5",),
        head_units=128,
        dropout_before_dense=0.4,
        dropout_after_dense=0.4,
        use_head_batchnorm=True,
    ),
    "efficientnetb0": BackboneSpec(
        name="efficientnetb0",
        img_size=224,
        preprocess_fn=efficientnet.preprocess_input,
        last_conv_layer="top_conv",
        builder=lambda shape: EfficientNetB0(
            include_top=False,
            weights=_cached_weights_or_imagenet("efficientnetb0_notop.h5"),
            input_shape=shape,
        ),
        finetune_from_prefixes=("block6", "block7", "top"),
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        img_size=224,
        preprocess_fn=resnet.preprocess_input,
        last_conv_layer="conv5_block3_out",
        builder=lambda shape: ResNet50(
            include_top=False,
            weights=_cached_weights_or_imagenet("resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"),
            input_shape=shape,
        ),
        finetune_from_prefixes=("conv5",),
    ),
    "inceptionv3": BackboneSpec(
        name="inceptionv3",
        img_size=299,
        preprocess_fn=inception_v3.preprocess_input,
        last_conv_layer="mixed10",
        builder=lambda shape: InceptionV3(include_top=False, weights="imagenet", input_shape=shape),
        finetune_from_prefixes=("mixed8", "mixed9", "mixed10"),
    ),
}


def get_spec(backbone: str) -> BackboneSpec:
    key = backbone.lower()
    if key not in BACKBONES:
        raise ValueError(f"Backbone inconnu : {backbone}. Choix : {list(BACKBONES)}")
    return BACKBONES[key]


def build_tl_model(backbone: str, num_classes: int = 4, dropout: float = 0.3):
    """Construit un modele de Transfer Learning "a plat".

    Retourne (model, base_model). Le base_model permet de geler/degeler.
    La sortie softmax est forcee en float32 (compatibilite mixed precision).
    """
    spec = get_spec(backbone)
    shape = (spec.img_size, spec.img_size, 3)

    base_model = spec.builder(shape)
    base_model.trainable = False

    inputs = layers.Input(shape=shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if spec.use_head_batchnorm:
        x = layers.BatchNormalization(name="bn_head")(x)
    if spec.dropout_before_dense is not None:
        x = layers.Dropout(spec.dropout_before_dense, name="dropout_1")(x)
    x = layers.Dense(spec.head_units, activation="relu", name=f"dense_{spec.head_units}")(x)
    x = layers.Dropout(spec.dropout_after_dense if spec.dropout_after_dense is not None else dropout, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"{spec.name}_tl")
    return model, base_model


def _freeze_batchnorm(model_or_layer) -> None:
    """Gele recursivement toutes les couches BatchNormalization.

    Critique en phase 2 : degeler les BN avec de petits batches detruit les
    statistiques pre-entrainees et fait diverger le fine-tuning.
    """
    for layer in getattr(model_or_layer, "layers", []):
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        elif hasattr(layer, "layers"):
            _freeze_batchnorm(layer)


def unfreeze_for_finetuning(base_model, spec: BackboneSpec) -> int:
    """Degele le dernier bloc du backbone, en gardant les BatchNorm gelees.

    Retourne le nombre de couches entrainables apres operation.
    """
    base_model.trainable = True
    prefixes = spec.finetune_from_prefixes
    for layer in base_model.layers:
        # On ne degele que les couches du/des dernier(s) bloc(s).
        layer.trainable = any(layer.name.startswith(p) for p in prefixes)
    # Quoi qu'il arrive, on regele toutes les BatchNorm.
    _freeze_batchnorm(base_model)
    return sum(1 for l in base_model.layers if l.trainable)


def get_callbacks(backbone: str, phase: int, paths: dict, patience: int = 5):
    """Callbacks standards : checkpoint, early stopping, reduce LR, CSV logger."""
    from tensorflow.keras.callbacks import (
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    ckpt = paths["models_transfer"] / f"{backbone}_best.keras"
    csv_log = paths["history"] / f"{backbone}_phase{phase}.csv"
    return [
        ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1),
        CSVLogger(str(csv_log)),
    ]


def train_tl_model(
    backbone: str,
    train_ds,
    val_ds,
    class_weights: dict,
    paths: dict,
    epochs_phase1: int = 12,
    epochs_phase2: int = 15,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-5,
    verbose: int = 1,
):
    """Entraine un modele TL selon le protocole 2 phases.

    `verbose` : 1 = barre de progression (notebook), 2 = une ligne par epoque
    (recommande pour un log d'execution headless en arriere-plan).

    Retourne (model, history_phase1, history_phase2).
    """
    spec = get_spec(backbone)
    model, base_model = build_tl_model(backbone)

    # --- Phase 1 : extraction de features (backbone gele) ---
    print(f"\n[{backbone}] Phase 1 - extraction (backbone gele), lr={lr_phase1}", flush=True)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_phase1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        class_weight=class_weights,
        callbacks=get_callbacks(backbone, phase=1, paths=paths),
        verbose=verbose,
    )

    # --- Phase 2 : fine-tuning (dernier bloc degele, BN gelees, recompile) ---
    n_trainable = unfreeze_for_finetuning(base_model, spec)
    print(f"\n[{backbone}] Phase 2 - fine-tuning ({n_trainable} couches degelees), lr={lr_phase2}", flush=True)
    model.compile(  # RECOMPILE obligatoire apres changement de trainable
        optimizer=optimizers.Adam(learning_rate=lr_phase2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        class_weight=class_weights,
        callbacks=get_callbacks(backbone, phase=2, paths=paths),
        verbose=verbose,
    )

    return model, history1, history2
