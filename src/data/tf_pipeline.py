"""Pipeline de donnees tf.data pour le Transfer Learning.

Points cles de conception :
- Chargement *base fichiers* (streaming depuis le disque) et non chargement
  NumPy massif en RAM : 21 165 images en 224x224x3 float32 ~ 12 Go, inteneable.
- Split stratifie 70/15/15 FIGE et persiste dans reports/splits/*.csv. Tous les
  modeles (classiques, hybride, TL) sont evalues sur le MEME test set -> seule
  facon d'avoir une comparaison defendable.
- Conversion niveaux de gris -> 3 canaux RGB (replication) car les poids ImageNet
  attendent 3 canaux.
- Le sous-dossier `masks/` de chaque classe est EXCLU (on ne garde que `images/`).
- Augmentation medicalement prudente sur le train uniquement (pas de flip vertical).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.utils.env import get_paths

AUTOTUNE = tf.data.AUTOTUNE

# Ordre canonique des classes (indices fixes pour tout le projet).
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


# --------------------------------------------------------------------------- #
# Construction et persistance du split stratifie
# --------------------------------------------------------------------------- #
def _list_all_images() -> pd.DataFrame:
    """Liste toutes les images (en excluant masks/) avec leur classe et label."""
    paths = get_paths()
    data_dir = paths["data"]
    rows = []
    for class_name in CLASS_NAMES:
        img_dir = data_dir / class_name / "images"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Dossier introuvable : {img_dir}")
        for pattern in ("*.png", "*.PNG", "*.jpg", "*.jpeg"):
            for img_path in sorted(img_dir.glob(pattern)):
                rows.append(
                    {
                        "filepath": str(img_path),
                        "class": class_name,
                        "label": CLASS_TO_IDX[class_name],
                    }
                )
    if not rows:
        raise FileNotFoundError(f"Aucune image trouvee sous {data_dir}")
    return pd.DataFrame(rows)


def build_split_csvs(seed: int = SEED, force: bool = False) -> dict:
    """Cree (si absents) les CSV de split stratifie 70/15/15.

    Le split est realise sur les CHEMINS de fichiers (pas les pixels) : aucune
    image ne peut se retrouver dans deux splits -> pas de fuite de donnees.

    Retourne un dict {split: DataFrame}.
    """
    paths = get_paths()
    split_dir = paths["splits"]
    split_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = {s: split_dir / f"{s}.csv" for s in ("train", "val", "test")}

    if not force and all(p.exists() for p in csv_paths.values()):
        return {s: pd.read_csv(p) for s, p in csv_paths.items()}

    df = _list_all_images()

    # train (70%) vs reste (30%)
    train_df, rest_df = train_test_split(
        df,
        test_size=VAL_FRACTION + TEST_FRACTION,
        stratify=df["label"],
        random_state=seed,
    )
    # reste -> val (15%) / test (15%) -> 50/50 du reste
    rel_test = TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=rel_test,
        stratify=rest_df["label"],
        random_state=seed,
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, frame in splits.items():
        frame.reset_index(drop=True).to_csv(csv_paths[name], index=False)
    return splits


def load_split_df(split: str) -> pd.DataFrame:
    """Charge un CSV de split (le cree si necessaire)."""
    paths = get_paths()
    csv_path = paths["splits"] / f"{split}.csv"
    if not csv_path.exists():
        build_split_csvs()
    return pd.read_csv(csv_path)


# --------------------------------------------------------------------------- #
# Construction des tf.data.Dataset
# --------------------------------------------------------------------------- #
def _build_augmenter() -> tf.keras.Sequential:
    """Augmentation medicalement prudente (train uniquement).

    Pas de flip vertical ni de transformations agressives : l'anatomie et les
    marqueurs gauche/droite sont informatifs en radiologie.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.03),       # ~ +/- 10 degres
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomFlip("horizontal"),
        ],
        name="augmenter",
    )


def _decode_image(filepath: tf.Tensor, img_size: int) -> tf.Tensor:
    """Lit un PNG niveaux de gris -> RGB float32 [0,255], redimensionne."""
    raw = tf.io.read_file(filepath)
    img = tf.image.decode_png(raw, channels=1)          # 1 canal
    img = tf.image.grayscale_to_rgb(img)                # replique -> 3 canaux
    img = tf.image.resize(img, [img_size, img_size])
    return tf.cast(img, tf.float32)


def _apply_lung_mask(img, filepath, img_size):
    """Met a zero tous les pixels HORS poumons via le masque fourni par le dataset.

    Le masque est a `<classe>/masks/<nom>.png` (binaire, poumons en blanc). On force
    ainsi le modele a se baser uniquement sur la region pulmonaire (anti-biais).
    """
    mask_path = tf.strings.regex_replace(filepath, "/images/", "/masks/")
    m = tf.image.decode_png(tf.io.read_file(mask_path), channels=1)
    m = tf.image.resize(m, [img_size, img_size])           # float [0,255]
    m = tf.cast(m > 127.0, tf.float32)                     # binaire (h,w,1)
    return img * m                                         # broadcast sur 3 canaux


def make_dataset(
    split: str,
    img_size: int,
    preprocess_fn,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool | None = None,
    apply_mask: bool = False,
) -> tf.data.Dataset:
    """Construit un tf.data.Dataset pret a etre passe a model.fit / predict.

    Parametres
    ----------
    split : "train" | "val" | "test"
    img_size : taille cible (224 pour ResNet/EfficientNet, 299 pour Inception)
    preprocess_fn : la fonction `preprocess_input` du backbone keras.applications
        (appliquee apres la mise a l'echelle RGB [0,255]).
    augment : applique l'augmentation (uniquement pertinent sur le train).
    shuffle : par defaut True pour le train, False sinon.

    Les labels sont encodes en one-hot (categorical_crossentropy).
    """
    df = load_split_df(split)
    if shuffle is None:
        shuffle = split == "train"

    filepaths = df["filepath"].to_numpy()
    labels = df["label"].to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)

    def _load(filepath, label):
        img = _decode_image(filepath, img_size)
        if apply_mask:
            img = _apply_lung_mask(img, filepath, img_size)
        onehot = tf.one_hot(label, NUM_CLASSES)
        return img, onehot

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)

    if augment:
        augmenter = _build_augmenter()
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # preprocess_input du backbone (attend du [0,255] RGB)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def get_labels(split: str) -> np.ndarray:
    """Labels entiers (ordre du CSV, non melange) pour l'evaluation."""
    return load_split_df(split)["label"].to_numpy()


def get_class_weights() -> dict:
    """Poids de classe 'balanced' calcules sur le TRAIN uniquement (anti-fuite)."""
    y_train = get_labels("train")
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def split_summary() -> pd.DataFrame:
    """Tableau recapitulatif des effectifs par classe et par split."""
    frames = []
    for split in ("train", "val", "test"):
        counts = load_split_df(split)["class"].value_counts()
        frames.append(counts.rename(split))
    summary = pd.concat(frames, axis=1).fillna(0).astype(int)
    summary = summary.reindex(CLASS_NAMES)
    summary["total"] = summary.sum(axis=1)
    return summary
