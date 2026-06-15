import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "COVID-19_Radiography_Dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
CNN_MODEL_PATH = REPORTS_DIR / "best_model.keras"
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_boosting.pkl"
HYBRID_METRICS_PATH = MODELS_DIR / "hybrid_boosting_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
BOOSTING_ESTIMATORS = 300
FEATURE_LAYER = "features"
RAW_SIZE = (256, 256)
CNN_SIZE = (128, 128)


def load_dataset():
    rows = []
    for category_dir in sorted(DATA_DIR.iterdir()):
        img_dir = category_dir / "images"
        if not img_dir.is_dir():
            continue
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            for image_path in sorted(img_dir.glob(pattern)):
                rows.append((image_path, category_dir.name))
    if not rows:
        raise FileNotFoundError(f"Aucune image trouvee dans {DATA_DIR}")
    return rows


def preprocess_for_cnn(image_path):
    # Reproduit le notebook 05 : resize 256 + /255, puis resize 128 + /255.
    image = Image.open(image_path).convert("L").resize(RAW_SIZE)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = cv2.resize(image_array, CNN_SIZE).astype(np.float32) / 255.0
    return image_array[..., np.newaxis]


def extract_features(feature_model, image_paths, batch_size=128):
    features = []
    total = len(image_paths)
    for start in range(0, total, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch = np.array([preprocess_for_cnn(path) for path in batch_paths], dtype=np.float32)
        batch_features = feature_model.predict(batch, verbose=0)
        features.append(batch_features.astype(np.float32))
        print(f"Features CNN : {min(start + batch_size, total)}/{total}")
    return np.vstack(features)


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele CNN introuvable : {CNN_MODEL_PATH}")

    rows = load_dataset()
    image_paths = np.array([path for path, _ in rows], dtype=object)
    labels = np.array([label for _, label in rows])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    print(f"Dataset : {len(image_paths)} images")
    print(f"Classes : {list(label_encoder.classes_)}")

    train_paths, test_paths, y_train, y_test = train_test_split(
        image_paths,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cnn_model = load_model(str(CNN_MODEL_PATH), compile=False)
    feature_model = Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer(FEATURE_LAYER).output,
    )

    print("Extraction des features train...")
    X_train_features = extract_features(feature_model, train_paths)
    print("Extraction des features test...")
    X_test_features = extract_features(feature_model, test_paths)

    print("Entrainement Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=BOOSTING_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    gb_model.fit(X_train_features, y_train)

    y_pred = gb_model.predict(X_test_features)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy test : {accuracy:.4f}")
    print(f"F1-macro test : {f1_macro:.4f}")

    test_samples = [
        {
            "image_path": str(path.relative_to(BASE_DIR)),
            "label": label_encoder.classes_[label_index],
        }
        for path, label_index in zip(test_paths, y_test)
    ]

    artifact = {
        "gb_model": gb_model,
        "classes": list(label_encoder.classes_),
        "test_samples": test_samples,
        "feature_layer": FEATURE_LAYER,
        "preprocessing": {
            "raw_size": RAW_SIZE,
            "cnn_size": CNN_SIZE,
            "double_normalization": True,
        },
        "training": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "boosting_estimators": BOOSTING_ESTIMATORS,
        },
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, HYBRID_MODEL_PATH)
    with HYBRID_METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "classification_report": report,
                "classes": list(label_encoder.classes_),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Modele hybride sauvegarde : {HYBRID_MODEL_PATH}")
    print(f"Metriques sauvegardees : {HYBRID_METRICS_PATH}")


if __name__ == "__main__":
    main()
