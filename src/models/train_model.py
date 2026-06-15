import os
import numpy as np
import pandas as pd
import joblib
import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog


DATA_DIR = os.path.join(os.path.dirname(__file__), "../..", "data", "COVID-19_Radiography_Dataset")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../..", "models")


def load_dataset():
    data = []
    for category in os.listdir(DATA_DIR):
        folder_images = os.path.join(DATA_DIR, category, "images")
        if os.path.isdir(folder_images):
            for file in os.listdir(folder_images):
                if file.endswith(".png") or file.endswith(".jpg"):
                    data.append({
                        "image_path": os.path.join(folder_images, file),
                        "label": category
                    })
    return pd.DataFrame(data)


def preprocess_image(path, target_size=(64, 64)):
    img = Image.open(path).convert("L")
    img = img.resize(target_size)
    return np.array(img, dtype=np.float32) / 255.0


def extract_hog_features(images, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    features = []
    for img in images:
        feat = hog(img, orientations=9, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, feature_vector=True)
        features.append(feat)
    return np.array(features)


def train_rf_hog():
    print("Chargement du dataset...")
    df = load_dataset()
    print(f"Dataset: {len(df)} images")

    print("Preprocessing des images...")
    X_list = [preprocess_image(p) for p in df["image_path"]]
    X_hog = extract_hog_features(X_list)
    print(f"HOG features shape: {X_hog.shape}")

    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    print(f"Classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entrainement du Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_hog.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_mapping.pkl"))
    print(f"\nModeles sauvegardes dans {MODELS_DIR}")


if __name__ == "__main__":
    train_rf_hog()