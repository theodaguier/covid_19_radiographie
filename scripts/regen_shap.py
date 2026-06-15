"""Regenere une figure SHAP lisible pour EfficientNetB0.

Bug de l'ancienne figure : les images passees a shap.image_plot etaient pretraitees
(hors [0,1]) -> rendu noir/blanc. Correctif : on calcule les valeurs SHAP sur les
images pretraitees (ce que le modele attend) mais on AFFICHE les images en [0,1].
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import shap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "transfer", "efficientnetb0_best.keras")
TEST = os.path.join(ROOT, "reports", "splits", "test.csv")
TRAIN = os.path.join(ROOT, "reports", "splits", "train.csv")
OUT = os.path.join(ROOT, "reports", "figures", "shap_efficientnetb0.png")
SIZE = 224
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]


def load_rgb(fp):
    img = Image.open(fp).convert("L").resize((SIZE, SIZE))
    arr = np.array(img, dtype="float32")
    return np.stack([arr, arr, arr], axis=-1)  # (H,W,3) en [0,255]


print("Chargement du modele...", flush=True)
model = tf.keras.models.load_model(MODEL)

test_df = pd.read_csv(TEST)
train_df = pd.read_csv(TRAIN)

# 1 image de test par classe (4 images, figure compacte et lisible)
sample_paths = []
for c in CLASSES:
    sub = test_df[test_df["class"] == c]
    sample_paths.append(sub.iloc[0]["filepath"])
samples_raw = np.stack([load_rgb(p) for p in sample_paths])          # [0,255]
samples_pp = preprocess_input(samples_raw.copy())                    # entree modele

# fond : 24 images d'entrainement
bg_paths = train_df.sample(n=24, random_state=42)["filepath"].tolist()
background = preprocess_input(np.stack([load_rgb(p) for p in bg_paths]))

print("Calcul SHAP (GradientExplainer)...", flush=True)
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(samples_pp)

# Mise en forme pour image_plot : liste (une entree par classe) d'arrays (N,H,W,3)
if isinstance(shap_values, np.ndarray):
    # forme (N,H,W,3,n_classes) -> liste de n_classes
    shap_values = [shap_values[..., k] for k in range(shap_values.shape[-1])]

display_imgs = samples_raw / 255.0                                   # [0,1] pour l'affichage
labels = np.array([CLASSES] * len(sample_paths))                    # une ligne de labels par image

print("Trace...", flush=True)
shap.image_plot(shap_values, display_imgs, labels=labels, show=False)
fig = plt.gcf()
fig.suptitle("Valeurs SHAP par classe - EfficientNetB0", y=1.02, fontsize=11)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"OK -> {OUT}", flush=True)
