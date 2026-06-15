"""
Script pour générer les figures manquantes dans reports/
à partir des notebooks et des données.

Lancez ce script AVANT de démarrer Streamlit si vous voulez
avoir toutes les figures. Certaines sections de l'app fonctionnent
sans ces figures (description textuelle).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path

SAVE_DIR = Path(__file__).resolve().parent.parent / "reports"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "COVID-19_Radiography_Dataset"

os.makedirs(SAVE_DIR, exist_ok=True)

print("Generation des figures complementaires...")

CLASSES = ['Normal', 'Lung_Opacity', 'COVID', 'Viral Pneumonia']

try:
    sample_size = 125
    np.random.seed(42)
    data = []
    for cls in CLASSES:
        img_dir = DATA_DIR / cls / "images"
        if not img_dir.exists():
            print(f"  [SKIP] {img_dir} non trouve")
            continue
        imgs = list(img_dir.glob("*.png"))
        chosen = np.random.choice(imgs, min(sample_size, len(imgs)), replace=False)
        for p in chosen:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
            data.append({
                'classe': cls,
                'moyenne': img.mean(),
                'ecart_type': img.std(),
                'contraste': img.std(),
                'entropie': -np.sum(img * np.log(img + 1e-8)) / (256 * 256),
                'gradient': cv2.Laplacian(img, cv2.CV_64F).std()
            })
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'Normal': '#2ecc71', 'Lung_Opacity': '#f39c12', 'COVID': '#e74c3c', 'Viral Pneumonia': '#9b59b6'}
    for cls in CLASSES:
        d = df[df['classe'] == cls]
        axes[0].scatter(d['moyenne'], d['ecart_type'], label=cls, alpha=0.5, c=colors.get(cls, '#95a5a6'))
    axes[0].set_xlabel("Intensite moyenne")
    axes[0].set_ylabel("Ecart-type (contraste)")
    axes[0].legend()
    axes[0].set_title("Intensite moyenne vs Ecart-type")
    axes[0].grid(True, alpha=0.3)

    sns.boxplot(data=df, x='classe', y='contraste', ax=axes[1], palette=colors)
    axes[1].set_title("Distribution du contraste par classe")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "intensite_vs_ecart_type.png", dpi=150)
    plt.close()
    print("  [OK] intensite_vs_ecart_type.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in CLASSES:
        d = df[df['classe'] == cls]
        ax.hist(d['entropie'], bins=30, alpha=0.5, label=cls, color=colors.get(cls, '#95a5a6'))
    ax.set_xlabel("Entropie")
    ax.set_ylabel("Frequence")
    ax.set_title("Distribution de l'entropie par classe")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "distribution_entropie.png", dpi=150)
    plt.close()
    print("  [OK] distribution_entropie.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in CLASSES:
        d = df[df['classe'] == cls]
        ax.hist(d['gradient'], bins=30, alpha=0.5, label=cls, color=colors.get(cls, '#95a5a6'))
    ax.set_xlabel("Gradient (Laplacien)")
    ax.set_ylabel("Frequence")
    ax.set_title("Distribution du gradient des contours par classe")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "distribution_gradient.png", dpi=150)
    plt.close()
    print("  [OK] distribution_gradient.png")

    feat_cols = ['moyenne', 'ecart_type', 'contraste', 'entropie', 'gradient']
    corr = df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title("Matrice de correlation des features")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "matrice_correlation_features.png", dpi=150)
    plt.close()
    print("  [OK] matrice_correlation_features.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    covid_means = df[df['classe'] == 'COVID']['moyenne']
    normal_means = df[df['classe'] == 'Normal']['moyenne']
    sns.kdeplot(covid_means, label="COVID", fill=True, color='#e74c3c', ax=ax)
    sns.kdeplot(normal_means, label="Normal", fill=True, color='#2ecc71', ax=ax)
    ax.set_xlabel("Moyenne d'intensite")
    ax.set_ylabel("Densite")
    ax.set_title("Comparaison des moyennes d'intensite : COVID vs Normal")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "comparaison_covid_normal.png", dpi=150)
    plt.close()
    print("  [OK] comparaison_covid_normal.png")

except Exception as e:
    print(f"  [ERROR] Generation des figures statistiques: {e}")

print("Generation terminee.")