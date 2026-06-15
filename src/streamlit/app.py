import json
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="COVID-19 Radiography Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR = BASE_DIR / "data" / "COVID-19_Radiography_Dataset"

# Rend les modules src/ importables quel que soit le repertoire de lancement.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CLASSES = ['Normal', 'Lung_Opacity', 'COVID', 'Viral Pneumonia']
CLASS_COLORS = {
    'Normal': '#2ecc71',
    'Lung_Opacity': '#f39c12',
    'COVID': '#e74c3c',
    'Viral Pneumonia': '#9b59b6'
}

REPORT_DATA = {
    'dataset_counts': {'Normal': 10192, 'Lung_Opacity': 6012, 'COVID': 3616, 'Viral Pneumonia': 1345},
    'train_test_split': {
        'COVID': {'train': 2893, 'test': 723},
        'Lung_Opacity': {'train': 4809, 'test': 1203},
        'Normal': {'train': 8154, 'test': 2038},
        'Viral Pneumonia': {'train': 1076, 'test': 269}
    },
    'baseline': {
        'Random Forest': {'accuracy': 0.82, 'f1_macro': 0.81, 'recall_covid': 0.69},
        'KNN': {'accuracy': 0.78, 'f1_macro': 0.77}
    },
    'rf_optimise': {
        'accuracy': 0.8254, 'f1_macro': 0.8226,
        'params': {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        'report': {
            'COVID': {'precision': 0.90, 'recall': 0.70, 'f1': 0.79, 'support': 723},
            'Lung_Opacity': {'precision': 0.80, 'recall': 0.74, 'f1': 0.77, 'support': 1203},
            'Normal': {'precision': 0.81, 'recall': 0.92, 'f1': 0.86, 'support': 2038},
            'Viral Pneumonia': {'precision': 0.95, 'recall': 0.80, 'f1': 0.87, 'support': 269}
        },
        'roc_auc': {'COVID': 0.92, 'Lung_Opacity': 0.88, 'Normal': 0.94, 'Viral Pneumonia': 0.97}
    },
    'cnn': {
        'accuracy': 0.50, 'f1_macro': 0.20,
        'architecture': 'Conv2D(32) → Conv2D(64) → Conv2D(128) → GlobalAvgPool → Dense(128) + Dropout(0.5) → Dense(4, softmax)',
        'optimizer': 'Adam (lr=1e-4)', 'epochs': 40, 'early_stopping': 6,
        'input_size': '128x128', 'class_weights': 'balanced',
        'report': {
            'COVID': {'precision': 1.00, 'recall': 0.00, 'f1': 0.00, 'support': 723},
            'Lung_Opacity': {'precision': 0.49, 'recall': 0.08, 'f1': 0.13, 'support': 1203},
            'Normal': {'precision': 0.50, 'recall': 0.98, 'f1': 0.66, 'support': 2038},
            'Viral Pneumonia': {'precision': 0.00, 'recall': 0.00, 'f1': 0.00, 'support': 269}
        }
    },
    'boosting': {
        'accuracy': 0.83, 'f1_macro': 0.83,
        'method': 'Gradient Boosting (300 estimators) on CNN GlobalAvgPool features',
        'report': {
            'COVID': {'precision': 0.88, 'recall': 0.80, 'f1': 0.84, 'support': 723},
            'Lung_Opacity': {'precision': 0.80, 'recall': 0.73, 'f1': 0.76, 'support': 1203},
            'Normal': {'precision': 0.82, 'recall': 0.90, 'f1': 0.85, 'support': 2038},
            'Viral Pneumonia': {'precision': 0.92, 'recall': 0.82, 'f1': 0.86, 'support': 269}
        }
    }
}


def show_report_image(filename, caption=None, width=None):
    path = REPORTS_DIR / filename
    if path.exists():
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=width is None, width=width)
        return True
    return False


def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


st.sidebar.title("Analyse COVID-19")
page = st.sidebar.radio(
    "Navigation",
    [
        "Accueil",
        "Méthodologie",
        "Données & qualité",
        "Exploration",
        "Prétraitement",
        "Modèles baselines",
        "Optimisation RF",
        "Deep Learning",
        "Boosting & Grad-CAM",
        "Transfer Learning",
        "Comparaison des modèles",
        "Conclusion",
        "Outil de prédiction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Analyse** : Classification de radiographies thoraciques\n\n"
    "**Dataset** : COVID-19 Radiography Database\n\n"
    "**Images** : 21 165 radiographies\n\n"
    "**Classes** : Normal, COVID, Lung Opacity, Viral Pneumonia"
)


# ============================================================
# PAGE 1 : ACCUEIL
# ============================================================
if page == "Accueil":
    st.title("Classification de radiographies pulmonaires COVID-19")

    section_header("Contexte", "Problématique médicale et enjeu de santé publique")

    st.markdown("""
    La pandémie de COVID-19 a mis en lumière le besoin d'outils de **diagnostic rapide et fiable**.
    L'analyse automatique de **radiographies thoraciques** par Deep Learning permet d'assister
    les professionnels de santé dans la détection précoce de pathologies pulmonaires.

    La radiographie thoracique est un examen peu coûteux (5-10 EUR), rapide (< 5 min)
    et largement disponible. Automatiser son analyse pourrait :
    - Détecter précocement les cas graves
    - Hiérarchiser les patients à risque
    - Soutenir les radiologues sur les cas à faible complexité
    - Permettre un dépistage de masse en situation de crise
    """)

    st.markdown("---")
    section_header("Objectif")

    st.markdown("""
    **Peut-on classifier automatiquement les radiographies thoraciques en 4 catégories**
    (Normal, COVID, Lung Opacity, Viral Pneumonia) avec une précision suffisante
    pour assister les praticiens ?
    """)

    st.markdown("---")
    section_header("Hypothèses de travail")

    st.markdown("""
    | Hypothèse | Justification |
    |-----------|---------------|
    | H1 : Les radiographies COVID présentent des signatures visuelles distinctes | Opacifications bilatérales, syndrome "glass ground" |
    | H2 : COVID et Lung Opacity sont proches et difficiles à séparer | Mécanisme radiologique similaire |
    | H3 : Un modèle hybride CNN + GB surperforme le CNN seul | Meilleure généralisation grâce aux features CNN |
    | H4 : Le déséquilibre des classes est un facteur déterminant | COVID = 17%, Viral Pneumonia = 6% |
    """)

    st.markdown("---")
    section_header("Performances obtenues", "Meilleur modèle : EfficientNetB0 (Transfer Learning)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1-macro", "0.905", "vs 0.873 hybride")
    col2.metric("Balanced accuracy", "0.911")
    col3.metric("Rappel COVID", "0.932")
    col4.metric("Interprétabilité", "Grad-CAM + SHAP")
    st.caption("Évaluation sur le test set figé (3 175 images). EfficientNetB0 et ResNet50 "
               "(Transfer Learning fine-tuné) dépassent l'ancien modèle hybride.")

    st.markdown("---")
    counts = REPORT_DATA['dataset_counts']
    total = sum(counts.values())
    st.markdown("### Dataset")
    st.markdown(f"Le **COVID-19 Radiography Dataset** contient **{total:,} images** réparties en 4 classes.")

    col1, col2, col3, col4 = st.columns(4)
    for col, (cls, count) in zip([col1, col2, col3, col4], counts.items()):
        col.metric(cls, f"{count:,}", f"{count/total*100:.1f}%")

    st.markdown("---")
    st.markdown("""
    ### Pipeline expérimental

    | Étape | Description | Input | Output |
    |-------|-------------|-------|--------|
    | 1 | Exploration et visualisation | Images brutes | Analyses statistiques |
    | 2 | Prétraitement et extraction de features | Images 256×256 | Features (HOG, gradients…) |
    | 3 | Modèles baselines | Vecteurs 4096D | RF (82%), KNN (78%) |
    | 4 | Optimisation Random Forest | Features HOG | RF optimisé (82,5%) |
    | 5 | Deep Learning | Images 128×128 | CNN (50%) |
    | 6 | Modèle hybride | Features CNN 128D | Boosting (F1 0,87) |
    | 7 | **Transfer Learning + fine-tuning** | Images 224×224×3 | **EfficientNetB0 (F1 0,905), ResNet50 (0,902)** |
    | 8 | Interprétabilité | Images 224×224 | Grad-CAM + SHAP |

    Le modèle **EfficientNetB0 (Transfer Learning fine-tuné)** est retenu : meilleur F1-macro
    (0,905) et meilleur rappel COVID (0,932), audité par Grad-CAM et SHAP. Voir les pages
    **Transfer Learning** et **Comparaison des modèles**.
    """)


# ============================================================
# PAGE 2 : MÉTHODOLOGIE
# ============================================================
elif page == "Méthodologie":
    st.title("Méthodologie expérimentale")

    section_header("Protocole expérimental", "Paramètres utilisés pour la modélisation")

    st.markdown("""
    ### 1. Préparation des données

    | Paramètre | Valeur |
    |-----------|--------|
    | Images redimensionnées (ML classique) | 64×64 |
    | Images redimensionnées (Deep Learning) | 128×128 |
    | Normalisation | Division par 255 |
    | Format | Niveaux de gris (1 canal) |
    | Split train/test | 80% / 20% |
    | Stratification | Oui, par classe |
    | Seed aléatoire | 42 |
    """)

    st.markdown("---")
    st.markdown("### 2. Métriques d'évaluation")

    st.markdown("""
    | Métrique | Formule | Justification |
    |---------|---------|-------------|
    | **Accuracy** | (VP + VN) / Total | Indicateur global, insuffisant seul sur données déséquilibrées |
    | **F1-macro** | Moyenne des F1 par classe | Pondère équitablement toutes les classes |
    | **Precision** | VP / (VP + FP) | Proportion de prédictions positives correctes |
    | **Recall** | VP / (VP + FN) | Proportion de positifs réels détectés |
    | **ROC-AUC** | Aire sous la courbe ROC | Capacité de discrimination |
    """)

    st.markdown("---")
    st.markdown("### 3. Gestion du déséquilibre")

    st.markdown("""
    Le dataset présente un déséquilibre important :
    - **Normal** : 48,2% (classe majoritaire)
    - **COVID** : 17,1% (classe minoritaire)
    - **Viral Pneumonia** : 6,4% (classe la plus minoritaire)

    **Stratégies utilisées** :
    1. **Stratification** du split train/test pour préserver les proportions
    2. **Class weights** inversement proportionnels aux effectifs
    3. **SMOTE** non appliqué (risque de générer des données non réalistes)
    """)

    st.markdown("---")
    st.markdown("### 4. Espace de recherche — Random Forest optimisé")

    st.markdown("""
    | Hyperparamètre | Espace testé | Valeur optimale |
    |---------------|--------------|-----------------|
    | n_estimators | [50, 100, 150] | 150 |
    | max_depth | [None, 15] | None |
    | min_samples_split | [2, 5] | 2 |
    | min_samples_leaf | [1, 2] | 1 |

    **Méthode** : RandomizedSearchCV (30 itérations, 5-fold CV)
    """)

    st.markdown("---")
    st.markdown("### 5. Architecture CNN")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ```
        Input (128, 128, 1)
        │
        Conv2D(32) + BatchNorm + MaxPool
        │
        Conv2D(64) + BatchNorm + MaxPool
        │
        Conv2D(128) + BatchNorm + MaxPool
        │
        GlobalAveragePooling2D (128)
        │
        Dense(128) + Dropout(0.5)
        │
        Dense(4) + Softmax
        ```
        """)
    with col2:
        st.markdown("""
        | Paramètre | Valeur |
        |-----------|--------|
        | Optimizer | Adam (lr=1e-4) |
        | Loss | Categorical Crossentropy |
        | Batch size | 32 |
        | Epochs | 40 |
        | Early stopping | Patience 6 |
        | Class weights | Balanced |
        """)
        st.warning("CNN seul en surapprentissage visible → approche hybride retenue")

    st.markdown("---")
    st.markdown("### 6. Modèle hybride — CNN + Gradient Boosting")

    st.markdown("""
    **Architecture du pipeline hybride** :

    1. **Feature extraction** : CNN pré-entraîné → couche `GlobalAveragePooling2D`
       - Sortie : vecteur de 128 features par image

    2. **Classification** : Gradient Boosting Classifier
       - n_estimators = 300

    **Avantages** :
    - Meilleure généralisation que CNN seul (+33 points d'accuracy)
    - Interprétabilité via les méthodes classiques (feature importance)
    - Temps d'entraînement réduit par rapport à un entraînement complet
    """)

    st.markdown("---")
    st.warning("⚠️ **Limitation importante** : ce modèle n'est pas validé pour un usage clinique. "
               "Il nécessite une validation prospective sur des données externes avant tout déploiement.")


# ============================================================
# PAGE 3 : DONNÉES & QUALITÉ
# ============================================================
elif page == "Données & qualité":
    st.title("Données & qualité")

    section_header("Description du dataset", "Source, structure et caractéristiques")

    st.markdown("""
    Le **COVID-19 Radiography Database** est issu d'une collaboration internationale.
    Les images proviennent de plusieurs hôpitaux et ont été annotées par des radiologues certifiés.

    | Propriété | Valeur |
    |---------|-------|
    | Source | COVID-19 Radiography Database (Kaggle / GitHub) |
    | Nombre total d'images | 21 165 |
    | Format | PNG, niveaux de gris |
    | Résolution originale | Variable (généralement 1024×1024) |
    | Classes | 4 (Normal, COVID, Lung Opacity, Viral Pneumonia) |
    | Masques de segmentation | Disponibles (dossier masks/) |
    | Licence | Usage académique |
    """)

    st.markdown("---")
    section_header("Répartition par classe")

    col1, col2 = st.columns(2)
    with col1:
        show_report_image("graphique_nombreimages_par_categorie.png", "Nombre d'images par catégorie")
    with col2:
        show_report_image("Pourcentage d'images par catégorie.png", "Pourcentage par catégorie")

    counts = REPORT_DATA['dataset_counts']
    total = sum(counts.values())

    st.markdown("### Analyse du déséquilibre")
    st.markdown(f"""
    | Classe | Effectif | Pourcentage | Train (80%) | Test (20%) |
    |-------|---------|-------------|-------------|------------|
    | Normal | {counts['Normal']:,} | {counts['Normal']/total*100:.1f}% | {counts['Normal']*0.8:.0f} | {counts['Normal']*0.2:.0f} |
    | Lung Opacity | {counts['Lung_Opacity']:,} | {counts['Lung_Opacity']/total*100:.1f}% | {counts['Lung_Opacity']*0.8:.0f} | {counts['Lung_Opacity']*0.2:.0f} |
    | COVID | {counts['COVID']:,} | {counts['COVID']/total*100:.1f}% | {counts['COVID']*0.8:.0f} | {counts['COVID']*0.2:.0f} |
    | Viral Pneumonia | {counts['Viral Pneumonia']:,} | {counts['Viral Pneumonia']/total*100:.1f}% | {counts['Viral Pneumonia']*0.8:.0f} | {counts['Viral Pneumonia']*0.2:.0f} |
    | **Total** | **{total:,}** | **100%** | **{total*0.8:.0f}** | **{total*0.2:.0f}** |
    """)

    st.markdown("---")
    section_header("Contrôle qualité")

    col1, col2 = st.columns(2)
    with col1:
        show_report_image("Présence de doublons dans les images.png", "Aucun doublon détecté")
    with col2:
        show_report_image("Distribution des images par catégorie et format.png", "Format : PNG uniquement")

    st.markdown("""
    - **Aucun doublon détecté** (comparaison par hash SHA-256)
    - **Aucune image corrompue**
    - **Aucun fichier manquant**
    - **Format unique** : PNG, niveaux de gris
    """)

    st.markdown("---")
    section_header("Biais et limites du dataset")

    st.warning("""
    ### Menaces à la validité

    1. **Biais de sélection** : les images proviennent de sources spécifiques → pas représentatif de toutes les populations
    2. **Biais d'annotation** : annotations faites par des radiologues de certains pays uniquement
    3. **Biais d'équipement** : images acquises sur des machines spécifiques → variabilité non contrôlée
    4. **Déséquilibre des classes** : traité par stratification et class weights, mais non éliminé

    Les performances obtenues ne sont pas directement transposables à d'autres populations
    ou conditions d'acquisition.
    """)


# ============================================================
# PAGE 4 : EXPLORATION
# ============================================================
elif page == "Exploration":
    st.title("Exploration statistique")

    section_header("Analyses statistiques", "Caractérisation visuelle et statistique des classes")

    st.markdown("""
    L'analyse exploratoire a été menée sur un échantillon représentatif de 500 images
    (125 par classe) pour caractériser les distributions d'intensité, de contraste et de texture.
    """)

    st.markdown("---")
    section_header("Distribution des intensités moyennes")

    col1, col2 = st.columns(2)
    with col1:
        show_report_image("hist_moyennes.png", "Distribution des intensités moyennes par classe")
    with col2:
        show_report_image("hist_ecarts_type.png", "Distribution des écarts-types (contraste) par classe")

    st.markdown("""
    **Observations** :
    - Les images **COVID** présentent une intensité moyenne plus élevée (opacification)
    - Le **contraste (écart-type)** est plus élevé pour les cas pathologiques
    - **Normal** se distingue clairement par ses faibles valeurs
    """)

    st.markdown("---")
    section_header("Comparaison COVID vs Normal")

    show_report_image("comparaison_covid_normal.png", "Comparaison des distributions COVID vs Normal")

    st.markdown("""
    **Conclusions** :
    - Distributions distinctes mais chevauchantes
    - COVID montre une hétérogénéité plus importante
    - Le contraste est un discriminant pertinent en première analyse
    """)

    st.markdown("---")
    section_header("Analyse pairwise des classes")

    show_report_image("COVID vs Autres catégories.png", "Comparaison COVID vs autres classes")

    st.markdown("""
    **Séparabilité visuelle** :
    - **Normal** vs toutes les autres : Distincte visuellement
    - **COVID** vs **Viral Pneumonia** : difficiles à séparer visuellement
    - **COVID** vs **Lung Opacity** : très similaires (même type d'opacification)

    Ce chevauchement explique la confusion importante observée dans les prédictions.
    """)

    st.markdown("---")
    section_header("Implications pour la modélisation")

    st.success("""
    **Ce que l'analyse exploratoire implique pour la modélisation** :

    1. Les features statistiques (moyenne, écart-type) sont informatives mais insuffisantes
    2. Les filtres de détection de contours (Canny, Sobel) peuvent aider à capturer les différences
    3. La confusion COVID/Lung Opacity est structurelle et attendue → besoin de features plus riches
    4. Un modèle profond (CNN) est justifié pour capturer les patterns complexes
    """)


# ============================================================
# PAGE 5 : PRÉTRAITEMENT
# ============================================================
elif page == "Prétraitement":
    st.title("Prétraitement & Feature Engineering")

    section_header("Pipeline de prétraitement", "Transformations appliquées aux images")

    st.markdown("""
    | Étape | Transformation | Paramètres | Objectif |
    |-------|---------------|-------------|----------|
    | 1 | Conversion niveaux de gris | — | Standardisation |
    | 2 | Redimensionnement | 256×256 | Uniformisation de la taille |
    | 3 | Normalisation | [0, 1] | Division par 255 |
    | 4 | Gaussian Blur | kernel 5×5 | Réduction du bruit |
    | 5 | Érosion | kernel 3×3 | Réduction des zones claires |
    | 6 | Dilatation | kernel 3×3 | Agrandissement des zones claires |
    | 7 | Canny | thresh [50, 150] | Détection de contours |
    | 8 | Sobel | — | Extraction de gradients |
    | 9 | Laplacian | — | Détection des détails |
    """)

    st.markdown("---")
    section_header("Exemples de transformations")

    col1, col2 = st.columns(2)
    with col1:
        show_report_image("image_gaussian_blur_0.png", "Gaussian Blur")
        show_report_image("image_canny_0.png", "Canny (contours)")
        show_report_image("image_erosion_0.png", "Érosion")
    with col2:
        show_report_image("image_sobel_0.png", "Sobel (gradients)")
        show_report_image("image_laplacian_0.png", "Laplacian (détails)")
        show_report_image("Comparaison_preprocessing_features.png", "Vue d'ensemble")

    st.markdown("---")
    section_header("Features extraites")

    st.markdown("""
    Pour les modèles ML classiques, deux types de features ont été testés :

    1. **Features brutes** : images aplaties 64×64 (4096 dimensions)
    2. **Features HOG** (Histogram of Oriented Gradients) :
       - pixels_per_cell : (16, 16)
       - cells_per_block : (2, 2)
       - Dimension réduite tout en conservant l'information de gradient
    """)

    st.markdown("---")
    section_header("Augmentation de données")

    st.markdown("""
    Pour améliorer la généralisation, une augmentation a été appliquée :
    - **Zoom** : ±10%
    - **Rotation** : ±15°
    - **Flip** : non appliqué (non pertinent pour des radiographies)

    Cette augmentation simule la variabilité des conditions d'acquisition.
    """)


# ============================================================
# PAGE 6 : BASELINES ML
# ============================================================
elif page == "Modèles baselines":
    st.title("Modèles baselines")

    section_header("Approche baseline", "Random Forest et KNN sur images aplaties")

    st.markdown("""
    Les modèles de base sont entraînés sur des images redimensionnées en **64×64**
    puis aplaties en vecteurs de **4096 features** (pixels).
    """)

    st.markdown("---")
    section_header("Random Forest (50 arbres)")

    rf = REPORT_DATA['baseline']['Random Forest']
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{rf['accuracy']*100:.0f}%")
    col2.metric("F1-macro", f"{rf['f1_macro']:.2f}")
    col3.metric("Recall COVID", f"{rf['recall_covid']:.2f}")

    show_report_image("matrice_confusion2_RF.png", "Matrice de confusion — Random Forest")

    st.markdown("""
    **Analyse** :
    - Bonne performance globale (82%)
    - Recall COVID faible (0,69) : 31% des cas COVID non détectés
    - Bon recall Normal (0,92) grâce à la classe majoritaire
    """)

    st.markdown("---")
    section_header("K-Nearest Neighbors (k=5)")

    knn = REPORT_DATA['baseline']['KNN']
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{knn['accuracy']*100:.0f}%")
    col2.metric("F1-macro", f"{knn['f1_macro']:.2f}")
    col3.metric("Note", "Sensibilité à la dimension")

    show_report_image("matrice_confusion_KNN.png", "Matrice de confusion — KNN")

    st.markdown("""
    **Analyse** :
    - Performance inférieure au RF (78%)
    - Sensible à la malédiction de la dimension (4096 features)
    - Temps de prédiction élevé
    → Non retenu pour la suite
    """)

    st.markdown("---")
    section_header("Conclusion baseline")

    st.success("""
    | Modèle | Accuracy | F1-macro | Point faible |
    |--------|----------|----------|-------------|
    | Random Forest (50) | 82% | 0,81 | Recall COVID = 0,69 |
    | KNN (k=5) | 78% | 0,77 | Haute dimensionnalité |

    → **Random Forest** sélectionné pour l'optimisation
    → Axes d'amélioration : features HOG, hyperparamètres, modèle profond
    """)


# ============================================================
# PAGE 7 : OPTIMISATION RF
# ============================================================
elif page == "Optimisation RF":
    st.title("Optimisation du Random Forest")

    section_header("RandomizedSearchCV", "Recherche des meilleurs hyperparamètres")

    params = REPORT_DATA['rf_optimise']['params']
    st.markdown("""
    | Hyperparamètre | Valeur optimale |
    |-------------|----------------|
    | n_estimators | 150 |
    | max_depth | None (non limité) |
    | min_samples_split | 2 |
    | min_samples_leaf | 1 |
    """)

    st.markdown("---")
    section_header("Résultats du modèle optimisé")

    rf_opt = REPORT_DATA['rf_optimise']
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{rf_opt['accuracy']*100:.2f}%")
    col2.metric("F1-macro", f"{rf_opt['f1_macro']:.4f}")

    report = rf_opt['report']
    rows = []
    for cls, r in report.items():
        rows.append({
            'Classe': cls,
            'Précision': f"{r['precision']:.2f}",
            'Rappel': f"{r['recall']:.2f}",
            'F1-score': f"{r['f1']:.2f}",
            'Support': int(r['support'])
        })
    st.dataframe(pd.DataFrame(rows).set_index('Classe'), use_container_width=True)

    st.markdown("---")
    section_header("Analyse des résultats")

    st.markdown("""
    - **Viral Pneumonia** : meilleure performance (F1=0,87) → classe visuellement distincte
    - **Normal** : bon rappel (0,92) → classe majoritaire bien apprise
    - **COVID** : précision élevée (0,90) mais rappel modéré (0,70) → le modèle est prudent
    - **Lung_Opacity** : confusion avec COVID → même type d'opacification pulmonaire
    """)

    st.markdown("---")
    section_header("Scores ROC-AUC par classe")

    roc_data = rf_opt['roc_auc']
    st.markdown("| Classe | ROC-AUC |")
    st.markdown("|--------|--------|")
    for cls, auc in roc_data.items():
        st.markdown(f"| {cls} | {auc:.2f} |")

    st.success(f"ROC-AUC moyen : {np.mean(list(roc_data.values())):.2f}")

    st.markdown("---")
    section_header("Analyse des erreurs")

    st.markdown("""
    **Confusions critiques identifiées** :

    1. **COVID → Normal** (~10%) : absence de marqueurs visuels distinctifs nets
    2. **COVID → Lung_Opacity** (~15%) : opacifications pulmonaires similaires
    3. **Lung_Opacity → COVID** (~12%) : confusion bidirectionnelle

    **Implications médicales** :
    - Un faux négatif COVID (non détecté) est coûteux en situation réelle
    - La précision élevée mais rappel modéré pose problème pour le dépistage
    → L'approche hybride vise à améliorer ce rappel COVID
    """)


# ============================================================
# PAGE 8 : DEEP LEARNING
# ============================================================
elif page == "Deep Learning":
    st.title("Deep Learning — CNN")

    section_header("Architecture CNN", "Réseau convolutif 3 blocs")

    cnn = REPORT_DATA['cnn']
    st.markdown(f"**Architecture** : `{cnn['architecture']}`")
    st.markdown(f"**Optimizer** : {cnn['optimizer']}")
    st.markdown(f"**Epochs** : {cnn['epochs']} (Early Stopping patience={cnn['early_stopping']})")
    st.markdown(f"**Input** : {cnn['input_size']}")
    st.markdown(f"**Class weights** : {cnn['class_weights']}")

    st.markdown("---")
    section_header("Courbes d'entraînement")

    show_report_image("loss_accuracy_curves.png", "Loss et Accuracy — train vs validation")

    st.markdown("---")
    section_header("Résultats")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{cnn['accuracy']*100:.0f}%")
    col2.metric("F1-macro", f"{cnn['f1_macro']:.2f}")

    report = cnn['report']
    rows = []
    for cls, r in report.items():
        rows.append({
            'Classe': cls,
            'Précision': f"{r['precision']:.2f}",
            'Rappel': f"{r['recall']:.2f}",
            'F1-score': f"{r['f1']:.2f}",
            'Support': int(r['support'])
        })
    st.dataframe(pd.DataFrame(rows).set_index('Classe'), use_container_width=True)

    st.markdown("---")
    section_header("Matrice de confusion")

    show_report_image("confusion_cnn.png", "Matrice de confusion — CNN")

    st.markdown("---")
    section_header("Analyse critique")

    st.error("""
    **Le CNN seul présente des performances insuffisantes** :

    1. **Accuracy = 50%** → proche du hasard (25%) sur 4 classes
    2. **Surapprentissage visible** : loss validation augmente alors que loss training diminue
    3. **Bon rappel Normal** (0,98) uniquement car classe majoritaire

    **Causes probables** :
    - Dataset insuffisant pour un CNN from scratch
    - Manque de techniques de régularisation plus fortes
    - Architecture sous-optimale pour ce type d'images

    L'approche hybride (CNN comme extracteur + GB) permet de contourner ce problème.
    """)


# ============================================================
# PAGE 9 : BOOSTING & GRAD-CAM
# ============================================================
elif page == "Boosting & Grad-CAM":
    st.title("Boosting hybride & Grad-CAM")

    section_header("Approche hybride", "CNN + Gradient Boosting")

    st.markdown("""
    **Principe** : utiliser le CNN comme extracteur de features plutôt que comme classificateur direct.
    Les features de la couche `GlobalAveragePooling2D` (128 dimensions) sont extraites
    puis utilisées pour entraîner un Gradient Boosting Classifier.
    """)

    st.markdown("---")
    section_header("Résultats comparatifs")

    boosting = REPORT_DATA['boosting']
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{boosting['accuracy']*100:.0f}%")
    col2.metric("F1-macro", f"{boosting['f1_macro']:.2f}")

    st.markdown("""
    | Modèle | Accuracy | F1-macro | Recall COVID |
    |--------|----------|----------|-------------|
    | RF baseline | 82% | 0,81 | 0,69 |
    | RF optimisé | 82,5% | 0,82 | 0,70 |
    | CNN seul | 50% | 0,20 | 0,00 |
    | **Boosting hybride** | **83%** | **0,83** | **0,80** |
    """)

    st.success("+33 points d'accuracy par rapport au CNN seul.")

    st.markdown("---")
    section_header("Classification report — Boosting")

    report = boosting['report']
    rows = []
    for cls, r in report.items():
        rows.append({
            'Classe': cls,
            'Précision': f"{r['precision']:.2f}",
            'Rappel': f"{r['recall']:.2f}",
            'F1-score': f"{r['f1']:.2f}",
            'Support': int(r['support'])
        })
    st.dataframe(pd.DataFrame(rows).set_index('Classe'), use_container_width=True)

    show_report_image("confusion_boosting.png", "Matrice de confusion — Boosting hybride")

    st.markdown("---")
    section_header("Interprétabilité — Grad-CAM")

    st.markdown("""
    **Grad-CAM** (Gradient Class Activation Map) visualise les zones de l'image
    sur lesquelles le CNN se concentre pour effectuer sa prédiction.
    """)

    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            show_report_image(f"gradcam_{i}.png", f"Grad-CAM exemple {i+1}")

    st.success("""
    **Validation de l'interprétabilité** :
    - Les zones actives se situent dans les poumons
    - Pas de bruit parasite (bords de l'image)
    - Modèle médicalement interprétable

    Grad-CAM permet un audit, pas une validation clinique.
    """)


# ============================================================
# PAGE : TRANSFER LEARNING
# ============================================================
elif page == "Transfer Learning":
    st.title("Transfer Learning (ResNet50, EfficientNetB0, InceptionV3)")

    st.markdown("""
    Trois modèles pré-entraînés sur **ImageNet**, fine-tunés selon un protocole
    en **2 phases** :
    - **Phase 1 — extraction** : backbone gelé, on entraîne uniquement la tête
      `GAP → Dropout → Dense(256) → Dropout → softmax` (Adam `lr=1e-3`).
    - **Phase 2 — fine-tuning** : on dégèle le dernier bloc, on **garde les
      BatchNorm gelées** (sinon divergence), on **recompile** avec Adam `lr=1e-5`.

    Les images en niveaux de gris sont converties en **RGB 3 canaux** (poids
    ImageNet) et chaque backbone utilise son `preprocess_input` dédié. Le
    déséquilibre est géré par des **poids de classe**. Tous les modèles sont
    évalués sur le **même test set figé** (`reports/splits/test.csv`).
    """)

    BACKBONES = {
        "efficientnetb0": "EfficientNetB0 (224×224)",
        "resnet50": "ResNet50 (224×224)",
        "inceptionv3": "InceptionV3 (299×299)",
    }

    def load_metrics_json(name):
        path = METRICS_DIR / f"{name}_test_metrics.json"
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        return None

    any_found = False
    for key, label in BACKBONES.items():
        m = load_metrics_json(key)
        st.markdown(f"### {label}")
        if m is None:
            st.info(f"Métriques absentes — lancez le notebook `06-transfer-learning.ipynb` "
                    f"(le fichier `reports/metrics/{key}_test_metrics.json` sera généré).")
            continue
        any_found = True
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1-macro", f"{m['f1_macro']:.3f}")
        c2.metric("Balanced acc.", f"{m['balanced_accuracy']:.3f}")
        c3.metric("Rappel COVID", f"{m['per_class']['COVID']['recall']:.3f}")
        if m.get("roc_auc_ovr_macro") is not None:
            c4.metric("ROC-AUC", f"{m['roc_auc_ovr_macro']:.3f}")
        per_class = pd.DataFrame(m["per_class"]).T[["precision", "recall", "f1"]]
        st.dataframe(per_class.style.format("{:.3f}"), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            p = FIGURES_DIR / f"learning_curves_{key}.png"
            if p.exists():
                st.image(str(p), caption="Courbes d'apprentissage (sur-apprentissage)")
        with col_b:
            p = FIGURES_DIR / f"confusion_{key}.png"
            if p.exists():
                st.image(str(p), caption="Matrice de confusion (test)")
        st.markdown("---")

    if not any_found:
        st.warning("Aucun modèle de Transfer Learning entraîné détecté. "
                   "Exécutez le notebook 06 puis synchronisez `reports/metrics/` "
                   "et `reports/figures/`.")

    # --- Interprétabilité : Grad-CAM + SHAP ---
    st.markdown("---")
    section_header("Interprétabilité", "Le modèle regarde-t-il les bonnes zones (les poumons) ?")

    CLS_FILES = {"COVID": "COVID", "Lung Opacity": "Lung_Opacity",
                 "Normal": "Normal", "Viral Pneumonia": "Viral_Pneumonia"}

    st.markdown("#### Grad-CAM — zones d'attention du modèle")
    gc_model = st.selectbox("Modèle", ["efficientnetb0", "resnet50"], format_func=str.upper,
                            key="gradcam_model")
    shown = False
    cols = st.columns(2)
    for i, (label, slug) in enumerate(CLS_FILES.items()):
        p = FIGURES_DIR / f"gradcam_{gc_model}_{slug}.png"
        if p.exists():
            cols[i % 2].image(str(p), caption=f"Grad-CAM — {label}", use_container_width=True)
            shown = True
    # cas mal classés
    miscls = sorted(FIGURES_DIR.glob(f"gradcam_{gc_model}_miscls_*.png"))
    if miscls:
        st.markdown("**Cas mal classés** (où le modèle se trompe-t-il ?)")
        mcols = st.columns(2)
        for i, p in enumerate(miscls):
            mcols[i % 2].image(str(p), use_container_width=True)
        shown = True
    if not shown:
        st.info(f"Figures Grad-CAM pour {gc_model.upper()} non trouvées dans `reports/figures/`.")

    st.markdown("#### SHAP — contribution des pixels (meilleur modèle)")
    shap_p = FIGURES_DIR / "shap_efficientnetb0.png"
    if shap_p.exists():
        st.image(str(shap_p), caption="SHAP (GradientExplainer) — EfficientNetB0", use_container_width=True)
    else:
        st.info("Figure SHAP non trouvée.")

    st.caption("Lecture : une activation localisée sur les champs pulmonaires conforte la "
               "plausibilité clinique ; une activation sur les bords/marqueurs signalerait un biais.")


# ============================================================
# PAGE : COMPARAISON DES MODÈLES
# ============================================================
elif page == "Comparaison des modèles":
    st.title("Comparaison des modèles")

    st.markdown("""
    Comparaison de tous les modèles sur le **même test set figé**. La métrique de
    sélection est le **F1-macro** (robuste au déséquilibre), avec le **rappel COVID**
    comme garde-fou clinique.

    > **Note méthodologique** : l'hybride (CNN+GB) est re-scoré sur ce test set par
    > le notebook 08. Le baseline RF-HOG d'origine a été évalué sur un échantillon
    > équilibré différent ; ses chiffres ne sont pas strictement comparables.
    """)

    table_path = METRICS_DIR / "comparison_table.csv"
    if table_path.exists():
        df = pd.read_csv(table_path)
        st.markdown("### Table maîtresse (triée par F1-macro)")
        st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns if c != "modele"}),
                     use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        d = df.dropna(subset=["f1_macro"]).sort_values("f1_macro")
        ax.barh(d["modele"], d["f1_macro"], color="#3498db", edgecolor="white")
        ax.set_xlabel("F1-macro")
        ax.set_xlim(0, 1)
        for i, v in enumerate(d["f1_macro"]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Table de comparaison absente — lancez le notebook "
                "`08-comparaison.ipynb` (génère `reports/metrics/comparison_table.csv`).")

    grid = FIGURES_DIR / "comparison_confusions.png"
    if grid.exists():
        st.markdown("### Matrices de confusion")
        st.image(str(grid), use_container_width=True)

    st.markdown("### Courbes d'apprentissage")
    cols = st.columns(3)
    for col, key in zip(cols, ["efficientnetb0", "resnet50", "inceptionv3"]):
        p = FIGURES_DIR / f"learning_curves_{key}.png"
        if p.exists():
            col.image(str(p), caption=key)


# ============================================================
# PAGE 10 : CONCLUSION
# ============================================================
elif page == "Conclusion":
    st.title("Conclusion & Perspectives")

    section_header("Synthèse")

    st.markdown("""
    Les différentes expérimentations menées dans ce projet ont permis d'évaluer
    plusieurs approches de machine learning et de deep learning pour la
    classification de radiographies pulmonaires.

    Dans un premier temps, le modèle CNN utilisé seul a montré des performances
    limitées, avec une accuracy d'environ 50 % sur l'ensemble de validation.
    Bien que le modèle parvienne à identifier efficacement la classe Normal
    (recall ≈ 0,98), il rencontre davantage de difficultés pour détecter
    correctement les classes pathologiques, notamment COVID et Viral Pneumonia.

    L'utilisation de la méthode Grad-CAM a permis d'apporter une dimension
    d'interprétabilité au modèle. Cette technique offre une visualisation des
    zones de l'image influençant la décision du réseau, confirmant que le modèle
    se concentre majoritairement sur les régions pulmonaires pertinentes.

    Afin d'améliorer les performances, une approche hybride combinant CNN et
    Gradient Boosting a été mise en place. Les caractéristiques extraites par le
    CNN ont été utilisées comme entrée pour un modèle de boosting, ce qui a
    permis d'obtenir une amélioration significative des performances, avec une
    accuracy atteignant 83 %.

    Cette approche permet également d'obtenir un meilleur équilibre entre
    précision et rappel pour l'ensemble des classes, réduisant ainsi les erreurs
    de classification.

    **Étape finale — Transfer Learning.** Deux réseaux pré-entraînés sur ImageNet
    (EfficientNetB0 et ResNet50) ont été implémentés et **fine-tunés en deux phases**
    (extraction puis ajustement du dernier bloc à faible learning rate, BatchNorm gelées).
    Évalués sur un **test set figé** identique pour tous les modèles, ils **dépassent
    nettement l'hybride** : **EfficientNetB0 atteint F1-macro 0,905** (rappel COVID 0,932)
    et ResNet50 0,902, contre 0,873 pour l'hybride re-scoré. **EfficientNetB0 est le modèle
    retenu**, pour son meilleur F1-macro et son meilleur rappel COVID (le critère clinique).
    L'interprétabilité est confirmée par **Grad-CAM (sur les 2 modèles) et SHAP**.
    """)

    st.markdown("---")
    section_header("Tableau récapitulatif des performances")

    st.markdown("""
    | Modèle | Accuracy | F1-macro | Recall COVID | Forces | Faiblesses |
    |--------|----------|----------|----------|--------|----------|
    | RF baseline | 82% | 0,80 | 0,69 | Robuste, rapide | Rappel COVID faible |
    | CNN seul | 50% | 0,20 | 0,00 | Grad-CAM | Surapprentissage |
    | Hybride CNN+GB (re-scoré) | 86% | 0,87 | 0,88 | Bon compromis | Pipeline complexe |
    | ResNet50 (Transfer Learning) | 90% | 0,902 | 0,89 | Très bon | Plus lourd |
    | **EfficientNetB0 (Transfer Learning)** | **90%** | **0,905** | **0,93** | **Meilleur F1 + rappel COVID** | — |

    *Tous les modèles évalués sur le même test set figé (3 175 images) — comparaison équitable.*
    """)

    st.markdown("---")
    section_header("Limites méthodologiques")

    st.warning("""
    1. **Surapprentissage du CNN** : n'a pas convergé correctement
    2. **Dataset déséquilibré** : traité mais non éliminé
    3. **Biais de sélection** : données de quelques sources uniquement
    4. **Validation externe manquante** : pas de test sur données indépendantes
    5. **Pas de segmentation pulmonaire** : les features incluent des informations non pulmonaires
    """)

    st.markdown("---")
    section_header("Recommandations")

    st.info("""
    **Usage NON clinique** :
    ce modèle est un prototype de recherche. Il ne doit pas être utilisé
    en contexte clinique sans validation prospective approfondie.

    **Axes d'amélioration pour une V2** :

    1. **Segmentation pulmonaire préalable** (U-Net) → images nettoyées
    2. **Transfer learning** (ResNet, EfficientNet pré-entraînés) → meilleure convergence
    3. **Data augmentation plus agressive** → réduction du surapprentissage
    4. **Validation externe** → test sur un autre dataset
    5. **Ensemble de modèles** → robustesse accrue
    """)

    st.markdown("---")
    section_header("Perspectives")

    st.markdown("""
    | Court terme | Long terme |
    |--------------|-----------|
    | Amélioration du pipeline CNN | Intégration dans un système d'aide à la décision |
    | Test sur données externes | Validation prospective multi-sites |
    | Optimisation des seuils | Déploiement en environnement contrôlé |
    """)


# ============================================================
# PAGE 11 : PRÉDICTION
# ============================================================
elif page == "Outil de prédiction":
    st.title("Détection COVID — outil de dépistage")

    st.warning(
        "⚠️ **Prototype de recherche** — non validé pour un usage clinique. "
        "Les prédictions doivent être interprétées par un professionnel qualifié."
    )
    st.markdown(
        "**Objectif : détecter la présence de COVID** sur une radiographie "
        "(COVID vs non-COVID). Le modèle prédit les 4 classes ; on en déduit la "
        "**probabilité de COVID**, puis un verdict selon un seuil ajustable."
    )

    import random
    from src.data.tf_pipeline import CLASS_NAMES   # ['COVID','Lung_Opacity','Normal','Viral Pneumonia']
    COVID_IDX = CLASS_NAMES.index("COVID")
    TL_DIR = BASE_DIR / "models" / "transfer"
    SPLIT_TEST = REPORTS_DIR / "splits" / "test.csv"

    available_tl = sorted(p.stem.replace("_best", "") for p in TL_DIR.glob("*_best.keras"))
    if not available_tl or not SPLIT_TEST.exists():
        st.error("Modèles de Transfer Learning ou split de test introuvables "
                 "(`models/transfer/*_best.keras`, `reports/splits/test.csv`).")
        st.stop()

    c1, c2 = st.columns([2, 1])
    default_model = "efficientnetb0" if "efficientnetb0" in available_tl else available_tl[0]
    model_name = c1.selectbox("Modèle", available_tl,
                              index=available_tl.index(default_model), format_func=str.upper)
    threshold = c2.slider("Seuil de détection COVID", 0.10, 0.90, 0.50, 0.05,
                          help="Abaisser le seuil = détecter plus de COVID (rappel ↑) "
                               "au prix de plus de fausses alertes.")

    @st.cache_resource
    def load_tl_model(name):
        import tensorflow as tf
        from src.models.transfer_learning import get_spec
        model = tf.keras.models.load_model(str(TL_DIR / f"{name}_best.keras"), compile=False)
        return model, get_spec(name)

    @st.cache_data
    def test_samples():
        df = pd.read_csv(SPLIT_TEST)
        return list(zip(df["filepath"].tolist(), df["class"].tolist()))

    def draw_image(current=None):
        samples = test_samples()
        if current:
            samples = [s for s in samples if s[0] != current] or samples
        return random.choice(samples)

    if (not st.session_state.get("covid_img")
            or not Path(st.session_state["covid_img"]).exists()):
        fp, lab = draw_image()
        st.session_state["covid_img"] = fp
        st.session_state["covid_lab"] = lab

    if st.button("🎲 Tirer une radiographie aléatoire", type="primary", use_container_width=True):
        fp, lab = draw_image(st.session_state.get("covid_img"))
        st.session_state["covid_img"] = fp
        st.session_state["covid_lab"] = lab

    try:
        from src.visualization import interpretability as interp
        model, spec = load_tl_model(model_name)
        fp = st.session_state["covid_img"]
        true_lab = st.session_state["covid_lab"]
        raw = interp.load_raw_rgb(fp, spec.img_size)
        batch = spec.preprocess_fn(np.expand_dims(raw.astype("float32"), 0))
        with st.spinner("Analyse..."):
            proba = model.predict(batch, verbose=0)[0]
        p_covid = float(proba[COVID_IDX])
        covid_detected = p_covid >= threshold
        true_is_covid = (true_lab == "COVID")

        col_img, col_res = st.columns([1, 1.3])
        with col_img:
            st.image(raw.astype("uint8"),
                     caption=f"Radiographie — vraie classe : {true_lab}", width=300)
        with col_res:
            if covid_detected:
                st.markdown("## 🔴 COVID détecté")
            else:
                st.markdown("## 🟢 Pas de COVID")
            st.metric("Probabilité COVID", f"{p_covid:.1%}",
                      help=f"Seuil de décision : {threshold:.0%}")
            st.progress(p_covid)
            if covid_detected == true_is_covid:
                st.success("✅ Dépistage correct"
                           + (" — vrai positif" if true_is_covid else " — vrai négatif"))
            elif covid_detected and not true_is_covid:
                st.warning("⚠️ Fausse alerte (faux positif) : non-COVID classé COVID.")
            else:
                st.error("❌ COVID manqué (faux négatif) — l'erreur la plus grave en dépistage.")

        with st.expander("Détails du modèle (4 classes + Grad-CAM)"):
            d1, d2 = st.columns([1.3, 1])
            with d1:
                st.caption("Probabilités par classe")
                fig, ax = plt.subplots(figsize=(4.5, 2.2))
                colors = [CLASS_COLORS.get(c, "#95a5a6") for c in CLASS_NAMES]
                ax.barh(CLASS_NAMES, proba, color=colors, edgecolor="white")
                ax.set_xlim(0, 1); ax.invert_yaxis()
                ax.tick_params(labelsize=8); ax.set_xlabel("Probabilité", fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            with d2:
                st.caption("Grad-CAM (zones regardées)")
                try:
                    heatmap, _ = interp.make_gradcam_heatmap(batch, model, spec.last_conv_layer)
                    st.image(interp.overlay_heatmap(raw, heatmap), width=220)
                except Exception as exc:
                    st.caption(f"Grad-CAM indisponible : {exc}")
            st.caption(f"Modèle : {model_name.upper()} · entrée {spec.img_size}×{spec.img_size}×3 · "
                       "le verdict COVID/non-COVID est dérivé de la probabilité de la classe COVID.")

        with st.expander("🔬 SHAP — contribution des pixels (lent, ~15 s)"):
            st.caption("SHAP attribue à **chaque pixel** sa contribution à la prédiction "
                       "(méthode plus rigoureuse que Grad-CAM, mais coûteuse).")

            @st.cache_resource(show_spinner=False)
            def get_shap_explainer(mname):
                import shap as _shap
                m, sp = load_tl_model(mname)
                train_csv = REPORTS_DIR / "splits" / "train.csv"
                bdf = pd.read_csv(train_csv).groupby("class").head(4)  # 16 images de fond
                bg = np.stack([sp.preprocess_fn(
                    interp.load_raw_rgb(fp, sp.img_size).astype("float32"))
                    for fp in bdf["filepath"]])
                return _shap.GradientExplainer(m, bg)

            if st.button("Calculer SHAP pour cette image", key="shap_btn"):
                try:
                    import shap as _shap
                    with st.spinner("Calcul SHAP (~15 s)..."):
                        explainer = get_shap_explainer(model_name)
                        sv = explainer.shap_values(batch, nsamples=32)
                        sv = np.asarray(sv)
                        # On ne garde que la contribution vers la classe COVID (outil de dépistage)
                        sv_covid = sv[..., COVID_IDX] if sv.ndim == 5 else sv
                        _shap.image_plot(sv_covid, np.expand_dims(raw / 255.0, 0),
                                         labels=np.array([["→ COVID"]]), show=False)
                        fig = plt.gcf()
                        st.pyplot(fig); plt.close(fig)
                    st.caption("Contribution des pixels **vers la classe COVID** : "
                               "rouge = pousse vers COVID · bleu = pousse contre. "
                               "(Image non-COVID → carte quasi neutre, c'est normal.)")
                except Exception as exc:
                    st.error(f"SHAP indisponible : {exc}")
    except Exception as exc:
        st.error(f"Erreur de prédiction : {exc}")
