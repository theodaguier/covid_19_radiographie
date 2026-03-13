import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import ndimage
from skimage import feature
from pathlib import Path

# ===== CONFIG =====
st.set_page_config(
    page_title="COVID-19 Radiography Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== PATHS =====
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data" / "COVID-19_Radiography_Dataset"

CLASSES = ['Normal', 'Lung_Opacity', 'COVID', 'Viral Pneumonia']
CLASS_COLORS = {
    'Normal': '#2ecc71',
    'Lung_Opacity': '#f39c12',
    'COVID': '#e74c3c',
    'Viral Pneumonia': '#9b59b6'
}


# ===== FEATURE EXTRACTION HOG =====
def extract_hog_features(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert('L'), dtype=np.float64) / 255.0
    return feature.hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2))


# ===== LOAD MODELS & METRICS =====
@st.cache_resource
def load_models():
    models = {}
    label_mapping = None
    for f in MODELS_DIR.glob("*.pkl"):
        name = f.stem
        obj = joblib.load(f)
        if 'label_mapping' in name:
            label_mapping = obj
        else:
            models[name] = obj
    return models, label_mapping


@st.cache_data
def load_metrics():
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def show_report_image(filename, caption=None, width=None):
    """Affiche une image du dossier reports/ si elle existe."""
    path = REPORTS_DIR / filename
    if path.exists():
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=width is None, width=width)
        return True
    return False


# ===== SIDEBAR =====
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller a",
    [
        "Presentation",
        "Exploration des donnees",
        "Preprocessing & Features",
        "Modelisation Baseline",
        "Optimisation & Metriques",
        "Deep Learning & Boosting",
        "Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Projet** : Analyse de radiographies pulmonaires COVID-19\n\n"
    "**Formation** : DataScientest\n\n"
    "**Dataset** : COVID-19 Radiography Database\n\n"
    "**Images** : 21 165 radiographies thoraciques\n\n"
    "**Classes** : Normal, COVID, Lung Opacity, Viral Pneumonia"
)

# ===== LOAD DATA =====
metrics = load_metrics()

# ===== PAGES =====

# ----- PAGE 1 : PRESENTATION -----
if page == "Presentation":
    st.title("Analyse de radiographies pulmonaires COVID-19")

    st.markdown("""
    ## Contexte

    La pandemie de COVID-19 a mis en lumiere le besoin d'outils de **diagnostic rapide et fiable**.
    L'analyse automatique de **radiographies thoraciques** par Machine Learning permet d'assister
    les professionnels de sante dans la detection de pathologies pulmonaires.

    ## Objectif

    Developper un modele de classification capable de distinguer **4 types de radiographies** :

    - **Normal** -- Poumons sains
    - **COVID** -- Infection COVID-19
    - **Lung Opacity** -- Opacite pulmonaire (pneumonie bacterienne, etc.)
    - **Viral Pneumonia** -- Pneumonie virale (non-COVID)

    ## Dataset
    """)

    if metrics:
        counts = metrics['dataset_counts']
    else:
        counts = {'Normal': 10192, 'Lung_Opacity': 6012, 'COVID': 3616, 'Viral Pneumonia': 1345}

    total = sum(counts.values())
    st.markdown(f"Le **COVID-19 Radiography Dataset** contient **{total:,} images** reparties en 4 classes.")

    col1, col2, col3, col4 = st.columns(4)
    for col, (cls, count) in zip([col1, col2, col3, col4], counts.items()):
        col.metric(cls, f"{count:,}", f"{count/total*100:.1f}%")

    st.markdown("""
    ## Pipeline du projet

    | Etape | Notebook | Description |
    |-------|----------|-------------|
    | 1 | `01-exploration-dataviz` | Analyse exploratoire, distribution des classes, doublons |
    | 2 | `02-preprocessing-feature-engineering` | Pretraitement (grayscale, resize 256x256, normalisation), 8 techniques de transformation |
    | 3 | `03-modelisation-baseline` | Random Forest (50 arbres) et KNN sur images aplaties 64x64 |
    | 4 | `04-optimisation-metriques` | Optimisation hyperparametres (RandomizedSearchCV), analyse des erreurs, courbes ROC |
    | 5 | `05-deep-learning-boosting` | CNN (3 blocs Conv2D), Gradient Boosting sur features CNN, Grad-CAM |

    ## Qualite des donnees

    - Aucun doublon detecte
    - Aucune image corrompue
    - Aucun fichier manquant
    - Format unique : PNG, niveaux de gris
    - **Defi principal** : desequilibre des classes (COVID = classe minoritaire)
    """)


# ----- PAGE 2 : EXPLORATION -----
elif page == "Exploration des donnees":
    st.title("Exploration des donnees")

    st.markdown("### Distribution des classes")

    col1, col2 = st.columns(2)
    with col1:
        show_report_image("graphique_nombreimages_par_categorie.png",
                          "Nombre d'images par categorie")
    with col2:
        show_report_image("Pourcentage d'images par catégorie.png",
                          "Pourcentage par categorie")

    if metrics:
        counts = metrics['dataset_counts']
    else:
        counts = {'Normal': 10192, 'Lung_Opacity': 6012, 'COVID': 3616, 'Viral Pneumonia': 1345}
    total = sum(counts.values())

    st.markdown(f"""
    **Observations** :
    - Le dataset est **desequilibre** : Normal ({counts.get('Normal',0)/total*100:.0f}%) domine largement
    - COVID ne represente que **{counts.get('COVID',0)/total*100:.0f}%** des images
    - Viral Pneumonia est la classe minoritaire (**{counts.get('Viral Pneumonia',0)/total*100:.0f}%**)
    """)

    st.markdown("### COVID vs Autres categories")
    show_report_image("COVID vs Autres catégories.png",
                      "Comparaison COVID vs autres classes")

    st.markdown("### Verification des doublons et formats")
    col1, col2 = st.columns(2)
    with col1:
        show_report_image("Présence de doublons dans les images.png",
                          "Aucun doublon detecte")
    with col2:
        show_report_image("Distribution des images par catégorie et format.png",
                          "Toutes les images sont au format PNG")

    st.markdown("### Exemples d'images par classe")
    show_report_image("Exemple d'image pour chaque catégorie.png",
                      "Echantillons representatifs de chaque categorie")


# ----- PAGE 3 : PREPROCESSING & FEATURES -----
elif page == "Preprocessing & Features":
    st.title("Preprocessing & Feature Engineering")

    st.markdown("""
    ## Pipeline de pretraitement

    1. **Conversion en niveaux de gris** -- Standardisation du format
    2. **Redimensionnement 256x256** -- Taille uniforme pour tous les modeles
    3. **Normalisation [0, 1]** -- Division des pixels par 255

    ## Techniques de transformation

    8 techniques appliquees pour enrichir l'analyse et extraire des features visuelles :
    """)

    st.markdown("### Exemples d'images pretraitees")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            show_report_image(f"image_pretraitee_{i}.png", f"Image {i+1}")

    st.markdown("### Transformations appliquees")
    col1, col2 = st.columns(2)
    with col1:
        show_report_image("image_gaussian_blur_0.png", "Gaussian Blur -- Reduction du bruit")
        show_report_image("image_canny_0.png", "Canny -- Detection de contours")
        show_report_image("image_erosion_0.png", "Erosion -- Reduction des zones claires")
    with col2:
        show_report_image("image_sobel_0.png", "Sobel -- Extraction des gradients")
        show_report_image("image_laplacian_0.png", "Laplacien -- Details fins et contours")

    st.markdown("### Comparaison globale des transformations")
    show_report_image("Comparaison_preprocessing_features.png",
                      "Vue d'ensemble de toutes les techniques de preprocessing")

    st.markdown("### Analyse statistique des intensites")
    col1, col2 = st.columns(2)
    with col1:
        show_report_image("hist_moyennes.png", "Distribution des intensites moyennes")
    with col2:
        show_report_image("hist_ecarts_type.png", "Distribution des ecarts-type")

    show_report_image("comparaison_covid_normal.png",
                      "Comparaison des distributions COVID vs Normal")

    st.markdown("""
    **Observations** :
    - Les images COVID presentent generalement une intensite moyenne plus elevee
    - La variance est plus grande pour les cas pathologiques
    - Les filtres de contour (Canny, Sobel) revelent des structures differentes selon la pathologie
    """)


# ----- PAGE 4 : MODELISATION BASELINE -----
elif page == "Modelisation Baseline":
    st.title("Modelisation Baseline")

    st.markdown("""
    ## Approche

    - Images redimensionnees en **64x64** puis aplaties en vecteurs de **4 096 features**
    - Split train/test : **80/20** stratifie
    - Encodage des labels : COVID(0), Lung_Opacity(1), Normal(2), Viral_Pneumonia(3)

    ## Modeles entraines
    """)

    st.markdown("### 1. Random Forest (50 arbres)")
    st.markdown("""
    - **Accuracy** : 82%
    - **F1-score macro** : 0.81
    - Meilleur recall : Normal (0.92)
    - Plus faible recall : COVID (0.69) -- marge d'amelioration
    """)
    show_report_image("matrice_confusion2_RF.png", "Matrice de confusion -- Random Forest")

    st.markdown("---")

    st.markdown("### 2. K-Nearest Neighbors (KNN)")
    st.markdown("""
    - **k = 5** voisins, poids par distance, metrique euclidienne
    - Performance inferieure au Random Forest sur ces images
    - Sensible a la haute dimensionnalite (4 096 features)
    """)
    show_report_image("matrice_confusion_KNN.png", "Matrice de confusion -- KNN")

    st.markdown("""
    ---
    ### Bilan baseline

    | Modele | Accuracy | F1-macro | Point faible |
    |--------|----------|----------|------------|
    | Random Forest (50) | 82% | 0.81 | Recall COVID = 0.69 |
    | KNN (k=5) | < RF | < RF | Haute dimensionnalite |

    Le **Random Forest** est clairement superieur. L'etape suivante consiste a optimiser ses hyperparametres.
    """)


# ----- PAGE 5 : OPTIMISATION & METRIQUES -----
elif page == "Optimisation & Metriques":
    st.title("Optimisation & Metriques")

    models_loaded, label_mapping = load_models()

    st.markdown("""
    ## Optimisation par RandomizedSearchCV

    Recherche des meilleurs hyperparametres du Random Forest :
    - **n_estimators** : 150 (optimal)
    - **max_depth** : None (pas de limite)
    - **min_samples_split** : 2
    - **min_samples_leaf** : 1
    """)

    st.markdown("### Resultats du modele optimise")

    if metrics:
        comp = metrics['comparaison']
        results = pd.DataFrame({
            'Modele': list(comp.keys()),
            'Accuracy': [v['accuracy'] for v in comp.values()],
            'F1-macro': [v['f1_macro'] for v in comp.values()]
        })
        results = results.set_index('Modele')

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(results.style.highlight_max(axis=0, color='#2ecc71'))

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(results))
            width = 0.35
            ax.bar(x - width / 2, results['Accuracy'], width, label='Accuracy', color='#3498db')
            ax.bar(x + width / 2, results['F1-macro'], width, label='F1-macro', color='#e74c3c')
            ax.set_xticks(x)
            ax.set_xticklabels(results.index, rotation=15)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.set_title("Performance du modele")
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig)

        report = metrics.get('best_model_report', {})
        best_name = metrics.get('best_model_name', '')

        st.subheader(f"Rapport de classification : {best_name}")

        report_rows = []
        for cls in CLASSES:
            if cls in report:
                r = report[cls]
                report_rows.append({
                    'Classe': cls,
                    'Precision': round(r['precision'], 2),
                    'Recall': round(r['recall'], 2),
                    'F1-score': round(r['f1-score'], 2),
                    'Support': int(r['support'])
                })
        if report_rows:
            st.dataframe(pd.DataFrame(report_rows).set_index('Classe'))

        st.markdown("""
        **Analyse des resultats** :
        - **Viral Pneumonia** : excellente precision (0.91) et recall (0.96) -- classe bien distincte
        - **COVID** : recall plus faible (0.73) -- confusion avec Lung Opacity
        - **Normal vs Lung Opacity** : performances similaires, confusion frequente entre ces classes
        """)

        if 'hog_params' in metrics:
            st.subheader("Parametres HOG")
            for k, v in metrics['hog_params'].items():
                st.markdown(f"- `{k}` = `{v}`")
    else:
        st.warning("Fichier metrics.json non trouve. Executez les notebooks pour generer les metriques.")


# ----- PAGE 6 : DEEP LEARNING & BOOSTING -----
elif page == "Deep Learning & Boosting":
    st.title("Deep Learning & Boosting")

    st.markdown("""
    ## Architecture CNN

    Reseau convolutif entraine sur les images redimensionnees en **128x128** :

    | Couche | Details |
    |--------|---------|
    | Conv2D Block 1 | 32 filtres, BatchNorm, MaxPool 2x2 |
    | Conv2D Block 2 | 64 filtres, BatchNorm, MaxPool 2x2 |
    | Conv2D Block 3 | 128 filtres, BatchNorm, MaxPool 2x2 |
    | GlobalAveragePooling2D | Reduction spatiale |
    | Dense | 128 unites, Dropout 0.5 |
    | Sortie | Softmax, 4 classes |

    **Entrainement** :
    - Optimizer : Adam (lr = 1e-4)
    - Loss : Categorical Crossentropy
    - Epochs : 40 (Early Stopping patience=6)
    - Poids de classes equilibres pour gerer le desequilibre
    """)

    st.markdown("### Courbes d'entrainement")
    show_report_image("loss_accuracy_curves.png",
                      "Evolution du loss et de l'accuracy (train vs validation)")

    st.markdown("### Matrice de confusion CNN")
    show_report_image("confusion_cnn.png", "Confusion matrix -- CNN")

    st.markdown("---")

    st.markdown("""
    ## Gradient Boosting sur features CNN

    Approche hybride : extraction des features de la couche **GlobalAveragePooling2D** du CNN,
    puis classification par **Gradient Boosting** (300 estimateurs).

    Cette methode combine la puissance d'extraction de features du deep learning
    avec la robustesse des methodes d'ensemble.
    """)

    show_report_image("confusion_boosting.png", "Confusion matrix -- Gradient Boosting sur features CNN")

    st.markdown("---")

    st.markdown("""
    ## Interpretabilite : Grad-CAM

    **Grad-CAM** (Gradient Class Activation Map) permet de visualiser les regions
    de l'image sur lesquelles le CNN se concentre pour prendre sa decision.

    Ces cartes de chaleur sont essentielles pour la **confiance medicale** :
    elles montrent si le modele regarde les bonnes zones (poumons) ou s'il se base
    sur des artefacts.
    """)

    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            show_report_image(f"gradcam_{i}.png", f"Grad-CAM exemple {i+1}")

    st.markdown("""
    **Interpretation** :
    - Les zones rouges/jaunes indiquent les regions les plus influentes pour la prediction
    - Un bon modele doit se concentrer sur les **zones pulmonaires**
    - Les artefacts externes (bords, texte) ne doivent pas influencer la decision
    """)


# ----- PAGE 7 : PREDICTION -----
elif page == "Prediction":
    st.title("Prediction sur une radiographie")

    models, label_mapping = load_models()

    if not models:
        st.error("Aucun modele trouve dans le dossier models/. Executez d'abord les notebooks.")
        st.stop()

    if label_mapping:
        if isinstance(list(label_mapping.keys())[0], int):
            inv_mapping = label_mapping
        else:
            inv_mapping = {v: k for k, v in label_mapping.items()}
    else:
        inv_mapping = {0: 'COVID', 1: 'Lung_Opacity', 2: 'Normal', 3: 'Viral Pneumonia'}

    model_name = st.selectbox("Choisir le modele", sorted(models.keys()))
    model = models[model_name]

    st.markdown("---")

    tab1, tab2 = st.tabs(["Uploader une image", "Image aleatoire du dataset"])

    with tab1:
        uploaded = st.file_uploader("Choisir une radiographie (PNG, JPG)", type=['png', 'jpg', 'jpeg'])
        if uploaded:
            image = Image.open(uploaded)
            predict_image = image

    with tab2:
        if st.button("Tirer une image aleatoire"):
            cls = np.random.choice(CLASSES)
            img_dir = DATA_DIR / cls / "images"
            if img_dir.exists():
                imgs = list(img_dir.glob("*.png"))
                chosen = np.random.choice(imgs)
                st.session_state['random_img'] = str(chosen)
                st.session_state['random_cls'] = cls

        if 'random_img' in st.session_state:
            image = Image.open(st.session_state['random_img'])
            predict_image = image
            st.info(f"Vraie classe : **{st.session_state['random_cls']}**")

    if 'predict_image' in dir():
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(predict_image, caption="Radiographie", use_container_width=True)

        with col2:
            hog_features = extract_hog_features(predict_image)
            X_pred = hog_features.reshape(1, -1)

            prediction = model.predict(X_pred)[0]
            predicted_class = inv_mapping.get(prediction, str(prediction))

            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_pred)[0]
                st.markdown("**Probabilites par classe :**")

                fig, ax = plt.subplots(figsize=(6, 3))
                classes_sorted = [inv_mapping.get(i, str(i)) for i in range(len(probas))]
                colors = [CLASS_COLORS.get(c, '#95a5a6') for c in classes_sorted]
                bars = ax.barh(classes_sorted, probas, color=colors, edgecolor='white')
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probabilite")
                for bar, p in zip(bars, probas):
                    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                            f'{p:.1%}', va='center', fontweight='bold')
                ax.spines[['top', 'right']].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

            color = CLASS_COLORS.get(predicted_class, '#95a5a6')
            st.markdown(
                f"### Diagnostic : <span style='color:{color}'>{predicted_class}</span>",
                unsafe_allow_html=True
            )
