import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import ndimage
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import os

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
DATA_DIR = BASE_DIR / "data" / "COVID-19_Radiography_Dataset"

CLASSES = ['Normal', 'Lung_Opacity', 'COVID', 'Viral Pneumonia']
CLASS_COLORS = {
    'Normal': '#2ecc71',
    'Lung_Opacity': '#f39c12',
    'COVID': '#e74c3c',
    'Viral Pneumonia': '#9b59b6'
}


# ===== FEATURE EXTRACTION (same as notebook) =====
def extract_features(image: Image.Image) -> pd.DataFrame:
    """Extrait les 5 features utilisées par le modèle."""
    img = image.convert('L')
    arr = np.array(img, dtype=np.float32)

    mean_int = arr.mean()
    std_int = arr.std()
    contrast = std_int / (mean_int + 1e-8)
    entropy = -np.sum((arr / 255) ** 2 * np.log((arr / 255) ** 2 + 1e-8))
    gradient = ndimage.sobel(arr).std()

    return pd.DataFrame([{
        'mean_intensity': mean_int,
        'std_intensity': std_int,
        'contrast': contrast,
        'entropy': entropy,
        'gradient': gradient
    }])


# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    models = {}
    label_mapping = None

    for f in MODELS_DIR.glob("*.pkl"):
        name = f.stem
        obj = joblib.load(f)
        if name == 'label_mapping':
            label_mapping = obj
        else:
            models[name] = obj

    return models, label_mapping


@st.cache_data
def load_dataset_stats():
    """Charge les stats du dataset."""
    counts = {'Normal': 10192, 'Lung_Opacity': 6012, 'COVID': 3616, 'Viral Pneumonia': 1345}
    total = sum(counts.values())
    return counts, total


@st.cache_data
def load_sample_images(n_per_class=3):
    """Charge quelques images exemples par classe."""
    samples = {}
    for cls in CLASSES:
        img_dir = DATA_DIR / cls / "images"
        if img_dir.exists():
            imgs = sorted(img_dir.glob("*.png"))[:n_per_class]
            samples[cls] = [str(p) for p in imgs]
    return samples


@st.cache_data
def compute_class_features(n_per_class=50):
    """Calcule les features sur un échantillon pour les visualisations."""
    rows = []
    for cls in CLASSES:
        img_dir = DATA_DIR / cls / "images"
        if not img_dir.exists():
            continue
        imgs = sorted(img_dir.glob("*.png"))[:n_per_class]
        for p in imgs:
            img = Image.open(p).convert('L')
            arr = np.array(img, dtype=np.float32)
            mean_int = arr.mean()
            std_int = arr.std()
            rows.append({
                'label': cls,
                'mean_intensity': mean_int,
                'std_intensity': std_int,
                'contrast': std_int / (mean_int + 1e-8),
                'entropy': -np.sum((arr / 255) ** 2 * np.log((arr / 255) ** 2 + 1e-8)),
                'gradient': ndimage.sobel(arr).std()
            })
    return pd.DataFrame(rows)


# ===== SIDEBAR =====
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Aller à",
    ["Présentation", "Exploration des données", "Visualisations", "Modélisation", "Prédiction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Projet** : Analyse de radiographies pulmonaires COVID-19\n\n"
    "**Formation** : DataScientest\n\n"
    "**Dataset** : COVID-19 Radiography Database"
)

# ===== PAGES =====

# ----- PAGE 1 : PRÉSENTATION -----
if page == "Présentation":
    st.title("Analyse de radiographies pulmonaires COVID-19")

    st.markdown("""
    ## Contexte

    La pandémie de COVID-19 a mis en lumière le besoin d'outils de **diagnostic rapide et fiable**.
    L'analyse automatique de **radiographies thoraciques** par Machine Learning permet d'assister
    les professionnels de santé dans la détection de pathologies pulmonaires.

    ## Objectif

    Développer un modèle de classification capable de distinguer **4 types de radiographies** :

    - **Normal** — Poumons sains
    - **COVID** — Infection COVID-19
    - **Lung Opacity** — Opacité pulmonaire (pneumonie bactérienne, etc.)
    - **Viral Pneumonia** — Pneumonie virale (non-COVID)

    ## Dataset

    Le **COVID-19 Radiography Dataset** contient **21 165 images** réparties en 4 classes.
    """)

    counts, total = load_dataset_stats()
    col1, col2, col3, col4 = st.columns(4)
    for col, (cls, count) in zip([col1, col2, col3, col4], counts.items()):
        col.metric(cls, f"{count:,}", f"{count/total*100:.1f}%")

    st.markdown("""
    ## Pipeline

    1. **Exploration & DataViz** — Analyse de la distribution, intensité, textures
    2. **Preprocessing** — Extraction de 5 features (intensité, contraste, entropie, gradient)
    3. **Modélisation** — Random Forest, Gradient Boosting, comparaison de 6 modèles
    4. **Optimisation** — RandomizedSearchCV sur les 2 meilleurs modèles
    """)


# ----- PAGE 2 : EXPLORATION -----
elif page == "Exploration des données":
    st.title("Exploration des données")

    counts, total = load_dataset_stats()

    st.subheader("Distribution des classes")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [CLASS_COLORS[c] for c in counts.keys()]
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f'{count}', ha='center', fontweight='bold')
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Répartition des images par catégorie")
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)

    st.markdown(f"""
    **Observations** :
    - Le dataset est **déséquilibré** : Normal ({counts['Normal']/total*100:.0f}%) domine largement
    - COVID ne représente que **{counts['COVID']/total*100:.0f}%** des images
    - Viral Pneumonia est la classe minoritaire (**{counts['Viral Pneumonia']/total*100:.0f}%**)
    """)

    st.subheader("Exemples d'images par classe")
    samples = load_sample_images(n_per_class=3)
    for cls in CLASSES:
        st.markdown(f"**{cls}**")
        if cls in samples:
            cols = st.columns(3)
            for i, img_path in enumerate(samples[cls]):
                img = Image.open(img_path)
                cols[i].image(img, use_container_width=True)


# ----- PAGE 3 : VISUALISATIONS -----
elif page == "Visualisations":
    st.title("Visualisations des features")

    with st.spinner("Calcul des features sur un échantillon..."):
        df_feat = compute_class_features(n_per_class=50)

    features = ['mean_intensity', 'std_intensity', 'contrast', 'entropy', 'gradient']
    feature_labels = {
        'mean_intensity': 'Intensité moyenne',
        'std_intensity': 'Écart-type intensité',
        'contrast': 'Contraste normalisé',
        'entropy': 'Entropie',
        'gradient': 'Gradient (Sobel)'
    }

    # Boxplots
    st.subheader("Distribution des features par classe")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, feat in zip(axes, features):
        palette = [CLASS_COLORS[c] for c in CLASSES]
        sns.boxplot(data=df_feat, x='label', y=feat, ax=ax, palette=palette, order=CLASSES)
        ax.set_title(feature_labels[feat], fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Scatter
    st.subheader("Intensité moyenne vs Écart-type")
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in CLASSES:
        subset = df_feat[df_feat['label'] == cls]
        ax.scatter(subset['mean_intensity'], subset['std_intensity'],
                   c=CLASS_COLORS[cls], label=cls, alpha=0.7, s=40)
    ax.set_xlabel("Intensité moyenne")
    ax.set_ylabel("Écart-type")
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)

    # Correlation matrix
    st.subheader("Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df_feat[features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=[feature_labels[f] for f in features],
                yticklabels=[feature_labels[f] for f in features], ax=ax)
    ax.set_title("Corrélations entre features")
    plt.tight_layout()
    st.pyplot(fig)


# ----- PAGE 4 : MODÉLISATION -----
elif page == "Modélisation":
    st.title("Modélisation")

    models, label_mapping = load_models()

    st.subheader("Approche")
    st.markdown("""
    **5 features extraites** de chaque radiographie :
    - `mean_intensity` — intensité moyenne des pixels
    - `std_intensity` — écart-type (contraste brut)
    - `contrast` — contraste normalisé
    - `entropy` — complexité de texture
    - `gradient` — intensité des contours (Sobel)

    **Modèles entraînés et comparés** :
    | Modèle | Description |
    |--------|-------------|
    | Logistic Regression | Baseline linéaire |
    | KNN | K plus proches voisins |
    | SVM (RBF) | Support Vector Machine |
    | Decision Tree | Arbre de décision |
    | Random Forest | Ensemble d'arbres (bagging) |
    | Gradient Boosting | Ensemble d'arbres (boosting) |
    """)

    st.subheader("Résultats comparatifs")

    results = pd.DataFrame({
        'Modèle': ['RF Baseline', 'RF Optimisé', 'GB Optimisé'],
        'Accuracy': [0.500, 0.543, 0.521],
        'F1-macro': [0.495, 0.526, 0.516]
    })
    results = results.set_index('Modèle')

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
        ax.set_title("Comparaison des modèles")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)

    st.subheader("Meilleur modèle : Random Forest Optimisé")
    st.markdown("""
    **Hyperparamètres optimisés** via `RandomizedSearchCV` (40 itérations, 5-fold CV) :
    - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

    **Performance sur le test set** :
    """)

    report_data = {
        'Classe': ['Normal', 'Lung_Opacity', 'COVID', 'Viral Pneumonia'],
        'Precision': [0.53, 0.50, 0.62, 0.47],
        'Recall': [0.64, 0.40, 0.72, 0.37],
        'F1-score': [0.58, 0.44, 0.67, 0.41]
    }
    st.dataframe(pd.DataFrame(report_data).set_index('Classe'))

    st.markdown("""
    **Observations** :
    - Le COVID est la classe **la mieux détectée** (F1 = 0.67) grâce à des patterns radiographiques distincts
    - Lung Opacity et Viral Pneumonia sont souvent confondus (aspects radiologiques similaires)
    - Les 5 features d'intensité/texture capturent des informations utiles mais limitées
    """)

    st.subheader("Modèles disponibles")
    for name in sorted(models.keys()):
        st.write(f"- `{name}`")


# ----- PAGE 5 : PRÉDICTION -----
elif page == "Prédiction":
    st.title("Prediction sur une radiographie")

    models, label_mapping = load_models()

    if not models:
        st.error("Aucun modèle trouvé dans le dossier models/. Exécutez d'abord les notebooks.")
        st.stop()

    # Inverse label mapping
    if label_mapping:
        inv_mapping = {v: k for k, v in label_mapping.items()}
    else:
        inv_mapping = {0: 'Normal', 1: 'Lung_Opacity', 2: 'COVID', 3: 'Viral Pneumonia'}

    # Model selection
    model_name = st.selectbox("Choisir le modèle", sorted(models.keys()))
    model = models[model_name]

    st.markdown("---")

    # Upload or sample
    tab1, tab2 = st.tabs(["Uploader une image", "Image aleatoire du dataset"])

    with tab1:
        uploaded = st.file_uploader("Choisir une radiographie (PNG, JPG)", type=['png', 'jpg', 'jpeg'])
        if uploaded:
            image = Image.open(uploaded)
            predict_image = image

    with tab2:
        if st.button("Tirer une image aléatoire"):
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

    # Prediction
    if 'predict_image' in dir():
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(predict_image, caption="Radiographie", use_container_width=True)

        with col2:
            features_df = extract_features(predict_image)

            st.markdown("**Features extraites :**")
            st.dataframe(features_df.T.rename(columns={0: 'Valeur'}).style.format("{:.2f}"))

            # Predict
            prediction = model.predict(features_df)[0]
            predicted_class = inv_mapping.get(prediction, str(prediction))

            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(features_df)[0]
                st.markdown("**Probabilités par classe :**")

                fig, ax = plt.subplots(figsize=(6, 3))
                classes_sorted = [inv_mapping.get(i, str(i)) for i in range(len(probas))]
                colors = [CLASS_COLORS.get(c, '#95a5a6') for c in classes_sorted]
                bars = ax.barh(classes_sorted, probas, color=colors, edgecolor='white')
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probabilité")
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
