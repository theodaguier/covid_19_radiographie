import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import ndimage
from skimage import io, feature
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
DATA_DIR = BASE_DIR / "data" / "COVID-19_Radiography_Dataset"

CLASSES = ['Normal', 'Lung_Opacity', 'COVID', 'Viral Pneumonia']
CLASS_COLORS = {
    'Normal': '#2ecc71',
    'Lung_Opacity': '#f39c12',
    'COVID': '#e74c3c',
    'Viral Pneumonia': '#9b59b6'
}


# ===== FEATURE EXTRACTION HOG (same as notebook cell 6) =====
def extract_hog_features(image: Image.Image) -> np.ndarray:
    """Extrait les features HOG comme dans le notebook."""
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


@st.cache_data
def load_sample_images(n_per_class=3):
    samples = {}
    for cls in CLASSES:
        img_dir = DATA_DIR / cls / "images"
        if img_dir.exists():
            imgs = sorted(img_dir.glob("*.png"))[:n_per_class]
            samples[cls] = [str(p) for p in imgs]
    return samples


@st.cache_data
def compute_class_features(n_per_class=50):
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
    "Aller a",
    ["Presentation", "Exploration des donnees", "Visualisations", "Modelisation", "Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Projet** : Analyse de radiographies pulmonaires COVID-19\n\n"
    "**Formation** : DataScientest\n\n"
    "**Dataset** : COVID-19 Radiography Database"
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
    ## Pipeline

    1. **Exploration & DataViz** -- Analyse de la distribution, intensite, textures
    2. **Feature engineering** -- Extraction de features HOG (Histogram of Oriented Gradients)
    3. **Modelisation** -- Random Forest sur features HOG
    4. **Prediction** -- Classification de nouvelles radiographies
    """)


# ----- PAGE 2 : EXPLORATION -----
elif page == "Exploration des donnees":
    st.title("Exploration des donnees")

    if metrics:
        counts = metrics['dataset_counts']
    else:
        counts = {'Normal': 10192, 'Lung_Opacity': 6012, 'COVID': 3616, 'Viral Pneumonia': 1345}
    total = sum(counts.values())

    st.subheader("Distribution des classes")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [CLASS_COLORS.get(c, '#95a5a6') for c in counts.keys()]
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f'{count}', ha='center', fontweight='bold')
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Repartition des images par categorie")
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)

    st.markdown(f"""
    **Observations** :
    - Le dataset est **desequilibre** : Normal ({counts.get('Normal',0)/total*100:.0f}%) domine largement
    - COVID ne represente que **{counts.get('COVID',0)/total*100:.0f}%** des images
    - Viral Pneumonia est la classe minoritaire (**{counts.get('Viral Pneumonia',0)/total*100:.0f}%**)
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

    with st.spinner("Calcul des features sur un echantillon..."):
        df_feat = compute_class_features(n_per_class=50)

    features = ['mean_intensity', 'std_intensity', 'contrast', 'entropy', 'gradient']
    feature_labels = {
        'mean_intensity': 'Intensite moyenne',
        'std_intensity': 'Ecart-type intensite',
        'contrast': 'Contraste normalise',
        'entropy': 'Entropie',
        'gradient': 'Gradient (Sobel)'
    }

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

    st.subheader("Intensite moyenne vs Ecart-type")
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in CLASSES:
        subset = df_feat[df_feat['label'] == cls]
        ax.scatter(subset['mean_intensity'], subset['std_intensity'],
                   c=CLASS_COLORS[cls], label=cls, alpha=0.7, s=40)
    ax.set_xlabel("Intensite moyenne")
    ax.set_ylabel("Ecart-type")
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)

    st.subheader("Matrice de correlation")
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df_feat[features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=[feature_labels[f] for f in features],
                yticklabels=[feature_labels[f] for f in features], ax=ax)
    ax.set_title("Correlations entre features")
    plt.tight_layout()
    st.pyplot(fig)


# ----- PAGE 4 : MODELISATION -----
elif page == "Modelisation":
    st.title("Modelisation")

    models, label_mapping = load_models()

    st.subheader("Approche")
    st.markdown("""
    **Features HOG (Histogram of Oriented Gradients)** extraites de chaque radiographie :
    - Capture les **contours et textures** de l'image
    - Parametres : `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)`
    - Beaucoup plus discriminant que de simples statistiques d'intensite

    **Modele** : Random Forest (100 arbres) entraine sur les features HOG
    """)

    st.subheader("Resultats")

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

        # Hyperparametres
        if 'hog_params' in metrics:
            st.subheader("Parametres HOG")
            for k, v in metrics['hog_params'].items():
                st.markdown(f"- `{k}` = `{v}`")
    else:
        st.warning("Fichier metrics.json non trouve. Executez le notebook pour generer les metriques.")

    st.subheader("Modeles disponibles")
    for name in sorted(models.keys()):
        st.write(f"- `{name}`")


# ----- PAGE 5 : PREDICTION -----
elif page == "Prediction":
    st.title("Prediction sur une radiographie")

    models, label_mapping = load_models()

    if not models:
        st.error("Aucun modele trouve dans le dossier models/. Executez d'abord les notebooks.")
        st.stop()

    # Inverse label mapping (label_mapping_hog is {int: str})
    if label_mapping:
        # label_mapping_hog format: {0: 'COVID', 1: 'Lung_Opacity', ...}
        if isinstance(list(label_mapping.keys())[0], int):
            inv_mapping = label_mapping
        else:
            inv_mapping = {v: k for k, v in label_mapping.items()}
    else:
        inv_mapping = {0: 'COVID', 1: 'Lung_Opacity', 2: 'Normal', 3: 'Viral Pneumonia'}

    # Model selection
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

    # Prediction
    if 'predict_image' in dir():
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(predict_image, caption="Radiographie", use_container_width=True)

        with col2:
            # Extract HOG features
            hog_features = extract_hog_features(predict_image)
            X_pred = hog_features.reshape(1, -1)

            # Predict
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
