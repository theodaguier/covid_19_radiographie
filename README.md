COVID 19 RADIOGRAPHIE
==============================

Classification automatique de radiographies thoraciques pour la detection du COVID-19 par Machine Learning et Deep Learning.

## Dataset

**COVID-19 Radiography Database** -- 21 165 images reparties en 4 classes :
- Normal
- COVID
- Lung Opacity
- Viral Pneumonia

## Installation

```bash
pip install -r requirements.txt
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01-exploration-dataviz` | Analyse exploratoire et visualisation des donnees |
| `02-preprocessing-feature-engineering` | Pretraitement et extraction de features |
| `03-modelisation-baseline` | Random Forest et KNN baseline |
| `04-optimisation-metriques` | Optimisation hyperparametres et analyse des metriques |
| `05-deep-learning-boosting` | CNN, Gradient Boosting et Grad-CAM |
| `06-transfer-learning` | Transfer Learning (VGG16, EfficientNetB0, ResNet50), fine-tuning 2 phases |
| `07-interpretabilite` | Interpretabilite : Grad-CAM (3 modeles) + SHAP (meilleur modele) |
| `08-comparaison` | Comparaison des modeles (F1-macro, courbes d'apprentissage, sur-apprentissage) |
| `Rattrapage (2).ipynb` | Notebook propre d'orchestration : VGG16, ResNet50 masque, comparaison et Grad-CAM de rattrapage |

> Les notebooks 06-08 s'executent de preference sur **Google Colab (GPU)**.
> Adapter `PROJECT_DIR` dans la cellule d'amorcage (chemin Google Drive).
> Le notebook de rattrapage ne duplique plus la logique : il appelle les modules `src`.

## Rapport de synthese

Le rapport complet repondant aux attendus du jury est dans
`reports/RAPPORT_SYNTHESE.md` (export PDF : `reports/RAPPORT_SYNTHESE.pdf`).

## Streamlit

Lancer l'application de demonstration :

```bash
streamlit run src/streamlit/app.py
```

## Organisation du projet

    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── data               <- Dataset (non versionne)
    ├── models             <- Modeles entraines
    │   └── transfer       <- Poids des modeles Transfer Learning (.keras, non versionnes)
    ├── notebooks          <- 8 notebooks du pipeline
    ├── references         <- Documentation et references
    ├── reports            <- Figures, metriques, splits, historiques, rapport de synthese
    │   ├── splits         <- Split stratifie fige train/val/test (CSV)
    │   ├── history        <- Historiques d'entrainement (CSV) -> courbes d'apprentissage
    │   ├── metrics        <- Metriques par modele (JSON) + table de comparaison
    │   └── figures        <- Confusions, ROC/PR, courbes, Grad-CAM, SHAP
    └── src
        ├── data           <- Pipeline tf.data (split, RGB, augmentation, poids de classe)
        ├── features       <- Extraction de features
        ├── models         <- Transfer learning, evaluation, comparaison, hybride
        ├── streamlit      <- Application Streamlit
        ├── utils          <- Detection environnement (Colab/local), GPU, mixed precision
        └── visualization  <- Visualisations + interpretabilite (Grad-CAM, SHAP)
