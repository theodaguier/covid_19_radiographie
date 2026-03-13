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
    ├── models             <- Modeles entraines (RF HOG, label mapping)
    ├── notebooks          <- 5 notebooks du pipeline
    ├── references         <- Documentation et references
    ├── reports            <- Graphiques et figures generes par les notebooks
    └── src
        ├── features       <- Extraction de features
        ├── models         <- Entrainement et prediction
        ├── streamlit      <- Application Streamlit
        └── visualization  <- Scripts de visualisation
