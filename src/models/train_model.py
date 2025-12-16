"""
Model training module for COVID-19 radiography classification.

Implements 3 baseline classifiers:
1. Logistic Regression (simple baseline)
2. Random Forest (ensemble method)
3. Support Vector Machine (robust classifier)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import joblib
from typing import Dict, Tuple, Any

# Feature columns
FEATURE_COLS = ['mean_intensity', 'std_intensity', 'contrast', 'entropy', 'gradient']


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Prepare data for training: encode labels, split, and scale features.
    
    Args:
        df: DataFrame with features and labels
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    X = df[FEATURE_COLS].values
    y = df['label'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of baseline models with balanced class weights.
    """
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            random_state=42,
            probability=True
        )
    }


def train_model(model, X_train: np.ndarray, y_train: np.ndarray, 
                cv_folds: int = 5) -> Dict[str, Any]:
    """
    Train a single model with cross-validation.
    
    Args:
        model: Sklearn model instance
        X_train: Training features (scaled)
        y_train: Training labels
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with trained model and CV scores
    """
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    return {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                   label_encoder: LabelEncoder) -> Dict[str, Any]:
    """
    Evaluate a trained model on test set.
    
    Args:
        model: Trained sklearn model
        X_test: Test features (scaled)
        y_test: Test labels
        label_encoder: LabelEncoder for class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        ),
        'y_pred': y_pred
    }


def train_all_models(df: pd.DataFrame = None, data_path: str = None,
                     save_dir: str = 'models', sample_size: int = None) -> Dict[str, Any]:
    """
    Train all baseline models and save results.
    
    Args:
        df: DataFrame with features (optional if data_path provided)
        data_path: Path to features CSV file
        save_dir: Directory to save trained models
        sample_size: Optional sample size for quick testing
        
    Returns:
        Dictionary with results for all models
    """
    # Load data if not provided
    if df is None:
        if data_path is None:
            raise ValueError("Either df or data_path must be provided")
        df = pd.read_csv(data_path)
    
    # Sample for quick testing
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using sample of {sample_size} instances for testing")
    
    print(f"Dataset size: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, le = prepare_data(df)
    
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples\n")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save scaler and label encoder
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(save_dir, 'label_encoder.joblib'))
    
    # Train and evaluate all models
    results = {}
    models = get_models()
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train
        train_result = train_model(model, X_train, y_train)
        print(f"  CV Accuracy: {train_result['cv_mean']:.4f} (+/- {train_result['cv_std']:.4f})")
        
        # Evaluate
        eval_result = evaluate_model(train_result['model'], X_test, y_test, le)
        print(f"  Test Accuracy: {eval_result['accuracy']:.4f}")
        print(f"  F1 Weighted: {eval_result['f1_weighted']:.4f}")
        
        # Save model
        model_path = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.joblib")
        joblib.dump(train_result['model'], model_path)
        print(f"  Model saved to {model_path}\n")
        
        # Store results
        results[name] = {
            'train': train_result,
            'eval': eval_result,
            'model_path': model_path
        }
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY - BASELINE MODELS COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'CV Acc':<12} {'Test Acc':<12} {'F1 Weighted':<12}")
    print("-" * 60)
    
    for name, res in results.items():
        cv_acc = res['train']['cv_mean']
        test_acc = res['eval']['accuracy']
        f1 = res['eval']['f1_weighted']
        print(f"{name:<25} {cv_acc:.4f}       {test_acc:.4f}       {f1:.4f}")
    
    print("=" * 60)
    
    return results


def get_feature_importance(model_path: str = 'models/random_forest.joblib') -> pd.DataFrame:
    """
    Get feature importance from Random Forest model.
    
    Args:
        model_path: Path to saved Random Forest model
        
    Returns:
        DataFrame with feature importances
    """
    rf = joblib.load(model_path)
    
    importance_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == "__main__":
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/features.csv"
    
    if os.path.exists(data_path):
        results = train_all_models(data_path=data_path)
    else:
        print(f"Error: Features file not found at {data_path}")
        print("Run build_features.py first to generate features.")
