"""
Model prediction and evaluation module for COVID-19 radiography classification.

Provides functions for:
- Loading trained models
- Making predictions on new images
- Evaluating model performance
- Generating visualizations (confusion matrix, feature importance)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_recall_fscore_support
)
import joblib
from typing import List, Dict, Any, Optional

from src.features.build_features import extract_features

# Feature columns (must match training)
FEATURE_COLS = ['mean_intensity', 'std_intensity', 'contrast', 'entropy', 'gradient']


def load_model(model_name: str, models_dir: str = 'models') -> Dict[str, Any]:
    """
    Load a trained model with its scaler and label encoder.
    
    Args:
        model_name: Name of the model file (without extension)
        models_dir: Directory containing saved models
        
    Returns:
        Dictionary with model, scaler, and label_encoder
    """
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    
    return {
        'model': joblib.load(model_path),
        'scaler': joblib.load(scaler_path),
        'label_encoder': joblib.load(le_path)
    }


def predict_image(image_path: str, model_name: str = 'random_forest',
                  models_dir: str = 'models') -> Dict[str, Any]:
    """
    Predict the class of a single image.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the model to use
        models_dir: Directory containing saved models
        
    Returns:
        Dictionary with prediction and probabilities
    """
    # Load model components
    components = load_model(model_name, models_dir)
    model = components['model']
    scaler = components['scaler']
    le = components['label_encoder']
    
    # Extract features
    features = extract_features(image_path)
    X = np.array([[features[col] for col in FEATURE_COLS]])
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    result = {
        'predicted_class': le.inverse_transform([prediction])[0],
        'features': features
    }
    
    # Add probabilities if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_scaled)[0]
        result['probabilities'] = dict(zip(le.classes_, proba))
    
    return result


def predict_batch(image_paths: List[str], model_name: str = 'random_forest',
                  models_dir: str = 'models') -> pd.DataFrame:
    """
    Predict classes for multiple images.
    
    Args:
        image_paths: List of image paths
        model_name: Name of the model to use
        models_dir: Directory containing saved models
        
    Returns:
        DataFrame with predictions for all images
    """
    results = []
    
    for path in image_paths:
        try:
            pred = predict_image(path, model_name, models_dir)
            pred['image_path'] = path
            results.append(pred)
        except Exception as e:
            print(f"Error predicting {path}: {e}")
            continue
    
    return pd.DataFrame(results)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: List[str], 
                          title: str = 'Confusion Matrix',
                          save_path: Optional[str] = None,
                          figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Plot a confusion matrix with annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize for percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'{title}\n(Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'{title}\n(Normalized)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_feature_importance(model_path: str = 'models/random_forest.joblib',
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance from Random Forest model.
    
    Args:
        model_path: Path to saved Random Forest model
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    rf = joblib.load(model_path)
    
    importance_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest - Feature Importance')
    
    # Add value labels
    for bar, val in zip(bars, importance_df['importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return fig


def compare_models(results: Dict[str, Dict], 
                   label_encoder,
                   save_dir: str = 'reports/figures') -> pd.DataFrame:
    """
    Generate comparison visualizations for all models.
    
    Args:
        results: Dictionary with results from train_all_models
        label_encoder: LabelEncoder for class names
        save_dir: Directory to save figures
        
    Returns:
        Summary DataFrame
    """
    os.makedirs(save_dir, exist_ok=True)
    
    summary_data = []
    
    for name, res in results.items():
        eval_res = res['eval']
        train_res = res['train']
        
        # Plot confusion matrix for each model
        y_test = res.get('y_test', None)
        if 'y_pred' in eval_res:
            cm_path = os.path.join(save_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
            # Note: Would need y_test passed to results for this to work
        
        # Collect summary metrics
        summary_data.append({
            'Model': name,
            'CV Accuracy': f"{train_res['cv_mean']:.4f} ± {train_res['cv_std']:.4f}",
            'Test Accuracy': f"{eval_res['accuracy']:.4f}",
            'F1 Weighted': f"{eval_res['f1_weighted']:.4f}",
            'Precision': f"{eval_res['precision_weighted']:.4f}",
            'Recall': f"{eval_res['recall_weighted']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = os.path.join(save_dir, 'model_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Model comparison saved to {summary_path}")
    
    return summary_df


def generate_report(results: Dict[str, Dict], label_encoder,
                    output_path: str = 'reports/baseline_results.md') -> str:
    """
    Generate a markdown report of model results.
    
    Args:
        results: Dictionary with results from train_all_models
        label_encoder: LabelEncoder for class names
        output_path: Path to save the report
        
    Returns:
        Report content as string
    """
    report = "# Baseline Models - Results Report\n\n"
    report += "## Summary\n\n"
    report += "| Model | CV Accuracy | Test Accuracy | F1 Weighted |\n"
    report += "|-------|-------------|---------------|-------------|\n"
    
    for name, res in results.items():
        cv_acc = res['train']['cv_mean']
        cv_std = res['train']['cv_std']
        test_acc = res['eval']['accuracy']
        f1 = res['eval']['f1_weighted']
        report += f"| {name} | {cv_acc:.4f} ± {cv_std:.4f} | {test_acc:.4f} | {f1:.4f} |\n"
    
    report += "\n## Detailed Results\n\n"
    
    for name, res in results.items():
        report += f"### {name}\n\n"
        report += "**Classification Report:**\n\n"
        
        class_report = res['eval']['classification_report']
        report += "| Class | Precision | Recall | F1-Score | Support |\n"
        report += "|-------|-----------|--------|----------|--------|\n"
        
        for class_name in label_encoder.classes_:
            if class_name in class_report:
                cr = class_report[class_name]
                report += f"| {class_name} | {cr['precision']:.4f} | {cr['recall']:.4f} | {cr['f1-score']:.4f} | {int(cr['support'])} |\n"
        
        report += "\n"
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Example: predict a single image
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = predict_image(image_path)
        print(f"Prediction: {result['predicted_class']}")
        if 'probabilities' in result:
            print("Probabilities:")
            for class_name, prob in sorted(result['probabilities'].items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {prob:.4f}")
    else:
        print("Usage: python predict_model.py <image_path>")
