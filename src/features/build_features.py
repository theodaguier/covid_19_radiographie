"""
Feature extraction module for COVID-19 radiography classification.

Extracts 5 statistical features from chest X-ray images:
- mean_intensity: Average pixel intensity
- std_intensity: Standard deviation of pixel intensity  
- contrast: Ratio of std to mean (coefficient of variation)
- entropy: Shannon entropy (texture complexity)
- gradient: Mean Sobel gradient (edge intensity)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy.stats import entropy as shannon_entropy


def extract_features(image_path: str) -> dict:
    """
    Extract 5 statistical features from a single image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with feature values
    """
    # Load image in grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)
    
    # 1. Mean intensity
    mean_intensity = np.mean(img_array)
    
    # 2. Standard deviation of intensity
    std_intensity = np.std(img_array)
    
    # 3. Contrast (coefficient of variation)
    contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
    
    # 4. Entropy (texture complexity)
    # Normalize to [0, 1] and compute histogram
    img_normalized = img_array / 255.0
    histogram, _ = np.histogram(img_normalized.flatten(), bins=256, range=(0, 1))
    histogram = histogram / histogram.sum()  # Normalize
    # Remove zeros to avoid log(0)
    histogram = histogram[histogram > 0]
    img_entropy = shannon_entropy(histogram, base=2)
    
    # 5. Gradient (edge intensity using Sobel)
    img_uint8 = img_array.astype(np.uint8)
    sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    
    return {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'contrast': contrast,
        'entropy': img_entropy,
        'gradient': gradient
    }


def build_feature_dataset(data_dir: str, output_path: str = None) -> pd.DataFrame:
    """
    Build a complete feature dataset from all images in the data directory.
    
    Args:
        data_dir: Path to COVID-19_Radiography_Dataset folder
        output_path: Optional path to save the CSV file
        
    Returns:
        DataFrame with features and labels for all images
    """
    # Define class directories
    classes = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
    
    features_list = []
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name, 'images')
        
        # Fallback if images are directly in class folder
        if not os.path.exists(class_dir):
            class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
            
        image_files = [f for f in os.listdir(class_dir) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {class_name}: {len(image_files)} images...")
        
        for i, filename in enumerate(image_files):
            image_path = os.path.join(class_dir, filename)
            
            try:
                features = extract_features(image_path)
                features['label'] = class_name
                features['image_path'] = image_path
                features_list.append(features)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(image_files)}")
    
    df = pd.DataFrame(features_list)
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved to {output_path}")
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df


def load_or_build_features(data_dir: str, cache_path: str = None) -> pd.DataFrame:
    """
    Load features from cache or build them if not available.
    
    Args:
        data_dir: Path to COVID-19_Radiography_Dataset folder
        cache_path: Path to cached CSV file
        
    Returns:
        DataFrame with features and labels
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)
    
    print("Building features from images...")
    return build_feature_dataset(data_dir, output_path=cache_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/COVID-19_Radiography_Dataset"
    
    output_path = "data/features.csv"
    
    df = build_feature_dataset(data_dir, output_path)
    print("\nFeature statistics:")
    print(df.describe())
