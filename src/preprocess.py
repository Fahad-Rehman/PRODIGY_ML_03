import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    """
    Load an image and preprocess it for SVM
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Resize to standard size
        image = cv2.resize(image, target_size)
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        image = image / 255.0
        
        return image
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def extract_features(images):
    """
    Extract features from images for SVM
    Simple approach: flatten the image
    """
    features = []
    for image in images:
        if image is not None:
            # Flatten the image to 1D array
            flattened = image.flatten()
            features.append(flattened)
    
    return np.array(features)

def show_sample_images(image_paths, labels, num_samples=5):
    """
    Display sample images to verify loading
    """
    plt.figure(figsize=(15, 5))
    
    for i in range(min(num_samples, len(image_paths))):
        image = load_and_preprocess_image(image_paths[i])
        if image is not None:
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(image)
            plt.title(f"Label: {'Cat' if labels[i] == 0 else 'Dog'}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_images.png')
    

def prepare_data_for_svm(image_paths, labels, target_size=(64, 64)):
    """
    Main function to prepare data for SVM training
    """
    print("Loading and preprocessing images...")
    images = []
    valid_labels = []
    valid_paths = []
    
    for i, path in enumerate(image_paths):
        image = load_and_preprocess_image(path, target_size)
        if image is not None:
            images.append(image)
            valid_labels.append(labels[i])
            valid_paths.append(path)
    
    print(f"Successfully loaded {len(images)} images")
    
    # Extract features
    print("Extracting features...")
    X = extract_features(images)
    y = np.array(valid_labels)
    
    print(f"Feature shape: {X.shape}")
    
    return X, y, valid_paths