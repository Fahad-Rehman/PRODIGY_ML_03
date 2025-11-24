import os
import pandas as pd
from tqdm import tqdm

def load_image_paths(data_path, is_train=True):
    """
    Load image paths and labels from the dataset
    """
    image_paths = []
    labels = []
    
    if is_train:
        # For training data - filenames contain labels
        for filename in tqdm(os.listdir(data_path), desc="Loading images"):
            if filename.startswith('cat'):
                image_paths.append(os.path.join(data_path, filename))
                labels.append(0)  # 0 for cat
            elif filename.startswith('dog'):
                image_paths.append(os.path.join(data_path, filename))
                labels.append(1)  # 1 for dog
    else:
        # For test data - no labels in filenames
        for filename in tqdm(os.listdir(data_path), desc="Loading test images"):
            if filename.endswith('.jpg'):
                image_paths.append(os.path.join(data_path, filename))
                labels.append(-1)  # -1 for unknown
    
    return image_paths, labels

def create_dataframe(image_paths, labels):
    """
    Create a pandas DataFrame with image paths and labels
    """
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
    return df