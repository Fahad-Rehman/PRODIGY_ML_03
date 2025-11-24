# Cats vs Dogs Classification using SVM

## 🐱🐶 Project Overview
This project implements a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs using the Kaggle Cats vs Dogs dataset.

## Dataset
- **Source**: [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Training Images**: ~25,000 images (12,500 cats, 12,500 dogs)
- **Test Images**: ~12,500 images
- **Image Format**: JPG files

## Project Structure
```text
Task3/
├── data/
│ ├── train/ # Training images (cat..jpg, dog..jpg)
│ └── test/ # Test images (*.jpg)
├── src/
│ ├── data_loader.py # Load and organize image paths
│ ├── preprocess.py # Image preprocessing and feature extraction
│ ├── train_svm.py # SVM model training and evaluation
│ └── evaluate.py # Model evaluation utilities
├── models/ # Saved trained models
├── results/ # Training results and visualizations
├── main.py # Main pipeline script
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## 🚀 Installation & Setup

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd cats_vs_dogs
2. Create virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Download and extract the dataset
    + Download from Kaggle
    + Extract `train.zip` to `data/train/`
    + Extract `test.zip` to `data/test/`

## Usage
Run the complete pipeline:
    ```bash
    python main.py
    ```
## Individual Components:
+ Data Loading: src/data_loader.py
+ Preprocessing: src/preprocess.py
+ Model Training: src/train_svm.py
+ Evaluation: src/evaluate.py

## Image Preprocessing
+ Image resizing (64x64, 128x128, or 256x256 pixels)
+ RGB color space conversion
+ Pixel value normalization [0, 1]
+ Image flattening for SVM input

## Model Architecture
+ Algorithm: Support Vector Machine (SVM)
+ Kernel: RBF (Radial Basis Function)
+ Feature Extraction: Flattened pixel values
+ Validation: 80-20 train-validation split

## Performance
+ Expected Accuracy: 60-75% with full dataset
+ Training Time: 30-120 minutes (depending on image size)

## Results
The pipeline generates:

+ Confusion matrix
+ Classification report
+ Sample image visualizations
+ Training accuracy plots
+ Saved model file (.pkl)

# Configuration

## To Use Entire Dataset:
In `main.py`, change these lines:
```bash
sample_size = 10000
    X, y, valid_paths = prepare_data_for_svm(
    train_paths[:sample_size], 
    train_labels[:sample_size],
    target_size=(64, 64)
    )
```
Change to:
```bash
# Use entire dataset
X, y, valid_paths = prepare_data_for_svm(
    train_paths,  # Remove slicing -> use all paths
    train_labels, # Remove slicing -> use all labels
    target_size=(128, 128)  # You can increase size for better accuracy
)
```

## Additional Improvements for Better Accuracy:
1. Increase image size (in `main.py`):
    ```bash
    target_size=(128, 128)  # Instead of (64, 64)
    # or
    target_size=(256, 256)  # Even better but slower
    ```
2. Use better features (in `preprocess.py`):
    + Consider using HOG (Histogram of Oriented Gradients) features
    + Or deep learning features from pre-trained models

# Important Notes
+ Memory Requirements: Using all images with larger sizes requires significant RAM (8GB+ recommended)
+ Training Time: Full training may take 1-2 hours
+ Storage: Dataset is ~800MB when extracted

# Future Improvements
+ Implement data augmentation
+ Use transfer learning with pre-trained CNNs
+ Try different feature extraction methods
+ Hyperparameter tuning for SVM

# License
This project is for educational purposes as part of the Prodigy Infotech internship program.