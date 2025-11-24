import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import joblib
import os

def train_svm_model(X, y, test_size=0.2, random_state=42):
    """
    Train SVM model on the image features
    """
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    
    # Initialize SVM classifier
    print("Training SVM classifier...")
    svm_classifier = SVC(
        kernel='rbf',        # Radial Basis Function kernel
        C=1.0,              # Regularization parameter
        gamma='scale',       # Kernel coefficient
        random_state=42,
        verbose=True
    )
    
    # Train the model
    svm_classifier.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = svm_classifier.predict(X_val)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    return svm_classifier, X_val, y_val, y_pred

def evaluate_model(model, X_val, y_val, y_pred, results_path="results"):
    """
    Evaluate the trained model and save results
    """
    os.makedirs(results_path, exist_ok=True)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Cat', 'Dog']))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cat', 'Dog'], 
                yticklabels=['Cat', 'Dog'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{results_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model
    model_path = "models/svm_cats_dogs.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def plot_training_results(y_val, y_pred, results_path="results"):
    """
    Plot training results
    """
    # Accuracy visualization
    accuracy = accuracy_score(y_val, y_pred)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(['Validation Accuracy'], [accuracy], color=['skyblue'])
    plt.ylim(0, 1)
    plt.title('Model Performance')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    class_distribution = [np.sum(y_val == 0), np.sum(y_val == 1)]
    plt.pie(class_distribution, labels=['Cats', 'Dogs'], autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
    plt.title('Class Distribution in Validation Set')
    
    plt.tight_layout()
    plt.savefig(f"{results_path}/training_results.png", dpi=300, bbox_inches='tight')
    plt.show()