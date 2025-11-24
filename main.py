from src.data_loader import load_image_paths, create_dataframe
from src.preprocess import prepare_data_for_svm, show_sample_images
from src.train_svm import train_svm_model, evaluate_model, plot_training_results
import os

def main():
    print("🐱🐶 Cats vs Dogs Classifier using SVM")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Load data
    print("\nStep 1: Loading image paths...")
    train_paths, train_labels = load_image_paths('data/train')
    
    # Show sample images
    print("\nShowing sample images...")
    show_sample_images(train_paths[:5], train_labels[:5])
    
    # Step 2: Preprocess data (use smaller subset for testing)
    print("\nStep 2: Preprocessing images...")
    
    sample_size = 10000
    X, y, valid_paths = prepare_data_for_svm(
    train_paths[:sample_size], 
    train_labels[:sample_size],
    target_size=(64, 64)
    )
    
    
    # Step 3: Train SVM model
    print("\nStep 3: Training SVM model...")
    model, X_val, y_val, y_pred = train_svm_model(X, y)
    
    # Step 4: Evaluate model
    print("\nStep 4: Evaluating model...")
    evaluate_model(model, X_val, y_val, y_pred)
    plot_training_results(y_val, y_pred)
    
    print("\nTraining completed!")
    print(f"Model saved in 'models/' folder")
    print(f"Results saved in 'results/' folder")

if __name__ == "__main__":
    main()