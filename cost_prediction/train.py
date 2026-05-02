import os
import joblib
import preprocessing
import models

def run_training_pipeline():
    # 1. Define paths
    db_path = "../data/inventory.db"  # Ensure this points to your database file
    model_dir = "model"
    model_path = os.path.join(model_dir, "model.pkt")

    # 2. Load and Preprocess Data
    print("Loading data...")
    X_train, X_test, y_train, y_test = preprocessing.load_and_clean_data(db_path)

    # 3. Train all models
    print("Training models...")
    trained_models = models.get_trained_models(X_train, y_train)

    # 4. Evaluate and find the best model (Lowest MAE)
    best_model = None
    best_model_name = ""
    lowest_mae = float('inf')

    print("\n--- Evaluation Results ---")
    for name, model in trained_models.items():
        metrics = models.evaluate_model(model, X_test, y_test, name)
        
        # Check if this model is better than the current best
        if metrics['mae'] < lowest_mae:
            lowest_mae = metrics['mae']
            best_model = model
            best_model_name = name

    # 5. Save the best model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(best_model, model_path)
    
    print(f"WINNER: {best_model_name} with the lowest MAE of {lowest_mae:.2f}")
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    run_training_pipeline()