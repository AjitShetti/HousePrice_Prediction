from src.data_preprocessing import load_data
from src.model_training import train_multiple_models, save_model
from src.model_evaluation import evaluate_model


def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_data()

    # Train models and get the best one
    best_model, results = train_multiple_models(X_train, X_test, y_train, y_test)

    # Evaluate the best model
    print("\nEvaluation of the best model:")
    metrics = evaluate_model(best_model, X_test, y_test)

    # Save the best model and scaler
    save_model(best_model, scaler)

    print("\nAll done!")


if __name__ == "__main__":
    main()
