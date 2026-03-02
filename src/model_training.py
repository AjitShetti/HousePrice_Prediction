from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib
import os

# local import to avoid circular dependencies
from .model_evaluation import evaluate_model


def train_multiple_models(X_train, X_test, y_train, y_test, evaluate: bool = True):
    """Train several regression models and optionally evaluate them.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : array-like
        Output of preprocessing loader (see :func:`data_preprocessing.load_data`).
    evaluate : bool, default True
        If True, compute R² and other metrics on the test set for each model.

    Returns
    -------
    best_model : estimator
        The model with highest R² score on the test set.
    results : dict
        Mapping model name → evaluation metrics (if ``evaluate``) or fitted estimator.
    """

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        if evaluate:
            metrics = evaluate_model(model, X_test, y_test)
            results[name] = metrics
            print(f"{name}: R² = {metrics['r2']:.4f}\n")
        else:
            results[name] = model

    # select best model based on R² if metrics available
    if evaluate:
        best_model_name = max(results, key=lambda k: results[k]['r2'])
    else:
        best_model_name = next(iter(models))

    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} ")

    return best_model, results



def save_model(model, scaler, model_path="artifacts/model.pkl", scaler_path="artifacts/scaler.pkl"):
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
