from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib
import os


def train_multiple_models(X_train, X_test, y_train, y_test):
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
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = round(r2, 4)
        print(f"{name}: R² = {r2:.4f}")

    # Pick best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name} with R² = {results[best_model_name]:.4f}")
    return best_model, results


def save_model(model, scaler, model_path="artifacts/model.pkl", scaler_path="artifacts/scaler.pkl"):
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
