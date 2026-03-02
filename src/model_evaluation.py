from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test, plot: bool = False):
    """Evaluate regression model performance.

    Parameters
    ----------
    model : estimator
        Trained scikit-learn-like estimator with ``predict``.
    X_test : array-like
        Features for testing.
    y_test : array-like
        True target values.
    plot : bool, default False
        If True, produces a scatter plot of actual vs. predicted values.

    Returns
    -------
    dict
        Dictionary containing mse, rmse, mae and r2.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.4f}")

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')
        plt.tight_layout()
        plt.show()

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
