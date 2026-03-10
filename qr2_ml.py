"""
qr2_ml.py

This file uses machine learning to answer the research question:
Can social media addiction score predict relationship conflicts?


This script also creates a visualization showing:
- the original data points
- the linear regression line
- the best ridge regression line

Run:
    python qr2_ml.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CSV_PATH: str = "social_media.csv"


def load_data(filename: str) -> pd.DataFrame:
    """Reads the CSV file and returns a DataFrame."""
    return pd.read_csv(filename)


def get_features_and_target(
    data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Selects the feature and target columns for QR2."""
    X: pd.DataFrame = data[["Addicted_Score"]]
    y: pd.Series = data["Conflicts_Over_Social_Media"]
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=0.20, random_state=163)


def train_linear_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    """Trains and returns a linear regression model."""
    model: LinearRegression = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression | Ridge,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[pd.Series, float, float]:
    """
    Evaluates a trained model using MAE and R^2.
    Returns predicted values, MAE, and R^2.
    """
    predictions = model.predict(X_test)
    mae: float = mean_absolute_error(y_test, predictions)
    r2: float = r2_score(y_test, predictions)
    return predictions, mae, r2


def train_best_ridge_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[Ridge, float, pd.Series, float, float]:
    """
    Trains multiple Ridge regression models with different alpha values.
    Returns the best model based on lowest MAE, along with its metrics.
    """
    alphas: list[float] = [0.1, 1, 10, 100]

    best_model: Ridge | None = None
    best_alpha: float | None = None
    best_predictions = None
    best_mae: float | None = None
    best_r2: float | None = None

    print("Ridge Regression Results")
    for alpha in alphas:
        model: Ridge = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        predictions, mae, r2 = evaluate_model(model, X_test, y_test)

        print(f"alpha = {alpha}")
        print(f"MAE = {mae:.4f}")
        print(f"R^2 = {r2:.4f}")
        print()

        if best_mae is None or mae < best_mae:
            best_model = model
            best_alpha = alpha
            best_predictions = predictions
            best_mae = mae
            best_r2 = r2

    assert best_model is not None
    assert best_alpha is not None
    assert best_predictions is not None
    assert best_mae is not None
    assert best_r2 is not None

    return best_model, best_alpha, best_predictions, best_mae, best_r2


def plot_results(
    data: pd.DataFrame,
    linear_model: LinearRegression,
    ridge_model: Ridge
) -> None:
    """
    Creates a scatter plot of the data and overlays regression lines
    for the linear regression model and the best ridge regression model.
    """
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=data,
        x="Addicted_Score",
        y="Conflicts_Over_Social_Media",
        alpha=0.6
    )

    x_vals = data[["Addicted_Score"]].sort_values(by="Addicted_Score")
    linear_y = linear_model.predict(x_vals)
    ridge_y = ridge_model.predict(x_vals)

    plt.plot(x_vals, linear_y, label="Linear Regression", linewidth=2)
    plt.plot(x_vals, ridge_y, label="Best Ridge Regression", linewidth=2)

    plt.title("Addiction Score vs Relationship Conflicts")
    plt.xlabel("Social Media Addiction Score")
    plt.ylabel("Number of Relationship Conflicts Over Social Media")
    plt.legend()
    plt.show()


def main() -> None:
    """Runs the full QR2 analysis."""
    print("Loading dataset...")
    data: pd.DataFrame = load_data(CSV_PATH)
    print("Dataset loaded successfully.\n")

    print("Selecting feature and target columns...")
    X, y = get_features_and_target(data)
    print("Feature: Addicted_Score")
    print("Target: Conflicts_Over_Social_Media\n")

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}\n")

    print("Training baseline linear regression model...")
    linear_model: LinearRegression = train_linear_model(X_train, y_train)
    _, linear_mae, linear_r2 = evaluate_model(linear_model, X_test, y_test)

    print("Baseline Linear Regression Results")
    print(f"MAE = {linear_mae:.4f}")
    print(f"R^2 = {linear_r2:.4f}\n")

    print("Training ridge regression models with different alpha values...")
    ridge_model, best_alpha, _, ridge_mae, ridge_r2 = train_best_ridge_model(
        X_train, y_train, X_test, y_test
    )

    print("Best Ridge Regression Model")
    print(f"Best alpha = {best_alpha}")
    print(f"MAE = {ridge_mae:.4f}")
    print(f"R^2 = {ridge_r2:.4f}\n")

    if ridge_mae < linear_mae:
        print(
            "Ridge regression performed better than the baseline"
            "based on MAE."
            )
    else:
        print(
            "Linear regression performed as well as or better than Ridge"
            "based on MAE."
            )

    plot_results(data, linear_model, ridge_model)


if __name__ == "__main__":
    main()
