"""
qr2_ml.py
Kimberly Gu

QR2: Can social media addiction score predict relationship conflicts?

This script trains:
1. Linear Regression (baseline)
2. Ridge Regression (with multiple alpha values)

Feature:
    Addicted_Score

Target:
    Conflicts_Over_Social_Media

Evaluation Metrics:
    - Mean Absolute Error (MAE)
    - R^2 score
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def get_features_and_target(
        df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extracts feature matrix X and target vector y from the dataset.

    Parameters:
        df (pd.DataFrame): The full dataset.

    Returns:
        tuple:
            X (pd.DataFrame): Feature matrix containing Addicted_Score.
            y (pd.Series): Target variable Conflicts_Over_Social_Media.
    """
    X = df[["Addicted_Score"]]
    y = df["Conflicts_Over_Social_Media"]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.

    Returns:
        tuple:
            X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=163
    )


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[float, float]:
    """
    Trains a model and evaluates it using MAE and R^2.

    Parameters:
        model: A scikit-learn regression model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing targets.

    Returns:
        tuple:
            mae (float): Mean Absolute Error.
            r2 (float): R^2 score.
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mae, r2


def main() -> None:
    """
    Main execution function for QR2 machine learning analysis.
    Loads data, trains models, evaluates performance, and prints results.
    """
    # Read dataset inside main()
    df = pd.read_csv("Students Social Media Addiction.csv")

    # Prepare data
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Baseline: Linear Regression
    linear_model = LinearRegression()
    lin_mae, lin_r2 = evaluate_model(
        linear_model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("Baseline Model: Linear Regression")
    print(f"MAE: {lin_mae:.4f}")
    print(f"R^2: {lin_r2:.4f}")
    print()
    # Ridge Regression (Hyperparameter Tuning)
    alphas = [0.1, 1.0, 10.0, 100.0]

    best_alpha = None
    best_mae = None
    best_r2 = None

    print("Ridge Regression Results:")
    for alpha in alphas:
        ridge_model = Ridge(alpha=alpha)
        mae, r2 = evaluate_model(
            ridge_model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        print(f"alpha={alpha}: MAE={mae:.4f}, R^2={r2:.4f}")

        if best_mae is None or mae < best_mae:
            best_alpha = alpha
            best_mae = mae
            best_r2 = r2
    print()
    print("Best Ridge Model (Lowest MAE)")
    print(f"alpha: {best_alpha}")
    print(f"MAE: {best_mae:.4f}")
    print(f"R^2: {best_r2:.4f}")
    print()
    # Simple Conclusion
    print("Conclusion:")

    if best_mae < lin_mae and best_r2 > lin_r2:
        print("Ridge regression outperformed the baseline linear regression on both MAE and R^2.")
        print("This suggests that Addicted_Score provides meaningful predictive signal for relationship conflicts.")
    else:
        print("Ridge regression did not clearly outperform the baseline model.")
        print("While Addicted_Score shows some predictive relationship, its predictive strength may be limited when used alone.")


if __name__ == "__main__":
    main()
