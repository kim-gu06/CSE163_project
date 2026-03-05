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
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# --------- RQ2 --------------
def get_features_and_target_qr2(
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


# --------- RQ3 --------------
def get_features_and_target_qr3(
        df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:

    X = df[["Mental_Health_Score", "Sleep_Hours_Per_Night"]]
    y = df["Addicted_Score"]
    return X, y


# ------ Shared Utilities -----------
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

# ============================
# PyTorch Model for RQ3
# ============================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)


def train_pytorch_model(X_train, y_train, X_test, y_test):

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Fixed epochs (no hyperparameter tuning)
    for _ in range(200):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()

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
    print("QR2: Predict Relationship Conflicts")
    # Prepare data
    X2, y2 = get_features_and_target_qr2(df)
    X2_train, X2_test, y2_train, y2_test = split_data(X2, y2)

    # Baseline: Linear Regression
    linear_model = LinearRegression()
    lin_mae, lin_r2 = evaluate_model(
        linear_model,
        X2_train,
        y2_train,
        X2_test,
        y2_test
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
            X2_train,
            y2_train,
            X2_test,
            y2_test
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
    print()

    
    # RQ3
    print("RQ3: Predict Addiction Score")
    X3, y3 = get_features_and_target_qr3(df)
    X3_train, X3_test, y3_train, y3_test = split_data(X3, y3)

    # Baseline Linear Regression
    lin3 = LinearRegression()
    lin3_mae, lin3_r2 = evaluate_model(
        lin3,
        X3_train,
        y3_train,
        X3_test,
        y3_test
    )

    print("Baseline Linear Regression")
    print(f"MAE: {lin3_mae:.4f}")
    print(f"R^2: {lin3_r2:.4f}")
    print()

    # PyTorch MLP
    mlp_mae, mlp_r2 = train_pytorch_model(
        X3_train,
        y3_train,
        X3_test,
        y3_test
    )

    print("PyTorch MLP")
    print(f"MAE: {mlp_mae:.4f}")
    print(f"R^2: {mlp_r2:.4f}\n")

    print("Conclusion:")
    if mlp_mae < lin3_mae and mlp_r2 > lin3_r2:
        print("Neural network improved performance. Mental health and sleep provide predictive information.")
    else:
        print("Neural network did not clearly outperform baseline; predictive strength may be limited.")


if __name__ == "__main__":
    main()
