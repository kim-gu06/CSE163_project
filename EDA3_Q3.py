'''
Isabella Le & Kimberly Gu
CSE 163 Section AD & AG

This file is used to answer rq3 to find a correlation between
average hours slept per night and mental health score to
social media addiction
'''
# importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt

# import neural network library
import torch
import torch.nn as nn

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# importing in numpy
import numpy as np


def load_data(filename: str) -> pd.DataFrame:
    """
    Reads the CSV file and returns a pandas DataFrame.

    parameters: filename: str > path to csv file

    returns: pd.DataFrame > a processed pandas dataframe

    """
    # loads in and process it into pandas dataframe
    return pd.read_csv(filename)


def get_features_and_target_qr3(
        df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function is used to take in the necessary columns
    needed to answer QR3:

    df: a pandas dataframe

    returns: a tuple[np.ndarray, np.ndarray]
    x: feature matrix
    y: target variable array
    """

    # grabs the exact data we need from certain columns
    X = df[["Mental_Health_Score", "Sleep_Hours_Per_Night"]].values
    y = df["Addicted_Score"].values
    return X, y


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """
    Trains a model and evaluates it using MAE and R^2.

    Parameters:
        model: A scikit-learn regression model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing targets.

    Returns:
        tuple: [float, float, np.ndarray]
            mae (float): Mean Absolute Error.
            r2 (float): R^2 score.
            predictions: predicted values for the test data
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mae, r2, predictions


def split_data(
    X: np.ndarray,
    y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training and testing sets.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable.

    Returns:
        tuple: [np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


class mlpregres(nn.Module):
    """
    This a class that creates a 3-layered MLP (neural network) that is used for
    regression

    It aims to predict a student's social media addiction
    """
    def __init__(self):
        """
        The function intialized the neural network itself
        """
        super().__init__()

        # this creates a 3-layed neural network
        self.model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 4),
                                   nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Defines the forward pass for the neural networks

        parameters:
        x: torch.tensor > an input tensor containing values

        returns: torch.tensor > the predicted addiction
        scores from neural network
        """
        return self.model(x)


def train_neural_network(x_train, y_train, x_test) -> np.ndarray:
    """
    This function uses PyTorch in order to create and train
    a neural network using a training dataset

    It is trained using MAE, Adam Optimzier, and 200 training epochs

    parameters:
    x_train: np.ndarray > feature data
    y_train: np.ndarray > training target
    x_test: np.ndarray > testing feature data

    returns:
    np.ndarray > the predicted addiction scores for the dataset

    """
    # completing the pyTorch section for MLP for neural networks
    x_train_ten = torch.tensor(x_train, dtype=torch.float32)
    x_test_ten = torch.tensor(x_test, dtype=torch.float32)
    y_train_ten = torch.tensor(y_train, dtype=torch.float32)

    # creates the neural network
    ml_model = mlpregres()

    # this finds the lost & optimizer

    # meassures the prediction error tells us about performance of model
    criterion = nn.L1Loss()

    # updates the network weights to reduce the errors
    optimize = torch.optim.Adam(ml_model.parameters(), lr=0.01)

    # this creates the training loop for the machine learning
    epochs = 200
    for i in range(epochs):

        # clear any old gradients
        optimize.zero_grad()

        # makest the predictions
        output = ml_model(x_train_ten).flatten()

        # measures the loss/error
        loss = criterion(output, y_train_ten)

        # calculates n computes the gradients
        loss.backward()

        # adjusting the network parameter to ensure accuracy
        optimize.step()

        if i % 20 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}")

    # tests the neural network
    with torch.no_grad():
        y_pred_ml = ml_model(x_test_ten).numpy()

    return y_pred_ml


def main() -> None:
    """
    Main execution function for QR3 machine learning analysis.
    Loads data, trains models, evaluates performance, and
    prints results and visualizations
    """
    # Read dataset inside main()
    df = load_data("Students Social Media Addiction.csv")
    x, y = get_features_and_target_qr3(df)

    x_train, x_test, y_train, y_test = split_data(x, y)

    # scaler makes it easeier for ML model standardizing dataset
    scaler = StandardScaler()

    x_train_scale = scaler.fit_transform(x_train)
    x_test_scale = scaler.transform(x_test)

    # creating the linear regression
    linear_mo = LinearRegression()
    linear_mo.fit(x_train_scale, y_train)

    mae_linear, r2_linear, prediction_linear = evaluate_model(linear_mo,
                                                              x_train_scale,
                                                              y_train,
                                                              x_test_scale,
                                                              y_test)
    print("QR3: Results")
    print()
    print("PyTorch ML Results:")

    # runs the function to get results from our the neural_networks
    pred_ml = train_neural_network(x_train_scale, y_train, x_test_scale)

    # evalulates the results from the torch re
    mae_ml = mean_absolute_error(y_test, pred_ml)
    r2_ml = r2_score(y_test, pred_ml)

    # printing out the MAE AND R2 for neural netowrk
    print()
    print(f"Mean Average Error {mae_ml}, R2: {r2_ml}")
    print()

    # prints out results for the linear model
    print("The Linear Regression Model Results:")
    print(f"MAE: {mae_linear}, R: {r2_linear}")
    print()

    plt.figure()
    plt.scatter(y_test, pred_ml, label="ML Model")
    plt.scatter(y_test, prediction_linear, label="Linear Regression")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], color="black", label="Prediction")
    plt.xlabel("Actual Addiction Score")
    plt.ylabel("Predicted Addiction Score")
    plt.title("Actual Addiction Score VS. Predicted")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
