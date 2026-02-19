"""
eda_plots.py

This script performs exploratory data analysis (EDA) on the
"Students Social Media Addiction.csv" dataset.

It generates and saves two visualizations:
1. Bubble plot of Addiction Score vs Conflicts Over Social Media
2. Scatter plot of Mental Health vs Sleep Hours, colored by Addiction Score
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Reads the dataset from a CSV file
    Returns pandas.DataFrame: The loaded dataset.
    """
    return pd.read_csv(csv_path)


def create_bubble_counts(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    """
    Counts frequency of each (x, y) pair for bubble plotting.

    Parameters:
        df (DataFrame): Input dataset
        x_col (str): X-axis column name
        y_col (str): Y-axis column name

    Returns:
        DataFrame: Aggregated counts for bubble sizes
    """
    counts = (
        df
        .groupby([x_col, y_col])
        .size()
        .reset_index(name="count")
    )
    return counts


def plot_addiction_vs_conflicts(df: pd.DataFrame) -> None:
    """
    Creates a bubble plot of Addiction Score vs Conflicts Over Social Media,
    bubble size represents frequency of identical (x, y) pairs.
    """
    x_col = "Addicted_Score"
    y_col = "Conflicts_Over_Social_Media"

    df_counts = create_bubble_counts(df, x_col, y_col)
    # adjust plot size
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df_counts,
        x=x_col,
        y=y_col,
        size="count",
        sizes=(50, 400),  # bubble size
        alpha=0.6  # transparency of points
    )
    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        scatter=False
    )
    plt.xlabel("Social Media Addiction Score")
    plt.ylabel("Relationship Conflicts Over Social Media")
    plt.title("Addiction Score vs Conflicts (Bubble Size = Frequency)")


def plot_mental_health_sleep(df: pd.DataFrame) -> None:
    """
    Creates a scatter plot of Mental Health Score vs Sleep Hours,
    colored by Addiction Score.
    """

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="Mental_Health_Score",
        y="Sleep_Hours_Per_Night",
        hue="Addicted_Score",
        palette="viridis"
    )

    plt.xlabel("Mental Health Score")
    plt.ylabel("Sleep Hours Per Night")
    plt.title("Mental Health, Sleep Duration, and Addiction Score")


def main() -> None:
    """
    Loads dataset, creates plots and saves images.
    """
    csv_path = "Students Social Media Addiction.csv"
    df = load_dataset(csv_path)
    # Set theme for aesthetics
    sns.set_theme(style="darkgrid")

    # Plot 1
    plot_addiction_vs_conflicts(df)
    plt.savefig("addiction_vs_conflicts_bubble.png",
                bbox_inches="tight")
    plt.close()
    # Plot 2
    plot_mental_health_sleep(df)
    plt.savefig("mental_health_sleep_addiction_scatter.png",
                bbox_inches="tight")
    plt.close()

    print("Images saved successfully.")


if __name__ == "__main__":
    main()
