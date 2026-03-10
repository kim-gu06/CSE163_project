'''
Isabella Le & Kimberly Gu
CSE 163
Final Project

This file was used to create our visualization for QR1
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def qr1() -> None:
    """
    This function was created to create a visualization for QR1 that
    analyzes the average social media addiciton score by each country
    """

    df = pd.read_csv('social_media.csv')

    # setting the theme
    sns.set_theme(style="darkgrid")

    # calculating the mean scores for each country in csv
    mean = df.groupby("Country")["Avg_Daily_Usage_Hours"].mean().reset_index()

    plt.figure(figsize=(15, 10))
    plt.scatter(mean["Avg_Daily_Usage_Hours"], mean["Country"])

    # labeling the visualization
    plt.title("Average Social Media Addiction Score by Country")
    plt.ylabel("Country")
    plt.yticks(fontsize=8, rotation=30)
    plt.xlabel("Average Hours Spent on Social Media Platforms")
    plt.show()


def main():
    """
    This function was created to run qr1 to create a visualization for our QR1
    """
    qr1()


if __name__ == "__main__":
    main()
