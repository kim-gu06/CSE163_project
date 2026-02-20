'''
Kimberly Gu & Isabella Le
Section AD & AG
CSE 163

This file is used to make our visualization
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv('Students Social Media Addiction.csv')

# Set theme
sns.set_theme(style="darkgrid")

# Keep only needed columns
cols = ["Addicted_Score", "Conflicts_Over_Social_Media"]
df_clean = df[cols].dropna()

# Count duplicate (x, y) pairs
df_counts = (
    df_clean
    .groupby(["Addicted_Score", "Conflicts_Over_Social_Media"])
    .size()
    .reset_index(name="count")
)

# plt.figure(figsize=(8, 6))

# Bubble scatter (size = frequency)
sns.scatterplot(
    data=df_counts,
    x="Addicted_Score",
    y="Conflicts_Over_Social_Media",
    size="count",
    sizes=(50, 400),
    alpha=0.6,  # Bubble color
)

# Regression line using original (non-aggregated) data
sns.regplot(
    data=df_clean,
    x="Addicted_Score",
    y="Conflicts_Over_Social_Media",
    scatter=False
)

plt.xlabel("Social Media Addiction Score")
plt.ylabel("Number of Relationship Conflicts Over Social Media")
plt.title("Addiction Score vs Relationship Conflicts "
          "(Bubble Size = Frequency)")
plt.show()


# visualization two
# Select relevant columns and drop missing values
cols = ["Mental_Health_Score", "Sleep_Hours_Per_Night", "Addicted_Score"]
df_clean = df[cols].dropna()

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_clean,
    x="Mental_Health_Score",
    y="Sleep_Hours_Per_Night",
    hue="Addicted_Score",
    palette="viridis",   # color for visualization
)

plt.title("Mental Health, Sleep Duration, and Social Media Addiction")
plt.xlabel("Mental Health Score")
plt.ylabel("Sleep Hours Per Night")
plt.show()
