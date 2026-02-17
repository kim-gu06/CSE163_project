"""
CSE 163
Kimberly Gu
This script reads the Student Social Media Addition datase
and print summary statistics
"""
import pandas as pd


def total(n):
    """
    Takes in a number n (int) and returns the sum of the numbers from
    0(inclusive) to n (inclusive). If n is negative, returns None.
    """
    if n < 0:
        return None
    else:
        result = 0
        for i in range(n + 1):
            result += i
        return result


def median(numbers):
    """
    Given a list of numbers, returns the median as float or integer.
    Assumptions:
        - The list is sorted from least to greatest
        - The list contains at least five numbers
    """
    n = len(numbers)

    result = None
    if len(numbers) % 2 == 1:
        result = numbers[n // 2]
    else:
        upper_num = numbers[n // 2]
        lower_num = numbers[n // 2 - 1]
        result = (upper_num + lower_num) / 2

    return result


def five_number_summary(numbers):
    """
    Given a sorted list of numbers, returns a tuple of the five-number
    summary: the minimum, first quartile, median, third quartile,
    and maximum.

    Assumptions:
        - The list is sorted from least to greatest
        - The list contains at least five numbers
    """
    n = len(numbers)

    if n % 2 == 0:
        lower_half = numbers[:n // 2]
        upper_half = numbers[n // 2:]
    else:
        lower_half = numbers[:n // 2]
        upper_half = numbers[n // 2 + 1:]

    return (numbers[0], median(lower_half),
            median(numbers), median(upper_half), numbers[-1])


def num_outliers(numbers):
    """
    Takes a sorted list, returns the numbers of data points that would
    be considered outliters based on the interquartile range.

    Assumptions:
        - The list is sorted from least to greatest
        - The list contains at least five numbers
        - The five-number summary is used to compute quartiles
    """
    # Unpack tuples to get their contents
    max, q1, med, q3, min = five_number_summary(numbers)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    count = 0
    for num in numbers:
        if num < lower_bound or num > upper_bound:
            count += 1
    return count


def text_normalize(text):
    """
    Takes a string and returns a new string that keeps only alphabetical
    characters(ignore whitespace, number, non-alphabet characters) and turns
    all characters to lowercase.
    """
    alphabet = "qwertyuiopasdfghjklzxcvbnm"
    result = ""
    for char in text:
        lower_char = char.lower()
        if lower_char in alphabet:
            result += lower_char
    return result


def main():
    # Load dataset
    df = pd.read_csv("Project/Students Social Media Addiction.csv")

    # Open output text file
    with open("summary_output.txt", "w") as f:

        # =========================
        # Quantitative Variables
        # =========================
        numeric_cols = df.select_dtypes(include=["int64", "float64"])

        f.write("QUANTITATIVE VARIABLE SUMMARIES\n")
        f.write("=" * 40 + "\n\n")

        for col in numeric_cols.columns:
            f.write(f"Variable: {col}\n")

            mean = df[col].mean()
            std = df[col].std()
            minimum = df[col].min()
            q1 = df[col].quantile(0.25)
            median = df[col].median()
            q3 = df[col].quantile(0.75)
            maximum = df[col].max()

            f.write(f"Mean: {mean}\n")
            f.write(f"Standard Deviation: {std}\n")
            f.write(f"Minimum: {minimum}\n")
            f.write(f"First Quartile (Q1): {q1}\n")
            f.write(f"Median: {median}\n")
            f.write(f"Third Quartile (Q3): {q3}\n")
            f.write(f"Maximum: {maximum}\n")
            f.write("-" * 40 + "\n\n")

        # =========================
        # Categorical Variables
        # =========================
        categorical_cols = df.select_dtypes(include=["object"])

        f.write("\nCATEGORICAL VARIABLE SUMMARIES\n")
        f.write("=" * 40 + "\n")

        for col in categorical_cols.columns:
            f.write(f"Variable: {col}\n")

            counts = df[col].value_counts()

            for value, count in counts.items():
                f.write(f"{value}: {count}\n")

            f.write("-" * 40 + "\n")


if __name__ == "__main__":
    main()

