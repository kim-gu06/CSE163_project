'''
Isabella Le and Kimberly Gu
EDA 2
CSE 163 Section AG & AD

This file is used to test and verify our code is working
as intended with a smaller csv file.
'''


import pandas as pd


def total(numbers):
    """
    Takes in a number n (int) and returns the sum of the numbers from
    0(inclusive) to n (inclusive). If n is negative, returns None.
    """
    result = 0
    for n in numbers:
        result += n
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


def load_csv(csv_path):
    """
    Reads a CSV file and returns its contents as a pandas DataFrame.
    """
    return pd.read_csv(csv_path, skiprows=1)


def is_quantitative(series):
    """
    Returns True if the given pandas Series contains numeric data.
    Returns False otherwise.
    """
    return pd.api.types.is_numeric_dtype(series)


def compute_mean(numbers):
    """
    Returns the mean (average) of a list of numbers.
    Assumes the list is non-empty.
    """
    return total(numbers) / len(numbers)


def compute_std(numbers, mean):
    """
    Returns the standard deviation of a list of numbers.

    Parameters:
        numbers: list of numeric values
        mean: the mean of the list

    Assumes the list is non-empty.
    """
    squared_diffs = []
    for n in numbers:
        squared_diffs.append((n - mean) ** 2)

    variance = total(squared_diffs) / len(numbers)
    return variance ** 0.5


def seven_number_summary(series):
    """
    Computes the seven-number summary for a quantitative column:
    mean, standard deviation, minimum, first quartile (Q1),
    median, third quartile (Q3), and maximum.

    Returns a dictionary mapping statistic names to values.
    """

    # Convert pandas Series to a Python list
    values = []
    for v in series:
        values.append(v)
    values.sort()   # Sort values from least to greatest
    # Compute mean and standard deviation
    mean = compute_mean(values)
    std = compute_std(values, mean)
    # Compute five-number summary (min, Q1, median, Q3, max)
    min_val, q1, med, q3, max_val = five_number_summary(values)

    # Return all seven statistics
    return {
        "Mean": mean,
        "Standard Deviation": std,
        "Minimum": min_val,
        "Q1": q1,
        "Median": med,
        "Q3": q3,
        "Maximum": max_val
    }


def categorical_summary(series):
    """
    Returns a dictionary counting how many times each unique value
    appears in the given pandas Series.
    """
    counts = {}
    # Count occurrences of each value
    for value in series:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    return counts


def write_quantitative(file, column, summary):
    """
    Writes the seven-number summary for a quantitative variable
    to an open text file.
    """
    # Write variable name
    file.write("Variable: " + column + "\n")
    # Write each statistic
    for key in summary:
        file.write(key + ": " + str(summary[key]) + "\n")
    # Blank line after each variable for readability
    file.write("\n")


def write_categorical(file, column, counts):
    """
    Writes the value counts for a categorical variable
    to an open text file.
    """
    # Write variable name
    file.write("Variable: " + column + "\n")
    # Write each category and its count
    for value in counts:
        file.write(str(value) + ": " + str(counts[value]) + "\n")
    # Blank line after each variable for readability
    file.write("\n")


def summarize_dataframe(df, output_txt, variables_of_interest=None):
    """
    Writes a summary of each variable in the DataFrame to a text file.

    If variables_of_interest is None, all columns in the DataFrame
    will be summarized. Otherwise, only the listed columns are used.
    """
    # Use all columns if none are specified
    if variables_of_interest is None:
        variables_of_interest = list(df.columns)

    # Open output file for writing
    with open(output_txt, "w") as file:
        file.write("VARIABLE SUMMARY REPORT\n\n")
        # Process each selected column
        for column in variables_of_interest:
            series = df[column]

            if is_quantitative(series):
                summary = seven_number_summary(series)
                write_quantitative(file, column, summary)
            else:
                counts = categorical_summary(series)
                write_categorical(file, column, counts)


def main():
    """
    Loads the CSV file, generates summaries for each variable,
    and writes the results to a text file.
    """
    input_csv = "/Users/isabellale/Desktop/CSE 163/final project/test.csv"
    output_txt = "variable_summary.txt"
    # Load dataset
    df = load_csv(input_csv)
    # Generate and write summaries
    summarize_dataframe(df, output_txt)

    print("Summary written to", output_txt)


if __name__ == "__main__":
    main()

