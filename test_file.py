import pandas as pd


# checking the file and testing it with a smaller test file
def test_smaller():
    '''
    For this function, it is created
    to test our code on a smaller dataset
    to verify the accuracy of our code better
    '''

    # reading in the csv file & SKIPPING ONE ROW FOR THE TEST DATASET
    df = pd.read_csv("/Users/isabellale/Downloads/test_socialmedia.csv",
                     skiprows=1)

    # Open output text file
    with open("summary_output.txt", "w") as f:

        # =========================
        # Quantitative Variables
        # =========================
        numeric_cols = df.select_dtypes(include=["int64", "float64"])

        print(numeric_cols)

        print()
        print(df.columns)
        print(df)

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


def main():
    """
    This function is used to test all of our functions
    """
    test_smaller()


if __name__ == "__main__":
    main()
