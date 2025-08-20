import pandas as pd


def find_min_psi_plus_xi(csv_filepath: str):
    """
    Loads data from a CSV file representing the conjectured lower boundary,
    finds the point that minimizes the sum of (psi + xi), and prints the
    corresponding parameter and correlation values.

    Args:
        csv_filepath: The path to the input CSV file.
    """
    try:
        # 1. Load the data from the CSV file
        df = pd.read_csv(csv_filepath)
        print(f"✅ Successfully loaded '{csv_filepath}'.")
        print(
            f"   Data contains {len(df)} rows and these columns: {list(df.columns)}\n"
        )

    except FileNotFoundError:
        print(f"❌ Error: The file '{csv_filepath}' was not found.")
        print("   Please ensure the CSV file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Check if the required columns exist
    required_cols = ["mu", "xi", "psi"]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: The CSV file must contain the columns {required_cols}.")
        return

    # 2. Calculate the sum psi + xi for each row
    df["psi_plus_xi"] = df["psi"] + df["xi"]

    # 3. Find the index (row number) of the overall minimum sum
    min_index = df["psi_plus_xi"].idxmin()

    # 4. Extract the full row of data where the minimum occurred
    result_row = df.loc[min_index]

    min_mu = result_row["mu"]
    min_xi = result_row["xi"]
    min_psi = result_row["psi"]
    min_sum = result_row["psi_plus_xi"]

    # 5. Print the final results in a clean, formatted way
    print("=" * 40)
    print("🔎 Analysis Results 🔎")
    print("Found the point that minimizes the sum (ψ + ξ):")
    print("=" * 40)
    print(f"  Optimal Parameter (μ): {min_mu:.4f}")
    print(f"  Chatterjee's xi (ξ):   {min_xi:.4f}")
    print(f"  Spearman's psi (ψ):   {min_psi:.4f}")
    print("-" * 28)
    print(f"  Minimum Sum (ψ + ξ):  {min_sum:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    # Define the name of the CSV file to be analyzed
    CSV_FILE = "lower_boundary_final_smooth.csv"

    # Run the analysis function
    find_min_psi_plus_xi(CSV_FILE)
