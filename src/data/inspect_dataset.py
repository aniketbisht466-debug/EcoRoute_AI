
"""
EcoRoute AI
Dataset Inspection Engine
Purpose: Inspect original external datasets
without modifying their contents.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "external" / "dryad_truck"


def inspect_csv(file_path):

    print("\n" + "=" * 70)
    print("FILE:", file_path.name)
    print("=" * 70)

    print(
        "File size:",
        round(file_path.stat().st_size / 1024, 2),
        "KB"
    )

    try:

        df = pd.read_csv(file_path)

        print("\nDATASET DIMENSIONS")
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])

        print("\nCOLUMN NAMES")

        for column in df.columns:
            print("-", column)

        print("\nFIRST FIVE RECORDS")
        print(df.head().to_string())

        print("\nDATA TYPES")
        print(df.dtypes)

        print("\nMISSING VALUES")
        print(df.isnull().sum())

        print("\nDUPLICATE ROWS")
        print(df.duplicated().sum())

        print("\nSTATISTICAL SUMMARY")
        print(df.describe(include="all").to_string())

    except Exception as error:

        print("Unable to inspect file:", error)


def main():

    print("=" * 70)
    print("ECOROUTE AI - DATASET INSPECTION")
    print("=" * 70)

    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    print("Dataset directory:", DATA_DIR)

    print("CSV files found:", len(csv_files))

    if not csv_files:
        print("No CSV files found.")
        print("Download and extract the dataset first.")
        return

    for file in csv_files:
        inspect_csv(file)

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
