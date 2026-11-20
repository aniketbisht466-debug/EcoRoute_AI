
"""
EcoRoute AI
Dataset Quality Audit - Version 1.0

Purpose:
    Validate all original vehicle trajectory files
    before cleaning and consolidation.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "external" / "dryad_truck"

REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


EXPECTED_COLUMNS = [
    "Speed",
    "FuelRate",
    "DistanceToIntersection"
]


def audit_file(file):

    df = pd.read_csv(file)

    original_rows = len(df)

    # Check expected columns
    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]

    # Identify scenario from filename
    filename = file.stem.lower()

    if filename.endswith("_baseline"):
        scenario = "baseline"

    elif filename.endswith("_eco"):
        scenario = "eco"

    else:
        scenario = "unknown"

    result = {
        "filename": file.relative_to(DATA_DIR).as_posix(),
        "scenario": scenario,
        "rows": original_rows,
        "columns": len(df.columns),
        "empty_file": original_rows == 0,
        "missing_columns": " | ".join(missing_columns),
        "extra_columns": " | ".join(extra_columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

    # Audit each expected measurement
    for column in EXPECTED_COLUMNS:

        if column not in df.columns:
            continue

        original = df[column]

        numeric = pd.to_numeric(
            original,
            errors="coerce"
        )

        non_numeric = (
            numeric.isna() & original.notna()
        )

        infinite = np.isinf(numeric)

        negative = (numeric < 0) & ~infinite

        result[f"{column}_non_numeric"] = int(
            non_numeric.sum()
        )

        result[f"{column}_infinite"] = int(
            infinite.sum()
        )

        result[f"{column}_negative"] = int(
            negative.sum()
        )

        valid = numeric[np.isfinite(numeric)]

        result[f"{column}_min"] = valid.min()
        result[f"{column}_max"] = valid.max()
        result[f"{column}_mean"] = valid.mean()

    return result


def main():

    files = sorted(DATA_DIR.rglob("*.csv"))

    print("=" * 65)
    print("ECOROUTE AI - DATA QUALITY AUDIT")
    print("=" * 65)

    print("CSV files detected:", len(files))

    if not files:
        print("ERROR: No dataset files found.")
        return

    results = []
    errors = []

    for index, file in enumerate(files, start=1):

        try:

            results.append(audit_file(file))

        except Exception as error:

            errors.append({
                "filename": file.relative_to(
                    DATA_DIR
                ).as_posix(),
                "error": str(error)
            })

        if index % 50 == 0:
            print(
                f"Audited {index}/{len(files)} files"
            )

    audit = pd.DataFrame(results)

    error_df = pd.DataFrame(
        errors,
        columns=["filename", "error"]
    )

    error_df.to_csv(
        REPORT_DIR / "dataset_audit_errors.csv",
        index=False
    )

    if audit.empty:
        print("ERROR: No files could be audited.")
        return

    audit.to_csv(
        REPORT_DIR / "dataset_quality_audit.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("AUDIT SUMMARY")
    print("=" * 65)

    print("Files discovered:", len(files))
    print("Files successfully read:", len(audit))
    print("Files failed:", len(errors))
    print("Total observations:", audit["rows"].sum())

    print("\nTotal missing cells:",
          audit["missing_values"].sum())

    print("Total duplicate rows:",
          audit["duplicate_rows"].sum())

    print("Empty files:",
          audit["empty_file"].sum())

    print("Files missing expected columns:",
          audit["missing_columns"].ne("").sum())

    print("Files with additional columns:",
          audit["extra_columns"].ne("").sum())

    print("\nDRIVING SCENARIOS")

    print(
        audit.groupby("scenario").agg(
            files=("filename", "count"),
            observations=("rows", "sum")
        )
    )

    print("\nMEASUREMENT QUALITY")

    for column in EXPECTED_COLUMNS:

        print(f"\n{column}")

        for suffix in [
            "non_numeric",
            "infinite",
            "negative"
        ]:

            name = f"{column}_{suffix}"

            if name in audit.columns:
                print(
                    f"  {suffix}:",
                    audit[name].sum()
                )

        min_col = f"{column}_min"
        max_col = f"{column}_max"

        if min_col in audit.columns:
            print("  Global minimum:",
                  audit[min_col].min())

        if max_col in audit.columns:
            print("  Global maximum:",
                  audit[max_col].max())

    print("\nAUDIT REPORT SAVED:")
    print(REPORT_DIR / "dataset_quality_audit.csv")

    print("\nOriginal dataset remains unchanged.")


if __name__ == "__main__":
    main()
