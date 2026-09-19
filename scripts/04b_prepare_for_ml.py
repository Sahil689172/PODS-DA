import pandas as pd
from pathlib import Path


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "outputs" / "integrated_market_data_2016_2026.csv"

OUTPUT_FILE = BASE_DIR / "outputs" / "integrated_market_data_ready_for_ml.csv"

SUMMARY_FILE = BASE_DIR / "outputs" / "tables" / "04b_ml_preparation_summary.csv"


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("STEP 04B - PREPARE INTEGRATED DATA FOR ML")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput file: {INPUT_FILE}")
print(f"Initial shape: {df.shape}")


# ==========================================================
# DATE CONVERSION
# ==========================================================

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# ==========================================================
# REMOVE OLD DA-1 NORMALIZED COLUMNS
# ==========================================================

old_normalized_columns = [
    "Open_Normalized",
    "High_Normalized",
    "Low_Normalized",
    "Close_Normalized",
    "Volume_Normalized"
]

existing_normalized_columns = [
    col for col in old_normalized_columns
    if col in df.columns
]

if existing_normalized_columns:

    print("\nRemoving old DA-1 normalized columns:")

    for col in existing_normalized_columns:
        print(f"  - {col}")

    df = df.drop(columns=existing_normalized_columns)

else:

    print("\nNo old DA-1 normalized columns found.")


# ==========================================================
# CHECK RAW NUMERICAL FEATURES
# ==========================================================

raw_price_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

print("\nRaw numerical columns retained:")

for col in raw_price_columns:

    if col in df.columns:
        print(f"  ✓ {col}")
    else:
        print(f"  ✗ {col} MISSING")


# ==========================================================
# CHECK SECTOR ENCODING
# ==========================================================

if "Sector_ID" in df.columns:

    print("\nSector_ID:")
    print("  ✓ Present")

    print(f"  Unique Sector_ID values: {df['Sector_ID'].nunique()}")

else:

    print("\nWARNING: Sector_ID is missing.")


# ==========================================================
# CHECK DUPLICATES
# ==========================================================

duplicate_count = df.duplicated(
    subset=["Date", "Ticker"]
).sum()

print(f"\nDuplicate Date + Ticker rows: {duplicate_count}")


# ==========================================================
# CHECK DATE RANGE
# ==========================================================

print(
    f"\nDate range: "
    f"{df['Date'].min().date()} → {df['Date'].max().date()}"
)

print(f"Unique tickers: {df['Ticker'].nunique()}")


# ==========================================================
# CHECK MISSING VALUES
# ==========================================================

missing_summary = (
    df.isna()
      .sum()
      .sort_values(ascending=False)
)

print("\nTop missing-value columns:")

print(
    missing_summary[
        missing_summary > 0
    ].head(15)
)


# ==========================================================
# VERIFY NO NORMALIZED COLUMNS REMAIN
# ==========================================================

remaining_normalized = [
    col for col in df.columns
    if "Normalized" in col
]

print("\nOld normalized columns remaining:")

if remaining_normalized:
    for col in remaining_normalized:
        print(f"  ✗ {col}")
else:
    print("  ✓ None")


# ==========================================================
# SAVE CLEAN ML-READY BASE DATA
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved ML-ready integrated dataset:")
print(f"  {OUTPUT_FILE}")

print(f"\nFinal shape: {df.shape}")


# ==========================================================
# CREATE SUMMARY TABLE
# ==========================================================

summary = pd.DataFrame({
    "Metric": [
        "Input Rows",
        "Input Columns",
        "Final Rows",
        "Final Columns",
        "Unique Tickers",
        "Unique Sectors",
        "Duplicate Date-Ticker Rows",
        "Old Normalized Columns Removed",
        "Date Start",
        "Date End"
    ],
    "Value": [
        "N/A",
        "N/A",
        len(df),
        len(df.columns),
        df["Ticker"].nunique(),
        df["Sector"].nunique(),
        duplicate_count,
        ", ".join(existing_normalized_columns)
        if existing_normalized_columns
        else "None",
        str(df["Date"].min().date()),
        str(df["Date"].max().date())
    ]
})

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print(f"\nSaved summary:")
print(f"  {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("STEP 04B COMPLETED")
print("=" * 70)