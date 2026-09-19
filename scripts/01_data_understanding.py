# ==========================================================
# PODS-DA
# Script 01 : Data Understanding
# ==========================================================

import pandas as pd
import os

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "nifty50_historical_data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "tables"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("\n" + "=" * 60)
print("             DATASET UNDERSTANDING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully.")

# ==========================================================
# BASIC DATASET INFORMATION
# ==========================================================

print("\n" + "=" * 60)
print("BASIC DATASET INFORMATION")
print("=" * 60)

print(f"\nNumber of Rows    : {df.shape[0]:,}")
print(f"Number of Columns : {df.shape[1]}")

# ==========================================================
# COLUMN NAMES
# ==========================================================

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

# ==========================================================
# DATA TYPES
# ==========================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

# ==========================================================
# FIRST 5 RECORDS
# ==========================================================

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(df.head())

# ==========================================================
# LAST 5 RECORDS
# ==========================================================

print("\n" + "=" * 60)
print("LAST 5 RECORDS")
print("=" * 60)

print(df.tail())

# ==========================================================
# DATE CONVERSION FOR INSPECTION
# ==========================================================

print("\n" + "=" * 60)
print("DATE RANGE")
print("=" * 60)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
    utc=True
)

print(f"\nEarliest Date : {df['Date'].min()}")
print(f"Latest Date   : {df['Date'].max()}")

# ==========================================================
# UNIQUE TICKERS
# ==========================================================

print("\n" + "=" * 60)
print("UNIQUE COMPANIES")
print("=" * 60)

print(f"\nNumber of Unique Tickers : {df['Ticker'].nunique()}")

print("\nSample Tickers:")

for ticker in df["Ticker"].dropna().unique()[:20]:
    print(ticker)

# ==========================================================
# UNIQUE SECTORS
# ==========================================================

print("\n" + "=" * 60)
print("SECTOR INFORMATION")
print("=" * 60)

print(f"\nNumber of Unique Sectors : {df['Sector'].nunique()}")

print("\nSectors:")

for sector in df["Sector"].dropna().unique():
    print(sector)

# ==========================================================
# MISSING VALUES
# ==========================================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing = pd.DataFrame({
    "Column": df.columns,
    "Missing_Values": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().sum().values / len(df) * 100
    )
})

print("\n")
print(missing.to_string(index=False))

# Save missing value report
missing.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "01_missing_value_report.csv"
    ),
    index=False
)

# ==========================================================
# DUPLICATE ANALYSIS
# ==========================================================

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

total_duplicates = df.duplicated().sum()

date_ticker_duplicates = df.duplicated(
    subset=["Date", "Ticker"]
).sum()

print(f"\nComplete Duplicate Rows : {total_duplicates:,}")

print(
    f"Duplicate Date-Ticker Combinations : "
    f"{date_ticker_duplicates:,}"
)

# ==========================================================
# NUMERICAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

numeric_summary = df.describe().T

print("\n")
print(numeric_summary)

numeric_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "01_numeric_summary.csv"
    )
)

# ==========================================================
# DATASET MEMORY USAGE
# ==========================================================

print("\n" + "=" * 60)
print("DATASET MEMORY USAGE")
print("=" * 60)

memory_mb = df.memory_usage(
    deep=True
).sum() / (1024 ** 2)

print(f"\nMemory Usage : {memory_mb:.2f} MB")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("DATA UNDERSTANDING COMPLETED")
print("=" * 60)

print(f"""
Original Dataset
----------------
Rows              : {df.shape[0]:,}
Columns           : {df.shape[1]}
Unique Tickers    : {df['Ticker'].nunique()}
Unique Sectors    : {df['Sector'].nunique()}
Earliest Date     : {df['Date'].min()}
Latest Date       : {df['Date'].max()}
Duplicate Rows    : {total_duplicates:,}
Date-Ticker Dupes : {date_ticker_duplicates:,}

Reports saved in:
outputs/tables/
""")

print("=" * 60)