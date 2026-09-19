# ==========================================================
# PODS-DA
# Step 02: Data Cleaning and Preprocessing
# ==========================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# 1. PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "nifty50_historical_data.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"

OUTPUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


# ==========================================================
# 2. LOAD DATA
# ==========================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)
original_columns = len(df.columns)

print(f"Original shape: {df.shape}")


# ==========================================================
# 3. DATE CONVERSION
# ==========================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Preserve the original local calendar date.
# Do NOT convert the timestamp to UTC because that can shift
# Indian dates backward by one day.

if hasattr(df["Date"].dt, "tz") and df["Date"].dt.tz is not None:
    df["Date"] = df["Date"].dt.tz_localize(None)


# ==========================================================
# 4. SORT DATA
# ==========================================================

df = df.sort_values(
    by=["Ticker", "Date"]
).reset_index(drop=True)


# ==========================================================
# 5. RECALCULATE PRICE-BASED FEATURES
# ==========================================================

print("\nRecalculating financial features...")

# Daily return
df["Daily_Return"] = (
    df.groupby("Ticker")["Close"]
    .pct_change()
)

# 20-day historical volatility
df["Volatility_20D"] = (
    df.groupby("Ticker")["Daily_Return"]
    .transform(
        lambda x: x.rolling(window=20).std()
    )
)

# 50-day moving average
df["MA_50"] = (
    df.groupby("Ticker")["Close"]
    .transform(
        lambda x: x.rolling(window=50).mean()
    )
)

# 200-day moving average
df["MA_200"] = (
    df.groupby("Ticker")["Close"]
    .transform(
        lambda x: x.rolling(window=200).mean()
    )
)


# ==========================================================
# 6. FILTER DATE RANGE
# ==========================================================

START_DATE = "2016-01-01"
END_DATE = "2026-01-30"

df = df[
    (df["Date"] >= START_DATE) &
    (df["Date"] <= END_DATE)
].copy()

print(
    f"\nDate range selected: "
    f"{START_DATE} to {END_DATE}"
)

print(
    f"Shape after date filtering: {df.shape}"
)


# ==========================================================
# 7. REMOVE COMPLETELY MISSING COLUMN
# ==========================================================

if "PEG_Ratio" in df.columns:

    df = df.drop(
        columns=["PEG_Ratio"]
    )

    print(
        "\nRemoved PEG_Ratio because it "
        "is completely missing."
    )


# ==========================================================
# 8. HANDLE STOCK SPLIT
# ==========================================================

# Missing Stock_Split values represent no recorded split
# in this dataset, so they are represented as 0.

if "Stock_Split" in df.columns:

    df["Stock_Split"] = (
        df["Stock_Split"].fillna(0)
    )


# ==========================================================
# 9. REMOVE DUPLICATE DATE-TICKER ROWS
# ==========================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["Date", "Ticker"]
)

duplicates_removed = (
    before_duplicates - len(df)
)

print(
    f"\nDuplicate Date-Ticker rows removed: "
    f"{duplicates_removed}"
)


# ==========================================================
# 10. VALIDATE OHLC DATA
# ==========================================================

invalid_ohlc = df[
    (df["High"] < df["Low"]) |
    (df["High"] < df["Open"]) |
    (df["High"] < df["Close"]) |
    (df["Low"] > df["Open"]) |
    (df["Low"] > df["Close"])
]

invalid_ohlc_count = len(invalid_ohlc)

print(
    f"Invalid OHLC rows found: "
    f"{invalid_ohlc_count}"
)

if invalid_ohlc_count > 0:

    df = df.drop(
        index=invalid_ohlc.index
    )


# ==========================================================
# 11. CHECK MISSING VALUES
# ==========================================================

print("\nMissing values after basic cleaning:")

print(
    df.isnull().sum()
)


# ==========================================================
# 12. HANDLE TECHNICAL FEATURE MISSING VALUES
# ==========================================================

# These four variables are required because they are
# calculated directly from historical price data.

technical_columns = [
    "Daily_Return",
    "Volatility_20D",
    "MA_50",
    "MA_200"
]

print(
    "\nMissing values in mandatory "
    "technical features:"
)

print(
    df[technical_columns].isnull().sum()
)


before_technical_cleaning = len(df)

df = df.dropna(
    subset=technical_columns
).copy()

technical_rows_removed = (
    before_technical_cleaning - len(df)
)

print(
    "\nRows removed because of missing "
    "technical features:",
    technical_rows_removed
)


# ==========================================================
# 13. RETAIN FUNDAMENTAL MISSING VALUES
# ==========================================================

# IMPORTANT:
#
# PE_Ratio, Dividend_Yield and Beta are NOT removed here.
#
# Some companies have these values completely unavailable
# in the historical dataset.
#
# Example:
# COALINDIA.NS  -> Beta missing
# INDUSINDBK.NS -> PE_Ratio and Dividend_Yield missing
#
# These values will be handled later during DA-2
# feature engineering/model preparation.

fundamental_columns = [
    "PE_Ratio",
    "Dividend_Yield",
    "Beta"
]

print(
    "\nFundamental missing values retained "
    "for later DA-2 handling:"
)

print(
    df[fundamental_columns].isnull().sum()
)


# ==========================================================
# 14. SECTOR ENCODING
# ==========================================================

sector_categories = sorted(
    df["Sector"].dropna().unique()
)

sector_mapping = {
    sector: index
    for index, sector
    in enumerate(sector_categories)
}

df["Sector_ID"] = (
    df["Sector"].map(sector_mapping)
)


# Save sector mapping

sector_mapping_df = pd.DataFrame(
    list(sector_mapping.items()),
    columns=[
        "Sector",
        "Sector_ID"
    ]
)

sector_mapping_df.to_csv(
    TABLE_DIR /
    "02_sector_encoding.csv",
    index=False
)


# ==========================================================
# 15. MIN-MAX NORMALIZATION
# ==========================================================

normalize_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

scaler = MinMaxScaler()

normalized_columns = [
    f"{column}_Normalized"
    for column in normalize_columns
]

df[normalized_columns] = (
    scaler.fit_transform(
        df[normalize_columns]
    )
)


# ==========================================================
# 16. FINAL SORTING
# ==========================================================

df = df.sort_values(
    by=["Date", "Ticker"]
).reset_index(drop=True)


# ==========================================================
# 17. FINAL MISSING VALUE REPORT
# ==========================================================

missing_report = pd.DataFrame({

    "Column": df.columns,

    "Missing_Count":
        df.isnull().sum().values,

    "Missing_Percentage":
        (
            df.isnull().mean().values
            * 100
        )
})

missing_report.to_csv(
    TABLE_DIR /
    "02_final_missing_value_report.csv",
    index=False
)


# ==========================================================
# 18. SAVE CLEANED DATA
# ==========================================================

cleaned_file = (
    OUTPUT_DIR /
    "historical_cleaned_2016_2026.csv"
)

df.to_csv(
    cleaned_file,
    index=False
)


# ==========================================================
# 19. SAVE CLEANING SUMMARY
# ==========================================================

summary = {

    "Original Rows":
        original_rows,

    "Final Rows":
        len(df),

    "Original Columns":
        original_columns,

    "Final Columns":
        len(df.columns),

    "Date Start":
        df["Date"].min(),

    "Date End":
        df["Date"].max(),

    "Unique Tickers":
        df["Ticker"].nunique(),

    "Unique Sectors":
        df["Sector"].nunique(),

    "Duplicate Date-Ticker Rows":
        df.duplicated(
            subset=["Date", "Ticker"]
        ).sum(),

    "Invalid OHLC Rows":
        invalid_ohlc_count,

    "Rows Removed - Technical Missing":
        technical_rows_removed,

    "Rows Removed - Duplicates":
        duplicates_removed
}

summary_df = pd.DataFrame(
    summary.items(),
    columns=[
        "Metric",
        "Value"
    ]
)

summary_df.to_csv(
    TABLE_DIR /
    "02_cleaning_summary.csv",
    index=False
)


# ==========================================================
# 20. DISPLAY FINAL RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETED")
print("=" * 70)

print(
    f"\nFinal shape: {df.shape}"
)

print(
    f"Final date range: "
    f"{df['Date'].min().date()} "
    f"→ "
    f"{df['Date'].max().date()}"
)

print(
    f"Unique tickers: "
    f"{df['Ticker'].nunique()}"
)

print(
    f"Unique sectors: "
    f"{df['Sector'].nunique()}"
)

print("\nFinal missing values:")

print(
    df.isnull().sum()
)

print("\nFinal fundamental missing values:")

print(
    df[fundamental_columns].isnull().sum()
)

print("\nFinal columns:")

for i, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"{i:2}. {column}"
    )

print("\nFiles created:")

print(
    f"1. {cleaned_file}"
)

print(
    f"2. "
    f"{TABLE_DIR / '02_sector_encoding.csv'}"
)

print(
    f"3. "
    f"{TABLE_DIR / '02_final_missing_value_report.csv'}"
)

print(
    f"4. "
    f"{TABLE_DIR / '02_cleaning_summary.csv'}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)