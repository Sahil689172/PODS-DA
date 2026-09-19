# ==========================================================
# PODS-DA
# Step 04: Historical + Current Data Integration
# ==========================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================================
# 1. PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

HISTORICAL_FILE = (
    BASE_DIR
    / "outputs"
    / "historical_cleaned_2016_2026.csv"
)

CURRENT_FILE = (
    BASE_DIR
    / "outputs"
    / "current_data"
    / "current_yfinance_2026_02_onward.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
)

TABLE_DIR = (
    OUTPUT_DIR
    / "tables"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# 2. LOAD DATASETS
# ==========================================================

print("=" * 70)
print("DATA INTEGRATION")
print("=" * 70)

print("\nLoading historical data...")

historical = pd.read_csv(
    HISTORICAL_FILE
)

print(
    f"Historical shape: "
    f"{historical.shape}"
)

print("\nLoading current yfinance data...")

current = pd.read_csv(
    CURRENT_FILE
)

print(
    f"Current shape: "
    f"{current.shape}"
)


# ==========================================================
# 3. CONVERT DATES
# ==========================================================

historical["Date"] = pd.to_datetime(
    historical["Date"],
    errors="coerce"
)

current["Date"] = pd.to_datetime(
    current["Date"],
    errors="coerce"
)


# ==========================================================
# 4. CHECK TICKERS
# ==========================================================

historical_tickers = set(
    historical["Ticker"]
    .dropna()
    .unique()
)

current_tickers = set(
    current["Ticker"]
    .dropna()
    .unique()
)

print("\nTicker comparison:")

print(
    f"Historical tickers: "
    f"{len(historical_tickers)}"
)

print(
    f"Current tickers: "
    f"{len(current_tickers)}"
)

missing_current = sorted(
    historical_tickers - current_tickers
)

extra_current = sorted(
    current_tickers - historical_tickers
)

print(
    f"Missing current tickers: "
    f"{len(missing_current)}"
)

print(
    f"Extra current tickers: "
    f"{len(extra_current)}"
)

if missing_current:
    print("\nTickers missing from current data:")

    for ticker in missing_current:
        print(f"  {ticker}")

if extra_current:
    print("\nExtra current tickers:")

    for ticker in extra_current:
        print(f"  {ticker}")


# ==========================================================
# 5. GET COMPANY / SECTOR MAPPING
# ==========================================================

company_mapping = (
    historical[
        [
            "Ticker",
            "Company_Name",
            "Sector"
        ]
    ]
    .drop_duplicates(
        subset=["Ticker"]
    )
)


# ==========================================================
# 6. ADD COMPANY AND SECTOR TO CURRENT DATA
# ==========================================================

current = current.merge(
    company_mapping,
    on="Ticker",
    how="left"
)


# ==========================================================
# 7. CHECK COMPANY / SECTOR MISSING VALUES
# ==========================================================

print(
    "\nMissing Company_Name values in current data:",
    current["Company_Name"].isnull().sum()
)

print(
    "Missing Sector values in current data:",
    current["Sector"].isnull().sum()
)


# ==========================================================
# 8. ADD COLUMNS REQUIRED TO MATCH HISTORICAL DATA
# ==========================================================

# These columns exist in the historical dataset but not
# directly in the yfinance price download.

columns_to_add = [
    "Dividend",
    "Stock_Split",
    "Market_Cap",
    "PE_Ratio",
    "Forward_PE",
    "Price_to_Book",
    "Dividend_Yield",
    "EPS",
    "Beta",
    "52Week_High",
    "52Week_Low"
]

for column in columns_to_add:

    if column not in current.columns:

        current[column] = np.nan


# ==========================================================
# 9. REMOVE OLD CALCULATED FEATURES
# ==========================================================

# We will recalculate these across the complete combined
# time series.

calculated_columns = [
    "Daily_Return",
    "Volatility_20D",
    "MA_50",
    "MA_200"
]

for column in calculated_columns:

    if column in historical.columns:
        historical = historical.drop(
            columns=[column]
        )

    if column in current.columns:
        current = current.drop(
            columns=[column]
        )


# ==========================================================
# 10. REMOVE NORMALIZED COLUMNS
# ==========================================================

# Normalization will be handled later in the ML pipeline.
# Keeping old normalization values would be incorrect after
# adding new data.

normalized_columns = [
    "Open_Normalized",
    "High_Normalized",
    "Low_Normalized",
    "Close_Normalized",
    "Volume_Normalized"
]

for column in normalized_columns:

    if column in historical.columns:
        historical = historical.drop(
            columns=[column]
        )


# ==========================================================
# 11. REMOVE SECTOR_ID FROM HISTORICAL
# ==========================================================

# We will recreate the encoding after integration.

if "Sector_ID" in historical.columns:

    historical = historical.drop(
        columns=["Sector_ID"]
    )


# ==========================================================
# 12. MATCH COLUMN STRUCTURE
# ==========================================================

# Columns expected in final integrated dataset.

final_columns = [
    "Date",
    "Ticker",
    "Company_Name",
    "Sector",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividend",
    "Stock_Split",
    "Market_Cap",
    "PE_Ratio",
    "Forward_PE",
    "Price_to_Book",
    "Dividend_Yield",
    "EPS",
    "Beta",
    "52Week_High",
    "52Week_Low"
]


# Add any missing columns to historical/current

for column in final_columns:

    if column not in historical.columns:
        historical[column] = np.nan

    if column not in current.columns:
        current[column] = np.nan


# Keep only the common final structure

historical = historical[
    final_columns
].copy()

current = current[
    final_columns
].copy()


# ==========================================================
# 13. COMBINE HISTORICAL + CURRENT
# ==========================================================

print("\nCombining datasets...")

combined = pd.concat(
    [
        historical,
        current
    ],
    ignore_index=True
)


print(
    f"Combined rows before duplicate removal: "
    f"{len(combined):,}"
)


# ==========================================================
# 14. REMOVE DUPLICATE DATE-TICKER ROWS
# ==========================================================

before_duplicates = len(
    combined
)

combined = (
    combined
    .drop_duplicates(
        subset=["Date", "Ticker"],
        keep="first"
    )
)

duplicates_removed = (
    before_duplicates
    - len(combined)
)

print(
    f"Duplicate Date-Ticker rows removed: "
    f"{duplicates_removed}"
)


# ==========================================================
# 15. SORT COMPLETE TIME SERIES
# ==========================================================

combined = (
    combined
    .sort_values(
        by=["Ticker", "Date"]
    )
    .reset_index(drop=True)
)


# ==========================================================
# 16. RECALCULATE DAILY RETURN
# ==========================================================

print(
    "\nRecalculating time-series features..."
)

combined["Daily_Return"] = (
    combined
    .groupby("Ticker")["Close"]
    .pct_change()
)


# ==========================================================
# 17. RECALCULATE VOLATILITY
# ==========================================================

combined["Volatility_20D"] = (
    combined
    .groupby("Ticker")["Daily_Return"]
    .transform(
        lambda x:
        x.rolling(
            window=20
        ).std()
    )
)


# ==========================================================
# 18. RECALCULATE MOVING AVERAGES
# ==========================================================

combined["MA_50"] = (
    combined
    .groupby("Ticker")["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=50
        ).mean()
    )
)

combined["MA_200"] = (
    combined
    .groupby("Ticker")["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=200
        ).mean()
    )
)


# ==========================================================
# 19. SECTOR ENCODING
# ==========================================================

sector_categories = sorted(
    combined["Sector"]
    .dropna()
    .unique()
)

sector_mapping = {
    sector: index
    for index, sector
    in enumerate(sector_categories)
}

combined["Sector_ID"] = (
    combined["Sector"]
    .map(sector_mapping)
)


# Save sector mapping

sector_mapping_df = pd.DataFrame(
    list(
        sector_mapping.items()
    ),
    columns=[
        "Sector",
        "Sector_ID"
    ]
)

sector_mapping_df.to_csv(
    TABLE_DIR
    / "04_sector_encoding.csv",
    index=False
)


# ==========================================================
# 20. FINAL SORTING
# ==========================================================

combined = (
    combined
    .sort_values(
        by=["Date", "Ticker"]
    )
    .reset_index(drop=True)
)


# ==========================================================
# 21. CHECK DATE TRANSITION
# ==========================================================

print("\nChecking historical/current transition...")

transition_data = combined[
    (
        combined["Date"]
        >= pd.Timestamp("2026-01-28")
    )
    &
    (
        combined["Date"]
        <= pd.Timestamp("2026-02-05")
    )
]

print(
    transition_data[
        [
            "Date",
            "Ticker",
            "Close",
            "Daily_Return"
        ]
    ].head(20)
)


# ==========================================================
# 22. FINAL DUPLICATE CHECK
# ==========================================================

final_duplicates = combined.duplicated(
    subset=["Date", "Ticker"]
).sum()

print(
    f"\nFinal duplicate Date-Ticker rows: "
    f"{final_duplicates}"
)


# ==========================================================
# 23. FINAL TICKER CHECK
# ==========================================================

final_tickers = combined[
    "Ticker"
].nunique()

print(
    f"Final unique tickers: "
    f"{final_tickers}"
)


# ==========================================================
# 24. FINAL DATA QUALITY REPORT
# ==========================================================

quality_report = pd.DataFrame({

    "Column":
        combined.columns,

    "Data_Type":
        combined.dtypes.astype(
            str
        ).values,

    "Missing_Count":
        combined.isnull().sum().values,

    "Missing_Percentage":
        (
            combined
            .isnull()
            .mean()
            * 100
        )
})

quality_report.to_csv(
    TABLE_DIR
    / "04_integrated_data_quality.csv",
    index=False
)


# ==========================================================
# 25. SAVE INTEGRATED DATASET
# ==========================================================

integrated_file = (
    OUTPUT_DIR
    / "integrated_market_data_2016_2026.csv"
)

combined.to_csv(
    integrated_file,
    index=False
)


# ==========================================================
# 26. SAVE INTEGRATION SUMMARY
# ==========================================================

summary = {

    "Historical Rows":
        len(historical),

    "Current Rows":
        len(current),

    "Integrated Rows":
        len(combined),

    "Duplicate Rows Removed":
        duplicates_removed,

    "Final Columns":
        len(combined.columns),

    "Unique Tickers":
        combined["Ticker"].nunique(),

    "Unique Sectors":
        combined["Sector"].nunique(),

    "Start Date":
        combined["Date"].min(),

    "End Date":
        combined["Date"].max(),

    "Final Duplicate Date-Ticker":
        final_duplicates
}

summary_df = pd.DataFrame(
    summary.items(),
    columns=[
        "Metric",
        "Value"
    ]
)

summary_df.to_csv(
    TABLE_DIR
    / "04_integration_summary.csv",
    index=False
)


# ==========================================================
# 27. FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 70)
print("DATA INTEGRATION COMPLETED")
print("=" * 70)

print(
    f"\nIntegrated shape: "
    f"{combined.shape}"
)

print(
    f"Date range: "
    f"{combined['Date'].min().date()} "
    f"→ "
    f"{combined['Date'].max().date()}"
)

print(
    f"Unique tickers: "
    f"{combined['Ticker'].nunique()}"
)

print(
    f"Unique sectors: "
    f"{combined['Sector'].nunique()}"
)

print(
    f"Final duplicate Date-Ticker rows: "
    f"{final_duplicates}"
)

print("\nMissing values:")

print(
    combined.isnull().sum()
)

print("\nFiles created:")

print(
    f"1. {integrated_file}"
)

print(
    "2. "
    f"{TABLE_DIR / '04_sector_encoding.csv'}"
)

print(
    "3. "
    f"{TABLE_DIR / '04_integrated_data_quality.csv'}"
)

print(
    "4. "
    f"{TABLE_DIR / '04_integration_summary.csv'}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)