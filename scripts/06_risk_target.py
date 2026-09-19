import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "feature_engineered_market_data_2016_2026.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "risk_target_dataset_2016_2026.csv"
)

TARGET_SUMMARY_FILE = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "06_risk_target_summary.csv"
)

QUANTILE_FILE = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "06_risk_quantile_thresholds.csv"
)


# ==========================================================
# START
# ==========================================================

print("=" * 70)
print("STEP 06 - FUTURE VOLATILITY AND RISK TARGET")
print("=" * 70)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

print(f"\nInput shape: {df.shape}")

print(
    f"Date range: "
    f"{df['Date'].min().date()} → "
    f"{df['Date'].max().date()}"
)

print(
    f"Unique tickers: "
    f"{df['Ticker'].nunique()}"
)


# ==========================================================
# CALCULATE FUTURE RETURNS
# ==========================================================

print("\nCalculating future 20-trading-day volatility...")


# Future daily return:
#
# At day t:
#
# future return 1 = return at t+1
# future return 2 = return at t+2
# ...
#
# We shift the existing daily return backwards so that
# future information is aligned with today's row.

future_return_columns = []

for i in range(1, 21):

    column_name = f"Future_Return_{i}D"

    df[column_name] = (
        df.groupby("Ticker")["Daily_Return"]
        .shift(-i)
    )

    future_return_columns.append(
        column_name
    )


# ==========================================================
# FUTURE 20-DAY VOLATILITY
# ==========================================================

# Standard deviation of the NEXT 20 trading-day returns.
#
# Annualization:
# sqrt(252) = approximate number of trading days/year.

df["Future_20D_Volatility"] = (
    df[future_return_columns]
    .std(axis=1)
    * np.sqrt(252)
)


# ==========================================================
# REMOVE TEMPORARY FUTURE RETURN COLUMNS
# ==========================================================

df = df.drop(
    columns=future_return_columns
)


# ==========================================================
# CHECK FUTURE VOLATILITY
# ==========================================================

print("\nFuture volatility statistics:")

print(
    df["Future_20D_Volatility"]
    .describe()
)


# ==========================================================
# CREATE QUANTILE THRESHOLDS
# ==========================================================

valid_future_volatility = (
    df["Future_20D_Volatility"]
    .dropna()
)

low_threshold = (
    valid_future_volatility
    .quantile(1 / 3)
)

high_threshold = (
    valid_future_volatility
    .quantile(2 / 3)
)


print("\nRisk thresholds:")

print(
    f"LOW / MEDIUM threshold: "
    f"{low_threshold:.6f}"
)

print(
    f"MEDIUM / HIGH threshold: "
    f"{high_threshold:.6f}"
)


# ==========================================================
# CREATE RISK LEVEL
# ==========================================================

def assign_risk(volatility):

    if pd.isna(volatility):

        return np.nan

    elif volatility <= low_threshold:

        return "LOW"

    elif volatility <= high_threshold:

        return "MEDIUM"

    else:

        return "HIGH"


df["Risk_Level"] = (
    df["Future_20D_Volatility"]
    .apply(assign_risk)
)


# ==========================================================
# NUMERIC TARGET
# ==========================================================

# Keep both the human-readable class and numeric encoding.

risk_mapping = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
}

df["Risk_Level_ID"] = (
    df["Risk_Level"]
    .map(risk_mapping)
)


# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

print("\n" + "-" * 70)
print("RISK LEVEL DISTRIBUTION")
print("-" * 70)

risk_counts = (
    df["Risk_Level"]
    .value_counts(
        dropna=False
    )
)

risk_percentages = (
    df["Risk_Level"]
    .value_counts(
        normalize=True,
        dropna=False
    )
    * 100
)


for risk in ["LOW", "MEDIUM", "HIGH"]:

    count = risk_counts.get(
        risk,
        0
    )

    percentage = risk_percentages.get(
        risk,
        0
    )

    print(
        f"{risk:<8} "
        f"{count:>8} rows "
        f"({percentage:.2f}%)"
    )


missing_target = (
    df["Risk_Level"]
    .isna()
    .sum()
)

print(
    f"\nMissing Risk_Level: "
    f"{missing_target}"
)


# ==========================================================
# CHECK TARGET BY TICKER
# ==========================================================

ticker_target_counts = (
    pd.crosstab(
        df["Ticker"],
        df["Risk_Level"]
    )
)

ticker_target_counts.to_csv(
    BASE_DIR
    / "outputs"
    / "tables"
    / "06_risk_distribution_by_ticker.csv"
)


# ==========================================================
# SAVE THRESHOLDS
# ==========================================================

thresholds = pd.DataFrame({
    "Threshold": [
        "LOW_MEDIUM",
        "MEDIUM_HIGH"
    ],
    "Future_20D_Volatility": [
        low_threshold,
        high_threshold
    ]
})

thresholds.to_csv(
    QUANTILE_FILE,
    index=False
)


# ==========================================================
# TARGET SUMMARY
# ==========================================================

target_summary = pd.DataFrame({

    "Metric": [
        "Total Rows",
        "Rows With Future Volatility",
        "Rows Without Future Volatility",
        "LOW Count",
        "MEDIUM Count",
        "HIGH Count",
        "LOW Percentage",
        "MEDIUM Percentage",
        "HIGH Percentage",
        "LOW-MEDIUM Threshold",
        "MEDIUM-HIGH Threshold"
    ],

    "Value": [

        len(df),

        df["Future_20D_Volatility"]
        .notna()
        .sum(),

        df["Future_20D_Volatility"]
        .isna()
        .sum(),

        risk_counts.get(
            "LOW",
            0
        ),

        risk_counts.get(
            "MEDIUM",
            0
        ),

        risk_counts.get(
            "HIGH",
            0
        ),

        risk_percentages.get(
            "LOW",
            0
        ),

        risk_percentages.get(
            "MEDIUM",
            0
        ),

        risk_percentages.get(
            "HIGH",
            0
        ),

        low_threshold,

        high_threshold
    ]
})

target_summary.to_csv(
    TARGET_SUMMARY_FILE,
    index=False
)


# ==========================================================
# SAVE DATASET
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================================
# FINAL CHECK
# ==========================================================

print("\n" + "=" * 70)
print("TARGET CREATION SUMMARY")
print("=" * 70)

print(
    f"\nFinal shape: {df.shape}"
)

print(
    f"Future volatility column: "
    f"Future_20D_Volatility"
)

print(
    f"Target column: "
    f"Risk_Level"
)

print(
    f"Numeric target: "
    f"Risk_Level_ID"
)

print(
    f"\nSaved dataset:"
)

print(
    f"  ✓ {OUTPUT_FILE}"
)

print(
    f"\nSaved target summary:"
)

print(
    f"  ✓ {TARGET_SUMMARY_FILE}"
)

print(
    f"\nSaved thresholds:"
)

print(
    f"  ✓ {QUANTILE_FILE}"
)

print("\n" + "=" * 70)
print("STEP 06 COMPLETED")
print("=" * 70)