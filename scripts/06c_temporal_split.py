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

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "temporal_split"
)

TABLE_DIR = (
    BASE_DIR
    / "outputs"
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
# TEMPORAL SPLIT DATES
# ==========================================================

TRAIN_START = "2016-01-01"
TRAIN_END = "2023-12-31"

VALIDATION_START = "2024-01-01"
VALIDATION_END = "2024-12-31"

TEST_START = "2025-01-01"
TEST_END = "2026-12-31"


# ==========================================================
# START
# ==========================================================

print("=" * 70)
print("STEP 06C - TEMPORAL SPLIT + LEAKAGE-SAFE RISK TARGET")
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
    f"Full date range: "
    f"{df['Date'].min().date()} → "
    f"{df['Date'].max().date()}"
)

print(
    f"Unique tickers: "
    f"{df['Ticker'].nunique()}"
)


# ==========================================================
# CALCULATE FUTURE 20-DAY VOLATILITY
# ==========================================================

print(
    "\nCalculating future 20-trading-day volatility..."
)


# We calculate the next 20 trading-day returns
# separately for each stock.

future_return_columns = []

for i in range(1, 21):

    column_name = f"_Future_Return_{i}"

    df[column_name] = (
        df.groupby("Ticker")["Daily_Return"]
        .shift(-i)
    )

    future_return_columns.append(
        column_name
    )


df["Future_20D_Volatility"] = (
    df[future_return_columns]
    .std(axis=1)
    * np.sqrt(252)
)


# ==========================================================
# REMOVE TEMPORARY COLUMNS
# ==========================================================

df = df.drop(
    columns=future_return_columns
)


# ==========================================================
# CREATE TEMPORAL SPLITS
# ==========================================================

train = df[
    (df["Date"] >= TRAIN_START)
    & (df["Date"] <= TRAIN_END)
].copy()

validation = df[
    (df["Date"] >= VALIDATION_START)
    & (df["Date"] <= VALIDATION_END)
].copy()

test = df[
    (df["Date"] >= TEST_START)
    & (df["Date"] <= TEST_END)
].copy()


# ==========================================================
# CHECK SPLIT SIZES
# ==========================================================

print("\n" + "-" * 70)
print("TEMPORAL SPLIT")
print("-" * 70)

print(
    f"Training   : {len(train):>8,} rows | "
    f"{train['Date'].min().date()} → "
    f"{train['Date'].max().date()}"
)

print(
    f"Validation : {len(validation):>8,} rows | "
    f"{validation['Date'].min().date()} → "
    f"{validation['Date'].max().date()}"
)

print(
    f"Test       : {len(test):>8,} rows | "
    f"{test['Date'].min().date()} → "
    f"{test['Date'].max().date()}"
)


# ==========================================================
# CHECK CHRONOLOGICAL ORDER
# ==========================================================

if train["Date"].max() < validation["Date"].min():

    print(
        "\n✓ Training occurs entirely before validation."
    )

else:

    raise ValueError(
        "ERROR: Training and validation periods overlap."
    )


if validation["Date"].max() < test["Date"].min():

    print(
        "✓ Validation occurs entirely before test."
    )

else:

    raise ValueError(
        "ERROR: Validation and test periods overlap."
    )


# ==========================================================
# FIT RISK THRESHOLDS ONLY ON TRAINING DATA
# ==========================================================

print("\n" + "-" * 70)
print("FITTING RISK THRESHOLDS ON TRAINING DATA ONLY")
print("-" * 70)


train_future_volatility = (
    train["Future_20D_Volatility"]
    .dropna()
)


low_threshold = (
    train_future_volatility
    .quantile(1 / 3)
)

high_threshold = (
    train_future_volatility
    .quantile(2 / 3)
)


print(
    f"\nTraining LOW/MEDIUM threshold: "
    f"{low_threshold:.6f}"
)

print(
    f"Training MEDIUM/HIGH threshold: "
    f"{high_threshold:.6f}"
)


# ==========================================================
# APPLY SAME THRESHOLDS TO ALL SPLITS
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


for dataset in [
    train,
    validation,
    test
]:

    dataset["Risk_Level"] = (
        dataset["Future_20D_Volatility"]
        .apply(assign_risk)
    )


# ==========================================================
# NUMERIC TARGET
# ==========================================================

risk_mapping = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
}


for dataset in [
    train,
    validation,
    test
]:

    dataset["Risk_Level_ID"] = (
        dataset["Risk_Level"]
        .map(risk_mapping)
    )


# ==========================================================
# TARGET DISTRIBUTION FUNCTION
# ==========================================================

def print_target_distribution(
    dataset,
    name
):

    counts = (
        dataset["Risk_Level"]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH"]
        )
        .fillna(0)
    )

    valid_count = counts.sum()

    print(
        f"\n{name}"
    )

    print(
        f"  LOW    : {int(counts['LOW']):>8,} "
        f"({counts['LOW'] / valid_count * 100:.2f}%)"
    )

    print(
        f"  MEDIUM : {int(counts['MEDIUM']):>8,} "
        f"({counts['MEDIUM'] / valid_count * 100:.2f}%)"
    )

    print(
        f"  HIGH   : {int(counts['HIGH']):>8,} "
        f"({counts['HIGH'] / valid_count * 100:.2f}%)"
    )

    print(
        f"  Missing: "
        f"{dataset['Risk_Level'].isna().sum():,}"
    )


# ==========================================================
# DISPLAY TARGET DISTRIBUTIONS
# ==========================================================

print("\n" + "-" * 70)
print("RISK DISTRIBUTION AFTER TRAINING-ONLY THRESHOLDING")
print("-" * 70)

print_target_distribution(
    train,
    "TRAINING"
)

print_target_distribution(
    validation,
    "VALIDATION"
)

print_target_distribution(
    test,
    "TEST"
)


# ==========================================================
# SAVE SPLITS
# ==========================================================

train_file = (
    OUTPUT_DIR
    / "train_2016_2023.csv"
)

validation_file = (
    OUTPUT_DIR
    / "validation_2024.csv"
)

test_file = (
    OUTPUT_DIR
    / "test_2025_2026.csv"
)


train.to_csv(
    train_file,
    index=False
)

validation.to_csv(
    validation_file,
    index=False
)

test.to_csv(
    test_file,
    index=False
)


# ==========================================================
# SAVE THRESHOLDS
# ==========================================================

threshold_table = pd.DataFrame({

    "Threshold": [
        "LOW_MEDIUM",
        "MEDIUM_HIGH"
    ],

    "Value": [
        low_threshold,
        high_threshold
    ],

    "Fitted_On": [
        "Training 2016-2023",
        "Training 2016-2023"
    ]
})


threshold_file = (
    TABLE_DIR
    / "06c_training_risk_thresholds.csv"
)

threshold_table.to_csv(
    threshold_file,
    index=False
)


# ==========================================================
# CREATE SPLIT SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Dataset": [
        "Training",
        "Validation",
        "Test"
    ],

    "Start_Date": [
        train["Date"].min().date(),
        validation["Date"].min().date(),
        test["Date"].min().date()
    ],

    "End_Date": [
        train["Date"].max().date(),
        validation["Date"].max().date(),
        test["Date"].max().date()
    ],

    "Rows": [
        len(train),
        len(validation),
        len(test)
    ],

    "Tickers": [
        train["Ticker"].nunique(),
        validation["Ticker"].nunique(),
        test["Ticker"].nunique()
    ],

    "Missing_Target": [
        train["Risk_Level"].isna().sum(),
        validation["Risk_Level"].isna().sum(),
        test["Risk_Level"].isna().sum()
    ]
})


summary_file = (
    TABLE_DIR
    / "06c_temporal_split_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ==========================================================
# TARGET DISTRIBUTION TABLE
# ==========================================================

distribution_rows = []

for dataset_name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test)
]:

    counts = (
        dataset["Risk_Level"]
        .value_counts()
    )

    valid = (
        dataset["Risk_Level"]
        .notna()
        .sum()
    )

    for risk in [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]:

        count = counts.get(
            risk,
            0
        )

        distribution_rows.append({

            "Dataset": dataset_name,

            "Risk_Level": risk,

            "Count": count,

            "Percentage": (
                count / valid * 100
                if valid > 0
                else 0
            )
        })


distribution_table = pd.DataFrame(
    distribution_rows
)

distribution_file = (
    TABLE_DIR
    / "06c_temporal_risk_distribution.csv"
)

distribution_table.to_csv(
    distribution_file,
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("STEP 06C COMPLETED")
print("=" * 70)

print("\nSaved datasets:")

print(
    f"  ✓ {train_file}"
)

print(
    f"  ✓ {validation_file}"
)

print(
    f"  ✓ {test_file}"
)

print("\nSaved tables:")

print(
    f"  ✓ {threshold_file}"
)

print(
    f"  ✓ {summary_file}"
)

print(
    f"  ✓ {distribution_file}"
)

print("\nImportant methodological rule:")
print(
    "Risk thresholds were fitted ONLY on training data "
    "(2016-2023)."
)

print(
    "The same thresholds were applied unchanged to "
    "validation and test data."
)

print("\n" + "=" * 70)