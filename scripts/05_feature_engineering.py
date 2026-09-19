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
    / "integrated_market_data_ready_for_ml.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "feature_engineered_market_data_2016_2026.csv"
)

FEATURE_SUMMARY_FILE = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "05_feature_summary.csv"
)

MISSING_REPORT_FILE = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "05_feature_missing_report.csv"
)


# ==========================================================
# START
# ==========================================================

print("=" * 70)
print("STEP 05 - FEATURE ENGINEERING")
print("=" * 70)


# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
FEATURE_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values(["Ticker", "Date"])
    .reset_index(drop=True)
)

print(f"\nInput file: {INPUT_FILE}")
print(f"Initial shape: {df.shape}")

print(
    f"Date range: "
    f"{df['Date'].min().date()} → "
    f"{df['Date'].max().date()}"
)

print(f"Unique tickers: {df['Ticker'].nunique()}")


# ==========================================================
# BASIC DATA VALIDATION
# ==========================================================

required_columns = [
    "Ticker",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Required columns are missing: {missing_required}"
    )

print("\nRequired columns verified.")


# ==========================================================
# GROUP BY TICKER
# ==========================================================

group = df.groupby("Ticker", group_keys=False)


# ==========================================================
# 1. DAILY RETURN
# ==========================================================

# Daily percentage change in closing price.

df["Daily_Return"] = (
    group["Close"].pct_change()
)


# ==========================================================
# 2. LOG RETURN
# ==========================================================

# Logarithmic daily return.

df["Log_Return"] = np.log(
    df["Close"] /
    group["Close"].shift(1)
)


# ==========================================================
# 3. PRICE RANGE FEATURES
# ==========================================================

# Absolute intraday price range.

df["Price_Range"] = (
    df["High"] - df["Low"]
)


# High-Low range relative to closing price.

df["High_Low_Range"] = (
    (df["High"] - df["Low"])
    / df["Close"]
)


# Open-Close percentage movement.

df["Open_Close_Range"] = (
    (df["Close"] - df["Open"])
    / df["Open"]
)


# ==========================================================
# 4. MOVING AVERAGES
# ==========================================================

df["MA_20"] = (
    group["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=20
        ).mean()
    )
)

df["MA_50"] = (
    group["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=50,
            min_periods=50
        ).mean()
    )
)

df["MA_100"] = (
    group["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=100,
            min_periods=100
        ).mean()
    )
)

df["MA_200"] = (
    group["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=200,
            min_periods=200
        ).mean()
    )
)


# ==========================================================
# 5. PRICE VS MOVING AVERAGE
# ==========================================================

df["Price_to_MA20"] = (
    df["Close"] /
    df["MA_20"]
)

df["Price_to_MA50"] = (
    df["Close"] /
    df["MA_50"]
)

df["Price_to_MA200"] = (
    df["Close"] /
    df["MA_200"]
)


# ==========================================================
# 6. MOMENTUM FEATURES
# ==========================================================

df["Momentum_5D"] = (
    group["Close"]
    .transform(
        lambda x:
        x / x.shift(5) - 1
    )
)

df["Momentum_10D"] = (
    group["Close"]
    .transform(
        lambda x:
        x / x.shift(10) - 1
    )
)

df["Momentum_20D"] = (
    group["Close"]
    .transform(
        lambda x:
        x / x.shift(20) - 1
    )
)


# ==========================================================
# 7. RATE OF CHANGE
# ==========================================================

df["ROC_10"] = (
    group["Close"]
    .transform(
        lambda x:
        x.pct_change(10)
    )
)

df["ROC_20"] = (
    group["Close"]
    .transform(
        lambda x:
        x.pct_change(20)
    )
)


# ==========================================================
# 8. VOLATILITY FEATURES
# ==========================================================

# Annualized rolling volatility.
# 252 = conventional number of trading days/year.

df["Volatility_5D"] = (
    group["Daily_Return"]
    .transform(
        lambda x:
        x.rolling(
            window=5,
            min_periods=5
        ).std()
        * np.sqrt(252)
    )
)

df["Volatility_10D"] = (
    group["Daily_Return"]
    .transform(
        lambda x:
        x.rolling(
            window=10,
            min_periods=10
        ).std()
        * np.sqrt(252)
    )
)

df["Volatility_20D"] = (
    group["Daily_Return"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=20
        ).std()
        * np.sqrt(252)
    )
)

df["Volatility_60D"] = (
    group["Daily_Return"]
    .transform(
        lambda x:
        x.rolling(
            window=60,
            min_periods=60
        ).std()
        * np.sqrt(252)
    )
)


# ==========================================================
# 9. DOWNSIDE VOLATILITY
# ==========================================================

# Downside deviation measures the magnitude of negative
# returns over a rolling window.
#
# Positive returns are treated as zero.
#
# Formula:
#
# sqrt(mean(min(Return, 0)^2))
#
# Annualized using sqrt(252).
#
# min_periods=5 ensures the feature becomes available
# relatively early and avoids an all-missing feature.

df["Downside_Return_Squared"] = (
    np.minimum(
        df["Daily_Return"],
        0
    ) ** 2
)

df["Downside_Volatility"] = (
    group["Downside_Return_Squared"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=5
        )
        .mean()
        .pow(0.5)
        * np.sqrt(252)
    )
)

# Temporary calculation column is removed.

df.drop(
    columns=["Downside_Return_Squared"],
    inplace=True
)


# ==========================================================
# 10. VOLUME FEATURES
# ==========================================================

df["Volume_MA20"] = (
    group["Volume"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=20
        ).mean()
    )
)

df["Volume_Ratio"] = (
    df["Volume"] /
    df["Volume_MA20"]
)

df["Volume_Change"] = (
    group["Volume"].pct_change()
)


# ==========================================================
# 11. MOVING AVERAGE RELATIONSHIPS
# ==========================================================

df["MA20_MA50_Ratio"] = (
    df["MA_20"] /
    df["MA_50"]
)

df["MA50_MA200_Ratio"] = (
    df["MA_50"] /
    df["MA_200"]
)


# ==========================================================
# 12. ROLLING 52-WEEK POSITION
# ==========================================================

rolling_52_high = (
    group["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=252,
            min_periods=252
        ).max()
    )
)

rolling_52_low = (
    group["Close"]
    .transform(
        lambda x:
        x.rolling(
            window=252,
            min_periods=252
        ).min()
    )
)

denominator = (
    rolling_52_high -
    rolling_52_low
)

# Avoid division by zero.

denominator = denominator.replace(
    0,
    np.nan
)

df["52Week_Position"] = (
    (df["Close"] - rolling_52_low)
    / denominator
)


# ==========================================================
# 13. FUNDAMENTAL MISSINGNESS INDICATORS
# ==========================================================

# These indicate whether fundamental information is available.
#
# Missing fundamental values themselves are NOT replaced by zero.

df["PE_Ratio_Missing"] = (
    df["PE_Ratio"]
    .isna()
    .astype(int)
)

df["Dividend_Yield_Missing"] = (
    df["Dividend_Yield"]
    .isna()
    .astype(int)
)

df["Beta_Missing"] = (
    df["Beta"]
    .isna()
    .astype(int)
)


# ==========================================================
# 14. CLEAN INFINITE VALUES
# ==========================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ==========================================================
# 15. ENGINEERED FEATURE LIST
# ==========================================================

engineered_features = [

    # Returns
    "Daily_Return",
    "Log_Return",

    # Price range
    "Price_Range",
    "High_Low_Range",
    "Open_Close_Range",

    # Moving averages
    "MA_20",
    "MA_50",
    "MA_100",
    "MA_200",

    # Price relative to moving averages
    "Price_to_MA20",
    "Price_to_MA50",
    "Price_to_MA200",

    # Momentum
    "Momentum_5D",
    "Momentum_10D",
    "Momentum_20D",

    # Rate of change
    "ROC_10",
    "ROC_20",

    # Volatility
    "Volatility_5D",
    "Volatility_10D",
    "Volatility_20D",
    "Volatility_60D",
    "Downside_Volatility",

    # Volume
    "Volume_MA20",
    "Volume_Ratio",
    "Volume_Change",

    # Moving average relationships
    "MA20_MA50_Ratio",
    "MA50_MA200_Ratio",

    # 52-week position
    "52Week_Position",

    # Fundamental availability
    "PE_Ratio_Missing",
    "Dividend_Yield_Missing",
    "Beta_Missing"
]


# ==========================================================
# 16. VERIFY FEATURES
# ==========================================================

print("\n" + "-" * 70)
print("ENGINEERED FEATURES")
print("-" * 70)

missing_features = []

for feature in engineered_features:

    if feature in df.columns:
        print(f"  ✓ {feature}")

    else:
        print(f"  ✗ {feature} MISSING")
        missing_features.append(feature)


if missing_features:

    raise ValueError(
        "Feature engineering failed. "
        f"Missing features: {missing_features}"
    )


# ==========================================================
# 17. FEATURE SUMMARY
# ==========================================================

feature_summary = []

for feature in engineered_features:

    feature_summary.append({

        "Feature": feature,

        "Data_Type": str(
            df[feature].dtype
        ),

        "Missing_Count": int(
            df[feature].isna().sum()
        ),

        "Missing_Percentage": (
            df[feature].isna().mean()
            * 100
        ),

        "Unique_Values": int(
            df[feature].nunique()
        ),

        "Mean": (
            df[feature].mean()
            if pd.api.types.is_numeric_dtype(
                df[feature]
            )
            else np.nan
        ),

        "Std": (
            df[feature].std()
            if pd.api.types.is_numeric_dtype(
                df[feature]
            )
            else np.nan
        ),

        "Min": (
            df[feature].min()
            if pd.api.types.is_numeric_dtype(
                df[feature]
            )
            else np.nan
        ),

        "Max": (
            df[feature].max()
            if pd.api.types.is_numeric_dtype(
                df[feature]
            )
            else np.nan
        )
    })


feature_summary_df = pd.DataFrame(
    feature_summary
)


# ==========================================================
# 18. MISSING VALUE REPORT
# ==========================================================

missing_report = (
    df[engineered_features]
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
    .reset_index()
)

missing_report.columns = [
    "Feature",
    "Missing_Count"
]

missing_report["Missing_Percentage"] = (
    missing_report["Missing_Count"]
    / len(df)
    * 100
)


# ==========================================================
# 19. CHECK FOR 100% MISSING FEATURES
# ==========================================================

all_missing_features = (
    missing_report[
        missing_report["Missing_Count"] == len(df)
    ]["Feature"]
    .tolist()
)

if all_missing_features:

    print("\nWARNING:")
    print("The following engineered features are 100% missing:")

    for feature in all_missing_features:
        print(f"  ✗ {feature}")

else:

    print(
        "\n✓ No engineered feature is 100% missing."
    )


# ==========================================================
# 20. PRINT IMPORTANT MISSINGNESS INFORMATION
# ==========================================================

print("\nMissing-value report:")
print(
    missing_report.to_string(
        index=False
    )
)


# ==========================================================
# 21. SAVE OUTPUTS
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

feature_summary_df.to_csv(
    FEATURE_SUMMARY_FILE,
    index=False
)

missing_report.to_csv(
    MISSING_REPORT_FILE,
    index=False
)


# ==========================================================
# 22. FINAL INFORMATION
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

print(
    f"\nFinal shape: {df.shape}"
)

print(
    f"Date range: "
    f"{df['Date'].min().date()} → "
    f"{df['Date'].max().date()}"
)

print(
    f"Unique tickers: "
    f"{df['Ticker'].nunique()}"
)

print(
    f"Engineered features created: "
    f"{len(engineered_features)}"
)

print("\nKey feature checks:")

print(
    f"  Daily_Return missing: "
    f"{df['Daily_Return'].isna().sum()}"
)

print(
    f"  Volatility_20D missing: "
    f"{df['Volatility_20D'].isna().sum()}"
)

print(
    f"  Downside_Volatility missing: "
    f"{df['Downside_Volatility'].isna().sum()}"
)

print(
    f"  52Week_Position missing: "
    f"{df['52Week_Position'].isna().sum()}"
)

print("\nOutput files:")

print(
    f"  ✓ {OUTPUT_FILE}"
)

print(
    f"  ✓ {FEATURE_SUMMARY_FILE}"
)

print(
    f"  ✓ {MISSING_REPORT_FILE}"
)

print("\n" + "=" * 70)
print("STEP 05 COMPLETED")
print("=" * 70)