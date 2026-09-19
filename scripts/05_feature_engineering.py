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
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

print(f"\nInput file: {INPUT_FILE}")
print(f"Initial shape: {df.shape}")

print(
    f"Date range: "
    f"{df['Date'].min().date()} → {df['Date'].max().date()}"
)

print(f"Unique tickers: {df['Ticker'].nunique()}")


# ==========================================================
# GROUP BY TICKER
# ==========================================================

group = df.groupby("Ticker", group_keys=False)


# ==========================================================
# 1. DAILY RETURN
# ==========================================================

# Recalculate to ensure consistency across the integrated dataset.

df["Daily_Return"] = group["Close"].pct_change()


# ==========================================================
# 2. LOG RETURN
# ==========================================================

df["Log_Return"] = np.log(
    df["Close"] / group["Close"].shift(1)
)


# ==========================================================
# 3. PRICE RANGE FEATURES
# ==========================================================

df["Price_Range"] = df["High"] - df["Low"]

df["High_Low_Range"] = (
    (df["High"] - df["Low"])
    / df["Close"]
)

df["Open_Close_Range"] = (
    (df["Close"] - df["Open"])
    / df["Open"]
)


# ==========================================================
# 4. MOVING AVERAGES
# ==========================================================

df["MA_20"] = (
    group["Close"]
    .transform(lambda x: x.rolling(20).mean())
)

df["MA_50"] = (
    group["Close"]
    .transform(lambda x: x.rolling(50).mean())
)

df["MA_100"] = (
    group["Close"]
    .transform(lambda x: x.rolling(100).mean())
)

df["MA_200"] = (
    group["Close"]
    .transform(lambda x: x.rolling(200).mean())
)


# ==========================================================
# 5. PRICE VS MOVING AVERAGE
# ==========================================================

df["Price_to_MA20"] = (
    df["Close"] / df["MA_20"]
)

df["Price_to_MA50"] = (
    df["Close"] / df["MA_50"]
)

df["Price_to_MA200"] = (
    df["Close"] / df["MA_200"]
)


# ==========================================================
# 6. MOMENTUM FEATURES
# ==========================================================

df["Momentum_5D"] = (
    group["Close"].transform(
        lambda x: x / x.shift(5) - 1
    )
)

df["Momentum_10D"] = (
    group["Close"].transform(
        lambda x: x / x.shift(10) - 1
    )
)

df["Momentum_20D"] = (
    group["Close"].transform(
        lambda x: x / x.shift(20) - 1
    )
)


# ==========================================================
# 7. RATE OF CHANGE
# ==========================================================

df["ROC_10"] = (
    group["Close"].transform(
        lambda x: x.pct_change(10)
    )
)

df["ROC_20"] = (
    group["Close"].transform(
        lambda x: x.pct_change(20)
    )
)


# ==========================================================
# 8. VOLATILITY FEATURES
# ==========================================================

# Annualized rolling volatility based on daily returns.
# sqrt(252) is the conventional number of trading days/year.

df["Volatility_5D"] = (
    group["Daily_Return"]
    .transform(
        lambda x: x.rolling(5).std() * np.sqrt(252)
    )
)

df["Volatility_10D"] = (
    group["Daily_Return"]
    .transform(
        lambda x: x.rolling(10).std() * np.sqrt(252)
    )
)

df["Volatility_20D"] = (
    group["Daily_Return"]
    .transform(
        lambda x: x.rolling(20).std() * np.sqrt(252)
    )
)

df["Volatility_60D"] = (
    group["Daily_Return"]
    .transform(
        lambda x: x.rolling(60).std() * np.sqrt(252)
    )
)


# ==========================================================
# 9. DOWNSIDE VOLATILITY
# ==========================================================

def downside_volatility(series):

    negative_returns = series.where(
        series < 0
    )

    return (
        negative_returns
        .rolling(20)
        .std()
        * np.sqrt(252)
    )


df["Downside_Volatility"] = (
    group["Daily_Return"]
    .transform(downside_volatility)
)


# ==========================================================
# 10. VOLUME FEATURES
# ==========================================================

df["Volume_MA20"] = (
    group["Volume"]
    .transform(
        lambda x: x.rolling(20).mean()
    )
)

df["Volume_Ratio"] = (
    df["Volume"] / df["Volume_MA20"]
)

df["Volume_Change"] = (
    group["Volume"].pct_change()
)


# ==========================================================
# 11. MOVING AVERAGE RELATIONSHIPS
# ==========================================================

df["MA20_MA50_Ratio"] = (
    df["MA_20"] / df["MA_50"]
)

df["MA50_MA200_Ratio"] = (
    df["MA_50"] / df["MA_200"]
)


# ==========================================================
# 12. ROLLING 52-WEEK POSITION
# ==========================================================

rolling_52_high = (
    group["Close"]
    .transform(
        lambda x: x.rolling(252).max()
    )
)

rolling_52_low = (
    group["Close"]
    .transform(
        lambda x: x.rolling(252).min()
    )
)

df["52Week_Position"] = (
    (df["Close"] - rolling_52_low)
    / (rolling_52_high - rolling_52_low)
)


# ==========================================================
# 13. FUNDAMENTAL MISSINGNESS INDICATORS
# ==========================================================

# These indicate whether fundamental information is available.
# We do NOT replace missing fundamental values with zero.

df["PE_Ratio_Missing"] = (
    df["PE_Ratio"].isna().astype(int)
)

df["Dividend_Yield_Missing"] = (
    df["Dividend_Yield"].isna().astype(int)
)

df["Beta_Missing"] = (
    df["Beta"].isna().astype(int)
)


# ==========================================================
# 14. CLEAN INF VALUES
# ==========================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ==========================================================
# FEATURE LIST
# ==========================================================

engineered_features = [
    "Daily_Return",
    "Log_Return",
    "Price_Range",
    "High_Low_Range",
    "Open_Close_Range",
    "MA_20",
    "MA_50",
    "MA_100",
    "MA_200",
    "Price_to_MA20",
    "Price_to_MA50",
    "Price_to_MA200",
    "Momentum_5D",
    "Momentum_10D",
    "Momentum_20D",
    "ROC_10",
    "ROC_20",
    "Volatility_5D",
    "Volatility_10D",
    "Volatility_20D",
    "Volatility_60D",
    "Downside_Volatility",
    "Volume_MA20",
    "Volume_Ratio",
    "Volume_Change",
    "MA20_MA50_Ratio",
    "MA50_MA200_Ratio",
    "52Week_Position",
    "PE_Ratio_Missing",
    "Dividend_Yield_Missing",
    "Beta_Missing"
]


# ==========================================================
# VERIFY FEATURES
# ==========================================================

print("\n" + "-" * 70)
print("ENGINEERED FEATURES")
print("-" * 70)

for feature in engineered_features:

    if feature in df.columns:
        print(f"  ✓ {feature}")
    else:
        print(f"  ✗ {feature} MISSING")


# ==========================================================
# FEATURE SUMMARY
# ==========================================================

feature_summary = []

for feature in engineered_features:

    feature_summary.append({
        "Feature": feature,
        "Data_Type": str(df[feature].dtype),
        "Missing_Count": int(df[feature].isna().sum()),
        "Missing_Percentage": (
            df[feature].isna().mean() * 100
        ),
        "Unique_Values": int(
            df[feature].nunique()
        )
    })


feature_summary_df = pd.DataFrame(
    feature_summary
)


# ==========================================================
# MISSING VALUE REPORT
# ==========================================================

missing_report = (
    df[engineered_features]
    .isna()
    .sum()
    .sort_values(ascending=False)
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
# SAVE OUTPUTS
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
# FINAL INFORMATION
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 70)

print(f"\nFinal shape: {df.shape}")

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