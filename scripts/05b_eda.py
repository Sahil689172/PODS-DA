import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    / "eda_da2"
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
# START
# ==========================================================

print("=" * 70)
print("STEP 05B - DA-2 EXPLORATORY DATA ANALYSIS")
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

print(f"Unique tickers: {df['Ticker'].nunique()}")
print(f"Unique sectors: {df['Sector'].nunique()}")


# ==========================================================
# 1. BASIC STATISTICS
# ==========================================================

print("\n" + "-" * 70)
print("BASIC STATISTICS")
print("-" * 70)

numeric_columns = [
    "Close",
    "Volume",
    "Daily_Return",
    "Volatility_20D",
    "MA_20",
    "MA_50",
    "MA_200",
    "Momentum_20D",
    "Volume_Ratio"
]

available_numeric = [
    col for col in numeric_columns
    if col in df.columns
]

summary = df[available_numeric].describe().T

summary.to_csv(
    TABLE_DIR / "05b_eda_summary_statistics.csv"
)

print(summary)


# ==========================================================
# 2. DAILY RETURN DISTRIBUTION
# ==========================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Daily_Return"].dropna(),
    bins=100,
    kde=True
)

plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_daily_return_distribution.png",
    dpi=300
)

plt.close()


# ==========================================================
# 3. VOLATILITY DISTRIBUTION
# ==========================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Volatility_20D"].dropna(),
    bins=100,
    kde=True
)

plt.title("Distribution of 20-Day Annualized Volatility")
plt.xlabel("20-Day Volatility")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_volatility_distribution.png",
    dpi=300
)

plt.close()


# ==========================================================
# 4. CLOSING PRICE TREND
# ==========================================================

# Calculate NIFTY-50 constituent average closing price
# for a broad visualization of the dataset over time.

daily_average_close = (
    df.groupby("Date")["Close"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    daily_average_close["Date"],
    daily_average_close["Close"]
)

plt.title(
    "Average Closing Price Across NIFTY 50 Constituents"
)

plt.xlabel("Date")
plt.ylabel("Average Closing Price")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_average_close_trend.png",
    dpi=300
)

plt.close()


# ==========================================================
# 5. CORRELATION HEATMAP
# ==========================================================

correlation_features = [
    "Daily_Return",
    "Log_Return",
    "High_Low_Range",
    "Open_Close_Range",
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
    "Volume_Ratio",
    "MA20_MA50_Ratio",
    "MA50_MA200_Ratio",
    "52Week_Position"
]

available_correlation_features = [
    col
    for col in correlation_features
    if col in df.columns
]

corr_matrix = (
    df[available_correlation_features]
    .corr()
)

corr_matrix.to_csv(
    TABLE_DIR / "05b_feature_correlation_matrix.csv"
)

plt.figure(
    figsize=(16, 13)
)

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.3
)

plt.title(
    "Correlation Heatmap of Engineered Features"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_feature_correlation_heatmap.png",
    dpi=300
)

plt.close()


# ==========================================================
# 6. AVERAGE VOLATILITY BY SECTOR
# ==========================================================

sector_volatility = (
    df.groupby("Sector")["Volatility_20D"]
    .mean()
    .sort_values(ascending=False)
)

sector_volatility.to_csv(
    TABLE_DIR / "05b_sector_volatility.csv"
)

plt.figure(figsize=(12, 7))

sector_volatility.plot(
    kind="bar"
)

plt.title(
    "Average 20-Day Volatility by Sector"
)

plt.xlabel("Sector")
plt.ylabel("Average 20-Day Volatility")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_sector_volatility.png",
    dpi=300
)

plt.close()


# ==========================================================
# 7. AVERAGE VOLATILITY BY STOCK
# ==========================================================

stock_volatility = (
    df.groupby("Ticker")["Volatility_20D"]
    .mean()
    .sort_values(ascending=False)
)

stock_volatility.to_csv(
    TABLE_DIR / "05b_stock_volatility.csv"
)

plt.figure(figsize=(14, 7))

stock_volatility.plot(
    kind="bar"
)

plt.title(
    "Average 20-Day Volatility by Stock"
)

plt.xlabel("Ticker")
plt.ylabel("Average 20-Day Volatility")

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_stock_volatility.png",
    dpi=300
)

plt.close()


# ==========================================================
# 8. VOLUME VS DAILY RETURN
# ==========================================================

# Sample the data to make the scatter plot manageable.

plot_sample = df[
    [
        "Volume",
        "Daily_Return"
    ]
].dropna()

if len(plot_sample) > 30000:

    plot_sample = plot_sample.sample(
        30000,
        random_state=42
    )


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=plot_sample,
    x="Volume",
    y="Daily_Return",
    alpha=0.3
)

plt.title(
    "Trading Volume vs Daily Return"
)

plt.xlabel("Volume")
plt.ylabel("Daily Return")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_volume_vs_return.png",
    dpi=300
)

plt.close()


# ==========================================================
# 9. PRICE VS MA200
# ==========================================================

plot_sample = df[
    [
        "Close",
        "MA_200"
    ]
].dropna()

if len(plot_sample) > 30000:

    plot_sample = plot_sample.sample(
        30000,
        random_state=42
    )


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=plot_sample,
    x="MA_200",
    y="Close",
    alpha=0.3
)

plt.title(
    "Closing Price vs 200-Day Moving Average"
)

plt.xlabel("MA 200")
plt.ylabel("Closing Price")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_close_vs_ma200.png",
    dpi=300
)

plt.close()


# ==========================================================
# 10. FEATURE DISTRIBUTIONS
# ==========================================================

distribution_features = [
    "Momentum_5D",
    "Momentum_20D",
    "Volatility_20D",
    "Volatility_60D",
    "Volume_Ratio",
    "Price_to_MA20",
    "Price_to_MA200",
    "52Week_Position"
]

for feature in distribution_features:

    if feature not in df.columns:
        continue

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df[feature].dropna(),
        bins=80,
        kde=True
    )

    plt.title(
        f"Distribution of {feature}"
    )

    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()

    filename = (
        feature.lower()
        + "_distribution.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close()


# ==========================================================
# 11. FEATURE MISSINGNESS
# ==========================================================

feature_columns = [
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
    "52Week_Position"
]

available_features = [
    col
    for col in feature_columns
    if col in df.columns
]

missing_percent = (
    df[available_features]
    .isna()
    .mean()
    * 100
).sort_values(ascending=False)

missing_percent.to_csv(
    TABLE_DIR / "05b_feature_missingness.csv"
)

plt.figure(figsize=(12, 7))

missing_percent.plot(
    kind="bar"
)

plt.title(
    "Missing Percentage of Engineered Features"
)

plt.xlabel("Feature")
plt.ylabel("Missing Percentage (%)")

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "09_feature_missingness.png",
    dpi=300
)

plt.close()


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 70)
print("DA-2 EDA COMPLETED")
print("=" * 70)

print("\nGraphs saved to:")

print(
    f"  {OUTPUT_DIR}"
)

print("\nTables saved to:")

print(
    f"  {TABLE_DIR}"
)

print("\nGenerated visualizations:")

for file in sorted(OUTPUT_DIR.glob("*.png")):

    print(
        f"  ✓ {file.name}"
    )

print("\n" + "=" * 70)