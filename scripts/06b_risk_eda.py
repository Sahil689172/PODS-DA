import pandas as pd
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
    / "risk_target_dataset_2016_2026.csv"
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
print("STEP 06B - RISK TARGET EDA")
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

print(
    f"Unique sectors: "
    f"{df['Sector'].nunique()}"
)


# ==========================================================
# RISK COUNTS
# ==========================================================

risk_order = [
    "LOW",
    "MEDIUM",
    "HIGH"
]

risk_counts = (
    df["Risk_Level"]
    .value_counts()
    .reindex(risk_order)
    .fillna(0)
)

risk_percentages = (
    risk_counts
    / risk_counts.sum()
    * 100
)


print("\n" + "-" * 70)
print("RISK CLASS DISTRIBUTION")
print("-" * 70)

for risk in risk_order:

    print(
        f"{risk:<8} "
        f"{int(risk_counts[risk]):>8} rows "
        f"({risk_percentages[risk]:.2f}%)"
    )


# ==========================================================
# 1. RISK CLASS DISTRIBUTION
# ==========================================================

plt.figure(figsize=(9, 6))

bars = plt.bar(
    risk_counts.index,
    risk_counts.values
)

plt.title(
    "Distribution of Future Risk Levels"
)

plt.xlabel("Risk Level")
plt.ylabel("Number of Observations")

for bar, value in zip(
    bars,
    risk_counts.values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{int(value):,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "10_risk_class_distribution.png",
    dpi=300
)

plt.close()


# ==========================================================
# 2. FUTURE VOLATILITY DISTRIBUTION
# ==========================================================

low_threshold = (
    df["Future_20D_Volatility"]
    .quantile(1 / 3)
)

high_threshold = (
    df["Future_20D_Volatility"]
    .quantile(2 / 3)
)


plt.figure(figsize=(11, 6))

sns.histplot(
    df["Future_20D_Volatility"].dropna(),
    bins=100,
    kde=True
)

plt.axvline(
    low_threshold,
    linestyle="--",
    linewidth=2,
    label=f"LOW/MEDIUM = {low_threshold:.4f}"
)

plt.axvline(
    high_threshold,
    linestyle="--",
    linewidth=2,
    label=f"MEDIUM/HIGH = {high_threshold:.4f}"
)

plt.title(
    "Future 20-Day Volatility with Risk Thresholds"
)

plt.xlabel(
    "Future 20-Day Annualized Volatility"
)

plt.ylabel("Frequency")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "11_future_volatility_thresholds.png",
    dpi=300
)

plt.close()


# ==========================================================
# 3. RISK DISTRIBUTION BY SECTOR
# ==========================================================

sector_risk = pd.crosstab(
    df["Sector"],
    df["Risk_Level"]
)

sector_risk = sector_risk.reindex(
    columns=risk_order,
    fill_value=0
)

sector_risk.to_csv(
    TABLE_DIR
    / "06b_sector_risk_distribution.csv"
)


sector_risk.plot(
    kind="bar",
    stacked=True,
    figsize=(13, 7)
)

plt.title(
    "Risk Level Distribution by Sector"
)

plt.xlabel("Sector")
plt.ylabel("Number of Observations")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.legend(
    title="Risk Level"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "12_risk_by_sector.png",
    dpi=300
)

plt.close()


# ==========================================================
# 4. AVERAGE FUTURE VOLATILITY BY SECTOR
# ==========================================================

sector_future_volatility = (
    df.groupby("Sector")
    ["Future_20D_Volatility"]
    .mean()
    .sort_values(
        ascending=False
    )
)

sector_future_volatility.to_csv(
    TABLE_DIR
    / "06b_sector_future_volatility.csv"
)


plt.figure(figsize=(13, 7))

bars = plt.bar(
    sector_future_volatility.index,
    sector_future_volatility.values
)

plt.title(
    "Average Future 20-Day Volatility by Sector"
)

plt.xlabel("Sector")
plt.ylabel(
    "Average Future 20-Day Volatility"
)

plt.xticks(
    rotation=45,
    ha="right"
)

for bar, value in zip(
    bars,
    sector_future_volatility.values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "13_average_future_volatility_sector.png",
    dpi=300
)

plt.close()


# ==========================================================
# 5. RISK DISTRIBUTION BY YEAR
# ==========================================================

df["Year"] = df["Date"].dt.year

year_risk = pd.crosstab(
    df["Year"],
    df["Risk_Level"]
)

year_risk = year_risk.reindex(
    columns=risk_order,
    fill_value=0
)

year_risk.to_csv(
    TABLE_DIR
    / "06b_year_risk_distribution.csv"
)


year_risk.plot(
    kind="bar",
    stacked=True,
    figsize=(13, 7)
)

plt.title(
    "Risk Level Distribution by Year"
)

plt.xlabel("Year")
plt.ylabel("Number of Observations")

plt.legend(
    title="Risk Level"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "14_risk_by_year.png",
    dpi=300
)

plt.close()


# ==========================================================
# 6. YEARLY AVERAGE FUTURE VOLATILITY
# ==========================================================

yearly_volatility = (
    df.groupby("Year")
    ["Future_20D_Volatility"]
    .mean()
)

yearly_volatility.to_csv(
    TABLE_DIR
    / "06b_yearly_future_volatility.csv"
)


plt.figure(figsize=(12, 6))

plt.plot(
    yearly_volatility.index,
    yearly_volatility.values,
    marker="o"
)

plt.title(
    "Average Future 20-Day Volatility by Year"
)

plt.xlabel("Year")

plt.ylabel(
    "Average Future 20-Day Volatility"
)

plt.xticks(
    yearly_volatility.index,
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "15_yearly_future_volatility.png",
    dpi=300
)

plt.close()


# ==========================================================
# 7. TARGET SUMMARY TABLE
# ==========================================================

target_summary = pd.DataFrame({

    "Risk_Level": risk_order,

    "Count": [
        int(risk_counts[risk])
        for risk in risk_order
    ],

    "Percentage": [
        risk_percentages[risk]
        for risk in risk_order
    ]
})

target_summary.to_csv(
    TABLE_DIR
    / "06b_risk_class_summary.csv",
    index=False
)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 70)
print("RISK TARGET EDA COMPLETED")
print("=" * 70)

print("\nThresholds:")

print(
    f"LOW → MEDIUM: {low_threshold:.6f}"
)

print(
    f"MEDIUM → HIGH: {high_threshold:.6f}"
)

print("\nVisualizations created:")

risk_eda_files = [
    "10_risk_class_distribution.png",
    "11_future_volatility_thresholds.png",
    "12_risk_by_sector.png",
    "13_average_future_volatility_sector.png",
    "14_risk_by_year.png",
    "15_yearly_future_volatility.png"
]

for filename in risk_eda_files:

    print(
        f"  ✓ {filename}"
    )

print("\nTables created:")

table_files = [
    "06b_sector_risk_distribution.csv",
    "06b_sector_future_volatility.csv",
    "06b_year_risk_distribution.csv",
    "06b_yearly_future_volatility.csv",
    "06b_risk_class_summary.csv"
]

for filename in table_files:

    print(
        f"  ✓ {filename}"
    )

print("\n" + "=" * 70)
print("STEP 06B COMPLETED")
print("=" * 70)