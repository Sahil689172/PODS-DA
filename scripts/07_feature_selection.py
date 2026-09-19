import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "temporal_split"
    / "train_2016_2023.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "feature_selection"
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
# CANDIDATE FEATURES
# ==========================================================

CANDIDATE_FEATURES = [

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
# START
# ==========================================================

print("=" * 70)
print("STEP 07 - FEATURE SELECTION")
print("=" * 70)


# ==========================================================
# LOAD TRAINING DATA ONLY
# ==========================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

print(f"\nTraining input: {INPUT_FILE}")
print(f"Training shape: {df.shape}")

print(
    f"Training period: "
    f"{df['Date'].min().date()} → "
    f"{df['Date'].max().date()}"
)

print(
    f"Unique tickers: "
    f"{df['Ticker'].nunique()}"
)


# ==========================================================
# VERIFY TARGET
# ==========================================================

if "Risk_Level_ID" not in df.columns:

    raise ValueError(
        "Risk_Level_ID not found in training data."
    )


# ==========================================================
# VERIFY CANDIDATE FEATURES
# ==========================================================

missing_candidates = [
    feature
    for feature in CANDIDATE_FEATURES
    if feature not in df.columns
]

if missing_candidates:

    raise ValueError(
        "Missing candidate features:\n"
        + "\n".join(missing_candidates)
    )


print(
    f"\nCandidate features: "
    f"{len(CANDIDATE_FEATURES)}"
)

print("\nAll candidate features found ✓")


# ==========================================================
# TARGET DATA
# ==========================================================

X = df[CANDIDATE_FEATURES].copy()

y = df["Risk_Level_ID"].copy()


# ==========================================================
# REMOVE ROWS WITHOUT TARGET
# ==========================================================

target_mask = y.notna()

X = X.loc[target_mask].copy()

y = y.loc[target_mask].copy()

print(
    f"\nRows with valid target: "
    f"{len(X):,}"
)


# ==========================================================
# MISSING VALUE REPORT
# ==========================================================

missing_report = pd.DataFrame({

    "Feature": CANDIDATE_FEATURES,

    "Missing_Count": [
        X[feature].isna().sum()
        for feature in CANDIDATE_FEATURES
    ]
})

missing_report["Missing_Percentage"] = (
    missing_report["Missing_Count"]
    / len(X)
    * 100
)

missing_report = missing_report.sort_values(
    "Missing_Percentage",
    ascending=False
)

missing_report.to_csv(
    TABLE_DIR
    / "07_feature_missingness.csv",
    index=False
)


print("\nTop missing candidate features:")

print(
    missing_report.head(10).to_string(
        index=False
    )
)


# ==========================================================
# MEDIAN IMPUTATION
# ==========================================================
#
# IMPORTANT:
# The imputer is fitted ONLY on training data.
#
# Validation and test data will NOT be used here.
#
# This is only for feature-selection calculations.
# The final ML pipeline will have its own preprocessing.

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=CANDIDATE_FEATURES,
    index=X.index
)


# ==========================================================
# 1. CORRELATION ANALYSIS
# ==========================================================

print("\n" + "-" * 70)
print("1. CORRELATION ANALYSIS")
print("-" * 70)

correlation_matrix = (
    X_imputed
    .corr()
)

correlation_matrix.to_csv(
    TABLE_DIR
    / "07_feature_correlation_matrix.csv"
)


# ----------------------------------------------------------
# Find highly correlated pairs
# ----------------------------------------------------------

correlation_pairs = []

for i in range(
    len(CANDIDATE_FEATURES)
):

    for j in range(
        i + 1,
        len(CANDIDATE_FEATURES)
    ):

        feature_1 = CANDIDATE_FEATURES[i]

        feature_2 = CANDIDATE_FEATURES[j]

        correlation = (
            correlation_matrix
            .loc[
                feature_1,
                feature_2
            ]
        )

        if abs(correlation) >= 0.90:

            correlation_pairs.append({

                "Feature_1": feature_1,

                "Feature_2": feature_2,

                "Correlation": correlation,

                "Absolute_Correlation": abs(
                    correlation
                )
            })


correlation_pairs_df = (
    pd.DataFrame(
        correlation_pairs
    )
    .sort_values(
        "Absolute_Correlation",
        ascending=False
    )
)


correlation_pairs_df.to_csv(
    TABLE_DIR
    / "07_high_correlation_pairs.csv",
    index=False
)


print(
    f"\nHighly correlated pairs "
    f"(|r| >= 0.90): "
    f"{len(correlation_pairs_df)}"
)

if len(correlation_pairs_df) > 0:

    print(
        correlation_pairs_df.head(15)
        .to_string(index=False)
    )

else:

    print(
        "No feature pairs exceeded "
        "the 0.90 threshold."
    )


# ==========================================================
# 2. MUTUAL INFORMATION
# ==========================================================

print("\n" + "-" * 70)
print("2. MUTUAL INFORMATION")
print("-" * 70)


mi_scores = mutual_info_classif(
    X_imputed,
    y.astype(int),
    random_state=42
)


mi_results = pd.DataFrame({

    "Feature": CANDIDATE_FEATURES,

    "Mutual_Information": mi_scores

})


mi_results = mi_results.sort_values(
    "Mutual_Information",
    ascending=False
).reset_index(
    drop=True
)


mi_results["MI_Rank"] = (
    mi_results.index + 1
)


mi_results.to_csv(
    TABLE_DIR
    / "07_mutual_information_ranking.csv",
    index=False
)


print("\nTop 15 features by Mutual Information:")

print(
    mi_results.head(15)
    .to_string(index=False)
)


# ==========================================================
# 3. EXTRA TREES FEATURE IMPORTANCE
# ==========================================================

print("\n" + "-" * 70)
print("3. EXTRA TREES FEATURE IMPORTANCE")
print("-" * 70)


extra_trees = ExtraTreesClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


extra_trees.fit(
    X_imputed,
    y.astype(int)
)


importance_results = pd.DataFrame({

    "Feature": CANDIDATE_FEATURES,

    "ExtraTrees_Importance":
        extra_trees.feature_importances_

})


importance_results = (
    importance_results
    .sort_values(
        "ExtraTrees_Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


importance_results["ET_Rank"] = (
    importance_results.index + 1
)


importance_results.to_csv(
    TABLE_DIR
    / "07_extratrees_importance.csv",
    index=False
)


print(
    "\nTop 15 features by Extra Trees importance:"
)

print(
    importance_results.head(15)
    .to_string(index=False)
)


# ==========================================================
# 4. COMBINE RANKINGS
# ==========================================================

print("\n" + "-" * 70)
print("4. COMBINED FEATURE RANKING")
print("-" * 70)


ranking = mi_results[
    [
        "Feature",
        "Mutual_Information",
        "MI_Rank"
    ]
].merge(

    importance_results[
        [
            "Feature",
            "ExtraTrees_Importance",
            "ET_Rank"
        ]
    ],

    on="Feature"

)


# Lower average rank = stronger overall ranking.

ranking["Average_Rank"] = (
    ranking["MI_Rank"]
    + ranking["ET_Rank"]
) / 2


ranking = ranking.sort_values(
    "Average_Rank"
).reset_index(
    drop=True
)


ranking["Combined_Rank"] = (
    ranking.index + 1
)


# ==========================================================
# TOP FEATURE FLAGS
# ==========================================================

TOP_N = 15

top_mi_features = set(
    mi_results
    .head(TOP_N)
    ["Feature"]
)

top_et_features = set(
    importance_results
    .head(TOP_N)
    ["Feature"]
)


ranking["Top_MI"] = (
    ranking["Feature"]
    .isin(top_mi_features)
)

ranking["Top_ExtraTrees"] = (
    ranking["Feature"]
    .isin(top_et_features)
)

ranking["Selected_By_Both"] = (
    ranking["Top_MI"]
    & ranking["Top_ExtraTrees"]
)


ranking.to_csv(
    TABLE_DIR
    / "07_combined_feature_ranking.csv",
    index=False
)


print(
    "\nCombined ranking:"
)

print(
    ranking.head(20)
    .to_string(index=False)
)


# ==========================================================
# 5. CORRELATION-BASED REDUNDANCY CHECK
# ==========================================================

print("\n" + "-" * 70)
print("5. REDUNDANCY CHECK")
print("-" * 70)


# We process features from strongest combined ranking
# to weaker ranking.
#
# If a feature is highly correlated with an already
# selected feature, it is treated as redundant.

selected_features = []

removed_redundant = []


for feature in ranking["Feature"]:

    if len(selected_features) == 0:

        selected_features.append(
            feature
        )

        continue


    max_correlation = (
        correlation_matrix
        .loc[
            feature,
            selected_features
        ]
        .abs()
        .max()
    )

    if max_correlation < 0.90:

        selected_features.append(
            feature
        )

    else:

        correlated_with = (
            correlation_matrix
            .loc[
                feature,
                selected_features
            ]
            .abs()
            .idxmax()
        )

        removed_redundant.append({

            "Removed_Feature": feature,

            "Kept_Feature": correlated_with,

            "Absolute_Correlation": (
                correlation_matrix
                .loc[
                    feature,
                    correlated_with
                ]
                .__abs__()
            )
        })


redundancy_df = pd.DataFrame(
    removed_redundant
)

redundancy_df.to_csv(
    TABLE_DIR
    / "07_redundant_features_removed.csv",
    index=False
)


# ==========================================================
# 6. FINAL FEATURE SET
# ==========================================================

# We want a manageable final set.
#
# Maximum selected features:
# 15
#
# The features are selected in combined-ranking order,
# while avoiding highly redundant features.

MAX_FEATURES = 15

final_features = (
    selected_features[:MAX_FEATURES]
)


# ==========================================================
# FINAL FEATURE TABLE
# ==========================================================

final_feature_table = ranking[
    ranking["Feature"]
    .isin(final_features)
].copy()


final_feature_table = (
    final_feature_table
    .sort_values(
        "Combined_Rank"
    )
)


final_feature_table["Final_Selected"] = True


final_feature_table.to_csv(
    TABLE_DIR
    / "07_final_selected_features.csv",
    index=False
)


# ==========================================================
# SAVE FINAL FEATURE LIST
# ==========================================================

feature_list_file = (
    OUTPUT_DIR
    / "selected_features.txt"
)

with open(
    feature_list_file,
    "w"
) as file:

    for feature in final_features:

        file.write(
            feature + "\n"
        )


# ==========================================================
# SUMMARY
# ==========================================================

summary = pd.DataFrame({

    "Metric": [

        "Candidate Features",

        "Top MI Features",

        "Top ExtraTrees Features",

        "Features Selected Before Max Limit",

        "Maximum Final Features",

        "Final Selected Features"

    ],

    "Value": [

        len(CANDIDATE_FEATURES),

        TOP_N,

        TOP_N,

        len(selected_features),

        MAX_FEATURES,

        len(final_features)

    ]

})


summary.to_csv(
    TABLE_DIR
    / "07_feature_selection_summary.csv",
    index=False
)


# ==========================================================
# PRINT FINAL FEATURES
# ==========================================================

print("\n" + "=" * 70)
print("FINAL SELECTED FEATURES")
print("=" * 70)

for i, feature in enumerate(
    final_features,
    start=1
):

    print(
        f"{i:>2}. {feature}"
    )


# ==========================================================
# FINAL INFORMATION
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE SELECTION COMPLETED")
print("=" * 70)

print(
    f"\nCandidate features: "
    f"{len(CANDIDATE_FEATURES)}"
)

print(
    f"Final selected features: "
    f"{len(final_features)}"
)

print(
    "\nFeature selection was performed "
    "using TRAINING DATA ONLY."
)

print(
    "\nImportant:"
)

print(
    "Validation and test data were NOT used "
    "for feature selection."
)

print(
    f"\nFeature list saved to:"
)

print(
    f"  ✓ {feature_list_file}"
)

print("\n" + "=" * 70)