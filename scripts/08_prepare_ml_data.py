import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------
# INPUT FILES
# ----------------------------------------------------------

TRAIN_FILE = (
    BASE_DIR
    / "outputs"
    / "temporal_split"
    / "train_2016_2023.csv"
)

VALIDATION_FILE = (
    BASE_DIR
    / "outputs"
    / "temporal_split"
    / "validation_2024.csv"
)

TEST_FILE = (
    BASE_DIR
    / "outputs"
    / "temporal_split"
    / "test_2025_2026.csv"
)

SELECTED_FEATURES_FILE = (
    BASE_DIR
    / "outputs"
    / "feature_selection"
    / "selected_features.txt"
)


# ----------------------------------------------------------
# OUTPUT DIRECTORY
# ----------------------------------------------------------

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "ml_ready"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ----------------------------------------------------------
# OUTPUT FILES
# ----------------------------------------------------------

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "train_ml_ready.csv"
)

VALIDATION_OUTPUT = (
    OUTPUT_DIR
    / "validation_ml_ready.csv"
)

TEST_OUTPUT = (
    OUTPUT_DIR
    / "test_ml_ready.csv"
)

SCALER_PARAMETERS_OUTPUT = (
    OUTPUT_DIR
    / "scaler_parameters.csv"
)

IMPUTER_PARAMETERS_OUTPUT = (
    OUTPUT_DIR
    / "imputer_parameters.csv"
)

PREPARATION_SUMMARY_OUTPUT = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "08_ml_preparation_summary.csv"
)

FEATURE_STATISTICS_OUTPUT = (
    BASE_DIR
    / "outputs"
    / "tables"
    / "08_ml_feature_statistics.csv"
)


# ==========================================================
# START
# ==========================================================

print("=" * 70)
print("STEP 08 - ML DATASET PREPARATION")
print("=" * 70)


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def load_dataset(file_path, dataset_name):

    print("\n" + "-" * 70)
    print(f"LOADING {dataset_name.upper()}")
    print("-" * 70)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{file_path}"
        )

    data = pd.read_csv(file_path)

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values(
            ["Date", "Ticker"]
        )
        .reset_index(drop=True)
    )

    print(f"File: {file_path}")
    print(f"Shape: {data.shape}")

    print(
        f"Date range: "
        f"{data['Date'].min().date()} → "
        f"{data['Date'].max().date()}"
    )

    print(
        f"Unique tickers: "
        f"{data['Ticker'].nunique()}"
    )

    return data


# ==========================================================
# LOAD TRAIN / VALIDATION / TEST
# ==========================================================

train = load_dataset(
    TRAIN_FILE,
    "Training Data"
)

validation = load_dataset(
    VALIDATION_FILE,
    "Validation Data"
)

test = load_dataset(
    TEST_FILE,
    "Test Data"
)


# ==========================================================
# LOAD SELECTED FEATURES
# ==========================================================

print("\n" + "-" * 70)
print("LOADING SELECTED FEATURES")
print("-" * 70)

if not SELECTED_FEATURES_FILE.exists():

    raise FileNotFoundError(
        f"Selected feature file not found:\n"
        f"{SELECTED_FEATURES_FILE}"
    )


with open(
    SELECTED_FEATURES_FILE,
    "r",
    encoding="utf-8"
) as file:

    selected_features = [
        line.strip()
        for line in file
        if line.strip()
    ]


print(
    f"Selected features loaded: "
    f"{len(selected_features)}"
)

for i, feature in enumerate(
    selected_features,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


# ==========================================================
# VERIFY SELECTED FEATURES
# ==========================================================

print("\n" + "-" * 70)
print("VERIFYING FEATURES")
print("-" * 70)

required_columns = (
    selected_features
    + [
        "Risk_Level",
        "Risk_Level_ID"
    ]
)


for dataset_name, data in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test)
]:

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataset_name} data is missing "
            f"columns: {missing_columns}"
        )

    print(
        f"✓ {dataset_name}: "
        f"all required columns present"
    )


# ==========================================================
# TARGET VALIDATION
# ==========================================================

print("\n" + "-" * 70)
print("TARGET VALIDATION")
print("-" * 70)


expected_risk_levels = {
    "LOW",
    "MEDIUM",
    "HIGH"
}

expected_risk_ids = {
    0,
    1,
    2
}


for dataset_name, data in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test)
]:

    risk_levels = set(
        data["Risk_Level"]
        .dropna()
        .unique()
    )

    risk_ids = set(
        data["Risk_Level_ID"]
        .dropna()
        .astype(int)
        .unique()
    )

    print(
        f"\n{dataset_name}:"
    )

    print(
        f"  Risk_Level values: "
        f"{sorted(risk_levels)}"
    )

    print(
        f"  Risk_Level_ID values: "
        f"{sorted(risk_ids)}"
    )

    unexpected_levels = (
        risk_levels -
        expected_risk_levels
    )

    unexpected_ids = (
        risk_ids -
        expected_risk_ids
    )

    if unexpected_levels:

        raise ValueError(
            f"Unexpected Risk_Level values "
            f"in {dataset_name}: "
            f"{unexpected_levels}"
        )

    if unexpected_ids:

        raise ValueError(
            f"Unexpected Risk_Level_ID values "
            f"in {dataset_name}: "
            f"{unexpected_ids}"
        )


# ==========================================================
# REMOVE ROWS WITH MISSING TARGET
# ==========================================================

print("\n" + "-" * 70)
print("TARGET MISSINGNESS")
print("-" * 70)


def remove_missing_target(
    data,
    dataset_name
):

    before = len(data)

    data = data.dropna(
        subset=[
            "Risk_Level",
            "Risk_Level_ID"
        ]
    ).copy()

    after = len(data)

    removed = before - after

    print(
        f"{dataset_name}: "
        f"{removed} rows removed"
    )

    print(
        f"{dataset_name}: "
        f"{after} rows remaining"
    )

    return data


train = remove_missing_target(
    train,
    "Training"
)

validation = remove_missing_target(
    validation,
    "Validation"
)

test = remove_missing_target(
    test,
    "Test"
)


# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

print("\n" + "-" * 70)
print("TARGET DISTRIBUTION")
print("-" * 70)


def print_target_distribution(
    data,
    dataset_name
):

    counts = (
        data["Risk_Level"]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH"],
            fill_value=0
        )
    )

    percentages = (
        counts /
        len(data) *
        100
    )

    print(
        f"\n{dataset_name}:"
    )

    for level in [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]:

        print(
            f"  {level:6s}: "
            f"{counts[level]:6d} "
            f"({percentages[level]:6.2f}%)"
        )


print_target_distribution(
    train,
    "Training"
)

print_target_distribution(
    validation,
    "Validation"
)

print_target_distribution(
    test,
    "Test"
)


# ==========================================================
# CREATE X AND y
# ==========================================================

X_train = train[
    selected_features
].copy()

y_train = train[
    "Risk_Level_ID"
].astype(int)

X_validation = validation[
    selected_features
].copy()

y_validation = validation[
    "Risk_Level_ID"
].astype(int)

X_test = test[
    selected_features
].copy()

y_test = test[
    "Risk_Level_ID"
].astype(int)


# ==========================================================
# ORIGINAL MISSINGNESS
# ==========================================================

print("\n" + "-" * 70)
print("FEATURE MISSINGNESS BEFORE IMPUTATION")
print("-" * 70)


missing_before = pd.DataFrame({

    "Feature":
        selected_features,

    "Train_Missing":
        [
            X_train[col].isna().sum()
            for col in selected_features
        ],

    "Validation_Missing":
        [
            X_validation[col].isna().sum()
            for col in selected_features
        ],

    "Test_Missing":
        [
            X_test[col].isna().sum()
            for col in selected_features
        ]
})

missing_before[
    "Train_Missing_Percentage"
] = (
    missing_before["Train_Missing"]
    / len(X_train)
    * 100
)

missing_before[
    "Validation_Missing_Percentage"
] = (
    missing_before["Validation_Missing"]
    / len(X_validation)
    * 100
)

missing_before[
    "Test_Missing_Percentage"
] = (
    missing_before["Test_Missing"]
    / len(X_test)
    * 100
)

print(
    missing_before.to_string(
        index=False
    )
)


# ==========================================================
# IMPUTATION
# ==========================================================

print("\n" + "-" * 70)
print("MEDIAN IMPUTATION")
print("-" * 70)

print(
    "Fitting imputer on TRAINING data only..."
)

imputer = SimpleImputer(
    strategy="median"
)


# IMPORTANT:
# Fit ONLY on training data.

X_train_imputed_array = (
    imputer.fit_transform(
        X_train
    )
)

X_validation_imputed_array = (
    imputer.transform(
        X_validation
    )
)

X_test_imputed_array = (
    imputer.transform(
        X_test
    )
)


# Convert back to DataFrames.

X_train_imputed = pd.DataFrame(
    X_train_imputed_array,
    columns=selected_features,
    index=X_train.index
)

X_validation_imputed = pd.DataFrame(
    X_validation_imputed_array,
    columns=selected_features,
    index=X_validation.index
)

X_test_imputed = pd.DataFrame(
    X_test_imputed_array,
    columns=selected_features,
    index=X_test.index
)


print(
    "✓ Training imputer fitted"
)

print(
    "✓ Validation transformed"
)

print(
    "✓ Test transformed"
)


# ==========================================================
# VERIFY NO MISSING VALUES AFTER IMPUTATION
# ==========================================================

print("\n" + "-" * 70)
print("VERIFYING IMPUTATION")
print("-" * 70)

train_missing_after = (
    X_train_imputed
    .isna()
    .sum()
    .sum()
)

validation_missing_after = (
    X_validation_imputed
    .isna()
    .sum()
    .sum()
)

test_missing_after = (
    X_test_imputed
    .isna()
    .sum()
    .sum()
)

print(
    f"Training missing values: "
    f"{train_missing_after}"
)

print(
    f"Validation missing values: "
    f"{validation_missing_after}"
)

print(
    f"Test missing values: "
    f"{test_missing_after}"
)

if (
    train_missing_after != 0
    or validation_missing_after != 0
    or test_missing_after != 0
):

    raise ValueError(
        "Missing values remain after imputation."
    )

print(
    "✓ No missing values remain"
)


# ==========================================================
# SAVE IMPUTER PARAMETERS
# ==========================================================

imputer_parameters = pd.DataFrame({

    "Feature":
        selected_features,

    "Training_Median":
        imputer.statistics_

})

imputer_parameters.to_csv(
    IMPUTER_PARAMETERS_OUTPUT,
    index=False
)


# ==========================================================
# STANDARDIZATION
# ==========================================================

print("\n" + "-" * 70)
print("FEATURE STANDARDIZATION")
print("-" * 70)

print(
    "Fitting StandardScaler on TRAINING data only..."
)

scaler = StandardScaler()


# IMPORTANT:
# Fit ONLY on training data.

X_train_scaled_array = (
    scaler.fit_transform(
        X_train_imputed
    )
)

X_validation_scaled_array = (
    scaler.transform(
        X_validation_imputed
    )
)

X_test_scaled_array = (
    scaler.transform(
        X_test_imputed
    )
)


# Convert to DataFrames.

X_train_scaled = pd.DataFrame(
    X_train_scaled_array,
    columns=selected_features,
    index=X_train.index
)

X_validation_scaled = pd.DataFrame(
    X_validation_scaled_array,
    columns=selected_features,
    index=X_validation.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled_array,
    columns=selected_features,
    index=X_test.index
)


print(
    "✓ Training scaler fitted"
)

print(
    "✓ Validation transformed"
)

print(
    "✓ Test transformed"
)


# ==========================================================
# VERIFY SCALING
# ==========================================================

print("\n" + "-" * 70)
print("SCALING VERIFICATION")
print("-" * 70)


train_means = (
    X_train_scaled
    .mean()
)

train_stds = (
    X_train_scaled
    .std()
)


print(
    f"Maximum absolute training mean: "
    f"{train_means.abs().max():.10f}"
)

print(
    f"Maximum deviation of training std "
    f"from 1: "
    f"{(train_stds - 1).abs().max():.10f}"
)


# ==========================================================
# SAVE SCALER PARAMETERS
# ==========================================================

scaler_parameters = pd.DataFrame({

    "Feature":
        selected_features,

    "Training_Mean":
        scaler.mean_,

    "Training_Std":
        scaler.scale_

})

scaler_parameters.to_csv(
    SCALER_PARAMETERS_OUTPUT,
    index=False
)


# ==========================================================
# CREATE FINAL ML DATASETS
# ==========================================================

print("\n" + "-" * 70)
print("CREATING FINAL ML DATASETS")
print("-" * 70)


def create_final_dataset(
    original_data,
    X_scaled,
    y,
    dataset_name
):

    result = pd.DataFrame(
        index=original_data.index
    )

    # Identification information.

    result["Date"] = (
        original_data["Date"]
        .values
    )

    result["Ticker"] = (
        original_data["Ticker"]
        .values
    )

    # Target.

    result["Risk_Level"] = (
        original_data["Risk_Level"]
        .values
    )

    result["Risk_Level_ID"] = (
        y.values
    )

    # Scaled features.

    for feature in selected_features:

        result[feature] = (
            X_scaled[feature]
            .values
        )

    print(
        f"{dataset_name}: "
        f"{result.shape}"
    )

    return result


train_ml = create_final_dataset(
    train,
    X_train_scaled,
    y_train,
    "Training"
)

validation_ml = create_final_dataset(
    validation,
    X_validation_scaled,
    y_validation,
    "Validation"
)

test_ml = create_final_dataset(
    test,
    X_test_scaled,
    y_test,
    "Test"
)


# ==========================================================
# FINAL DATASET VALIDATION
# ==========================================================

print("\n" + "-" * 70)
print("FINAL DATASET VALIDATION")
print("-" * 70)


def validate_final_dataset(
    data,
    dataset_name
):

    expected_columns = (
        [
            "Date",
            "Ticker",
            "Risk_Level",
            "Risk_Level_ID"
        ]
        + selected_features
    )

    if list(data.columns) != expected_columns:

        raise ValueError(
            f"{dataset_name} columns do not "
            "match expected structure."
        )

    feature_missing = (
        data[selected_features]
        .isna()
        .sum()
        .sum()
    )

    if feature_missing != 0:

        raise ValueError(
            f"{dataset_name} contains "
            f"{feature_missing} missing feature values."
        )

    invalid_targets = set(
        data["Risk_Level_ID"]
        .unique()
    ) - {0, 1, 2}

    if invalid_targets:

        raise ValueError(
            f"{dataset_name} contains "
            f"invalid target IDs: "
            f"{invalid_targets}"
        )

    print(
        f"✓ {dataset_name} validated"
    )


validate_final_dataset(
    train_ml,
    "Training"
)

validate_final_dataset(
    validation_ml,
    "Validation"
)

validate_final_dataset(
    test_ml,
    "Test"
)


# ==========================================================
# FEATURE STATISTICS
# ==========================================================

feature_statistics = []

for feature in selected_features:

    feature_statistics.append({

        "Feature": feature,

        "Train_Mean": (
            train_ml[feature].mean()
        ),

        "Train_Std": (
            train_ml[feature].std()
        ),

        "Validation_Mean": (
            validation_ml[feature].mean()
        ),

        "Validation_Std": (
            validation_ml[feature].std()
        ),

        "Test_Mean": (
            test_ml[feature].mean()
        ),

        "Test_Std": (
            test_ml[feature].std()
        )
    })


feature_statistics_df = pd.DataFrame(
    feature_statistics
)

feature_statistics_df.to_csv(
    FEATURE_STATISTICS_OUTPUT,
    index=False
)


# ==========================================================
# SAVE FINAL DATASETS
# ==========================================================

train_ml.to_csv(
    TRAIN_OUTPUT,
    index=False
)

validation_ml.to_csv(
    VALIDATION_OUTPUT,
    index=False
)

test_ml.to_csv(
    TEST_OUTPUT,
    index=False
)


# ==========================================================
# PREPARATION SUMMARY
# ==========================================================

summary = [

    {
        "Dataset": "Training",
        "Start_Date":
            train_ml["Date"].min().date(),
        "End_Date":
            train_ml["Date"].max().date(),
        "Rows":
            len(train_ml),
        "Features":
            len(selected_features),
        "Missing_Feature_Values":
            int(
                train_ml[
                    selected_features
                ]
                .isna()
                .sum()
                .sum()
            )
    },

    {
        "Dataset": "Validation",
        "Start_Date":
            validation_ml["Date"].min().date(),
        "End_Date":
            validation_ml["Date"].max().date(),
        "Rows":
            len(validation_ml),
        "Features":
            len(selected_features),
        "Missing_Feature_Values":
            int(
                validation_ml[
                    selected_features
                ]
                .isna()
                .sum()
                .sum()
            )
    },

    {
        "Dataset": "Test",
        "Start_Date":
            test_ml["Date"].min().date(),
        "End_Date":
            test_ml["Date"].max().date(),
        "Rows":
            len(test_ml),
        "Features":
            len(selected_features),
        "Missing_Feature_Values":
            int(
                test_ml[
                    selected_features
                ]
                .isna()
                .sum()
                .sum()
            )
    }
]

summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(
    PREPARATION_SUMMARY_OUTPUT,
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("STEP 08 - ML DATASET PREPARATION COMPLETED")
print("=" * 70)

print(
    f"\nSelected features: "
    f"{len(selected_features)}"
)

print(
    f"Training rows: "
    f"{len(train_ml)}"
)

print(
    f"Validation rows: "
    f"{len(validation_ml)}"
)

print(
    f"Test rows: "
    f"{len(test_ml)}"
)

print("\nTarget mapping:")
print("  LOW    = 0")
print("  MEDIUM = 1")
print("  HIGH   = 2")

print("\nPreprocessing:")
print("  ✓ Median imputation")
print("  ✓ Imputer fitted on training data only")
print("  ✓ StandardScaler")
print("  ✓ Scaler fitted on training data only")
print("  ✓ Validation transformed using training parameters")
print("  ✓ Test transformed using training parameters")

print("\nOutput files:")

print(
    f"  ✓ {TRAIN_OUTPUT}"
)

print(
    f"  ✓ {VALIDATION_OUTPUT}"
)

print(
    f"  ✓ {TEST_OUTPUT}"
)

print(
    f"  ✓ {SCALER_PARAMETERS_OUTPUT}"
)

print(
    f"  ✓ {IMPUTER_PARAMETERS_OUTPUT}"
)

print(
    f"  ✓ {PREPARATION_SUMMARY_OUTPUT}"
)

print(
    f"  ✓ {FEATURE_STATISTICS_OUTPUT}"
)

print("\n" + "=" * 70)
print("READY FOR STEP 09 - MODEL TRAINING")
print("=" * 70)