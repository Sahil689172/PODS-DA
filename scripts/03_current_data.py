# ==========================================================
# PODS-DA
# Step 03: Download Current Market Data using yfinance
# ==========================================================

import pandas as pd
import yfinance as yf
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

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "current_data"
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
# 2. SETTINGS
# ==========================================================

START_DATE = "2026-02-01"

# yfinance END date is exclusive.
# Add one day so today's available data can be included.

END_DATE = (
    pd.Timestamp.today().normalize()
    + pd.Timedelta(days=1)
).strftime("%Y-%m-%d")


# ==========================================================
# 3. YAHOO FINANCE TICKER MAPPING
# ==========================================================

# Historical dataset ticker -> current Yahoo Finance ticker
#
# LTIMindtree was renamed and Yahoo Finance currently uses
# LTM.NS for the security.
#
# We download using LTM.NS but convert it back to LTIM.NS
# in our project so the historical and current datasets
# use the same ticker identifier.

YAHOO_TICKER_MAPPING = {
    "LTIM.NS": "LTM.NS"
}


# ==========================================================
# 4. LOAD HISTORICAL DATA
# ==========================================================

print("=" * 70)
print("CURRENT MARKET DATA DOWNLOAD")
print("=" * 70)

print("\nLoading historical cleaned dataset...")

historical = pd.read_csv(
    HISTORICAL_FILE
)

print(
    f"Historical rows loaded: "
    f"{len(historical):,}"
)

print(
    f"Historical tickers: "
    f"{historical['Ticker'].nunique()}"
)


# ==========================================================
# 5. GET PROJECT TICKER LIST
# ==========================================================

project_tickers = sorted(
    historical["Ticker"]
    .dropna()
    .unique()
)

print("\nProject tickers:")

for ticker in project_tickers:
    print(f"  {ticker}")

print(
    f"\nTotal project tickers: "
    f"{len(project_tickers)}"
)


# ==========================================================
# 6. DOWNLOAD CURRENT DATA
# ==========================================================

print("\n" + "=" * 70)
print("DOWNLOADING FROM YFINANCE")
print("=" * 70)

print(
    f"\nPeriod: "
    f"{START_DATE} → {END_DATE}"
)


all_data = []
download_status = []


for i, project_ticker in enumerate(
    project_tickers,
    start=1
):

    # ------------------------------------------------------
    # Determine Yahoo ticker
    # ------------------------------------------------------

    yahoo_ticker = YAHOO_TICKER_MAPPING.get(
        project_ticker,
        project_ticker
    )

    print(
        f"\n[{i}/{len(project_tickers)}] "
        f"Downloading {project_ticker}..."
    )

    if yahoo_ticker != project_ticker:

        print(
            f"  Yahoo Finance ticker: "
            f"{yahoo_ticker}"
        )

    try:

        # --------------------------------------------------
        # Download
        # --------------------------------------------------

        data = yf.download(
            yahoo_ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
            actions=False
        )


        # --------------------------------------------------
        # Check data
        # --------------------------------------------------

        if data.empty:

            print(
                f"  WARNING: No data returned "
                f"for {yahoo_ticker}"
            )

            download_status.append({

                "Project_Ticker":
                    project_ticker,

                "Yahoo_Ticker":
                    yahoo_ticker,

                "Status":
                    "NO_DATA",

                "Rows":
                    0,

                "First_Date":
                    None,

                "Last_Date":
                    None
            })

            continue


        # --------------------------------------------------
        # Handle MultiIndex columns
        # --------------------------------------------------

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = (
                data.columns
                .get_level_values(0)
            )


        # --------------------------------------------------
        # Reset index
        # --------------------------------------------------

        data = data.reset_index()


        # --------------------------------------------------
        # Standardize column names
        # --------------------------------------------------

        data.columns = [
            str(column).strip()
            for column in data.columns
        ]


        # --------------------------------------------------
        # Add PROJECT ticker
        # --------------------------------------------------

        # Important:
        # We use the historical/project ticker here,
        # NOT the Yahoo Finance ticker.

        data["Ticker"] = project_ticker


        # --------------------------------------------------
        # Select required columns
        # --------------------------------------------------

        required_columns = [

            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Ticker"
        ]

        data = data[
            [
                column
                for column in required_columns
                if column in data.columns
            ]
        ]


        # --------------------------------------------------
        # Convert Date
        # --------------------------------------------------

        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )


        # --------------------------------------------------
        # Remove invalid dates
        # --------------------------------------------------

        data = data.dropna(
            subset=["Date"]
        )


        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        all_data.append(data)


        # --------------------------------------------------
        # Status
        # --------------------------------------------------

        download_status.append({

            "Project_Ticker":
                project_ticker,

            "Yahoo_Ticker":
                yahoo_ticker,

            "Status":
                "SUCCESS",

            "Rows":
                len(data),

            "First_Date":
                data["Date"].min(),

            "Last_Date":
                data["Date"].max()
        })


        print(
            f"  SUCCESS: "
            f"{len(data)} rows"
        )

        print(
            f"  Date range: "
            f"{data['Date'].min().date()} → "
            f"{data['Date'].max().date()}"
        )


    except Exception as e:

        print(
            f"  ERROR: "
            f"{project_ticker} -> {e}"
        )

        download_status.append({

            "Project_Ticker":
                project_ticker,

            "Yahoo_Ticker":
                yahoo_ticker,

            "Status":
                "ERROR",

            "Rows":
                0,

            "First_Date":
                None,

            "Last_Date":
                None
        })


# ==========================================================
# 7. CHECK WHETHER ANY DATA WAS DOWNLOADED
# ==========================================================

if len(all_data) == 0:

    raise RuntimeError(
        "No ticker data was successfully downloaded."
    )


# ==========================================================
# 8. COMBINE ALL CURRENT DATA
# ==========================================================

print("\n" + "=" * 70)
print("COMBINING CURRENT DATA")
print("=" * 70)

current_data = pd.concat(
    all_data,
    ignore_index=True
)


# ==========================================================
# 9. REMOVE DUPLICATES
# ==========================================================

before_duplicates = len(
    current_data
)

current_data = (
    current_data
    .drop_duplicates(
        subset=["Date", "Ticker"]
    )
    .sort_values(
        by=["Date", "Ticker"]
    )
    .reset_index(drop=True)
)

duplicates_removed = (
    before_duplicates
    - len(current_data)
)


print(
    f"Duplicate Date-Ticker rows removed: "
    f"{duplicates_removed}"
)


# ==========================================================
# 10. SAVE CURRENT DATA
# ==========================================================

current_file = (
    OUTPUT_DIR
    / "current_yfinance_2026_02_onward.csv"
)

current_data.to_csv(
    current_file,
    index=False
)


# ==========================================================
# 11. SAVE DOWNLOAD STATUS
# ==========================================================

status_df = pd.DataFrame(
    download_status
)

status_file = (
    TABLE_DIR
    / "03_yfinance_download_status.csv"
)

status_df.to_csv(
    status_file,
    index=False
)


# ==========================================================
# 12. DATA QUALITY REPORT
# ==========================================================

quality_report = pd.DataFrame({

    "Column":
        current_data.columns,

    "Data_Type":
        current_data.dtypes.astype(str).values,

    "Missing_Count":
        current_data.isnull().sum().values,

    "Missing_Percentage":
        (
            current_data
            .isnull()
            .mean()
            * 100
        )

})

quality_file = (
    TABLE_DIR
    / "03_current_data_quality.csv"
)

quality_report.to_csv(
    quality_file,
    index=False
)


# ==========================================================
# 13. DOWNLOAD SUMMARY
# ==========================================================

successful = (
    status_df["Status"]
    == "SUCCESS"
).sum()

failed = (
    status_df["Status"]
    != "SUCCESS"
).sum()


# ==========================================================
# 14. FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 70)
print("CURRENT DATA DOWNLOAD COMPLETED")
print("=" * 70)

print(
    f"\nRows downloaded: "
    f"{len(current_data):,}"
)

print(
    f"Unique tickers downloaded: "
    f"{current_data['Ticker'].nunique()}"
)

print(
    f"Successful tickers: "
    f"{successful}"
)

print(
    f"Failed/no-data tickers: "
    f"{failed}"
)

print(
    f"Duplicate rows removed: "
    f"{duplicates_removed}"
)

print(
    f"Date range: "
    f"{current_data['Date'].min().date()} "
    f"→ "
    f"{current_data['Date'].max().date()}"
)

print("\nCurrent data columns:")

for i, column in enumerate(
    current_data.columns,
    start=1
):

    print(
        f"{i:2}. {column}"
    )


print("\nMissing values:")

print(
    current_data.isnull().sum()
)


print("\nFiles created:")

print(
    f"1. {current_file}"
)

print(
    f"2. {status_file}"
)

print(
    f"3. {quality_file}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)