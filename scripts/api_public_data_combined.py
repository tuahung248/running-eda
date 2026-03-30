import os
import json
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Strava OAuth credentials — stored in .env, never hard-coded here
STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
STRAVA_REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")

# Activity type we care about (Strava uses "Run" for outdoor runs)
RUN_ACTIVITY_TYPES = {"Run", "run", "running", "Running"}
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


SCHEMA = [
    "id",
    "name",
    "sport_type",
    "start_date",
    "moving_time",
    "elapsed_time",
    "distance",
    "total_elevation_gain",
    "average_speed",
    "average_heart_rate",
    "max_heart_rate",
    "kudos_count",
    "comment_count",
    "achievement_count",
    "visibility",
    "data_source",
]


def load_public_dataset(filepath: str | Path) -> pd.DataFrame:
    """
    Load a public Strava CSV, standardise column names, and keep only runs.

    The CSV may use different column names depending on its source.
    We map the most common variants to our shared schema.

    Parameters
    ----------
    filepath : path to the CSV file

    Returns
    -------
    DataFrame conforming to SCHEMA (run activities only)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Public dataset not found: {filepath}")

    print(f"Loading public dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Raw rows loaded: {len(df):,}")

    rename_map = {
        "type": "sport_type",
        "sport_type": "sport_type",
        "activity_type": "sport_type",
        "start_date_local": "start_date",
        "date": "start_date",
        "start_date": "start_date",
        "activity_date": "start_date",
        "distance": "distance",
        "distance_km": "distance",
        "moving_time": "moving_time",
        "elapsed_time": "elapsed_time",
        "total_elevation_gain": "total_elevation_gain",
        "elevation_gain": "total_elevation_gain",
        "elev_gain": "total_elevation_gain",
        "average_speed": "average_speed",
        "max_speed": "max_speed",
        "average_heartrate": "average_heart_rate",
        "average_heart_rate": "average_heart_rate",
        "avg_hr": "average_heart_rate",
        "max_heartrate": "max_heart_rate",
        "max_heart_rate": "max_heart_rate",
        "kudos_count": "kudos_count",
        "comment_count": "comment_count",
        "achievement_count": "achievement_count",
        "visibility": "visibility",
        "name": "name",
        "id": "id",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    required = ["distance", "moving_time", "start_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after renaming: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    df["start_date"] = df["start_date"].dt.tz_localize(None)  # → naive datetime

    if "sport_type" in df.columns:
        before = len(df)
        df = df[df["sport_type"].isin(RUN_ACTIVITY_TYPES)].copy()
        print(
            f"  Kept {len(df):,} run rows (dropped {before - len(df):,} non-run rows)"
        )
    else:
        print("  No sport_type / type column found — assuming all rows are runs.")
        df["sport_type"] = "Run"

    df["sport_type"] = "Run"

    df["data_source"] = "public_dataset"
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = np.nan

    print(
        f"  Date range: {df['start_date'].min().date()} → {df['start_date'].max().date()}"
    )
    return df[SCHEMA].copy()


def _refresh_access_token() -> str:

    if not all([STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN]):
        raise EnvironmentError("Missing Strava credentials. ")

    response = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "refresh_token": STRAVA_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
        timeout=10,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise ValueError(f"Unexpected token response: {response.json()}")
    return token


def _fetch_activity_pages(access_token: str, max_results: int = 3000) -> list[dict]:

    activities = []
    page = 1

    while len(activities) < max_results:
        response = requests.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"page": page, "per_page": 200},
            timeout=15,
        )
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        activities.extend(batch)
        print(f"  Fetched {len(activities):,} activities so far...")
        page += 1

    return activities[:max_results]


def _parse_activity(activity: dict) -> dict:

    return {
        "id": activity.get("id"),
        "name": activity.get("name"),
        "sport_type": activity.get("sport_type") or activity.get("type"),
        "start_date": activity.get("start_date_local") or activity.get("start_date"),
        "moving_time": activity.get("moving_time"),
        "elapsed_time": activity.get("elapsed_time"),
        "distance": activity.get("distance"),  # metres
        "total_elevation_gain": activity.get("total_elevation_gain"),
        "average_speed": activity.get("average_speed"),  # m/s
        "max_speed": activity.get("max_speed"),  # m/s
        "average_heart_rate": activity.get("average_heartrate"),
        "max_heart_rate": activity.get("max_heartrate"),
        "kudos_count": activity.get("kudos_count"),
        "comment_count": activity.get("comment_count"),
        "achievement_count": activity.get("achievement_count"),
        "visibility": activity.get("visibility"),
        "data_source": "strava_api",
    }


def get_strava_runs(max_results: int = 3000) -> pd.DataFrame:

    print("\nConnecting to Strava API...")
    access_token = _refresh_access_token()
    print("  Access token obtained.")

    raw_activities = _fetch_activity_pages(access_token, max_results=max_results)
    print(f"  Total activities fetched: {len(raw_activities):,}")

    records = [_parse_activity(a) for a in raw_activities]
    df = pd.DataFrame(records)

    df["start_date"] = pd.to_datetime(df["start_date"], utc=True).dt.tz_localize(None)

    # Filter to runs only
    before = len(df)
    df = df[df["sport_type"].isin(RUN_ACTIVITY_TYPES)].copy()
    df["sport_type"] = "Run"
    print(f"  Kept {len(df):,} runs (dropped {before - len(df):,} non-run activities)")
    print(
        f"  Date range: {df['start_date'].min().date()} → {df['start_date'].max().date()}"
    )

    for col in SCHEMA:
        if col not in df.columns:
            df[col] = np.nan

    return df[SCHEMA].copy()


def _synthetic_rule_based(real_data: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)

    distance_mean = real_data["distance"].mean()
    distance_std = real_data["distance"].std()
    time_mean = real_data["moving_time"].mean()
    time_std = real_data["moving_time"].std()
    elev_mean = real_data["total_elevation_gain"].mean()

    synthetic = pd.DataFrame(
        {
            "id": np.nan,
            "name": [f"Synthetic Run {i}" for i in range(n)],
            "sport_type": "Run",
            "start_date": pd.date_range(
                start=real_data["start_date"].max() + timedelta(days=1),
                periods=n,
                freq="D",
            ),
            "moving_time": rng.normal(time_mean, time_std, n).clip(60, 25_000),
            "elapsed_time": np.nan,
            "distance": rng.normal(distance_mean, distance_std, n).clip(500, 50_000),
            "total_elevation_gain": rng.gamma(
                shape=2,
                scale=max(elev_mean / 2, 1),
                size=n,
            ),
            "average_speed": np.nan,
            "max_speed": np.nan,
            "average_heart_rate": np.nan,
            "max_heart_rate": np.nan,
            "kudos_count": np.nan,
            "comment_count": np.nan,
            "achievement_count": np.nan,
            "visibility": np.nan,
            "data_source": "synthetic_rule_based",
        }
    )
    print(f"  Generated {len(synthetic):,} synthetic runs (rule-based)")
    return synthetic[SCHEMA].copy()


def _synthetic_ctgan(real_data: pd.DataFrame, n: int) -> pd.DataFrame:

    try:
        from ctgan import CTGAN
    except ImportError:
        print("  ctgan is not installed — falling back to rule-based synthesis.")
        print("      Install it with:  pip install ctgan")
        return _synthetic_rule_based(real_data, n)

    feature_cols = ["distance", "moving_time", "total_elevation_gain"]
    train = real_data[feature_cols].dropna()

    print(f"  Training CTGAN on {len(train):,} real activities...")
    model = CTGAN()
    model.fit(train, epochs=100, verbose=False)

    synthetic_features = model.sample(n)
    synthetic_features["id"] = np.nan
    synthetic_features["name"] = [f"Synthetic Run {i}" for i in range(n)]
    synthetic_features["sport_type"] = "Run"
    synthetic_features["start_date"] = pd.date_range(
        start=real_data["start_date"].max() + timedelta(days=1),
        periods=n,
        freq="D",
    )
    synthetic_features["elapsed_time"] = np.nan
    synthetic_features["average_speed"] = np.nan
    synthetic_features["max_speed"] = np.nan
    synthetic_features["average_heart_rate"] = np.nan
    synthetic_features["max_heart_rate"] = np.nan
    synthetic_features["kudos_count"] = np.nan
    synthetic_features["comment_count"] = np.nan
    synthetic_features["achievement_count"] = np.nan
    synthetic_features["visibility"] = np.nan
    synthetic_features["data_source"] = "synthetic_ctgan"

    for col in SCHEMA:
        if col not in synthetic_features.columns:
            synthetic_features[col] = np.nan

    print(f"  Generated {len(synthetic_features):,} synthetic runs (CTGAN)")
    return synthetic_features[SCHEMA].copy()


def generate_synthetic_runs(
    real_data: pd.DataFrame,
    n: int = 200,
    method: str = "rule_based",
) -> pd.DataFrame:

    print(f"\nGenerating {n:,} synthetic runs using method='{method}'...")
    if method == "ctgan":
        return _synthetic_ctgan(real_data, n)
    elif method == "rule_based":
        return _synthetic_rule_based(real_data, n)
    else:
        raise ValueError(
            f"Unknown synthetic method '{method}'. Choose 'rule_based' or 'ctgan'."
        )


def combine_sources(
    public_csv_paths: list[Path] | None = None,
    use_strava_api: bool = True,
    add_synthetic: bool = False,
    n_synthetic: int = 200,
    synthetic_method: str = "rule_based",
) -> pd.DataFrame:
    frames = []

    # --- Public CSV datasets ---
    if public_csv_paths:
        for csv_path in public_csv_paths:
            try:
                frames.append(load_public_dataset(csv_path))
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping {csv_path.name}: {e}")

    # --- Strava API ---
    if use_strava_api:
        try:
            frames.append(get_strava_runs(max_results=3000))
        except Exception as e:
            print(f"Could not fetch Strava data: {e}")

    if not frames:
        raise RuntimeError(
            "No data was loaded. "
        )
    real_combined = pd.concat(frames, ignore_index=True)
    if add_synthetic:
        frames.append(
            generate_synthetic_runs(
                real_combined, n=n_synthetic, method=synthetic_method
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("start_date").reset_index(drop=True)
    for col in SCHEMA:
        if col not in combined.columns:
            combined[col] = np.nan

    return combined[SCHEMA].copy()


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total runs         : {len(df):,}")
    print(
        f"Date range         : {df['start_date'].min().date()} → {df['start_date'].max().date()}"
    )
    print(
        f"Days spanned       : {(df['start_date'].max() - df['start_date'].min()).days:,}"
    )
    total_km = df["distance"].sum() / 1000
    avg_km = df["distance"].mean() / 1000
    print(f"Total distance     : {total_km:,.1f} km")
    print(f"Avg run distance   : {avg_km:.2f} km")
    print(f"\nRows by source:")
    print(df["data_source"].value_counts().to_string())
    print(f"\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print("=" * 60)


def save_dataset(df: pd.DataFrame, output_path: str | Path = None) -> Path:
    if output_path is None:
        output_path = PROCESSED_DIR / "combined_runs.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return output_path


if __name__ == "__main__":

    ALL_CANDIDATES = [
        RAW_DIR / "Strava Running Data.csv",
        RAW_DIR / "strava_data.csv",
        RAW_DIR / "strava_full_data.csv",
    ]
    FOUND_CSVS = [p for p in ALL_CANDIDATES if p.exists()]

    if FOUND_CSVS:
        print(f"Found {len(FOUND_CSVS)} public dataset(s):")
        for p in FOUND_CSVS:
            print(f"  • {p}")
    else:
        print("No public CSVs found in data/raw/ — will rely on Strava API only.")
    USE_STRAVA_API = True
    # Synthetic augmentation — only enable if you need more data for EDA
    ADD_SYNTHETIC = False
    N_SYNTHETIC = 200
    SYNTHETIC_METHOD = "rule_based"
    combined = combine_sources(
        public_csv_paths=FOUND_CSVS or None,
        use_strava_api=USE_STRAVA_API,
        add_synthetic=ADD_SYNTHETIC,
        n_synthetic=N_SYNTHETIC,
        synthetic_method=SYNTHETIC_METHOD,
    )

    print_summary(combined)
    save_dataset(combined)
