from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "combined_public_data.csv"
SUPPORTED_EXTENSIONS = {".csv", ".json"}


def discover_input_files(raw_dir: Path) -> dict[str, list[Path]]:
    """Recursively discover all supported files under raw_dir."""
    discovered = {"csv": [], "json": []}
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            discovered["csv"].append(path)
        elif suffix == ".json":
            discovered["json"].append(path)

    discovered["csv"].sort()
    discovered["json"].sort()
    return discovered


def _dataframe_from_json_payload(payload: Any) -> pd.DataFrame:
    """Convert common JSON payload shapes into a DataFrame."""
    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame()
        return pd.json_normalize(payload)

    if isinstance(payload, dict):
        # Prefer top-level keys that look like record containers.
        for value in payload.values():
            if isinstance(value, list):
                return pd.json_normalize(value)
        return pd.json_normalize(payload)

    # Fallback for scalar or unsupported structures.
    return pd.json_normalize(payload)


def load_csv_file(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with encoding fallbacks."""
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error: Exception | None = None

    for encoding in encodings_to_try:
        try:
            try:
                df = pd.read_csv(path, encoding=encoding)
            except pd.errors.ParserError:
                # Retry with Python engine + auto delimiter detection (handles ; and mixed quirks).
                df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            if encoding != "utf-8":
                print(
                    f"[INFO] Used fallback encoding '{encoding}' for {path.as_posix()}"
                )
            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
            continue

    # Final fallback: replace invalid bytes so critical files are still ingestible.
    try:
        print(
            f"[WARN] Decoding issues in {path.as_posix()} - using replacement fallback"
        )
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            try:
                return pd.read_csv(fh)
            except pd.errors.ParserError:
                fh.seek(0)
                return pd.read_csv(fh, sep=None, engine="python")
    except Exception as exc:
        if last_error is not None:
            raise ValueError(f"{last_error}; fallback failed with: {exc}") from exc
        raise


def load_json_file(path: Path) -> pd.DataFrame:
    """Load a JSON file and normalize common nested formats."""
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return _dataframe_from_json_payload(payload)


def load_single_file(path: Path, raw_dir: Path) -> tuple[pd.DataFrame | None, int]:
    """
    Load one file into a DataFrame and add metadata columns.

    Returns
    -------
    (df_or_none, row_count)
        If load fails, returns (None, 0) and logs a warning.
    """
    file_type = path.suffix.lower().lstrip(".")
    relative_path = path.relative_to(raw_dir).as_posix()

    try:
        if file_type == "csv":
            df = load_csv_file(path)
        elif file_type == "json":
            df = load_json_file(path)
        else:
            print(f"[WARN] Unsupported file type skipped: {relative_path}")
            return None, 0

        df = df.copy()
        df["source_file"] = relative_path
        df["source_type"] = file_type
        row_count = len(df)
        print(f"[OK] Loaded {relative_path} ({file_type}) -> {row_count:,} rows")
        return df, row_count
    except Exception as exc:
        print(f"[WARN] Failed to load {relative_path}: {exc}")
        return None, 0


def combine_dataframes(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine all DataFrames with union of columns."""
    if not frames:
        return pd.DataFrame(columns=["source_file", "source_type"])
    return pd.concat(frames, ignore_index=True, sort=False)


def save_combined_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    """Ensure output directory exists and write the combined CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_pipeline(
    raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_FILE
) -> pd.DataFrame:
    """Execute discovery, load, combine, and save pipeline."""
    print("=" * 72)
    print("Public Data Combine Pipeline")
    print("=" * 72)
    print(f"Input directory : {raw_dir}")
    print(f"Output file     : {output_path}")

    discovered = discover_input_files(raw_dir)
    csv_files = discovered["csv"]
    json_files = discovered["json"]
    all_files = csv_files + json_files

    print("\nFile discovery summary")
    print(f"  CSV files  : {len(csv_files):,}")
    print(f"  JSON files : {len(json_files):,}")
    print(f"  Total files: {len(all_files):,}")

    frames: list[pd.DataFrame] = []
    loaded_rows_total = 0

    for path in all_files:
        df, row_count = load_single_file(path, raw_dir)
        if df is None:
            continue
        frames.append(df)
        loaded_rows_total += row_count

    combined = combine_dataframes(frames)
    save_combined_dataframe(combined, output_path)

    print("\nFinal output summary")
    print(f"  Files loaded       : {len(frames):,}/{len(all_files):,}")
    print(f"  Total rows loaded  : {loaded_rows_total:,}")
    print(
        f"  Final shape        : {combined.shape[0]:,} rows x {combined.shape[1]:,} columns"
    )
    print(f"  Output path        : {output_path}")
    print("=" * 72)

    return combined


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
