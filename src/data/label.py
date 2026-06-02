"""Apply the category schema to the processed Spiir export data.

Reads all yearly Spiir parquet files, maps Spiir category names to our schema
using v2.yaml, drops excluded rows, and writes a single labeled training file.

Output: data/processed/training/labeled.parquet
Columns: date, description, amount, currency, source, account_name, label
"""

from pathlib import Path

import pandas as pd

from src.config.load_config import (
    get_excluded_spiir,
    get_manual_review_patterns,
    get_spiir_mappings,
    load_category_config,
)

DEFAULT_CONFIG = "configs/categories/v2.yaml"
DEFAULT_SPIIR_DIR = "data/processed/spiir"
DEFAULT_OUTPUT = "data/processed/training/labeled.parquet"


def build_training_data(
    config_path: str = DEFAULT_CONFIG,
    spiir_dir: str = DEFAULT_SPIIR_DIR,
    output_path: str = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    config = load_category_config(config_path)
    spiir_map = get_spiir_mappings(config)
    excluded = get_excluded_spiir(config)

    # Load all yearly Spiir parquet files
    frames = [
        pd.read_parquet(p)
        for p in sorted(Path(spiir_dir).glob("*.parquet"))
    ]
    df = pd.concat(frames, ignore_index=True)

    # Drop rows where Spiir gave no category (uncategorized transactions)
    df = df.dropna(subset=["category"])

    # Drop excluded Spiir categories
    df = df[~df["category"].isin(excluded)].copy()

    # Drop transactions that require manual labeling (e.g. MobilePay)
    review_patterns = get_manual_review_patterns(config)
    if review_patterns:
        combined = "|".join(f"(?:{p})" for p in review_patterns)
        needs_review = df["description"].str.contains(combined, case=False, regex=True, na=False)
        n_dropped = needs_review.sum()
        print(f"Excluded {n_dropped} manual-review rows ({', '.join(review_patterns)})")
        df = df[~needs_review].copy()

    # Map remaining Spiir categories to our schema
    df["label"] = df["category"].map(spiir_map)

    # Any Spiir category not covered by the map becomes unmapped — surface them
    unmapped = df[df["label"].isna()]["category"].unique()
    if len(unmapped):
        print(f"Warning: {len(unmapped)} unmapped Spiir categories (dropped):")
        for c in sorted(unmapped):
            print(f"  - {c}")
        df = df.dropna(subset=["label"])

    output_cols = ["date", "description", "amount", "currency", "source", "account_name", "label"]
    df = df[output_cols].reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    return df
