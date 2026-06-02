"""Preprocessing orchestration for transaction data sources.

Two independent pipelines:

  preprocess_spiir(input_path, output_dir)
      Historic labeled data from the Spiir app. Partitioned by year.
      Output: data/processed/spiir/YYYY.parquet

  preprocess_lsb(input_path, output_dir)
      Ongoing bank transaction exports. Partitioned by quarter.
      Merges with existing parquet files and deduplicates so overlapping
      bank exports do not create duplicate rows.
      Output: data/processed/bank/YYYY-QN.parquet
"""

from pathlib import Path

import pandas as pd

from src.data.ingest.lsb import parse_lsb
from src.data.ingest.spiir import parse_spiir

# Columns used to identify duplicate transactions when merging bank exports
_DEDUP_COLS = ["date", "description", "amount"]


def _quarter_label(ts: pd.Timestamp) -> str:
    q = (ts.month - 1) // 3 + 1
    return f"{ts.year}-Q{q}"


def _write_parquet_merge(df: pd.DataFrame, path: Path) -> int:
    """Write df to path, merging with any existing file and deduplicating.

    Returns the number of new rows added (0 on first write).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        before = len(existing)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=_DEDUP_COLS, keep="first")
        new_rows = len(combined) - before
    else:
        combined = df.copy()
        new_rows = len(combined)

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return new_rows


def preprocess_spiir(input_path: str, output_dir: str) -> None:
    """Parse a Spiir export and write one parquet file per year.

    Existing yearly files are overwritten since Spiir data is a one-time
    historic export that does not change.
    """
    df = parse_spiir(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for year, group in df.groupby(df["date"].dt.year):
        path = output_dir / f"{year}.parquet"
        group = group.sort_values("date").reset_index(drop=True)
        group.to_parquet(path, index=False)
        print(f"  {path.name}: {len(group)} rows")

    print(f"Spiir: wrote {df['date'].dt.year.nunique()} yearly files to {output_dir}")


def preprocess_lsb(input_path: str, output_dir: str) -> None:
    """Parse an LSB bank export and merge into quarterly parquet files.

    Safe to run repeatedly with overlapping exports — duplicates are removed
    using (date, description, amount) as the deduplication key.
    """
    df = parse_lsb(input_path)
    output_dir = Path(output_dir)

    df["_quarter"] = df["date"].apply(_quarter_label)

    for quarter, group in df.groupby("_quarter"):
        group = group.drop(columns=["_quarter"])
        path = output_dir / f"{quarter}.parquet"
        new_rows = _write_parquet_merge(group, path)
        print(f"  {path.name}: +{new_rows} new rows (total {len(pd.read_parquet(path))})")

    print(f"LSB: processed {len(df)} rows from {input_path}")
