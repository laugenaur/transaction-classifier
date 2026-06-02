"""Parser for Spiir budget app transaction exports.

Spiir exports a semicolon-separated CSV with a header row, Danish number
formatting, and rich category metadata. This is historic labeled data used
for training the classifier.

Expected columns (subset used):
  Date, Description, Amount, Currency, AccountName,
  MainCategoryName, CategoryName
"""

from pathlib import Path

import pandas as pd

from src.data.clean import clean_description, parse_danish_amount

OUTPUT_COLS = [
    "date",
    "description",
    "amount",
    "currency",
    "source",
    "account_name",
    "main_category",
    "category",
]


def parse_spiir(path: str | Path) -> pd.DataFrame:
    """Parse a Spiir export CSV and return a normalized DataFrame.

    Returns columns: date (datetime64), description (str), amount (float64),
    currency (str), source (str), account_name (str), main_category (str),
    category (str, NaN when uncategorized).
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=";",
        quotechar='"',
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date"], format="%d-%m-%Y"),
            "description": df["Description"].apply(clean_description),
            "amount": df["Amount"].apply(parse_danish_amount),
            "currency": df["Currency"],
            "source": "spiir",
            "account_name": df["AccountName"],
            "main_category": df["MainCategoryName"].replace("", pd.NA),
            "category": df["CategoryName"].replace("", pd.NA),
        }
    )

    return out[OUTPUT_COLS].reset_index(drop=True)
