"""Parser for Lån og Spar Bank (LSB) bank export formats.

LSB exports three CSV variants, all semicolon-separated with Danish number
formatting and no header row:

  simple_no_balance    4 cols: date, description, amount, currency
  simple_with_balance  5 cols: date, description, amount, balance, currency
  advanced            16 cols: see _ADV column constants below

The format is auto-detected from the column count.
"""

from pathlib import Path

import pandas as pd

from src.data.clean import clean_description, parse_danish_amount, parse_danish_date

# Column positions in the advanced export (0-indexed)
_ADV_NOTE = 0
_ADV_DESC = 1
_ADV_AMOUNT = 4
_ADV_BOOKING_DATE = 8  # matches the date shown in the simple format
_ADV_TX_REF = 11

OUTPUT_COLS = ["date", "description", "amount", "currency", "source"]


def _detect_format(ncols: int) -> str:
    if ncols == 4:
        return "simple_no_balance"
    if ncols == 5:
        return "simple_with_balance"
    if ncols >= 12:
        return "advanced"
    raise ValueError(f"Unrecognized LSB format: {ncols} columns")


def _parse_simple(df_raw: pd.DataFrame, has_balance: bool) -> pd.DataFrame:
    cols = (
        ["date", "description", "amount", "balance", "currency"]
        if has_balance
        else ["date", "description", "amount", "currency"]
    )
    df = df_raw.copy()
    df.columns = cols
    out = df[["date", "description", "amount", "currency"]].copy()
    out["source"] = "lsb_simple"
    return out


def _parse_advanced(df_raw: pd.DataFrame) -> pd.DataFrame:
    note = df_raw.iloc[:, _ADV_NOTE].astype(str).str.strip()
    desc = df_raw.iloc[:, _ADV_DESC].astype(str).str.strip()

    # Use description column; fall back to note when description is empty
    combined_desc = desc.where(desc != "", other=note)

    out = pd.DataFrame(
        {
            "date": df_raw.iloc[:, _ADV_BOOKING_DATE],
            "description": combined_desc,
            "amount": df_raw.iloc[:, _ADV_AMOUNT],
            "currency": "DKK",
            "source": "lsb_advanced",
        }
    )
    return out


def parse_lsb(path: str | Path) -> pd.DataFrame:
    """Parse any LSB export CSV and return a normalized DataFrame.

    Returns columns: date (datetime64), description (str), amount (float64),
    currency (str), source (str).
    """
    path = Path(path)
    df_raw = pd.read_csv(
        path,
        sep=";",
        header=None,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        quotechar='"',
    )

    fmt = _detect_format(len(df_raw.columns))

    if fmt == "simple_no_balance":
        df = _parse_simple(df_raw, has_balance=False)
    elif fmt == "simple_with_balance":
        df = _parse_simple(df_raw, has_balance=True)
    else:
        df = _parse_advanced(df_raw)

    df["date"] = df["date"].apply(parse_danish_date)
    df["amount"] = df["amount"].apply(parse_danish_amount)
    df["description"] = df["description"].apply(clean_description)

    return df[OUTPUT_COLS].reset_index(drop=True)
