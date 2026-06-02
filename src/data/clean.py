import re
import pandas as pd


def parse_danish_amount(s: str) -> float:
    """Convert Danish-formatted number string to float.

    Danish format uses '.' as thousands separator and ',' as decimal:
    '-2.000,00' -> -2000.0, '6.410,70' -> 6410.7
    """
    s = str(s).strip()
    s = s.replace(".", "").replace(",", ".")
    return float(s)


def parse_danish_date(s: str) -> pd.Timestamp:
    """Parse DD-MM-YYYY date string."""
    return pd.to_datetime(s, format="%d-%m-%Y")


def clean_description(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
