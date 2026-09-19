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


# LSB raw exports prefix card payments and direct debits with these strings.
# Stripping them recovers the merchant name the model was trained to recognise.
_LSB_PREFIXES = re.compile(
    r"^(kontaktløs dankort|dankort-køb|visa/dankort|betalingsservice)\s+",
    re.IGNORECASE,
)
# LSB appends transaction reference codes that carry no category signal.
_LSB_SUFFIXES = re.compile(
    r"\s+(nota|notanr|aftalenr\.?)\s*[a-z0-9]+$",
    re.IGNORECASE,
)


def clean_description(text: str, strip_lsb_boilerplate: bool = False) -> str:
    text = str(text).lower().strip()
    if strip_lsb_boilerplate:
        text = _LSB_PREFIXES.sub("", text)
        text = _LSB_SUFFIXES.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
