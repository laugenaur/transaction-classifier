"""Formatters for sure.am import files.

sure.am accepts three CSV import types:

  Accounts      Account type*, Name*, Balance*, Currency, Balance Date
  Categories    name*, color, parent_category, lucide_icon
  Transactions  date*, amount*, name, currency, category, tags, account, notes

Typical usage for ongoing bank exports:

  config = load_category_config("configs/categories/v2.yaml")
  export_categories(config, "data/export/sure_am/categories.csv")

  predictor = Predictor("models/runs/<run_id>/best_model.pt")
  df_bank   = parse_lsb("data/raw/lsb_simpel_uden_balance.csv")
  df_pred   = predictor.predict_dataframe(df_bank)
  export_transactions(df_pred, config, "data/export/sure_am/2026-Q1.csv")

For historic Spiir data (one-time):

  export_spiir_accounts(spiir_csv, config, "data/export/sure_am/spiir_accounts.csv")
  export_spiir_transactions(spiir_dir, config, "data/export/sure_am/")
"""

import re
from pathlib import Path

import pandas as pd

from src.config.load_config import (
    get_excluded_spiir,
    get_manual_review_patterns,
    get_spiir_mappings,
)
from src.data.clean import parse_danish_amount


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _label_to_sure_map(config: dict) -> dict[str, str]:
    """model label → sure.am category name."""
    return {
        cat["name"]: cat.get("sure_am", {}).get("name", cat["name"])
        for cat in config.get("categories", [])
    }


def _write(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Wrote {len(df):>5} rows  →  {path}")


# ── Categories ─────────────────────────────────────────────────────────────────

def export_categories(config: dict, output_path: str | Path) -> pd.DataFrame:
    """Write sure.am category import CSV from v2.yaml.

    Order: parent categories first, then model categories, then rule-based
    categories (e.g. Transfer) so sure.am resolves all references in one pass.
    """
    rows = []

    for parent in config.get("sure_am_parents", []):
        rows.append({
            "name*": parent["name"],
            "color": parent.get("color", ""),
            "parent_category": "",
            "lucide_icon": parent.get("icon", ""),
        })

    for cat in config.get("categories", []):
        sure = cat.get("sure_am", {})
        rows.append({
            "name*": sure.get("name", cat["name"]),
            "color": sure.get("color", ""),
            "parent_category": sure.get("parent", ""),
            "lucide_icon": sure.get("icon", ""),
        })

    # Rule-based categories (Transfer) — detected by rule, not the model,
    # but still need to exist in sure.am as a category.
    for rb in config.get("rule_based", []):
        rows.append({
            "name*": rb["name"].capitalize(),
            "color": "#94a3b8",
            "parent_category": "",
            "lucide_icon": "arrow-left-right",
        })

    df = pd.DataFrame(rows)
    _write(df, output_path)
    return df


# ── Ongoing bank transactions (LSB) ───────────────────────────────────────────

def export_transactions(
    df: pd.DataFrame,
    config: dict,
    output_path: str | Path,
    account: str = "",
) -> pd.DataFrame:
    """Write sure.am transaction import CSV from a predicted transaction DataFrame.

    df must be the output of Predictor.predict_dataframe(), containing:
    date, amount, description, currency, label, confidence,
    needs_review, review_reason.
    """
    l2s = _label_to_sure_map(config)

    out = pd.DataFrame({
        "date*":    pd.to_datetime(df["date"]).dt.strftime("%m/%d/%Y"),
        "amount*":  df["amount"],
        "name":     df["description"],
        "currency": df.get("currency", "DKK"),
        "category": df["label"].map(l2s).fillna(""),
        "tags":     df["needs_review"].apply(lambda x: "needs_review" if x else ""),
        "account":  account,
        "notes":    df.apply(_bank_notes, axis=1),
    })

    _write(out, output_path)
    return out


def _bank_notes(row) -> str:
    if row["review_reason"] == "low_confidence":
        return f"confidence: {row['confidence']:.2f}"
    if row["review_reason"] == "pattern_match":
        return "awaiting manual label"
    return ""


# ── Historic Spiir data (one-time) ────────────────────────────────────────────

def _infer_account_type(name: str) -> str:
    """Guess sure.am account type from the Spiir account name."""
    n = name.lower()
    if any(k in n for k in ("opsparing", "savings")):
        return "Savings"
    if any(k in n for k in ("kredit", "kassekredit", "credit")):
        return "Credit Card"
    return "Checking"


def export_spiir_accounts(
    spiir_csv: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Derive latest balance per account from the Spiir export CSV.

    Reads the original CSV (not the processed parquets) because the Balance
    column was not carried through to the parquet files.
    """
    df = pd.read_csv(
        spiir_csv,
        sep=";", quotechar='"', encoding="utf-8-sig",
        dtype=str, keep_default_na=False,
    )
    df["Date"]    = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Balance"] = df["Balance"].replace("", None).apply(
        lambda x: parse_danish_amount(x) if x is not None else None
    )

    # Latest transaction per account carries the most recent balance
    latest = (
        df.sort_values("Date")
        .groupby("AccountName", sort=False)
        .last()
        .reset_index()
    )

    out = pd.DataFrame({
        "Account type*": latest["AccountName"].apply(_infer_account_type),
        "Name*":         latest["AccountName"],
        "Balance*":      latest["Balance"].fillna(0.0),
        "Currency":      latest["Currency"],
        "Balance Date":  latest["Date"].dt.strftime("%m/%d/%Y"),
    })

    _write(out, output_path)
    return out


def export_spiir_transactions(
    spiir_dir: str | Path,
    config: dict,
    output_dir: str | Path,
) -> None:
    """Write one sure.am transaction CSV per year from the processed Spiir parquets.

    Category mapping:
      Kontooverførsel  → Transfer  (rule-based label, no manual review needed)
      Udlæg / Ignorer  → empty category + needs_review tag
      NaN category     → empty category + needs_review tag
      MobilePay desc   → empty category + needs_review tag
      Everything else  → mapped through spiir_map → sure.am name
    """
    spiir_map  = get_spiir_mappings(config)
    excluded   = get_excluded_spiir(config)
    l2s        = _label_to_sure_map(config)
    patterns   = get_manual_review_patterns(config)
    review_re  = (
        re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
        if patterns else None
    )

    parquet_files = sorted(Path(spiir_dir).glob("*.parquet"))

    for parquet_path in parquet_files:
        year = parquet_path.stem          # e.g. "2024"
        df   = pd.read_parquet(parquet_path)

        rows = []
        for _, tx in df.iterrows():
            cat  = tx["category"] if pd.notna(tx.get("category")) else None
            desc = str(tx["description"])

            # Determine sure.am category and review status
            if cat == "Kontooverførsel":
                sure_cat    = "Transfer"
                tag         = ""
                note        = ""
            elif cat in excluded or cat is None:
                sure_cat    = ""
                tag         = "needs_review"
                note        = "excluded" if cat in excluded else "uncategorized"
            elif review_re and review_re.search(desc):
                sure_cat    = ""
                tag         = "needs_review"
                note        = "awaiting manual label"
            else:
                label       = spiir_map.get(cat)
                sure_cat    = l2s.get(label, "") if label else ""
                tag         = "" if sure_cat else "needs_review"
                note        = "" if sure_cat else f"unmapped: {cat}"

            rows.append({
                "date*":    pd.to_datetime(tx["date"]).strftime("%m/%d/%Y"),
                "amount*":  tx["amount"],
                "name":     desc,
                "currency": tx.get("currency", "DKK"),
                "category": sure_cat,
                "tags":     tag,
                "account":  tx.get("account_name", ""),
                "notes":    note,
            })

        out = pd.DataFrame(rows)
        _write(out, Path(output_dir) / f"spiir_{year}.csv")
