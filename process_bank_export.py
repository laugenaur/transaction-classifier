#!/usr/bin/env python3
"""Process a raw LSB bank export through the classifier and load it into sure.am.

The script always writes a sure.am import CSV (a handy audit trail). With
``--push`` it additionally posts every transaction straight to a running sure.am
instance via its REST API, so nothing has to be uploaded by hand. Re-running is
safe: each transaction carries a stable id and sure.am de-duplicates server-side.

Mandatory arguments:
  bank_export   Path to the raw LSB bank export CSV (format is auto-detected).
  output_file   Destination path for the sure.am transaction import CSV.
  account_name  Account name exactly as it appears in sure.am.

Optional arguments:
  --config PATH       Category config YAML. Defaults to the highest-version file
                      found in configs/categories/.
  --model PATH        Path to best_model.pt. Defaults to the trained run with the
                      highest test macro-F1 in models/runs/.
  --push              After exporting the CSV, push transactions to sure.am's API.
  --dry-run           With --push, print what would be sent without posting.
  --sure-url URL      sure.am base URL. Default: env SURE_BASE_URL or
                      http://localhost:3000.
  --sure-api-key KEY  API key (Settings -> API Keys, read_write scope).
                      Default: env SURE_API_KEY.
  --source NAME       Idempotency source tag. Default: lsb-classifier.

Examples:
  # Just write the CSV (latest config and best model resolved automatically)
  python process_bank_export.py \\
      data/raw/lsb_simpel_uden_balance.csv \\
      data/export/sure_am/2026-Q2.csv \\
      "LSB Checking"

  # Write the CSV and push to a local sure.am (API key from the environment)
  export SURE_API_KEY=your_key_here    # from sure.am Settings -> API Keys
  python process_bank_export.py \\
      data/raw/lsb_simpel_uden_balance.csv \\
      data/export/sure_am/2026-Q2.csv \\
      "LSB Checking" --push

  # Preview the API payloads without sending anything
  python process_bank_export.py \\
      data/raw/lsb_simpel_uden_balance.csv \\
      data/export/sure_am/2026-Q2.csv \\
      "LSB Checking" --push --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make `from src.xxx import ...` work when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent))

import yaml


def _find_latest_config() -> Path:
    """Return the category config YAML with the highest metadata.version."""
    configs = list((Path("configs") / "categories").glob("*.yaml"))
    if not configs:
        sys.exit("Error: no category config found in configs/categories/")

    def _version(path: Path) -> float:
        try:
            with open(path) as f:
                return float(yaml.safe_load(f).get("metadata", {}).get("version", 0))
        except Exception:
            return 0.0

    return max(configs, key=_version)


def _find_best_model() -> Path:
    """Return best_model.pt from the run with the highest test_macro_f1."""
    runs_dir = Path("models") / "runs"
    if not runs_dir.exists():
        sys.exit("Error: models/runs/ not found — train a model first.")

    best_f1, best_pt = -1.0, None
    for config_json in runs_dir.glob("*/config.json"):
        try:
            data   = json.loads(config_json.read_text())
            f1     = float(data.get("test_macro_f1", -1))
            pt     = config_json.parent / "best_model.pt"
            if f1 > best_f1 and pt.exists():
                best_f1, best_pt = f1, pt
        except Exception:
            continue

    if best_pt is None:
        sys.exit("Error: no trained model found in models/runs/ — train a model first.")

    return best_pt


def _parse_args() -> argparse.Namespace:
    doc_lines = __doc__.splitlines()
    examples_start = doc_lines.index("Examples:")
    parser = argparse.ArgumentParser(
        description="Process a raw LSB bank export through the classifier, export a "
                    "sure.am CSV, and optionally push it to sure.am's API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(doc_lines[examples_start:]),  # examples section
    )
    parser.add_argument(
        "bank_export",
        type=Path,
        help="Raw LSB bank export CSV (format auto-detected from column count).",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Destination path for the sure.am transaction import CSV.",
    )
    parser.add_argument(
        "account_name",
        type=str,
        help="Account name exactly as it appears in sure.am (e.g. 'LSB Checking').",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Category config YAML. Defaults to the highest-version file in configs/categories/.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to best_model.pt. Defaults to the run with the highest test macro-F1.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="After exporting the CSV, push transactions to sure.am's REST API.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --push, print the transactions that would be sent without posting.",
    )
    parser.add_argument(
        "--sure-url",
        type=str,
        default=os.environ.get("SURE_BASE_URL", "http://localhost:3000"),
        metavar="URL",
        help="sure.am base URL. Default: env SURE_BASE_URL or http://localhost:3000.",
    )
    parser.add_argument(
        "--sure-api-key",
        type=str,
        default=os.environ.get("SURE_API_KEY", ""),
        metavar="KEY",
        help="sure.am API key (read_write scope). Default: env SURE_API_KEY.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="lsb-classifier",
        metavar="NAME",
        help="Idempotency source tag stored on each transaction. Default: lsb-classifier.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Validate mandatory input ───────────────────────────────────────────────
    if not args.bank_export.exists():
        sys.exit(f"Error: bank export file not found: {args.bank_export}")

    if args.push and not args.dry_run and not args.sure_api_key:
        sys.exit(
            "Error: --push needs an API key. Set SURE_API_KEY or pass --sure-api-key "
            "(create one in sure.am under Settings -> API Keys, read_write scope).\n"
            "       Tip: use --push --dry-run to preview payloads without a key."
        )

    # ── Resolve optional defaults ──────────────────────────────────────────────
    config_path = args.config or _find_latest_config()
    model_path  = args.model  or _find_best_model()

    if not config_path.exists():
        sys.exit(f"Error: config file not found: {config_path}")
    if not model_path.exists():
        sys.exit(f"Error: model file not found: {model_path}")

    # ── Print resolved configuration ───────────────────────────────────────────
    # Imports are deferred to here so argument errors fail fast, before
    # PyTorch and Transformers spend several seconds loading.
    print(f"Bank export  : {args.bank_export}")
    print(f"Output       : {args.output_file}")
    print(f"Account      : {args.account_name!r}")
    print(f"Config       : {config_path}")
    print(f"Model        : {model_path}")
    print()

    # ── Import pipeline modules ────────────────────────────────────────────────
    from src.config.load_config import load_category_config
    from src.data.ingest.lsb import parse_lsb
    from src.export.sure_am import export_transactions
    from src.inference.predict import Predictor

    # ── Run pipeline ───────────────────────────────────────────────────────────
    config    = load_category_config(str(config_path))
    predictor = Predictor(model_path, config_path=str(config_path))

    print("Parsing bank export...")
    df_bank = parse_lsb(args.bank_export)
    print(f"  {len(df_bank)} transactions parsed")

    print("Running classifier...")
    df_pred = predictor.predict_dataframe(df_bank)

    print("Exporting to sure.am format...")
    export_transactions(
        df=df_pred,
        config=config,
        output_path=args.output_file,
        account=args.account_name,
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    auto   = (~df_pred["needs_review"]).sum()
    review = df_pred["needs_review"].sum()
    total  = len(df_pred)

    reasons = df_pred[df_pred["needs_review"]]["review_reason"].value_counts()

    print()
    print("─" * 42)
    print(f"  Auto-labeled   {auto:>4}  ({100 * auto / total:.0f}%)")
    for reason, count in reasons.items():
        label = "pattern match" if reason == "pattern_match" else "low confidence"
        print(f"  Needs review   {count:>4}  ({100 * count / total:.0f}%)  [{label}]")
    print(f"  {'─' * 36}")
    print(f"  Total          {total:>4}")
    print("─" * 42)

    # ── Push to sure.am (optional) ─────────────────────────────────────────────
    if args.push or args.dry_run:
        from src.load.sure_client import SureClient

        mode = "dry-run (no data sent)" if args.dry_run else f"pushing to {args.sure_url}"
        print()
        print(f"sure.am {mode} ...")

        client = SureClient(
            base_url=args.sure_url,
            api_key=args.sure_api_key,
            source=args.source,
            dry_run=args.dry_run,
        )
        try:
            result = client.push_dataframe(
                df=df_pred, config=config, account_name=args.account_name
            )
        except Exception as exc:  # connection refused, bad account name, etc.
            sys.exit(f"Error: could not push to sure.am: {exc}")

        print(result.summary())


if __name__ == "__main__":
    main()
