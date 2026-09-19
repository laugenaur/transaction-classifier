# sure.am Import Pipeline

A guide for processing raw transaction data into sure.am-ready CSV files.

---

## Overview

There are two independent pipelines:

| Pipeline | When to run | Input | Output |
|---|---|---|---|
| **Spiir** | Once — historic import | Spiir export CSV | `spiir_accounts.csv`, `spiir_YYYY.csv` per year |
| **LSB bank** | Each new bank export | LSB export CSV | `YYYY-QN.csv` |

Both pipelines share a common category configuration (`configs/categories/v2.yaml`) and produce CSVs that import directly into sure.am.

---

## Prerequisites

**Activate the conda environment** before running any commands:

```bash
conda activate transaction-classifier
```

All Python commands below are run from the **project root**
(`/Users/lauge/Projects/transaction-classifier`).

**Active model:** `models/runs/run_20260602_224238/best_model.pt`
- Trained on category schema v2.0
- Test accuracy: 85.9% | Macro F1: 0.723
- Auto-labels ~74% of bank transactions at the 0.70 confidence threshold

---

## Part 1 — Spiir historic import (one-time)

Run this once per Spiir export file. The three steps produce all files needed to
fully set up sure.am with your transaction history.

### Step 1 — Preprocess the Spiir CSV

Parses the raw Spiir export and writes one Parquet file per year to
`data/processed/spiir/`.

```python
from src.data.preprocess import preprocess_spiir

preprocess_spiir(
    input_path="data/export from spiir/alle-poster-2026-04-19.csv",
    output_dir="data/processed/spiir",
)
```

Expected output:
```
data/processed/spiir/2017.parquet  ...  2026.parquet
```

### Step 2 — Export accounts CSV

Reads the original Spiir CSV to extract the latest balance for each account.
Import this into sure.am **before** importing any transactions.

```python
from src.config.load_config import load_category_config
from src.export.sure_am import export_spiir_accounts

export_spiir_accounts(
    spiir_csv="data/export from spiir/alle-poster-2026-04-19.csv",
    output_path="data/export/sure_am/spiir_accounts.csv",
)
```

Output: `data/export/sure_am/spiir_accounts.csv`

> **Check before importing:** `Andels kassekredit` exports with balance `0.00`
> because Spiir had no valid balance recorded. Verify the real balance in your
> bank and correct it manually in sure.am after import.

### Step 3 — Export categories CSV

Generates all 18 sure.am categories (parents, model categories, and Transfer).
Import this into sure.am **before** importing transactions.

```python
from src.config.load_config import load_category_config
from src.export.sure_am import export_categories

config = load_category_config("configs/categories/v2.yaml")
export_categories(config, "data/export/sure_am/categories.csv")
```

Output: `data/export/sure_am/categories.csv`

### Step 4 — Export transaction CSVs

Writes one CSV per year. Each transaction is either auto-labeled or flagged
`needs_review`.

```python
from src.config.load_config import load_category_config
from src.export.sure_am import export_spiir_transactions

config = load_category_config("configs/categories/v2.yaml")
export_spiir_transactions(
    spiir_dir="data/processed/spiir",
    config=config,
    output_dir="data/export/sure_am",
)
```

Output: `data/export/sure_am/spiir_2017.csv` through `spiir_2026.csv`

Coverage across 5,099 transactions:
- **83.5% auto-labeled** (category set, ready to import)
- **16.5% needs review** (empty category, tagged `needs_review` in sure.am)

---

## Part 2 — LSB bank export (recurring)

Run this each time you download a new export from the bank.
Supports all three LSB export formats — the format is detected automatically.

### Step 1 — Place the export file

Copy the downloaded CSV into `data/raw/`. The filename is not important.

```
data/raw/lsb_simpel_uden_balance.csv   ← simple export (4 columns)
data/raw/lsb_simpel_med_balance.csv    ← simple export with balance (5 columns)
data/raw/lsb_avanceret_eksport.csv     ← advanced export (16 columns)
```

### Step 2 — Run the full pipeline

Parses the bank CSV, strips LSB boilerplate from descriptions, runs the
classifier, and writes the sure.am import file in one block:

```python
from src.config.load_config import load_category_config
from src.data.ingest.lsb import parse_lsb
from src.export.sure_am import export_transactions
from src.inference.predict import Predictor

config    = load_category_config("configs/categories/v2.yaml")
predictor = Predictor("models/runs/run_20260602_224238/best_model.pt")

df_bank = parse_lsb("data/raw/lsb_simpel_uden_balance.csv")
df_pred = predictor.predict_dataframe(df_bank)

export_transactions(
    df=df_pred,
    config=config,
    output_path="data/export/sure_am/2026-Q2.csv",   # name by quarter
    account="LSB Checking",                            # must match account name in sure.am
)
```

Output: `data/export/sure_am/2026-Q2.csv`

> **Naming convention:** name the output file by the quarter it covers, e.g.
> `2026-Q2.csv`. This prevents accidentally importing the same period twice.

---

## sure.am import order

Import files into sure.am in this order to avoid reference errors:

```
1. categories.csv       (Settings → Categories → Import)
2. spiir_accounts.csv   (Accounts → Import)
3. spiir_2017.csv       (Transactions → Import)
   spiir_2018.csv
   ...
   spiir_2026.csv
```

For ongoing LSB exports, just import the quarterly file:
```
4. 2026-Q2.csv          (Transactions → Import)
```

---

## Interpreting needs_review transactions

All transactions flagged `needs_review` arrive in sure.am with an empty
category. Filter by tag in sure.am to find them.

| `notes` value | Reason | Action |
|---|---|---|
| `awaiting manual label` | MobilePay or other pattern-matched payment | Assign category manually |
| `confidence: 0.xx` | Model prediction below 0.70 threshold | Review suggested category in `notes`; confirm or correct |
| `excluded` | Udlæg or Ignorer tag in Spiir | Assign category or delete |
| `uncategorized` | Transaction had no category in Spiir | Assign category manually |

---

## Updating the model

When enough manually-labeled transactions have accumulated:

1. Export corrected transactions from sure.am
2. Add them to the training data in `data/processed/training/`
3. Retrain: `python -m src.training.train`
4. Update the model path in Step 2 of the LSB pipeline to the new run ID

Each retraining run is saved to `models/runs/run_YYYYMMDD_HHMMSS/` with its
full performance report in `eval_report.json`.
