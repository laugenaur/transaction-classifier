"""Fine-tuning loop for transaction classification.

Each run is saved to models/runs/run_YYYYMMDD_HHMMSS/ containing:
  config.json        hyperparameters, schema version, data split sizes
  training_log.json  loss and accuracy per epoch
  best_model.pt      weights at lowest validation loss
  eval_report.json   per-class F1 on the held-out test set

Run from project root:
  python -m src.training.train
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config.load_config import get_category_names, load_category_config
from src.training.dataset import TransactionDataset, stratified_split
from src.training.evaluate import evaluate, print_report, save_report

# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_NAME    = "xlm-roberta-base"
BATCH_SIZE    = 32
EPOCHS        = 5
LR_BACKBONE   = 2e-5
LR_HEAD       = 1e-4
WARMUP_FRAC   = 0.1
MAX_GRAD_NORM = 1.0
MAX_LENGTH    = 64
SEED          = 42

CONFIG_PATH   = "configs/categories/v2.yaml"
DATA_PATH     = "data/processed/training/labeled.parquet"
RUNS_DIR      = Path("models/runs")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["label"].to(device),
        )
        outputs.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += outputs.loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["label"].to(device),
        )
        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["label"].to(device)).sum().item()
        total   += batch["label"].size(0)

    return total_loss / len(loader), correct / total


def main():
    device = get_device()
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_id}  |  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    config     = load_category_config(CONFIG_PATH)
    categories = get_category_names(config)
    label2id   = {name: i for i, name in enumerate(categories)}
    id2label   = {i: name for name, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df        = pd.read_parquet(DATA_PATH)
    dataset   = TransactionDataset(df, label2id, tokenizer, max_length=MAX_LENGTH)

    train_ds, val_ds, test_ds = stratified_split(dataset, seed=SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Save run config ────────────────────────────────────────────────────────
    run_config = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "category_config": CONFIG_PATH,
        "category_schema_version": config["metadata"]["version"],
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_backbone": LR_BACKBONE,
            "lr_head": LR_HEAD,
            "warmup_frac": WARMUP_FRAC,
            "max_grad_norm": MAX_GRAD_NORM,
            "max_length": MAX_LENGTH,
            "seed": SEED,
        },
        "data": {
            "file": DATA_PATH,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "n_classes": len(categories),
            "classes": categories,
        },
        "started_at": datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(categories)
    ).to(device)

    head_params    = list(model.classifier.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ],
        weight_decay=0.01,
    )

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{len(train_ds)} train  {len(val_ds)} val  {len(test_ds)} test  "
          f"|  {len(train_loader)} batches/epoch  |  {warmup_steps} warmup steps\n")
    print(f"{'epoch':<7} {'train_loss':>10} {'val_loss':>9} {'val_acc':>8}")
    print("─" * 38)

    training_log = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

        marker = " ←" if is_best else ""
        print(f"{epoch:<7} {train_loss:>10.4f} {val_loss:>9.4f} {val_acc:>8.3f}{marker}")

        training_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })

    (run_dir / "training_log.json").write_text(json.dumps(training_log, indent=2))

    # ── Final evaluation on test set (best checkpoint) ────────────────────────
    print(f"\nLoading best checkpoint (epoch {best_epoch}) for test evaluation...")
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    report = evaluate(model, test_loader, device, id2label)
    save_report(report, run_dir / "eval_report.json")
    print_report(report)

    # Append completion metadata to config
    run_config["completed_at"] = datetime.now().isoformat()
    run_config["best_epoch"] = best_epoch
    run_config["best_val_loss"] = round(best_val_loss, 6)
    run_config["test_accuracy"] = round(report["accuracy"], 4)
    run_config["test_macro_f1"] = round(report["macro avg"]["f1-score"], 4)
    run_config["test_weighted_f1"] = round(report["weighted avg"]["f1-score"], 4)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    print(f"\nArtifacts saved to {run_dir}/")


if __name__ == "__main__":
    main()
