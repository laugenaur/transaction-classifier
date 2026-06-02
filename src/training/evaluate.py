"""Evaluation utilities for transaction classification.

evaluate() runs inference on any DataLoader and returns a structured metrics
dict with per-class precision, recall, F1 and support, plus macro/weighted
averages. Accuracy is included for quick scanning but macro-F1 is the primary
metric — it treats all classes equally regardless of size.

print_report() renders a readable table to stdout.
"""

import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    id2label: dict[int, str],
) -> dict:
    """Return sklearn classification report as a nested dict.

    Keys: one entry per class name, plus 'accuracy', 'macro avg',
    'weighted avg'. Each class entry has: precision, recall, f1-score, support.
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(batch["label"].tolist())

    n_classes   = len(id2label)
    label_names = [id2label[i] for i in range(n_classes)]

    return classification_report(
        all_labels,
        all_preds,
        labels=list(range(n_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )


def print_report(report: dict) -> None:
    """Print per-class metrics table, sorted by F1 descending."""
    skip = {"accuracy", "macro avg", "weighted avg"}
    classes = [(k, v) for k, v in report.items() if k not in skip]
    classes.sort(key=lambda x: x[1]["f1-score"], reverse=True)

    print(f"\n{'category':<18} {'precision':>9} {'recall':>7} {'f1':>6} {'n':>5}")
    print("─" * 48)
    for name, m in classes:
        print(
            f"{name:<18} {m['precision']:>9.3f} {m['recall']:>7.3f}"
            f" {m['f1-score']:>6.3f} {int(m['support']):>5}"
        )
    print("─" * 48)
    print(
        f"{'macro avg':<18} {report['macro avg']['precision']:>9.3f}"
        f" {report['macro avg']['recall']:>7.3f}"
        f" {report['macro avg']['f1-score']:>6.3f}"
        f" {int(report['weighted avg']['support']):>5}"
    )
    print(
        f"{'weighted avg':<18} {report['weighted avg']['precision']:>9.3f}"
        f" {report['weighted avg']['recall']:>7.3f}"
        f" {report['weighted avg']['f1-score']:>6.3f}"
    )
    print(f"\n  accuracy: {report['accuracy']:.3f}")


def save_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2))
