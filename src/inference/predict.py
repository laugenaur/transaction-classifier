"""Transaction category inference.

Routing works in two stages before any label is assigned:

  1. Pattern check (pre-model)
     Descriptions matching manual_review_patterns in v2.yaml are flagged
     needs_review immediately — the model never runs on them.

  2. Confidence threshold (post-model)
     Predictions whose max softmax probability is below
     inference_confidence_threshold are flagged needs_review as "unknown".
     These are transactions the model has seen but cannot classify confidently.

Everything else receives an auto-assigned label.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.load_config import (
    get_category_names,
    get_confidence_threshold,
    get_manual_review_patterns,
    load_category_config,
)
from src.data.clean import clean_description


@dataclass
class Prediction:
    description: str       # cleaned input text
    label: str | None      # predicted category, or None when needs_review
    confidence: float      # max softmax probability (0–1)
    needs_review: bool
    review_reason: str     # "pattern_match" | "low_confidence" | ""


class Predictor:
    def __init__(
        self,
        model_path: str | Path,
        config_path: str = "configs/categories/v2.yaml",
        device: torch.device | None = None,
    ):
        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        config           = load_category_config(config_path)
        categories       = get_category_names(config)
        self.id2label    = {i: name for i, name in enumerate(categories)}
        self.threshold   = get_confidence_threshold(config)
        patterns         = get_manual_review_patterns(config)
        self._review_re  = (
            re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
            if patterns else None
        )

        model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base", num_labels=len(categories)
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device).eval()

    def _needs_pattern_review(self, description: str) -> bool:
        return self._review_re is not None and bool(self._review_re.search(description))

    @torch.no_grad()
    def predict(self, descriptions: list[str]) -> list[Prediction]:
        """Classify a list of raw transaction descriptions."""
        cleaned = [clean_description(d) for d in descriptions]

        # Stage 1 — pattern routing (no model call needed for these)
        needs_pattern = [self._needs_pattern_review(d) for d in cleaned]

        # Stage 2 — run model only on transactions that passed the pattern check
        model_indices = [i for i, flag in enumerate(needs_pattern) if not flag]
        results: list[Prediction | None] = [None] * len(cleaned)

        # Fill pattern-flagged results immediately
        for i, flag in enumerate(needs_pattern):
            if flag:
                results[i] = Prediction(
                    description=cleaned[i],
                    label=None,
                    confidence=0.0,
                    needs_review=True,
                    review_reason="pattern_match",
                )

        if model_indices:
            batch_texts = [cleaned[i] for i in model_indices]
            encoding = self.tokenizer(
                batch_texts,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
            )
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, pred_ids = probs.max(dim=-1)

            for list_pos, orig_idx in enumerate(model_indices):
                conf  = confidences[list_pos].item()
                pred  = pred_ids[list_pos].item()
                label = self.id2label[pred]

                if conf < self.threshold:
                    results[orig_idx] = Prediction(
                        description=cleaned[orig_idx],
                        label=None,
                        confidence=conf,
                        needs_review=True,
                        review_reason="low_confidence",
                    )
                else:
                    results[orig_idx] = Prediction(
                        description=cleaned[orig_idx],
                        label=label,
                        confidence=conf,
                        needs_review=False,
                        review_reason="",
                    )

        return results

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add label, confidence and needs_review columns to a transaction DataFrame.

        Expects a 'description' column. Returns the DataFrame with three new columns.
        """
        preds = self.predict(df["description"].tolist())
        out = df.copy()
        out["label"]         = [p.label for p in preds]
        out["confidence"]    = [round(p.confidence, 4) for p in preds]
        out["needs_review"]  = [p.needs_review for p in preds]
        out["review_reason"] = [p.review_reason for p in preds]
        return out
