import yaml


def load_category_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_label_mappings(config: dict) -> dict:
    """alias -> category name, used for rule-based labeling of new transactions."""
    mappings = {}
    for category in config.get("categories", []):
        label = category["name"]
        for alias in category.get("aliases", []):
            mappings[alias] = label
    return mappings


def get_spiir_mappings(config: dict) -> dict:
    """Spiir category name -> our category name, for mapping historic export data."""
    mappings = {}
    for category in config.get("categories", []):
        label = category["name"]
        for spiir_name in category.get("spiir_map", []):
            mappings[spiir_name] = label
    return mappings


def get_excluded_spiir(config: dict) -> set:
    """Spiir category names that should be dropped from training data entirely."""
    return set(config.get("excluded_spiir", []))


def get_category_names(config: dict) -> list[str]:
    """Ordered list of model class names."""
    return [c["name"] for c in config.get("categories", [])]


def get_manual_review_patterns(config: dict) -> list[str]:
    """Regex patterns for descriptions that require manual labeling.

    Transactions matching any pattern are excluded from training and skipped
    at inference time — they go to a manual review queue instead.
    """
    return config.get("manual_review_patterns", [])


def get_confidence_threshold(config: dict) -> float:
    """Min softmax probability for a prediction to be auto-labeled.

    Predictions below this threshold are marked unknown and routed to manual
    review instead of being auto-labeled.
    """
    return config.get("inference_confidence_threshold", 0.70)
