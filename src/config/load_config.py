import yaml
from pathlib import Path

def load_category_config(path: str) -> dict:
    """
    Load category configuration from a YAML file.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the category configuration.
    """

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config

def get_label_mappings(config: dict) -> dict:
    """
    Get label mappings from the category configuration.

    Args:
        config (dict): The category configuration dictionary.

    Returns:
        dict: A dictionary containing the label mappings.
    """
    label_mappings = {}
    for category in config.get("categories", []):
        label = category["name"]
        for alias in category.get("aliases", []):
            label_mappings[alias] = label
    return label_mappings

