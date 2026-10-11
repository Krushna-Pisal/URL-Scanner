"""Refresh explanatory report fields without fitting models or changing metrics.

This utility recomputes dataset-shape diagnostics from the current training
CSV and adds explicit types to existing computed feature-importance values.
Model metrics, fold results, thresholds, and ablations are preserved verbatim.
Run scripts/train.py for a full fresh evaluation when that longer fit is wanted.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import DATA_PATH, REPORT_PATH, _print_dataset_sanity


def refresh_report_diagnostics() -> None:
    """Update dataset warnings and importance labels while preserving scores."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates(subset="url").dropna(subset=["url", "label"]).copy()
    data["url"] = data["url"].astype(str).str.strip()
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    sanity, _feature_removal_decision = _print_dataset_sanity(data)
    report["sanity"] = sanity
    report["https_share_by_class"] = {
        label: summary["https_share"] for label, summary in sanity["classes"].items()
    }

    for model_name, features in report["top_features"].items():
        for feature in features:
            old_value = feature.pop("mean_value")
            if model_name == "logistic_regression":
                feature["importance_type"] = "absolute_standardized_logistic_coefficient"
                feature["mean_signed_coefficient"] = old_value
            elif model_name == "random_forest":
                feature["importance_type"] = "random_forest_impurity_importance"
                feature["mean_importance"] = old_value
            else:
                feature["importance_type"] = "weighted_absolute_base_model_importance"
                feature["mean_importance"] = old_value

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Preserved existing model metrics and ablations; refreshed diagnostics and feature labels.")
    print("Updated report:", REPORT_PATH)


if __name__ == "__main__":
    refresh_report_diagnostics()
