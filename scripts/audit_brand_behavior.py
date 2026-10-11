"""Audit brand features and the saved model against dataset-provided labels.

This is a targeted diagnostic over a labeled dataset, not an independently
verified URL collection. Counts describe the supplied CSV only and do not
estimate population performance or statistical significance.
"""

import hashlib
import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.features import (
    BRAND_DOMAINS, extract_features, feature_extractor_fingerprint, _registered_domain,
)

DATA_PATH = ROOT / "data" / "raw" / "urls.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORT_PATH = ROOT / "reports" / "brand_behavior_audit.json"
OFFICIAL_DOMAINS = {domain for brand, domain in BRAND_DOMAINS.items() if brand != "gpay"}
MAX_EXAMPLES_PER_CATEGORY = 100


def _url_hash(url: str) -> str:
    """Identify audit examples without copying potentially unsafe URLs to reports."""
    return hashlib.sha256(url.encode("utf-8", errors="replace")).hexdigest()


def _component_category(url: str, label: int, registered: str) -> str | None:
    """Describe where a brand mention occurs without using it as a label."""
    try:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower()
    except ValueError:
        return None
    registered_label = registered.split(".", 1)[0]
    subdomain = hostname[: -(len(registered) + 1)] if hostname.endswith("." + registered) else ""
    searchable_brands = [name.replace(" ", "") for name in BRAND_DOMAINS if name != "gpay"]
    if registered in OFFICIAL_DOMAINS and label == 0:
        return "dataset_benign_official_bank_domain"
    if any(brand in registered_label.replace("-", "") for brand in searchable_brands):
        return "dataset_labeled_brand_like_registered_domain"
    if any(brand in subdomain for brand in searchable_brands):
        return "dataset_benign_brand_mention_in_subdomain" if label == 0 else "dataset_phishing_brand_mention_in_subdomain"
    if any(brand in parsed.path.lower() + parsed.query.lower() for brand in searchable_brands):
        return "dataset_benign_brand_mention_in_path_or_query" if label == 0 else "dataset_phishing_brand_mention_in_path_or_query"
    return None


def audit(*, model_path: Path = MODEL_PATH, report_path: Path = REPORT_PATH,
          feature_cache_path: Path | None = None) -> dict[str, Any]:
    """Compute feature error counts and model errors on targeted labeled cases."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    data = pd.read_csv(DATA_PATH).dropna(subset=["url", "label"]).drop_duplicates("url")
    data["url"] = data["url"].astype(str).str.strip()
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    artifact = joblib.load(model_path)
    feature_names = list(artifact["feature_names"])
    cached_features = None
    if feature_cache_path is not None:
        cache = joblib.load(feature_cache_path)
        if len(cache["features"]) != len(data):
            raise ValueError("Feature cache row count does not match the cleaned dataset")
        cached_features = cache["features"]
    rows: list[dict[str, Any]] = []
    candidates: dict[str, list[dict[str, Any]]] = {}
    feature_false_positives = []
    feature_false_negatives = []

    for row_number, item in enumerate(data.itertuples(index=False)):
        url, label = str(item.url), int(item.label)
        try:
            hostname = (urlsplit(url).hostname or "").lower()
        except ValueError:
            hostname = ""
        registered = _registered_domain(hostname) if hostname else ""
        values = (cached_features[row_number] if cached_features is not None
                  else extract_features(url))
        category = _component_category(url, label, registered)
        row = {"url": url, "label": label, "features": values, "category": category}
        if values["brand_impersonation_flag"] == 1 and label == 0:
            feature_false_positives.append(row)
        if label == 1 and values["brand_similarity"] >= 0.80 and values["brand_impersonation_flag"] == 0:
            feature_false_negatives.append(row)
        if category:
            candidates.setdefault(category, []).append(row)

    selected: list[dict[str, Any]] = []
    for category in sorted(candidates):
        category_rows = candidates[category]
        for label in (0, 1):
            selected.extend([
                row for row in category_rows if row["label"] == label
            ][:MAX_EXAMPLES_PER_CATEGORY])
    if selected:
        frames = pd.DataFrame([
            [row["features"].get(name, 0.0) for name in feature_names]
            for row in selected
        ], columns=feature_names)
        scores = artifact["best_model"].predict_proba(frames)[:, 1]
    else:
        scores = np.asarray([], dtype=float)

    per_category: dict[str, Any] = {}
    for row, score in zip(selected, scores):
        predicted = int(score >= 0.5)
        target = per_category.setdefault(row["category"], {
            "dataset_label_rows": {
                "0": sum(case["label"] == 0 for case in candidates[row["category"]]),
                "1": sum(case["label"] == 1 for case in candidates[row["category"]]),
            },
            "evaluated_rows_by_label": {"0": 0, "1": 0},
            "selected_rows": 0, "model_false_positives_at_0_5": 0,
            "model_false_negatives_at_0_5": 0, "examples": [],
        })
        target["selected_rows"] += 1
        target["evaluated_rows_by_label"][str(row["label"])] += 1
        target["model_false_positives_at_0_5"] += int(row["label"] == 0 and predicted == 1)
        target["model_false_negatives_at_0_5"] += int(row["label"] == 1 and predicted == 0)
        if len(target["examples"]) < 12:
            target["examples"].append({
                "url_sha256": _url_hash(row["url"]),
                "dataset_label": row["label"],
                "brand_similarity": row["features"]["brand_similarity"],
                "brand_impersonation_flag": row["features"]["brand_impersonation_flag"],
                "model_score": float(score),
                "model_prediction_at_0_5": predicted,
            })

    report = {
        "audit_type": "targeted_diagnostic_not_population_estimate",
        "label_source": "label column in data/raw/urls.csv; labels not independently verified",
        "model_artifact": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
        "feature_cache_used": str(feature_cache_path) if feature_cache_path else None,
        "source_rows": int(len(data)),
        "feature_schema_matches_extractor": feature_names == [
            name for name in extract_features("") if name in feature_names
        ],
        "artifact_feature_fingerprint_present": bool(artifact.get("feature_extractor_sha256")),
        "artifact_feature_fingerprint_matches_current": (
            artifact.get("feature_extractor_sha256") == feature_extractor_fingerprint()
            if artifact.get("feature_extractor_sha256") else None
        ),
        "model_name": artifact.get("best_model_name", "unknown"),
        "model_prediction_cutoff": 0.5,
        "score_calibration": "uncalibrated predict_proba score used for ranking",
        "selected_case_limit_per_category_per_label": MAX_EXAMPLES_PER_CATEGORY,
        "feature_flag_errors_over_dataset": {
            "false_positive_count": len(feature_false_positives),
            "false_negative_count": len(feature_false_negatives),
            "false_positive_examples": [
                {"url_sha256": _url_hash(row["url"]), "dataset_label": row["label"],
                 "brand_similarity": row["features"]["brand_similarity"]}
                for row in feature_false_positives[:20]
            ],
            "false_negative_examples": [
                {"url_sha256": _url_hash(row["url"]), "dataset_label": row["label"],
                 "brand_similarity": row["features"]["brand_similarity"]}
                for row in feature_false_negatives[:20]
            ],
        },
        "targeted_model_cases": per_category,
        "limitations": [
            "Dataset labels may be noisy and do not independently verify official ownership or malicious behavior.",
            "Case selection is rule-targeted and is not random; counts are not statistically significant estimates.",
            "URLs are represented by hashes so the report does not reproduce dataset URLs.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved brand behavior audit: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--feature-cache", type=Path)
    arguments = parser.parse_args()
    audit(model_path=arguments.model_path, report_path=arguments.report_path,
          feature_cache_path=arguments.feature_cache)
