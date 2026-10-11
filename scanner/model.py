"""Load the trained classifier and explain its lexical risk estimate."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from scanner.features import extract_features
from scanner.evidence import build_scan_evidence


LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.7
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"

FEATURE_EXPLANATIONS = {
    "url_length": "The URL is unusually long.",
    "hostname_length": "The hostname is unusually long.",
    "dot_count": "The hostname has many dot-separated parts.",
    "subdomain_count": "The hostname has several subdomains.",
    "hyphen_count": "The URL contains several hyphens.",
    "digit_ratio": "The URL contains a high share of digits.",
    "hostname_entropy": "The hostname has a varied, less predictable character pattern.",
    "has_at": "The URL contains an @ marker.",
    "path_has_double_slash": "The path contains a double slash.",
    "host_is_ipv4": "The hostname is a raw IPv4 address.",
    "uses_https": "The URL uses HTTPS.",
    "has_explicit_port": "The URL specifies a port.",
    "path_depth": "The URL path has several levels.",
    "query_length": "The URL has a long query string.",
    "query_param_count": "The URL has many query parameters.",
    "has_punycode": "The hostname contains punycode.",
    "suspicious_tld": "The hostname uses a configured suspicious top-level domain.",
    "suspicious_keyword_count": "The URL contains account or sign-in terms.",
    "is_url_shortener": "The hostname is a configured URL shortener.",
    "brand_similarity": "The hostname resembles a configured bank or payment brand.",
    "brand_impersonation_flag": "The registered domain resembles a bank or payment brand.",
    "apk_link": "The URL path ends with an APK file.",
    "upi_bait_keywords": "The URL contains UPI or payment-bait terms.",
}


@lru_cache(maxsize=1)
def _load_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the large artifact once per process.

    Replacing the artifact requires an application restart so predictions
    cannot mix model versions within a running process.
    """
    return joblib.load(path)


def _feature_frame(url: str, names: list[str]) -> pd.DataFrame:
    """Keep inference columns in exactly the training order."""
    features = extract_features(url)
    return pd.DataFrame([[features.get(name, 0.0) for name in names]], columns=names)


def predict(url: str) -> dict[str, Any]:
    """Return phishing probability, named severity level, and plain reasons."""
    artifact = _load_artifact()
    names = artifact["feature_names"]
    features = extract_features(url)
    frame = pd.DataFrame([[features.get(name, 0.0) for name in names]], columns=names)
    probability = float(artifact["best_model"].predict_proba(frame)[0, 1])
    if probability < LOW_THRESHOLD:
        level = LOW
    elif probability <= HIGH_THRESHOLD:
        level = MEDIUM
    else:
        level = HIGH
    best_name = str(artifact.get("best_model_name", "trained_classifier"))
    attribution: dict[str, Any] = {
        "method": "not_available_for_stacking_ensemble",
        "items": [],
    }
    # Only report a per-instance coefficient decomposition when the selected
    # scorer itself is a logistic pipeline with the expected scaler.
    if best_name.lower() in {"logistic", "logistic_regression", "logistic regression"}:
        estimator = artifact["best_model"]
        if hasattr(estimator, "named_steps") and {"scaler", "classifier"}.issubset(estimator.named_steps):
            scaler = estimator.named_steps["scaler"]
            classifier = estimator.named_steps["classifier"]
            if hasattr(classifier, "coef_"):
                scaled = scaler.transform(frame)[0]
                contributions = classifier.coef_[0] * scaled
                attribution = {
                    "method": "standardized_logistic_log_odds_contribution",
                    "items": [
                        {
                            "feature": names[index],
                            "observed_value": float(features.get(names[index], 0.0)),
                            "scaled_value": float(scaled[index]),
                            "coefficient": float(classifier.coef_[0][index]),
                            "contribution_log_odds": float(contributions[index]),
                            "direction": "raises" if contributions[index] > 0 else "lowers",
                        }
                        for index in sorted(range(len(names)), key=lambda i: abs(float(contributions[i])), reverse=True)[:5]
                        if contributions[index] != 0
                    ],
                }
    evidence = build_scan_evidence(
        url=url,
        score=probability,
        level=level,
        scoring_method="trained_classifier_predict_proba",
        model_name=best_name,
        model_features=names,
        feature_values=features,
        model_attribution=attribution,
        model_version=artifact.get("artifact_version"),
    )
    return {
        "probability": probability,
        "level": level,
        # These are transparent rule reasons and are distinct from model
        # attribution; a stacking score is not explained by another model.
        "reasons": evidence["heuristics"]["signals"],
        "features": features,
        "feature_names": names,
        "model_name": best_name,
        "model_version": artifact.get("artifact_version"),
        "model_attribution": attribution,
        "evidence": evidence,
    }


def top_contributing_features(
    url: str, limit: int = 5, artifact: dict[str, Any] | None = None
) -> list[str]:
    """Describe the separate logistic baseline's largest contributions.

    Contributions are coefficient multiplied by the feature value after the
    same StandardScaler used during training. This describes direction and
    relative contribution, not causation. This helper does not explain the
    selected stacking model and is therefore not used by :func:`predict`.
    """
    saved = artifact if artifact is not None else _load_artifact()
    estimator = saved["logistic_model"]
    names = saved["feature_names"]
    frame = _feature_frame(url, names)
    pipeline_scaler = estimator.named_steps["scaler"]
    logistic = estimator.named_steps["classifier"]
    scaled = pipeline_scaler.transform(frame)[0]
    contributions = logistic.coef_[0] * scaled
    ranked = sorted(enumerate(contributions), key=lambda item: abs(float(item[1])), reverse=True)
    explanations: list[str] = []
    for index, contribution in ranked:
        if len(explanations) >= limit or contribution == 0:
            break
        feature_name = names[index]
        direction = "raises" if contribution > 0 else "lowers"
        explanations.append(
            f"{FEATURE_EXPLANATIONS.get(feature_name, feature_name)} This {direction} the model score."
        )
    return explanations
