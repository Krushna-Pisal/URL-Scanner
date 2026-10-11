"""Evaluate URL models with stratified, registered-domain-grouped folds."""

import ipaddress
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.features import extract_features
from scanner.heuristics import score_features


DATA_PATH = ROOT / "data" / "raw" / "urls.csv"
REPORT_PATH = ROOT / "reports" / "metrics.json"
MODEL_PATH = ROOT / "models" / "model.joblib"
FOLD_COUNT = 5
RANDOM_STATE = 42
SCHEME_GAP_WARNING = 0.20
# A 50% gap in average URL length is large enough to make class origin an
# obvious shortcut for a lexical classifier; 5 percentage points is a
# meaningful difference for query prevalence at this dataset's scale.
LENGTH_RATIO_WARNING = 1.5
QUERY_SHARE_GAP_WARNING = 0.05
HTTPS_SHARE_GAP_WARNING = 0.20
FOLD_PHISH_SHARE_GAP_WARNING = 0.05
RECALL_FPR_TARGETS = (0.01, 0.05)
URL_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
OFFLINE_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())
METRIC_NAMES = (
    "recall", "false_positive_rate", "precision", "f1", "roc_auc",
    "recall_at_1pct_fpr", "recall_at_5pct_fpr",
)


def registered_domain(url: str) -> str:
    """Return the registrable domain using tldextract's bundled suffix list.

    Disabling suffix-list URLs keeps this calculation offline. Private suffixes
    use tldextract's defaults, so hosted sites under platforms such as
    weebly.com share the platform's registrable-domain group. Raw IPs remain
    their own groups because they do not have public suffixes.
    """
    try:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
    except (ValueError, AttributeError):
        return ""
    if not host:
        return ""
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        extracted = OFFLINE_EXTRACTOR(host)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        return extracted.domain or host


def _normalize_url_and_record_scheme(url: str) -> tuple[str, bool]:
    """Add HTTP before parsing scheme-less URLs and retain original status."""
    original_had_scheme = bool(URL_SCHEME_PATTERN.match(url.strip()))
    normalized = url.strip() if original_had_scheme else "http://" + url.strip()
    return normalized, original_had_scheme


def _url_shape(url: str) -> tuple[bool, bool]:
    """Return whether a normalized URL has a non-empty path and query."""
    try:
        parsed = urlsplit(url)
        return bool(parsed.path), bool(parsed.query)
    except ValueError:
        return False, False


def _print_dataset_sanity(data: pd.DataFrame) -> tuple[dict[str, Any], bool]:
    """Print class-wise URL structure checks before extracting features."""
    normalized_and_scheme = data["url"].map(_normalize_url_and_record_scheme)
    data["url"] = normalized_and_scheme.map(lambda item: item[0])
    data["original_had_scheme"] = normalized_and_scheme.map(lambda item: item[1])
    data["url_length_for_sanity"] = data["url"].str.len()
    shapes = data["url"].map(_url_shape)
    data["has_nonempty_path"] = shapes.map(lambda item: item[0])
    data["has_query"] = shapes.map(lambda item: item[1])

    summaries: dict[str, Any] = {}
    print("DATASET SANITY")
    print("Normalized URLs without an original scheme receive http:// before parsing.")
    for label in (0, 1):
        rows = data.loc[data["label"] == label]
        summary = {
            "row_count": int(len(rows)),
            "original_scheme_share": float(rows["original_had_scheme"].mean()),
            "mean_url_length": float(rows["url_length_for_sanity"].mean()),
            "nonempty_path_share": float(rows["has_nonempty_path"].mean()),
            "query_share": float(rows["has_query"].mean()),
            "https_share": float(rows["url"].str.lower().str.startswith("https://").mean()),
        }
        summaries[str(label)] = summary
        print(
            f"Class {label}: rows={summary['row_count']}, "
            f"original_scheme_share={summary['original_scheme_share']:.6f}, "
            f"mean_url_length={summary['mean_url_length']:.6f}, "
            f"nonempty_path_share={summary['nonempty_path_share']:.6f}, "
            f"query_share={summary['query_share']:.6f}, "
            f"https_share={summary['https_share']:.6f}"
        )

    scheme_gap = abs(summaries["0"]["original_scheme_share"] - summaries["1"]["original_scheme_share"])
    lengths = [summaries[str(label)]["mean_url_length"] for label in (0, 1)]
    length_ratio = max(lengths) / min(lengths) if min(lengths) else float("inf")
    query_share_gap = abs(summaries["0"]["query_share"] - summaries["1"]["query_share"])
    https_share_gap = abs(summaries["0"]["https_share"] - summaries["1"]["https_share"])
    warnings: list[str] = []
    if scheme_gap > SCHEME_GAP_WARNING:
        warnings.append(f"original scheme share differs by {scheme_gap:.6f}")
    if length_ratio > LENGTH_RATIO_WARNING:
        warnings.append(
            f"mean URL length ratio is {length_ratio:.6f} (warning threshold {LENGTH_RATIO_WARNING:.2f})"
        )
    if query_share_gap > QUERY_SHARE_GAP_WARNING:
        warnings.append(
            f"query-string share differs by {query_share_gap:.6f} "
            f"(warning threshold {QUERY_SHARE_GAP_WARNING:.2f})"
        )
    if https_share_gap > HTTPS_SHARE_GAP_WARNING:
        warnings.append(
            f"HTTPS share differs by {https_share_gap:.6f} "
            f"(warning threshold {HTTPS_SHARE_GAP_WARNING:.2f}); inspect its association with the label"
        )
    for warning in warnings:
        print("WARNING: Dataset sanity:", warning)
    remove_https = scheme_gap > SCHEME_GAP_WARNING
    if remove_https:
        print("Removed uses_https from ML features because original-scheme share differs sharply by class.")
    data.drop(columns=["url_length_for_sanity", "has_nonempty_path", "has_query"], inplace=True)
    return {
        "classes": summaries,
        "scheme_share_gap": float(scheme_gap),
        "mean_url_length_ratio": float(length_ratio),
        "query_share_gap": float(query_share_gap),
        "https_share_gap": float(https_share_gap),
        "diagnostic_thresholds": {
            "scheme_share_gap": SCHEME_GAP_WARNING,
            "mean_url_length_ratio": LENGTH_RATIO_WARNING,
            "query_share_gap": QUERY_SHARE_GAP_WARNING,
            "https_share_gap": HTTPS_SHARE_GAP_WARNING,
        },
        "warnings": warnings,
        "uses_https_removed_from_ml": remove_https,
    }, remove_https


def _make_logistic() -> Pipeline:
    """Scale numeric columns for logistic regression's coefficient model."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def _build_models() -> dict[str, Any]:
    """Build the existing classifier configurations with fixed seeds."""
    forest = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    stacking = StackingClassifier(
        estimators=[
            ("logistic", _make_logistic()),
            ("forest", RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            )),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        stack_method="predict_proba",
    )
    return {"logistic_regression": _make_logistic(), "random_forest": forest, "stacking": stacking}


def _best_heuristic_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Choose the threshold with best F1 using only this fold's train rows."""
    candidate_thresholds = np.unique(scores)
    candidate_f1 = [f1_score(labels, scores >= threshold, zero_division=0)
                    for threshold in candidate_thresholds]
    best_index = int(np.argmax(candidate_f1))
    return float(candidate_thresholds[best_index]), float(candidate_f1[best_index])


def _recall_at_false_positive_rate(labels: np.ndarray, scores: np.ndarray, limit: float) -> float:
    """Take the greatest ROC true-positive rate that stays under the FPR limit."""
    false_positive_rates, true_positive_rates, _ = roc_curve(labels, scores)
    allowed = true_positive_rates[false_positive_rates <= limit]
    return float(np.max(allowed)) if len(allowed) else 0.0


def _fold_metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Calculate classification and low-FPR metrics for one held-out fold."""
    negatives = int(np.sum(labels == 0))
    false_positives = int(np.sum((labels == 0) & (predictions == 1)))
    return {
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(false_positives / negatives) if negatives else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "recall_at_1pct_fpr": _recall_at_false_positive_rate(labels, scores, 0.01),
        "recall_at_5pct_fpr": _recall_at_false_positive_rate(labels, scores, 0.05),
    }


def _summarize_folds(fold_results: list[dict[str, float]]) -> dict[str, Any]:
    """Report every fold and population mean/std (ddof=0) for each metric."""
    return {
        "mean": {name: float(np.mean([row[name] for row in fold_results])) for name in METRIC_NAMES},
        "std": {name: float(np.std([row[name] for row in fold_results], ddof=0)) for name in METRIC_NAMES},
        "folds": fold_results,
        "std_ddof": 0,
    }


def _importance(model: Any, feature_names: list[str]) -> np.ndarray:
    """Return a feature-aligned coefficient or importance vector."""
    if isinstance(model, Pipeline):
        return np.asarray(model.named_steps["classifier"].coef_[0], dtype=float)
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    meta_weights = np.abs(model.final_estimator_.coef_[0])
    importance = np.zeros(len(feature_names), dtype=float)
    for weight, base in zip(meta_weights, model.estimators_):
        if isinstance(base, Pipeline):
            base_values = np.abs(base.named_steps["classifier"].coef_[0])
        else:
            base_values = base.feature_importances_
        importance += weight * np.abs(base_values)
    return importance


def _top_feature_report(
    feature_names: list[str], model_importances: dict[str, list[np.ndarray]]
) -> dict[str, list[dict[str, Any]]]:
    """Label signed coefficients separately from nonnegative importance scores."""
    result: dict[str, list[dict[str, Any]]] = {}
    for model_name, fold_vectors in model_importances.items():
        absolute_means = np.mean(np.abs(fold_vectors), axis=0)
        entries = []
        for index in np.argsort(absolute_means)[::-1][:10]:
            entry: dict[str, Any] = {
                "feature": feature_names[index],
                "mean_absolute_importance": float(absolute_means[index]),
            }
            if model_name == "logistic_regression":
                entry["importance_type"] = "absolute_standardized_logistic_coefficient"
                entry["mean_signed_coefficient"] = float(
                    np.mean([vector[index] for vector in fold_vectors])
                )
            elif model_name == "random_forest":
                entry["importance_type"] = "random_forest_impurity_importance"
                entry["mean_importance"] = float(
                    np.mean([vector[index] for vector in fold_vectors])
                )
            else:
                entry["importance_type"] = "weighted_absolute_base_model_importance"
                entry["mean_importance"] = float(
                    np.mean([vector[index] for vector in fold_vectors])
                )
            entries.append(entry)
        result[model_name] = entries
    return result


def _coefficient_sign(value: float) -> str:
    """Describe coefficient direction without rounding away its sign."""
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def train() -> dict[str, Any]:
    """Evaluate models by domain-stratified folds, ablate, and save a final model."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    if not {"url", "label"}.issubset(data.columns):
        raise ValueError("data/raw/urls.csv must contain url and label columns")
    data = data.drop_duplicates(subset="url").dropna(subset=["url", "label"]).copy()
    data["url"] = data["url"].astype(str).str.strip()
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("Training data must contain both labels, 0 and 1")

    sanity_report, remove_https = _print_dataset_sanity(data)
    feature_rows = [extract_features(url) for url in data["url"]]
    features = pd.DataFrame(feature_rows)
    if remove_https:
        features = features.drop(columns=["uses_https"])
    feature_names = list(features.columns)
    labels = data["label"].to_numpy()
    groups = data["url"].map(registered_domain).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=RANDOM_STATE)
    folds = list(splitter.split(features, labels, groups))

    overall_phishing_share = float(np.mean(labels == 1))
    group_frame = pd.DataFrame({"group": groups, "label": labels})
    group_counts = group_frame.groupby(["group", "label"]).size().unstack(fill_value=0)
    for label in (0, 1):
        if label not in group_counts.columns:
            group_counts[label] = 0
    group_counts["row_count"] = group_counts[0] + group_counts[1]
    largest = group_counts.sort_values("row_count", ascending=False).head(10)
    print("GROUPED CROSS-VALIDATION")
    print(f"Unique groups: {len(group_counts)}")
    print("10 largest groups (rows, label 0, label 1):")
    largest_groups = []
    for name, row in largest.iterrows():
        item = {"group": str(name), "row_count": int(row["row_count"]),
                "label_0": int(row[0]), "label_1": int(row[1])}
        largest_groups.append(item)
        print(f"  {item['group']}: rows={item['row_count']}, label_0={item['label_0']}, label_1={item['label_1']}")

    fold_shares = []
    fold_warnings = []
    for fold_number, (train_indices, test_indices) in enumerate(folds, start=1):
        train_labels, test_labels = labels[train_indices], labels[test_indices]
        test_share = float(np.mean(test_labels == 1))
        share_gap = abs(test_share - overall_phishing_share)
        fold_info = {
            "fold": fold_number,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_class_rows": {str(label): int(np.sum(train_labels == label)) for label in (0, 1)},
            "test_class_rows": {str(label): int(np.sum(test_labels == label)) for label in (0, 1)},
            "test_phishing_share": test_share,
            "overall_phishing_share": overall_phishing_share,
            "phishing_share_gap": float(share_gap),
        }
        fold_shares.append(fold_info)
        print(
            f"Fold {fold_number}: train_rows={fold_info['train_rows']}, test_rows={fold_info['test_rows']}, "
            f"train_class_rows={fold_info['train_class_rows']}, test_class_rows={fold_info['test_class_rows']}, "
            f"test_phishing_share={test_share:.6f}, overall_share={overall_phishing_share:.6f}"
        )
        if share_gap > FOLD_PHISH_SHARE_GAP_WARNING:
            warning = (f"fold {fold_number} phishing share differs from overall share by "
                       f"{share_gap:.6f} (threshold {FOLD_PHISH_SHARE_GAP_WARNING:.2f})")
            fold_warnings.append(warning)
            print("WARNING:", warning)

    https_shares = {
        str(label): float(features.loc[labels == label, "uses_https"].mean())
        if "uses_https" in features.columns else None
        for label in (0, 1)
    }
    print("HTTPS share by class:", json.dumps(https_shares, sort_keys=True))

    # These are heuristic probabilities, not trained ML predictions. The old
    # fixed 0.5 cutoff flagged nothing because the weighted rule scores for
    # this dataset stayed below 0.5; each fold now chooses its cutoff on train.
    # Reuse the extracted rows: score_url would parse and extract the same
    # 115k URLs a second time before cross-validation starts.
    heuristic_scores = np.array([score_features(row)[0] for row in feature_rows], dtype=float)
    default_train_flags = int(np.sum(heuristic_scores >= 0.5))
    print(f"Default heuristic threshold 0.5 flagged {default_train_flags} URLs in the full dataset.")

    model_names = list(_build_models())
    model_fold_results: dict[str, list[dict[str, float]]] = {name: [] for name in model_names}
    heuristic_fold_results: list[dict[str, float]] = []
    heuristic_thresholds = []
    model_importances: dict[str, list[np.ndarray]] = {name: [] for name in model_names}
    coefficient_signs: list[dict[str, str | int | None]] = []
    fold_models: dict[str, list[Any]] = {name: [] for name in model_names}

    print("FOLD METRICS")
    for fold_number, (train_indices, test_indices) in enumerate(folds, start=1):
        y_train, y_test = labels[train_indices], labels[test_indices]
        threshold, train_f1 = _best_heuristic_threshold(heuristic_scores[train_indices], y_train)
        heuristic_thresholds.append({"threshold": threshold, "training_f1": train_f1})
        heuristic_predictions = (heuristic_scores[test_indices] >= threshold).astype(int)
        baseline_metrics = _fold_metrics(y_test, heuristic_scores[test_indices], heuristic_predictions)
        heuristic_fold_results.append(baseline_metrics)
        print(f"fold={fold_number} heuristic threshold={threshold:.12g} train_f1={train_f1:.6f} "
              f"metrics={json.dumps(baseline_metrics, sort_keys=True)}")

        for name, model in _build_models().items():
            x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            metrics = _fold_metrics(y_test, probabilities, (probabilities >= 0.5).astype(int))
            model_fold_results[name].append(metrics)
            model_importances[name].append(_importance(model, feature_names))
            fold_models[name].append(model)
            print(f"fold={fold_number} {name} metrics={json.dumps(metrics, sort_keys=True)}")

            if name == "logistic_regression":
                coefficients = model.named_steps["classifier"].coef_[0]
                indices_by_name = {feature: index for index, feature in enumerate(feature_names)}
                signs = {
                    feature: _coefficient_sign(float(coefficients[indices_by_name[feature]]))
                    if feature in indices_by_name else None
                    for feature in ("uses_https", "is_url_shortener")
                }
                coefficient_signs.append({"fold": fold_number, **signs})
                print(f"fold={fold_number} logistic coefficient signs: {json.dumps(signs, sort_keys=True)}")

    summaries: dict[str, Any] = {
        "heuristic": _summarize_folds(heuristic_fold_results),
        **{name: _summarize_folds(rows) for name, rows in model_fold_results.items()},
    }
    print("MEAN +/- POPULATION STANDARD DEVIATION BY MODEL")
    for name, summary in summaries.items():
        print(f"{name}: " + json.dumps({
            metric: {"mean": summary["mean"][metric], "std": summary["std"][metric]}
            for metric in METRIC_NAMES
        }, sort_keys=True))

    # The heuristic remains a reported baseline; ablation requires a learned
    # estimator, so best-model selection compares the three trained models.
    best_name = max(model_names, key=lambda name: summaries[name]["mean"]["recall_at_1pct_fpr"])
    print("Best model by mean recall at 1% FPR:", best_name)

    ablation_groups = {
        "digit_ratio": ["digit_ratio"],
        "uses_https": ["uses_https"],
        "banking_fraud_features": [
            "brand_similarity", "brand_impersonation_flag", "apk_link", "upi_bait_keywords"
        ],
        "is_url_shortener": ["is_url_shortener"],
    }
    ablations: dict[str, Any] = {}
    print("ABLATIONS (BEST MODEL)")
    for ablation_name, removed_features in ablation_groups.items():
        retained_features = [feature for feature in feature_names if feature not in removed_features]
        fold_results = []
        for train_indices, test_indices in folds:
            model = _build_models()[best_name]
            model.fit(features.iloc[train_indices][retained_features], labels[train_indices])
            probabilities = model.predict_proba(features.iloc[test_indices][retained_features])[:, 1]
            fold_results.append(_fold_metrics(
                labels[test_indices], probabilities, (probabilities >= 0.5).astype(int)
            ))
        ablations[ablation_name] = {
            "removed_features": removed_features,
            "metrics": _summarize_folds(fold_results),
        }
        print(f"{ablation_name} removed={removed_features}: " + json.dumps({
            metric: {"mean": ablations[ablation_name]["metrics"]["mean"][metric],
                     "std": ablations[ablation_name]["metrics"]["std"][metric]}
            for metric in METRIC_NAMES
        }, sort_keys=True))

    logistic_sign_summary = {
        feature: [item[feature] for item in coefficient_signs]
        for feature in ("uses_https", "is_url_shortener")
    }
    report: dict[str, Any] = {
        "evaluation": {
            "splitter": "StratifiedGroupKFold",
            "fold_count": FOLD_COUNT,
            "random_state": RANDOM_STATE,
            "metric_standard_deviation_ddof": 0,
            "best_model_selection_metric": "mean_recall_at_1pct_fpr",
            "overall_phishing_share": overall_phishing_share,
            "folds": fold_shares,
            "warnings": fold_warnings,
        },
        "sanity": sanity_report,
        "group_diagnostics": {
            "unique_groups": int(len(group_counts)),
            "largest_groups": largest_groups,
        },
        "https_share_by_class": https_shares,
        "logistic_coefficient_signs_by_fold": coefficient_signs,
        "logistic_coefficient_signs": logistic_sign_summary,
        "heuristic_thresholds_by_fold": heuristic_thresholds,
        "models": summaries,
        "best_model": best_name,
        "ablation": ablations,
        "top_features": _top_feature_report(feature_names, model_importances),
    }

    high_metrics = []
    for model_name, summary in summaries.items():
        for metric_name, value in summary["mean"].items():
            if value > 0.99:
                high_metrics.append(f"{model_name}.{metric_name} mean={value:.6f}")
    if high_metrics:
        warning = "Check for dataset bias or leakage: " + ", ".join(high_metrics)
        report["warning"] = warning
        print("WARNING:", warning)

    # Refit the selected estimator and explanation model on all available rows.
    final_models = _build_models()
    best_model = final_models[best_name].fit(features, labels)
    logistic_model = final_models["logistic_regression"].fit(features, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "best_model": best_model,
        "best_model_name": best_name,
        "logistic_model": logistic_model,
        "feature_names": feature_names,
    }, MODEL_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FULL METRICS REPORT")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved model: {MODEL_PATH}")
    return report


if __name__ == "__main__":
    train()
