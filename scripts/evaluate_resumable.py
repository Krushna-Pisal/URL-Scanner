"""Run the existing grouped evaluation with isolated, resumable checkpoints.

The full mode keeps the five-fold StratifiedGroupKFold evaluation, existing
models, thresholds, metrics, and four ablations from scripts.train. Checkpoints
are stored under reports/evaluation_runs and never replace the historical
metrics report or the model used by the API. Use --resume after interruption.
The --smoke mode samples rows and is only a pipeline correctness check; its
metrics are not a full evaluation.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train as pipeline
from scanner.features import extract_features, feature_extractor_fingerprint
from scanner.heuristics import score_features

DATA_PATH = ROOT / "data" / "raw" / "urls.csv"
DEFAULT_RUNS = ROOT / "reports" / "evaluation_runs"
SMOKE_ROWS = 3000


def _atomic_json(path: Path, value: Any) -> None:
    """Write a complete JSON checkpoint before replacing its prior copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_checkpoint(path: Path, resume: bool, calculate: Any) -> Any:
    """Reuse a finished stage on resume, otherwise calculate and save it."""
    if resume and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    result = calculate()
    _atomic_json(path, result)
    return result


def _digest_file(path: Path) -> str:
    """Hash the source CSV so checkpoints cannot cross dataset versions."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_rows(urls: list[str], run_dir: Path, resume: bool) -> tuple[pd.DataFrame, float, float]:
    """Load or create the feature matrix without repeating completed parsing."""
    cache_path = run_dir / "features.joblib"
    if resume and cache_path.is_file():
        load_started = time.perf_counter()
        cached = joblib.load(cache_path)
        load_seconds = time.perf_counter() - load_started
        return pd.DataFrame(cached["features"]), float(cached["seconds"]), load_seconds
    started = time.perf_counter()
    rows = [extract_features(url) for url in urls]
    frame = pd.DataFrame(rows)
    seconds = time.perf_counter() - started
    temporary = cache_path.with_suffix(".joblib.tmp")
    joblib.dump({"features": frame.to_dict(orient="records"), "seconds": seconds}, temporary)
    temporary.replace(cache_path)
    return frame, seconds, 0.0


def _folds(labels: np.ndarray, groups: np.ndarray, run_dir: Path, resume: bool) -> tuple[list[tuple[np.ndarray, np.ndarray]], float, float]:
    """Load or create fixed-seed grouped fold indices."""
    path = run_dir / "fold_indices.json"
    if resume and path.is_file():
        load_started = time.perf_counter()
        value = json.loads(path.read_text(encoding="utf-8"))
        folds = [(np.asarray(row["train"], dtype=int), np.asarray(row["test"], dtype=int))
                 for row in value["folds"]]
        load_seconds = time.perf_counter() - load_started
        return folds, float(value["seconds"]), load_seconds
    started = time.perf_counter()
    splitter = StratifiedGroupKFold(
        n_splits=pipeline.FOLD_COUNT, shuffle=True, random_state=pipeline.RANDOM_STATE
    )
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    seconds = time.perf_counter() - started
    _atomic_json(path, {"seconds": seconds, "folds": [
        {"train": tr.tolist(), "test": te.tolist()} for tr, te in folds
    ]})
    return folds, seconds, 0.0


def _heuristic_vector(feature_frame: pd.DataFrame, run_dir: Path,
                      resume: bool) -> tuple[np.ndarray, float, float]:
    """Cache per-row heuristic scores because resume uses them for every fold."""
    path = run_dir / "heuristic_scores.joblib"
    if resume and path.is_file():
        started = time.perf_counter()
        cached = joblib.load(path)
        return np.asarray(cached["scores"], dtype=float), float(cached["calculation_seconds"]), time.perf_counter() - started
    started = time.perf_counter()
    scores = np.asarray([
        score_features(row)[0] for row in feature_frame.to_dict(orient="records")
    ], dtype=float)
    seconds = time.perf_counter() - started
    temporary = path.with_suffix(".joblib.tmp")
    joblib.dump({"scores": scores.tolist(), "calculation_seconds": seconds}, temporary)
    temporary.replace(path)
    return scores, seconds, 0.0


def _evaluate_model(model_name: str, features: pd.DataFrame, labels: np.ndarray,
                    train_indices: np.ndarray, test_indices: np.ndarray) -> dict[str, Any]:
    """Fit one configured model for one outer fold and preserve its evidence."""
    model = pipeline._build_models()[model_name]
    started = time.perf_counter()
    model.fit(features.iloc[train_indices], labels[train_indices])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = model.predict_proba(features.iloc[test_indices])[:, 1]
    predict_seconds = time.perf_counter() - started
    metrics = pipeline._fold_metrics(
        labels[test_indices], scores, (scores >= 0.5).astype(int)
    )
    signs: dict[str, str] | None = None
    if model_name == "logistic_regression":
        coefficients = model.named_steps["classifier"].coef_[0]
        signs = {
            feature: pipeline._coefficient_sign(float(coefficients[features.columns.get_loc(feature)]))
            if feature in features.columns else "not_in_feature_set"
            for feature in ("uses_https", "is_url_shortener")
        }
    return {
        "metrics": metrics,
        "importance": pipeline._importance(model, list(features.columns)).tolist(),
        "coefficient_signs": signs,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def _evaluate_ablation(model_name: str, retained: list[str], features: pd.DataFrame,
                       labels: np.ndarray, train_indices: np.ndarray,
                       test_indices: np.ndarray) -> dict[str, Any]:
    """Fit one ablation fold using the unchanged model configuration."""
    model = pipeline._build_models()[model_name]
    started = time.perf_counter()
    model.fit(features.iloc[train_indices][retained], labels[train_indices])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = model.predict_proba(features.iloc[test_indices][retained])[:, 1]
    predict_seconds = time.perf_counter() - started
    return {
        "metrics": pipeline._fold_metrics(
            labels[test_indices], scores, (scores >= 0.5).astype(int)
        ),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def evaluate(*, smoke: bool = False, resume: bool = False, run_dir: Path | None = None) -> dict[str, Any]:
    """Run complete or smoke grouped evaluation in a separate run directory."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")
    source_digest = _digest_file(DATA_PATH)
    if run_dir is None:
        run_dir = DEFAULT_RUNS / (("smoke-" if smoke else "full-") + source_digest[:16])
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    configuration = {
        "status": "smoke_only" if smoke else "full_grouped_evaluation",
        "source_csv_sha256": source_digest,
        "fold_count": pipeline.FOLD_COUNT,
        "splitter": "StratifiedGroupKFold",
        "random_state": pipeline.RANDOM_STATE,
        "metric_threshold": 0.5,
        "model_configuration": {
            "logistic_regression": {"max_iter": 2000, "class_weight": "balanced", "random_state": 42},
            "random_forest": {"n_estimators": 300, "class_weight": "balanced", "random_state": 42, "n_jobs": -1},
            "stacking": {
                "base_logistic_max_iter": 2000, "base_forest_estimators": 200,
                "final_logistic_max_iter": 2000, "class_weight": "balanced",
                "random_state": 42, "internal_cv": 5, "n_jobs": -1,
            },
        },
        "metrics": list(pipeline.METRIC_NAMES),
        "low_fpr_targets": list(pipeline.RECALL_FPR_TARGETS),
        "python_version": platform.python_version(),
        "ablations": ["digit_ratio", "uses_https", "banking_fraud_features", "is_url_shortener"],
        "smoke_row_limit": SMOKE_ROWS if smoke else None,
        "feature_extractor": "scanner.features.extract_features",
        "feature_extractor_sha256": feature_extractor_fingerprint(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != configuration:
            raise ValueError("Run directory configuration differs; choose another run directory")
        if not resume:
            raise FileExistsError("Run directory already exists; pass --resume to continue it")
    elif resume:
        raise FileNotFoundError("Cannot resume: this run directory has no manifest")
    else:
        _atomic_json(manifest_path, configuration)

    timings: dict[str, float] = {}
    started = time.perf_counter()
    data = pd.read_csv(DATA_PATH)
    timings["csv_read_seconds"] = time.perf_counter() - started
    if not {"url", "label"}.issubset(data.columns):
        raise ValueError("data/raw/urls.csv must contain url and label columns")
    data = data.drop_duplicates(subset="url").dropna(subset=["url", "label"]).copy()
    data["url"] = data["url"].astype(str).str.strip()
    data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("Training data must contain both labels, 0 and 1")
    if smoke and len(data) > SMOKE_ROWS:
        data = data.sample(n=SMOKE_ROWS, random_state=pipeline.RANDOM_STATE).sort_index().copy()

    sanity, remove_https = pipeline._print_dataset_sanity(data)
    feature_frame, timings["feature_extraction_seconds"], timings["feature_cache_load_seconds"] = _artifact_rows(
        data["url"].tolist(), run_dir, resume
    )
    if remove_https:
        feature_frame = feature_frame.drop(columns=["uses_https"])
    labels = data["label"].to_numpy()
    started = time.perf_counter()
    groups = data["url"].map(pipeline.registered_domain).to_numpy()
    timings["domain_grouping_seconds"] = time.perf_counter() - started
    folds, timings["grouped_fold_generation_seconds"], timings["fold_cache_load_seconds"] = _folds(
        labels, groups, run_dir, resume
    )

    model_names = list(pipeline._build_models())
    fold_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in model_names}
    fold_importances: dict[str, list[np.ndarray]] = {name: [] for name in model_names}
    coefficient_signs = []
    baseline_rows = []
    heuristic_thresholds = []
    heuristic_scores, timings["heuristic_scoring_seconds"], timings["heuristic_score_cache_load_seconds"] = _heuristic_vector(
        feature_frame, run_dir, resume
    )
    for fold_number, (train_indices, test_indices) in enumerate(folds, 1):
        fold_path = run_dir / "folds" / f"fold_{fold_number}"
        threshold, train_f1 = pipeline._best_heuristic_threshold(
            heuristic_scores[train_indices], labels[train_indices]
        )
        baseline_path = fold_path / "heuristic.json"
        baseline = _json_checkpoint(baseline_path, resume, lambda: {
            "threshold": threshold,
            "training_f1": train_f1,
            "metrics": pipeline._fold_metrics(
                labels[test_indices], heuristic_scores[test_indices],
                (heuristic_scores[test_indices] >= threshold).astype(int),
            ),
        })
        baseline_rows.append(baseline["metrics"])
        heuristic_thresholds.append({
            "fold": fold_number,
            "threshold": float(baseline["threshold"]),
            "training_f1": float(baseline["training_f1"]),
        })
        for name in model_names:
            path = fold_path / f"{name}.json"
            result = _json_checkpoint(path, resume, lambda n=name: _evaluate_model(
                n, feature_frame, labels, train_indices, test_indices
            ))
            fold_metrics[name].append(result["metrics"])
            fold_importances[name].append(np.asarray(result["importance"], dtype=float))
            if name == "logistic_regression":
                coefficient_signs.append({"fold": fold_number, **result["coefficient_signs"]})
            print(f"fold={fold_number} model={name} fit_seconds={result['fit_seconds']:.3f} "
                  f"predict_seconds={result['predict_seconds']:.3f} metrics="
                  f"{json.dumps(result['metrics'], sort_keys=True)}", flush=True)
        print(f"fold={fold_number} heuristic threshold={baseline['threshold']:.8g} "
              f"metrics={json.dumps(baseline['metrics'], sort_keys=True)}", flush=True)

    summaries = {"heuristic": pipeline._summarize_folds(baseline_rows)}
    summaries.update({name: pipeline._summarize_folds(rows) for name, rows in fold_metrics.items()})
    best_name = max(model_names, key=lambda name: summaries[name]["mean"]["recall_at_1pct_fpr"])
    overall_phishing_share = float(np.mean(labels == 1))
    fold_diagnostics = []
    for fold_number, (train_indices, test_indices) in enumerate(folds, 1):
        train_labels, test_labels = labels[train_indices], labels[test_indices]
        test_share = float(np.mean(test_labels == 1))
        fold_diagnostics.append({
            "fold": fold_number,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_class_rows": {str(label): int(np.sum(train_labels == label)) for label in (0, 1)},
            "test_class_rows": {str(label): int(np.sum(test_labels == label)) for label in (0, 1)},
            "test_phishing_share": test_share,
            "overall_phishing_share": overall_phishing_share,
            "phishing_share_gap": abs(test_share - overall_phishing_share),
        })
    fold_warnings = [
        f"fold {row['fold']} phishing share differs by {row['phishing_share_gap']:.6f}"
        for row in fold_diagnostics if row["phishing_share_gap"] > pipeline.FOLD_PHISH_SHARE_GAP_WARNING
    ]
    ablation_groups = {
        "digit_ratio": ["digit_ratio"],
        "uses_https": ["uses_https"],
        "banking_fraud_features": ["brand_similarity", "brand_impersonation_flag", "apk_link", "upi_bait_keywords"],
        "is_url_shortener": ["is_url_shortener"],
    }
    ablations = {}
    for ablation_name, removed in ablation_groups.items():
        retained = [name for name in feature_frame.columns if name not in removed]
        results = []
        for fold_number, (train_indices, test_indices) in enumerate(folds, 1):
            path = run_dir / "ablations" / ablation_name / f"fold_{fold_number}.json"
            result = _json_checkpoint(path, resume, lambda: _evaluate_ablation(
                best_name, retained, feature_frame, labels, train_indices, test_indices
            ))
            results.append(result["metrics"])
            print(f"ablation={ablation_name} fold={fold_number} "
                  f"fit_seconds={result['fit_seconds']:.3f}", flush=True)
        ablations[ablation_name] = {
            "removed_features": removed,
            "metrics": pipeline._summarize_folds(results),
        }

    timings["evaluation_elapsed_seconds"] = float(sum(
        value["fit_seconds"] + value["predict_seconds"]
        for name in model_names for fold in range(1, len(folds) + 1)
        for value in [json.loads((run_dir / "folds" / f"fold_{fold}" / f"{name}.json").read_text(encoding="utf-8"))]
    ))
    timings["ablation_elapsed_seconds"] = float(sum(
        json.loads(path.read_text(encoding="utf-8"))["fit_seconds"]
        + json.loads(path.read_text(encoding="utf-8"))["predict_seconds"]
        for path in (run_dir / "ablations").glob("*/*.json")
    ))
    report = {
        "evaluation_status": "smoke_test_not_full_evaluation" if smoke else "complete_five_fold_grouped_evaluation",
        "configuration": configuration,
        "evaluation": {
            "splitter": "StratifiedGroupKFold",
            "fold_count": pipeline.FOLD_COUNT,
            "random_state": pipeline.RANDOM_STATE,
            "metric_standard_deviation_ddof": 0,
            "best_model_selection_metric": "mean_recall_at_1pct_fpr",
            "overall_phishing_share": overall_phishing_share,
            "folds": fold_diagnostics,
            "warnings": fold_warnings,
        },
        "row_count": int(len(data)),
        "class_rows": {str(label): int(np.sum(labels == label)) for label in (0, 1)},
        "overall_phishing_share": overall_phishing_share,
        "unique_groups": int(len(set(groups))),
        "fold_diagnostics": fold_diagnostics,
        "sanity": sanity,
        "uses_https_removed_from_ml": remove_https,
        "heuristic_thresholds_by_fold": heuristic_thresholds,
        "timings_seconds": timings,
        "models": summaries,
        "best_model": best_name,
        "ablation": ablations,
        "top_features": pipeline._top_feature_report(list(feature_frame.columns), fold_importances),
        "logistic_coefficient_signs_by_fold": coefficient_signs,
    }
    high_metrics = [
        f"{model_name}.{metric_name} mean={value:.6f}"
        for model_name, summary in summaries.items()
        for metric_name, value in summary["mean"].items()
        if value > 0.99
    ]
    if high_metrics:
        report["warning"] = "Check for dataset bias or leakage: " + ", ".join(high_metrics)
    report_path = run_dir / ("smoke_metrics.json" if smoke else "fresh_metrics.json")
    if not smoke:
        # Keep this candidate isolated; the active API artifact and old report
        # remain untouched until a human explicitly chooses to deploy it.
        candidate = run_dir / "candidate_model.joblib"
        candidate_metadata_path = run_dir / "candidate_metadata.json"
        if resume and candidate.is_file() and candidate_metadata_path.is_file():
            candidate_metadata = json.loads(candidate_metadata_path.read_text(encoding="utf-8"))
            if candidate_metadata.get("feature_extractor_sha256") != feature_extractor_fingerprint():
                raise ValueError("Candidate artifact feature fingerprint does not match current code")
        else:
            candidate_started = time.perf_counter()
            final_model = pipeline._build_models()[best_name].fit(feature_frame, labels)
            logistic_model = pipeline._make_logistic().fit(feature_frame, labels)
            candidate_fit_seconds = time.perf_counter() - candidate_started
            temporary = candidate.with_suffix(".joblib.tmp")
            joblib.dump({
                "best_model": final_model,
                "best_model_name": best_name,
                "logistic_model": logistic_model,
                "feature_names": list(feature_frame.columns),
                "feature_extractor_sha256": feature_extractor_fingerprint(),
            }, temporary)
            temporary.replace(candidate)
            candidate_metadata = {
                "candidate_fit_seconds": candidate_fit_seconds,
                "feature_extractor_sha256": feature_extractor_fingerprint(),
                "candidate_artifact": candidate.name,
            }
            _atomic_json(candidate_metadata_path, candidate_metadata)
        timings["candidate_model_fit_seconds"] = float(candidate_metadata["candidate_fit_seconds"])
        report["timings_seconds"] = timings
        report["candidate_artifact"] = {
            "path": candidate.name,
            "feature_extractor_sha256": candidate_metadata["feature_extractor_sha256"],
        }
    _atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {report['evaluation_status']} report: {report_path}", flush=True)
    return report


def main() -> None:
    """Expose isolated smoke and resumable full modes on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="sample rows for a correctness check only")
    parser.add_argument("--resume", action="store_true", help="reuse matching completed checkpoints")
    parser.add_argument("--run-dir", type=Path, help="checkpoint directory; defaults under reports/evaluation_runs")
    args = parser.parse_args()
    evaluate(smoke=args.smoke, resume=args.resume, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
