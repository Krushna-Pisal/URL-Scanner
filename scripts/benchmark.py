"""Benchmark in-process, single-process URL inference one URL at a time.

Each timed call includes lexical feature extraction and one trained-model
``predict_proba`` call. The model artifact is loaded once before warm-up, and
warm-up calls are excluded from latency and throughput timing. The report
records Python version, sample size, and warm-up count so the measurement can
be interpreted. It does not measure HTTP, database, process startup, artifact
loading, batch prediction, or network access.
"""

import json
import random
import sys
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.features import extract_features


DATA_PATH = ROOT / "data" / "raw" / "urls.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORT_PATH = ROOT / "reports" / "benchmark.json"
SAMPLE_SIZE = 1_000
WARMUP_LIMIT = 100
RANDOM_SEED = 42


def _predict_one(url: str, artifact: dict) -> float:
    """Extract one URL's features and return its phishing probability."""
    features = extract_features(url)
    names = artifact["feature_names"]
    row = pd.DataFrame([[features.get(name, 0.0) for name in names]], columns=names)
    return float(artifact["best_model"].predict_proba(row)[0, 1])


def benchmark() -> dict[str, float | int | str]:
    """Time the configured sequential URL sample after untimed model warm-up."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Benchmark data not found: {DATA_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}; run scripts/train.py first")
    data = pd.read_csv(DATA_PATH)
    if "url" not in data.columns:
        raise ValueError("data/raw/urls.csv must contain a url column")
    urls = data["url"].dropna().astype(str).tolist()
    if not urls:
        raise ValueError("No non-empty URLs are available to benchmark")

    randomizer = random.Random(RANDOM_SEED)
    sample = [randomizer.choice(urls) for _ in range(SAMPLE_SIZE)]
    artifact = joblib.load(MODEL_PATH)
    # Keep inference within this process; n_jobs changes execution only, not
    # the fitted trees or the per-URL probability calculation.
    prediction_model = artifact["best_model"]
    single_process_params = {
        name: 1 for name in prediction_model.get_params(deep=True)
        if name.endswith("n_jobs")
    }
    if single_process_params:
        prediction_model.set_params(**single_process_params)
    warmup_calls = min(WARMUP_LIMIT, len(sample))
    for url in sample[:warmup_calls]:
        _predict_one(url, artifact)

    timed_urls = sample
    latencies: list[float] = []
    start_all = time.perf_counter()
    for url in timed_urls:
        started = time.perf_counter()
        _predict_one(url, artifact)
        latencies.append(time.perf_counter() - started)
    elapsed = time.perf_counter() - start_all
    result: dict[str, float | int | str] = {
        "measurement": "in-process URL scoring: feature extraction plus trained model predict_proba, one URL per call",
        "execution_mode": "single process, sequential calls",
        "estimator_n_jobs": 1,
        "python_version": platform.python_version(),
        "sample_size": len(timed_urls),
        "source_rows": len(urls),
        "warmup_calls": warmup_calls,
        "warmup_timing": "excluded from throughput and latency",
        "model_load_timing": "excluded; artifact loaded once before warm-up",
        "throughput_urls_per_second": float(len(timed_urls) / elapsed),
        "p50_latency_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved benchmark report: {REPORT_PATH}")
    return result


if __name__ == "__main__":
    benchmark()
