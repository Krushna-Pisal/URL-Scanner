"""Compare extraction, pre-extracted model inference, and full scoring.

Each stage processes the same deterministic URL sample one URL per call in a
single Python process. The artifact and precomputed feature rows are prepared
before timing. Each stage receives the same untimed warm-up count. Feature
extraction timing covers ``extract_features`` only; inference timing covers
one model ``predict_proba`` call over an already-built one-row feature frame;
end-to-end timing covers extraction, feature-frame construction, and model
prediction. Model loading, CSV reading, and sample selection are excluded.
The script does not send the sample URLs over a network.
"""

import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner.features import extract_features


DATA_PATH = ROOT / "data" / "raw" / "urls.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORT_PATH = ROOT / "reports" / "benchmark_stages.json"
SAMPLE_SIZE = 1_000
WARMUP_CALLS = 100
RANDOM_SEED = 42


def _score_one(url: str, artifact: dict[str, Any]) -> float:
    """Run the same feature extraction and one-row model inference as the API."""
    features = extract_features(url)
    names = artifact["feature_names"]
    row = pd.DataFrame([[features.get(name, 0.0) for name in names]], columns=names)
    return float(artifact["best_model"].predict_proba(row)[0, 1])


def _time_stage(function: Callable[[int], Any], sample_size: int) -> dict[str, float]:
    """Time sequential single-item calls after caller-controlled warm-up."""
    latencies = []
    start_total = time.perf_counter()
    for index in range(sample_size):
        started = time.perf_counter()
        function(index)
        latencies.append(time.perf_counter() - started)
    elapsed = time.perf_counter() - start_total
    return {
        "throughput_urls_per_second": float(sample_size / elapsed),
        "p50_latency_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
    }


def benchmark_stages() -> dict[str, Any]:
    """Measure each scoring stage with shared URLs and explicit timing edges."""
    data = pd.read_csv(DATA_PATH)
    if "url" not in data.columns:
        raise ValueError("data/raw/urls.csv must contain a url column")
    source_urls = data["url"].dropna().astype(str).tolist()
    if not source_urls:
        raise ValueError("No non-empty URLs are available to benchmark")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    sampler = random.Random(RANDOM_SEED)
    urls = [sampler.choice(source_urls) for _ in range(SAMPLE_SIZE)]
    artifact = joblib.load(MODEL_PATH)
    model = artifact["best_model"]
    n_jobs = {name: 1 for name in model.get_params(deep=True) if name.endswith("n_jobs")}
    if n_jobs:
        model.set_params(**n_jobs)
    names = artifact["feature_names"]

    # Prepare inference inputs before timing so the middle stage isolates model work.
    extracted = [extract_features(url) for url in urls]
    rows = [pd.DataFrame([[values.get(name, 0.0) for name in names]], columns=names)
            for values in extracted]
    warmup_calls = min(WARMUP_CALLS, len(urls))
    stages: dict[str, dict[str, float]] = {}
    stage_functions: dict[str, Callable[[int], Any]] = {
        "feature_extraction": lambda index: extract_features(urls[index]),
        "model_inference_pre_extracted": lambda index: model.predict_proba(rows[index]),
        "full_end_to_end": lambda index: _score_one(urls[index], artifact),
    }
    for name, function in stage_functions.items():
        for index in range(warmup_calls):
            function(index)
        stages[name] = _time_stage(function, len(urls))

    report: dict[str, Any] = {
        "measurement": "stage comparison: extraction only; predict_proba on pre-extracted one-row frames; and end-to-end extraction plus frame construction plus predict_proba",
        "execution_mode": "single process, sequential one-URL calls",
        "python_version": platform.python_version(),
        "sample_size": len(urls),
        "source_rows": len(source_urls),
        "random_seed": RANDOM_SEED,
        "warmup_calls_per_stage": warmup_calls,
        "warmup_timing": "excluded from each stage's throughput and latency",
        "model_loading": "excluded; loaded once before warm-up",
        "sample_selection_and_precomputation": "excluded from timing; same sampled URLs used for all stages",
        "estimator_n_jobs": 1,
        "stages": stages,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved stage benchmark report: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    benchmark_stages()
