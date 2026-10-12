# URL Threat Scanner

URL Threat Scanner is a explainable phishing-risk prototype. It extracts lexical features from a URL string, combines a transparent rule score with optional scikit-learn classifiers, and exposes scan results through a FastAPI service backed by SQLite. It never opens the supplied URL or makes DNS, WHOIS, or other network lookups.

## Architecture

```text
Client
  |
  | POST /scan {"url": "..."}
  v
FastAPI validation and optional API-key check
  |
  v
Lexical feature extraction (string only)
  |
  +--> Trained model when models/model.joblib exists
  |       or transparent heuristic fallback
  v
SQLite scan history --> response with score, level, reasons, model, time

Training CSV --> scripts/train.py --> reports/metrics.json
                                 \--> models/model.joblib
Training CSV --> scripts/benchmark.py --> reports/benchmark.json
```

## Where a URL scorer fits in a secure web gateway

A URL scorer runs at the gateway decision point before the gateway allows or blocks a request. It can provide a risk signal to the gateway's policy alongside other controls. This project keeps analysis string-only: fetching an untrusted URL could create server-side request forgery (SSRF) paths, while avoiding fetches also makes scoring faster and deterministic for the same input and model. A lexical score alone should not be treated as a complete gateway policy.

## Motivation

Includes banking-fraud signals motivated by a Bank of India hackathon proposal I co-designed (not shortlisted).

## Dataset

Supply a CSV at `data/raw/urls.csv` with columns `url,label`, where `1` means phishing and `0` means legitimate. The raw data directory is ignored by Git; do not commit source datasets or derived records unless their licenses explicitly allow redistribution.

The evaluation summarized below uses URL-Phish from Mendeley Data, DOI [10.17632/65z9twcx3r](https://doi.org/10.17632/65z9twcx3r), Version 2: 100,000 benign URLs from trusted institutional domains and 16,600 phishing URLs from PhishTank (November 2024 to September 2025). Obtain the dataset from its Mendeley Data record, check its current license and terms, and keep the downloaded source file out of Git. To prepare the scanner's input, place the downloaded CSV at `data/download/Dataset.csv` and run `python scripts/prepare_dataset.py`; the script emits `data/raw/urls.csv`.

Possible phishing or malware URL sources include [PhishTank's verified online phishing feed](https://phishtank.org/developer_info.php) and [URLhaus data exports](https://urlhaus.abuse.ch/api/). PhishTank's bulk feed may require an application key for regular downloads; review its current developer terms and attribution requirements. URLhaus requires an Auth-Key for downloads, describes fair-use community access, and directs commercial use to its enhanced service and terms. Treat the sources as candidates for a research dataset, not interchangeable labels: URLhaus focuses on malware distribution and is not a general phishing-only source.

For legitimate examples, use a documented, permitted source such as a dated [Tranco ranking](https://tranco-list.eu/) as a candidate list, or a hand-curated list of official domains. Popularity does not prove that a URL is safe, and a domain list is not equivalent to a verified legitimate URL label. Before downloading, training on, or redistributing any source, read the source's current license, terms, attribution rules, and rate limits. Record source, retrieval date, label definition, and any filtering in your own dataset notes. Keep the raw data private when redistribution rights are unclear.

After preparing the CSV, train and benchmark locally:

```powershell
python -m scripts.train
python -m scripts.benchmark
```

`scripts.train` writes a single grouped train/test evaluation to `reports/metrics.json` and saves an estimator to `models/model.joblib`. For a reproducible five-fold grouped evaluation with resumable fold and ablation checkpoints, run `python scripts/evaluate_resumable.py`; use `--smoke` for a small correctness run and `--resume` to continue an existing full run. Full-run outputs are written to an isolated `reports/evaluation_runs/` directory so historical reports and the active model are not overwritten. The fresh results below were computed by that script. The benchmark measures in-process scoring of one URL per call, including feature extraction and model prediction; artifact loading and warm-up calls are excluded. Its sample size and timing setup are recorded in `reports/benchmark.json`.

## Docker

Build and run the API from the repository root:

```powershell
docker build -t url-threat-scanner .
docker run --rm -p 8000:8000 -e URL_SCANNER_DB_PATH=/app/data/scans.sqlite3 url-threat-scanner
```

Without a trained artifact the API uses the rule-based heuristic and reports `"model": "heuristic"`. To serve the trained classifier, mount the locally generated `models` directory read-only at `/app/models`. Set `URL_SCANNER_API_KEY` to require an `X-API-Key` header; if it is unset, API-key authentication is disabled.

## API example

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/scan `
  -ContentType "application/json" -Body '{"url":"https://example.com/login"}'
```

The response shape is shown with placeholders because its values are computed at request time:

```json
{
  "url": "https://example.com/login",
  "score": "<computed uncalibrated score from 0 to 1>",
  "level": "<LOW, MEDIUM, or HIGH>",
  "reasons": ["<plain-English feature reason, if any>"],
  "model": "<heuristic or trained_model>",
  "scanned_at": "<UTC timestamp>"
}
```

The score is a model output and is **not a calibrated probability**. The classifier's default decision cutoff is 0.5. The UI's LOW/MEDIUM/HIGH bands use 0.3 and 0.7 for presentation, so those labels do not change the classifier cutoff.

`GET /history?limit=20` returns recent stored scans. `GET /health` is a public health check. Scan URLs are never fetched.

## Results

These fresh results are the mean ± standard deviation across five `StratifiedGroupKFold` test folds, grouped by registered domain, from `reports/evaluation_runs/full-7dda65c11dc56804/fresh_metrics.json`. Precision, recall, F1, ROC-AUC, and false-positive rate use the report's 0.5 classification cutoff; the heuristic threshold is selected on each training fold only. Values are dataset-specific and not a claim of real-world accuracy.

| Model | Precision | Recall | F1 | ROC-AUC | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic baseline | 0.691 ± 0.050 | 0.624 ± 0.125 | 0.653 ± 0.094 | 0.797 ± 0.064 | 0.046 ± 0.005 |
| Logistic regression | 0.543 ± 0.006 | 0.845 ± 0.091 | 0.660 ± 0.035 | 0.951 ± 0.014 | 0.119 ± 0.010 |
| Random forest | 0.884 ± 0.017 | 0.687 ± 0.132 | 0.767 ± 0.094 | 0.959 ± 0.029 | 0.015 ± 0.002 |
| Stacking | 0.776 ± 0.020 | 0.809 ± 0.120 | 0.789 ± 0.072 | 0.968 ± 0.015 | 0.039 ± 0.002 |

The predeclared selection rule chooses stacking by mean recall at 1% FPR. Its reported recall at 1% FPR is 0.636 ± 0.121. Removing `digit_ratio`, `uses_https`, all four banking-fraud features, or `is_url_shortener` changed stacking mean recall at 1% FPR to 0.506 ± 0.139, 0.510 ± 0.124, 0.649 ± 0.104, and 0.609 ± 0.117, respectively. These are ablation results, not feature-tuning decisions.

The measured benchmark in `reports/benchmark.json` used 1,000 URLs in one process, one URL per call, with 100 warm-up calls excluded and the model loaded once before timing: **15.00 URLs/sec**, p50 **61.71 ms**, p95 **98.84 ms**. It includes feature extraction and model prediction.

## CI

[![CI](https://github.com/Krushna-Pisal/URL-Scanner/actions/workflows/ci.yml/badge.svg)](https://github.com/Krushna-Pisal/URL-Scanner/actions/workflows/ci.yml)

## Limitations

- Features are lexical only; this does not inspect page content or behavior.
- Dataset bias and label quality can make reported performance misleading.
- There is no live threat intelligence or reputation lookup.
- The scanner cannot see what a page displays after navigation.
- The configured brand list is small and does not cover every bank or payment service.
- Attackers can use clean-looking URLs that evade lexical features.
- Dataset diagnostics show substantial differences in URL length and query-string frequency between benign and phishing classes; these patterns may inflate evaluation results. Results are specific to this dataset, whose benign URLs are institutional websites, and do not establish real-world accuracy.
- Grouping uses `tldextract` with its suffix-list network refresh disabled. This uses bundled public-suffix data but is not a universal domain identity boundary.

Future work: add safe APK static analysis for linked Android packages without executing them.
