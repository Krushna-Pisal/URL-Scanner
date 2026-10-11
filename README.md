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

Possible phishing or malware URL sources include [PhishTank's verified online phishing feed](https://phishtank.org/developer_info.php) and [URLhaus data exports](https://urlhaus.abuse.ch/api/). PhishTank's bulk feed may require an application key for regular downloads; review its current developer terms and attribution requirements. URLhaus requires an Auth-Key for downloads, describes fair-use community access, and directs commercial use to its enhanced service and terms. Treat the sources as candidates for a research dataset, not interchangeable labels: URLhaus focuses on malware distribution and is not a general phishing-only source.

For legitimate examples, use a documented, permitted source such as a dated [Tranco ranking](https://tranco-list.eu/) as a candidate list, or a hand-curated list of official domains. Popularity does not prove that a URL is safe, and a domain list is not equivalent to a verified legitimate URL label. Before downloading, training on, or redistributing any source, read the source's current license, terms, attribution rules, and rate limits. Record source, retrieval date, label definition, and any filtering in your own dataset notes. Keep the raw data private when redistribution rights are unclear.

After preparing the CSV, train and benchmark locally:

```powershell
python -m scripts.train
python -m scripts.benchmark
```

Training uses a domain-grouped held-out split and writes computed evaluation values to `reports/metrics.json`; it saves the selected estimator to `models/model.joblib`. The benchmark scores a seeded sample of 10,000 dataset rows and writes measured throughput and latency to `reports/benchmark.json`. Both reports are generated artifacts, not hand-entered results.

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
  "score": "<computed probability from 0 to 1>",
  "level": "<LOW, MEDIUM, or HIGH>",
  "reasons": ["<plain-English feature reason, if any>"],
  "model": "<heuristic or trained_model>",
  "scanned_at": "<UTC timestamp>"
}
```

`GET /history?limit=20` returns recent stored scans. `GET /health` is a public health check. Scan URLs are never fetched.

## Results

The table below is populated only from generated report files. The training dataset has not been supplied in this checkout, so neither report is currently available and no measurements are shown.

| Report | Values |
| --- | --- |
| `reports/metrics.json` | Not generated: `data/raw/urls.csv` is absent |
| `reports/benchmark.json` | Not generated: `data/raw/urls.csv` is absent |

## CI

[![CI](https://github.com/<owner>/<repository>/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/<repository>/actions/workflows/ci.yml)


## Limitations

- Features are lexical only; this does not inspect page content or behavior.
- Dataset bias and label quality can make reported performance misleading.
- There is no live threat intelligence or reputation lookup.
- The scanner cannot see what a page displays after navigation.
- The configured brand list is small and does not cover every bank or payment service.
- Attackers can use clean-looking URLs that evade lexical features.
- The domain grouping in training uses a simple last-two-label approximation and can group some domains incorrectly.

Future work: add safe APK static analysis for linked Android packages without executing them.
