"""FastAPI service that scores URL text and stores scan history."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from scanner.explanations import explain_scan
from scanner.evidence import build_scan_evidence
from scanner.features import extract_features
from scanner.heuristics import score_features
from scanner.storage import insert_scan, list_scans


load_dotenv()
app = FastAPI(title="URL Threat Scanner")
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")


class ScanRequest(BaseModel):
    """Validate the submitted text before any feature extraction happens."""

    url: str = Field(min_length=1, max_length=2048)


class ExplainRequest(BaseModel):
    """Accept only the scanner's completed assessment evidence, not a URL."""

    score: float = Field(ge=0, le=1)
    level: str = Field(pattern="^(LOW|MEDIUM|HIGH)$")
    reasons: list[Annotated[str, Field(max_length=300)]] = Field(max_length=12)
    evidence: dict[str, Any] | None = None


def _database_path() -> Path:
    """Read the database location on each request for test and deployment config."""
    return Path(os.environ.get("URL_SCANNER_DB_PATH", "data/scans.sqlite3"))


def _check_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Enforce a configured key while leaving local development unprotected."""
    expected = os.environ.get("URL_SCANNER_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _scan_url(url: str) -> dict[str, Any]:
    """Use a trained classifier when available, otherwise use transparent rules."""
    try:
        from scanner.model import predict

        prediction = predict(url)
        return {
            "score": prediction["probability"],
            "level": prediction["level"],
            "reasons": [item["description"] for item in prediction["reasons"]],
            "model": "trained_model",
            "evidence": prediction["evidence"],
        }
    except FileNotFoundError:
        # Phase 2's dataset may not yet be supplied; keep API scans useful
        # while clearly naming the rule-based model in each stored result.
        features = extract_features(url)
        score, reasons, _signals = score_features(features)
        if score < 0.3:
            level = "LOW"
        elif score <= 0.7:
            level = "MEDIUM"
        else:
            level = "HIGH"
        evidence = build_scan_evidence(
            url, score, level, "weighted_heuristic_rules", "heuristic",
            feature_values=features,
            model_attribution={"method": "not_available_for_heuristic_scoring", "items": []},
        )
        return {"score": score, "level": level, "reasons": reasons, "model": "heuristic", "evidence": evidence}


@app.post("/scan", dependencies=[Depends(_check_api_key)])
def scan(request: ScanRequest) -> dict[str, Any]:
    """Score the URL characters only; never request the submitted address."""
    result = _scan_url(request.url)
    result.update({
        "url": request.url,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    })
    insert_scan(_database_path(), result)
    return result


@app.get("/history", dependencies=[Depends(_check_api_key)])
def history(limit: int = Query(default=20, ge=1, le=1000)) -> list[dict[str, Any]]:
    """Return the newest persisted scan records."""
    return list_scans(_database_path(), limit)


@app.get("/health")
def health() -> dict[str, str]:
    """Report process health without touching the network or scan database."""
    return {"status": "ok"}


@app.get("/config")
def public_config() -> dict[str, bool]:
    """Expose feature availability, never credential values."""
    return {
        "api_key_required": bool(os.environ.get("URL_SCANNER_API_KEY")),
        "ai_explanations_enabled": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.post("/explain", dependencies=[Depends(_check_api_key)])
def explain(request: ExplainRequest) -> dict[str, Any]:
    """Explain scanner evidence; the AI cannot change the assessment."""
    bounded_reasons = [reason[:300] for reason in request.reasons]
    return explain_scan(request.score, request.level, bounded_reasons, request.evidence)


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """Serve the same-origin browser interface."""
    return FileResponse(WEB_ROOT / "index.html")
