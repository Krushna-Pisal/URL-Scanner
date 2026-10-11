"""Model loading is cached and inference keeps the shared feature schema."""

from pathlib import Path

import numpy as np
import pytest

from scanner import model
from scanner.features import extract_features, feature_extractor_fingerprint


def test_artifact_is_loaded_once_for_same_path(monkeypatch):
    calls = []
    artifacts = [{"generation": "first"}, {"generation": "replacement"}]

    def fake_load(path):
        calls.append(path)
        return artifacts[len(calls) - 1]

    model._load_artifact.cache_clear()
    monkeypatch.setattr(model.joblib, "load", fake_load)
    path = Path("cached-test-model.joblib")
    try:
        first = model._load_artifact(path)
        # Loading another file version is invisible until an explicit refresh.
        second = model._load_artifact(path)
        assert first is second
        assert second["generation"] == "first"
        assert calls == [path]
        model._load_artifact.cache_clear()
        refreshed = model._load_artifact(path)
        assert refreshed["generation"] == "replacement"
        assert calls == [path, path]
    finally:
        model._load_artifact.cache_clear()


def test_training_and_inference_import_the_same_feature_extractor():
    from scripts import train

    assert train.extract_features is extract_features
    assert model.extract_features is extract_features


def test_predict_builds_model_frame_in_artifact_feature_order(monkeypatch):
    captured = {}

    class FakeClassifier:
        def predict_proba(self, frame):
            captured["columns"] = list(frame.columns)
            captured["row"] = frame.iloc[0].to_dict()
            return np.array([[0.8, 0.2]])

    names = list(extract_features("https://x.example/").keys())
    monkeypatch.setattr(model, "_load_artifact", lambda: {
        "feature_names": names,
        "best_model": FakeClassifier(),
        "best_model_name": "test_model",
    })
    prediction = model.predict("https://paytm.example.com/login")
    assert captured["columns"] == names
    assert captured["row"] == extract_features("https://paytm.example.com/login")
    assert prediction["score"] == 0.2
    assert "probability" not in prediction


@pytest.mark.parametrize(("score", "expected_level"), [
    (0.299, model.LOW), (0.3, model.MEDIUM), (0.5, model.MEDIUM),
    (0.7, model.MEDIUM), (0.701, model.HIGH),
])
def test_score_risk_bands_are_separate_named_thresholds(monkeypatch, score, expected_level):
    class FakeClassifier:
        def predict_proba(self, frame):
            return np.array([[1.0 - score, score]])

    monkeypatch.setattr(model, "_load_artifact", lambda: {
        "feature_names": [], "best_model": FakeClassifier(), "best_model_name": "test_model",
    })
    result = model.predict("https://example.com/")
    assert result["score"] == score
    assert result["level"] == expected_level


def test_dashboard_explains_uncalibrated_score_and_evaluation_cutoff():
    from pathlib import Path

    page = Path("web/index.html").read_text(encoding="utf-8")
    script = Path("web/static/app.js").read_text(encoding="utf-8")
    assert "uncalibrated" in page
    assert "0.50 cutoff" in page
    assert "0.30" in page and "0.70" in page
    assert 'toFixed(2)' in script
    assert '"%"' not in script


def test_predict_rejects_artifact_with_different_feature_semantics(monkeypatch):
    monkeypatch.setattr(model, "_load_artifact", lambda: {
        "feature_names": [],
        "feature_extractor_sha256": "different-feature-code",
        "best_model": object(),
        "best_model_name": "test_model",
    })
    with pytest.raises(RuntimeError, match="different URL features"):
        model.predict("https://example.com/")


def test_feature_fingerprint_is_stable_and_hexadecimal():
    fingerprint = feature_extractor_fingerprint()
    assert len(fingerprint) == 64
    assert int(fingerprint, 16) >= 0
    assert feature_extractor_fingerprint() == fingerprint

