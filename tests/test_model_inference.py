"""Model loading is cached and inference keeps the shared feature schema."""

from pathlib import Path

import numpy as np

from scanner import model
from scanner.features import extract_features


def test_artifact_is_loaded_once_for_same_path(monkeypatch):
    calls = []
    artifact = {"feature_names": []}

    def fake_load(path):
        calls.append(path)
        return artifact

    model._load_artifact.cache_clear()
    monkeypatch.setattr(model.joblib, "load", fake_load)
    path = Path("cached-test-model.joblib")
    try:
        first = model._load_artifact(path)
        second = model._load_artifact(path)
        assert first is second is artifact
        assert calls == [path]
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
    assert prediction["probability"] == 0.2

