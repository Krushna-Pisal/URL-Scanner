"""Evidence fields are factual, bounded, and separate from model inputs."""

from scanner.evidence import build_scan_evidence, supplemental_observations
from scanner.features import extract_features


def test_evidence_has_versioned_model_and_heuristic_sections_without_url_text():
    url = "https://xn--e1afmkfd.example/login/kyc.apk?upi=1"
    features = extract_features(url)
    evidence = build_scan_evidence(url, .72, "HIGH", "trained_classifier_predict_proba",
                                   "stacking", list(features), features)
    serialized = str(evidence)
    assert evidence["schema_version"] == "url-threat-scan-evidence.v1"
    assert evidence["features"]["model_input_values"] == features
    assert "heuristic_only_observations" in evidence["features"]
    assert evidence["model_attribution"]["items"] == []
    assert url not in serialized
    assert "xn--e1afmkfd.example" not in serialized


def test_supplemental_observations_record_shape_without_component_text():
    observed = supplemental_observations("https://例え.example/login/verify?upi=%41")
    assert observed["unicode_hostname"] == 1
    assert observed["percent_encoded_octet_count"] == 1
    assert observed["account_security_keyword_counts_by_component"]["path"] == 2
    assert observed["financial_bait_keyword_counts_by_component"]["query"] == 1
    assert "例え" not in str(observed)


def test_malformed_url_has_safe_supplemental_defaults():
    observed = supplemental_observations("http://[bad")
    assert observed["unicode_hostname"] == 0
    assert observed["path_length"] == 0


def test_missing_or_malformed_model_values_are_ignored_or_defaulted():
    evidence = build_scan_evidence("https://example.org", .2, "LOW", "method", "test",
                                   ["uses_https", "url_length", "unknown"],
                                   {"uses_https": "bad", "url_length": 44})
    assert evidence["features"]["model_input_values"] == {"uses_https": 0.0, "url_length": 44.0}

