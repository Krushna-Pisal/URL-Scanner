"""Rule scorer tests verify bounded, explainable output."""

from scanner.features import extract_features
from scanner.heuristics import WEIGHTS, score_features, score_url


def test_clean_url_has_zero_score_and_no_reasons():
    assert score_url("https://example.com/") == (0.0, [])


def test_risky_url_has_score_and_plain_english_reasons():
    score, reasons = score_url("http://192.168.1.1/login.apk")
    assert 0 < score <= 1
    assert reasons
    assert all(isinstance(reason, str) for reason in reasons)


def test_score_is_bounded_for_many_signals():
    score, _ = score_url("http://paytm-secure-login.top//kyc/refund.apk?x=123")
    assert 0 <= score <= 1


def test_every_weight_has_a_positive_value():
    assert WEIGHTS
    assert all(weight > 0 for weight in WEIGHTS.values())


def test_reusing_extracted_features_preserves_heuristic_score():
    url = "https://paytm-secure-login.top//kyc/refund.apk?x=123"
    assert score_features(extract_features(url))[0] == score_url(url)[0]
