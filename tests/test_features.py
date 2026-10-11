"""Hand-made examples verify feature behavior, not model metrics."""

import pytest

from scanner.features import _levenshtein_distance, extract_features


@pytest.mark.parametrize(("url", "expected"), [
    ("https://example.com", 19.0),
    ("http://example.com/a/b", 2.0),
    ("https://a.b.example.com", 23.0),
])
def test_basic_lengths_and_subdomains(url, expected):
    features = extract_features(url)
    if url.endswith("example.com"):
        assert features["url_length"] == expected
    else:
        assert features["path_depth"] == expected


def test_dot_and_subdomain_counts():
    f = extract_features("https://a.b.example.com")
    assert f["dot_count"] == 3
    assert f["subdomain_count"] == 2


def test_hyphen_count():
    assert extract_features("https://a-b.example.com/x-y")["hyphen_count"] == 2


def test_digit_ratio():
    assert extract_features("https://a1.com")["digit_ratio"] > 0


def test_entropy_is_positive_for_varied_hostname():
    assert extract_features("https://abcdef.com")["hostname_entropy"] > 0


def test_at_marker():
    assert extract_features("https://user@example.com")["has_at"] == 1


def test_double_slash_inside_path():
    assert extract_features("https://example.com//login")["path_has_double_slash"] == 1


@pytest.mark.parametrize("url", ["http://192.168.0.1/", "https://255.255.255.255"])
def test_ipv4_hosts(url):
    assert extract_features(url)["host_is_ipv4"] == 1


def test_non_ipv4_hostname():
    assert extract_features("https://example.com")["host_is_ipv4"] == 0


def test_https_flag():
    assert extract_features("HTTPS://example.com")["uses_https"] == 1


def test_explicit_port():
    assert extract_features("https://example.com:8443/a")["has_explicit_port"] == 1


def test_path_depth():
    assert extract_features("https://example.com/a/b/")["path_depth"] == 2


def test_query_length_and_parameter_count():
    f = extract_features("https://example.com/?a=1&b=2")
    assert f["query_length"] == 7
    assert f["query_param_count"] == 2


def test_punycode_flag():
    assert extract_features("https://xn--e1afmkfd.example/")["has_punycode"] == 1


def test_suspicious_tld():
    assert extract_features("https://example.zip")["suspicious_tld"] == 1


def test_suspicious_words_count():
    assert extract_features("https://example.com/login/verify")["suspicious_keyword_count"] == 2


def test_shortener_flag():
    assert extract_features("https://bit.ly/abc")["is_url_shortener"] == 1


def test_legitimate_bank_brand_domain():
    f = extract_features("https://www.hdfcbank.com/")
    assert f["brand_similarity"] > 0
    assert f["brand_impersonation_flag"] == 0


def test_legitimate_payment_brand_domain():
    f = extract_features("https://phonepe.com/")
    assert f["brand_similarity"] == 1
    assert f["brand_impersonation_flag"] == 0


def test_lookalike_brand_domain():
    f = extract_features("https://paytmm.example/")
    assert f["brand_similarity"] >= 0.8
    assert f["brand_impersonation_flag"] == 1


@pytest.mark.parametrize("url", [
    "https://paytm.example.com/",
    "https://paytm.com.evil.org/",
    "https://evil.com/paytm?brand=hdfc",
])
def test_brand_in_subdomain_path_or_query_is_not_hostname_impersonation(url):
    assert extract_features(url)["brand_impersonation_flag"] == 0


@pytest.mark.parametrize("suffix", ["/paytm/login", "/?next=paytm", "/?bank=hdfc"])
def test_brand_mentions_in_path_or_query_do_not_change_hostname_similarity(suffix):
    baseline = extract_features("https://example.com/")
    observed = extract_features("https://example.com" + suffix)
    assert observed["brand_similarity"] == baseline["brand_similarity"]
    assert observed["brand_impersonation_flag"] == baseline["brand_impersonation_flag"]


@pytest.mark.parametrize("url", ["https://secure.paytm.com/", "https://www.hdfcbank.com/"])
def test_official_brand_subdomains_are_not_flagged(url):
    assert extract_features(url)["brand_impersonation_flag"] == 0


def test_brand_prefix_and_suffix_lookalike_is_flagged():
    assert extract_features("https://secure-paytm-login.com/")["brand_impersonation_flag"] == 1


def test_apk_path_suffix_case_insensitive():
    assert extract_features("https://example.com/app.APK")["apk_link"] == 1


def test_upi_bait_count():
    assert extract_features("https://example.com/kyc/refund")["upi_bait_keywords"] == 2


@pytest.mark.parametrize(("left", "right", "expected"), [
    ("", "", 0), ("kitten", "sitting", 3), ("bank", "bank", 0),
    ("abc", "", 3), ("", "xyz", 3),
])
def test_levenshtein_distance(left, right, expected):
    assert _levenshtein_distance(left, right) == expected


@pytest.mark.parametrize("url", ["http://[broken", "http://example.com:99999", "", None])
def test_malformed_or_non_string_inputs_return_safe_defaults(url):
    features = extract_features(url)
    assert features["url_length"] == 0
    assert all(isinstance(value, float) for value in features.values())
