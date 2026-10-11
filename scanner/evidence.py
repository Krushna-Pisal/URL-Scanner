"""Build bounded, URL-free evidence for the explanation and dashboard.

The trained feature vector stays unchanged. A few descriptive observations
are computed separately and labeled heuristic-only so they cannot be mistaken
for inputs to the saved classifier.
"""

import re
from typing import Any
from urllib.parse import urlsplit

from scanner.features import SUSPICIOUS_WORDS, UPI_BAIT_WORDS, extract_features
from scanner.heuristics import score_features


FEATURE_DESCRIPTIONS = {
    "url_length": "Number of characters in the submitted URL.",
    "hostname_length": "Number of characters in the parsed hostname.",
    "dot_count": "Dots in the hostname.",
    "subdomain_count": "Hostname labels before the registrable domain, approximately counted.",
    "hyphen_count": "Hyphens in the URL text.",
    "digit_ratio": "Share of URL characters that are digits.",
    "hostname_entropy": "Character entropy of the hostname; variety is not proof of randomness.",
    "has_at": "Whether an @ marker occurs in the URL text.",
    "path_has_double_slash": "Whether the parsed path contains a double slash.",
    "host_is_ipv4": "Whether the hostname parses as a dotted IPv4 address.",
    "uses_https": "Whether the URL scheme is HTTPS; encryption does not prove legitimacy.",
    "has_explicit_port": "Whether the URL specifies a port.",
    "path_depth": "Non-empty path segments.",
    "query_length": "Characters in the query string.",
    "query_param_count": "Ampersand-separated query parts, not necessarily valid parameters.",
    "has_punycode": "Whether the hostname contains an xn-- label.",
    "suspicious_tld": "Whether the final hostname label is in the local configured list.",
    "suspicious_keyword_count": "Occurrences of configured account/security terms in URL text.",
    "is_url_shortener": "Whether the hostname exactly matches a configured shortener.",
    "brand_similarity": "Best string similarity to a configured brand alias on the registered-domain label.",
    "brand_impersonation_flag": "Whether that label is similar and not an exact configured official domain.",
    "apk_link": "Whether the path ends in .apk, case-insensitively.",
    "upi_bait_keywords": "Occurrences of configured payment-bait terms in URL text.",
}

_ENCODED_OCTET = re.compile(r"%[0-9a-fA-F]{2}")
_SPECIAL_CHARACTER = re.compile(r"[^A-Za-z0-9\s]")


def supplemental_observations(url: str) -> dict[str, Any]:
    """Return URL-shape facts that are not part of the trained feature schema."""
    if not isinstance(url, str):
        url = ""
    try:
        parsed = urlsplit(url.strip())
        hostname = parsed.hostname or ""
        path = parsed.path
        query = parsed.query
    except (ValueError, UnicodeError, AttributeError):
        hostname, path, query = "", "", ""

    components = {"hostname": hostname.lower(), "path": path.lower(), "query": query.lower()}

    def keyword_counts(words: tuple[str, ...]) -> dict[str, int]:
        return {
            component: sum(text.count(word) for word in words)
            for component, text in components.items()
        }

    special_count = len(_SPECIAL_CHARACTER.findall(url))
    encoded_count = len(_ENCODED_OCTET.findall(url))
    return {
        "special_character_count": special_count,
        "special_character_ratio": special_count / len(url) if url else 0.0,
        "percent_encoded_octet_count": encoded_count,
        "unicode_hostname": int(any(ord(character) > 127 for character in hostname)),
        "path_length": len(path),
        "account_security_keyword_counts_by_component": keyword_counts(SUSPICIOUS_WORDS),
        "financial_bait_keyword_counts_by_component": keyword_counts(UPI_BAIT_WORDS),
    }


def build_scan_evidence(
    url: str,
    score: float,
    level: str,
    scoring_method: str,
    model_name: str,
    model_features: list[str] | None = None,
    feature_values: dict[str, float] | None = None,
    model_attribution: dict[str, Any] | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Create versioned evidence without including the raw URL or components."""
    values = feature_values if feature_values is not None else extract_features(url)
    feature_names = model_features or []
    heuristic_score, _reasons, signals = score_features(values)
    safe_model_values = {}
    for name in feature_names:
        if name not in FEATURE_DESCRIPTIONS:
            continue
        try:
            value = float(values.get(name, 0.0))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if value == value and abs(value) != float("inf"):
            safe_model_values[name] = value
    evidence: dict[str, Any] = {
        "schema_version": "url-threat-scan-evidence.v1",
        "assessment": {
            "score": float(score),
            "level": level,
            "score_meaning": (
                "Classifier or heuristic score for ranking URL risk; not a calibrated "
                "probability that the destination is malicious."
            ),
        },
        "scoring": {"method": scoring_method, "model_name": model_name},
        "features": {
            "model_input_values": safe_model_values,
            "model_feature_names": list(feature_names),
            "descriptions": {name: FEATURE_DESCRIPTIONS[name] for name in safe_model_values
                             if name in FEATURE_DESCRIPTIONS},
            "heuristic_only_observations": supplemental_observations(url),
        },
        "heuristics": {
            "score": float(heuristic_score),
            "signals": signals,
            "score_note": "Rule contributions describe the separate heuristic score, not the ML score.",
        },
        "model_attribution": model_attribution or {
            "method": "not_available_for_this_scoring_method",
            "items": [],
        },
    }
    if model_version:
        evidence["scoring"]["model_version"] = model_version
    return evidence
