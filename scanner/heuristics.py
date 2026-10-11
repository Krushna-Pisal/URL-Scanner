"""Small transparent rule scorer for the lexical features."""

from scanner.features import extract_features
from math import isfinite


# Keeping each contribution here makes the rule score easy to explain and tune.
WEIGHTS = {
    "host_is_ipv4": 0.16,
    "has_at": 0.12,
    "has_punycode": 0.12,
    "suspicious_tld": 0.08,
    "is_url_shortener": 0.08,
    "brand_impersonation_flag": 0.18,
    "apk_link": 0.10,
    "suspicious_keyword_count": 0.04,
    "upi_bait_keywords": 0.04,
    "path_has_double_slash": 0.03,
    "digit_ratio": 0.02,
    "has_explicit_port": 0.03,
}


def score_url(url: str) -> tuple[float, list[str]]:
    """Score a URL from 0 to 1 and explain every active rule."""
    features = extract_features(url)
    score, reasons, _signals = score_features(features)
    return score, reasons


def score_features(
    features: dict[str, float]
) -> tuple[float, list[str], list[dict[str, float | str]]]:
    """Score existing feature values and expose exact rule contributions.

    The contribution is the rule weight times its capped feature value. It is
    evidence about the heuristic score only, never a decomposition of an ML
    model's score.
    """
    reasons: list[str] = []
    signals: list[dict[str, float | str]] = []
    score = 0.0
    explanations = {
        "host_is_ipv4": "The hostname is a raw IPv4 address.",
        "has_at": "The URL contains an @ marker that can disguise its destination.",
        "has_punycode": "The hostname contains a punycode label.",
        "suspicious_tld": "The hostname uses a configured suspicious top-level domain.",
        "is_url_shortener": "The hostname is a configured URL shortener.",
        "brand_impersonation_flag": "The registered domain resembles a configured bank or payment brand.",
        "apk_link": "The URL path ends with an Android APK file.",
        "suspicious_keyword_count": "The URL contains sign-in or account-related terms.",
        "upi_bait_keywords": "The URL contains UPI or payment-bait terms.",
        "path_has_double_slash": "The URL path contains an extra double slash.",
        "digit_ratio": "The URL contains many digits.",
        "has_explicit_port": "The URL specifies an explicit port.",
    }
    for name, weight in WEIGHTS.items():
        try:
            value = float(features.get(name, 0.0))
        except (TypeError, ValueError, OverflowError):
            value = 0.0
        if not isfinite(value):
            value = 0.0
        if value > 0:
            # Counts are bounded so a repeated word cannot overwhelm the rules.
            contribution = weight * min(value, 1.0)
            score += contribution
            reasons.append(explanations[name])
            signals.append({
                "feature": name,
                "observed_value": float(value),
                "rule_weight": float(weight),
                "score_addition": float(contribution),
                "direction": "increases_heuristic_score",
                "description": explanations[name],
            })
    return min(1.0, max(0.0, score)), reasons, signals
