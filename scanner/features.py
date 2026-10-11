"""Extract explainable URL strings that can indicate phishing.

Every feature is lexical: the function parses the supplied text but never
opens a connection. Length, unusual punctuation, digits, and entropy can
signal obfuscation; path/query words and depth can imitate sign-in flows;
punycode and raw IPs can hide a destination; and risky TLDs/shorteners can
conceal destination identity. Banking features detect lookalike brands,
APK delivery, and common UPI lures. These are clues, not proof of fraud.
"""

import hashlib
import inspect
import json
from math import log2
import re
from urllib.parse import urlsplit


SUSPICIOUS_TLDS = {"zip", "mov", "top", "xyz", "work"}
URL_SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd"}
BRAND_DOMAINS = {
    "bank of india": "bankofindia.co.in",
    "sbi": "sbi.co.in",
    "hdfc": "hdfcbank.com",
    "icici": "icicibank.com",
    "axis": "axisbank.com",
    "kotak": "kotak.com",
    "phonepe": "phonepe.com",
    "paytm": "paytm.com",
    "gpay": "google.com",
    "bhim": "bhimupi.org.in",
}
# Short aliases are checked only in the registered-domain label. Brand text in
# a path, query, or unrelated subdomain therefore cannot trigger impersonation.
BRAND_ALIASES = {
    "bank of india": ("bankofindia",),
    "sbi": ("sbi",),
    "hdfc": ("hdfc", "hdfcbank"),
    "icici": ("icici", "icicibank"),
    "axis": ("axis", "axisbank"),
    "kotak": ("kotak", "kotakmahindra"),
    "phonepe": ("phonepe",),
    "paytm": ("paytm",),
    "gpay": ("gpay", "googlepay"),
    "bhim": ("bhim", "bhimupi"),
}
SUSPICIOUS_WORDS = ("login", "verify", "secure", "update", "account", "bank")
UPI_BAIT_WORDS = ("kyc", "upi", "reward", "refund", "blocked", "expired", "cashback")


def _levenshtein_distance(left: str, right: str) -> int:
    """Return edit distance using a row-by-row dynamic programming table.

    Each cell is the cheapest cost to transform prefixes: insertion,
    deletion, or substitution. Keeping only the previous row saves memory.
    """
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _normalize_brand_text(value: str) -> str:
    """Normalize common visual substitutions before comparing brand names."""
    return value.lower().replace("0", "o").replace("1", "l").replace("rn", "m").replace("-", "")


def _registered_domain(hostname: str) -> str:
    """Approximate a registered domain without DNS or a suffix database."""
    labels = [part for part in hostname.lower().strip(".").split(".") if part]
    if len(labels) >= 3 and labels[-1] == "in" and labels[-2] in {"co", "org", "net", "gov", "ac"}:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else (labels[0] if labels else "")


def extract_features(url: str) -> dict[str, float]:
    """Return numeric, safe-default URL features without making network calls.

    Parsing errors and non-string input produce the same all-zero feature
    shape. Features are simple signals: long/complex strings, credential-like
    markers, suspicious vocabulary, hidden host spellings, lookalike financial
    brands, and executable or UPI-bait paths can all occur in phishing URLs.
    """
    names = (
        "url_length", "hostname_length", "dot_count", "subdomain_count", "hyphen_count",
        "digit_ratio", "hostname_entropy", "has_at", "path_has_double_slash", "host_is_ipv4",
        "uses_https", "has_explicit_port", "path_depth", "query_length", "query_param_count",
        "has_punycode", "suspicious_tld", "suspicious_keyword_count", "is_url_shortener",
        "brand_similarity", "brand_impersonation_flag", "apk_link", "upi_bait_keywords",
    )
    result = {name: 0.0 for name in names}
    if not isinstance(url, str):
        return result
    try:
        parsed = urlsplit(url.strip())
        hostname = (parsed.hostname or "").lower().rstrip(".")
        # Accessing .port validates malformed and out-of-range port values.
        port = parsed.port
    except (ValueError, UnicodeError, AttributeError):
        return result

    path, query = parsed.path, parsed.query
    result.update({
        "url_length": float(len(url)),
        "hostname_length": float(len(hostname)),
        "dot_count": float(hostname.count(".")),
        "subdomain_count": float(max(0, len(hostname.split(".")) - 2)) if hostname else 0.0,
        "hyphen_count": float(url.count("-")),
        "digit_ratio": float(sum(char.isdigit() for char in url) / len(url)) if url else 0.0,
        "has_at": float("@" in url),
        "path_has_double_slash": float("//" in path),
        "uses_https": float(parsed.scheme.lower() == "https"),
        "has_explicit_port": float(port is not None),
        "path_depth": float(sum(bool(part) for part in path.split("/"))),
        "query_length": float(len(query)),
        "query_param_count": float(len(query.split("&"))) if query else 0.0,
        "has_punycode": float("xn--" in hostname),
        "suspicious_keyword_count": float(sum(url.lower().count(word) for word in SUSPICIOUS_WORDS)),
        "is_url_shortener": float(hostname in URL_SHORTENERS),
        "apk_link": float(path.lower().endswith(".apk")),
        "upi_bait_keywords": float(sum(url.lower().count(word) for word in UPI_BAIT_WORDS)),
    })
    try:
        parts = hostname.split(".")
        result["host_is_ipv4"] = float(len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts))
    except (ValueError, OverflowError):
        result["host_is_ipv4"] = 0.0
    if hostname:
        frequencies = {char: hostname.count(char) / len(hostname) for char in set(hostname)}
        result["hostname_entropy"] = float(-sum(p * log2(p) for p in frequencies.values()))
        tld = hostname.rsplit(".", 1)[-1]
        result["suspicious_tld"] = float(tld in SUSPICIOUS_TLDS)

        registered = _registered_domain(hostname)
        registered_label = registered.split(".", 1)[0]
        normalized_registered = _normalize_brand_text(registered_label)
        # Keep the whole registered label to detect joined suffix tricks, and
        # its hyphen-separated parts to detect misleading prefix/suffix forms.
        candidates = [normalized_registered] + [
            _normalize_brand_text(part)
            for part in re.split(r"[^a-zA-Z0-9]+", registered_label)
            if part
        ]
        best_score, best_brand = 0.0, ""
        for brand, aliases in BRAND_ALIASES.items():
            for alias in aliases:
                normalized_alias = _normalize_brand_text(alias)
                score = max(
                    1.0 - _levenshtein_distance(candidate, normalized_alias)
                    / max(len(candidate), len(normalized_alias), 1)
                    for candidate in candidates
                )
                if score > best_score:
                    best_score, best_brand = score, brand
        result["brand_similarity"] = float(max(0.0, best_score))
        official = BRAND_DOMAINS.get(best_brand, "")
        result["brand_impersonation_flag"] = float(
            best_score >= 0.80 and registered not in {official}
        )
    return result


def feature_extractor_fingerprint() -> str:
    """Fingerprint the feature logic and settings for model provenance checks.

    The model artifact stores this value so inference can detect a changed
    feature implementation instead of silently feeding it different values.
    """
    implementation = "\n".join(inspect.getsource(function) for function in (
        _levenshtein_distance, _normalize_brand_text, _registered_domain, extract_features,
    ))
    settings = json.dumps({
        "suspicious_tlds": sorted(SUSPICIOUS_TLDS),
        "url_shorteners": sorted(URL_SHORTENERS),
        "brand_domains": BRAND_DOMAINS,
        "brand_aliases": BRAND_ALIASES,
        "suspicious_words": SUSPICIOUS_WORDS,
        "upi_bait_words": UPI_BAIT_WORDS,
    }, sort_keys=True)
    return hashlib.sha256((implementation + settings).encode("utf-8")).hexdigest()
