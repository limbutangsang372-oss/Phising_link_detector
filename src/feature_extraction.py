"""
Feature extraction for phishing URL detection.

The feature extraction is defensive and educational. It analyses URL structure only.
It does not visit websites, download web pages, collect credentials, or interact
with external systems.
"""

import re
from urllib.parse import urlparse

try:
    import tldextract
except ImportError:
    tldextract = None


SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank",
    "signin", "password", "confirm", "billing", "wallet", "urgent",
    "free", "prize", "warning", "suspend"
]


FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_slashes",
    "num_question_marks",
    "num_equals",
    "num_at_symbols",
    "num_digits",
    "uses_https",
    "has_ip_address",
    "subdomain_length",
    "domain_length",
    "tld_length",
    "suspicious_word_count",
]


def count_digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def has_ip_address(url: str) -> int:
    pattern = r"(?:\d{1,3}\.){3}\d{1,3}"
    return int(bool(re.search(pattern, url)))


def normalise_url(url: str) -> str:
    url = str(url).strip()
    if not url:
        return url
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return "http://" + url
    return url


def extract_domain_parts(url: str):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if tldextract:
        extracted = tldextract.extract(url)
        domain = extracted.domain
        suffix = extracted.suffix
        subdomain = extracted.subdomain
    else:
        parts = hostname.split(".")
        domain = parts[-2] if len(parts) >= 2 else hostname
        suffix = parts[-1] if len(parts) >= 2 else ""
        subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""

    return hostname, domain, suffix, subdomain


def extract_url_features(url: str) -> dict:
    url = normalise_url(url)
    parsed = urlparse(url)
    hostname, domain, suffix, subdomain = extract_domain_parts(url)

    lower_url = url.lower()

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_at_symbols": url.count("@"),
        "num_digits": count_digits(url),
        "uses_https": int(parsed.scheme == "https"),
        "has_ip_address": has_ip_address(url),
        "subdomain_length": len(subdomain),
        "domain_length": len(domain),
        "tld_length": len(suffix),
        "suspicious_word_count": sum(word in lower_url for word in SUSPICIOUS_WORDS),
    }


def transform_urls_to_features(urls):
    return [extract_url_features(url) for url in urls]
