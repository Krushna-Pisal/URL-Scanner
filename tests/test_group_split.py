"""Grouped split check: one registered-domain group cannot cross partitions."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from scripts.train import registered_domain


def test_group_split_has_zero_domain_overlap():
    urls = [
        "https://login.example.com/a", "https://www.example.com/b",
        "https://secure.other.org/a", "https://other.org/b",
        "https://shop.sample.net/a", "https://sample.net/b",
        "https://pay.demo.io/a", "https://demo.io/b",
    ]
    groups = np.array([registered_domain(url) for url in urls])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=7)
    train_indices, test_indices = next(splitter.split(pd.DataFrame({"url": urls}), labels, groups))
    assert set(groups[train_indices]).isdisjoint(set(groups[test_indices]))


def test_registered_domain_uses_public_suffix_for_uk():
    assert registered_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"


def test_registered_domain_uses_public_suffix_for_japan():
    assert registered_domain("https://x.ac.jp/") == "x.ac.jp"


def test_registered_domain_keeps_hosting_platform_as_group():
    assert registered_domain("https://customer.weebly.com/page") == "weebly.com"


def test_registered_domain_keeps_raw_ip_as_its_own_group():
    assert registered_domain("http://192.0.2.10/path") == "192.0.2.10"
