"""Checks for dataset-bias warnings and explicit feature-importance labels."""

import numpy as np
import pandas as pd

from scripts.train import _print_dataset_sanity, _top_feature_report


def test_sanity_diagnostics_warn_for_length_query_and_https_gaps(capsys):
    data = pd.DataFrame({
        "url": [
            "https://a.co/", "https://b.co/",
            "https://phishing.example.com/" + "x" * 80 + "?account=1",
            "http://phishing2.example.com/" + "y" * 80 + "?verify=1",
        ],
        "label": [0, 0, 1, 1],
    })
    summary, remove_https = _print_dataset_sanity(data)
    output = capsys.readouterr().out
    warnings = " ".join(summary["warnings"])
    assert "mean URL length ratio" in warnings
    assert "query-string share" in warnings
    assert "HTTPS share" in warnings
    assert "https_share=" in output
    # A HTTPS prevalence gap is a diagnostic only; it never drops the feature.
    assert not remove_https
    assert summary["diagnostic_thresholds"]["query_share_gap"] == 0.05


def test_top_feature_report_distinguishes_signed_coefficients_from_importance():
    features = ["positive", "negative"]
    fold_values = {
        "logistic_regression": [np.array([2.0, -1.0])],
        "random_forest": [np.array([0.7, 0.3])],
        "stacking": [np.array([1.4, 0.6])],
    }
    report = _top_feature_report(features, fold_values)
    logistic = report["logistic_regression"][0]
    assert logistic["importance_type"] == "absolute_standardized_logistic_coefficient"
    assert logistic["mean_signed_coefficient"] == 2.0
    assert "mean_importance" not in logistic
    assert report["random_forest"][0]["importance_type"] == "random_forest_impurity_importance"
    assert report["stacking"][0]["importance_type"] == "weighted_absolute_base_model_importance"
    assert all("mean_importance" in report[name][0] for name in ("random_forest", "stacking"))
