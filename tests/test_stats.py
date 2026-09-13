from __future__ import annotations

import numpy as np
import pytest

from fsdetect.stats import bootstrap_auroc, mcnemar_exact, paired_bootstrap_auroc_diff, spread


def test_mcnemar_is_one_when_nothing_is_discordant():
    a = np.array([True, False, True])
    assert mcnemar_exact(a, a) == {"a_only": 0, "b_only": 0, "discordant": 0, "exact_two_sided_p": 1.0}


def test_mcnemar_counts_each_direction_and_is_symmetric_in_p():
    a = np.array([True, True, True, False])
    b = np.array([False, False, False, False])
    forward = mcnemar_exact(a, b)
    assert (forward["a_only"], forward["b_only"], forward["discordant"]) == (3, 0, 3)
    assert mcnemar_exact(b, a)["exact_two_sided_p"] == forward["exact_two_sided_p"]


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 40)
    scores = np.where(y == 1, rng.normal(1.0, 1.0, 100), rng.normal(0.0, 1.0, 100))
    result = bootstrap_auroc(y, scores, resamples=200, seed=0)
    low, high = result["ci95"]
    assert low <= result["auroc"] <= high


def test_bootstrap_survives_a_heavily_imbalanced_split():
    # An unstratified resample of this would routinely draw zero positives and roc_auc_score
    # would raise, so the stratified draw is what makes the interval computable at all.
    y = np.array([0] * 199 + [1])
    scores = np.linspace(0, 1, 200)
    assert bootstrap_auroc(y, scores, resamples=50, seed=1)["ci95"][0] <= 1.0


def test_paired_diff_of_a_detector_against_itself_is_zero_and_includes_zero():
    y = np.array([0] * 50 + [1] * 50)
    scores = np.linspace(0, 1, 100)
    result = paired_bootstrap_auroc_diff(y, scores, scores, resamples=100, seed=0)
    assert result["diff"] == 0.0
    assert result["excludes_zero"] is False


def test_paired_diff_detects_a_real_gap():
    y = np.array([0] * 50 + [1] * 50)
    good = np.concatenate([np.zeros(50), np.ones(50)])
    coin = np.tile([0.0, 1.0], 50)
    result = paired_bootstrap_auroc_diff(y, good, coin, resamples=200, seed=0)
    assert result["diff"] > 0
    assert result["excludes_zero"] is True


def test_spread_of_a_single_value_reports_zero_std():
    assert spread([0.9])["std"] == 0.0
    with pytest.raises(ValueError):
        spread([])


def test_score_combined_rejects_a_missing_column():
    from sklearn.feature_extraction.text import TfidfVectorizer

    from fsdetect.detect import score_combined

    vec = TfidfVectorizer().fit(["alpha beta"])
    with pytest.raises(ValueError, match="missing"):
        score_combined(([("closing", vec)], None), {"structural": ["x"]})
