from __future__ import annotations

from math import comb

import numpy as np
from sklearn.metrics import roc_auc_score


def mcnemar_exact(decisions_a: np.ndarray, decisions_b: np.ndarray) -> dict[str, float | int]:
    a_only = int(np.sum(decisions_a & ~decisions_b))
    b_only = int(np.sum(~decisions_a & decisions_b))
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "discordant": 0, "exact_two_sided_p": 1.0}
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": n,
        "exact_two_sided_p": round(min(1.0, 2 * tail), 6),
    }


def _resample_indices(y: np.ndarray, resamples: int, seed: int) -> list[np.ndarray]:
    # Stratified resampling: an unstratified draw can return a fold with one class absent, and
    # roc_auc_score raises on that rather than returning a value to average over.
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    return [
        np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        for _ in range(resamples)
    ]


def bootstrap_auroc(y: np.ndarray, scores: np.ndarray, resamples: int, seed: int) -> dict[str, float]:
    values = [roc_auc_score(y[take], scores[take]) for take in _resample_indices(y, resamples, seed)]
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "auroc": round(float(roc_auc_score(y, scores)), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "resamples": resamples,
    }


def paired_bootstrap_auroc_diff(
    y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, resamples: int, seed: int
) -> dict[str, float | bool]:
    diffs = [
        roc_auc_score(y[take], scores_a[take]) - roc_auc_score(y[take], scores_b[take])
        for take in _resample_indices(y, resamples, seed)
    ]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": round(float(roc_auc_score(y, scores_a) - roc_auc_score(y, scores_b)), 4),
        "ci95": [round(float(low), 4), round(float(high), 4)],
        "excludes_zero": bool(low > 0 or high < 0),
        "resamples": resamples,
    }


def spread(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("spread needs at least one value")
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std(ddof=1)), 4) if len(array) > 1 else 0.0,
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
        "seeds": len(array),
    }
