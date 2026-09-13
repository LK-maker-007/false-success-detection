from __future__ import annotations

from math import comb, erfc, sqrt

import numpy as np
from sklearn.metrics import roc_auc_score


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, float]:
    # DeLong 1988, computed by the midrank method of Sun and Xu 2014. This is the standard test for
    # two ROC curves scored on the same samples; a bootstrap answers the same question less directly.
    positives = np.asarray(y) == 1
    scores = np.vstack([np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)])
    pos, neg = scores[:, positives], scores[:, ~positives]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        raise ValueError(f"DeLong needs both classes, got {m} positives and {n} negatives")

    tx = np.vstack([_midrank(row) for row in pos])
    ty = np.vstack([_midrank(row) for row in neg])
    tz = np.vstack([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(2)])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n

    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    diff = float(aucs[0] - aucs[1])
    if variance <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "z": 0.0, "p": 1.0}
    z = diff / sqrt(variance)
    return {
        "auc_a": round(float(aucs[0]), 4),
        "auc_b": round(float(aucs[1]), 4),
        "diff": round(diff, 4),
        "se": round(sqrt(variance), 4),
        "ci95": [round(diff - 1.96 * sqrt(variance), 4), round(diff + 1.96 * sqrt(variance), 4)],
        "z": round(z, 4),
        "p": erfc(abs(z) / sqrt(2)),
    }


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    # Holm 1979. Controls the family-wise error rate across a set of comparisons without assuming
    # they are independent, which they are not here: the same test rows feed every one of them.
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    total = len(ordered)
    out: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        adjusted = min(1.0, max(running, (total - rank) * p))
        running = adjusted
        out[name] = {"p": p, "p_holm": adjusted, "significant_at_alpha": adjusted <= alpha}
    return out


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
