from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from fsdetect.corpus import Record

# Appendix C of arXiv 2606.09863.
TEXT_VECTORIZER = {
    "ngram_range": (1, 2),
    "max_features": 30000,
    "min_df": 2,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "lowercase": True,
}

# actions + arguments, the configuration that won the ablation in results/feature_ablation.json
STRUCTURAL_VECTORIZER = {
    "ngram_range": (1, 3),
    "max_features": 30000,
    "min_df": 2,
    "sublinear_tf": True,
    "token_pattern": r"\S+",
    "lowercase": False,
}

ACTION_VECTORIZER = {
    "ngram_range": (1, 3),
    "max_features": 30000,
    "min_df": 2,
    "sublinear_tf": True,
    "token_pattern": r"\S+",
    "lowercase": False,
}

Detector = tuple[TfidfVectorizer, LogisticRegression]
Combined = tuple[list[tuple[str, TfidfVectorizer]], LogisticRegression]
COMBINED_SPEC = {"closing": TEXT_VECTORIZER, "structural": STRUCTURAL_VECTORIZER}
Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def build_dataset(labels: dict[str, list[Record]]) -> tuple[list[Record], np.ndarray]:
    positives, negatives = labels["FS"], labels["TS"]
    if not positives or not negatives:
        raise ValueError(f"FS vs TS needs both classes, got {len(positives)} FS and {len(negatives)} TS")
    rows = positives + negatives
    y = np.array([1] * len(positives) + [0] * len(negatives))
    return rows, y


def stratified_split(y: np.ndarray, seed: int) -> Split:
    index = np.arange(len(y))
    train, rest = train_test_split(index, test_size=0.30, random_state=seed, stratify=y)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed, stratify=y[rest])
    return train, val, test


def fit(rows: list[Record], y: np.ndarray, train: np.ndarray, field: str, config: dict, seed: int) -> Detector:
    vectorizer = TfidfVectorizer(**config)
    features = vectorizer.fit_transform([rows[i][field] for i in train])
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    classifier.fit(features, y[train])
    return vectorizer, classifier


def score(detector: Detector, texts: list[str]) -> np.ndarray:
    vectorizer, classifier = detector
    return classifier.predict_proba(vectorizer.transform(texts))[:, 1]


def fit_combined(rows: list[Record], y: np.ndarray, train: np.ndarray, specs: dict[str, dict], seed: int) -> Combined:
    # Prose and action tokens need different tokenizers, so they are vectorized separately and
    # stacked rather than concatenated into one string.
    fitted: list[tuple[str, TfidfVectorizer]] = []
    blocks = []
    for field, config in specs.items():
        vectorizer = TfidfVectorizer(**config)
        blocks.append(vectorizer.fit_transform([rows[i][field] for i in train]))
        fitted.append((field, vectorizer))
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
    classifier.fit(hstack(blocks).tocsr(), y[train])
    return fitted, classifier


def score_combined(detector: Combined, texts: dict[str, list[str]]) -> np.ndarray:
    fitted, classifier = detector
    missing = [field for field, _ in fitted if field not in texts]
    if missing:
        raise ValueError(f"score_combined needs a column per fitted field, missing {missing}")
    blocks = [vectorizer.transform(texts[field]) for field, vectorizer in fitted]
    return classifier.predict_proba(hstack(blocks).tocsr())[:, 1]


def auroc(detector: Detector, texts: list[str], y: np.ndarray) -> float:
    return float(roc_auc_score(y, score(detector, texts)))


def precision_at_flag_rate(scores: np.ndarray, y: np.ndarray, flag_rate: float) -> float:
    n_flagged = max(1, int(round(flag_rate * len(scores))))
    flagged = np.argsort(-scores)[:n_flagged]
    return float(y[flagged].mean())
