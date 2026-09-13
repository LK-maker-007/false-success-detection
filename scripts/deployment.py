"""Measure scoring latency and test whether a detector transfers between benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fsdetect.appworld import build_dataset as appworld_dataset
from fsdetect.appworld import load as appworld_load
from fsdetect.corpus import closing_message, load, otel_tokens, split_labels
from fsdetect.detect import STRUCTURAL_VECTORIZER, build_dataset, stratified_split


def fit_on(texts: list[str], y: np.ndarray) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(**STRUCTURAL_VECTORIZER)
    features = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=42)
    classifier.fit(features, y)
    return vectorizer, classifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--appworld", type=Path, default=Path("data/appworld/leaderboard/experiments/outputs"))
    parser.add_argument("--latency-n", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("results/deployment.json"))
    args = parser.parse_args()

    records, _ = load(args.data)
    rows, y = build_dataset(split_labels(records))
    train, _, test = stratified_split(y, 42)
    vectorizer, classifier = fit_on([rows[i]["otel_zero_config"] for i in train], y[train])

    # Latency, measured on raw trajectories so extraction is included. A deployment pays that cost.
    raw: list[list[dict]] = []
    for path in sorted(args.data.rglob("*.json")):
        if path.name == "_bucket_keys.json" or len(raw) >= args.latency_n:
            continue
        for sim in json.loads(path.read_text(encoding="utf-8"))["simulations"]:
            messages = sim.get("messages")
            if messages and closing_message(messages) is not None:
                raw.append(messages)
            if len(raw) >= args.latency_n:
                break
    turns = [len(m) for m in raw]
    print(f"latency sample: {len(raw)} trajectories, median {sorted(turns)[len(turns) // 2]} messages", flush=True)

    extract_ms, score_ms = [], []
    for messages in raw:
        t0 = time.perf_counter()
        tokens = otel_tokens(messages)
        t1 = time.perf_counter()
        classifier.predict_proba(vectorizer.transform([tokens]))[:, 1]
        t2 = time.perf_counter()
        extract_ms.append((t1 - t0) * 1000)
        score_ms.append((t2 - t1) * 1000)

    total = [a + b for a, b in zip(extract_ms, score_ms, strict=True)]

    def pct(values: list[float]) -> dict[str, float]:
        a = np.asarray(values)
        return {
            "mean_ms": round(float(a.mean()), 3),
            "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "max_ms": round(float(a.max()), 3),
        }

    latency = {"extract": pct(extract_ms), "score": pct(score_ms), "end_to_end": pct(total)}
    print(f"  extract    {latency['extract']}")
    print(f"  score      {latency['score']}")
    print(f"  end to end {latency['end_to_end']}", flush=True)

    batched_start = time.perf_counter()
    classifier.predict_proba(vectorizer.transform([otel_tokens(m) for m in raw]))[:, 1]
    batched_ms = (time.perf_counter() - batched_start) * 1000 / len(raw)
    print(f"  batched, per trajectory {batched_ms:.3f} ms", flush=True)

    # Cross-benchmark transfer. Feature vocabularies are built from different action
    # representations, so the expectation is chance. Measured rather than assumed.
    aw_records, _ = appworld_load(args.appworld, require_explicit_status=False)
    aw_records, aw_y_list = appworld_dataset(aw_records)
    aw_y = np.array(aw_y_list)
    aw_texts = [f"{r['endpoints']} {r['arguments']}" for r in aw_records]
    aw_vectorizer, aw_classifier = fit_on(aw_texts, aw_y)

    tau2_vocab = set(vectorizer.vocabulary_)
    aw_vocab = set(aw_vectorizer.vocabulary_)
    overlap = tau2_vocab & aw_vocab
    print(f"\nvocabulary: tau2 {len(tau2_vocab)}, appworld {len(aw_vocab)}, shared {len(overlap)}")
    if overlap:
        print(f"  shared examples: {sorted(overlap)[:8]}")

    tau2_on_aw = roc_auc_score(aw_y, classifier.predict_proba(vectorizer.transform(aw_texts))[:, 1])
    aw_on_tau2 = roc_auc_score(
        y[test],
        aw_classifier.predict_proba(aw_vectorizer.transform([rows[i]["otel_zero_config"] for i in test]))[:, 1],
    )
    print(f"  tau2-trained scored on appworld : {tau2_on_aw:.4f}")
    print(f"  appworld-trained scored on tau2 : {aw_on_tau2:.4f}")
    print("  chance is 0.5; a transferable detector would be well above it", flush=True)

    empty_aw = sum(1 for t in aw_texts if not (set(t.split()) & tau2_vocab))
    print(f"  appworld trajectories with zero known features: {empty_aw}/{len(aw_texts)}")

    report = {
        "latency": latency,
        "latency_batched_per_trajectory_ms": round(batched_ms, 3),
        "latency_sample": {"n": len(raw), "median_messages": sorted(turns)[len(turns) // 2]},
        "cross_benchmark": {
            "tau2_vocab": len(tau2_vocab),
            "appworld_vocab": len(aw_vocab),
            "shared_features": len(overlap),
            "shared_examples": sorted(overlap)[:20],
            "tau2_model_on_appworld_auroc": round(float(tau2_on_aw), 4),
            "appworld_model_on_tau2_auroc": round(float(aw_on_tau2), 4),
            "appworld_rows_with_no_known_feature": empty_aw,
            "appworld_rows": len(aw_texts),
        },
        "provenance": {
            "data_dir": str(args.data),
            "appworld_dir": str(args.appworld),
            "message_counts": dict(Counter(min(t, 60) for t in turns).most_common(5)),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
