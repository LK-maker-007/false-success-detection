"""Break the text detector with templated confident-completion rewrites, and measure the crossover."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path

import numpy as np
import sklearn

from fsdetect.attack import STYLES, rewrite
from fsdetect.corpus import load, split_labels
from fsdetect.detect import (
    ACTION_VECTORIZER,
    TEXT_VECTORIZER,
    auroc,
    build_dataset,
    fit,
    precision_at_flag_rate,
    score,
    stratified_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flag-rate", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("results/adversarial.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    labels = split_labels(records)
    rows, y = build_dataset(labels)
    train, _, test = stratified_split(y, args.seed)

    text_detector = fit(rows, y, train, "closing", TEXT_VECTORIZER, args.seed)
    action_detector = fit(rows, y, train, "actions", ACTION_VECTORIZER, args.seed)

    test_closings = [rows[i]["closing"] for i in test]
    test_actions = [rows[i]["actions"] for i in test]
    y_test = y[test]

    clean = {
        "text": round(auroc(text_detector, test_closings, y_test), 4),
        "structural": round(auroc(action_detector, test_actions, y_test), 4),
    }
    print(f"clean  text {clean['text']:.4f}  structural {clean['structural']:.4f}", flush=True)

    # Attack A rewrites honest failures. Reward is untouched, so every rewritten trajectory is still
    # a failure whose closing now asserts completion, which a working detector should flag.
    honest = labels["HF"]
    baseline_flags = score(text_detector, [r["closing"] for r in honest])
    attack_a = {"unmodified": round(float((baseline_flags > 0.5).mean()), 4)}
    for style in STYLES:
        rng = random.Random(args.seed)
        flipped = score(text_detector, [rewrite(r["closing"], style, rng) for r in honest])
        attack_a[style] = round(float((flipped > 0.5).mean()), 4)
        print(f"attack A  {style:<18} flip {attack_a[style]:.1%}", flush=True)

    attack_b: dict[str, dict[str, float]] = {}
    for style in STYLES:
        rng = random.Random(args.seed)
        attacked = [
            rewrite(t, style, rng) if label == 1 else t for t, label in zip(test_closings, y_test, strict=True)
        ]
        probabilities = score(text_detector, attacked)
        attack_b[style] = {
            "text_auroc": round(auroc(text_detector, attacked, y_test), 4),
            "text_precision_at_flag_rate": round(precision_at_flag_rate(probabilities, y_test, args.flag_rate), 4),
            "distinct_rewrites": len(set(attacked)),
        }
        print(f"attack B  {style:<18} AUROC {attack_b[style]['text_auroc']:.4f}", flush=True)

    worst = min(v["text_auroc"] for v in attack_b.values())
    crossed = worst < clean["structural"]
    print(
        f"\nworst text {worst:.4f}  structural clean {clean['structural']:.4f}  "
        f"{'CROSSOVER' if crossed else 'NO CROSSOVER'}",
        flush=True,
    )

    report = {
        "clean": clean,
        "attack_a_flip_rate": attack_a,
        "attack_b_auroc": attack_b,
        "worst_text_auroc": worst,
        "crossover": crossed,
        # The structural detector reads no message text, so its score under both attacks is
        # identical to its clean score by construction. It is a control, not a measurement.
        "structural_invariant_by_construction": True,
        "provenance": {
            "seed": args.seed,
            "data_dir": str(args.data),
            "records": len(records),
            "dropped": dropped,
            "counts": {k: len(v) for k, v in labels.items()},
            "attack_a_population": len(honest),
            "attack_b_positives": int(y_test.sum()),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
