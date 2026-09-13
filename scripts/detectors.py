"""Train the text and structural false-success detectors and report clean AUROC."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from fsdetect.corpus import load, split_labels
from fsdetect.detect import (
    ACTION_VECTORIZER,
    COMBINED_SPEC,
    STRUCTURAL_VECTORIZER,
    TEXT_VECTORIZER,
    auroc,
    build_dataset,
    fit,
    fit_combined,
    precision_at_flag_rate,
    score,
    score_combined,
    stratified_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flag-rate", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("results/detectors.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    labels = split_labels(records)
    rows, y = build_dataset(labels)
    train, val, test = stratified_split(y, args.seed)
    print(f"records {len(records)}  FS {len(labels['FS'])}  TS {len(labels['TS'])}", flush=True)
    print(f"HF {len(labels['HF'])}  ambiguous {len(labels['AMB'])}  dropped {dropped}", flush=True)
    print(f"train {len(train)}  val {len(val)}  test {len(test)}", flush=True)

    detectors = {
        "text": ("closing", TEXT_VECTORIZER),
        "structural": ("structural", STRUCTURAL_VECTORIZER),
        "tool_names_only": ("actions", ACTION_VECTORIZER),
        "arguments_only": ("arguments", STRUCTURAL_VECTORIZER),
    }
    report: dict[str, object] = {}
    for name, (field, config) in detectors.items():
        detector = fit(rows, y, train, field, config, args.seed)
        texts = [rows[i][field] for i in test]
        probabilities = score(detector, texts)
        report[name] = {
            "field": field,
            "auroc": round(auroc(detector, texts, y[test]), 4),
            "precision_at_flag_rate": round(precision_at_flag_rate(probabilities, y[test], args.flag_rate), 4),
            "vocabulary": len(detector[0].vocabulary_),
        }
        print(
            f"{name:<12} AUROC {report[name]['auroc']:.4f}  "
            f"P@{args.flag_rate:.0%} {report[name]['precision_at_flag_rate']:.4f}",
            flush=True,
        )

    both = fit_combined(rows, y, train, COMBINED_SPEC, args.seed)
    combined_scores = score_combined(
        both,
        {field: [rows[i][field] for i in test] for field in COMBINED_SPEC},
    )
    report["combined"] = {
        "field": "+".join(COMBINED_SPEC),
        "auroc": round(float(roc_auc_score(y[test], combined_scores)), 4),
        "precision_at_flag_rate": round(precision_at_flag_rate(combined_scores, y[test], args.flag_rate), 4),
        "vocabulary": sum(len(v.vocabulary_) for _, v in both[0]),
    }
    print(
        f"{'combined':<12} AUROC {report['combined']['auroc']:.4f}  "
        f"P@{args.flag_rate:.0%} {report['combined']['precision_at_flag_rate']:.4f}",
        flush=True,
    )

    report["provenance"] = {
        "seed": args.seed,
        "data_dir": str(args.data),
        "trajectory_files": len(sorted(args.data.rglob("*.json"))) - 1,
        "records": len(records),
        "dropped": dropped,
        "counts": {k: len(v) for k, v in labels.items()},
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
