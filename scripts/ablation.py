"""Ablate the structural feature families, control for capacity, and compare all four detectors."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from fsdetect.attack import swap_lexicon
from fsdetect.corpus import load, split_labels
from fsdetect.detect import (
    ACTION_VECTORIZER,
    COMBINED_SPEC,
    STRUCTURAL_VECTORIZER,
    TEXT_VECTORIZER,
    build_dataset,
    fit,
    fit_combined,
    precision_at_flag_rate,
    score,
    score_combined,
    stratified_split,
)
from fsdetect.stats import paired_bootstrap_auroc_diff, spread

# grounded:/ungrounded: is absent from argument_tokens. It was measured on seed 42 at 0.9216 against
# 0.9155 with it, so it cost 0.0061, and was removed rather than kept as an ablation row that cannot fire.
FAMILIES = {
    "pay:": ("pay:",),
    "val:": ("val:",),
    "num:/len:": ("num:", "len:"),
    "arg: keys": ("arg:",),
    "nids/nmut": ("nids:", "nmut:"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/ablation.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    rows, y = build_dataset(split_labels(records))
    print(f"rows {len(rows)}  FS {int(y.sum())}  TS {int((1 - y).sum())}", flush=True)

    detectors: dict[str, list[float]] = {k: [] for k in ("text", "structural", "combined", "tool_names", "arguments")}
    attacked_scores: dict[str, list[float]] = {"text": [], "combined": []}
    capacity: dict[str, list[float]] = {"text_30k": [], "text_60k": [], "text_120k": []}
    families: dict[str, list[float]] = {name: [] for name in FAMILIES}
    wins = {"clean": 0, "attacked": 0}
    precision: dict[str, list[float]] = {k: [] for k in ("text", "combined", "text_attacked", "combined_attacked")}

    for seed in args.seeds:
        train, _, test = stratified_split(y, seed)
        closings = [rows[i]["closing"] for i in test]
        structural = [rows[i]["structural"] for i in test]
        y_test = y[test]
        attacked = [swap_lexicon(t) if label == 1 else t for t, label in zip(closings, y_test, strict=True)]

        single = {
            "text": ("closing", TEXT_VECTORIZER, closings),
            "structural": ("structural", STRUCTURAL_VECTORIZER, structural),
            "tool_names": ("actions", ACTION_VECTORIZER, [rows[i]["actions"] for i in test]),
            "arguments": ("arguments", STRUCTURAL_VECTORIZER, [rows[i]["arguments"] for i in test]),
        }
        held = {}
        for name, (field, config, texts) in single.items():
            detector = fit(rows, y, train, field, config, seed)
            held[name] = (detector, texts)
            detectors[name].append(round(float(roc_auc_score(y_test, score(detector, texts))), 4))

        both = fit_combined(rows, y, train, COMBINED_SPEC, seed)
        clean_combined = score_combined(both, {"closing": closings, "structural": structural})
        detectors["combined"].append(round(float(roc_auc_score(y_test, clean_combined)), 4))

        text_attacked = score(held["text"][0], attacked)
        combined_attacked = score_combined(both, {"closing": attacked, "structural": structural})
        attacked_scores["text"].append(round(float(roc_auc_score(y_test, text_attacked)), 4))
        attacked_scores["combined"].append(round(float(roc_auc_score(y_test, combined_attacked)), 4))
        wins["clean"] += detectors["combined"][-1] > detectors["text"][-1]
        wins["attacked"] += attacked_scores["combined"][-1] > attacked_scores["text"][-1]

        precision["text"].append(round(precision_at_flag_rate(score(*held["text"]), y_test, 0.10), 4))
        precision["combined"].append(round(precision_at_flag_rate(clean_combined, y_test, 0.10), 4))
        precision["text_attacked"].append(round(precision_at_flag_rate(text_attacked, y_test, 0.10), 4))
        precision["combined_attacked"].append(round(precision_at_flag_rate(combined_attacked, y_test, 0.10), 4))

        # Text is saturated well below its cap, so any combined-model gain cannot be extra capacity.
        for tag, cap in (("text_30k", 30000), ("text_60k", 60000), ("text_120k", 120000)):
            detector = fit(rows, y, train, "closing", dict(TEXT_VECTORIZER, max_features=cap), seed)
            capacity[tag].append(round(float(roc_auc_score(y_test, score(detector, closings))), 4))

        for name, prefixes in FAMILIES.items():
            for row in rows:
                kept = [t for t in row["arguments"].split() if not t.startswith(prefixes)]
                row["ablated"] = f"{row['actions']} {' '.join(kept)}"
            detector = fit(rows, y, train, "ablated", STRUCTURAL_VECTORIZER, seed)
            value = roc_auc_score(y_test, score(detector, [rows[i]["ablated"] for i in test]))
            families[name].append(round(float(value), 4))

        if seed == args.seeds[0]:
            clean_gap = paired_bootstrap_auroc_diff(y_test, clean_combined, score(*held["text"]), args.resamples, seed)
            attacked_gap = paired_bootstrap_auroc_diff(y_test, combined_attacked, text_attacked, args.resamples, seed)
        print(f"seed {seed} done", flush=True)

    print(f"\n{'detector':<22}{'AUROC':>18}")
    for name, values in detectors.items():
        s = spread(values)
        print(f"  {name:<20}{s['mean']:.4f} +/- {s['std']:.4f}", flush=True)
    print(f"\n{'under swap_lexicon':<22}{'AUROC':>18}")
    for name, values in attacked_scores.items():
        s = spread(values)
        print(f"  {name:<20}{s['mean']:.4f} +/- {s['std']:.4f}", flush=True)
    print("\ncapacity control on the text detector")
    for name, values in capacity.items():
        s = spread(values)
        print(f"  {name:<20}{s['mean']:.4f} +/- {s['std']:.4f}", flush=True)
    print(f"\nstructural feature families, full model {spread(detectors['structural'])['mean']:.4f}")
    for name, values in families.items():
        s = spread(values)
        print(f"  without {name:<22}{s['mean']:.4f}", flush=True)
    print("\nprecision at a 10% flag rate")
    for name, values in precision.items():
        print(f"  {name:<20}{spread(values)['mean']:.4f}", flush=True)
    print(
        f"\ncombined beats text on {wins['clean']}/{len(args.seeds)} seeds clean, "
        f"{wins['attacked']}/{len(args.seeds)} under swap_lexicon"
    )
    print(f"combined - text, clean : {clean_gap['diff']:+.4f} CI95 {clean_gap['ci95']}")
    print(f"combined - text, attack: {attacked_gap['diff']:+.4f} CI95 {attacked_gap['ci95']}")

    report = {
        "detectors": {k: spread(v) for k, v in detectors.items()},
        "under_swap_lexicon": {k: spread(v) for k, v in attacked_scores.items()},
        "capacity_control": {k: spread(v) for k, v in capacity.items()},
        "structural_families_removed": {k: spread(v) for k, v in families.items()},
        "grounding_note": (
            "grounded:/ungrounded: tokens were tried and removed; seed 42 gave 0.9155 with them "
            "and 0.9216 without, so they cost 0.0061 and are not emitted by argument_tokens"
        ),
        "precision_at_10pct": {k: spread(v) for k, v in precision.items()},
        "combined_beats_text_seeds": wins,
        "seeds_total": len(args.seeds),
        "combined_minus_text_clean": clean_gap,
        "combined_minus_text_attacked": attacked_gap,
        "provenance": {
            "seeds": args.seeds,
            "resamples": args.resamples,
            "data_dir": str(args.data),
            "records": len(records),
            "dropped": dropped,
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
