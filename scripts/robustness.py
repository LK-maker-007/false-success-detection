"""Re-run the detectors and every attack across seeds, and put an interval on the crossover claim."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from fsdetect.attack import DEGENERATE_RE, STYLES, assert_completion, rewrite, swap_lexicon
from fsdetect.corpus import load, split_labels
from fsdetect.detect import (
    COMBINED_SPEC,
    STRUCTURAL_VECTORIZER,
    TEXT_VECTORIZER,
    build_dataset,
    fit,
    fit_combined,
    score,
    score_combined,
    stratified_split,
)
from fsdetect.stats import bootstrap_auroc, mcnemar_exact, paired_bootstrap_auroc_diff, spread

# Templates replace the message; swap_lexicon edits it and keeps every fact. The templates are the
# weak floor, kept so the two can be compared rather than because they are the realistic attack.
TRANSFORMS: dict[str, object] = {
    **{style: (lambda text, rng, s=style: rewrite(text, s, rng)) for style in STYLES},
    "swap_lexicon": lambda text, rng: swap_lexicon(text),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/robustness.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    labels = split_labels(records)
    rows, y = build_dataset(labels)

    # 86.5% of HF is the harness transfer token, which carries no content to rewrite. Attack A runs
    # on the substantive remainder so the flip rate is not dominated by one repeated string.
    honest = [r for r in labels["HF"] if not DEGENERATE_RE.search(r["closing"])]
    print(f"records {len(records)}  FS {len(labels['FS'])}  TS {len(labels['TS'])}", flush=True)
    print(f"attack A population: {len(honest)} substantive HF of {len(labels['HF'])}", flush=True)

    runs: dict[str, list[float]] = {"text_clean": [], "structural_clean": [], "combined_clean": []}
    runs.update({f"text_{name}": [] for name in TRANSFORMS})
    runs.update({f"combined_{name}": [] for name in TRANSFORMS})
    runs["flip_assert_completion"] = []
    runs.update({f"flip_{style}": [] for style in STYLES})
    intervals: dict[str, object] = {}

    for seed in args.seeds:
        train, _, test = stratified_split(y, seed)
        text = fit(rows, y, train, "closing", TEXT_VECTORIZER, seed)
        actions = fit(rows, y, train, "structural", STRUCTURAL_VECTORIZER, seed)
        both = fit_combined(rows, y, train, COMBINED_SPEC, seed)

        closings = [rows[i]["closing"] for i in test]
        y_test = y[test]
        text_scores = score(text, closings)
        action_scores = score(actions, [rows[i]["structural"] for i in test])
        runs["text_clean"].append(round(float(roc_auc_score(y_test, text_scores)), 4))
        runs["structural_clean"].append(round(float(roc_auc_score(y_test, action_scores)), 4))
        structural_text = [rows[i]["structural"] for i in test]
        combined_clean = score_combined(both, {"closing": closings, "structural": structural_text})
        runs["combined_clean"].append(round(float(roc_auc_score(y_test, combined_clean)), 4))

        worst_name, worst_scores, worst_auroc = None, None, 1.0
        for name, transform in TRANSFORMS.items():
            rng = random.Random(seed)
            attacked = [transform(t, rng) if label == 1 else t for t, label in zip(closings, y_test, strict=True)]
            attacked_scores = score(text, attacked)
            value = round(float(roc_auc_score(y_test, attacked_scores)), 4)
            runs[f"text_{name}"].append(value)
            # The adversary controls the closing message, not the tool calls, so only the text
            # column is rewritten and the structural column is passed through untouched.
            combined_attacked = score_combined(both, {"closing": attacked, "structural": structural_text})
            runs[f"combined_{name}"].append(round(float(roc_auc_score(y_test, combined_attacked)), 4))
            if value < worst_auroc:
                worst_name, worst_scores, worst_auroc = name, attacked_scores, value

        for style in STYLES:
            rng = random.Random(seed)
            rewritten = [rewrite(r["closing"], style, rng) for r in honest]
            runs[f"flip_{style}"].append(round(float((score(text, rewritten) > 0.5).mean()), 4))
        surgical = [assert_completion(r["closing"]) for r in honest]
        runs["flip_assert_completion"].append(round(float((score(text, surgical) > 0.5).mean()), 4))

        if seed == args.seeds[0]:
            # Both detectors score the same test rows, so the crossover margin is paired and gets a
            # paired interval rather than two independent ones.
            intervals["text_clean"] = bootstrap_auroc(y_test, text_scores, args.resamples, seed)
            intervals["structural_clean"] = bootstrap_auroc(y_test, action_scores, args.resamples, seed)
            intervals[f"text_{worst_name}"] = bootstrap_auroc(y_test, worst_scores, args.resamples, seed)
            intervals["crossover_margin"] = paired_bootstrap_auroc_diff(
                y_test, action_scores, worst_scores, args.resamples, seed
            )
            intervals["worst_transform"] = worst_name
            baseline_flags = score(text, [r["closing"] for r in honest]) > 0.5
            intervals["attack_a_mcnemar_surgical_vs_unmodified"] = mcnemar_exact(
                score(text, surgical) > 0.5, baseline_flags
            )
            intervals["attack_a_unmodified_flip_rate"] = round(float(baseline_flags.mean()), 4)

        print(
            f"seed {seed}  text {runs['text_clean'][-1]:.4f}  structural {runs['structural_clean'][-1]:.4f}  "
            f"worst {worst_name} {worst_auroc:.4f}",
            flush=True,
        )

    summary = {key: spread(values) for key, values in runs.items()}
    crossovers = sum(
        1
        for i in range(len(args.seeds))
        if min(runs[f"text_{n}"][i] for n in TRANSFORMS) < runs["structural_clean"][i]
    )
    swap_crossovers = sum(
        1 for i in range(len(args.seeds)) if runs["text_swap_lexicon"][i] < runs["structural_clean"][i]
    )

    print()
    for key in (
        "text_clean",
        "structural_clean",
        "combined_clean",
        *[f"text_{n}" for n in TRANSFORMS],
        *[f"combined_{n}" for n in TRANSFORMS],
    ):
        s = summary[key]
        print(f"{key:<28} {s['mean']:.4f} +/- {s['std']:.4f}   [{s['min']:.4f}, {s['max']:.4f}]", flush=True)
    print()
    for key in ("flip_assert_completion", *[f"flip_{s}" for s in STYLES]):
        s = summary[key]
        print(f"{key:<28} {s['mean']:.1%} +/- {s['std']:.1%}", flush=True)
    print(f"\ncrossover, any transform:  {crossovers}/{len(args.seeds)} seeds", flush=True)
    print(f"crossover, swap_lexicon:   {swap_crossovers}/{len(args.seeds)} seeds", flush=True)
    print(
        f"margin {intervals['crossover_margin']['diff']:+.4f} "
        f"CI95 {intervals['crossover_margin']['ci95']} "
        f"excludes zero: {intervals['crossover_margin']['excludes_zero']}",
        flush=True,
    )

    report = {
        "across_seeds": summary,
        "crossover_seeds_any_transform": crossovers,
        "crossover_seeds_swap_lexicon": swap_crossovers,
        "crossover_seeds_total": len(args.seeds),
        "intervals_on_first_seed": intervals,
        "provenance": {
            "seeds": args.seeds,
            "resamples": args.resamples,
            "data_dir": str(args.data),
            "records": len(records),
            "dropped": dropped,
            "counts": {k: len(v) for k, v in labels.items()},
            "attack_a_population": len(honest),
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
