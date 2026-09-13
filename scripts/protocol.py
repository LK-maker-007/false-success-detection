"""Compare a row-random split against a task-disjoint one, and test the claim under each."""

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
    build_dataset,
    fit,
    fit_combined,
    precision_at_flag_rate,
    random_split,
    score,
    score_combined,
    split_balance,
    task_disjoint_split,
    task_groups,
)
from fsdetect.stats import delong_test, holm_bonferroni, spread

DETECTORS = {
    "text": ("closing", TEXT_VECTORIZER),
    "tool_names": ("actions", ACTION_VECTORIZER),
    "arguments": ("arguments", STRUCTURAL_VECTORIZER),
    "structural": ("structural", STRUCTURAL_VECTORIZER),
}

# The comparison the project's claim actually names is arguments vs tool_names. It was never run.
COMPARISONS = {
    "arguments_vs_tool_names": ("arguments", "tool_names"),
    "structural_vs_tool_names": ("structural", "tool_names"),
    "structural_vs_arguments": ("structural", "arguments"),
    "combined_vs_text": ("combined", "text"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--output", type=Path, default=Path("results/protocol.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    rows, y = build_dataset(split_labels(records))
    groups = task_groups(rows)
    print(f"rows {len(rows)}  distinct tasks {len(set(groups))}  positives {int(y.sum())} ({y.mean():.1%})")

    protocols = {
        "random_row": lambda seed: random_split(y, seed),
        "task_disjoint": lambda seed: task_disjoint_split(rows, y, seed),
    }
    auroc: dict[str, dict[str, list[float]]] = {p: {d: [] for d in [*DETECTORS, "combined"]} for p in protocols}
    precision: dict[str, dict[str, list[float]]] = {p: {d: [] for d in [*DETECTORS, "combined"]} for p in protocols}
    leakage: dict[str, list[int]] = {p: [] for p in protocols}
    balance: dict[str, dict] = {}
    held: dict[str, dict[str, np.ndarray]] = {}
    y_held: dict[str, np.ndarray] = {}

    for protocol, make_split in protocols.items():
        for seed in args.seeds:
            train, val, test = make_split(seed)
            leakage[protocol].append(len(set(groups[train]) & set(groups[test])))
            y_test = y[test]
            scores: dict[str, np.ndarray] = {}
            for name, (field, config) in DETECTORS.items():
                detector = fit(rows, y, train, field, config, seed)
                scores[name] = score(detector, [rows[i][field] for i in test])
            both = fit_combined(rows, y, train, COMBINED_SPEC, seed)
            scores["combined"] = score_combined(both, {f: [rows[i][f] for i in test] for f in COMBINED_SPEC})

            for name, values in scores.items():
                auroc[protocol][name].append(round(float(roc_auc_score(y_test, values)), 4))
                precision[protocol][name].append(round(precision_at_flag_rate(values, y_test, 0.10, seed), 4))
            if seed == args.seeds[0]:
                held[protocol] = scores
                y_held[protocol] = y_test
                balance[protocol] = split_balance(y, {"train": train, "val": val, "test": test})
        print(f"{protocol}: tasks shared train/test {set(leakage[protocol])}", flush=True)

    print(f"\n{'detector':<14}{'random row':>22}{'task disjoint':>22}{'delta':>10}")
    for name in [*DETECTORS, "combined"]:
        a, b = spread(auroc["random_row"][name]), spread(auroc["task_disjoint"][name])
        print(
            f"  {name:<12}{a['mean']:>10.4f} +/- {a['std']:.4f}{b['mean']:>12.4f} +/- {b['std']:.4f}"
            f"{b['mean'] - a['mean']:>+10.4f}",
            flush=True,
        )

    tests: dict[str, dict] = {}
    for protocol in protocols:
        raw = {
            name: delong_test(y_held[protocol], held[protocol][a], held[protocol][b])
            for name, (a, b) in COMPARISONS.items()
        }
        holm = holm_bonferroni({k: v["p"] for k, v in raw.items()})
        tests[protocol] = {
            name: {
                **result,
                "p_holm": holm[name]["p_holm"],
                "significant_at_0.05": holm[name]["significant_at_alpha"],
                "diff_across_seeds": spread(
                    [
                        a - b
                        for a, b in zip(
                            auroc[protocol][COMPARISONS[name][0]],
                            auroc[protocol][COMPARISONS[name][1]],
                            strict=True,
                        )
                    ]
                ),
                "wins_across_seeds": sum(
                    1
                    for a, b in zip(
                        auroc[protocol][COMPARISONS[name][0]],
                        auroc[protocol][COMPARISONS[name][1]],
                        strict=True,
                    )
                    if a > b
                ),
            }
            for name, result in raw.items()
        }
        print(f"\nDeLong under {protocol}, confirmatory seed {args.seeds[0]}, Holm across {len(raw)}")
        print(f"  {'comparison':<28}{'diff':>9}{'p_holm':>11}{'sig':>6}{'wins':>7}{'across seeds':>20}")
        for name, row in tests[protocol].items():
            across = row["diff_across_seeds"]
            print(
                f"  {name:<28}{row['diff']:>+9.4f}{row['p_holm']:>11.2e}"
                f"{'yes' if row['significant_at_0.05'] else 'NO':>6}"
                f"{row['wins_across_seeds']:>4}/{len(args.seeds)}"
                f"{across['mean']:>+13.4f} +/- {across['std']:.4f}",
                flush=True,
            )

    report = {
        "auroc": {p: {d: spread(v) for d, v in det.items()} for p, det in auroc.items()},
        "precision_at_10pct": {p: {d: spread(v) for d, v in det.items()} for p, det in precision.items()},
        "tasks_shared_train_test": {p: sorted(set(v)) for p, v in leakage.items()},
        "split_balance_first_seed": balance,
        "delong_holm": tests,
        "distinct_tasks": len(set(groups)),
        "provenance": {
            "seeds": args.seeds,
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
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
