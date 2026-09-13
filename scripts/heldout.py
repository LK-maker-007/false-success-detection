"""Score tau2 detectors on unseen submissions. Note this holds out agents, never tasks."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

from fsdetect.corpus import load, split_labels
from fsdetect.detect import (
    COMBINED_SPEC,
    STRUCTURAL_VECTORIZER,
    TEXT_VECTORIZER,
    build_dataset,
    fit,
    fit_combined,
    precision_at_flag_rate,
    score,
    score_combined,
    task_groups,
)
from fsdetect.stats import spread

SINGLE = {"text": ("closing", TEXT_VECTORIZER), "structural": ("structural", STRUCTURAL_VECTORIZER)}


def evaluate_fold(rows, y, train, test) -> dict[str, float]:
    out = {}
    for name, (field, config) in SINGLE.items():
        detector = fit(rows, y, train, field, config, 0)
        out[name] = float(roc_auc_score(y[test], score(detector, [rows[i][field] for i in test])))
    both = fit_combined(rows, y, train, COMBINED_SPEC, 0)
    scores = score_combined(both, {f: [rows[i][f] for i in test] for f in COMBINED_SPEC})
    out["combined"] = float(roc_auc_score(y[test], scores))
    out["combined_p_at_10"] = precision_at_flag_rate(scores, y[test], 0.10)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--output", type=Path, default=Path("results/heldout.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    rows, y = build_dataset(split_labels(records))
    groups = np.array([r["submission"] for r in rows])
    # tau2 is a complete grid: every one of the 12 submissions runs every one of the 278 tasks.
    # Holding out a submission therefore holds out no task at all, so these numbers measure
    # generalisation to an unseen agent while every task is still seen. Agent-disjoint and
    # task-disjoint cannot both hold on this benchmark; see scripts/protocol.py for the latter.
    tasks = task_groups(rows)
    coverage = {s: len({t for t, g in zip(tasks, groups, strict=True) if g == s}) for s in sorted(set(groups))}
    print(
        f"task coverage per submission: min {min(coverage.values())} "
        f"max {max(coverage.values())} of {len(set(tasks))}",
        flush=True,
    )
    print(f"rows {len(rows)}  FS {int(y.sum())}  submissions {len(set(groups))}", flush=True)

    # Protocol matching AppWorld: 30% of submissions held out, repeated over seeds.
    shuffle_runs: dict[str, list[float]] = {k: [] for k in ("text", "structural", "combined", "combined_p_at_10")}
    for seed in args.seeds:
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed).split(rows, y, groups))
        fold = evaluate_fold(rows, y, train, test)
        for k, v in fold.items():
            shuffle_runs[k].append(round(v, 4))
    print("\nheld out 30% of submissions, 10 seeds")
    for name, values in shuffle_runs.items():
        s = spread(values)
        print(f"  {name:<20}{s['mean']:.4f} +/- {s['std']:.4f}   [{s['min']:.4f}, {s['max']:.4f}]", flush=True)

    # One submission at a time, so every submission is scored as the unseen one.
    loso: dict[str, dict[str, float]] = {}
    skipped = {}
    for train, test in LeaveOneGroupOut().split(rows, y, groups):
        held = groups[test][0]
        if len(set(y[test])) < 2:
            skipped[held] = {"fs": int(y[test].sum()), "ts": int((1 - y[test]).sum())}
            continue
        fold = evaluate_fold(rows, y, train, test)
        fold["n"] = len(test)
        fold["fs"] = int(y[test].sum())
        loso[held] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fold.items()}

    print(f"\nleave one submission out, {len(loso)} scorable of {len(set(groups))}")
    print(f"  {'held out':<44}{'n':>6}{'FS':>5}{'text':>9}{'struct':>9}{'comb':>9}")
    for name, row in sorted(loso.items(), key=lambda kv: kv[1]["combined"]):
        print(
            f"  {name:<44}{row['n']:>6}{row['fs']:>5}{row['text']:>9.4f}"
            f"{row['structural']:>9.4f}{row['combined']:>9.4f}",
            flush=True,
        )
    if skipped:
        print(f"  skipped, one class only: {skipped}")

    for key in ("text", "structural", "combined"):
        s = spread([row[key] for row in loso.values()])
        print(f"\n  LOSO {key:<12}{s['mean']:.4f} +/- {s['std']:.4f}   [{s['min']:.4f}, {s['max']:.4f}]", flush=True)

    # Precision at a fixed flag rate is bounded by the held-out agent's own failure rate, so it
    # moves with base rate rather than with detector quality. This is why a shipped threshold
    # cannot be fixed: it has to be fitted against the deployment's own prevalence.
    rates = np.array([row["fs"] / row["n"] for row in loso.values()])
    precisions = np.array([row["combined_p_at_10"] for row in loso.values()])
    correlation = float(np.corrcoef(rates, precisions)[0, 1])
    print(f"\n  fold FS rate vs P@10% correlation: r = {correlation:.3f}", flush=True)
    print(
        f"  fold FS rate range {rates.min():.1%} to {rates.max():.1%}, "
        f"P@10% range {precisions.min():.4f} to {precisions.max():.4f}",
        flush=True,
    )

    wins = sum(1 for row in loso.values() if row["combined"] > row["text"])
    print(f"\n  combined beats text on {wins}/{len(loso)} held-out submissions")

    report = {
        "group_shuffle_30pct": {k: spread(v) for k, v in shuffle_runs.items()},
        "leave_one_submission_out": loso,
        "loso_summary": {k: spread([r[k] for r in loso.values()]) for k in ("text", "structural", "combined")},
        "loso_fs_rate_vs_precision_r": round(correlation, 4),
        "loso_combined_beats_text": wins,
        "loso_folds": len(loso),
        "skipped_single_class": skipped,
        "protocol": "agent-disjoint, NOT task-disjoint; every held-out submission's tasks appear in training",
        "task_coverage_per_submission": coverage,
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
