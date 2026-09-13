"""Test whether call arguments beat endpoints alone on AppWorld, the way they do on tau2."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from fsdetect.appworld import build_dataset, load
from fsdetect.detect import precision_at_flag_rate
from fsdetect.stats import paired_bootstrap_auroc_diff, spread

VECTORIZER = {
    "ngram_range": (1, 2),
    "max_features": 30000,
    "min_df": 2,
    "sublinear_tf": True,
    "token_pattern": r"\S+",
    "lowercase": False,
}


def evaluate(records: list[dict], y: np.ndarray, fields: tuple[str, ...], seed: int, groups: np.ndarray):
    # Split by submission, not by task. Tasks recur across submissions, so a random split would put
    # the same task's trace in train and test and inflate every number.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train, test = next(splitter.split(np.arange(len(y)), y, groups))
    vectorizer = TfidfVectorizer(**VECTORIZER)
    x_train = vectorizer.fit_transform([" ".join(records[i][f] for f in fields) for i in train])
    x_test = vectorizer.transform([" ".join(records[i][f] for f in fields) for i in test])
    classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=seed)
    classifier.fit(x_train, y[train])
    scores = classifier.predict_proba(x_test)[:, 1]
    return roc_auc_score(y[test], scores), scores, y[test]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/appworld/leaderboard/experiments/outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/appworld.json"))
    args = parser.parse_args()

    report: dict[str, object] = {}
    for explicit in (True, False):
        tag = "explicit_status" if explicit else "any_completion_claim"
        records, dropped = load(args.data, require_explicit_status=explicit)
        records, y_list = build_dataset(records)
        y = np.array(y_list)
        groups = np.array([r["submission"] for r in records])
        print(f"\n=== {tag} ===", flush=True)
        print(
            f"tasks {len(records)}  FS {int(y.sum())}  TS {int((1 - y).sum())}  submissions {len(set(groups))}",
            flush=True,
        )

        runs: dict[str, list[float]] = {"endpoints": [], "arguments": [], "both": []}
        precision: dict[str, list[float]] = {"endpoints": [], "both": []}
        held: dict[str, tuple] = {}
        for seed in args.seeds:
            for name, fields in (
                ("endpoints", ("endpoints",)),
                ("arguments", ("arguments",)),
                ("both", ("endpoints", "arguments")),
            ):
                auc, scores, y_test = evaluate(records, y, fields, seed, groups)
                runs[name].append(round(float(auc), 4))
                if name in precision:
                    precision[name].append(round(precision_at_flag_rate(scores, y_test, 0.10), 4))
                if seed == args.seeds[0]:
                    held[name] = (scores, y_test)

        wins = sum(1 for i in range(len(args.seeds)) if runs["both"][i] > runs["endpoints"][i])
        gap = paired_bootstrap_auroc_diff(
            held["both"][1], held["both"][0], held["endpoints"][0], args.resamples, args.seeds[0]
        )
        for name, values in runs.items():
            s = spread(values)
            print(f"  {name:<12}{s['mean']:.4f} +/- {s['std']:.4f}   [{s['min']:.4f}, {s['max']:.4f}]", flush=True)
        print(f"  both beats endpoints on {wins}/{len(args.seeds)} seeds", flush=True)
        print(
            f"  both - endpoints {gap['diff']:+.4f} CI95 {gap['ci95']} excludes zero {gap['excludes_zero']}",
            flush=True,
        )
        for name, values in precision.items():
            print(f"  P@10% {name:<10}{spread(values)['mean']:.4f}", flush=True)

        report[tag] = {
            "tasks": len(records),
            "fs": int(y.sum()),
            "ts": int((1 - y).sum()),
            "submissions": len(set(groups)),
            "dropped": dropped,
            "auroc": {k: spread(v) for k, v in runs.items()},
            "precision_at_10pct": {k: spread(v) for k, v in precision.items()},
            "both_beats_endpoints_seeds": wins,
            "seeds_total": len(args.seeds),
            "both_minus_endpoints": gap,
        }

    report["provenance"] = {
        "seeds": args.seeds,
        "resamples": args.resamples,
        "data_dir": str(args.data),
        "split": "GroupShuffleSplit by submission",
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
