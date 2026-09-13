"""Score the detector using only fields OpenTelemetry GenAI carries, the deployment feature set."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from fsdetect.corpus import MUTATING_TOOLS, load, split_labels
from fsdetect.detect import (
    STRUCTURAL_VECTORIZER,
    build_dataset,
    fit,
    precision_at_flag_rate,
    score,
    stratified_split,
)
from fsdetect.stats import delong_test, holm_bonferroni, spread

# What each variant is allowed to read.
VARIANTS = {
    "full_structural": "tau2-native: requestor, final role, hardcoded mutating list, arguments",
    "otel_with_tool_list": "gen_ai.tool.name, gen_ai.tool.call.arguments, error status, caller's mutating list",
    "otel_zero_config": "the same three fields, no mutating list, every call scored",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--output", type=Path, default=Path("results/portability.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    labels = split_labels(records)
    rows, y = build_dataset(labels)
    print(f"rows {len(rows)}  FS {int(y.sum())}  TS {int((1 - y).sum())}", flush=True)

    runs: dict[str, list[float]] = {k: [] for k in VARIANTS}
    precision: dict[str, list[float]] = {k: [] for k in VARIANTS}
    held: dict[str, tuple] = {}
    for seed in args.seeds:
        train, _, test = stratified_split(y, seed)
        for name in VARIANTS:
            field = "structural" if name == "full_structural" else name
            detector = fit(rows, y, train, field, STRUCTURAL_VECTORIZER, seed)
            scores = score(detector, [rows[i][field] for i in test])
            runs[name].append(round(float(roc_auc_score(y[test], scores)), 4))
            precision[name].append(round(precision_at_flag_rate(scores, y[test], 0.10), 4))
            if seed == args.seeds[0]:
                held[name] = (scores, y[test])

    print(f"\n{'variant':<24}{'AUROC':>20}{'P@10%':>9}")
    for name, values in runs.items():
        s = spread(values)
        print(f"  {name:<22}{s['mean']:.4f} +/- {s['std']:.4f}{spread(precision[name])['mean']:>9.4f}", flush=True)

    y_first = held["full_structural"][1]
    tests = {
        "full_vs_otel_with_tool_list": ("full_structural", "otel_with_tool_list"),
        "full_vs_otel_zero_config": ("full_structural", "otel_zero_config"),
        "otel_with_tool_list_vs_zero_config": ("otel_with_tool_list", "otel_zero_config"),
    }
    results = {k: delong_test(y_first, held[a][0], held[b][0]) for k, (a, b) in tests.items()}
    holm = holm_bonferroni({k: v["p"] for k, v in results.items()})
    print(f"\nDeLong on the confirmatory split, seed {args.seeds[0]}, Holm across {len(tests)}")
    print(f"  {'comparison':<38}{'diff':>9}{'p':>11}{'p_holm':>11}  sig")
    for name, result in results.items():
        flag = "yes" if holm[name]["significant_at_alpha"] else "NO"
        print(
            f"  {name:<38}{result['diff']:>+9.4f}{result['p']:>11.2e}{holm[name]['p_holm']:>11.2e}  {flag}", flush=True
        )

    report = {
        "variants": VARIANTS,
        "auroc": {k: spread(v) for k, v in runs.items()},
        "precision_at_10pct": {k: spread(v) for k, v in precision.items()},
        "delong_holm": {
            k: {**v, "p_holm": holm[k]["p_holm"], "significant_at_0.05": holm[k]["significant_at_alpha"]}
            for k, v in results.items()
        },
        "mutating_tool_list_size": len(MUTATING_TOOLS),
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
