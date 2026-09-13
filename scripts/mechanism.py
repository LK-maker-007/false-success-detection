"""Test what false success is: a claim without a matching action, or the right action done wrong."""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

from fsdetect.corpus import claim_is_unsupported, load, split_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--output", type=Path, default=Path("results/mechanism.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    labels = split_labels(records)

    # If false success were a claim the agent never acted on, the unsupported-claim rate would be
    # far higher for FS than for TS. It is lower.
    consistency = {}
    for name in ("FS", "TS"):
        rows = labels[name]
        verdicts = [claim_is_unsupported(r["closing"], r["succeeded"]) for r in rows]
        claiming = [v for v in verdicts if v is not None]
        unsupported = sum(claiming)
        consistency[name] = {
            "trajectories": len(rows),
            "claiming_an_operation": len(claiming),
            "claim_without_matching_call": unsupported,
            "unsupported_rate": round(unsupported / len(claiming), 4) if claiming else None,
        }
    ratio = consistency["FS"]["unsupported_rate"] / consistency["TS"]["unsupported_rate"]

    print(f"{'class':<6}{'n':>7}{'claims op':>11}{'no match':>10}{'rate':>9}")
    for name, row in consistency.items():
        print(
            f"{name:<6}{row['trajectories']:>7}{row['claiming_an_operation']:>11}"
            f"{row['claim_without_matching_call']:>10}{row['unsupported_rate']:>8.1%}",
            flush=True,
        )
    print(
        f"\nFS rate / TS rate = {ratio:.2f}x  "
        f"{'inverted, claims are supported' if ratio < 1 else 'FS claims less supported'}",
        flush=True,
    )

    # If not an absent action, which reward component actually fails?
    failed_components = Counter()
    checks = Counter()
    for record in labels["FS"]:
        info = record["reward_info"]
        breakdown = info.get("reward_breakdown") or {}
        failed = tuple(sorted(k for k, v in breakdown.items() if v != 1.0)) or ("none_recorded",)
        failed_components[failed] += 1
        if (info.get("db_check") or {}).get("db_match") is False:
            checks["db_check"] += 1
        if any(not a.get("action_match", True) for a in (info.get("action_checks") or [])):
            checks["action_check"] += 1
        if any(not a.get("met", True) for a in (info.get("nl_assertions") or [])):
            checks["nl_assertion"] += 1
        if any(not c.get("met", True) for c in (info.get("communicate_checks") or [])):
            checks["communicate_check"] += 1

    total = len(labels["FS"])
    print(f"\nreward components failing, over {total} false successes")
    for combo, n in failed_components.most_common():
        print(f"  {n:>5} ({n / total:>5.1%})  {'+'.join(combo)}", flush=True)
    print("\nindividual checks failing")
    for name, n in checks.most_common():
        print(f"  {name:<20}{n:>5} ({n / total:>5.1%})", flush=True)

    report = {
        "claim_consistency": consistency,
        "fs_over_ts_unsupported_ratio": round(ratio, 4),
        "reward_components_failing": {"+".join(k): v for k, v in failed_components.most_common()},
        "individual_checks_failing": dict(checks),
        "false_successes": total,
        "provenance": {
            "data_dir": str(args.data),
            "records": len(records),
            "dropped": dropped,
            "counts": {k: len(v) for k, v in labels.items()},
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
