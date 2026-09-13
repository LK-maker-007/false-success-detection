"""Measure false-success prevalence on AppWorld, including submissions that log only their verdict."""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

from fsdetect.appworld import completion_claim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/appworld/leaderboard/experiments/outputs"))
    parser.add_argument("--output", type=Path, default=Path("results/appworld_prevalence.json"))
    args = parser.parse_args()

    experiments = sorted(p for p in args.data.iterdir() if (p / "evaluations").is_dir())
    if not experiments:
        raise ValueError(f"{args.data} contains no unpacked experiments")

    per_submission: dict[str, dict] = {}
    overall = Counter()
    for experiment in experiments:
        counts = Counter()
        for evaluation_file in sorted((experiment / "evaluations").glob("*.json")):
            evaluations = json.loads(evaluation_file.read_text(encoding="utf-8"))["individual"]
            for task_id, record in evaluations.items():
                call_file = experiment / "tasks" / task_id / "logs/api_calls.jsonl"
                if not call_file.exists():
                    counts["no_trace"] += 1
                    continue
                calls = [json.loads(line) for line in call_file.read_text(encoding="utf-8").splitlines()]
                _, status = completion_claim(calls)
                if status != "success":
                    counts["no_explicit_success_claim"] += 1
                    continue
                counts["claimed_success"] += 1
                counts["false_success" if not record["success"] else "true_success"] += 1
                counts["failures"] += 0 if record["success"] else 1
            counts["all_failures"] += sum(1 for r in evaluations.values() if not r["success"])

        claimed = counts["claimed_success"]
        if not claimed:
            continue
        per_submission[experiment.name] = {
            "claimed_success": claimed,
            "false_success": counts["false_success"],
            "true_success": counts["true_success"],
            "all_failures": counts["all_failures"],
            "fs_rate_among_success_claims": round(counts["false_success"] / claimed, 4),
            "fs_share_of_all_failures": (
                round(counts["false_success"] / counts["all_failures"], 4) if counts["all_failures"] else None
            ),
        }
        overall.update(counts)

    print(f"{'submission':<58}{'claims':>8}{'FS':>6}{'FS/claims':>11}{'FS/fails':>10}")
    for name, row in sorted(per_submission.items(), key=lambda kv: -kv[1]["claimed_success"]):
        share = row["fs_share_of_all_failures"]
        print(
            f"{name:<58}{row['claimed_success']:>8}{row['false_success']:>6}"
            f"{row['fs_rate_among_success_claims']:>10.1%}"
            f"{(f'{share:.1%}' if share is not None else '-'):>10}",
            flush=True,
        )

    pooled_claimed = overall["claimed_success"]
    pooled = {
        "submissions_with_explicit_claims": len(per_submission),
        "claimed_success": pooled_claimed,
        "false_success": overall["false_success"],
        "fs_rate_among_success_claims": round(overall["false_success"] / pooled_claimed, 4),
        "all_failures": overall["all_failures"],
        "fs_share_of_all_failures": round(overall["false_success"] / overall["all_failures"], 4),
    }
    print(f"\npooled over {pooled['submissions_with_explicit_claims']} submissions with explicit claims")
    print(f"  false success among success claims : {pooled['fs_rate_among_success_claims']:.1%}")
    print(f"  false success as share of failures : {pooled['fs_share_of_all_failures']:.1%}")
    print("  reference reports 75.8% of AppWorld self-assessing trajectories [arXiv 2606.09863]")

    report = {
        "per_submission": per_submission,
        "pooled": pooled,
        "provenance": {"data_dir": str(args.data), "python": platform.python_version()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
