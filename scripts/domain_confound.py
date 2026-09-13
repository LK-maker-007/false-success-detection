"""Test whether telecom's low false-success rate is dual control or a working reward function."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from fsdetect.corpus import load, split_labels

# Issue #384 lists these as scored on inaction alone: the gold action changes no database state and
# the written criteria sit in no reward_basis. Telecom appears nowhere in the issue.
AUTO_PASS_TASKS = {
    "airline": frozenset(
        [
            "0",
            "1",
            "2",
            "4",
            "5",
            "6",
            "9",
            "10",
            "13",
            "26",
            "27",
            "28",
            "31",
            "34",
            "36",
            "38",
            "41",
            "43",
            "45",
            "46",
            "47",
            "48",
            "49",
        ]
    ),
    "retail": frozenset(["10", "12", "25", "50", "57", "65"]),
    "telecom": frozenset(),
}
EMPTY_COMMUNICATE_NOTE = "No communicate_info to evaluate"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--output", type=Path, default=Path("results/domain_confound.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    labels = split_labels(records)

    report: dict[str, object] = {}
    print(
        f"{'domain':<10}{'n':>7}{'failures':>10}{'FS':>6}{'FS/fail':>9}{'FS/all':>8}"
        f"{'empty gate':>12}{'#384 tasks':>12}"
    )
    for domain in ("airline", "retail", "telecom"):
        rows = [r for r in records if r["domain"] == domain]
        failures = [r for r in labels["FS"] + labels["HF"] + labels["AMB"] if r["domain"] == domain]
        false_success = [r for r in labels["FS"] if r["domain"] == domain]
        empty_gate = sum(
            1
            for r in rows
            if ((r["reward_info"].get("info") or {}).get("communicate") or {}).get("note") == EMPTY_COMMUNICATE_NOTE
            and "COMMUNICATE" in (r["reward_info"].get("reward_basis") or [])
        )
        flagged = [r for r in rows if str(r["task_id"]) in AUTO_PASS_TASKS[domain]]
        report[domain] = {
            "trajectories": len(rows),
            "failures": len(failures),
            "false_successes": len(false_success),
            "fs_share_of_failures": round(len(false_success) / len(failures), 4) if failures else None,
            "fs_share_of_all": round(len(false_success) / len(rows), 4),
            "empty_communicate_gate": empty_gate,
            "empty_communicate_share": round(empty_gate / len(rows), 4),
            "issue_384_tasks": len(AUTO_PASS_TASKS[domain]),
            "trajectories_on_384_list": len(flagged),
            "share_on_384_list": round(len(flagged) / len(rows), 4),
        }
        d = report[domain]
        print(
            f"{domain:<10}{d['trajectories']:>7}{d['failures']:>10}{d['false_successes']:>6}"
            f"{(d['fs_share_of_failures'] or 0):>8.1%}{d['fs_share_of_all']:>8.1%}"
            f"{d['empty_communicate_share']:>11.1%}{d['issue_384_tasks']:>12}",
            flush=True,
        )

    airline, retail, telecom = (report[d] for d in ("airline", "retail", "telecom"))
    single = airline["false_successes"] + retail["false_successes"]
    single_fail = airline["failures"] + retail["failures"]
    ratio = (single / single_fail) / telecom["fs_share_of_failures"] if telecom["fs_share_of_failures"] else None

    print(f"\nsingle-control (airline+retail) FS share of failures: {single / single_fail:.1%}")
    print(f"dual-control (telecom)                              : {telecom['fs_share_of_failures']:.1%}")
    print(f"ratio                                               : {ratio:.1f}x" if ratio else "")
    print("\nthe confound: the two single-control domains are exactly the two with an auto-passing")
    print("reward gate. telecom appears in no part of issue #384 and has the working evaluator.")
    print(
        f"  empty COMMUNICATE gate: airline {airline['empty_communicate_share']:.1%}, "
        f"retail {retail['empty_communicate_share']:.1%}, telecom {telecom['empty_communicate_share']:.1%}"
    )
    print(
        f"  tasks named in #384   : airline {airline['issue_384_tasks']}, "
        f"retail {retail['issue_384_tasks']}, telecom {telecom['issue_384_tasks']}"
    )
    print("\nThis does not show dual control has no effect. It shows the comparison used to")
    print("demonstrate it cannot separate control mode from evaluator validity.")

    report["summary"] = {
        "single_control_fs_share_of_failures": round(single / single_fail, 4),
        "dual_control_fs_share_of_failures": telecom["fs_share_of_failures"],
        "ratio": round(ratio, 2) if ratio else None,
        "confound": (
            "the two single-control domains are the two named in tau2-bench issue #384 as having "
            "reward gates that auto-pass; telecom is named nowhere in it. Control mode and "
            "evaluator validity are perfectly collinear across the three domains, so this "
            "comparison cannot attribute the difference to either."
        ),
    }
    report["provenance"] = {
        "data_dir": str(args.data),
        "records": len(records),
        "dropped": dropped,
        "issue": "https://github.com/sierra-research/tau2-bench/issues/384",
        "python": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
