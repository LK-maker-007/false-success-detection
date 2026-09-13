"""Test whether the reference corpus, 9876 trajectories at 8146 successes and 1730 failures, exists."""

from __future__ import annotations

import argparse
import json
import platform
import urllib.parse
import urllib.request
from collections import Counter
from email.utils import parsedate_to_datetime
from itertools import combinations
from pathlib import Path

from fsdetect.corpus import closing_message

BUCKET = "https://sierra-tau-bench-public.s3.amazonaws.com"
TARGET_TOTAL = 9876
TARGET_SUCCESS = 8146
TARGET_FAILURE = 1730
# The leaderboard as of the reference work's arXiv date. The next manifest commit is 2026-06-22.
MANIFEST_COMMIT = "c7e436f33"


def file_counts(path: Path) -> tuple[int, int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    total = success = failure = 0
    for sim in payload["simulations"]:
        reward = (sim.get("reward_info") or {}).get("reward")
        messages = sim.get("messages")
        if reward is None or not messages or closing_message(messages) is None:
            continue
        total += 1
        success += reward == 1.0
        failure += reward == 0.0
    return total, success, failure


def meet_in_the_middle(items: list[tuple[str, int, int]]) -> list[tuple]:
    half = len(items) // 2
    left, right = items[:half], items[half:]

    def reachable(part):
        out: dict[tuple[int, int], list[int]] = {}
        for mask in range(1 << len(part)):
            success = failure = 0
            for i in range(len(part)):
                if mask >> i & 1:
                    success += part[i][1]
                    failure += part[i][2]
            if success <= TARGET_SUCCESS and failure <= TARGET_FAILURE:
                out.setdefault((success, failure), []).append(mask)
        return out

    table = reachable(left)
    hits = []
    for mask_r in range(1 << len(right)):
        success = failure = 0
        for i in range(len(right)):
            if mask_r >> i & 1:
                success += right[i][1]
                failure += right[i][2]
        complement = (TARGET_SUCCESS - success, TARGET_FAILURE - failure)
        for mask_l in table.get(complement, []):
            hits.append((mask_l, mask_r))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--check-timestamps", action="store_true", help="HEAD every key against S3")
    parser.add_argument("--output", type=Path, default=Path("results/corpus_provenance.json"))
    args = parser.parse_args()

    files = sorted(f for f in args.data.rglob("*.json") if f.name != "_bucket_keys.json")
    per_file = [(f"{f.parent.name}/{f.name}", *file_counts(f)) for f in files]
    per_submission: dict[str, list[int]] = {}
    for name, total, success, failure in per_file:
        row = per_submission.setdefault(name.split("/")[0], [0, 0, 0])
        row[0] += total
        row[1] += success
        row[2] += failure

    pooled_success = sum(v[1] for v in per_submission.values())
    pooled_failure = sum(v[2] for v in per_submission.values())
    print(f"{'submission':<46}{'n':>7}{'succ':>7}{'fail':>7}{'fail%':>8}")
    for name, (total, success, failure) in sorted(per_submission.items()):
        print(f"{name:<46}{total:>7}{success:>7}{failure:>7}{failure / total:>7.1%}", flush=True)
    print(
        f"{'POOLED':<46}{sum(v[0] for v in per_submission.values()):>7}{pooled_success:>7}"
        f"{pooled_failure:>7}{pooled_failure / (pooled_success + pooled_failure):>7.1%}"
    )
    print(
        f"{'REFERENCE':<46}{TARGET_TOTAL:>7}{TARGET_SUCCESS:>7}{TARGET_FAILURE:>7}"
        f"{TARGET_FAILURE / TARGET_TOTAL:>7.1%}"
    )

    submission_hits = [
        combo
        for r in range(1, len(per_submission) + 1)
        for combo in combinations(sorted(per_submission), r)
        if (sum(per_submission[k][1] for k in combo), sum(per_submission[k][2] for k in combo))
        == (TARGET_SUCCESS, TARGET_FAILURE)
    ]
    file_hits = meet_in_the_middle(per_file)
    print(f"\nsubsets of whole submissions matching ({TARGET_SUCCESS}, {TARGET_FAILURE}): {len(submission_hits)}")
    print(f"subsets of individual files matching, exhaustive: {len(file_hits)}")

    termination = Counter()
    for f in files:
        for sim in json.loads(f.read_text(encoding="utf-8"))["simulations"]:
            reward = (sim.get("reward_info") or {}).get("reward")
            if reward is None:
                continue
            termination[(sim.get("termination_reason"), reward == 0.0)] += 1
    by_reason = {}
    for reason in {r for r, _ in termination}:
        fails = termination[(reason, True)]
        total = fails + termination[(reason, False)]
        by_reason[str(reason)] = {"n": total, "failures": fails, "rate": round(fails / total, 4)}
    print("\nfailure rate by termination_reason, none reaches the required 17.5%")
    for reason, row in sorted(by_reason.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {reason:<24}{row['n']:>7}{row['rate']:>8.1%}", flush=True)

    timestamps = None
    if args.check_timestamps:
        keys = json.loads((args.data / "_bucket_keys.json").read_text())
        wanted = {f"{f.parent.name}/{f.name}" for f in files}
        stamps = []
        for key, _ in keys:
            if "/trajectories/" not in key:
                continue
            label = f"{key.split('/')[1]}/{key.rsplit('/', 1)[-1]}"
            if label not in wanted:
                continue
            request = urllib.request.Request(f"{BUCKET}/{urllib.parse.quote(key)}", method="HEAD")
            stamps.append(parsedate_to_datetime(urllib.request.urlopen(request).headers["Last-Modified"]))
        timestamps = {
            "checked": len(stamps),
            "earliest": min(stamps).isoformat(),
            "latest": max(stamps).isoformat(),
            "modified_after_2026_06": sum(1 for d in stamps if d.strftime("%Y-%m") > "2026-06"),
        }
        print(f"\nS3 Last-Modified: {timestamps}")

    report = {
        "target": {"total": TARGET_TOTAL, "success": TARGET_SUCCESS, "failure": TARGET_FAILURE},
        "manifest_commit": MANIFEST_COMMIT,
        "per_submission": {k: {"n": v[0], "success": v[1], "failure": v[2]} for k, v in per_submission.items()},
        "pooled": {
            "success": pooled_success,
            "failure": pooled_failure,
            "failure_rate": round(pooled_failure / (pooled_success + pooled_failure), 4),
        },
        "submission_subsets_matching": len(submission_hits),
        "file_subsets_matching_exhaustive": len(file_hits),
        "files_searched": len(per_file),
        "failure_rate_by_termination_reason": by_reason,
        "s3_timestamps": timestamps,
        "provenance": {"data_dir": str(args.data), "python": platform.python_version()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
