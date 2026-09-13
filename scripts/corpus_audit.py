"""Audit the corpus and the attack for the defects that inflate a detector score without being visible in one."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from fsdetect.attack import STYLES, entity_of, rewrite
from fsdetect.corpus import load, split_labels
from fsdetect.detect import ACTION_VECTORIZER, TEXT_VECTORIZER, auroc, build_dataset, fit, stratified_split

# Control tokens the harness emits in place of a customer-facing message. They are perfectly
# separable and belong to one class only, so a detector can score them without reading language.
DEGENERATE_RE = re.compile(r"TRANSFERRED TO A HUMAN AGENT|###STOP###", re.IGNORECASE)

# Tasks tau2-bench issue #384 reports as scored on inaction alone, because the gold action
# changes no database state and the written criteria are in no reward_basis.
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
}
EMPTY_COMMUNICATE_NOTE = "No communicate_info to evaluate"


def length_stats(rows: list[dict]) -> dict[str, float | int]:
    lengths = sorted(len(r["closing"].split()) for r in rows)
    n = len(lengths)
    return {
        "n": n,
        "mean_words": round(statistics.mean(lengths), 1),
        "median_words": lengths[n // 2],
        "p10": lengths[n // 10],
        "p90": lengths[(9 * n) // 10],
        "max": lengths[-1],
        "at_most_3_words": sum(1 for x in lengths if x <= 3),
        "degenerate": sum(1 for r in rows if DEGENERATE_RE.search(r["closing"])),
    }


def ablate(labels: dict[str, list[dict]], keep, seed: int) -> dict[str, float]:
    filtered = {k: [r for r in v if keep(r)] for k, v in labels.items()}
    rows, y = build_dataset(filtered)
    train, _, test = stratified_split(y, seed)
    scores = {}
    for field, config in (("closing", TEXT_VECTORIZER), ("actions", ACTION_VECTORIZER)):
        detector = fit(rows, y, train, field, config, seed)
        scores[field] = round(auroc(detector, [rows[i][field] for i in test], y[test]), 4)
    scores["rows"] = len(rows)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tau2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/corpus_audit.json"))
    args = parser.parse_args()

    records, dropped = load(args.data)
    labels = split_labels(records)

    lengths = {name: length_stats(rows) for name, rows in labels.items()}
    for name, stats in lengths.items():
        print(
            f"{name:<5} n={stats['n']:>5} mean={stats['mean_words']:>6} median={stats['median_words']:>4} "
            f"<=3w={stats['at_most_3_words']:>4} degenerate={stats['degenerate']:>4}",
            flush=True,
        )

    # Issue #384: a COMMUNICATE gate with no criteria contributes 1.0 while measuring nothing.
    flagged: dict[str, dict[str, float | int]] = {}
    for domain, task_ids in AUTO_PASS_TASKS.items():
        rows = [r for r in records if r["domain"] == domain]
        info = [(r, r["reward_info"]) for r in rows]
        empty_gate = sum(
            1
            for _, i in info
            if ((i.get("info") or {}).get("communicate") or {}).get("note") == EMPTY_COMMUNICATE_NOTE
            and "COMMUNICATE" in (i.get("reward_basis") or [])
        )
        on_list = [r for r in rows if str(r["task_id"]) in task_ids]
        rewarded = sum(1 for r in on_list if r["reward"] == 1.0)
        base = sum(1 for r in rows if r["reward"] == 1.0)
        flagged[domain] = {
            "simulations": len(rows),
            "empty_communicate_gate": empty_gate,
            "empty_communicate_share": round(empty_gate / len(rows), 4),
            "on_issue_384_list": len(on_list),
            "on_list_share": round(len(on_list) / len(rows), 4),
            "on_list_full_reward": rewarded,
            "on_list_full_reward_rate": round(rewarded / len(on_list), 4) if on_list else None,
            "base_full_reward_rate": round(base / len(rows), 4),
        }
        f = flagged[domain]
        print(
            f"{domain:<9} empty COMMUNICATE gate {f['empty_communicate_share']:>6.1%}   "
            f"on #384 list {f['on_list_share']:>6.1%}   "
            f"of those full reward {f['on_list_full_reward_rate']:>6.1%} "
            f"vs base {f['base_full_reward_rate']:.1%}",
            flush=True,
        )

    dataset = labels["FS"] + labels["TS"]
    counts = Counter(r["closing"] for r in dataset)
    repeated = {text: n for text, n in counts.items() if n > 1}
    duplicates = {
        "distinct_closings": len(counts),
        "rows": len(dataset),
        "closings_appearing_more_than_once": len(repeated),
        "rows_covered_by_repeats": sum(repeated.values()),
        "top": [
            {
                "count": n,
                "fs": sum(1 for r in labels["FS"] if r["closing"] == text),
                "ts": sum(1 for r in labels["TS"] if r["closing"] == text),
                "text": text[:120],
            }
            for text, n in sorted(repeated.items(), key=lambda kv: -kv[1])[:5]
        ],
    }
    print(f"\ndistinct closings {duplicates['distinct_closings']} of {duplicates['rows']} rows", flush=True)

    seen: set[str] = set()

    def first_occurrence(row: dict) -> bool:
        if row["closing"] in seen:
            return False
        seen.add(row["closing"])
        return True

    ablations = {
        "as_published": ablate(labels, lambda r: True, args.seed),
        "degenerate_removed": ablate(labels, lambda r: not DEGENERATE_RE.search(r["closing"]), args.seed),
        "deduplicated": ablate(labels, first_occurrence, args.seed),
    }
    for name, scores in ablations.items():
        print(f"{name:<22} text {scores['closing']:.4f}  structural {scores['actions']:.4f}", flush=True)

    # The attack rewrites into a fixed template pool, so the number of distinct strings the detector
    # actually sees bounds how much the flip rate can be trusted, whatever the trajectory count is.
    honest = labels["HF"]
    entities = Counter(entity_of(r["closing"]) for r in honest)
    diversity = {}
    for style in STYLES:
        rng = random.Random(args.seed)
        rewritten = [rewrite(r["closing"], style, rng) for r in honest]
        diversity[style] = {"trajectories": len(rewritten), "distinct_strings": len(set(rewritten))}
    print(
        f"\nattack A: {len(honest)} trajectories collapse to "
        f"{diversity['direct']['distinct_strings']} distinct strings",
        flush=True,
    )

    report = {
        "closing_length_by_class": lengths,
        "duplicate_closings": duplicates,
        "issue_384_auto_pass": flagged,
        "ablations": ablations,
        "attack_diversity": {
            "entity_unresolved": entities["request"],
            "entity_unresolved_share": round(entities["request"] / len(honest), 4),
            "top_entities": dict(entities.most_common(6)),
            "by_style": diversity,
        },
        "provenance": {"seed": args.seed, "data_dir": str(args.data), "dropped": dropped},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
