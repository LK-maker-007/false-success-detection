from __future__ import annotations

import json
import re
from pathlib import Path

MUTATING_METHODS = frozenset({"post", "put", "patch", "delete"})
COMPLETION_URL = "/supervisor/message"
ID_LIKE = re.compile(r"^[A-Za-z0-9_.:/@+-]{8,}$")

Record = dict[str, object]


def endpoint_tokens(calls: list[dict]) -> str:
    tokens: list[str] = []
    for call in calls:
        method = call.get("method")
        url = call.get("url") or ""
        app = url.strip("/").split("/")[0] or "root"
        tokens.append(f"ep:{method}:{url}")
        tokens.append(f"app:{method}:{app}")
        if method in MUTATING_METHODS:
            tokens.append(f"mut:{app}")
    tokens.append(f"ncalls:{min(len(calls) // 10, 12)}")
    tokens.append(f"nmut:{min(sum(1 for c in calls if c.get('method') in MUTATING_METHODS), 20)}")
    return " ".join(tokens)


def argument_tokens(calls: list[dict]) -> str:
    # Same hypothesis as the tau2 extractor: the endpoint says which operation was attempted, the
    # arguments say whether it was the right one. Opaque ids and tokens are abstracted to their key.
    tokens: list[str] = []
    for call in calls:
        if call.get("method") not in MUTATING_METHODS:
            continue
        url = call.get("url") or ""
        for key, value in (call.get("data") or {}).items():
            tokens.append(f"argk:{url}:{key}")
            if isinstance(value, bool):
                tokens.append(f"val:{key}={value}")
            elif isinstance(value, (int, float)):
                tokens.append(f"num:{key}={min(int(value), 10)}")
            elif isinstance(value, list):
                tokens.append(f"len:{key}={min(len(value), 5)}")
            elif isinstance(value, str):
                if value and not ID_LIKE.match(value) and len(value) <= 24:
                    tokens.append(f"val:{key}={value.lower().replace(' ', '_')}")
                else:
                    tokens.append(f"opaque:{key}")
    return " ".join(tokens) or "no_mutation"


def completion_claim(calls: list[dict]) -> tuple[bool, str | None]:
    # Submissions differ: full_code_refl and cuga send complete_task(status="success"), react and
    # plan_exec send a bare complete_task() with no status. Reaching the endpoint at all is an
    # implicit claim of completion; only some submissions make it explicit.
    messages = [c for c in calls if c.get("url") == COMPLETION_URL and isinstance(c.get("data"), dict)]
    if not messages:
        return False, None
    return True, messages[-1]["data"].get("status")


def load(root: Path, *, require_explicit_status: bool = True) -> tuple[list[Record], dict[str, int]]:
    if not root.exists():
        raise ValueError(f"{root} does not exist; clone appworld-leaderboard and unpack the bundles first")
    experiments = sorted(p for p in root.iterdir() if (p / "evaluations").is_dir())
    if not experiments:
        raise ValueError(f"{root} contains no unpacked experiments; run appworld unpack first")

    records: list[Record] = []
    dropped = {
        "no_api_calls_file": 0,
        "no_completion_call": 0,
        "no_explicit_status": 0,
        "claimed_failure": 0,
        "status_call_only": 0,
    }
    for experiment in experiments:
        for evaluation_file in sorted((experiment / "evaluations").glob("*.json")):
            evaluations = json.loads(evaluation_file.read_text(encoding="utf-8"))["individual"]
            for task_id, record in evaluations.items():
                call_file = experiment / "tasks" / task_id / "logs/api_calls.jsonl"
                if not call_file.exists():
                    dropped["no_api_calls_file"] += 1
                    continue
                calls = [json.loads(line) for line in call_file.read_text(encoding="utf-8").splitlines()]
                completed, status = completion_claim(calls)
                if not completed:
                    dropped["no_completion_call"] += 1
                    continue
                if status is None and require_explicit_status:
                    dropped["no_explicit_status"] += 1
                    continue
                if status is not None and status != "success":
                    dropped["claimed_failure"] += 1
                    continue
                # A trace whose only call is the completion itself carries no behaviour to model.
                if len(calls) <= 1:
                    dropped["status_call_only"] += 1
                    continue
                records.append(
                    {
                        "submission": experiment.name,
                        "split": evaluation_file.stem,
                        "task_id": task_id,
                        "success": bool(record["success"]),
                        "explicit_status": status is not None,
                        "endpoints": endpoint_tokens(calls),
                        "arguments": argument_tokens(calls),
                    }
                )
    return records, dropped


def build_dataset(records: list[Record]) -> tuple[list[Record], list[int]]:
    # Positives are trajectories that claimed completion and failed, negatives are those that
    # claimed completion and passed. Same FS vs TS contrast as the tau2 corpus.
    y = [0 if r["success"] else 1 for r in records]
    if not any(y) or all(y):
        raise ValueError(f"FS vs TS needs both classes, got {sum(y)} FS and {len(y) - sum(y)} TS")
    return records, y
