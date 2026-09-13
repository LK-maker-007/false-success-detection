from __future__ import annotations

import json
import re
from pathlib import Path

FS_ASSERT_RE = re.compile(
    r"\b(successfully|has\s+been|have\s+been|is\s+(now\s+)?(complete|done|processed|"
    r"booked|cancelled|canceled|updated|submitted|confirmed|refunded|approved))\b"
    r"|\brefund(ed)?\s*(of\s+)?\$?\d+"
    r"|\byou(?:\'re|\s+are)\s+(all\s+set|good\s+to\s+go)"
    r"|\byour\s+(reservation|booking|order|return|refund|cancellation|change)\s+"
    r"(has\s+been\s+|is\s+)(complete|confirmed|processed|submitted|approved)"
    r"|\b(processed|completed|finalized|issued)\s+(the\s+)?(refund|cancellation|change)",
    re.IGNORECASE,
)

HONEST_FAILURE_RE = re.compile(
    r"\bI\s+(cannot|can\'t|am\s+unable|am\s+not\s+able)\b"
    r"|\b(transferring|transferred|transfer)\s+(you\s+)?(to\s+)?(a\s+)?"
    r"(human|agent|representative|specialist)"
    r"|\bunable\s+to\s+(process|complete|fulfill|approve)"
    r"|\bshall\s+I\s+proceed\b"
    r"|\bwould\s+you\s+like\s+(me\s+)?to\s+proceed"
    r"|\bI\s+(\'m|am)\s+sorry\b"
    r"|\brequires?\s+manual\s+(review|intervention|approval)"
    r"|\bI\s+don\'t\s+have\s+(the\s+)?(authority|ability|access)",
    re.IGNORECASE,
)

DOMAINS = ("airline", "retail", "telecom")

Record = dict[str, object]


def closing_message(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content") or ""
    return None


def label_failure(closing: str) -> str:
    low = closing.lower()
    asserts = bool(FS_ASSERT_RE.search(low))
    admits = bool(HONEST_FAILURE_RE.search(low))
    return "FS" if asserts and not admits else "HF" if admits and not asserts else "AMB"


def action_tokens(messages: list[dict]) -> str:
    tokens: list[str] = []
    pending: dict[str, str] = {}
    for message in messages:
        # In dual-control telecom the user simulator issues tool calls too, so tool_calls are
        # registered from any role. Registering only from assistant orphaned 209 tool responses.
        for call in message.get("tool_calls") or []:
            who = call.get("requestor") or message.get("role")
            name = (call.get("function") or {}).get("name") or call.get("name") or "unknown"
            tokens.append(f"call:{who}:{name}")
            pending[call.get("id")] = f"{who}:{name}"
        if message.get("role") == "tool":
            name = pending.get(message.get("id"), "orphan")
            tokens.append(f"err:{name}" if message.get("error") else f"ok:{name}")
    tokens.append(f"ncalls:{min(sum(t.startswith('call:') for t in tokens), 20)}")
    tokens.append(f"nerr:{min(sum(t.startswith('err:') for t in tokens), 10)}")
    tokens.append(f"lastrole:{messages[-1].get('role')}")
    return " ".join(tokens)


MUTATING_TOOLS = frozenset(
    {
        "cancel_reservation",
        "book_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "send_certificate",
        "cancel_pending_order",
        "modify_pending_order_items",
        "modify_pending_order_address",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        "make_payment",
        "send_payment_request",
        "suspend_line",
        "resume_line",
        "enable_roaming",
        "disable_roaming",
        "set_apn_settings",
        "reset_apn_settings",
        "set_network_mode_preference",
    }
)

ID_RE = re.compile(r"^(#?[A-Z0-9]{5,}|[a-z]+_[a-z]+_\d+|[A-Z]\d+|\d{6,})$")
PAYMENT_RE = re.compile(r"^(gift_card|credit_card|paypal|certificate)_")


def _leaves(value: object, key: str):
    if isinstance(value, dict):
        for inner_key, inner in value.items():
            yield from _leaves(inner, inner_key)
    elif isinstance(value, list):
        for inner in value:
            yield from _leaves(inner, key)
    else:
        yield key, value


def argument_tokens(messages: list[dict]) -> str:
    # Tool names alone cannot separate cancelling the right reservation from the wrong one: 93.1% of
    # false successes fail the DB check while the call itself succeeded. The discriminating detail is
    # in the arguments. Concrete ids are abstracted away, only their shape and enum values are kept.
    # Whether an id also appears in the user's turns was tried and dropped: it fires on nearly every
    # trajectory and cost 0.0061 AUROC.
    tokens: list[str] = []
    ids_touched: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            name = call.get("name")
            if name not in MUTATING_TOOLS:
                continue
            for key, raw in (call.get("arguments") or {}).items():
                tokens.append(f"arg:{name}:{key}")
                if isinstance(raw, list):
                    tokens.append(f"len:{name}:{key}={min(len(raw), 5)}")
                for leaf_key, value in _leaves(raw, key):
                    if isinstance(value, bool):
                        tokens.append(f"val:{leaf_key}={value}")
                    elif isinstance(value, (int, float)):
                        tokens.append(f"num:{leaf_key}={min(int(value), 10)}")
                    elif isinstance(value, str) and value:
                        payment = PAYMENT_RE.match(value)
                        if payment:
                            tokens.append(f"pay:{payment.group(1)}")
                        if ID_RE.match(value):
                            ids_touched.add(value)
                        elif len(value) <= 24:
                            tokens.append(f"val:{leaf_key}={value.lower().replace(' ', '_')}")
    tokens.append(f"nids:{min(len(ids_touched), 6)}")
    tokens.append(f"nmut:{min(sum(1 for t in tokens if t.startswith('arg:')), 10)}")
    return " ".join(tokens)


# Operation a closing message claims, mapped to the tools that could have performed it. Used to
# test whether false successes are claims without a matching action. They are not; see
# scripts/mechanism.py.
CLAIMED_OPERATIONS: dict[str, tuple[re.Pattern[str], frozenset[str]]] = {
    "cancel": (
        re.compile(r"\bcancell?(ed|ation)\b", re.IGNORECASE),
        frozenset({"cancel_reservation", "cancel_pending_order"}),
    ),
    "book": (
        re.compile(r"\b(booked|reservation (?:is |has been )?(?:created|confirmed))\b", re.IGNORECASE),
        frozenset({"book_reservation"}),
    ),
    "refund": (
        re.compile(r"\brefund(ed)?\b|\bcredited\b", re.IGNORECASE),
        frozenset(
            {
                "cancel_reservation",
                "cancel_pending_order",
                "return_delivered_order_items",
                "exchange_delivered_order_items",
                "send_certificate",
            }
        ),
    ),
    "update": (
        re.compile(r"\b(updated|changed|modified|upgraded|downgraded)\b", re.IGNORECASE),
        frozenset(
            {
                "update_reservation_flights",
                "update_reservation_baggages",
                "update_reservation_passengers",
                "modify_pending_order_items",
                "modify_pending_order_address",
                "modify_pending_order_payment",
                "modify_user_address",
            }
        ),
    ),
    "return": (
        re.compile(r"\breturn(ed)?\b", re.IGNORECASE),
        frozenset({"return_delivered_order_items"}),
    ),
    "exchange": (
        re.compile(r"\bexchang(ed|e)\b", re.IGNORECASE),
        frozenset({"exchange_delivered_order_items"}),
    ),
    "payment": (
        re.compile(r"\bpayment (?:has been |was )?(?:made|processed|sent)\b", re.IGNORECASE),
        frozenset({"make_payment", "send_payment_request"}),
    ),
    "line": (
        re.compile(
            r"\b(suspended|resumed|roaming (?:is )?(?:enabled|disabled)|apn (?:is )?(?:set|reset))\b",
            re.IGNORECASE,
        ),
        frozenset(
            {
                "suspend_line",
                "resume_line",
                "enable_roaming",
                "disable_roaming",
                "set_apn_settings",
                "reset_apn_settings",
                "set_network_mode_preference",
            }
        ),
    ),
}


def successful_tools(messages: list[dict]) -> frozenset[str]:
    pending: dict[str, str] = {}
    succeeded: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            pending[call.get("id")] = call.get("name")
        if message.get("role") == "tool" and not message.get("error"):
            name = pending.get(message.get("id"))
            if name:
                succeeded.add(name)
    return frozenset(succeeded)


def claimed_operations(closing: str) -> frozenset[str]:
    return frozenset(name for name, (pattern, _) in CLAIMED_OPERATIONS.items() if pattern.search(closing))


def claim_is_unsupported(closing: str, succeeded: frozenset[str]) -> bool | None:
    claimed = claimed_operations(closing)
    if not claimed:
        return None
    allowed = frozenset().union(*(CLAIMED_OPERATIONS[name][1] for name in claimed))
    return not (allowed & succeeded)


def domain_of(filename: str) -> str | None:
    return next((d for d in DOMAINS if f"_{d}_" in filename), None)


def load(data_dir: Path) -> tuple[list[Record], dict[str, int]]:
    files = sorted(f for f in data_dir.rglob("*.json") if f.name != "_bucket_keys.json")
    if not files:
        raise ValueError(f"{data_dir} contains no trajectory files; run scripts/fetch_tau2.py first")

    records: list[Record] = []
    dropped = {"missing_reward": 0, "no_messages": 0, "no_assistant_turn": 0}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sim in payload["simulations"]:
            reward = (sim.get("reward_info") or {}).get("reward")
            messages = sim.get("messages")
            if not messages:
                dropped["no_messages"] += 1
                continue
            if reward is None:
                dropped["missing_reward"] += 1
                continue
            closing = closing_message(messages)
            if closing is None:
                dropped["no_assistant_turn"] += 1
                continue
            records.append(
                {
                    "submission": path.parent.name,
                    "domain": domain_of(path.name),
                    "task_id": sim["task_id"],
                    "trial": sim.get("trial"),
                    "reward": reward,
                    "closing": closing,
                    "actions": action_tokens(messages),
                    "arguments": argument_tokens(messages),
                    "structural": f"{action_tokens(messages)} {argument_tokens(messages)}",
                    "succeeded": successful_tools(messages),
                    "reward_info": sim.get("reward_info") or {},
                }
            )

    # A reward that is neither 0.0 nor 1.0 would fall out of both classes and silently shrink the
    # dataset rather than raise, so the split is checked rather than assumed.
    fractional = [r for r in records if r["reward"] not in (0.0, 1.0)]
    if fractional:
        raise ValueError(
            f"{len(fractional)} trajectories carry a reward that is neither 0.0 nor 1.0 "
            f"(e.g. {[r['reward'] for r in fractional[:3]]}); the FS/TS split assumes binary reward"
        )
    return records, dropped


def split_labels(records: list[Record]) -> dict[str, list[Record]]:
    failures = [r for r in records if r["reward"] == 0.0]
    for record in failures:
        record["label"] = label_failure(record["closing"])
    return {
        "FS": [r for r in failures if r["label"] == "FS"],
        "HF": [r for r in failures if r["label"] == "HF"],
        "AMB": [r for r in failures if r["label"] == "AMB"],
        "TS": [r for r in records if r["reward"] == 1.0],
    }
