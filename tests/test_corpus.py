from __future__ import annotations

import json

import pytest

from fsdetect.corpus import (
    action_tokens,
    argument_tokens,
    closing_message,
    domain_of,
    label_failure,
    load,
    split_labels,
)


def test_label_requires_exactly_one_pattern():
    assert label_failure("Your refund has been processed.") == "FS"
    assert label_failure("I cannot process that request.") == "HF"
    assert label_failure("I cannot do that, but your order has been processed.") == "AMB"
    assert label_failure("Anything else today?") == "AMB"


def test_closing_message_takes_last_assistant_not_last_message():
    messages = [
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "thanks"},
    ]
    assert closing_message(messages) == "second"
    assert closing_message([{"role": "user", "content": "hi"}]) is None


def test_action_tokens_credit_user_simulator_calls():
    messages = [
        {"role": "user", "tool_calls": [{"id": "a", "name": "get_bill", "requestor": "user"}]},
        {"role": "tool", "id": "a", "error": False},
        {"role": "assistant", "tool_calls": [{"id": "b", "name": "refund", "requestor": "assistant"}]},
        {"role": "tool", "id": "b", "error": True},
    ]
    tokens = action_tokens(messages).split()
    assert "call:user:get_bill" in tokens
    assert "ok:user:get_bill" in tokens
    assert "err:assistant:refund" in tokens
    assert "orphan" not in action_tokens(messages)


def test_action_tokens_mark_unmatched_tool_response():
    messages = [{"role": "tool", "id": "missing", "error": False}]
    assert "ok:orphan" in action_tokens(messages).split()


def test_domain_of():
    assert domain_of("gpt-5_airline_default_gpt-4.1_4trials.json") == "airline"
    assert domain_of("banking_knowledge_results.json") is None


def test_load_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no trajectory files"):
        load(tmp_path)


def test_load_rejects_fractional_reward(tmp_path):
    payload = {
        "simulations": [
            {
                "task_id": 1,
                "reward_info": {"reward": 0.5},
                "messages": [{"role": "assistant", "content": "done"}],
            }
        ]
    }
    (tmp_path / "m_airline_x_4trials.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="neither 0.0 nor 1.0"):
        load(tmp_path)


def test_load_counts_each_drop_reason(tmp_path):
    payload = {
        "simulations": [
            {"task_id": 1, "reward_info": {"reward": 1.0}, "messages": []},
            {"task_id": 2, "reward_info": {}, "messages": [{"role": "assistant", "content": "x"}]},
            {"task_id": 3, "reward_info": {"reward": 0.0}, "messages": [{"role": "user", "content": "x"}]},
            {"task_id": 4, "reward_info": {"reward": 1.0}, "messages": [{"role": "assistant", "content": "x"}]},
        ]
    }
    (tmp_path / "m_retail_x_4trials.json").write_text(json.dumps(payload), encoding="utf-8")
    records, dropped = load(tmp_path)
    assert len(records) == 1
    assert dropped == {"missing_reward": 1, "no_messages": 1, "no_assistant_turn": 1}


def test_split_labels_partitions_failures_and_keeps_successes():
    records = [
        {"reward": 0.0, "closing": "Your refund has been processed."},
        {"reward": 0.0, "closing": "I cannot help with that."},
        {"reward": 0.0, "closing": "Anything else?"},
        {"reward": 1.0, "closing": "Your refund has been processed."},
    ]
    labels = split_labels(records)
    assert [len(labels[k]) for k in ("FS", "HF", "AMB", "TS")] == [1, 1, 1, 1]


def test_argument_tokens_keep_enums_and_payment_type_but_not_the_id():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "a",
                    "name": "update_reservation_flights",
                    "arguments": {
                        "reservation_id": "XEHM4B",
                        "cabin": "economy",
                        "flights": [{"flight_number": "HAT005"}, {"flight_number": "HAT178"}],
                        "payment_id": "gift_card_8190333",
                    },
                }
            ],
        }
    ]
    tokens = argument_tokens(messages).split()
    assert "val:cabin=economy" in tokens
    assert "pay:gift_card" in tokens
    assert "len:update_reservation_flights:flights=2" in tokens
    # one reservation, two flight numbers and one payment method are four distinct entities.
    assert "nids:4" in tokens
    assert not any("XEHM4B" in t or "8190333" in t for t in tokens)


def test_argument_tokens_ignore_read_only_tools():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "a", "name": "get_order_details", "arguments": {"order_id": "#W123456"}}],
        }
    ]
    assert argument_tokens(messages).split() == ["nids:0", "nmut:0"]
