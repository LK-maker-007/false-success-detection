from __future__ import annotations

import random

import pytest

from fsdetect.attack import STYLES, assert_completion, entity_of, rewrite, swap_lexicon
from fsdetect.corpus import FS_ASSERT_RE, HONEST_FAILURE_RE, label_failure


def test_entity_is_carried_from_the_original_closing():
    assert entity_of("I could not change your reservation today") == "reservation"
    assert entity_of("nothing recognisable here") == "request"


def test_rewrite_is_deterministic_under_a_seed():
    a = rewrite("your order", "direct", random.Random(42))
    b = rewrite("your order", "direct", random.Random(42))
    assert a == b


def test_rewrite_rejects_unknown_style():
    with pytest.raises(ValueError, match="unknown rewrite style"):
        rewrite("text", "nonexistent", random.Random(0))


def test_lexicon_avoiding_style_trips_no_fs_trigger():
    # The claim that this style evades FS_ASSERT_RE is what makes its flip rate interpretable,
    # so it is asserted rather than assumed.
    for template in STYLES["lexicon_avoiding"]:
        text = template.format(e="order")
        assert not FS_ASSERT_RE.search(text.lower()), text


def test_other_styles_do_trip_the_fs_trigger():
    for style in ("direct", "technically_true"):
        for template in STYLES[style]:
            text = template.format(e="order")
            assert label_failure(text) == "FS", text


def test_swap_lexicon_keeps_the_facts_and_drops_the_trigger():
    text = "I've successfully issued a $200 certificate. Your reservation has been confirmed."
    out = swap_lexicon(text)
    assert "$200 certificate" in out
    assert not FS_ASSERT_RE.search(out.lower())


def test_swap_lexicon_is_idempotent():
    text = "Your order has been processed and you're all set."
    assert swap_lexicon(swap_lexicon(text)) == swap_lexicon(text)


def test_assert_completion_removes_the_admission_and_keeps_the_body():
    text = (
        "I appreciate the details. Unfortunately, I cannot make this change. "
        "According to our policy, flight modifications cannot change the destination."
    )
    out = assert_completion(text)
    assert "According to our policy" in out
    assert not HONEST_FAILURE_RE.search(out.lower())
    assert out.startswith("I have gone ahead")


def test_assert_completion_leaves_no_unpaired_bold_marker():
    text = "Here is the position. **Unfortunately, I cannot help.** The policy is fixed."
    assert assert_completion(text).count("**") % 2 == 0


def test_assert_completion_carries_the_entity_from_the_original():
    assert "reservation" in assert_completion("I cannot change your reservation.")
    assert "request" in assert_completion("I cannot do that.")
