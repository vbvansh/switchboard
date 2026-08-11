"""Grading must be strict enough to be meaningful and fair to every model."""

from __future__ import annotations

import pytest

from eval.grading import Check, extract_answer, strip_thinking


# --- Thinking blocks -------------------------------------------------------


def test_think_blocks_are_removed() -> None:
    """qwen3 emits these; the scratchpad must not be graded."""
    text = "<think>Maybe 41? No, 45.</think>\nANSWER: 45"
    assert "<think>" not in strip_thinking(text)
    assert "Maybe 41" not in strip_thinking(text)


def test_unclosed_think_block_is_removed() -> None:
    """A truncated reply can leave <think> open; it must not survive."""
    assert strip_thinking("before <think>ramble ramble") == "before"


def test_thinking_does_not_leak_into_the_answer() -> None:
    """Regression: a number inside <think> must not be graded as the answer."""
    text = "<think>I first guessed 99</think>\nANSWER: 45"
    answer, _ = extract_answer(text)
    assert Check("numeric", 45).grade(answer)
    assert not Check("numeric", 99).grade(answer)


# --- Answer extraction -----------------------------------------------------


def test_marked_answer_is_extracted() -> None:
    assert extract_answer("Some working.\nANSWER: 42") == ("42", True)


def test_last_marker_wins() -> None:
    """If a model restates the marker, the final one is its conclusion."""
    answer, _ = extract_answer("ANSWER: 10\nWait, recomputing.\nANSWER: 12")
    assert answer == "12"


def test_marker_is_case_insensitive() -> None:
    answer, followed = extract_answer("answer: 7")
    assert (answer, followed) == ("7", True)


def test_missing_marker_is_flagged_but_still_graded() -> None:
    """Format failures are recorded, not silently punished."""
    answer, followed = extract_answer("The result is 42")
    assert followed is False
    assert Check("numeric", 42).grade(answer)


def test_text_after_the_marker_line_is_ignored() -> None:
    answer, _ = extract_answer("ANSWER: 42\nHope that helps!")
    assert answer == "42"


# --- Numeric checks --------------------------------------------------------


@pytest.mark.parametrize(
    "answer", ["45", " 45 ", "45.0", "The answer is 45", "1,000 minus 955 is 45"]
)
def test_numeric_accepts_reasonable_forms(answer: str) -> None:
    assert Check("numeric", 45).grade(answer)


def test_numeric_takes_the_last_number_as_the_conclusion() -> None:
    """Working appears before the result, so the final number is the answer."""
    assert Check("numeric", 21).grade("3 times 7, so 3 * 7 = 21")


def test_numeric_rejects_wrong_values() -> None:
    assert not Check("numeric", 45).grade("46")
    assert not Check("numeric", 45).grade("no idea")


def test_numeric_handles_thousands_separators() -> None:
    assert Check("numeric", 5050).grade("5,050")


def test_numeric_handles_negatives() -> None:
    assert Check("numeric", -12).grade("-12")


# --- Exact and contains ----------------------------------------------------


def test_exact_ignores_case_and_punctuation() -> None:
    check = Check("exact", "Au")
    assert check.grade("au")
    assert check.grade("Au.")
    assert not check.grade("Ag")


def test_contains_requires_every_value() -> None:
    check = Check("contains", values=("tokyo",))
    assert check.grade("The capital is Tokyo.")
    assert not check.grade("The capital is Kyoto.")


def test_unknown_check_type_is_rejected() -> None:
    """A typo in a task file must fail loudly, not grade everything wrong."""
    with pytest.raises(ValueError, match="Unknown check type"):
        Check("nonsense", "x").grade("x")
