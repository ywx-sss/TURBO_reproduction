"""Reject sampling rules (TURBO §4.1)."""

from __future__ import annotations

from trace_utils import answers_match, extract_answer_from_target, wrap_sft_target

MIN_REASONING_LEN = 40
MAX_REASONING_LEN = 4000
MAX_THINK_COUNT = 3


def reject_reason(rec: dict, reasoning: str) -> tuple[bool, str]:
    if len(reasoning) < MIN_REASONING_LEN:
        return False, "too_short"
    if len(reasoning) > MAX_REASONING_LEN:
        return False, "too_verbose"
    if reasoning.lower().count("therefore") > MAX_THINK_COUNT:
        return False, "contradictory_verbose"

    target = wrap_sft_target(reasoning, rec["answer"])
    extracted = extract_answer_from_target(target)
    if not extracted:
        return False, "format_error"
    if not answers_match(extracted, rec["answer"]):
        return False, "answer_mismatch"

    # reject if reasoning explicitly states a different final numeric/text conclusion
    last_line = reasoning.strip().splitlines()[-1].lower()
    gold = rec["answer"].strip().lower()
    if "answer is" in last_line or "final answer" in last_line:
        if gold not in last_line and not any(
            tok in last_line for tok in gold.split() if len(tok) > 2
        ):
            return False, "reasoning_contradicts_gold"

    return True, "ok"
