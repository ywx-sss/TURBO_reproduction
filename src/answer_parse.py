"""Parse model output into a predicted answer string."""

from __future__ import annotations

import re

from trace_utils import extract_answer_from_target

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def parse_pred_answer(raw: str) -> str:
    tagged = extract_answer_from_target(raw)
    if tagged:
        return tagged
    m = _ANSWER_RE.search(raw)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return raw.strip()
