"""Reasoning trace helpers for TURBO trace generation."""

from __future__ import annotations

import re
from typing import Optional

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def wrap_sft_target(reasoning: str, answer: str) -> str:
    r = reasoning.strip()
    a = answer.strip()
    return f"{THINK_OPEN}{r}{THINK_CLOSE}{ANSWER_OPEN}{a}{ANSWER_CLOSE}"


def extract_answer_from_target(text: str) -> Optional[str]:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\$,]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("percent", "%")
    return s


def answers_match(pred: str, gold: str) -> bool:
    """Match pred against gold; gold may be WTQ-style 'a|b' alternatives."""
    for part in gold.split("|"):
        g = part.strip()
        if not g:
            continue
        if _answers_match_one(pred, g):
            return True
    return False


def _answers_match_one(pred: str, gold: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return True
    # numeric
    try:
        return abs(float(re.findall(r"-?\d+\.?\d*", p)[0]) - float(re.findall(r"-?\d+\.?\d*", g)[0])) < 1e-4
    except (IndexError, ValueError):
        pass
    # multi-choice: gold may be name only
    if g in p or p in g:
        return True
    return False


def wtq_fallback_reasoning(rec: dict) -> str:
    return (
        f"Read the table (context: {rec.get('table_context', 'table')}). "
        f"Locate the cells needed for: {rec['question']} "
        f"Then derive the answer step by step."
    )


TRACE_USER_TEMPLATE = """You are generating a structure-aware reasoning trace for tabular QA.

Given the structured table (Markdown), question, and the correct final answer, write a clear step-by-step reasoning chain that logically derives the answer from the table. Reference specific rows, columns, or cells. Do NOT output answer tags — reasoning text only.

## Table
{table_md}

## Question
{question}

## Correct Answer (for guidance)
{answer}

Write the reasoning trace:"""


def solution_fallback_reasoning(rec: dict) -> str:
    """Use dataset gold solution when DeepSeek API is unavailable."""
    sol = (rec.get("solution_reference") or "").strip()
    if sol:
        return sol
    if rec.get("dataset") == "WTQ":
        return wtq_fallback_reasoning(rec)
    return (
        f"Read the table titled '{rec.get('table_title') or 'table'}'. "
        f"Apply the operations needed for: {rec['question']}"
    )
