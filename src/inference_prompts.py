"""Prompts for image-only vs structured-table (ST) MLLM inference."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful assistant. For each question, first think through your reasoning, "
    "then provide an answer. Format your response as:\n"
    "Your reasoning process"
    "<answer>Your final answer</answer>"
)


def build_user_text_image_only(question: str, choices: list[str] | None) -> str:
    parts = ["Look at the table image and answer the question.", "", f"## Question\n{question}"]
    if choices:
        opts = "\n".join(f"- {c}" for c in choices)
        parts.extend(["", "## Choices", opts])
    return "\n".join(parts)


def build_user_text_with_st(table_md: str, question: str, choices: list[str] | None) -> str:
    parts = [
        "Look at the table image. You are also given the same table as structured Markdown.",
        "Use both the image and the Markdown to reason, then answer.",
        "",
        "## Structured table (Markdown)",
        table_md.strip(),
        "",
        f"## Question\n{question}",
    ]
    if choices:
        opts = "\n".join(f"- {c}" for c in choices)
        parts.extend(["", "## Choices", opts])
    return "\n".join(parts)
