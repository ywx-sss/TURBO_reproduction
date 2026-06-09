"""DeepSeek API client for trace generation — reads DEEPSEEK_* from data.env."""

from __future__ import annotations

import re

from config import get_env, load_env_file
from trace_utils import TRACE_USER_TEMPLATE

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"
_ANSWER_BLOCK = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK = re.compile(
    re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE),
    re.DOTALL | re.IGNORECASE,
)


def get_deepseek_config() -> dict[str, str]:
    load_env_file()
    api_key = get_env("DEEPSEEK_API_KEY") or ""
    return {
        "api_key": api_key.strip(),
        "base_url": (get_env("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip(),
        "model": (get_env("DEEPSEEK_MODEL") or "deepseek-reasoner").strip(),
    }


def is_api_configured() -> bool:
    return bool(get_deepseek_config()["api_key"])


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return key[:4] + "..." + key[-4:]


def clean_reasoning_text(text: str) -> str:
    """Remove answer/think wrappers from model output; keep reasoning body only."""
    t = text.strip()
    t = _ANSWER_BLOCK.sub("", t).strip()
    m = _THINK_BLOCK.search(t)
    if m:
        t = m.group(1).strip()
    else:
        t = t.replace(THINK_OPEN, "").replace(THINK_CLOSE, "").strip()
    return t


def _extract_message_text(message) -> str:
    """Support deepseek-reasoner (reasoning_content) and chat models (content)."""
    reasoning = getattr(message, "reasoning_content", None) or ""
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    content = getattr(message, "content", None) or ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                chunks.append(str(block["text"]).strip())
            elif hasattr(block, "text"):
                chunks.append(str(block.text).strip())
        return "\n".join(chunks).strip()
    return ""


def generate_reasoning_trace(table_md: str, question: str, answer: str) -> tuple[str, str]:
    """
    Call DeepSeek OpenAI-compatible API.
    Returns (reasoning_text, teacher_label = model name).
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai: pip install openai") from e

    cfg = get_deepseek_config()
    if not cfg["api_key"]:
        raise RuntimeError("DEEPSEEK_API_KEY not set in data.env")

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    prompt = TRACE_USER_TEMPLATE.format(table_md=table_md, question=question, answer=answer)

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce faithful step-by-step reasoning for table questions. "
                    "Output reasoning text only — no XML tags, no separate final answer line."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    raw = _extract_message_text(resp.choices[0].message)
    if not raw:
        raise RuntimeError("Empty response from DeepSeek API")
    return clean_reasoning_text(raw), cfg["model"]
