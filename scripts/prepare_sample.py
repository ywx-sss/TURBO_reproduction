#!/usr/bin/env python3
"""Sample 500 TABMWP train examples with Markdown table + image paths."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "TabMWP" / "OpenDataLab___TabMWP" / "raw"
TABLES_DIR = RAW / "tables"
OUT_DIR = ROOT / "data" / "processed"
SAMPLE_SIZE = 500
SEED = 42


def table_to_markdown(table_text: str, title: str | None = None) -> str:
    lines = [ln.strip() for ln in (table_text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    parts = []
    if title:
        parts.append(f"### {title}\n")
    header = [c.strip() for c in lines[0].split("|")]
    parts.append("| " + " | ".join(header) + " |")
    parts.append("| " + " | ".join(["---"] * len(header)) + " |")
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split("|")]
        if len(cells) == len(header):
            parts.append("| " + " | ".join(cells) + " |")
    return "\n".join(parts)


def format_answer(item: dict) -> str:
    ans = str(item.get("answer", "")).strip()
    unit = item.get("unit")
    if unit and str(unit).lower() not in ("none", "null", ""):
        return f"{ans} {unit}".strip()
    return ans


def build_record(pid: str, item: dict) -> dict | None:
    img = TABLES_DIR / f"{pid}.png"
    if not img.exists():
        return None
    return {
        "id": pid,
        "dataset": "TABMWP",
        "split": item.get("split", "train"),
        "question": item.get("question", "").strip(),
        "choices": item.get("choices") or None,
        "answer": format_answer(item),
        "table_title": item.get("table_title"),
        "table_md": table_to_markdown(item.get("table", ""), item.get("table_title")),
        "table_raw": item.get("table", ""),
        "image_path": str(img.relative_to(ROOT)).replace("\\", "/"),
        "solution_reference": item.get("solution") or "",
        "ques_type": item.get("ques_type"),
        "ans_type": item.get("ans_type"),
        "grade": item.get("grade"),
        "row_num": item.get("row_num"),
        "column_num": item.get("column_num"),
    }


def main() -> None:
    train_path = RAW / "problems_train.json"
    if not TABLES_DIR.is_dir():
        raise SystemExit(f"[ERROR] tables/ not found. Unzip tables.zip under {RAW}")

    data = json.loads(train_path.read_text(encoding="utf-8"))
    train_items = [(pid, item) for pid, item in data.items() if item.get("split") == "train"]
    print(f"Train pool: {len(train_items)}")

    rng = random.Random(SEED)
    rng.shuffle(train_items)

    selected: list[dict] = []
    missing_img = 0
    for pid, item in train_items:
        rec = build_record(pid, item)
        if rec is None:
            missing_img += 1
            continue
        selected.append(rec)
        if len(selected) >= SAMPLE_SIZE:
            break

    if len(selected) < SAMPLE_SIZE:
        raise SystemExit(f"[ERROR] Only {len(selected)} samples (missing images: {missing_img})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "train_500.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "task": "prepare_sample",
        "sample_size": len(selected),
        "seed": SEED,
        "source": str(train_path),
        "tables_dir": str(TABLES_DIR),
        "missing_image_skipped": missing_img,
        "ques_type": {},
        "ans_type": {},
        "output": str(jsonl_path),
    }
    for rec in selected:
        stats["ques_type"][rec["ques_type"]] = stats["ques_type"].get(rec["ques_type"], 0) + 1
        stats["ans_type"][rec["ans_type"]] = stats["ans_type"].get(rec["ans_type"], 0) + 1

    stats_path = OUT_DIR / "train_500_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(selected)} samples -> {jsonl_path}")
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()
