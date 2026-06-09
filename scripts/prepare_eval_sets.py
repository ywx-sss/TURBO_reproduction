#!/usr/bin/env python3
"""Sample held-out eval sets: 250 TabMWP + 250 WTQ (no overlap with train_500)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtq_utils import WTQ_ROOT, build_wtq_record, load_wtq_tsv  # noqa: E402

TABMWP_RAW = ROOT / "data" / "TabMWP" / "OpenDataLab___TabMWP" / "raw"
TABLES_DIR = TABMWP_RAW / "tables"
OUT_DIR = ROOT / "data" / "processed"
IMAGES_DIR = OUT_DIR / "wtq_eval_images"
DEFAULT_WTQ_TSV = WTQ_ROOT / "data" / "training.tsv"
TRAIN_SEED = 42
TRAIN_SIZE = 500
DEFAULT_EVAL_SIZE = 250


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


def build_tabmwp_record(pid: str, item: dict) -> dict | None:
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


def load_train_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def sample_tabmwp_eval(size: int, seed: int, exclude_ids: set[str]) -> list[dict]:
    train_path = TABMWP_RAW / "problems_train.json"
    if not TABLES_DIR.is_dir():
        raise SystemExit(f"[ERROR] TabMWP tables/ not found under {TABMWP_RAW}")

    data = json.loads(train_path.read_text(encoding="utf-8"))
    train_items = [(pid, item) for pid, item in data.items() if item.get("split") == "train"]

    rng = random.Random(seed)
    rng.shuffle(train_items)

    selected: list[dict] = []
    skipped_train = 0
    missing_img = 0
    for pid, item in train_items:
        if pid in exclude_ids:
            skipped_train += 1
            continue
        rec = build_tabmwp_record(pid, item)
        if rec is None:
            missing_img += 1
            continue
        selected.append(rec)
        if len(selected) >= size:
            break

    if len(selected) < size:
        raise SystemExit(
            f"[ERROR] TabMWP: only {len(selected)} eval samples "
            f"(need {size}; skipped_train={skipped_train}, missing_img={missing_img})"
        )
    return selected


def sample_wtq_eval(tsv: Path, size: int, seed: int, exclude_ids: set[str]) -> list[dict]:
    if not tsv.is_file():
        raise SystemExit(f"[ERROR] Missing WTQ TSV: {tsv}")

    pool = load_wtq_tsv(tsv)
    rng = random.Random(seed)
    rng.shuffle(pool)

    selected: list[dict] = []
    skipped_train = 0
    skipped_build = 0
    for item in pool:
        if item["id"] in exclude_ids:
            skipped_train += 1
            continue
        rec = build_wtq_record(
            item,
            wtq_root=WTQ_ROOT,
            project_root=ROOT,
            images_dir=IMAGES_DIR,
        )
        if rec is None:
            skipped_build += 1
            continue
        selected.append(rec)
        if len(selected) >= size:
            break

    if len(selected) < size:
        raise SystemExit(
            f"[ERROR] WTQ: only {len(selected)} eval samples "
            f"(need {size}; skipped_train={skipped_train}, skipped_build={skipped_build})"
        )
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TabMWP/WTQ eval sets (held out from train_500)")
    parser.add_argument("--size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED, help="same as train sampling seed")
    parser.add_argument("--wtq-tsv", type=Path, default=DEFAULT_WTQ_TSV)
    args = parser.parse_args()

    tabmwp_train_ids = load_train_ids(OUT_DIR / "train_500.jsonl")
    wtq_train_ids = load_train_ids(OUT_DIR / "wtq_train_500.jsonl")

    tabmwp_eval = sample_tabmwp_eval(args.size, args.seed, tabmwp_train_ids)
    wtq_eval = sample_wtq_eval(args.wtq_tsv, args.size, args.seed, wtq_train_ids)

    overlap_tab = {r["id"] for r in tabmwp_eval} & tabmwp_train_ids
    overlap_wtq = {r["id"] for r in wtq_eval} & wtq_train_ids
    if overlap_tab or overlap_wtq:
        raise SystemExit(f"[ERROR] Train/eval overlap: TabMWP={len(overlap_tab)} WTQ={len(overlap_wtq)}")

    tab_path = OUT_DIR / f"tabmwp_eval_{args.size}.jsonl"
    wtq_path = OUT_DIR / f"wtq_eval_{args.size}.jsonl"
    write_jsonl(tab_path, tabmwp_eval)
    write_jsonl(wtq_path, wtq_eval)

    stats = {
        "task": "prepare_eval_sets",
        "eval_size_per_dataset": args.size,
        "seed": args.seed,
        "train_seed": TRAIN_SEED,
        "train_size": TRAIN_SIZE,
        "tabmwp_train_ids_excluded": len(tabmwp_train_ids),
        "wtq_train_ids_excluded": len(wtq_train_ids),
        "tabmwp_eval_count": len(tabmwp_eval),
        "wtq_eval_count": len(wtq_eval),
        "outputs": {
            "tabmwp": str(tab_path),
            "wtq": str(wtq_path),
        },
    }
    stats_path = OUT_DIR / f"eval_{args.size}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"TabMWP eval: {len(tabmwp_eval)} -> {tab_path}")
    print(f"WTQ eval:    {len(wtq_eval)} -> {wtq_path}")
    print(f"Stats:       {stats_path}")


if __name__ == "__main__":
    main()
