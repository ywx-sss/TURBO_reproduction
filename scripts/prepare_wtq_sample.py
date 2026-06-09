#!/usr/bin/env python3
"""Sample WikiTableQuestions training split → JSONL + rendered table PNGs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "src"
import sys

sys.path.insert(0, str(sys_path))

from wtq_utils import WTQ_ROOT, build_wtq_record, load_wtq_tsv  # noqa: E402

OUT_DIR = ROOT / "data" / "processed"
IMAGES_DIR = OUT_DIR / "wtq_images"
DEFAULT_TSV = WTQ_ROOT / "data" / "training.tsv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WTQ sample (JSONL + table PNGs)")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="wtq_train_500", help="output basename")
    args = parser.parse_args()

    if not args.tsv.is_file():
        raise SystemExit(f"[ERROR] Missing {args.tsv}")

    pool = load_wtq_tsv(args.tsv)
    print(f"WTQ pool: {len(pool)} examples from {args.tsv.name}")

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    selected: list[dict] = []
    skipped = 0
    for item in pool:
        rec = build_wtq_record(
            item,
            wtq_root=WTQ_ROOT,
            project_root=ROOT,
            images_dir=IMAGES_DIR,
        )
        if rec is None:
            skipped += 1
            continue
        selected.append(rec)
        if len(selected) >= args.size:
            break

    if len(selected) < args.size:
        raise SystemExit(f"[ERROR] Only {len(selected)} samples (skipped {skipped})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / f"{args.tag}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "task": "prepare_wtq_sample",
        "dataset": "WTQ",
        "split": "training",
        "sample_size": len(selected),
        "seed": args.seed,
        "source": str(args.tsv),
        "skipped": skipped,
        "images_dir": str(IMAGES_DIR),
        "output": str(jsonl_path),
    }
    stats_path = OUT_DIR / f"{args.tag}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(selected)} samples -> {jsonl_path}")
    print(f"Images -> {IMAGES_DIR}")
    print(f"Stats -> {stats_path}")


if __name__ == "__main__":
    main()
