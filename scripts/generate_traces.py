#!/usr/bin/env python3
"""
Generate structure-aware reasoning traces + reject sampling (TURBO §4.1).

Reads data.env: DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
Modes: auto | api | fallback
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_env_file  # noqa: E402
from deepseek_client import generate_reasoning_trace, get_deepseek_config, is_api_configured  # noqa: E402
from reject_sampling import reject_reason  # noqa: E402
from trace_utils import solution_fallback_reasoning, wrap_sft_target  # noqa: E402

IN_PATH = ROOT / "data" / "processed" / "train_500.jsonl"


def output_paths(tag: str) -> dict[str, Path]:
    d = ROOT / "data" / "processed"
    return {
        "accepted": d / f"{tag}_bridged.jsonl",
        "rejected": d / f"{tag}_rejected.jsonl",
        "stats": d / f"{tag}_bridged_stats.json",
        "checkpoint": d / f"{tag}_traces_checkpoint.jsonl",
    }


def resolve_mode(requested: str) -> str:
    if requested == "auto":
        return "api" if is_api_configured() else "fallback"
    return requested


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint_ids(checkpoint: Path) -> set[str]:
    if not checkpoint.exists():
        return set()
    return {json.loads(ln)["id"] for ln in checkpoint.read_text(encoding="utf-8").splitlines() if ln.strip()}


def generate_reasoning(rec: dict, mode: str) -> tuple[str, str]:
    if mode == "api":
        return generate_reasoning_trace(rec["table_md"], rec["question"], rec["answer"])
    teacher = "wtq_placeholder" if rec.get("dataset") == "WTQ" else "tabmwp_solution"
    return solution_fallback_reasoning(rec), teacher


def main() -> None:
    load_env_file()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=IN_PATH)
    parser.add_argument("--mode", choices=("auto", "fallback", "api"), default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    tag = args.input.stem
    paths = output_paths(tag)
    OUT_ACCEPTED = paths["accepted"]
    OUT_REJECTED = paths["rejected"]
    OUT_STATS = paths["stats"]
    CHECKPOINT = paths["checkpoint"]

    if not args.input.exists():
        print(f"[ERROR] Missing input: {args.input}")
        sys.exit(1)

    mode = resolve_mode(args.mode)
    if mode == "api" and not is_api_configured():
        print("[ERROR] --mode api requires DEEPSEEK_API_KEY in data.env")
        sys.exit(1)

    cfg = get_deepseek_config() if mode == "api" else {}
    if mode == "api":
        print(f"DeepSeek API: base_url={cfg['base_url']} model={cfg['model']}", flush=True)

    records = load_jsonl(args.input)
    if args.limit > 0:
        records = records[: args.limit]

    done_ids = load_checkpoint_ids(CHECKPOINT) if args.resume else set()
    if not args.resume:
        for p in (OUT_ACCEPTED, OUT_REJECTED, CHECKPOINT):
            if p.exists():
                p.unlink()

    accepted: list[dict] = []
    rejected: list[dict] = []
    reasons = Counter()
    teacher = Counter()
    errors = []

    t0 = time.time()
    for i, rec in enumerate(records):
        if rec["id"] in done_ids:
            continue
        try:
            reasoning, teacher_name = generate_reasoning(rec, mode)
            if mode == "api" and args.sleep > 0:
                time.sleep(args.sleep)
        except Exception as e:
            errors.append({"id": rec["id"], "error": str(e)})
            row = {**rec, "reject_reason": "api_error", "error": str(e)}
            rejected.append(row)
            append_jsonl(OUT_REJECTED, row)
            append_jsonl(CHECKPOINT, row)
            reasons["api_error"] += 1
            continue

        ok, reason = reject_reason(rec, reasoning)
        reasons[reason] += 1
        teacher[teacher_name] += 1

        row = {
            **rec,
            "reasoning": reasoning,
            "sft_target": wrap_sft_target(reasoning, rec["answer"]),
            "teacher": teacher_name,
            "reject_reason": reason if not ok else None,
        }
        append_jsonl(CHECKPOINT, row)

        if ok:
            accepted.append(row)
            append_jsonl(OUT_ACCEPTED, row)
        else:
            rejected.append(row)
            append_jsonl(OUT_REJECTED, row)

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(
                f"  processed {i+1}/{len(records)} | accepted={len(accepted)} rejected={len(rejected)}",
                flush=True,
            )

    elapsed = time.time() - t0
    stats = {
        "task": "generate_traces",
        "dataset_tag": tag,
        "input": str(args.input),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "deepseek_model": cfg.get("model") if mode == "api" else None,
        "deepseek_base_url": cfg.get("base_url") if mode == "api" else None,
        "teacher": dict(teacher),
        "input_count": len(records),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accept_rate": round(len(accepted) / len(records), 4) if records else 0,
        "reject_reasons": dict(reasons),
        "api_errors": errors[:20],
        "elapsed_seconds": round(elapsed, 1),
        "outputs": {"accepted": str(OUT_ACCEPTED), "rejected": str(OUT_REJECTED)},
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Trace generation complete ===")
    print(f"Mode: {mode}")
    print(f"Accepted: {len(accepted)} / {len(records)} ({stats['accept_rate']*100:.1f}%)")
    print(f"Rejected: {len(rejected)}")
    print(f"Reasons: {dict(reasons)}")
    if errors:
        print(f"API errors: {len(errors)}")
    print(f"Stats: {OUT_STATS}")


if __name__ == "__main__":
    main()
