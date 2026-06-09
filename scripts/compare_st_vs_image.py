#!/usr/bin/env python3
"""
Compare TABMWP inference: image-only vs image + structured table (ST).

Uses local Qwen3-VL (default 4B, 4-bit). Primary deliverable: bar chart PNG
(reports/compare_st_vs_image.png), aligned with paper Figure 1 style.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer_parse import parse_pred_answer  # noqa: E402
from plot_compare import plot_compare_chart  # noqa: E402
from trace_utils import answers_match  # noqa: E402
from vl_inference import DEFAULT_MODEL_2B, DEFAULT_MODEL_4B, Qwen3VLRunner  # noqa: E402

IN_PATH = ROOT / "data" / "processed" / "train_500.jsonl"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def compare_paths(tag: str) -> dict[str, Path]:
    return {
        "results": PROCESSED / f"{tag}_compare_results.jsonl",
        "stats": PROCESSED / f"{tag}_compare_stats.json",
        "checkpoint": PROCESSED / f"{tag}_compare_checkpoint.jsonl",
        "chart": REPORTS / f"compare_{tag}.png",
    }

CONDITIONS = ("image_only", "with_st")


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


def load_checkpoint(checkpoint: Path) -> dict[str, set[str]]:
    done: dict[str, set[str]] = {c: set() for c in CONDITIONS}
    if not checkpoint.exists():
        return done
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["id"]
        for c in CONDITIONS:
            if row.get(c, {}).get("raw") is not None:
                done[c].add(sid)
    return done


def merge_checkpoint_rows(checkpoint: Path) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    if not checkpoint.exists():
        return merged
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["id"]
        if sid not in merged:
            merged[sid] = {"id": sid, "answer": row.get("answer"), "question": row.get("question")}
        for c in CONDITIONS:
            if c in row and row[c]:
                merged[sid][c] = row[c]
    return merged


def eval_condition(runner: Qwen3VLRunner, rec: dict, condition: str) -> dict:
    if condition == "image_only":
        raw = runner.predict_image_only(rec, ROOT)
    elif condition == "with_st":
        raw = runner.predict_with_st(rec, ROOT)
    else:
        raise ValueError(condition)
    pred = parse_pred_answer(raw)
    correct = answers_match(pred, rec["answer"])
    return {"raw": raw, "pred_answer": pred, "correct": correct}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare image-only vs with-ST (Qwen3-VL)")
    parser.add_argument("--input", type=Path, default=IN_PATH)
    parser.add_argument("--model", default="4b", choices=("2b", "4b"))
    parser.add_argument("--model-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-visual-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--chart-out", type=Path, default=None)
    parser.add_argument("--chart-title", default="")
    parser.add_argument("--conditions", default="both", choices=("both", "image_only", "with_st"))
    parser.add_argument("--plot-only", action="store_true", help="Regenerate chart from stats JSON")
    args = parser.parse_args()

    tag = args.input.stem
    paths = compare_paths(tag)
    OUT_RESULTS = paths["results"]
    OUT_STATS = paths["stats"]
    CHECKPOINT = paths["checkpoint"]
    chart_out = args.chart_out or paths["chart"]
    chart_title = args.chart_title or f"Impact of structured tables (ST) — {tag}"

    if args.plot_only:
        if not OUT_STATS.exists():
            print(f"[ERROR] Missing {OUT_STATS}")
            sys.exit(1)
        stats = json.loads(OUT_STATS.read_text(encoding="utf-8"))
        chart = plot_compare_chart(stats, chart_out, title=chart_title)
        print(f"Chart: {chart}")
        return

    model_id = args.model_id or (DEFAULT_MODEL_2B if args.model == "2b" else DEFAULT_MODEL_4B)
    run_conds = list(CONDITIONS) if args.conditions == "both" else [args.conditions]

    if not args.input.exists():
        print(f"[ERROR] Missing input: {args.input}")
        sys.exit(1)

    records = load_jsonl(args.input)
    if args.limit > 0:
        records = records[: args.limit]

    for rec in records:
        img = ROOT / rec["image_path"]
        if not img.exists():
            print(f"[ERROR] Image not found: {img}")
            sys.exit(1)

    done = load_checkpoint(CHECKPOINT) if args.resume else {c: set() for c in CONDITIONS}
    if not args.resume:
        for p in (OUT_RESULTS, CHECKPOINT):
            if p.exists():
                p.unlink()

    merged = merge_checkpoint_rows(CHECKPOINT) if args.resume else {}

    try:
        import torch

        if not torch.cuda.is_available():
            print("[ERROR] CUDA GPU required. Reinstall torch with cu124 — see docs/INFERENCE_COMPARE.md")
            sys.exit(1)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    except ImportError:
        print("[ERROR] Install VL env: .\\scripts\\install_vl_env.ps1")
        sys.exit(1)

    runner = Qwen3VLRunner(
        model_id=model_id,
        load_4bit=not args.no_4bit,
        max_visual_tokens=args.max_visual_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    t0 = time.time()
    for i, rec in enumerate(records):
        sid = rec["id"]
        row = merged.get(sid, {"id": sid, "answer": rec["answer"], "question": rec["question"]})

        for cond in run_conds:
            if sid in done[cond]:
                continue
            print(f"  [{i+1}/{len(records)}] id={sid} {cond} ...", flush=True)
            try:
                row[cond] = eval_condition(runner, rec, cond)
            except Exception as e:
                row[cond] = {"raw": None, "pred_answer": None, "correct": False, "error": str(e)}
                print(f"    [WARN] {cond} failed: {e}", flush=True)

        merged[sid] = row
        append_jsonl(CHECKPOINT, row)
        if (i + 1) % 5 == 0 or (i + 1) == len(records):
            print(f"  checkpoint {i+1}/{len(records)}", flush=True)

    results = [merged[r["id"]] for r in records if r["id"] in merged]
    OUT_RESULTS.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in results) + ("\n" if results else ""),
        encoding="utf-8",
    )

    stats = {
        "task": "compare_st_vs_image",
        "dataset_tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "load_4bit": not args.no_4bit,
        "input": str(args.input),
        "sample_count": len(records),
        "conditions": run_conds,
        "judge": "answers_match (not Qwen2.5-72B)",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    for cond in run_conds:
        evaluated = [r for r in results if cond in r and r[cond].get("pred_answer") is not None]
        correct = sum(1 for r in evaluated if r[cond].get("correct"))
        stats[cond] = {
            "n": len(evaluated),
            "correct": correct,
            "accuracy": round(correct / len(evaluated), 4) if evaluated else None,
            "errors": sum(1 for r in results if cond in r and r[cond].get("error")),
        }
    if "image_only" in stats and "with_st" in stats:
        a = stats["image_only"].get("accuracy")
        b = stats["with_st"].get("accuracy")
        if a is not None and b is not None:
            stats["delta_with_st_minus_image_only"] = round(b - a, 4)

    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    chart_path = None
    try:
        chart_path = plot_compare_chart(stats, chart_out, title=chart_title)
        stats["chart"] = str(chart_path)
        OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    except ValueError as e:
        print(f"\n[WARN] Chart skipped: {e}")

    print("\n=== Compare complete ===")
    for cond in run_conds:
        s = stats[cond]
        acc = s["accuracy"]
        pct = f"{acc * 100:.1f}%" if acc is not None else "n/a"
        print(f"  {cond}: {s['correct']}/{s['n']} ({pct})")
    if stats.get("delta_with_st_minus_image_only") is not None:
        d = stats["delta_with_st_minus_image_only"]
        print(f"  delta (with_st - image_only): {d:+.4f}")
    if chart_path:
        print(f"Chart:   {chart_path}")
    print(f"Stats:   {OUT_STATS}")
    print(f"Details: {OUT_RESULTS}")


if __name__ == "__main__":
    main()
