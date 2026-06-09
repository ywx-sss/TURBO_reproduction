#!/usr/bin/env python3
"""
Evaluate base vs SFT-tuned Qwen3-VL on held-out TabMWP/WTQ eval sets.

Paper test setting: image-only (no structured table at inference).
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

from config import configure_hf_mirror  # noqa: E402
from answer_parse import parse_pred_answer  # noqa: E402
from plot_compare import plot_compare_chart  # noqa: E402
from trace_utils import answers_match  # noqa: E402
from vl_inference import DEFAULT_MODEL_2B, DEFAULT_MODEL_4B, Qwen3VLRunner  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
DEFAULT_EVAL = [
    PROCESSED / "tabmwp_eval_250.jsonl",
    PROCESSED / "wtq_eval_250.jsonl",
]


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


def eval_paths(tag: str) -> dict[str, Path]:
    return {
        "results": PROCESSED / f"{tag}_sft_eval_results.jsonl",
        "stats": PROCESSED / f"{tag}_sft_eval_stats.json",
        "checkpoint": PROCESSED / f"{tag}_sft_eval_checkpoint.jsonl",
        "chart": REPORTS / f"sft_eval_{tag}.png",
    }


def load_checkpoint(checkpoint: Path) -> dict[str, set[str]]:
    done: dict[str, set[str]] = {"base": set(), "sft": set()}
    if not checkpoint.exists():
        return done
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["id"]
        for cond in done:
            if row.get(cond, {}).get("raw") is not None:
                done[cond].add(sid)
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
            merged[sid] = {
                "id": sid,
                "dataset": row.get("dataset"),
                "answer": row.get("answer"),
                "question": row.get("question"),
            }
        for cond in ("base", "sft"):
            if cond in row and row[cond]:
                merged[sid][cond] = row[cond]
    return merged


def predict(runner: Qwen3VLRunner, rec: dict) -> dict:
    raw = runner.predict_image_only(rec, ROOT)
    pred = parse_pred_answer(raw)
    correct = answers_match(pred, rec["answer"])
    return {"raw": raw, "pred_answer": pred, "correct": correct}


def summarize(results: list[dict], cond: str) -> dict:
    evaluated = [r for r in results if cond in r and r[cond].get("pred_answer") is not None]
    correct = sum(1 for r in evaluated if r[cond].get("correct"))
    return {
        "n": len(evaluated),
        "correct": correct,
        "accuracy": round(correct / len(evaluated), 4) if evaluated else None,
        "errors": sum(1 for r in results if cond in r and r[cond].get("error")),
    }


def run_eval(
    runner: Qwen3VLRunner,
    records: list[dict],
    cond: str,
    done: set[str],
    merged: dict[str, dict],
    checkpoint: Path,
) -> None:
    for i, rec in enumerate(records):
        sid = rec["id"]
        if sid in done:
            continue
        row = merged.get(
            sid,
            {
                "id": sid,
                "dataset": rec.get("dataset"),
                "answer": rec["answer"],
                "question": rec["question"],
            },
        )
        print(f"  [{i+1}/{len(records)}] id={sid} {cond} ...", flush=True)
        try:
            row[cond] = predict(runner, rec)
        except Exception as e:
            row[cond] = {"raw": None, "pred_answer": None, "correct": False, "error": str(e)}
            print(f"    [WARN] {cond} failed: {e}", flush=True)
        merged[sid] = row
        append_jsonl(checkpoint, row)


def main() -> None:
    configure_hf_mirror()
    parser = argparse.ArgumentParser(description="Compare base vs SFT model on eval sets")
    parser.add_argument("--input", type=Path, action="append", default=[], help="eval jsonl (repeatable)")
    parser.add_argument("--lora-dir", type=Path, default=ROOT / "models" / "sft_lora")
    parser.add_argument("--model", default="2b", choices=("2b", "4b"))
    parser.add_argument("--model-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--max-visual-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    eval_paths_list = args.input or DEFAULT_EVAL
    for p in eval_paths_list:
        if not p.exists():
            print(f"[ERROR] Missing eval set: {p}")
            print("Run: python scripts/prepare_eval_sets.py")
            sys.exit(1)

    model_id = args.model_id or (DEFAULT_MODEL_2B if args.model == "2b" else DEFAULT_MODEL_4B)
    lora_dir = args.lora_dir if args.lora_dir.is_dir() else None
    if not args.skip_sft and lora_dir is None:
        print(f"[ERROR] LoRA adapter not found: {args.lora_dir}")
        print("Train first: python scripts/train_sft.py")
        sys.exit(1)

    try:
        import torch

        if not torch.cuda.is_available():
            print("[ERROR] CUDA GPU required")
            sys.exit(1)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    except ImportError:
        print("[ERROR] Install VL env: .\\scripts\\install_vl_env.ps1")
        sys.exit(1)

    all_records: list[dict] = []
    for p in eval_paths_list:
        rows = load_jsonl(p)
        if args.limit > 0:
            rows = rows[: args.limit]
        all_records.extend(rows)

    for rec in all_records:
        img = ROOT / rec["image_path"]
        if not img.exists():
            print(f"[ERROR] Image not found: {img}")
            sys.exit(1)

    tag = "combined" if len(eval_paths_list) > 1 else eval_paths_list[0].stem
    paths = eval_paths(tag)
    checkpoint = paths["checkpoint"]
    done = load_checkpoint(checkpoint) if args.resume else {"base": set(), "sft": set()}
    if not args.resume:
        for p in (paths["results"], checkpoint):
            if p.exists():
                p.unlink()
    merged = merge_checkpoint_rows(checkpoint) if args.resume else {}

    t0 = time.time()

    if not args.skip_base:
        print(f"\n=== Base model ({model_id}) ===", flush=True)
        base_runner = Qwen3VLRunner(
            model_id=model_id,
            load_4bit=True,
            max_visual_tokens=args.max_visual_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        run_eval(base_runner, all_records, "base", done["base"], merged, checkpoint)
        del base_runner

    if not args.skip_sft and lora_dir is not None:
        print(f"\n=== SFT model ({model_id} + {lora_dir}) ===", flush=True)
        sft_runner = Qwen3VLRunner(
            model_id=model_id,
            load_4bit=True,
            lora_path=lora_dir,
            max_visual_tokens=args.max_visual_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        run_eval(sft_runner, all_records, "sft", done["sft"], merged, checkpoint)
        del sft_runner

    results = [merged[r["id"]] for r in all_records if r["id"] in merged]
    paths["results"].write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in results) + ("\n" if results else ""),
        encoding="utf-8",
    )

    stats = {
        "task": "eval_sft",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "lora_dir": str(lora_dir) if lora_dir else None,
        "eval_inputs": [str(p) for p in eval_paths_list],
        "sample_count": len(all_records),
        "condition": "image_only",
        "judge": "answers_match",
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    per_dataset: dict[str, dict] = {}
    for ds in sorted({r.get("dataset", "UNK") for r in all_records}):
        ds_results = [merged[r["id"]] for r in all_records if r.get("dataset") == ds and r["id"] in merged]
        per_dataset[ds] = {
            "base": summarize(ds_results, "base"),
            "sft": summarize(ds_results, "sft"),
        }
        b, s = per_dataset[ds]["base"].get("accuracy"), per_dataset[ds]["sft"].get("accuracy")
        if b is not None and s is not None:
            per_dataset[ds]["delta_sft_minus_base"] = round(s - b, 4)

    stats["overall"] = {
        "base": summarize(results, "base"),
        "sft": summarize(results, "sft"),
    }
    ob, os_ = stats["overall"]["base"].get("accuracy"), stats["overall"]["sft"].get("accuracy")
    if ob is not None and os_ is not None:
        stats["overall"]["delta_sft_minus_base"] = round(os_ - ob, 4)
    stats["per_dataset"] = per_dataset

    paths["stats"].write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    chart_stats = {
        "image_only": stats["overall"].get("base", {}),
        "with_st": stats["overall"].get("sft", {}),
        "sample_count": len(all_records),
    }
    try:
        chart = plot_compare_chart(
            chart_stats,
            paths["chart"],
            title=f"SFT eval (image-only): base vs tuned — {tag}",
        )
        stats["chart"] = str(chart)
        paths["stats"].write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    except ValueError as e:
        print(f"[WARN] Chart skipped: {e}")

    print("\n=== SFT eval complete ===")
    for ds, ds_stats in per_dataset.items():
        b = ds_stats["base"]
        s = ds_stats["sft"]
        bp = f"{b['accuracy']*100:.1f}%" if b.get("accuracy") is not None else "n/a"
        sp = f"{s['accuracy']*100:.1f}%" if s.get("accuracy") is not None else "n/a"
        print(f"  {ds}: base {b['correct']}/{b['n']} ({bp}) -> sft {s['correct']}/{s['n']} ({sp})")
    if stats["overall"].get("delta_sft_minus_base") is not None:
        d = stats["overall"]["delta_sft_minus_base"]
        print(f"  overall delta (sft - base): {d:+.4f}")
    print(f"Stats: {paths['stats']}")


if __name__ == "__main__":
    main()
