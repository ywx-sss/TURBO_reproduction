#!/usr/bin/env python3
"""
QLoRA SFT on bridged reasoning traces (TURBO §4.1).

Trains Qwen3-VL on image + structured table + question -> sft_target.
Default: Qwen3-VL-2B + 4-bit LoRA for RTX 4060 8GB.

IMPORTANT: use .\\scripts\\run_train_sft.ps1 (system `python` may be wrong).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("[train_sft] starting ...", flush=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import configure_hf_mirror  # noqa: E402
from sft_utils import (  # noqa: E402
    VLSFTDataset,
    collate_sft_batch,
    merge_bridged,
    set_processor_pixel_budget,
)

DEFAULT_MODEL_2B = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MODEL_4B = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_TRAIN_PATHS = [
    ROOT / "data" / "processed" / "train_500_bridged.jsonl",
    ROOT / "data" / "processed" / "wtq_train_500_bridged.jsonl",
]
DEFAULT_OUT = ROOT / "models" / "sft_lora"
TRAIN_STATE_NAME = "train_state.json"


def build_sample_indices(dataset_len: int, epochs: float) -> list[int]:
    n_samples = max(1, math.ceil(dataset_len * epochs))
    indices = list(range(dataset_len))
    if n_samples < dataset_len:
        indices = indices[:n_samples]
    elif epochs > 1:
        reps = math.ceil(epochs)
        indices = (indices * reps)[:n_samples]
    return indices


def save_checkpoint(
    model,
    processor,
    output_dir: Path,
    *,
    step_i: int,
    next_sample_index: int,
    loss: float,
    total_steps: int,
    tag: str | None = None,
) -> Path:
    ckpt_dir = output_dir / (tag or f"checkpoint-{step_i}")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ckpt_dir))
    processor.save_pretrained(str(ckpt_dir))
    state = {
        "completed_optimizer_steps": step_i,
        "next_sample_index": next_sample_index,
        "total_optimizer_steps": total_steps,
        "last_loss": loss,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (ckpt_dir / TRAIN_STATE_NAME).write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    latest_dir = output_dir / "checkpoint-latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(ckpt_dir, latest_dir)
    return ckpt_dir


def load_resume_state(resume_from: Path) -> dict:
    state_path = resume_from / TRAIN_STATE_NAME
    if not state_path.is_file():
        print(f"[ERROR] Missing {state_path} — not a train_sft checkpoint")
        sys.exit(1)
    return json.loads(state_path.read_text(encoding="utf-8"))


def run_training_loop(
    model,
    processor,
    dataset: VLSFTDataset,
    pad_id: int,
    output_dir: Path,
    *,
    epochs: float,
    lr: float,
    grad_accum: int,
    save_steps: int,
    resume_from: Path | None,
) -> float:
    import torch

    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
    except Exception:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )

    model.train()
    indices = build_sample_indices(len(dataset), epochs)
    total_steps = max(1, math.ceil(len(indices) / grad_accum))

    start_sample = 0
    step_i = 0
    if resume_from is not None:
        state = load_resume_state(resume_from)
        start_sample = int(state["next_sample_index"])
        step_i = int(state["completed_optimizer_steps"])
        print(
            f"Resuming from {resume_from} "
            f"(step {step_i}/{total_steps}, sample {start_sample}/{len(indices)})",
            flush=True,
        )
        if start_sample >= len(indices):
            print("[WARN] Checkpoint already finished all samples; saving final adapter only.")
            return float(state.get("last_loss", 0.0))

    running_loss = 0.0
    micro = 0
    avg = 0.0

    print(
        f"Starting SFT ({len(indices)} samples, ~{total_steps} optimizer steps, "
        f"grad_accum={grad_accum}, save_steps={save_steps}) ...",
        flush=True,
    )
    optimizer.zero_grad(set_to_none=True)

    for i, idx in enumerate(indices):
        if i < start_sample:
            continue

        batch = collate_sft_batch([dataset[idx]], pad_id)
        batch = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }
        with torch.cuda.amp.autocast(dtype=torch.float16):
            loss = model(**batch).loss / grad_accum
        loss.backward()
        running_loss += float(loss.item()) * grad_accum
        micro += 1
        del batch

        if micro % grad_accum == 0 or (i + 1) == len(indices):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_i += 1
            avg = running_loss / micro
            print(f"  step {step_i}/{total_steps}  loss={avg:.4f}", flush=True)
            running_loss = 0.0
            micro = 0

            if save_steps > 0 and step_i % save_steps == 0:
                ckpt = save_checkpoint(
                    model,
                    processor,
                    output_dir,
                    step_i=step_i,
                    next_sample_index=i + 1,
                    loss=avg,
                    total_steps=total_steps,
                )
                print(f"  checkpoint saved: {ckpt}", flush=True)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return avg if step_i else 0.0


def main() -> None:
    configure_hf_mirror()
    parser = argparse.ArgumentParser(description="QLoRA SFT on bridged tabular reasoning traces")
    parser.add_argument("--train", type=Path, action="append", default=[], help="bridged jsonl (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="2b", choices=("2b", "4b"))
    parser.add_argument("--model-id", default="")
    parser.add_argument("--epochs", type=float, default=0.5, help="fractional epochs OK (default 0.5 for 8GB)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=16, help="micro-batches per optimizer step")
    parser.add_argument(
        "--max-visual-tokens",
        type=int,
        default=256,
        help="lower = less VRAM (try 192 if OOM)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=20,
        help="save LoRA checkpoint every N optimizer steps (0=final only)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="checkpoint dir (e.g. models/sft_lora/checkpoint-latest)",
    )
    parser.add_argument("--limit", type=int, default=0, help="debug: train on first N samples")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    train_paths = args.train or DEFAULT_TRAIN_PATHS
    for p in train_paths:
        if not p.exists():
            print(f"[ERROR] Missing training file: {p}")
            sys.exit(1)

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig
    except ImportError as e:
        print(f"[ERROR] SFT deps missing: {e}")
        print("Install: pip install -r requirements-vl.txt -r requirements-sft.txt")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU required for SFT")
        sys.exit(1)

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelCls

    model_id = args.model_id or (DEFAULT_MODEL_2B if args.model == "2b" else DEFAULT_MODEL_4B)
    records = merge_bridged(train_paths)
    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        print("[ERROR] No training records with sft_target")
        sys.exit(1)

    for rec in records:
        img = ROOT / rec["image_path"]
        if not img.is_file():
            print(f"[ERROR] Missing image: {img}")
            sys.exit(1)

    print(f"Training samples: {len(records)}", flush=True)
    print(f"Model: {model_id}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"Memory settings: max_visual_tokens={args.max_visual_tokens}, "
        f"grad_accum={args.grad_accum}, epochs={args.epochs}",
        flush=True,
    )
    print("[train_sft] loading model (~30s) ...", flush=True)

    resume_from = args.resume_from
    if resume_from is not None and not resume_from.is_dir():
        print(f"[ERROR] --resume-from not found: {resume_from}")
        sys.exit(1)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=args.local_files_only)
    set_processor_pixel_budget(processor, args.max_visual_tokens)

    model = ModelCls.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    if resume_from is not None:
        model = PeftModel.from_pretrained(model, str(resume_from), is_trainable=True)
        print(f"Loaded adapter from {resume_from}", flush=True)
    else:
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = VLSFTDataset(records, processor, ROOT)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    final_loss = run_training_loop(
        model,
        processor,
        dataset,
        pad_id,
        args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        grad_accum=args.grad_accum,
        save_steps=args.save_steps,
        resume_from=resume_from,
    )

    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    stats = {
        "task": "train_sft",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "method": "QLoRA 4-bit (manual loop, no HF Trainer)",
        "train_files": [str(p) for p in train_paths],
        "train_count": len(records),
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "max_visual_tokens": args.max_visual_tokens,
        "save_steps": args.save_steps,
        "resume_from": str(resume_from) if resume_from else None,
        "final_loss": final_loss,
        "output_dir": str(args.output_dir),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    stats_path = args.output_dir / "train_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SFT complete ===")
    print(f"Adapter saved: {args.output_dir}")
    print(f"Stats: {stats_path}")
    if args.save_steps > 0:
        print(f"Latest checkpoint: {args.output_dir / 'checkpoint-latest'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        print(
            "Tip: if OOM, retry with --max-visual-tokens 192 --grad-accum 32 "
            "or resume from models/sft_lora/checkpoint-latest",
            flush=True,
        )
        raise
