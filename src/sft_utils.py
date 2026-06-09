"""SFT helpers: merge bridged data, build Qwen3-VL chat messages, label masking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inference_prompts import SYSTEM_PROMPT, build_user_text_with_st


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_bridged(paths: list[Path]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        for rec in load_jsonl(path):
            key = f"{rec.get('dataset', 'UNK')}:{rec['id']}"
            if key in seen:
                continue
            if not rec.get("sft_target"):
                continue
            seen.add(key)
            merged.append(rec)
    return merged


def image_abs_path(rec: dict, root: Path) -> str:
    img = root / rec["image_path"]
    if not img.is_file():
        raise FileNotFoundError(f"Missing image: {img}")
    return str(img.resolve())


def build_sft_messages(rec: dict, root: Path) -> list[dict]:
    """Train-time messages: image + structured table + question -> bridged target."""
    img = image_abs_path(rec, root)
    user_text = build_user_text_with_st(rec["table_md"], rec["question"], rec.get("choices"))
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": rec["sft_target"]}]},
    ]


def build_prompt_messages(rec: dict, root: Path, *, with_st: bool = True) -> list[dict]:
    """Inference messages (no assistant turn)."""
    from inference_prompts import build_user_text_image_only

    img = image_abs_path(rec, root)
    if with_st:
        user_text = build_user_text_with_st(rec["table_md"], rec["question"], rec.get("choices"))
    else:
        user_text = build_user_text_image_only(rec["question"], rec.get("choices"))
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def set_processor_pixel_budget(processor: Any, max_visual_tokens: int) -> None:
    patch = 32
    longest = max(256, min(max_visual_tokens, 1280)) * patch * patch
    shortest = 256 * patch * patch
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.size = {"longest_edge": longest, "shortest_edge": shortest}


def tokenize_sft_example(
    processor: Any,
    rec: dict,
    root: Path,
) -> dict[str, Any]:
    """Return model inputs + labels (prompt tokens masked with -100)."""
    full_messages = build_sft_messages(rec, root)
    prompt_messages = full_messages[:2]

    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    img = image_abs_path(rec, root)
    prompt_inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
    full_inputs = processor(text=[full_text], images=[img], return_tensors="pt")

    input_ids = full_inputs["input_ids"][0]
    labels = input_ids.clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:prompt_len] = -100

    out: dict[str, Any] = {"labels": labels}
    for key, val in full_inputs.items():
        if hasattr(val, "shape") and val.shape[0] == 1:
            out[key] = val[0]
        else:
            out[key] = val
    return out


class VLSFTDataset:
    def __init__(self, records: list[dict], processor: Any, root: Path) -> None:
        self.records = records
        self.processor = processor
        self.root = root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return tokenize_sft_example(self.processor, self.records[idx], self.root)


def collate_sft_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = []
    attention_mask = []
    labels = []

    optional_keys = [k for k in batch[0] if k not in ("input_ids", "attention_mask", "labels")]
    stacked: dict[str, list[Any]] = {k: [] for k in optional_keys}

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad = max_len - seq_len
        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
        for key in optional_keys:
            val = item[key]
            if hasattr(val, "shape") and val.dim() == 0:
                stacked[key].append(val.unsqueeze(0))
            else:
                stacked[key].append(val)

    out: dict[str, Any] = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }
    for key, vals in stacked.items():
        if not all(hasattr(v, "shape") for v in vals):
            out[key] = vals[0]
            continue
        if key == "image_grid_thw":
            out[key] = torch.stack([v.reshape(-1) for v in vals], dim=0)
        elif key in ("pixel_values", "pixel_values_videos"):
            out[key] = torch.cat(vals, dim=0)
        elif vals[0].dim() == 0:
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = torch.stack(vals, dim=0)
    return out
