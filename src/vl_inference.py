"""Qwen3-VL local inference (4-bit) for compare_st_vs_image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import configure_hf_mirror

from inference_prompts import SYSTEM_PROMPT, build_user_text_image_only, build_user_text_with_st

DEFAULT_MODEL_4B = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MODEL_2B = "Qwen/Qwen3-VL-2B-Instruct"


def _image_source(path: Path) -> str:
    """Local absolute path (Windows: do NOT use file:// URI — processor mis-parses it)."""
    if not path.is_file():
        raise FileNotFoundError(f"Table image not found: {path}")
    return str(path.resolve())


def _set_processor_pixel_budget(processor: Any, max_visual_tokens: int) -> None:
    """Limit vision tokens for 8GB GPUs (Qwen3-VL uses 32× spatial compression)."""
    patch = 32
    longest = max(256, min(max_visual_tokens, 1280)) * patch * patch
    shortest = 256 * patch * patch
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        processor.image_processor.size = {"longest_edge": longest, "shortest_edge": shortest}


class Qwen3VLRunner:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_4B,
        load_4bit: bool = True,
        lora_path: str | Path | None = None,
        max_visual_tokens: int = 640,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.load_4bit = load_4bit
        self.lora_path = Path(lora_path) if lora_path else None
        self.max_visual_tokens = max_visual_tokens
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        configure_hf_mirror()
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError(
                "Failed to import torch/transformers for VL inference.\n"
                f"Cause: {e}\n"
                "Fix: pip install 'transformers>=4.57.0,<5.0'  (avoid transformers 5.x with torch 2.6)\n"
                "See docs/INFERENCE_COMPARE.md"
            ) from e

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelCls

        kwargs: dict[str, Any] = {"device_map": "auto"}
        if self.load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["dtype"] = torch.float16
        else:
            kwargs["dtype"] = "auto"

        print(f"Loading {self.model_id} (4bit={self.load_4bit}) ...", flush=True)
        proc_path = str(self.lora_path) if self.lora_path and (self.lora_path / "preprocessor_config.json").exists() else self.model_id
        self._processor = AutoProcessor.from_pretrained(proc_path)
        _set_processor_pixel_budget(self._processor, self.max_visual_tokens)
        self._model = ModelCls.from_pretrained(self.model_id, **kwargs)
        if self.lora_path is not None:
            from peft import PeftModel

            print(f"Loading LoRA adapter: {self.lora_path}", flush=True)
            self._model = PeftModel.from_pretrained(self._model, str(self.lora_path))
        self._model.eval()
        print("Model ready.", flush=True)

    def _build_messages(self, image_path: Path, user_text: str) -> list[dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _image_source(image_path)},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def generate(self, image_path: Path, user_text: str) -> str:
        import torch

        self.load()
        assert self._model is not None and self._processor is not None

        messages = self._build_messages(image_path, user_text)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated)
        ]
        text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text.strip()

    def predict_image_only(self, rec: dict, root: Path) -> str:
        img = root / rec["image_path"]
        text = build_user_text_image_only(rec["question"], rec.get("choices"))
        return self.generate(img, text)

    def predict_with_st(self, rec: dict, root: Path) -> str:
        img = root / rec["image_path"]
        text = build_user_text_with_st(rec["table_md"], rec["question"], rec.get("choices"))
        return self.generate(img, text)
