"""WikiTableQuestions (WTQ) helpers: TSV parse, CSV→Markdown, table→PNG."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

WTQ_ROOT = Path(__file__).resolve().parents[1] / "data" / "WikiTableQuestions-1.0.2-compact" / "WikiTableQuestions"


def unescape_wtq_field(text: str) -> str:
    """Reverse WTQ TSV escaping: \\n \\p \\\\"""
    return (
        text.replace("\\p", "|")
        .replace("\\n", "\n")
        .replace("\\\\", "\\")
        .strip()
    )


def load_wtq_tsv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            rec = dict(zip(header, parts[:4]))
            rows.append(
                {
                    "id": rec["id"],
                    "question": unescape_wtq_field(rec["utterance"]),
                    "table_context": rec["context"].strip(),
                    "answer": unescape_wtq_field(rec["targetValue"]),
                }
            )
    return rows


def read_csv_table(csv_path: Path) -> list[list[str]]:
    text = csv_path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text))
    return [[cell.strip() for cell in row] for row in reader if any(c.strip() for c in row)]


def csv_rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    norm = [r + [""] * (ncols - len(r)) for r in rows]
    header = norm[0]
    parts = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for row in norm[1:]:
        parts.append("| " + " | ".join(row) + " |")
    return "\n".join(parts)


def csv_to_raw_text(rows: list[list[str]]) -> str:
    return "\n".join(",".join(row) for row in rows)


def render_table_png(rows: list[list[str]], out_path: Path, *, max_rows: int = 40, max_cols: int = 12) -> None:
    """Render CSV grid as PNG for Qwen3-VL (matplotlib)."""
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("empty table")
    clipped = [r[:max_cols] for r in rows[:max_rows]]
    ncols = max(len(r) for r in clipped)
    clipped = [r + [""] * (ncols - len(r)) for r in clipped]
    if len(clipped) == 1:
        clipped.append([""] * ncols)

    fig_w = min(24, max(6, ncols * 1.8))
    fig_h = min(28, max(2.5, len(clipped) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    ax.axis("off")
    tbl = ax.table(
        cellText=clipped[1:],
        colLabels=clipped[0],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)


def safe_image_name(example_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", example_id)


def build_wtq_record(
    item: dict,
    *,
    wtq_root: Path,
    project_root: Path,
    images_dir: Path,
) -> dict | None:
    csv_rel = item["table_context"]
    csv_path = wtq_root / csv_rel.replace("/", "\\") if "\\" not in csv_rel else wtq_root / csv_rel
    if not csv_path.is_file():
        csv_path = wtq_root / csv_rel
    if not csv_path.is_file():
        return None

    rows = read_csv_table(csv_path)
    if not rows:
        return None

    img_name = f"{safe_image_name(item['id'])}.png"
    img_abs = images_dir / img_name
    render_table_png(rows, img_abs)

    return {
        "id": item["id"],
        "dataset": "WTQ",
        "split": "training",
        "question": item["question"],
        "choices": None,
        "answer": item["answer"],
        "table_title": None,
        "table_md": csv_rows_to_markdown(rows),
        "table_raw": csv_to_raw_text(rows),
        "table_context": csv_rel,
        "image_path": str(img_abs.relative_to(project_root)).replace("\\", "/"),
        "solution_reference": "",
        "ques_type": "free_text",
        "ans_type": "extractive_text",
    }
