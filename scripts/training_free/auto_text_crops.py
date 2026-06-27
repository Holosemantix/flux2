"""Detect text boxes in a reference image and pass them as HR crop references.

This helper is for the training-free reference-pyramid experiments. It detects text
regions, converts them to `--crop 0:text_i:x0,y0,x1,y1` specs, and can either print
the probe command or run `flux2_attention_mass_probe.py` directly.

Optional OCR backends:
    pip install easyocr
    pip install paddleocr

Examples:
    python scripts/training_free/auto_text_crops.py \
        --image ref.png \
        --engine easyocr \
        --dry-run

    python scripts/training_free/auto_text_crops.py \
        --image ref.png \
        --model-path /path/to/FLUX.2-klein-base-4B \
        --model-type base \
        --local-files-only \
        --prompt "Change the background but preserve the small text from the reference." \
        --simulate-group-balance \
        --run-probe
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass
class TextBox:
    x0: int
    y0: int
    x1: int
    y1: int
    text: str = ""
    confidence: float = 1.0
    label: str = ""

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def padded(self, pad: int, width: int, height: int) -> "TextBox":
        return TextBox(
            x0=max(0, self.x0 - pad),
            y0=max(0, self.y0 - pad),
            x1=min(width, self.x1 + pad),
            y1=min(height, self.y1 + pad),
            text=self.text,
            confidence=self.confidence,
            label=self.label,
        )

    def crop_spec(self, image_index: int = 0) -> str:
        return f"{image_index}:{self.label}:{self.x0},{self.y0},{self.x1},{self.y1}"


def _points_to_box(points) -> tuple[int, int, int, int]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))


def detect_text_easyocr(image_path: Path, langs: list[str], min_confidence: float, gpu: bool) -> list[TextBox]:
    try:
        import easyocr
    except ImportError as exc:
        raise SystemExit("easyocr is not installed. Install it with: pip install easyocr") from exc

    reader = easyocr.Reader(langs, gpu=gpu)
    results = reader.readtext(str(image_path), detail=1, paragraph=False)
    boxes: list[TextBox] = []
    for item in results:
        if len(item) < 3:
            continue
        points, text, confidence = item[0], str(item[1]), float(item[2])
        if confidence < min_confidence:
            continue
        x0, y0, x1, y1 = _points_to_box(points)
        boxes.append(TextBox(x0=x0, y0=y0, x1=x1, y1=y1, text=text, confidence=confidence))
    return boxes


def _looks_like_quad(points) -> bool:
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        return False
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return False
    return True


def _parse_paddle_items(obj) -> list[TextBox]:
    """Best-effort parser for PaddleOCR v2/v3 style nested outputs."""

    boxes: list[TextBox] = []
    if isinstance(obj, dict):
        # Some newer PaddleOCR APIs expose arrays under keys such as dt_polys,
        # rec_texts, and rec_scores.
        polys = obj.get("dt_polys") or obj.get("rec_polys") or obj.get("boxes")
        texts = obj.get("rec_texts") or obj.get("texts") or []
        scores = obj.get("rec_scores") or obj.get("scores") or []
        if polys is not None:
            for idx, poly in enumerate(polys):
                if not _looks_like_quad(poly):
                    continue
                text = str(texts[idx]) if idx < len(texts) else ""
                confidence = float(scores[idx]) if idx < len(scores) else 1.0
                x0, y0, x1, y1 = _points_to_box(poly)
                boxes.append(TextBox(x0=x0, y0=y0, x1=x1, y1=y1, text=text, confidence=confidence))
        return boxes

    if not isinstance(obj, (list, tuple)):
        return boxes

    if len(obj) >= 2 and _looks_like_quad(obj[0]) and isinstance(obj[1], (list, tuple)):
        text = str(obj[1][0]) if len(obj[1]) > 0 else ""
        confidence = float(obj[1][1]) if len(obj[1]) > 1 else 1.0
        x0, y0, x1, y1 = _points_to_box(obj[0])
        return [TextBox(x0=x0, y0=y0, x1=x1, y1=y1, text=text, confidence=confidence)]

    for item in obj:
        boxes.extend(_parse_paddle_items(item))
    return boxes


def detect_text_paddleocr(image_path: Path, lang: str, min_confidence: float, gpu: bool) -> list[TextBox]:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit("paddleocr is not installed. Install it with: pip install paddleocr") from exc

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=gpu)
        raw = ocr.ocr(str(image_path), cls=True)
    except TypeError:
        ocr = PaddleOCR(lang=lang)
        raw = ocr.ocr(str(image_path)) if hasattr(ocr, "ocr") else ocr.predict(input=str(image_path))

    boxes = _parse_paddle_items(raw)
    return [box for box in boxes if box.confidence >= min_confidence]


def clamp_and_filter_boxes(
    boxes: Iterable[TextBox],
    *,
    image_width: int,
    image_height: int,
    padding: int,
    min_width: int,
    min_height: int,
    min_area: int,
) -> list[TextBox]:
    filtered: list[TextBox] = []
    for box in boxes:
        box = TextBox(
            x0=max(0, min(image_width, box.x0)),
            y0=max(0, min(image_height, box.y0)),
            x1=max(0, min(image_width, box.x1)),
            y1=max(0, min(image_height, box.y1)),
            text=box.text,
            confidence=box.confidence,
        ).padded(padding, image_width, image_height)
        if box.width < min_width or box.height < min_height or box.area < min_area:
            continue
        filtered.append(box)
    return filtered


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _gap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def should_merge(a: TextBox, b: TextBox, merge_gap: int, y_overlap_threshold: float) -> bool:
    y_overlap = _overlap_len(a.y0, a.y1, b.y0, b.y1)
    min_height = max(1, min(a.height, b.height))
    x_gap = _gap_len(a.x0, a.x1, b.x0, b.x1)
    y_gap = _gap_len(a.y0, a.y1, b.y0, b.y1)

    same_line = y_overlap / min_height >= y_overlap_threshold and x_gap <= merge_gap
    nearby_block = x_gap <= merge_gap and y_gap <= merge_gap
    return same_line or nearby_block


def merge_text_boxes(boxes: list[TextBox], merge_gap: int, y_overlap_threshold: float) -> list[TextBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b.y0, b.x0))
    used = [False] * len(boxes)
    merged: list[TextBox] = []

    for i, box in enumerate(boxes):
        if used[i]:
            continue
        used[i] = True
        group = [box]
        changed = True
        while changed:
            changed = False
            current = TextBox(
                x0=min(g.x0 for g in group),
                y0=min(g.y0 for g in group),
                x1=max(g.x1 for g in group),
                y1=max(g.y1 for g in group),
                text=" ".join(g.text for g in group if g.text),
                confidence=max(g.confidence for g in group),
            )
            for j, candidate in enumerate(boxes):
                if used[j]:
                    continue
                if should_merge(current, candidate, merge_gap, y_overlap_threshold):
                    used[j] = True
                    group.append(candidate)
                    changed = True
        merged.append(
            TextBox(
                x0=min(g.x0 for g in group),
                y0=min(g.y0 for g in group),
                x1=max(g.x1 for g in group),
                y1=max(g.y1 for g in group),
                text=" ".join(g.text for g in group if g.text),
                confidence=max(g.confidence for g in group),
            )
        )
    return sorted(merged, key=lambda b: (b.y0, b.x0))


def assign_labels(boxes: list[TextBox], prefix: str, max_crops: int | None) -> list[TextBox]:
    boxes = sorted(boxes, key=lambda b: (b.y0, b.x0))
    if max_crops is not None and max_crops > 0 and len(boxes) > max_crops:
        # Keep the largest/highest-confidence boxes, then restore reading order.
        boxes = sorted(boxes, key=lambda b: (b.area, b.confidence), reverse=True)[:max_crops]
        boxes = sorted(boxes, key=lambda b: (b.y0, b.x0))
    for idx, box in enumerate(boxes):
        box.label = f"{prefix}_{idx}"
    return boxes


def save_outputs(image_path: Path, boxes: list[TextBox], output_dir: Path, save_crops: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None

    crop_dir = output_dir / "text_crops"
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    for box in boxes:
        draw.rectangle([box.x0, box.y0, box.x1, box.y1], outline="red", width=3)
        draw.text((box.x0, max(0, box.y0 - 12)), box.label, fill="red", font=font)
        if save_crops:
            crop = Image.open(image_path).convert("RGB").crop((box.x0, box.y0, box.x1, box.y1))
            crop.save(crop_dir / f"{box.label}.png")

    image.save(output_dir / "text_boxes_preview.png")
    with (output_dir / "text_boxes.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(box) | {"crop_spec": box.crop_spec(0)} for box in boxes], f, indent=2, ensure_ascii=False)


def build_probe_command(args: argparse.Namespace, boxes: list[TextBox]) -> list[str]:
    command = [
        sys.executable,
        "scripts/training_free/flux2_attention_mass_probe.py",
        "--image",
        str(args.image),
        "--image-label",
        args.image_label,
        "--prompt",
        args.prompt,
        "--output-dir",
        str(args.probe_output_dir or args.output_dir / "probe"),
    ]
    if args.model_path:
        command += ["--model-path", args.model_path]
    if args.model:
        command += ["--model", args.model]
    command += ["--model-type", args.model_type]
    if args.local_files_only:
        command.append("--local-files-only")
    if args.num_inference_steps is not None:
        command += ["--num-inference-steps", str(args.num_inference_steps)]
    if args.guidance_scale is not None:
        command += ["--guidance-scale", str(args.guidance_scale)]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.simulate_group_balance:
        command.append("--simulate-group-balance")
    if args.apply_group_balance:
        command.append("--apply-group-balance")
    for box in boxes:
        command += ["--crop", box.crop_spec(0)]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-detect text boxes and pass them as HR crop refs.")
    parser.add_argument("--image", type=Path, required=True, help="Reference image to inspect.")
    parser.add_argument("--image-label", default="global", help="Label for the whole reference image group.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/auto_text_crops"))
    parser.add_argument("--engine", choices=("easyocr", "paddleocr"), default="easyocr")
    parser.add_argument("--langs", default="en,ch_sim", help="EasyOCR languages, comma-separated.")
    parser.add_argument("--paddle-lang", default="ch", help="PaddleOCR language, e.g. ch or en.")
    parser.add_argument("--gpu", action="store_true", help="Use OCR GPU mode when supported by the backend.")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--padding", type=int, default=24)
    parser.add_argument("--min-width", type=int, default=8)
    parser.add_argument("--min-height", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--merge", action="store_true", help="Merge nearby OCR boxes into larger text blocks.")
    parser.add_argument("--merge-gap", type=int, default=32)
    parser.add_argument("--y-overlap-threshold", type=float, default=0.45)
    parser.add_argument("--max-crops", type=int, default=12, help="Maximum number of text crops to pass to the model.")
    parser.add_argument("--label-prefix", default="text")
    parser.add_argument("--no-save-crops", action="store_true")

    # Probe/model args forwarded when --run-probe is set.
    parser.add_argument("--run-probe", action="store_true", help="Run flux2_attention_mass_probe.py after OCR.")
    parser.add_argument("--dry-run", action="store_true", help="Only print detected crop specs and the probe command.")
    parser.add_argument("--probe-output-dir", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-type", choices=("auto", "distilled", "base"), default="base")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", default="Change the background but preserve the small text from the reference.")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simulate-group-balance", action="store_true")
    parser.add_argument("--apply-group-balance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    image_width, image_height = image.size

    if args.engine == "easyocr":
        langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
        boxes = detect_text_easyocr(args.image, langs, args.min_confidence, args.gpu)
    else:
        boxes = detect_text_paddleocr(args.image, args.paddle_lang, args.min_confidence, args.gpu)

    boxes = clamp_and_filter_boxes(
        boxes,
        image_width=image_width,
        image_height=image_height,
        padding=args.padding,
        min_width=args.min_width,
        min_height=args.min_height,
        min_area=args.min_area,
    )
    if args.merge:
        boxes = merge_text_boxes(boxes, args.merge_gap, args.y_overlap_threshold)
    boxes = assign_labels(boxes, args.label_prefix, args.max_crops)

    save_outputs(args.image, boxes, args.output_dir, save_crops=not args.no_save_crops)

    print(f"Detected {len(boxes)} text crop(s).")
    for box in boxes:
        print(f"  --crop {box.crop_spec(0)}  # conf={box.confidence:.3f}, text={box.text!r}")

    command = build_probe_command(args, boxes)
    print("\nProbe command:")
    print(" ".join(shlex.quote(part) for part in command))

    if args.run_probe and not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
