"""Local-OCR helper for generating text crop specs.

This is a compact companion to auto_text_crops.py. It is useful when OCR weights
are already downloaded and the environment should not auto-download anything.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def bbox_from_points(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def pad_box(box, pad, width, height):
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


def detect_easyocr(args):
    import easyocr

    reader = easyocr.Reader(
        [x.strip() for x in args.langs.split(",") if x.strip()],
        gpu=args.gpu,
        model_storage_directory=str(args.easyocr_model_dir) if args.easyocr_model_dir else None,
        user_network_directory=str(args.easyocr_user_network_dir) if args.easyocr_user_network_dir else None,
        download_enabled=not args.ocr_local_files_only,
        detect_network=args.easyocr_detect_network,
    )
    out = []
    for points, text, score in reader.readtext(str(args.image), detail=1, paragraph=False):
        if float(score) < args.min_confidence:
            continue
        out.append({"box": bbox_from_points(points), "text": str(text), "score": float(score)})
    return out


def parse_paddle(obj):
    if isinstance(obj, dict):
        polys = obj.get("dt_polys") or obj.get("rec_polys") or obj.get("boxes") or []
        texts = obj.get("rec_texts") or obj.get("texts") or []
        scores = obj.get("rec_scores") or obj.get("scores") or []
        return [
            {
                "box": bbox_from_points(poly),
                "text": str(texts[i]) if i < len(texts) else "",
                "score": float(scores[i]) if i < len(scores) else 1.0,
            }
            for i, poly in enumerate(polys)
            if isinstance(poly, (list, tuple)) and len(poly) == 4
        ]
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and isinstance(obj[0], (list, tuple)) and len(obj[0]) == 4:
            text = str(obj[1][0]) if isinstance(obj[1], (list, tuple)) and obj[1] else ""
            score = float(obj[1][1]) if isinstance(obj[1], (list, tuple)) and len(obj[1]) > 1 else 1.0
            return [{"box": bbox_from_points(obj[0]), "text": text, "score": score}]
        ans = []
        for item in obj:
            ans.extend(parse_paddle(item))
        return ans
    return []


def detect_paddleocr(args):
    from paddleocr import PaddleOCR

    kwargs = {"lang": args.paddle_lang}
    if args.paddle_det_model_dir:
        kwargs["det_model_dir"] = str(args.paddle_det_model_dir)
    if args.paddle_rec_model_dir:
        kwargs["rec_model_dir"] = str(args.paddle_rec_model_dir)
    if args.paddle_cls_model_dir:
        kwargs["cls_model_dir"] = str(args.paddle_cls_model_dir)
    try:
        ocr = PaddleOCR(use_angle_cls=True, use_gpu=args.gpu, **kwargs)
        raw = ocr.ocr(str(args.image), cls=True)
    except TypeError:
        ocr = PaddleOCR(**kwargs)
        raw = ocr.ocr(str(args.image)) if hasattr(ocr, "ocr") else ocr.predict(input=str(args.image))
    return [x for x in parse_paddle(raw) if x["score"] >= args.min_confidence]


def maybe_merge(items, gap):
    if not items:
        return []
    items = sorted(items, key=lambda x: (x["box"][1], x["box"][0]))
    merged = []
    for item in items:
        x0, y0, x1, y1 = item["box"]
        if not merged:
            merged.append(item)
            continue
        px0, py0, px1, py1 = merged[-1]["box"]
        same_line = min(y1, py1) - max(y0, py0) > 0.4 * max(1, min(y1 - y0, py1 - py0))
        near = x0 - px1 <= gap and abs(y0 - py0) <= gap
        if same_line and near:
            merged[-1]["box"] = (min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1))
            merged[-1]["text"] = (merged[-1].get("text", "") + " " + item.get("text", "")).strip()
            merged[-1]["score"] = max(float(merged[-1].get("score", 0)), float(item.get("score", 0)))
        else:
            merged.append(item)
    return merged


def make_probe_command(args, crop_specs):
    cmd = [
        sys.executable,
        "scripts/training_free/flux2_attention_mass_probe.py",
        "--image",
        str(args.image),
        "--image-label",
        args.image_label,
        "--prompt",
        args.prompt,
        "--output-dir",
        str(args.probe_output_dir),
        "--model-type",
        args.model_type,
    ]
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.num_inference_steps is not None:
        cmd += ["--num-inference-steps", str(args.num_inference_steps)]
    if args.guidance_scale is not None:
        cmd += ["--guidance-scale", str(args.guidance_scale)]
    if args.simulate_group_balance:
        cmd.append("--simulate-group-balance")
    if args.apply_group_balance:
        cmd.append("--apply-group-balance")
    for spec in crop_specs:
        cmd += ["--crop", spec]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--engine", choices=["easyocr", "paddleocr"], default="easyocr")
    p.add_argument("--langs", default="en,ch_sim")
    p.add_argument("--paddle-lang", default="ch")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--padding", type=int, default=24)
    p.add_argument("--max-crops", type=int, default=12)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--merge-gap", type=int, default=32)
    p.add_argument("--image-label", default="global")
    p.add_argument("--label-prefix", default="text")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/auto_text_crops_local"))
    p.add_argument("--ocr-local-files-only", action="store_true")
    p.add_argument("--easyocr-model-dir", type=Path)
    p.add_argument("--easyocr-user-network-dir", type=Path)
    p.add_argument("--easyocr-detect-network", choices=["craft", "dbnet18"], default="craft")
    p.add_argument("--paddle-det-model-dir", type=Path)
    p.add_argument("--paddle-rec-model-dir", type=Path)
    p.add_argument("--paddle-cls-model-dir", type=Path)
    p.add_argument("--run-probe", action="store_true")
    p.add_argument("--model-path")
    p.add_argument("--model-type", choices=["auto", "base", "distilled"], default="base")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--prompt", default="Change the background but preserve the small text from the reference.")
    p.add_argument("--num-inference-steps", type=int)
    p.add_argument("--guidance-scale", type=float)
    p.add_argument("--simulate-group-balance", action="store_true")
    p.add_argument("--apply-group-balance", action="store_true")
    p.add_argument("--probe-output-dir", type=Path, default=Path("outputs/probe_auto_text"))
    args = p.parse_args()

    im = Image.open(args.image).convert("RGB")
    width, height = im.size
    items = detect_easyocr(args) if args.engine == "easyocr" else detect_paddleocr(args)
    if args.merge:
        items = maybe_merge(items, args.merge_gap)
    items = sorted(items, key=lambda x: ((x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]), x["score"]), reverse=True)
    items = items[: args.max_crops] if args.max_crops > 0 else items
    items = sorted(items, key=lambda x: (x["box"][1], x["box"][0]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(im)
    crops = []
    records = []
    for i, item in enumerate(items):
        box = pad_box(item["box"], args.padding, width, height)
        label = f"{args.label_prefix}_{i}"
        spec = f"0:{label}:{box[0]},{box[1]},{box[2]},{box[3]}"
        crops.append(spec)
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(0, box[1] - 12)), label, fill="red")
        Image.open(args.image).convert("RGB").crop(box).save(args.output_dir / f"{label}.png")
        records.append({"label": label, "crop_spec": spec, **item, "padded_box": box})
        print(f"--crop {spec}  # conf={item['score']:.3f}, text={item.get('text', '')!r}")

    im.save(args.output_dir / "text_boxes_preview.png")
    (args.output_dir / "text_boxes.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    cmd = make_probe_command(args, crops)
    print("\nProbe command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    if args.run_probe:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
