"""Run a small FLUX.2 [klein] smoke test through Diffusers.

Examples:
    python scripts/flux2_klein_diffusers.py \
        --prompt "A cat holding a sign that says hello world" \
        --output flux-klein.png

    python scripts/flux2_klein_diffusers.py \
        --prompt "Turn the input cat into a dog" \
        --image cat.png \
        --output flux-klein-edit.png

    python scripts/flux2_klein_diffusers.py \
        --prompt "Combine the subjects from both references into one scene" \
        --image ref_a.png \
        --image ref_b.png \
        --output flux-klein-multiref.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch


def _load_diffusers():
    try:
        from diffusers import Flux2KleinPipeline
        from diffusers.utils import load_image
    except ImportError as exc:
        raise SystemExit(
            "Diffusers support is not installed. Install it with:\n"
            "  pip install -e '.[diffusers]'\n"
            "or:\n"
            "  pip install git+https://github.com/huggingface/diffusers.git"
        ) from exc

    return Flux2KleinPipeline, load_image


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_from_arg(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        if device in {"cuda", "mps"}:
            return torch.bfloat16
        return torch.float32

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def _load_images(load_image, image_paths: Iterable[str]):
    return [load_image(path) for path in image_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test FLUX.2 [klein] through the Diffusers pipeline.")
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Hugging Face model id. Defaults to the Apache-2.0 FLUX.2 [klein] 4B model.",
    )
    parser.add_argument(
        "--prompt",
        default="A cat holding a sign that says hello world",
        help="Text prompt for text-to-image or image editing.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional reference image path or URL. Repeat for multi-reference editing.",
    )
    parser.add_argument("--output", default="flux-klein.png", help="Output image path.")
    parser.add_argument("--height", type=int, default=1024, help="Text-to-image output height.")
    parser.add_argument("--width", type=int, default=1024, help="Text-to-image output width.")
    parser.add_argument(
        "--force-size-for-edit",
        action="store_true",
        help="Also pass --height/--width when reference images are supplied.",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale for [klein] distilled models.")
    parser.add_argument("--num-inference-steps", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", default=_default_device(), help="Device to run on: cuda, mps, or cpu.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model dtype. Defaults to bfloat16 on CUDA/MPS and float32 on CPU.",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Move the whole pipeline to --device instead of using model CPU offload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Flux2KleinPipeline, load_image = _load_diffusers()

    device = args.device
    dtype = _dtype_from_arg(args.dtype, device)

    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype)
    if args.no_cpu_offload:
        pipe = pipe.to(device)
    else:
        pipe.enable_model_cpu_offload()

    generator_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    call_kwargs = {
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generator": generator,
    }

    if args.image:
        images = _load_images(load_image, args.image)
        call_kwargs["image"] = images[0] if len(images) == 1 else images
        if args.force_size_for_edit:
            call_kwargs["height"] = args.height
            call_kwargs["width"] = args.width
    else:
        call_kwargs["height"] = args.height
        call_kwargs["width"] = args.width

    image = pipe(**call_kwargs).images[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
