"""Run a small FLUX.2 [klein] smoke test through Diffusers.

Examples:
    # Local base checkpoint, fully offline/local file loading.
    python scripts/flux2_klein_diffusers.py \
        --model-path /path/to/FLUX.2-klein-base-4B \
        --model-type base \
        --local-files-only \
        --prompt "A cat holding a sign that says hello world" \
        --output flux-klein-base.png

    # Single-reference editing with a local base checkpoint.
    python scripts/flux2_klein_diffusers.py \
        --model-path /path/to/FLUX.2-klein-base-4B \
        --model-type base \
        --local-files-only \
        --prompt "Turn the input cat into a dog" \
        --image cat.png \
        --output flux-klein-edit.png

    # Multi-reference editing.
    python scripts/flux2_klein_diffusers.py \
        --model-path /path/to/FLUX.2-klein-base-4B \
        --model-type base \
        --local-files-only \
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


DEFAULT_DISTILLED_MODEL = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"


def _load_diffusers():
    try:
        from diffusers import Flux2KleinPipeline
        from diffusers.utils import load_image
    except ImportError as exc:
        raise SystemExit(
            "Diffusers support is not installed. Install your editable Diffusers checkout with:\n"
            "  git clone https://github.com/huggingface/diffusers\n"
            "  cd diffusers && pip install -e .\n"
            "or install the optional extra from this repo with:\n"
            "  pip install -e '.[diffusers]'"
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


def _resolve_model_arg(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    if args.model:
        return args.model
    if args.model_type == "base":
        return DEFAULT_BASE_MODEL
    return DEFAULT_DISTILLED_MODEL


def _infer_model_type(model_id: str, requested_type: str) -> str:
    if requested_type != "auto":
        return requested_type
    model_name = str(model_id).lower()
    if "base" in model_name:
        return "base"
    return "distilled"


def _resolve_sampling_defaults(args: argparse.Namespace, model_type: str) -> tuple[int, float]:
    if model_type == "base":
        default_steps = 50
        default_guidance = 4.0
    else:
        default_steps = 4
        default_guidance = 1.0

    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else default_steps
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else default_guidance
    return num_inference_steps, guidance_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test FLUX.2 [klein] through the Diffusers pipeline.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Hugging Face model id or local Diffusers model directory. "
            f"Defaults to {DEFAULT_DISTILLED_MODEL} unless --model-type base is set."
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Explicit local Diffusers model directory. This takes precedence over --model. "
            "The directory should contain model_index.json and Diffusers subfolders."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "distilled", "base"),
        default="auto",
        help=(
            "Controls default sampling values. auto infers base when 'base' appears in the model path/name. "
            "base defaults to 50 steps and guidance 4.0; distilled defaults to 4 steps and guidance 1.0."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Pass local_files_only=True to Diffusers from_pretrained for offline local-weight tests.",
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
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Guidance scale. Defaults to 4.0 for base models and 1.0 for distilled models.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps. Defaults to 50 for base models and 4 for distilled models.",
    )
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

    model_id = _resolve_model_arg(args)
    model_type = _infer_model_type(model_id, args.model_type)
    num_inference_steps, guidance_scale = _resolve_sampling_defaults(args, model_type)

    device = args.device
    dtype = _dtype_from_arg(args.dtype, device)

    print(f"Loading model from: {model_id}")
    print(f"Resolved model_type={model_type}, num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale}")

    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    if args.no_cpu_offload:
        pipe = pipe.to(device)
    else:
        pipe.enable_model_cpu_offload()

    generator_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    call_kwargs = {
        "prompt": args.prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
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
