# Running FLUX.2 [klein] with Diffusers

This repository contains the native BFL inference implementation. For quick compatibility checks against Hugging Face Diffusers, install the optional Diffusers extra and run the smoke-test script added in `scripts/flux2_klein_diffusers.py`.

## Install

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[diffusers]" --extra-index-url https://download.pytorch.org/whl/cu129 --no-cache-dir
```

Alternatively, install Diffusers directly from `main`:

```sh
pip install git+https://github.com/huggingface/diffusers.git
```

If your environment does not already have Hugging Face credentials, log in before running gated or rate-limited model downloads:

```sh
hf auth login
```

## Text-to-image smoke test

```sh
python scripts/flux2_klein_diffusers.py \
  --prompt "A cat holding a sign that says hello world" \
  --height 1024 \
  --width 1024 \
  --guidance-scale 1.0 \
  --num-inference-steps 4 \
  --seed 0 \
  --output flux-klein.png
```

The script defaults to `black-forest-labs/FLUX.2-klein-4B`, `bfloat16` on CUDA/MPS, 4 inference steps, guidance scale `1.0`, and model CPU offload.

## Single-reference editing smoke test

```sh
python scripts/flux2_klein_diffusers.py \
  --prompt "Turn the input cat into a dog" \
  --image cat.png \
  --output flux-klein-edit.png
```

`--image` accepts a local path or URL supported by `diffusers.utils.load_image`.

## Multi-reference editing smoke test

Pass `--image` multiple times:

```sh
python scripts/flux2_klein_diffusers.py \
  --prompt "Combine the subjects from both references into one scene" \
  --image ref_a.png \
  --image ref_b.png \
  --output flux-klein-multiref.png
```

## Training-free attention mass probe

For the first reference-token-density diagnostic, use:

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --prompt "Change the scene but preserve the important details from the reference." \
  --image ref.png \
  --image-label global \
  --output-dir outputs/probe_baseline
```

The probe can add HR crop references with `--crop INDEX:LABEL:X0,Y0,X1,Y1`, record grouped attention mass, simulate group-size balancing, and optionally apply group-size balancing to the real attention call.

See [docs/training_free_attention_mass_probe.md](training_free_attention_mass_probe.md) for the step-by-step baseline, reference-pyramid, simulated-balancing, and actual-balancing experiments.

## Useful flags

- `--model`: change the Hugging Face model id, for example to a 9B or base checkpoint.
- `--no-cpu-offload`: move the full pipeline to `--device` instead of using `enable_model_cpu_offload()`.
- `--dtype`: choose `auto`, `bfloat16`, `float16`, or `float32`.
- `--force-size-for-edit`: also pass `height` and `width` during reference-image editing.

## Why this lives behind an optional extra

The native inference path in this repository should stay self-contained and reproducible. Diffusers support moves quickly, and the FLUX.2 [klein] model card recommends installing Diffusers from GitHub `main` for the `Flux2KleinPipeline`. Keeping it as an optional extra avoids forcing every native inference install to track Diffusers `main`.
