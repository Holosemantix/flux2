# Running FLUX.2 [klein] with Diffusers

This repository contains the native BFL inference implementation. For quick compatibility checks against Hugging Face Diffusers, install or compile an editable Diffusers checkout and run the smoke-test script added in `scripts/flux2_klein_diffusers.py`.

## Install with editable Diffusers

```sh
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .

cd /path/to/flux2
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129 --no-cache-dir
```

If multiple Diffusers installs exist, force your editable checkout to be imported first:

```sh
export PYTHONPATH=/path/to/diffusers/src:$PYTHONPATH
```

The optional extra is still available if you do not need to edit Diffusers internals:

```sh
pip install -e ".[diffusers]" --extra-index-url https://download.pytorch.org/whl/cu129 --no-cache-dir
```

If your environment does not already have Hugging Face credentials, log in before running gated or rate-limited model downloads:

```sh
hf auth login
```

## Local model paths

Both Diffusers scripts accept a local model directory:

```sh
--model-path /path/to/FLUX.2-klein-base-4B --local-files-only
```

The local directory should be a Diffusers-format checkpoint, normally containing `model_index.json` and component subfolders such as `transformer`, `vae`, `text_encoder`, and `tokenizer`. If your files are only original BFL `.safetensors` weights, use the native BFL CLI path/env-var flow instead of this Diffusers script.

`--model-type` controls default sampling values:

| `--model-type` | Default steps | Default guidance |
|---|---:|---:|
| `base` | 50 | 4.0 |
| `distilled` | 4 | 1.0 |
| `auto` | inferred from model path/name | inferred |

For local base checkpoints whose folder name does not contain `base`, pass `--model-type base` explicitly.

## Text-to-image smoke test with a local base model

```sh
python scripts/flux2_klein_diffusers.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "A cat holding a sign that says hello world" \
  --height 1024 \
  --width 1024 \
  --seed 0 \
  --output flux-klein-base.png
```

Override base defaults when needed:

```sh
python scripts/flux2_klein_diffusers.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --num-inference-steps 20 \
  --guidance-scale 3.5 \
  --prompt "A cat holding a sign that says hello world" \
  --output flux-klein-base-fast.png
```

## Single-reference editing smoke test

```sh
python scripts/flux2_klein_diffusers.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Turn the input cat into a dog" \
  --image cat.png \
  --output flux-klein-edit.png
```

`--image` accepts a local path or URL supported by `diffusers.utils.load_image`.

## Multi-reference editing smoke test

Pass `--image` multiple times:

```sh
python scripts/flux2_klein_diffusers.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Combine the subjects from both references into one scene" \
  --image ref_a.png \
  --image ref_b.png \
  --output flux-klein-multiref.png
```

## Training-free attention mass probe

For the first reference-token-density diagnostic, use the same local base path:

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Change the scene but preserve the important details from the reference." \
  --image ref.png \
  --image-label global \
  --output-dir outputs/probe_baseline
```

The probe can add HR crop references with `--crop INDEX:LABEL:X0,Y0,X1,Y1`, record grouped attention mass, simulate group-size balancing, and optionally apply group-size balancing to the real attention call.

For text-heavy images, generate these crop specs automatically:

```sh
python scripts/training_free/auto_text_crops.py \
  --image ref.png \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --simulate-group-balance \
  --run-probe
```

See [docs/training_free_attention_mass_probe.md](training_free_attention_mass_probe.md) for the step-by-step baseline, reference-pyramid, simulated-balancing, and actual-balancing experiments. See [docs/auto_text_crops.md](auto_text_crops.md) for OCR setup and multi-text-region handling.

## Useful flags

- `--model-path`: explicit local Diffusers checkpoint directory. Takes precedence over `--model`.
- `--model`: Hugging Face model id or local Diffusers directory.
- `--model-type`: `auto`, `base`, or `distilled`; controls default steps/guidance.
- `--local-files-only`: pass `local_files_only=True` to Diffusers `from_pretrained`.
- `--no-cpu-offload`: move the full pipeline to `--device` instead of using `enable_model_cpu_offload()`.
- `--dtype`: choose `auto`, `bfloat16`, `float16`, or `float32`.
- `--force-size-for-edit`: also pass `height` and `width` during reference-image editing.

## Why this lives behind an optional extra

The native inference path in this repository should stay self-contained and reproducible. Diffusers support moves quickly, and the FLUX.2 [klein] model card recommends installing Diffusers for the `Flux2KleinPipeline`. Keeping it as an optional path avoids forcing every native inference install to track Diffusers `main`.
