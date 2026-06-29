# Local base model paths for FLUX.2 [klein] Diffusers tests

Both Diffusers scripts accept local Diffusers-format model directories:

```sh
--model-path /path/to/FLUX.2-klein-base-4B --model-type base --local-files-only
```

Use `--model-type base` for base checkpoints. This sets defaults to:

- `num_inference_steps=50`
- `guidance_scale=4.0`

For faster diagnostics, override them explicitly:

```sh
--num-inference-steps 10 --guidance-scale 4.0
```

The local model directory should contain `model_index.json` and Diffusers component subfolders such as `transformer`, `vae`, `text_encoder`, and `tokenizer`. If you only have original BFL `.safetensors` weights, use the native BFL CLI/environment-variable path instead of the Diffusers scripts.

## Smoke test

```sh
export PYTHONPATH=/path/to/diffusers/src:$PYTHONPATH

python scripts/flux2_klein_diffusers.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "A cat holding a sign that says hello world" \
  --output flux-klein-base.png
```

## Attention mass probe

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Change the background but preserve the small text from the reference." \
  --image ref.png \
  --image-label global \
  --crop 0:text_hr:120,80,520,220 \
  --simulate-group-balance \
  --output-dir outputs/probe_pyramid_sim_balance
```

## Automatic text crops

To avoid manually entering text coordinates, install an OCR backend and let the helper generate `--crop` specs:

```sh
pip install easyocr

python scripts/training_free/auto_text_crops.py \
  --image ref.png \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Change the background but preserve the small text from the reference." \
  --simulate-group-balance \
  --run-probe \
  --probe-output-dir outputs/probe_auto_text
```

See `docs/auto_text_crops.md` for OCR options, `--image-label`, merging many text boxes, and limiting crop count.
