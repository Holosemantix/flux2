# Training-free attention mass probe for FLUX.2 [klein]

This guide covers the first low-resource experiments for reference-aware dynamic resolution editing. The goal is to test the hypothesis that reference token density changes attention mass before doing any LoRA or post-training.

The probe script is:

```sh
scripts/training_free/flux2_attention_mass_probe.py
```

It works by replacing the FLUX.2 attention processors at runtime. You can use it with an editable local Diffusers checkout, and later move the same logic directly into Diffusers internals if needed.

For text-heavy references, the OCR helper can generate crop specs automatically:

```sh
scripts/training_free/auto_text_crops.py
```

See [docs/auto_text_crops.md](auto_text_crops.md) for OCR setup, `--image-label`, and multi-text-region handling.

## 0. Editable Diffusers setup

```sh
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
cd /path/to/flux2
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129 --no-cache-dir
```

If you want to force the local checkout to be used first:

```sh
export PYTHONPATH=/path/to/diffusers/src:$PYTHONPATH
```

## 0.1 Local base model paths

The probe accepts local Diffusers-format model directories:

```sh
--model-path /path/to/FLUX.2-klein-base-4B --model-type base --local-files-only
```

The local directory should contain `model_index.json` and Diffusers component subfolders. If the folder name does not include `base`, pass `--model-type base` explicitly so the script uses base defaults:

- `num_inference_steps=50`
- `guidance_scale=4.0`

You can override both with `--num-inference-steps` and `--guidance-scale` for faster diagnostics.

## 1. Baseline: whole reference only

Run with one reference image and record the first double-stream and first single-stream attention layers:

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Change the scene but preserve the important details from the reference." \
  --image ref.png \
  --image-label global \
  --output-dir outputs/probe_baseline \
  --seed 0
```

For faster first-pass debugging on a base model, override the steps:

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --num-inference-steps 10 \
  --guidance-scale 4.0 \
  --prompt "Change the scene but preserve the important details from the reference." \
  --image ref.png \
  --image-label global \
  --output-dir outputs/probe_baseline_fast \
  --seed 0
```

Outputs:

- `sample.png`: generated image
- `attention_mass.jsonl`: per-record grouped attention mass
- `attention_mass_summary.csv`: flattened CSV summary
- `layout.json`: model path, base/distilled mode, sampling values, text/output/ref token lengths, and installed processor names

The most useful columns in the CSV are:

- `prefix`: `raw`, or `sim_group_balance` if simulated balancing is enabled
- `group`: `text`, `output`, or `ref:<label>`
- `mass`: average attention mass assigned to that group
- `tokens`: token count in that group
- `mass_per_token`: `mass / tokens`

## 2. Reference pyramid: whole reference + HR crop

Add a high-resolution crop as another reference group. Crop format is:

```text
INDEX:LABEL:X0,Y0,X1,Y1
```

Example:

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Change the background but preserve the small text from the reference." \
  --image ref.png \
  --image-label global \
  --crop 0:text_hr:120,80,520,220 \
  --output-dir outputs/probe_pyramid \
  --seed 0
```

This tests whether the added HR crop receives disproportionate attention mass simply because it contributes more tokens.

To generate text crops automatically:

```sh
python scripts/training_free/auto_text_crops.py \
  --image ref.png \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Change the background but preserve the small text from the reference." \
  --run-probe
```

## 3. Simulated group-size balancing

This does not change the generated image. It records raw attention and also records a simulated group-balanced version where every key in a reference group receives `-log(num_group_tokens)` as a logit correction.

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
  --output-dir outputs/probe_pyramid_sim_balance \
  --seed 0
```

Compare `raw` vs `sim_group_balance` rows in `attention_mass_summary.csv`. If the HR crop's group mass drops after simulated balancing while `mass_per_token` remains reasonable, that supports the token-count-bias hypothesis.

## 4. Actual group-size balancing

This applies the group-size logit correction to the real attention call. It is more invasive and may affect quality, so treat it as a diagnostic intervention.

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Change the background but preserve the small text from the reference." \
  --image ref.png \
  --image-label global \
  --crop 0:text_hr:120,80,520,220 \
  --apply-group-balance \
  --output-dir outputs/probe_pyramid_apply_balance \
  --seed 0
```

If your local attention backend rejects additive masks, force native attention before importing Diffusers:

```sh
export DIFFUSERS_ATTN_BACKEND=native
```

## 5. Record more layers

Default layers:

- `transformer_blocks.0.attn`
- `single_transformer_blocks.0.attn`

Record specific layers:

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --image ref.png \
  --record-layer transformer_blocks.3.attn \
  --record-layer single_transformer_blocks.12.attn \
  --max-records 32 \
  --output-dir outputs/probe_layers
```

Record every attention layer until `--max-records` is hit:

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --image ref.png \
  --record-all-layers \
  --max-records 64 \
  --output-dir outputs/probe_all_layers
```

For base models with classifier-free guidance, the transformer can be called for both conditional and unconditional passes. Increase `--max-records` if you need more layers or more timesteps.

## 6. First experiment table

Run the same prompt and seed across these four settings:

| Setting | Whole ref | HR crop | Sim balance | Actual balance |
|---|---:|---:|---:|---:|
| A | yes | no | no | no |
| B | yes | yes | no | no |
| C | yes | yes | yes | no |
| D | yes | yes | no | yes |

Readouts:

1. Does the HR crop get much higher `mass` than expected from its semantics?
2. Does `sim_group_balance` reduce crop mass roughly by token count?
3. Does actual balancing reduce unwanted reference leakage or hurt preserve quality?
4. Does the output image improve when using `global LR + local HR crop` compared with only a resized global ref?

This is the smallest training-free validation before adding interactive masks, no-reference regions, or LoRA.
