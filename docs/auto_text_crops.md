# Automatic text crops for reference-pyramid tests

`--crop` in `flux2_attention_mass_probe.py` is a high-resolution reference-crop spec. It does **not** mark an output/editing mask. It appends a local crop from the whole reference image as an extra reference group.

This helper detects text boxes and turns them into crop specs automatically:

```sh
scripts/training_free/auto_text_crops.py
```

## Install an OCR backend

Use either EasyOCR:

```sh
pip install easyocr
```

or PaddleOCR:

```sh
pip install paddleocr
```

EasyOCR defaults to `en,ch_sim` in the helper. PaddleOCR defaults to `ch`.

## Detect text boxes only

```sh
python scripts/training_free/auto_text_crops.py \
  --image ref.png \
  --engine easyocr \
  --dry-run
```

Outputs:

- `outputs/auto_text_crops/text_boxes_preview.png`: preview with boxes and labels
- `outputs/auto_text_crops/text_boxes.json`: detected boxes and generated crop specs
- `outputs/auto_text_crops/text_crops/*.png`: saved text crops

The printed crop specs look like:

```sh
--crop 0:text_0:120,80,520,220
--crop 0:text_1:600,410,880,470
```

The `0` means the crop comes from the first `--image` passed to the probe.

## Run the attention mass probe directly

```sh
python scripts/training_free/auto_text_crops.py \
  --image ref.png \
  --engine easyocr \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Change the background but preserve the small text from the reference." \
  --simulate-group-balance \
  --run-probe \
  --output-dir outputs/auto_text_crops \
  --probe-output-dir outputs/probe_auto_text
```

This runs a command equivalent to:

```sh
python scripts/training_free/flux2_attention_mass_probe.py \
  --model-path /path/to/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --image ref.png \
  --image-label global \
  --crop 0:text_0:... \
  --crop 0:text_1:... \
  --simulate-group-balance \
  --output-dir outputs/probe_auto_text
```

## What `--image-label` means

`--image-label` is only the name of the whole reference image group in the attention summary. For example:

```sh
--image ref.png --image-label global
```

will produce a group named:

```text
ref:global
```

If OCR creates two text crops, the summary will also contain:

```text
ref:text_0
ref:text_1
```

The label does not tell the model what to do. It is used for bookkeeping in `attention_mass_summary.csv` and `layout.json`.

## Multiple text regions

By default, each detected text box becomes a separate crop:

```text
text_0, text_1, text_2, ...
```

This is best when you want to measure which text region receives attention mass.

If there are too many small OCR boxes, use one of these controls:

### Keep only the most important boxes

```sh
--max-crops 6
```

The helper keeps the largest/highest-confidence boxes, then restores reading order.

### Merge nearby OCR boxes

```sh
--merge --merge-gap 32
```

This groups nearby words/characters into larger text blocks. It is useful for signs, logos, posters, and UI text where OCR returns many tiny boxes.

### Add more context around each text crop

```sh
--padding 40
```

For image editing, avoid cropping too tightly around the glyphs. A little context usually helps the reference image tokens remain useful.

## Recommended first test

Use one text-heavy reference image and run these four settings with the same prompt/seed:

1. Global reference only: no OCR crops.
2. Global reference + auto text crops.
3. Global reference + auto text crops + `--simulate-group-balance`.
4. Global reference + auto text crops + `--apply-group-balance`.

Compare `attention_mass_summary.csv` for `ref:global` vs `ref:text_i`, and compare the generated images for small-text preservation and unwanted reference leakage.
