# Local OCR models for automatic text crop detection

Use this helper when OCR weights are already downloaded locally and the machine should not auto-download model files:

```sh
scripts/training_free/auto_text_crops_local.py
```

## EasyOCR

EasyOCR's `Reader` accepts `model_storage_directory`, `user_network_directory`, and `download_enabled`. Put the downloaded `.pth` files under the model directory and run with `--ocr-local-files-only`.

Common EasyOCR detector files:

- `craft_mlt_25k.pth` for `--easyocr-detect-network craft`
- `pretrained_ic15_res18.pt` for `--easyocr-detect-network dbnet18`

Example:

```sh
python scripts/training_free/auto_text_crops_local.py \
  --image test_imgs/costume.png \
  --engine easyocr \
  --langs en,ch_sim \
  --easyocr-model-dir models/pretrained/easyocr/model \
  --easyocr-user-network-dir models/pretrained/easyocr/user_network \
  --easyocr-detect-network craft \
  --ocr-local-files-only \
  --dry-run
```

If you do not already have the EasyOCR files, run once without `--ocr-local-files-only` on a machine with internet, then copy the cache directory. EasyOCR defaults to `~/.EasyOCR/model` when no model directory is provided.

## PaddleOCR

PaddleOCR accepts separate local inference model directories:

- `--paddle-det-model-dir`: text detection model
- `--paddle-rec-model-dir`: text recognition model
- `--paddle-cls-model-dir`: angle classifier model

Example:

```sh
python scripts/training_free/auto_text_crops_local.py \
  --image test_imgs/costume.png \
  --engine paddleocr \
  --paddle-lang ch \
  --paddle-det-model-dir models/pretrained/paddleocr/ch_PP-OCRv4_det_infer \
  --paddle-rec-model-dir models/pretrained/paddleocr/ch_PP-OCRv4_rec_infer \
  --paddle-cls-model-dir models/pretrained/paddleocr/ch_ppocr_mobile_v2.0_cls_infer \
  --dry-run
```

## Directly run the attention mass probe

```sh
python scripts/training_free/auto_text_crops_local.py \
  --image test_imgs/costume.png \
  --engine easyocr \
  --easyocr-model-dir models/pretrained/easyocr/model \
  --ocr-local-files-only \
  --model-path models/pretrained/black-forest-labs/FLUX.2-klein-base-4B \
  --model-type base \
  --local-files-only \
  --prompt "Change the background to a London street but preserve the small text from the reference." \
  --num-inference-steps 10 \
  --guidance-scale 4.0 \
  --simulate-group-balance \
  --run-probe \
  --probe-output-dir results/training_free/probe_auto_text_sim
```

## Multiple text regions

By default, each detected text region becomes a separate crop group named `text_0`, `text_1`, etc. Use:

```sh
--merge --merge-gap 32
```

to merge nearby word/character boxes into larger text blocks. Use:

```sh
--max-crops 6
```

to limit how many text regions are passed as HR reference crops.
