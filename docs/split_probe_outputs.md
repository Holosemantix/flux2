# Split probe outputs into small upload chunks

When `attention_mass_summary.csv` or `attention_mass.jsonl` is too large to upload, split the probe output directory into many small folders:

```sh
python scripts/training_free/split_probe_outputs.py \
  --input-dir results/training_free/probe_auto_text_sim_v2 \
  --clean
```

This writes:

```text
results/training_free/probe_auto_text_sim_v2/upload_chunks/
  manifest.json
  summary_chunks/
    chunk_0000/
      attention_mass_summary.csv
      layout.json
      manifest.json
    chunk_0001/
      attention_mass_summary.csv
      layout.json
      manifest.json
    ...
```

Each `summary_chunks/chunk_xxxx` folder is targeted to stay below `95,000` bytes by default. Upload these chunk folders one by one.

## Include raw JSONL chunks

By default the splitter only chunks `attention_mass_summary.csv`, because this is usually enough for analysis. To also split the raw `attention_mass.jsonl` file:

```sh
python scripts/training_free/split_probe_outputs.py \
  --input-dir results/training_free/probe_auto_text_sim_v2 \
  --include-jsonl \
  --clean
```

This additionally creates:

```text
upload_chunks/jsonl_chunks/chunk_0000/attention_mass.jsonl
upload_chunks/jsonl_chunks/chunk_0001/attention_mass.jsonl
...
```

## Change the size limit

```sh
python scripts/training_free/split_probe_outputs.py \
  --input-dir results/training_free/probe_auto_text_sim_v2 \
  --max-folder-bytes 90000 \
  --clean
```

Use a value below the platform upload limit to leave room for metadata.

## What to upload first

Upload the `summary_chunks` folders first. They contain:

- `attention_mass_summary.csv`: the shard needed for mass analysis
- `layout.json`: copied metadata for the run
- `manifest.json`: shard metadata

Only upload `jsonl_chunks` if the summary CSV is insufficient.
