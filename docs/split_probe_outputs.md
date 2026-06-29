# Split probe outputs into small upload chunks

When `attention_mass_summary.csv` or `attention_mass.jsonl` is too large to upload, split the probe output directory into many small folders:

```sh
python scripts/training_free/split_probe_outputs.py \
  --input-dir results/training_free/probe_auto_text_sim_v2 \
  --clean
```

This writes prefixed folder names and prefixed file names:

```text
results/training_free/probe_auto_text_sim_v2/upload_chunks/
  upload_chunks_manifest.json
  summary_chunks/
    summary_csv_chunk_0000/
      summary_csv_chunk_0000_attention_mass_summary.csv
      summary_csv_chunk_0000_layout.json
      summary_csv_chunk_0000_manifest.json
    summary_csv_chunk_0001/
      summary_csv_chunk_0001_attention_mass_summary.csv
      summary_csv_chunk_0001_layout.json
      summary_csv_chunk_0001_manifest.json
    ...
```

The naming rule is:

```text
folder: <content_prefix>_chunk_<idx4>
file:   <content_prefix>_chunk_<idx4>_<original_file_role>
```

For summary CSV chunks, `content_prefix=summary_csv`. For raw JSONL chunks, `content_prefix=raw_jsonl`.

Each `summary_chunks/summary_csv_chunk_xxxx` folder is targeted to stay below `95,000` bytes by default. Upload these chunk folders one by one.

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
upload_chunks/jsonl_chunks/raw_jsonl_chunk_0000/
  raw_jsonl_chunk_0000_attention_mass.jsonl
  raw_jsonl_chunk_0000_layout.json
  raw_jsonl_chunk_0000_manifest.json
upload_chunks/jsonl_chunks/raw_jsonl_chunk_0001/
  raw_jsonl_chunk_0001_attention_mass.jsonl
  raw_jsonl_chunk_0001_layout.json
  raw_jsonl_chunk_0001_manifest.json
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

Upload the `summary_chunks/summary_csv_chunk_xxxx` folders first. They contain:

- `summary_csv_chunk_xxxx_attention_mass_summary.csv`: the shard needed for mass analysis
- `summary_csv_chunk_xxxx_layout.json`: copied metadata for the run
- `summary_csv_chunk_xxxx_manifest.json`: shard metadata, including content prefix and idx

Only upload `jsonl_chunks` if the summary CSV is insufficient.
