"""Split FLUX.2 training-free probe outputs into small uploadable chunks.

The normal probe output directory can contain large CSV/JSONL files. This helper
creates a separate `upload_chunks/` directory with many small folders. Each folder
contains a shard of `attention_mass_summary.csv` plus small metadata files and is
kept below a configurable byte limit, defaulting to 95,000 bytes.

Chunk folder names and file names include a content prefix and chunk index, e.g.:

    summary_csv_chunk_0000/summary_csv_chunk_0000_attention_mass_summary.csv
    raw_jsonl_chunk_0000/raw_jsonl_chunk_0000_attention_mass.jsonl

By default it chunks only `attention_mass_summary.csv` + `layout.json`, because
that is usually enough for analysis. Use `--include-jsonl` to also create a
separate set of raw JSONL chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_MAX_BYTES = 95_000
SUMMARY_PREFIX = "summary_csv"
JSONL_PREFIX = "raw_jsonl"
ROOT_MANIFEST_NAME = "upload_chunks_manifest.json"


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def chunk_stem(prefix: str, idx: int) -> str:
    return f"{prefix}_chunk_{idx:04d}"


def prefixed_filename(prefix: str, idx: int, suffix: str) -> str:
    return f"{chunk_stem(prefix, idx)}_{suffix}"


def make_chunk_dir(output_root: Path, group_name: str, prefix: str, idx: int) -> Path:
    chunk_dir = output_root / group_name / chunk_stem(prefix, idx)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    return chunk_dir


def rows_to_csv_text(header: list[str], rows: list[list[str]]) -> str:
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def split_csv(
    *,
    input_csv: Path,
    layout_text: str,
    output_root: Path,
    max_bytes: int,
    group_name: str = "summary_chunks",
    prefix: str = SUMMARY_PREFIX,
) -> list[Path]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing CSV file: {input_csv}")

    chunks: list[Path] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: list[list[str]] = []
        chunk_idx = 0

        def flush() -> None:
            nonlocal rows, chunk_idx
            if not rows:
                return
            chunk_dir = make_chunk_dir(output_root, group_name, prefix, chunk_idx)
            csv_text = rows_to_csv_text(header, rows)
            csv_name = prefixed_filename(prefix, chunk_idx, "attention_mass_summary.csv")
            layout_name = prefixed_filename(prefix, chunk_idx, "layout.json")
            manifest_name = prefixed_filename(prefix, chunk_idx, "manifest.json")

            (chunk_dir / csv_name).write_text(csv_text, encoding="utf-8")
            if layout_text:
                (chunk_dir / layout_name).write_text(layout_text, encoding="utf-8")
            write_json(
                chunk_dir / manifest_name,
                {
                    "kind": "attention_mass_summary_csv",
                    "content_prefix": prefix,
                    "chunk_index": chunk_idx,
                    "chunk_folder": chunk_dir.name,
                    "data_file": csv_name,
                    "layout_file": layout_name if layout_text else None,
                    "manifest_file": manifest_name,
                    "source_file": str(input_csv),
                    "num_rows": len(rows),
                    "folder_size_bytes": folder_size(chunk_dir),
                    "max_folder_bytes": max_bytes,
                },
            )
            chunks.append(chunk_dir)
            rows = []
            chunk_idx += 1

        fixed_overhead = byte_len(layout_text) + 2048
        for row in reader:
            candidate_rows = rows + [row]
            candidate_text = rows_to_csv_text(header, candidate_rows)
            if rows and byte_len(candidate_text) + fixed_overhead > max_bytes:
                flush()
                candidate_rows = [row]
                candidate_text = rows_to_csv_text(header, candidate_rows)
            rows = candidate_rows
            if byte_len(candidate_text) + fixed_overhead > max_bytes and len(rows) == 1:
                # A single row is unusually large. Write it anyway so the user
                # can see exactly which record caused the oversize chunk.
                flush()
        flush()

    return chunks


def split_jsonl(
    *,
    input_jsonl: Path,
    layout_text: str,
    output_root: Path,
    max_bytes: int,
    group_name: str = "jsonl_chunks",
    prefix: str = JSONL_PREFIX,
) -> list[Path]:
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Missing JSONL file: {input_jsonl}")

    chunks: list[Path] = []
    lines: list[str] = []
    chunk_idx = 0
    fixed_overhead = byte_len(layout_text) + 2048

    def flush() -> None:
        nonlocal lines, chunk_idx
        if not lines:
            return
        chunk_dir = make_chunk_dir(output_root, group_name, prefix, chunk_idx)
        jsonl_name = prefixed_filename(prefix, chunk_idx, "attention_mass.jsonl")
        layout_name = prefixed_filename(prefix, chunk_idx, "layout.json")
        manifest_name = prefixed_filename(prefix, chunk_idx, "manifest.json")

        (chunk_dir / jsonl_name).write_text("".join(lines), encoding="utf-8")
        if layout_text:
            (chunk_dir / layout_name).write_text(layout_text, encoding="utf-8")
        write_json(
            chunk_dir / manifest_name,
            {
                "kind": "attention_mass_jsonl",
                "content_prefix": prefix,
                "chunk_index": chunk_idx,
                "chunk_folder": chunk_dir.name,
                "data_file": jsonl_name,
                "layout_file": layout_name if layout_text else None,
                "manifest_file": manifest_name,
                "source_file": str(input_jsonl),
                "num_lines": len(lines),
                "folder_size_bytes": folder_size(chunk_dir),
                "max_folder_bytes": max_bytes,
            },
        )
        chunks.append(chunk_dir)
        lines = []
        chunk_idx += 1

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            candidate = lines + [line]
            if lines and byte_len("".join(candidate)) + fixed_overhead > max_bytes:
                flush()
                candidate = [line]
            lines = candidate
            if byte_len("".join(lines)) + fixed_overhead > max_bytes and len(lines) == 1:
                # A single JSON line is too large. Write it as a standalone
                # oversize chunk rather than dropping data.
                flush()
        flush()

    return chunks


def chunk_manifest_files(chunks: Iterable[Path], prefix: str) -> list[str]:
    manifests: list[str] = []
    for chunk_dir in chunks:
        candidates = sorted(chunk_dir.glob(f"{prefix}_chunk_*_manifest.json"))
        if candidates:
            manifests.append(str(candidates[0]))
        else:
            manifests.append(str(chunk_dir))
    return manifests


def write_root_manifest(
    *,
    input_dir: Path,
    output_root: Path,
    max_bytes: int,
    summary_chunks: Iterable[Path],
    jsonl_chunks: Iterable[Path],
) -> None:
    summary_chunks = list(summary_chunks)
    jsonl_chunks = list(jsonl_chunks)
    payload = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "max_folder_bytes": max_bytes,
        "summary_content_prefix": SUMMARY_PREFIX,
        "jsonl_content_prefix": JSONL_PREFIX,
        "summary_chunks": [str(p) for p in summary_chunks],
        "summary_chunk_manifests": chunk_manifest_files(summary_chunks, SUMMARY_PREFIX),
        "jsonl_chunks": [str(p) for p in jsonl_chunks],
        "jsonl_chunk_manifests": chunk_manifest_files(jsonl_chunks, JSONL_PREFIX),
        "num_summary_chunks": len(summary_chunks),
        "num_jsonl_chunks": len(jsonl_chunks),
        "naming_rule": {
            "folder": "<content_prefix>_chunk_<idx4>",
            "file": "<content_prefix>_chunk_<idx4>_<original_file_role>",
            "examples": [
                "summary_csv_chunk_0000/summary_csv_chunk_0000_attention_mass_summary.csv",
                "summary_csv_chunk_0000/summary_csv_chunk_0000_layout.json",
                "raw_jsonl_chunk_0000/raw_jsonl_chunk_0000_attention_mass.jsonl",
            ],
        },
        "notes": [
            "Upload folders under summary_chunks first; they contain prefixed summary CSV plus prefixed layout JSON.",
            "JSONL chunks are optional and can be large if a single record is large.",
        ],
    }
    write_json(output_root / ROOT_MANIFEST_NAME, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split probe output files into small uploadable chunk folders.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Probe output directory containing summary/jsonl/layout.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root for chunks. Defaults to <input-dir>/upload_chunks.",
    )
    parser.add_argument(
        "--max-folder-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Target maximum total bytes per chunk folder. Default: 95000.",
    )
    parser.add_argument("--include-jsonl", action="store_true", help="Also split attention_mass.jsonl into jsonl_chunks.")
    parser.add_argument("--clean", action="store_true", help="Delete the output chunk directory before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_root = args.output_dir or (input_dir / "upload_chunks")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    layout_text = read_text_if_exists(input_dir / "layout.json")
    summary_chunks = split_csv(
        input_csv=input_dir / "attention_mass_summary.csv",
        layout_text=layout_text,
        output_root=output_root,
        max_bytes=args.max_folder_bytes,
    )

    jsonl_chunks = []
    if args.include_jsonl:
        jsonl_chunks = split_jsonl(
            input_jsonl=input_dir / "attention_mass.jsonl",
            layout_text=layout_text,
            output_root=output_root,
            max_bytes=args.max_folder_bytes,
        )

    write_root_manifest(
        input_dir=input_dir,
        output_root=output_root,
        max_bytes=args.max_folder_bytes,
        summary_chunks=summary_chunks,
        jsonl_chunks=jsonl_chunks,
    )

    print(f"Wrote chunks under: {output_root}")
    print(f"Root manifest: {output_root / ROOT_MANIFEST_NAME}")
    print(f"Summary chunks: {len(summary_chunks)}")
    print(f"JSONL chunks: {len(jsonl_chunks)}")
    for chunk_dir in summary_chunks[:5]:
        print(f"  {chunk_dir} ({folder_size(chunk_dir)} bytes)")
    if len(summary_chunks) > 5:
        print("  ...")


if __name__ == "__main__":
    main()
