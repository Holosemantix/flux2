"""Training-free attention-mass probe for FLUX.2 [klein] in Diffusers.

This script is intentionally written as a runtime hook instead of a permanent Diffusers
patch. It lets you run the first diagnostic experiments with an editable local Diffusers
checkout, then move the same logic into Diffusers internals once the probe is useful.

The probe measures how much attention mass output/noise tokens allocate to text tokens,
output tokens, and each reference-image group. It also supports a lightweight simulated
or actual group-size compensation test for reference token-count bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


DEFAULT_DISTILLED_MODEL = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"


@dataclass
class TokenGroup:
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


class AttentionMassRecorder:
    """Records grouped attention mass from output/noise query tokens to reference keys."""

    def __init__(
        self,
        *,
        query_sample_size: int,
        record_layers: set[str],
        record_all_layers: bool,
        max_records: int,
        simulate_group_balance: bool,
        apply_group_balance: bool,
        group_balance_strength: float,
    ) -> None:
        self.query_sample_size = query_sample_size
        self.record_layers = record_layers
        self.record_all_layers = record_all_layers
        self.max_records = max_records
        self.simulate_group_balance = simulate_group_balance
        self.apply_group_balance = apply_group_balance
        self.group_balance_strength = group_balance_strength

        self.pipe = None
        self.text_len: int | None = None
        self.output_len: int | None = None
        self.ref_lengths: list[int] = []
        self.ref_labels: list[str] = []
        self.records: list[dict[str, Any]] = []
        self.installed_processors: list[str] = []

    def attach_pipe(self, pipe) -> None:
        self.pipe = pipe

    def set_text_len(self, text_len: int) -> None:
        self.text_len = int(text_len)

    def set_output_len(self, output_len: int) -> None:
        self.output_len = int(output_len)

    def set_refs(self, ref_lengths: list[int], ref_labels: list[str]) -> None:
        self.ref_lengths = [int(x) for x in ref_lengths]
        labels = list(ref_labels)
        while len(labels) < len(ref_lengths):
            labels.append(f"ref{len(labels)}")
        self.ref_labels = labels[: len(ref_lengths)]

    def _groups(self, seq_len: int) -> list[TokenGroup]:
        if self.text_len is None or self.output_len is None:
            return []
        groups = [
            TokenGroup("text", 0, self.text_len),
            TokenGroup("output", self.text_len, self.text_len + self.output_len),
        ]
        cursor = self.text_len + self.output_len
        for label, length in zip(self.ref_labels, self.ref_lengths):
            groups.append(TokenGroup(f"ref:{label}", cursor, cursor + length))
            cursor += length
        return [g for g in groups if 0 <= g.start < g.end <= seq_len]

    def should_record(self, layer_name: str) -> bool:
        if len(self.records) >= self.max_records:
            return False
        if self.record_all_layers:
            return True
        return layer_name in self.record_layers or any(layer_name.endswith(x) for x in self.record_layers)

    def key_bias_vector(self, seq_len: int, *, device, dtype) -> torch.Tensor | None:
        """Return per-key logit bias for group-size compensation.

        The default compensation subtracts log(n_group) from every key in a ref group.
        If all logits are equal, a 4-token crop then receives the same total mass as a
        1-token crop instead of 4x more mass. This is a diagnostic approximation of
        density compensation, not a semantic importance prior.
        """

        if not (self.simulate_group_balance or self.apply_group_balance):
            return None
        groups = self._groups(seq_len)
        if not groups:
            return None
        bias = torch.zeros(seq_len, device=device, dtype=torch.float32)
        for group in groups:
            if not group.name.startswith("ref:") or group.length <= 1:
                continue
            bias[group.start : group.end] -= self.group_balance_strength * math.log(group.length)
        return bias.to(dtype=dtype)

    def attn_mask(self, seq_len: int, *, device, dtype) -> torch.Tensor | None:
        if not self.apply_group_balance:
            return None
        bias = self.key_bias_vector(seq_len, device=device, dtype=dtype)
        if bias is None:
            return None
        return bias.view(1, 1, 1, seq_len)

    @torch.no_grad()
    def record(self, layer_name: str, kind: str, query: torch.Tensor, key: torch.Tensor) -> None:
        if not self.should_record(layer_name):
            return
        if self.text_len is None or self.output_len is None:
            return

        seq_len = key.shape[1]
        groups = self._groups(seq_len)
        if not groups:
            return

        q_start = self.text_len
        q_end = min(self.text_len + self.output_len, query.shape[1])
        if q_start >= q_end:
            return

        sample_count = min(self.query_sample_size, q_end - q_start)
        if sample_count <= 0:
            return
        rel = torch.linspace(0, q_end - q_start - 1, sample_count, device=query.device).round().long().unique()
        q_idx = rel + q_start

        q = query.index_select(1, q_idx).float()
        k = key.float()
        logits = torch.einsum("bqhd,bkhd->bhqk", q, k) * (q.shape[-1] ** -0.5)

        def summarize(prefix: str, attn: torch.Tensor) -> dict[str, Any]:
            group_stats = {}
            for group in groups:
                mass = attn[..., group.start : group.end].sum(dim=-1).mean().item()
                group_stats[group.name] = {
                    "mass": mass,
                    "tokens": group.length,
                    "mass_per_token": mass / max(group.length, 1),
                    "start": group.start,
                    "end": group.end,
                }
            entropy = (-(attn.clamp_min(1e-20) * attn.clamp_min(1e-20).log()).sum(dim=-1)).mean().item()
            return {"prefix": prefix, "entropy": entropy, "groups": group_stats}

        raw_attn = torch.softmax(logits, dim=-1)
        summaries = [summarize("raw", raw_attn)]

        if self.simulate_group_balance:
            bias = self.key_bias_vector(seq_len, device=query.device, dtype=torch.float32)
            if bias is not None:
                sim_attn = torch.softmax(logits + bias.view(1, 1, 1, seq_len), dim=-1)
                summaries.append(summarize("sim_group_balance", sim_attn))

        timestep = None
        if self.pipe is not None and getattr(self.pipe, "current_timestep", None) is not None:
            t = self.pipe.current_timestep
            timestep = float(t.detach().float().cpu().item()) if torch.is_tensor(t) else float(t)

        self.records.append(
            {
                "layer": layer_name,
                "kind": kind,
                "timestep": timestep,
                "seq_len": seq_len,
                "query_sample_count": int(q_idx.numel()),
                "text_len": self.text_len,
                "output_len": self.output_len,
                "ref_lengths": self.ref_lengths,
                "ref_labels": self.ref_labels,
                "summaries": summaries,
            }
        )

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "attention_mass.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        csv_path = out_dir / "attention_mass_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "record_id",
                    "layer",
                    "kind",
                    "timestep",
                    "prefix",
                    "group",
                    "mass",
                    "tokens",
                    "mass_per_token",
                    "entropy",
                    "start",
                    "end",
                ],
            )
            writer.writeheader()
            for record_id, record in enumerate(self.records):
                for summary in record["summaries"]:
                    for group_name, stats in summary["groups"].items():
                        writer.writerow(
                            {
                                "record_id": record_id,
                                "layer": record["layer"],
                                "kind": record["kind"],
                                "timestep": record["timestep"],
                                "prefix": summary["prefix"],
                                "group": group_name,
                                "mass": stats["mass"],
                                "tokens": stats["tokens"],
                                "mass_per_token": stats["mass_per_token"],
                                "entropy": summary["entropy"],
                                "start": stats["start"],
                                "end": stats["end"],
                            }
                        )
        print(f"Wrote {jsonl_path}")
        print(f"Wrote {csv_path}")


def _combine_attention_mask(attention_mask, extra_mask):
    if extra_mask is None:
        return attention_mask
    if attention_mask is None:
        return extra_mask
    return attention_mask + extra_mask


class MassFlux2AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, recorder: AttentionMassRecorder, layer_name: str, flux2_mod) -> None:
        self.recorder = recorder
        self.layer_name = layer_name
        self.flux2_mod = flux2_mod

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        kv_cache=None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = self.flux2_mod._get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = self.flux2_mod.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = self.flux2_mod.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if kv_cache_mode is None:
            self.recorder.record(self.layer_name, "double", query, key)
            attention_mask = _combine_attention_mask(
                attention_mask,
                self.recorder.attn_mask(key.shape[1], device=key.device, dtype=query.dtype),
            )
            hidden_states = self.flux2_mod.dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            num_txt_tokens = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
                ref_start = num_txt_tokens
                ref_end = num_txt_tokens + num_ref_tokens
                kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())
            if kv_cache_mode == "extract" and num_ref_tokens > 0:
                hidden_states = self.flux2_mod._flux2_kv_causal_attention(
                    query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
                )
            elif kv_cache_mode == "cached" and kv_cache is not None:
                hidden_states = self.flux2_mod._flux2_kv_causal_attention(
                    query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
                )
            else:
                hidden_states = self.flux2_mod.dispatch_attention_fn(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return (hidden_states, encoder_hidden_states) if encoder_hidden_states is not None else hidden_states


class MassFlux2ParallelSelfAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, recorder: AttentionMassRecorder, layer_name: str, flux2_mod) -> None:
        self.recorder = recorder
        self.layer_name = layer_name
        self.flux2_mod = flux2_mod

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        kv_cache=None,
        kv_cache_mode: str | None = None,
        num_txt_tokens: int = 0,
        num_ref_tokens: int = 0,
    ) -> torch.Tensor:
        hidden_states_proj = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states_proj, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = self.flux2_mod.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = self.flux2_mod.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if kv_cache_mode is None:
            self.recorder.record(self.layer_name, "single", query, key)
            attention_mask = _combine_attention_mask(
                attention_mask,
                self.recorder.attn_mask(key.shape[1], device=key.device, dtype=query.dtype),
            )
            attn_output = self.flux2_mod.dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        else:
            if kv_cache_mode == "extract" and kv_cache is not None and num_ref_tokens > 0:
                ref_start = num_txt_tokens
                ref_end = num_txt_tokens + num_ref_tokens
                kv_cache.store(key[:, ref_start:ref_end].clone(), value[:, ref_start:ref_end].clone())
            if kv_cache_mode == "extract" and num_ref_tokens > 0:
                attn_output = self.flux2_mod._flux2_kv_causal_attention(
                    query, key, value, num_txt_tokens, num_ref_tokens, backend=self._attention_backend
                )
            elif kv_cache_mode == "cached" and kv_cache is not None:
                attn_output = self.flux2_mod._flux2_kv_causal_attention(
                    query, key, value, num_txt_tokens, 0, kv_cache=kv_cache, backend=self._attention_backend
                )
            else:
                attn_output = self.flux2_mod.dispatch_attention_fn(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )

        attn_output = attn_output.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        return attn.to_out(hidden_states)


def install_layout_capture(pipe, recorder: AttentionMassRecorder, ref_labels: list[str]) -> None:
    recorder.attach_pipe(pipe)
    orig_encode_prompt = pipe.encode_prompt
    orig_prepare_latents = pipe.prepare_latents
    orig_prepare_image_latents = pipe.prepare_image_latents

    def encode_prompt_wrapper(*args, **kwargs):
        prompt_embeds, text_ids = orig_encode_prompt(*args, **kwargs)
        recorder.set_text_len(prompt_embeds.shape[1])
        return prompt_embeds, text_ids

    def prepare_latents_wrapper(*args, **kwargs):
        latents, latent_ids = orig_prepare_latents(*args, **kwargs)
        recorder.set_output_len(latents.shape[1])
        return latents, latent_ids

    def prepare_image_latents_wrapper(*args, **kwargs):
        images = kwargs.get("images", args[0] if args else None)
        lengths = []
        if images is not None:
            token_scale = pipe.vae_scale_factor * 2
            for image in images:
                height, width = image.shape[-2:]
                lengths.append((height // token_scale) * (width // token_scale))
        recorder.set_refs(lengths, ref_labels)
        return orig_prepare_image_latents(*args, **kwargs)

    pipe.encode_prompt = encode_prompt_wrapper
    pipe.prepare_latents = prepare_latents_wrapper
    pipe.prepare_image_latents = prepare_image_latents_wrapper


def install_attention_processors(pipe, recorder: AttentionMassRecorder) -> None:
    from diffusers.models.transformers import transformer_flux2 as flux2_mod

    for name, module in pipe.transformer.named_modules():
        if not hasattr(module, "processor") or not hasattr(module, "set_processor"):
            continue
        proc_name = module.processor.__class__.__name__
        if proc_name in {"Flux2AttnProcessor", "Flux2KVAttnProcessor"}:
            module.set_processor(MassFlux2AttnProcessor(recorder, name, flux2_mod))
            recorder.installed_processors.append(name)
        elif proc_name in {"Flux2ParallelSelfAttnProcessor", "Flux2KVParallelSelfAttnProcessor"}:
            module.set_processor(MassFlux2ParallelSelfAttnProcessor(recorder, name, flux2_mod))
            recorder.installed_processors.append(name)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_from_arg(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if device in {"cuda", "mps"} else torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]


def _load_diffusers():
    try:
        from diffusers import Flux2KleinPipeline
        from diffusers.utils import load_image
    except ImportError as exc:
        raise SystemExit(
            "Diffusers is not importable. For editable Diffusers use, run something like:\n"
            "  git clone https://github.com/huggingface/diffusers\n"
            "  cd diffusers && pip install -e .\n"
            "Then rerun this script with that environment active."
        ) from exc
    return Flux2KleinPipeline, load_image


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


def _parse_crop_spec(spec: str) -> tuple[int, str, tuple[int, int, int, int]]:
    try:
        image_idx, label, box = spec.split(":", 2)
        coords = tuple(int(x) for x in box.split(","))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Crop format must be INDEX:LABEL:X0,Y0,X1,Y1, e.g. 0:face:100,120,420,520") from exc
    if len(coords) != 4:
        raise ValueError("Crop box must contain exactly four comma-separated integers")
    return int(image_idx), label, coords


def load_reference_images(load_image, image_paths: Iterable[str], image_labels: list[str], crop_specs: list[str]):
    images = [load_image(path).convert("RGB") for path in image_paths]
    labels = list(image_labels)
    while len(labels) < len(images):
        labels.append(f"ref{len(labels)}")

    for spec in crop_specs:
        image_idx, label, box = _parse_crop_spec(spec)
        if image_idx < 0 or image_idx >= len(images):
            raise ValueError(f"Crop spec {spec!r} uses image index {image_idx}, but only {len(images)} images exist")
        images.append(images[image_idx].crop(box))
        labels.append(label)
    return images, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-free attention mass probe for Diffusers FLUX.2 [klein].")
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
    parser.add_argument("--prompt", default="Edit the input while preserving the important reference details.")
    parser.add_argument("--image", action="append", default=[], help="Reference image path or URL. Repeat for multiref.")
    parser.add_argument("--image-label", action="append", default=[], help="Optional label for each --image.")
    parser.add_argument(
        "--crop",
        action="append",
        default=[],
        help="Append an HR crop ref: INDEX:LABEL:X0,Y0,X1,Y1. Example: --crop 0:text:100,120,420,220",
    )
    parser.add_argument("--output-dir", default="outputs/attention_mass_probe")
    parser.add_argument("--output-name", default="sample.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--force-size-for-edit", action="store_true")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--query-sample-size", type=int, default=256)
    parser.add_argument("--record-layer", action="append", default=[])
    parser.add_argument("--record-all-layers", action="store_true")
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--simulate-group-balance", action="store_true")
    parser.add_argument("--apply-group-balance", action="store_true")
    parser.add_argument("--group-balance-strength", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Flux2KleinPipeline, load_image = _load_diffusers()

    model_id = _resolve_model_arg(args)
    model_type = _infer_model_type(model_id, args.model_type)
    num_inference_steps, guidance_scale = _resolve_sampling_defaults(args, model_type)

    device = args.device
    dtype = _dtype_from_arg(args.dtype, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record_layers = set(args.record_layer) or {"transformer_blocks.0.attn", "single_transformer_blocks.0.attn"}
    recorder = AttentionMassRecorder(
        query_sample_size=args.query_sample_size,
        record_layers=record_layers,
        record_all_layers=args.record_all_layers,
        max_records=args.max_records,
        simulate_group_balance=args.simulate_group_balance,
        apply_group_balance=args.apply_group_balance,
        group_balance_strength=args.group_balance_strength,
    )

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

    images, labels = load_reference_images(load_image, args.image, args.image_label, args.crop)
    install_layout_capture(pipe, recorder, labels)
    install_attention_processors(pipe, recorder)
    print(f"Installed probe processors on {len(recorder.installed_processors)} attention modules")

    generator_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    call_kwargs = {
        "prompt": args.prompt,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
    }
    if images:
        call_kwargs["image"] = images[0] if len(images) == 1 else images
        if args.force_size_for_edit:
            call_kwargs["height"] = args.height
            call_kwargs["width"] = args.width
    else:
        call_kwargs["height"] = args.height
        call_kwargs["width"] = args.width

    image = pipe(**call_kwargs).images[0]
    output_path = out_dir / args.output_name
    image.save(output_path)
    print(f"Saved {output_path}")

    recorder.write(out_dir)
    layout = {
        "model": model_id,
        "model_type": model_type,
        "local_files_only": args.local_files_only,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "text_len": recorder.text_len,
        "output_len": recorder.output_len,
        "ref_labels": recorder.ref_labels,
        "ref_lengths": recorder.ref_lengths,
        "installed_processors": recorder.installed_processors,
        "record_layers": sorted(record_layers),
        "record_all_layers": args.record_all_layers,
        "simulate_group_balance": args.simulate_group_balance,
        "apply_group_balance": args.apply_group_balance,
        "group_balance_strength": args.group_balance_strength,
    }
    (out_dir / "layout.json").write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_dir / 'layout.json'}")


if __name__ == "__main__":
    main()
