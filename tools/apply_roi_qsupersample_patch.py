#!/usr/bin/env python3
"""Apply the next ROI attention experiment patch.

This helper edits the reconstructed files under code/ in-place:
  - code/transformer_flux2.py
  - code/refine_model.py
  - code/Dit_pipeline.py

Why a helper instead of hand-replacing the whole transformer file?
The transformer file is large and this branch is meant to be copied/diffed into the
real refiner tree. A deterministic patch script keeps the change small, inspectable,
and repeatable. Run from repository root:

    python tools/apply_roi_qsupersample_patch.py

The patch is idempotent: if it sees that roi_variant already exists, it leaves the
file unchanged for that section.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER = ROOT / "code" / "transformer_flux2.py"
REFINE = ROOT / "code" / "refine_model.py"
DIT = ROOT / "code" / "Dit_pipeline.py"


class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, tag: str) -> str:
    if old not in text:
        raise PatchError(f"[{tag}] pattern not found")
    return text.replace(old, new, 1)


def insert_before(text: str, marker: str, insertion: str, tag: str) -> str:
    if insertion.strip() in text:
        return text
    if marker not in text:
        raise PatchError(f"[{tag}] marker not found")
    return text.replace(marker, insertion + marker, 1)


def patch_transformer(text: str) -> str:
    # ------------------------------------------------------------------
    # 1) Config fields for the next experiment.
    # ------------------------------------------------------------------
    if "roi_variant: str = \"interpolate\"" not in text:
        text = replace_once(
            text,
            "    roi_down_layer: int = -1\n    # ===== Version A（真·高清 ref 重编码）=====\n",
            "    roi_down_layer: int = -1\n"
            "    # ===== Version B-2（Q-only supersampling，不插值 V、不下采样虚拟 latent）=====\n"
            "    roi_variant: str = \"interpolate\"  # 'interpolate'=旧 B; 'q_supersample'=下一步实验\n"
            "    roi_subsample: int = 2              # 每个 native noise face token 生成 m×m 个子查询\n"
            "    roi_agg_mode: str = \"mean\"        # 子查询输出聚合：mean / center\n"
            "    roi_split_branches: bool = True     # lq/ref 分支分开 softmax，避免互相稀释\n"
            "    roi_detail_beta: float = 0.5        # split 分支下 ref detail 分支权重，0=纯 lq, 1=纯 ref\n"
            "    # ===== Version A（真·高清 ref 重编码）=====\n",
            "transformer-config-fields",
        )

    # ------------------------------------------------------------------
    # 2) Small PE bug/risk fix: use token-center coordinates for virtual ids.
    #    Bbox is half-open [y1,y2), so y2 itself is not a token center.
    # ------------------------------------------------------------------
    text = text.replace(
        "    ys = torch.linspace(float(y1), float(y2), P, device=device)\n"
        "    xs = torch.linspace(float(x1), float(x2), P, device=device)\n",
        "    ys = torch.linspace(float(y1) + 0.5, float(y2) - 0.5, P, device=device)\n"
        "    xs = torch.linspace(float(x1) + 0.5, float(x2) - 0.5, P, device=device)\n",
    )

    # ------------------------------------------------------------------
    # 3) Add Q-only supersampling implementation.
    # ------------------------------------------------------------------
    qsup_code = r'''

def _make_rect_pos_ids(
    sample_box: Tuple[int, int, int, int],
    orig_box: Tuple[int, int, int, int],
    t_val: int,
    mode: str,
    target_box: Optional[Tuple[int, int, int, int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Native rectangular token ids using token-center coordinates.

    This is the native-token counterpart of _make_roi_pos_ids. It does not
    interpolate token contents; it only assigns RoPE ids to already existing
    K tokens. For ref with mode='pe3', the ref rectangle is mapped into the
    target lq face coordinate frame.
    """
    y1, x1, y2, x2 = sample_box
    h = max(int(y2 - y1), 1)
    w = max(int(x2 - x1), 1)
    ys = torch.linspace(float(y1) + 0.5, float(y2) - 0.5, h, device=device)
    xs = torch.linspace(float(x1) + 0.5, float(x2) - 0.5, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy = gy.reshape(-1)
    gx = gx.reshape(-1)

    if mode == "pe2":
        oy1, ox1, oy2, ox2 = orig_box
        cy = (oy1 + oy2) / 2.0
        cx = (ox1 + ox2) / 2.0
        ry = max(y2 - y1, 1e-6) / max(oy2 - oy1, 1e-6)
        rx = max(x2 - x1, 1e-6) / max(ox2 - ox1, 1e-6)
        gy = cy + (gy - cy) / ry
        gx = cx + (gx - cx) / rx
    elif mode == "pe3" and target_box is not None:
        ty1, tx1, ty2, tx2 = target_box
        ny = (gy - (float(y1) + 0.5)) / max(y2 - y1, 1e-6)
        nx = (gx - (float(x1) + 0.5)) / max(x2 - x1, 1e-6)
        gy = ty1 + 0.5 + ny * max(ty2 - ty1, 1e-6)
        gx = tx1 + 0.5 + nx * max(tx2 - tx1, 1e-6)

    t = torch.full_like(gy, float(t_val))
    l = torch.zeros_like(gy)
    return torch.stack([t, gy, gx, l], dim=-1)


def _make_subquery_pos_ids(
    box: Tuple[int, int, int, int],
    m: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate m×m sub-query positions around each native token center.

    The content vector is not interpolated; each native q is repeated m² times.
    Only the RoPE position is offset inside the native token cell. This avoids
    the low-pass failure of interpolating latent V and downsampling it back.
    """
    y1, x1, y2, x2 = box
    ys = torch.arange(y1, y2, device=device, dtype=torch.float32)
    xs = torch.arange(x1, x2, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base_y = gy.reshape(-1)
    base_x = gx.reshape(-1)

    offsets = (torch.arange(m, device=device, dtype=torch.float32) + 0.5) / float(m) - 0.5
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    oy = oy.reshape(-1)
    ox = ox.reshape(-1)

    sub_y = base_y[:, None] + oy[None, :]
    sub_x = base_x[:, None] + ox[None, :]
    sub_y = sub_y.reshape(-1)
    sub_x = sub_x.reshape(-1)
    t = torch.full_like(sub_y, float(_ROI_T_NOISE))
    l = torch.zeros_like(sub_y)
    return torch.stack([t, sub_y, sub_x, l], dim=-1)


def _aggregate_subquery_outputs(x: torch.Tensor, n_native: int, m: int, mode: str) -> torch.Tensor:
    """[B, n_native*m*m, H, D] -> [B, n_native, H, D]."""
    B, _, Hh, D = x.shape
    x = x.reshape(B, n_native, m * m, Hh, D)
    if mode == "center":
        return x[:, :, (m * m) // 2]
    return x.mean(dim=2)


def _q_supersample_roi_attention(
    output: torch.Tensor,
    q_pre: torch.Tensor, k_pre: torch.Tensor, v_pre: torch.Tensor,
    ranges: Dict[str, Tuple[int, int]],
    id_patch_pairs: List[Dict[str, Tuple[int, int, int, int]]],
    latent_h: int, latent_w: int,
    m: int, pe_mode: str, include_lq: bool, agg_mode: str,
    split_branches: bool, detail_beta: float, noise_alpha: float,
    exp_lq: float, exp_ref: float, exp_min: int,
    rope_theta: int, axes_dim,
    backend=None, parallel_config=None,
) -> torch.Tensor:
    """Version B-2: Q-only supersampling.

    Difference from previous Virtual ROI-QKV:
      * does NOT bilinear-upsample latent V;
      * does NOT create P×P virtual latent outputs and downsample them;
      * repeats each native noise-face query m² times with sub-token RoPE ids;
      * lets these sub-queries attend to native lq/ref K/V;
      * aggregates only attention outputs back to the original native token.

    This tests whether attention-logit/query resolution alone can improve ID
    detail transfer without the low-pass artifact of interpolation-based ROI.
    """
    m = max(1, _as_int(m))
    B, S, Hh, D = q_pre.shape
    noise_start, noise_end = ranges["noise"]
    lq_start, lq_end = ranges["lq"]
    ref_start, ref_end = ranges["ref"]
    if not id_patch_pairs or noise_end <= noise_start or ref_end <= ref_start:
        return output
    dev = q_pre.device
    detail_beta = float(detail_beta)

    for pair in id_patch_pairs:
        lq_bbox = pair["lq"]
        ref_bbox = pair["ref"]
        ly1, lx1, ly2, lx2 = lq_bbox
        h_f, w_f = ly2 - ly1, lx2 - lx1
        if h_f <= 0 or w_f <= 0:
            continue

        nq_idx = _build_2d_rect_indices(ly1, lx1, ly2, lx2, latent_w, dev) + noise_start
        n_native = int(nq_idx.numel())
        q_native = q_pre[:, nq_idx]
        q_sub = q_native[:, :, None].expand(B, n_native, m * m, Hh, D).reshape(B, n_native * m * m, Hh, D)
        q_ids = _make_subquery_pos_ids(lq_bbox, m, device=dev)
        qc, qs = _virtual_rope_freqs(q_ids, rope_theta, axes_dim)
        q_sub = apply_rotary_emb(q_sub, (qc, qs), sequence_dim=1)

        ref_exp = _expand_bbox(ref_bbox, exp_ref, exp_min, latent_h, latent_w)
        ry1, rx1, ry2, rx2 = ref_exp
        rk_idx = _build_2d_rect_indices(ry1, rx1, ry2, rx2, latent_w, dev) + ref_start
        k_ref = k_pre[:, rk_idx]
        v_ref = v_pre[:, rk_idx]
        ref_ids = _make_rect_pos_ids(ref_exp, ref_bbox, _ROI_T_REF, pe_mode, target_box=lq_bbox, device=dev)
        rc, rs = _virtual_rope_freqs(ref_ids, rope_theta, axes_dim)
        k_ref = apply_rotary_emb(k_ref, (rc, rs), sequence_dim=1)

        if include_lq and lq_end > lq_start:
            lq_exp = _expand_bbox(lq_bbox, exp_lq, exp_min, latent_h, latent_w)
            ey1, ex1, ey2, ex2 = lq_exp
            lk_idx = _build_2d_rect_indices(ey1, ex1, ey2, ex2, latent_w, dev) + lq_start
            k_lq = k_pre[:, lk_idx]
            v_lq = v_pre[:, lk_idx]
            lq_ids = _make_rect_pos_ids(lq_exp, lq_bbox, _ROI_T_LQ, pe_mode, target_box=lq_bbox, device=dev)
            lc, ls = _virtual_rope_freqs(lq_ids, rope_theta, axes_dim)
            k_lq = apply_rotary_emb(k_lq, (lc, ls), sequence_dim=1)

            if split_branches:
                out_lq = _dispatch_attention(q_sub, k_lq, v_lq, num_heads=Hh,
                                             backend=backend, parallel_config=parallel_config)
                out_ref = _dispatch_attention(q_sub, k_ref, v_ref, num_heads=Hh,
                                              backend=backend, parallel_config=parallel_config)
                sub_out = (1.0 - detail_beta) * out_lq + detail_beta * out_ref
            else:
                sub_out = _dispatch_attention(
                    q_sub, torch.cat([k_lq, k_ref], dim=1), torch.cat([v_lq, v_ref], dim=1),
                    num_heads=Hh, backend=backend, parallel_config=parallel_config,
                )
        else:
            sub_out = _dispatch_attention(q_sub, k_ref, v_ref, num_heads=Hh,
                                          backend=backend, parallel_config=parallel_config)

        native_out = _aggregate_subquery_outputs(sub_out, n_native, m, agg_mode)
        if _ROI_DEBUG:
            print(f"[roi-qsub] m={m} agg={agg_mode} split={split_branches} beta={detail_beta:.2f} "
                  f"noise={h_f}x{w_f}tok q_sub={tuple(q_sub.shape)} ref_k={k_ref.shape[1]}", flush=True)
        output[:, nq_idx] = (1.0 - noise_alpha) * output[:, nq_idx] + noise_alpha * native_out

    return output
'''
    text = insert_before(text, "\ndef _maybe_virtual_roi(\n", qsup_code, "transformer-qsup-functions")

    # ------------------------------------------------------------------
    # 4) Route roi_variant=q_supersample in _maybe_virtual_roi.
    # ------------------------------------------------------------------
    old = """    if rope_theta is None or rope_axes_dim is None:\n        logger.warning(\"roi_mode 需要 rope_theta/rope_axes_dim，但未传入，跳过 Version B。\")\n        return output\n    return _virtual_roi_qkv_attention(\n        output, q_pre, k_pre, v_pre, ranges, id_patch_pairs, latent_h, latent_w,\n        P=getattr(id_patch_config, \"roi_size\", 24),\n        pe_mode=getattr(id_patch_config, \"roi_pe_mode\", \"pe2\"),\n        include_lq=getattr(id_patch_config, \"roi_include_lq\", True),\n        noise_alpha=getattr(id_patch_config, \"noise_alpha\", 0.5),\n        exp_lq=getattr(id_patch_config, \"expand_ratio_lq\", 1.0),\n        exp_ref=getattr(id_patch_config, \"expand_ratio_ref\", 1.0),\n        exp_min=getattr(id_patch_config, \"expand_min_size\", 0),\n        rope_theta=rope_theta, axes_dim=rope_axes_dim,\n        backend=backend, parallel_config=parallel_config,\n    )\n"""
    new = """    if rope_theta is None or rope_axes_dim is None:\n        logger.warning(\"roi_mode 需要 rope_theta/rope_axes_dim，但未传入，跳过 Version B。\")\n        return output\n\n    roi_variant = getattr(id_patch_config, \"roi_variant\", \"interpolate\")\n    if roi_variant == \"q_supersample\":\n        return _q_supersample_roi_attention(\n            output, q_pre, k_pre, v_pre, ranges, id_patch_pairs, latent_h, latent_w,\n            m=getattr(id_patch_config, \"roi_subsample\", 2),\n            pe_mode=getattr(id_patch_config, \"roi_pe_mode\", \"pe3\"),\n            include_lq=getattr(id_patch_config, \"roi_include_lq\", True),\n            agg_mode=getattr(id_patch_config, \"roi_agg_mode\", \"mean\"),\n            split_branches=getattr(id_patch_config, \"roi_split_branches\", True),\n            detail_beta=getattr(id_patch_config, \"roi_detail_beta\", 0.5),\n            noise_alpha=getattr(id_patch_config, \"noise_alpha\", 0.5),\n            exp_lq=getattr(id_patch_config, \"expand_ratio_lq\", 1.0),\n            exp_ref=getattr(id_patch_config, \"expand_ratio_ref\", 1.0),\n            exp_min=getattr(id_patch_config, \"expand_min_size\", 0),\n            rope_theta=rope_theta, axes_dim=rope_axes_dim,\n            backend=backend, parallel_config=parallel_config,\n        )\n\n    return _virtual_roi_qkv_attention(\n        output, q_pre, k_pre, v_pre, ranges, id_patch_pairs, latent_h, latent_w,\n        P=getattr(id_patch_config, \"roi_size\", 24),\n        pe_mode=getattr(id_patch_config, \"roi_pe_mode\", \"pe2\"),\n        include_lq=getattr(id_patch_config, \"roi_include_lq\", True),\n        noise_alpha=getattr(id_patch_config, \"noise_alpha\", 0.5),\n        exp_lq=getattr(id_patch_config, \"expand_ratio_lq\", 1.0),\n        exp_ref=getattr(id_patch_config, \"expand_ratio_ref\", 1.0),\n        exp_min=getattr(id_patch_config, \"expand_min_size\", 0),\n        rope_theta=rope_theta, axes_dim=rope_axes_dim,\n        backend=backend, parallel_config=parallel_config,\n    )\n"""
    text = replace_once(text, old, new, "transformer-route-qsup")
    return text


def patch_refine(text: str) -> str:
    if "roi_variant: str = \"interpolate\"" not in text:
        text = replace_once(
            text,
            "    roi_down_layer: int = -1\n    # ===== Version A（真·高清 ref 重编码）=====\n",
            "    roi_down_layer: int = -1\n"
            "    # ===== Version B-2（Q-only supersampling）=====\n"
            "    roi_variant: str = \"interpolate\"\n"
            "    roi_subsample: int = 2\n"
            "    roi_agg_mode: str = \"mean\"\n"
            "    roi_split_branches: bool = True\n"
            "    roi_detail_beta: float = 0.5\n"
            "    # ===== Version A（真·高清 ref 重编码）=====\n",
            "refine-config-fields",
        )
    if "roi_variant=kwargs.get('id_patch_roi_variant'" not in text:
        text = replace_once(
            text,
            "                roi_down_layer=kwargs.get('id_patch_roi_down_layer', -1),\n                roi_ref_reencode=kwargs.get('id_patch_roi_ref_reencode', False),\n",
            "                roi_down_layer=kwargs.get('id_patch_roi_down_layer', -1),\n"
            "                roi_variant=kwargs.get('id_patch_roi_variant', 'interpolate'),\n"
            "                roi_subsample=kwargs.get('id_patch_roi_subsample', 2),\n"
            "                roi_agg_mode=kwargs.get('id_patch_roi_agg_mode', 'mean'),\n"
            "                roi_split_branches=kwargs.get('id_patch_roi_split_branches', True),\n"
            "                roi_detail_beta=kwargs.get('id_patch_roi_detail_beta', 0.5),\n"
            "                roi_ref_reencode=kwargs.get('id_patch_roi_ref_reencode', False),\n",
            "refine-constructor-fields",
        )
    text = text.replace(
        "        ys = torch.linspace(float(y1), float(y2), gh)\n"
        "        xs = torch.linspace(float(x1), float(x2), gw)\n",
        "        # bbox 是半开区间 [y1,y2) / [x1,x2)，用 token-center 坐标避免 RoPE 端点拉伸\n"
        "        ys = torch.linspace(float(y1) + 0.5, float(y2) - 0.5, gh)\n"
        "        xs = torch.linspace(float(x1) + 0.5, float(x2) - 0.5, gw)\n",
    )
    return text


def patch_dit(text: str) -> str:
    if "id_patch_roi_variant" not in text:
        text = replace_once(
            text,
            "            dit_params['id_patch_roi_down_layer'] = self.cfg.get('id_patch_roi_down_layer', -1)\n\n            # ===== Version A（真·高清 ref 重编码）=====\n",
            "            dit_params['id_patch_roi_down_layer'] = self.cfg.get('id_patch_roi_down_layer', -1)\n"
            "            # ===== Version B-2（Q-only supersampling）=====\n"
            "            dit_params['id_patch_roi_variant'] = self.cfg.get('id_patch_roi_variant', 'interpolate')\n"
            "            dit_params['id_patch_roi_subsample'] = self.cfg.get('id_patch_roi_subsample', 2)\n"
            "            dit_params['id_patch_roi_agg_mode'] = self.cfg.get('id_patch_roi_agg_mode', 'mean')\n"
            "            dit_params['id_patch_roi_split_branches'] = self.cfg.get('id_patch_roi_split_branches', True)\n"
            "            dit_params['id_patch_roi_detail_beta'] = self.cfg.get('id_patch_roi_detail_beta', 0.5)\n\n"
            "            # ===== Version A（真·高清 ref 重编码）=====\n",
            "dit-load-modules-fields",
        )
    return text


def patch_file(path: Path, fn) -> bool:
    text = path.read_text(encoding="utf-8")
    new = fn(text)
    if new == text:
        print(f"unchanged: {path}")
        return False
    path.write_text(new, encoding="utf-8")
    print(f"patched:   {path}")
    return True


def main() -> None:
    changed = False
    changed |= patch_file(TRANSFORMER, patch_transformer)
    changed |= patch_file(REFINE, patch_refine)
    changed |= patch_file(DIT, patch_dit)
    if changed:
        print("\nPatch applied. Recommended checks:")
        print("  python -m py_compile code/transformer_flux2.py code/refine_model.py code/Dit_pipeline.py")
        print("  ROI_DEBUG=1 run your q_supersample cfg on roi_max_faces=1")
    else:
        print("\nAll targets already contain the q_supersample patch.")


if __name__ == "__main__":
    main()
