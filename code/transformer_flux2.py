# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin, AttentionModuleMixin
from ..attention_dispatch import dispatch_attention_fn
from ..cache_utils import CacheMixin
from ..embeddings import (
    TimestepEmbedding,
    Timesteps,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class IdPatchConfig:
    """ID Patch Attention 的配置。

    在指定层中，lq/ref 的 ID 区域只与对应的 ref/lq ID 区域做 cross-attention，
    非 ID 区域及其他 segment 交互保持 full attention。

    Version A（Expanded-KV Only）参数：
    - expand_ratio_lq:  被 cross-attend 的 lq patch 的扩大倍数 (r_t)，结构侧
    - expand_ratio_ref: 被 cross-attend 的 ref patch 的扩大倍数 (r_s)，细节侧  ← 主旋钮
    - expand_min_size:  扩大后 bbox 在 token 网格上的最小边长（token 数），对小脸兜底

    Version A'（Noise-Fixup）参数：
    - fixup_lqref: 是否保留原始 lq/ref 段 fix-up（默认 True = 现有行为）。设 False 可单独测 A'。
    - fixup_noise: 是否对 noise(输出) 段人脸 query 做 fix-up，让输出直接 attend [lq结构 + ref细节]。
    - noise_alpha: 残差融合强度，output = (1-α)·全注意力 + α·ID注意力，建议 0.4~0.6。

    全默认值 (1.0, 1.0, 0, fixup_lqref=True, fixup_noise=False) 时，与原始 ID Patch Attention 逐 bit 相同。
    """
    idx_single_window: List[int] = field(default_factory=list)  # single block 中使用 id patch attention 的层
    idx_double_window: List[int] = field(default_factory=list)  # double block 中使用 id patch attention 的层
    expand_ratio_lq: float = 1.0
    expand_ratio_ref: float = 1.0
    expand_min_size: int = 0
    fixup_lqref: bool = True
    fixup_noise: bool = False
    noise_alpha: float = 0.5
    # ===== Version B（Virtual ROI-QKV）=====
    roi_mode: bool = False        # 开启 Version B：人脸 ROI 升采样到 P×P 做 attention 再降采样回写
    roi_size: int = 24            # P，虚拟 ROI 边长（token）
    roi_pe_mode: str = "pe2"      # 虚拟 token 位置编码：'pe1'连续/'pe2'压回原框/'pe3'映射到target脸
    roi_include_lq: bool = True   # KV 是否包含 lq 结构 ROI（[lq + ref] vs 仅 ref）
    roi_max_faces: int = -1       # 仅处理前 N 个匹配脸（-1=全部）；P=64 等大开销诊断时设 1~2
    # B-1.5 persist（跨层保持高分辨率）
    roi_persist: bool = False
    roi_up_layer: int = -1
    roi_down_layer: int = -1
    # ===== Version B-2（Q-only supersampling，不插值 V、不下采样虚拟 latent）=====
    roi_variant: str = "interpolate"  # 'interpolate'=旧 B; 'q_supersample'=下一步实验
    roi_subsample: int = 2              # 每个 native noise face token 生成 m×m 个子查询
    roi_agg_mode: str = "mean"        # 子查询输出聚合：mean / center
    roi_split_branches: bool = True     # lq/ref 分支分开 softmax，避免互相稀释
    roi_detail_beta: float = 0.5        # split 分支下 ref detail 分支权重，0=纯 lq, 1=纯 ref
    # ===== Version A（真·高清 ref 重编码）=====
    # noise 脸 query(native,不上采)attend 序列里的高清 ref_hr token(真高频),残差注入。
    roi_ref_reencode: bool = False
    ref_crop_size: int = 512      # ref 脸 crop 在像素空间 resize 到的边长(/16);Dit_pipeline 用


def _get_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    if hasattr(attn, 'fused_projections') and attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


_HAS_NPU = False
try:
    import torch_npu
    _HAS_NPU = True
except ImportError:
    pass


def _nfa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    使用 npu_fusion_attention (NFA) 计算注意力。

    NFA 是 CANN 深度优化的 fused attention 算子，在 Ascend 910B 上
    比 Triton kernel 快 2.5~2.8x（benchmark 验证）。

    Args:
        query:     [B, S_q, H, D]  — BSND 格式，与模型内部 layout 一致
        key:       [B, S_kv, H, D]
        value:     [B, S_kv, H, D]
        num_heads: 注意力头数 (H)
        scale:     缩放因子，默认 1/sqrt(D)

    Returns:
        output: [B, S_q, H, D]
    """
    D = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # 直接使用 BSND layout，避免 transpose + contiguous 的额外显存拷贝
    output = torch_npu.npu_fusion_attention(
        query, key, value,
        head_num=num_heads,
        input_layout="BSND",
        scale=scale,
    )[0]

    return output


def _dispatch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: Optional[int] = None,
    scale: Optional[float] = None,
    attn_mask: Optional[torch.Tensor] = None,
    backend=None,
    parallel_config=None,
) -> torch.Tensor:
    """
    统一注意力分发：NPU 上优先使用 NFA，否则回退到 dispatch_attention_fn。
    """
    if _HAS_NPU and query.device.type == 'npu':
        if num_heads is None:
            num_heads = query.shape[2]  # [B, S, H, D] 中的 H
        return _nfa_attention(query, key, value, num_heads, scale=scale)
    else:
        return dispatch_attention_fn(
            query, key, value,
            attn_mask=attn_mask,
            backend=backend,
            parallel_config=parallel_config,
        )


def _compute_segment_ranges(
    txt_len: int,
    seq_noise: int,
    seq_lq: int,
    seq_ref: int,
    img_first: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """
    计算各 segment 在合并序列中的 (start, end) 范围。

    Args:
        txt_len:   text token 数量
        seq_noise: noise latent token 数量
        seq_lq:    low-quality latent token 数量
        seq_ref:   reference latent token 数量
        img_first: True 时图像在前 [noise, lq, ref, txt]；
                   False 时文本在前 [txt, noise, lq, ref]

    Returns:
        dict 含 'txt' / 'noise' / 'lq' / 'ref' 各自的 (start, end)
    """
    if img_first:
        noise_start = 0
        noise_end = noise_start + seq_noise
        lq_start = noise_end
        lq_end = lq_start + seq_lq
        ref_start = lq_end
        ref_end = ref_start + seq_ref
        txt_start = ref_end
        txt_end = txt_start + txt_len
    else:
        txt_start = 0
        txt_end = txt_start + txt_len
        noise_start = txt_end
        noise_end = noise_start + seq_noise
        lq_start = noise_end
        lq_end = lq_start + seq_lq
        ref_start = lq_end
        ref_end = ref_start + seq_ref

    return {
        'txt': (txt_start, txt_end),
        'noise': (noise_start, noise_end),
        'lq': (lq_start, lq_end),
        'ref': (ref_start, ref_end),
    }


def _build_2d_rect_indices(
    y1: int, x1: int, y2: int, x2: int,
    latent_w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    将 2D 矩形区域 (y1, x1) ~ (y2, x2) 转换为 row-major 展平后的 1D token 索引。
    """
    ys = torch.arange(y1, y2, device=device)           # [patch_h]
    xs = torch.arange(x1, x2, device=device)           # [patch_w]
    row_offsets = ys * latent_w                          # [patch_h]
    indices = (row_offsets.unsqueeze(1) + xs.unsqueeze(0)).reshape(-1)  # [patch_h * patch_w]
    return indices


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    ratio: float,
    min_size: int,
    latent_h: int,
    latent_w: int,
) -> Tuple[int, int, int, int]:
    """
    [Version A 新增] 围绕中心扩大 bbox，并 clip 到 token 网格边界。

    Args:
        bbox:     (y1, x1, y2, x2)，token 网格坐标（与 id_patch_pairs 一致）
        ratio:    宽高扩大倍数（1.0 = 不变）
        min_size: 扩大后的最小边长（token 数），用于对小目标兜底（0 = 关闭）
        latent_h: token 网格高（= patch_data 的 H，= seq // latent_w 的行数上界）
        latent_w: token 网格宽

    Returns:
        (ny1, nx1, ny2, nx2)，clip 到 [0, latent_h) x [0, latent_w)。
        ratio=1.0 且 min_size<=原边长 时，返回值与输入 bbox 相同。
    """
    y1, x1, y2, x2 = bbox
    h = y2 - y1
    w = x2 - x1
    cy = (y1 + y2) / 2.0
    cx = (x1 + x2) / 2.0
    new_h = max(h * ratio, float(min_size))
    new_w = max(w * ratio, float(min_size))
    ny1 = max(0, int(round(cy - new_h / 2.0)))
    nx1 = max(0, int(round(cx - new_w / 2.0)))
    ny2 = min(latent_h, int(round(cy + new_h / 2.0)))
    nx2 = min(latent_w, int(round(cx + new_w / 2.0)))
    # 兜底：避免数值意外导致空框
    ny2 = max(ny2, ny1 + 1)
    nx2 = max(nx2, nx1 + 1)
    ny2 = min(ny2, latent_h)
    nx2 = min(nx2, latent_w)
    return (ny1, nx1, ny2, nx2)


def _id_patch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ranges: Dict[str, Tuple[int, int]],
    id_patch_pairs: List[Dict[str, Tuple[int, int, int, int]]],
    latent_h: int,
    latent_w: int,
    expand_ratio_lq: float = 1.0,
    expand_ratio_ref: float = 1.0,
    expand_min_size: int = 0,
    fixup_lqref: bool = True,
    fixup_noise: bool = False,
    noise_alpha: float = 0.5,
    backend=None,
    parallel_config=None,
) -> torch.Tensor:
    """
    ID Patch Attention：lq/ref/noise 的 ID 区域做定向 cross-attention。

    Version A（Expanded-KV Only，fixup_lqref=True）：
        - lq/ref query 取「原始」bbox，被 cross-attend 的一侧 KV 取「扩大后」bbox（E_i）；
        - 用已 apply RoPE 的现有 token，零 PE 改动（PE-0）。
        - 注意：lq query 的 base 含整张 lq+noise，ref_patch 占比极小 → 对输出影响被稀释（实验证实近乎不可见）。

    Version A'（Noise-Fixup，fixup_noise=True）：
        - 直接对 noise(输出) 段人脸 query 做 fix-up（noise 与 lq 空间对齐，共用 bbox）；
        - KV = [局部 lq（结构）+ 扩大 ref（细节）]，紧凑 base 不放全图，避免稀释；
        - 残差融合写回 noise：output = (1-noise_alpha)·全注意力 + noise_alpha·ID注意力。
        - 这是让「输出脸」真正响应 ref 的关键路径（lq 给结构、ref 给细节，靠内容匹配软对齐姿态差异）。

    全默认值时与原实现逐 bit 相同。
    """
    B, S, H, D = query.shape

    txt_start, txt_end = ranges['txt']
    noise_start, noise_end = ranges['noise']
    lq_start, lq_end = ranges['lq']
    ref_start, ref_end = ranges['ref']

    # ---- 边界情况：无匹配对或无 lq/ref segment → 直接 full attention ----
    if (not id_patch_pairs
            or lq_end <= lq_start
            or ref_end <= ref_start):
        return _dispatch_attention(
            query, key, value,
            num_heads=H,
            backend=backend,
            parallel_config=parallel_config,
        )

    # ============ Step 1: 全量 full attention ============
    output = _dispatch_attention(
        query, key, value,
        num_heads=H,
        backend=backend,
        parallel_config=parallel_config,
    )

    # ============ Step 2: 为 fix-up 预提取各 segment 的 KV ============
    lq_key = key[:, lq_start:lq_end]
    lq_value = value[:, lq_start:lq_end]
    ref_key = key[:, ref_start:ref_end]
    ref_value = value[:, ref_start:ref_end]

    base_parts_k = []
    base_parts_v = []
    if txt_end > txt_start:
        base_parts_k.append(key[:, txt_start:txt_end])
        base_parts_v.append(value[:, txt_start:txt_end])
    if noise_end > noise_start:
        base_parts_k.append(key[:, noise_start:noise_end])
        base_parts_v.append(value[:, noise_start:noise_end])

    base_for_lq_k = torch.cat(base_parts_k + [lq_key], dim=1)
    base_for_lq_v = torch.cat(base_parts_v + [lq_value], dim=1)

    base_for_ref_k = torch.cat(base_parts_k + [ref_key], dim=1)
    base_for_ref_v = torch.cat(base_parts_v + [ref_value], dim=1)

    # ============ Step 3: 逐对 fix-up ID patch 的注意力 ============
    for pair in id_patch_pairs:
        lq_bbox = pair['lq']
        ref_bbox = pair['ref']

        # [Version A] 被 cross-attend 的一侧使用「扩大后」bbox 构建 KV
        lq_bbox_exp = _expand_bbox(lq_bbox, expand_ratio_lq, expand_min_size, latent_h, latent_w)
        ref_bbox_exp = _expand_bbox(ref_bbox, expand_ratio_ref, expand_min_size, latent_h, latent_w)

        # query 索引：原始 bbox（写回范围不变）
        lq_local_indices = _build_2d_rect_indices(
            lq_bbox[0], lq_bbox[1], lq_bbox[2], lq_bbox[3],
            latent_w, query.device,
        )
        ref_local_indices = _build_2d_rect_indices(
            ref_bbox[0], ref_bbox[1], ref_bbox[2], ref_bbox[3],
            latent_w, query.device,
        )

        # KV 索引：扩大后 bbox
        lq_kv_indices = _build_2d_rect_indices(
            lq_bbox_exp[0], lq_bbox_exp[1], lq_bbox_exp[2], lq_bbox_exp[3],
            latent_w, query.device,
        )
        ref_kv_indices = _build_2d_rect_indices(
            ref_bbox_exp[0], ref_bbox_exp[1], ref_bbox_exp[2], ref_bbox_exp[3],
            latent_w, query.device,
        )

        # ===== Version A：lq/ref 段 fix-up（被 attend 的一侧用扩大 bbox 的现有 token）=====
        if fixup_lqref:
            # ---- Fix-up lq_patch_i queries（attend 到 base + 扩大后的 ref patch）----
            if lq_local_indices.numel() > 0:
                lq_global_indices = lq_local_indices + lq_start

                lq_patch_query = query[:, lq_global_indices]
                ref_patch_key = ref_key[:, ref_kv_indices]
                ref_patch_value = ref_value[:, ref_kv_indices]

                combined_k = torch.cat([base_for_lq_k, ref_patch_key], dim=1)
                combined_v = torch.cat([base_for_lq_v, ref_patch_value], dim=1)

                lq_patch_out = _dispatch_attention(
                    lq_patch_query, combined_k, combined_v,
                    num_heads=H,
                    backend=backend,
                    parallel_config=parallel_config,
                )
                output[:, lq_global_indices] = lq_patch_out

            # ---- Fix-up ref_patch_i queries（attend 到 base + 扩大后的 lq patch）----
            if ref_local_indices.numel() > 0:
                ref_global_indices = ref_local_indices + ref_start

                ref_patch_query = query[:, ref_global_indices]
                lq_patch_key = lq_key[:, lq_kv_indices]
                lq_patch_value = lq_value[:, lq_kv_indices]

                combined_k = torch.cat([base_for_ref_k, lq_patch_key], dim=1)
                combined_v = torch.cat([base_for_ref_v, lq_patch_value], dim=1)

                ref_patch_out = _dispatch_attention(
                    ref_patch_query, combined_k, combined_v,
                    num_heads=H,
                    backend=backend,
                    parallel_config=parallel_config,
                )
                output[:, ref_global_indices] = ref_patch_out

        # ===== Version A'：noise(输出) 段 fix-up（lq 结构 + ref 细节，残差注入）=====
        # noise 与 lq 空间对齐，共用 lq 的 bbox/索引；query 取原始 bbox，写回也只在原始 bbox。
        if fixup_noise and lq_local_indices.numel() > 0 and noise_end > noise_start:
            noise_q_indices = lq_local_indices + noise_start
            noise_q = query[:, noise_q_indices]

            # 紧凑 base：结构←局部 lq（扩大 r_t），细节←ref（扩大 r_s）；不放整图，避免稀释
            k_struct = lq_key[:, lq_kv_indices]
            v_struct = lq_value[:, lq_kv_indices]
            k_detail = ref_key[:, ref_kv_indices]
            v_detail = ref_value[:, ref_kv_indices]

            combined_k = torch.cat([k_struct, k_detail], dim=1)
            combined_v = torch.cat([v_struct, v_detail], dim=1)

            noise_out = _dispatch_attention(
                noise_q, combined_k, combined_v,
                num_heads=H,
                backend=backend,
                parallel_config=parallel_config,
            )
            # 残差融合：保留全注意力的全局一致性，noise_alpha 注入 ID 细节
            output[:, noise_q_indices] = (
                (1.0 - noise_alpha) * output[:, noise_q_indices] + noise_alpha * noise_out
            )

    return output


# ============================================================================
# Version B：Virtual ROI-QKV（attention 内把人脸 ROI 升采样到 P×P，高密度 attend，
# 再降采样回 native，残差注入 noise 段）。单趟、无 crop、无后处理。
# ============================================================================

# 各段的 stream/frame id（来自 KleinLatentProcessor.prepare_image_ids(scale=10)）：
# noise=prepare_latent_ids → T=0；lq=第0个 condition → T=10；ref=第1个 → T=20。
_ROI_T_NOISE = 0
_ROI_T_LQ = 10
_ROI_T_REF = 20

import os as _os
_ROI_DEBUG = bool(int(_os.environ.get("ROI_DEBUG", "0")))
_ROI_PERSIST_WARNED = False


def _as_int(v) -> int:
    """把尺寸值强制成 python int（防御 list/tuple/np/0-d tensor，例如 roi_size 被写成 [24,24]）。"""
    if isinstance(v, (tuple, list)):
        v = v[0]
    if hasattr(v, "item"):
        v = v.item()
    return int(v)


def _resample_tokens_2d(x: torch.Tensor, h, w, out_h, out_w) -> torch.Tensor:
    """
    把一段 row-major 排列的 token 在 2D 上 bilinear 重采样。
    x: [B, h*w, Hh, D] → [B, out_h*out_w, Hh, D]
    """
    h, w, out_h, out_w = _as_int(h), _as_int(w), _as_int(out_h), _as_int(out_w)
    B, n, Hh, D = x.shape
    # [B, h*w, Hh, D] -> [B, h, w, Hh*D] -> [B, Hh*D, h, w]
    x = x.reshape(B, h, w, Hh * D).permute(0, 3, 1, 2).contiguous()
    orig_dtype = x.dtype
    x = F.interpolate(x.float(), size=(out_h, out_w), mode="bilinear", align_corners=False).to(orig_dtype)
    # [B, Hh*D, out_h, out_w] -> [B, out_h*out_w, Hh, D]
    x = x.permute(0, 2, 3, 1).contiguous().reshape(B, out_h * out_w, Hh, D)
    return x


def _make_roi_pos_ids(
    sample_box: Tuple[int, int, int, int],
    orig_box: Tuple[int, int, int, int],
    P: int,
    t_val: int,
    mode: str,
    target_box: Optional[Tuple[int, int, int, int]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    为 P×P 虚拟 token 生成 4D 位置 id (T, H, W, L)。
    - sample_box: 实际采样所用的（扩大后）框，决定连续真实坐标范围。
    - orig_box:   原始人脸框（PE-2 压缩的参照中心/范围）。
    - mode: 'pe1' 连续真实坐标 / 'pe2' 压回原框范围 / 'pe3' 映射到 target_box 坐标系。
    返回 [P*P, 4]（float）。
    """
    y1, x1, y2, x2 = sample_box
    ys = torch.linspace(float(y1) + 0.5, float(y2) - 0.5, P, device=device)
    xs = torch.linspace(float(x1) + 0.5, float(x2) - 0.5, P, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy = gy.reshape(-1)
    gx = gx.reshape(-1)

    if mode == "pe2":
        oy1, ox1, oy2, ox2 = orig_box
        cy = (oy1 + oy2) / 2.0
        cx = (ox1 + ox2) / 2.0
        ry = max(y2 - y1, 1e-6) / max(oy2 - oy1, 1e-6)   # = 扩大倍数
        rx = max(x2 - x1, 1e-6) / max(ox2 - ox1, 1e-6)
        gy = cy + (gy - cy) / ry
        gx = cx + (gx - cx) / rx
    elif mode == "pe3" and target_box is not None:
        ty1, tx1, ty2, tx2 = target_box
        ny = (gy - y1) / max(y2 - y1, 1e-6)
        nx = (gx - x1) / max(x2 - x1, 1e-6)
        gy = ty1 + ny * (ty2 - ty1)
        gx = tx1 + nx * (tx2 - tx1)
    # 'pe1': 保持 sample_box 连续真实坐标

    t = torch.full_like(gy, float(t_val))
    l = torch.zeros_like(gy)
    return torch.stack([t, gy, gx, l], dim=-1)   # [P*P, 4]


def _virtual_rope_freqs(ids: torch.Tensor, theta: int, axes_dim) -> Tuple[torch.Tensor, torch.Tensor]:
    """复刻 Flux2PosEmbed.forward，为虚拟 token 算 (cos, sin)，各 [N, sum(axes_dim)]。"""
    is_npu = ids.device.type == "npu"
    is_mps = ids.device.type == "mps"
    freqs_dtype = torch.float32 if (is_npu or is_mps) else torch.float64
    pos = ids.float()
    cos_out, sin_out = [], []
    for i in range(len(axes_dim)):
        c, s = get_1d_rotary_pos_embed(
            axes_dim[i], pos[..., i], theta=theta,
            repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype,
        )
        cos_out.append(c)
        sin_out.append(s)
    cos = torch.cat(cos_out, dim=-1).to(ids.device)
    sin = torch.cat(sin_out, dim=-1).to(ids.device)
    return cos, sin


def _virtual_roi_qkv_attention(
    output: torch.Tensor,
    q_pre: torch.Tensor, k_pre: torch.Tensor, v_pre: torch.Tensor,
    ranges: Dict[str, Tuple[int, int]],
    id_patch_pairs: List[Dict[str, Tuple[int, int, int, int]]],
    latent_h: int, latent_w: int,
    P: int, pe_mode: str, include_lq: bool, noise_alpha: float,
    exp_lq: float, exp_ref: float, exp_min: int,
    rope_theta: int, axes_dim,
    backend=None, parallel_config=None,
) -> torch.Tensor:
    """
    Version B 主体：noise 脸 query 与 ref(/lq) KV 都 ROIAlign 到 P×P，高密度 attend，
    降采样回 native noise 脸 token，noise_alpha 残差写回。用 PRE-RoPE 的 q/k/v，对虚拟 token 重配 RoPE。
    output: [B,S,Hh,D]（主全注意力结果，会被原地融合）。
    """
    P = _as_int(P)   # 防御：roi_size 被写成 [24,24] 之类
    B, S, Hh, D = q_pre.shape
    noise_start, noise_end = ranges["noise"]
    lq_start, lq_end = ranges["lq"]
    ref_start, ref_end = ranges["ref"]
    if not id_patch_pairs or noise_end <= noise_start or ref_end <= ref_start:
        return output
    dev = q_pre.device

    for pair in id_patch_pairs:
        lq_bbox = pair["lq"]
        ref_bbox = pair["ref"]
        ly1, lx1, ly2, lx2 = lq_bbox
        h_f, w_f = ly2 - ly1, lx2 - lx1
        if h_f <= 0 or w_f <= 0:
            continue

        ref_exp = _expand_bbox(ref_bbox, exp_ref, exp_min, latent_h, latent_w)
        lq_exp = _expand_bbox(lq_bbox, exp_lq, exp_min, latent_h, latent_w)

        # ---- 虚拟 query：noise 脸(原始框) 升到 P×P ----
        nq_idx = _build_2d_rect_indices(ly1, lx1, ly2, lx2, latent_w, dev) + noise_start
        virt_q = _resample_tokens_2d(q_pre[:, nq_idx], h_f, w_f, P, P)               # [B,P²,Hh,D]
        q_ids = _make_roi_pos_ids(lq_bbox, lq_bbox, P, _ROI_T_NOISE, "pe1", device=dev)
        qc, qs = _virtual_rope_freqs(q_ids, rope_theta, axes_dim)
        virt_q = apply_rotary_emb(virt_q, (qc, qs), sequence_dim=1)

        # ---- 虚拟 ref KV（细节）：扩大 ref 区域 升到 P×P ----
        ry1, rx1, ry2, rx2 = ref_exp
        rk_idx = _build_2d_rect_indices(ry1, rx1, ry2, rx2, latent_w, dev) + ref_start
        virt_k_ref = _resample_tokens_2d(k_pre[:, rk_idx], ry2 - ry1, rx2 - rx1, P, P)
        virt_v_ref = _resample_tokens_2d(v_pre[:, rk_idx], ry2 - ry1, rx2 - rx1, P, P)
        ref_ids = _make_roi_pos_ids(ref_exp, ref_bbox, P, _ROI_T_REF, pe_mode, target_box=lq_bbox, device=dev)
        rc, rs = _virtual_rope_freqs(ref_ids, rope_theta, axes_dim)
        virt_k_ref = apply_rotary_emb(virt_k_ref, (rc, rs), sequence_dim=1)

        ks, vs = [virt_k_ref], [virt_v_ref]

        # ---- 可选 虚拟 lq KV（结构）：扩大 lq 区域 升到 P×P ----
        if include_lq and lq_end > lq_start:
            ey1, ex1, ey2, ex2 = lq_exp
            lk_idx = _build_2d_rect_indices(ey1, ex1, ey2, ex2, latent_w, dev) + lq_start
            virt_k_lq = _resample_tokens_2d(k_pre[:, lk_idx], ey2 - ey1, ex2 - ex1, P, P)
            virt_v_lq = _resample_tokens_2d(v_pre[:, lk_idx], ey2 - ey1, ex2 - ex1, P, P)
            lq_ids = _make_roi_pos_ids(lq_exp, lq_bbox, P, _ROI_T_LQ, pe_mode, target_box=lq_bbox, device=dev)
            lc, ls = _virtual_rope_freqs(lq_ids, rope_theta, axes_dim)
            virt_k_lq = apply_rotary_emb(virt_k_lq, (lc, ls), sequence_dim=1)
            ks = [virt_k_lq, virt_k_ref]
            vs = [virt_v_lq, virt_v_ref]

        virt_k = torch.cat(ks, dim=1)
        virt_v = torch.cat(vs, dim=1)

        if _ROI_DEBUG:
            print(f"[roi] P={P} pe={pe_mode} noise_face={h_f}x{w_f} "
                  f"virt_q={tuple(virt_q.shape)} virt_k={tuple(virt_k.shape)}", flush=True)

        virt_out = _dispatch_attention(virt_q, virt_k, virt_v, num_heads=Hh,
                                       backend=backend, parallel_config=parallel_config)  # [B,P²,Hh,D]
        out_native = _resample_tokens_2d(virt_out, P, P, h_f, w_f)                          # [B,h_f*w_f,Hh,D]
        output[:, nq_idx] = (1.0 - noise_alpha) * output[:, nq_idx] + noise_alpha * out_native

    return output



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

def _maybe_virtual_roi(
    output, q_pre, k_pre, v_pre, ranges, id_patch_pairs, latent_h, latent_w,
    id_patch_config, rope_theta, rope_axes_dim, backend, parallel_config,
):
    """从 id_patch_config 取 Version B(per-layer)参数并调用 _virtual_roi_qkv_attention。
    persist 模式由 forward 处理(_persist_append/_persist_collapse),不走这里。"""
    if rope_theta is None or rope_axes_dim is None:
        logger.warning("roi_mode 需要 rope_theta/rope_axes_dim，但未传入，跳过 Version B。")
        return output

    roi_variant = getattr(id_patch_config, "roi_variant", "interpolate")
    if roi_variant == "q_supersample":
        return _q_supersample_roi_attention(
            output, q_pre, k_pre, v_pre, ranges, id_patch_pairs, latent_h, latent_w,
            m=getattr(id_patch_config, "roi_subsample", 2),
            pe_mode=getattr(id_patch_config, "roi_pe_mode", "pe3"),
            include_lq=getattr(id_patch_config, "roi_include_lq", True),
            agg_mode=getattr(id_patch_config, "roi_agg_mode", "mean"),
            split_branches=getattr(id_patch_config, "roi_split_branches", True),
            detail_beta=getattr(id_patch_config, "roi_detail_beta", 0.5),
            noise_alpha=getattr(id_patch_config, "noise_alpha", 0.5),
            exp_lq=getattr(id_patch_config, "expand_ratio_lq", 1.0),
            exp_ref=getattr(id_patch_config, "expand_ratio_ref", 1.0),
            exp_min=getattr(id_patch_config, "expand_min_size", 0),
            rope_theta=rope_theta, axes_dim=rope_axes_dim,
            backend=backend, parallel_config=parallel_config,
        )

    return _virtual_roi_qkv_attention(
        output, q_pre, k_pre, v_pre, ranges, id_patch_pairs, latent_h, latent_w,
        P=getattr(id_patch_config, "roi_size", 24),
        pe_mode=getattr(id_patch_config, "roi_pe_mode", "pe2"),
        include_lq=getattr(id_patch_config, "roi_include_lq", True),
        noise_alpha=getattr(id_patch_config, "noise_alpha", 0.5),
        exp_lq=getattr(id_patch_config, "expand_ratio_lq", 1.0),
        exp_ref=getattr(id_patch_config, "expand_ratio_ref", 1.0),
        exp_min=getattr(id_patch_config, "expand_min_size", 0),
        rope_theta=rope_theta, axes_dim=rope_axes_dim,
        backend=backend, parallel_config=parallel_config,
    )


def _ref_hr_attention(
    output: torch.Tensor,
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    ranges: Dict[str, Tuple[int, int]],
    id_patch_pairs: List[Dict],
    latent_h: int, latent_w: int,
    exp_lq: float, exp_min: int, noise_alpha: float,
    backend=None, parallel_config=None,
) -> torch.Tensor:
    """
    Version A：noise 脸 query（native，原始 bbox，不上采→不引入低通糊）attend
    一个紧凑 KV = [局部 lq(结构) + 高清 ref_hr token(真高频，已在序列里、已被投影+RoPE)]，
    noise_alpha 残差写回 noise 脸。
    每个 pair 需带 'ref_hr'=(img_local_start, img_local_end)（refine_model 写入）。
    用 POST-RoPE 的 q/k/v（ref_hr 在序列里，已 RoPE）。
    """
    B, S, H, D = query.shape
    noise_start, _ = ranges["noise"]
    lq_start, lq_end = ranges["lq"]
    dev = query.device
    for pair in id_patch_pairs:
        rh = pair.get("ref_hr")
        if rh is None:
            continue
        ly1, lx1, ly2, lx2 = pair["lq"]
        if (ly2 - ly1) <= 0 or (lx2 - lx1) <= 0:
            continue
        nq_idx = _build_2d_rect_indices(ly1, lx1, ly2, lx2, latent_w, dev) + noise_start
        noise_q = query[:, nq_idx]                                  # native，不重采样

        # 结构：局部(扩大) lq 区域
        ey1, ex1, ey2, ex2 = _expand_bbox(pair["lq"], exp_lq, exp_min, latent_h, latent_w)
        lk_idx = _build_2d_rect_indices(ey1, ex1, ey2, ex2, latent_w, dev) + lq_start
        k_struct, v_struct = key[:, lk_idx], value[:, lk_idx]

        # 细节：高清 ref_hr 段（global = noise_start + image-local）
        g0, g1 = noise_start + int(rh[0]), noise_start + int(rh[1])
        k_detail, v_detail = key[:, g0:g1], value[:, g0:g1]

        ck = torch.cat([k_struct, k_detail], dim=1)
        cv = torch.cat([v_struct, v_detail], dim=1)
        if _ROI_DEBUG:
            print(f"[refhr] noise_q={tuple(noise_q.shape)} lq_struct={k_struct.shape[1]} "
                  f"ref_hr={k_detail.shape[1]} (range {g0}:{g1})", flush=True)
        out_i = _dispatch_attention(noise_q, ck, cv, num_heads=H,
                                    backend=backend, parallel_config=parallel_config)
        output[:, nq_idx] = (1.0 - noise_alpha) * output[:, nq_idx] + noise_alpha * out_i
    return output


def _persist_append(hidden_img, img_ids, pos_embed, text_rope, id_patch_config,
                    id_patch_pairs, latent_h, latent_w, seq_noise, seq_lq):
    """[B-1.5 persist · 三块影子] 对每个 ID,在 image 序列尾部追加 P×P 高密度"脸影子":
      - noise 影子(query/写回):exact 脸框, noise 段, T=0
      - lq 影子(结构):扩大脸框, lq 段, T=10
      - ref 影子(细节):扩大脸框, ref 段, T=20
    都是从对应段的 native 脸区**插值**而来;扩展 img_ids、重算 concat_rotary_emb。
    返回 (extended_hidden_img, new_concat_rope, persist_state)。
    persist_state = {'faces': [(noise_local_idx, h_f, w_f, P, noise_off)], 'total': int}
    （collapse 只回写 noise 影子;lq/ref 影子仅供注意力,丢弃。）
    """
    P = _as_int(getattr(id_patch_config, "roi_size", 24))
    pe_mode = getattr(id_patch_config, "roi_pe_mode", "pe2")
    exp_lq = getattr(id_patch_config, "expand_ratio_lq", 1.0)
    exp_ref = getattr(id_patch_config, "expand_ratio_ref", 1.0)
    exp_min = getattr(id_patch_config, "expand_min_size", 0)
    dev = hidden_img.device
    lq_base = seq_noise               # lq 段在 image 内的起点
    ref_base = seq_noise + seq_lq     # ref 段在 image 内的起点

    def _shadow(seg_base, box, t_val):
        y1, x1, y2, x2 = box
        idx = _build_2d_rect_indices(y1, x1, y2, x2, latent_w, dev) + seg_base
        sh = _resample_tokens_2d(hidden_img[:, idx].unsqueeze(2), y2 - y1, x2 - x1, P, P).squeeze(2)
        return sh                                                   # [B, P², C]

    faces, appended, appended_ids, off = [], [], [], 0
    for pair in id_patch_pairs:
        lq_bbox, ref_bbox = pair["lq"], pair["ref"]
        ly1, lx1, ly2, lx2 = lq_bbox
        h_f, w_f = ly2 - ly1, lx2 - lx1
        if h_f <= 0 or w_f <= 0:
            continue
        lq_exp = _expand_bbox(lq_bbox, exp_lq, exp_min, latent_h, latent_w)
        ref_exp = _expand_bbox(ref_bbox, exp_ref, exp_min, latent_h, latent_w)

        if _ROI_DEBUG:
            s = 16  # token->px：VAE 8x 下采 + patchify 2x = 16x
            rh, rw = ref_exp[2] - ref_exp[0], ref_exp[3] - ref_exp[1]
            print(f"[roi-persist] 脸: noise {h_f}x{w_f}tok(~{h_f*s}x{w_f*s}px), "
                  f"ref源(扩) {rh}x{rw}tok(~{rh*s}x{rw*s}px); 影子 P={P}(~{P*s}px-equiv); "
                  f"对标1k需 P≈{1024//s}, 当前 P/64={P/64:.2f}。"
                  f"注:影子是插值放大,真细节上限=源 token 数(~{max(h_f, w_f, rh, rw)}tok)", flush=True)

        # noise 影子（query/写回）：exact 脸框
        noise_off = off
        appended.append(_shadow(0, lq_bbox, _ROI_T_NOISE))
        appended_ids.append(_make_roi_pos_ids(lq_bbox, lq_bbox, P, _ROI_T_NOISE, "pe1",
                                              target_box=lq_bbox, device=dev))
        off += P * P
        # lq 影子（结构）：扩大脸框
        appended.append(_shadow(lq_base, lq_exp, _ROI_T_LQ))
        appended_ids.append(_make_roi_pos_ids(lq_exp, lq_bbox, P, _ROI_T_LQ, pe_mode,
                                              target_box=lq_bbox, device=dev))
        off += P * P
        # ref 影子（细节）：扩大脸框
        appended.append(_shadow(ref_base, ref_exp, _ROI_T_REF))
        appended_ids.append(_make_roi_pos_ids(ref_exp, ref_bbox, P, _ROI_T_REF, pe_mode,
                                              target_box=lq_bbox, device=dev))
        off += P * P

        faces.append((_build_2d_rect_indices(ly1, lx1, ly2, lx2, latent_w, dev), h_f, w_f, P, noise_off))

    if not appended:
        return hidden_img, None, None
    hidden_ext = torch.cat([hidden_img, torch.cat(appended, dim=1)], dim=1)
    img_ids_ext = torch.cat([img_ids.float(), torch.cat(appended_ids, dim=0).to(dev)], dim=0)
    img_rope = pos_embed(img_ids_ext)
    new_concat = (torch.cat([text_rope[0], img_rope[0]], dim=0),
                  torch.cat([text_rope[1], img_rope[1]], dim=0))
    if _ROI_DEBUG:
        print(f"[roi-persist] 3-shadow append {off} tokens ({len(faces)} faces x3, P={P}), "
              f"seq {hidden_img.shape[1]}->{hidden_ext.shape[1]}", flush=True)
    return hidden_ext, new_concat, {"faces": faces, "total": off}


def _persist_collapse(hidden_full, persist_state, num_txt_tokens, noise_alpha):
    """[B-1.5 persist] 单流阶段后(hidden_full=[B, txt+image_ext, C]):把每个 ID 的
    **noise 影子**降采样回 native,残差融合进 noise 脸位置;lq/ref 影子丢弃,裁掉全部尾部。"""
    total = persist_state["total"]
    tail = hidden_full[:, -total:]
    body = hidden_full[:, :-total]
    for (idx, h_f, w_f, P, noise_off) in persist_state["faces"]:
        face_hr = tail[:, noise_off:noise_off + P * P].unsqueeze(2)       # [B, P², 1, C]
        native = _resample_tokens_2d(face_hr, P, P, h_f, w_f).squeeze(2)  # [B, h*w, C]
        tgt = idx + num_txt_tokens                                       # noise 脸在 [txt,noise,...] 的全局位置
        body[:, tgt] = (1.0 - noise_alpha) * body[:, tgt] + noise_alpha * native
    return body


# ============ Processor 类 ============
class Flux2SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Flux2FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: float = 3.0,
        inner_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


class Flux2AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attn_segments: Optional[Tuple[int, int, int]] = None,
        txt_len: Optional[int] = None,
        layer_idx: Optional[int] = None,
        is_single_block: bool = False,
        id_patch_config: Optional["IdPatchConfig"] = None,
        id_patch_pairs: Optional[List[Dict[str, Tuple[int, int, int, int]]]] = None,
        latent_h: Optional[int] = None,
        latent_w: Optional[int] = None,
        rope_theta: Optional[int] = None,
        rope_axes_dim: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
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

        # 保存 PRE-RoPE 的 q/k/v（Version B 用：对虚拟 token 重配 RoPE）
        q_pre, k_pre, v_pre = query, key, value

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # ============ ID Patch Attention 逻辑 ============
        use_id_patch = (
            id_patch_config is not None
            and id_patch_pairs is not None
            and attn_segments is not None
            and txt_len is not None
            and latent_h is not None
            and latent_w is not None
        )

        if use_id_patch and layer_idx is not None:
            active_layers = (
                id_patch_config.idx_double_window
                if not is_single_block
                else id_patch_config.idx_single_window
            )
            if layer_idx not in active_layers:
                use_id_patch = False

        # ============ 路由 ============
        if use_id_patch:
            sn, sl, sr = attn_segments
            st = txt_len
            ranges = _compute_segment_ranges(st, sn, sl, sr, img_first=False)
            roi_mode = getattr(id_patch_config, "roi_mode", False)
            ref_reencode = getattr(id_patch_config, "roi_ref_reencode", False)

            hidden_states = _id_patch_attention(
                query, key, value,
                ranges=ranges,
                id_patch_pairs=id_patch_pairs,
                latent_h=latent_h,
                latent_w=latent_w,
                expand_ratio_lq=getattr(id_patch_config, "expand_ratio_lq", 1.0),
                expand_ratio_ref=getattr(id_patch_config, "expand_ratio_ref", 1.0),
                expand_min_size=getattr(id_patch_config, "expand_min_size", 0),
                fixup_lqref=getattr(id_patch_config, "fixup_lqref", True),
                # roi_mode / ref_reencode 时 noise 注入交给对应路径，避免重复
                fixup_noise=getattr(id_patch_config, "fixup_noise", False) and not roi_mode and not ref_reencode,
                noise_alpha=getattr(id_patch_config, "noise_alpha", 0.5),
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

            if ref_reencode:
                # Version A：noise 脸 query attend 序列里的高清 ref_hr token
                hidden_states = _ref_hr_attention(
                    hidden_states, query, key, value, ranges, id_patch_pairs, latent_h, latent_w,
                    exp_lq=getattr(id_patch_config, "expand_ratio_lq", 1.0),
                    exp_min=getattr(id_patch_config, "expand_min_size", 0),
                    noise_alpha=getattr(id_patch_config, "noise_alpha", 0.5),
                    backend=self._attention_backend, parallel_config=self._parallel_config,
                )
            elif roi_mode and not getattr(id_patch_config, "roi_persist", False):
                hidden_states = _maybe_virtual_roi(
                    hidden_states, q_pre, k_pre, v_pre, ranges, id_patch_pairs,
                    latent_h, latent_w, id_patch_config, rope_theta, rope_axes_dim,
                    self._attention_backend, self._parallel_config,
                )
        else:
            hidden_states = dispatch_attention_fn(
                query, key, value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class Flux2Attention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = Flux2AttnProcessor
    _available_processors = [Flux2AttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias
        self.fused_projections = False

        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by "
                f"{self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Flux2ParallelSelfAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attn_segments: Optional[Tuple[int, int, int]] = None,
        txt_len: Optional[int] = None,
        layer_idx: Optional[int] = None,
        is_single_block: bool = True,
        id_patch_config: Optional["IdPatchConfig"] = None,
        id_patch_pairs: Optional[List[Dict[str, Tuple[int, int, int, int]]]] = None,
        latent_h: Optional[int] = None,
        latent_w: Optional[int] = None,
        rope_theta: Optional[int] = None,
        rope_axes_dim: Optional[Tuple[int, ...]] = None,
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

        # 保存 PRE-RoPE 的 q/k/v（Version B 用）
        q_pre, k_pre, v_pre = query, key, value

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # ============ ID Patch Attention 逻辑 ============
        use_id_patch = (
            id_patch_config is not None
            and id_patch_pairs is not None
            and attn_segments is not None
            and txt_len is not None
            and latent_h is not None
            and latent_w is not None
        )

        if use_id_patch and layer_idx is not None:
            active_layers = id_patch_config.idx_single_window
            if layer_idx not in active_layers:
                use_id_patch = False

        # ============ 路由 ============
        if use_id_patch:
            sn, sl, sr = attn_segments
            st = txt_len
            ranges = _compute_segment_ranges(st, sn, sl, sr, img_first=False)
            roi_mode = getattr(id_patch_config, "roi_mode", False)
            ref_reencode = getattr(id_patch_config, "roi_ref_reencode", False)

            attn_output = _id_patch_attention(
                query, key, value,
                ranges=ranges,
                id_patch_pairs=id_patch_pairs,
                latent_h=latent_h,
                latent_w=latent_w,
                expand_ratio_lq=getattr(id_patch_config, "expand_ratio_lq", 1.0),
                expand_ratio_ref=getattr(id_patch_config, "expand_ratio_ref", 1.0),
                expand_min_size=getattr(id_patch_config, "expand_min_size", 0),
                fixup_lqref=getattr(id_patch_config, "fixup_lqref", True),
                fixup_noise=getattr(id_patch_config, "fixup_noise", False) and not roi_mode and not ref_reencode,
                noise_alpha=getattr(id_patch_config, "noise_alpha", 0.5),
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

            if ref_reencode:
                attn_output = _ref_hr_attention(
                    attn_output, query, key, value, ranges, id_patch_pairs, latent_h, latent_w,
                    exp_lq=getattr(id_patch_config, "expand_ratio_lq", 1.0),
                    exp_min=getattr(id_patch_config, "expand_min_size", 0),
                    noise_alpha=getattr(id_patch_config, "noise_alpha", 0.5),
                    backend=self._attention_backend, parallel_config=self._parallel_config,
                )
            elif roi_mode and not getattr(id_patch_config, "roi_persist", False):
                attn_output = _maybe_virtual_roi(
                    attn_output, q_pre, k_pre, v_pre, ranges, id_patch_pairs,
                    latent_h, latent_w, id_patch_config, rope_theta, rope_axes_dim,
                    self._attention_backend, self._parallel_config,
                )
        else:
            attn_output = dispatch_attention_fn(
                query, key, value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )

        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


class Flux2ParallelSelfAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = Flux2ParallelSelfAttnProcessor
    _available_processors = [Flux2ParallelSelfAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        self.to_qkv_mlp_proj = torch.nn.Linear(
            self.query_dim, self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=bias
        )
        self.mlp_act_fn = Flux2SwiGLU()

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_out = torch.nn.Linear(self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by "
                f"{self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
            processor=Flux2ParallelSelfAttnProcessor(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        split_hidden_states: bool = False,
        text_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        else:
            return hidden_states


class Flux2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            processor=Flux2AttnProcessor(),
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_params_img: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        temb_mod_params_txt: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs

        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class Flux2PosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
        )

        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
            )
        else:
            self.guidance_embedder = None

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))

        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))
            time_guidance_emb = timesteps_emb + guidance_emb
            return time_guidance_emb
        else:
            return timesteps_emb


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets

        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        mod = self.act_fn(temb)
        mod = self.linear(mod)

        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


class Flux2Transformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "img_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        seq_noise: Optional[int] = None,
        seq_lq: Optional[int] = None,
        seq_ref: Optional[int] = None,
        latent_h: Optional[int] = None,
        latent_w: Optional[int] = None,
        id_patch_config: Optional["IdPatchConfig"] = None,
        id_patch_pairs: Optional[List[Dict[str, Tuple[int, int, int, int]]]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            joint_attention_kwargs = {}
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # ============ 限制处理的人脸数（P=64 等大开销诊断用）============
        if id_patch_config is not None and id_patch_pairs:
            _mf = getattr(id_patch_config, "roi_max_faces", -1)
            if _mf is not None:
                _mf = _as_int(_mf)   # 防御：yaml 写成 [1]/(1,) 等
                if _mf >= 0 and len(id_patch_pairs) > _mf:
                    if _ROI_DEBUG:
                        print(f"[roi] roi_max_faces={_mf}: 用前 {_mf}/{len(id_patch_pairs)} 张脸", flush=True)
                    id_patch_pairs = id_patch_pairs[:_mf]

        # ============ Version B-1.5 persist：循环前在 image 尾部追加高密度脸影子 token ============
        persist_state = None
        if (id_patch_config is not None and id_patch_pairs and seq_noise is not None
                and latent_h is not None
                and getattr(id_patch_config, "roi_mode", False)
                and getattr(id_patch_config, "roi_persist", False)):
            hidden_states, _new_rope, persist_state = _persist_append(
                hidden_states, img_ids, self.pos_embed, text_rotary_emb,
                id_patch_config, id_patch_pairs, latent_h, latent_w,
                seq_noise, (seq_lq or 0),
            )
            if persist_state is not None:
                concat_rotary_emb = _new_rope

        # ============ 构建 ID Patch 参数 ============
        id_patch_attention_kwargs = {}
        if id_patch_config is not None and id_patch_pairs is not None and seq_noise is not None and latent_h is not None:
            sn = seq_noise
            sl = seq_lq or 0
            sr = seq_ref or 0
            id_patch_attention_kwargs = {
                "id_patch_config": id_patch_config,
                "id_patch_pairs": id_patch_pairs,
                "attn_segments": (sn, sl, sr),
                "txt_len": num_txt_tokens,
                "latent_h": latent_h,
                "latent_w": latent_w,
                # Version B 虚拟 token 重配 RoPE 所需
                "rope_theta": self.pos_embed.theta,
                "rope_axes_dim": self.pos_embed.axes_dim,
            }

        # Double Stream Transformer Blocks
        for index_block, block in enumerate(self.transformer_blocks):
            block_attention_kwargs = joint_attention_kwargs.copy()

            if id_patch_attention_kwargs:
                block_attention_kwargs.update(id_patch_attention_kwargs)
                block_attention_kwargs["layer_idx"] = index_block
                block_attention_kwargs["is_single_block"] = False

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    block_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_params_img=double_stream_mod_img,
                    temb_mod_params_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                )

        # Single Stream Transformer Blocks
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            block_attention_kwargs = joint_attention_kwargs.copy()

            if id_patch_attention_kwargs:
                block_attention_kwargs.update(id_patch_attention_kwargs)
                block_attention_kwargs["layer_idx"] = index_block
                block_attention_kwargs["is_single_block"] = True

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    block_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod_params=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                )

        # ============ Version B-1.5 persist：循环后把脸影子降采样回写进 noise 脸并裁掉尾部 ============
        if persist_state is not None:
            hidden_states = _persist_collapse(
                hidden_states, persist_state, num_txt_tokens,
                getattr(id_patch_config, "noise_alpha", 0.5),
            )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
