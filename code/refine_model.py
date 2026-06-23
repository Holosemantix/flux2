import os
import sys
import numpy as np
from typing import List
from pathlib import Path
from dataclasses import dataclass, field
import torch
import torch.distributed as dist
from diffusers.models import Flux2Transformer2DModel

from algorithms.pangu_i2i.refiner.model_pipeline.klein_utils import KleinLatentProcessor
import time


@dataclass
class QuestConfig:
    """Quest attention 算法的配置

    2D Block 划分说明：
    - block_size: 在 latent space 中，每个 block 覆盖 block_size x block_size 的 patch
    - query_block_size: 在 latent space 中，每个 query block 覆盖 query_block_size x query_block_size 的 patch

    例如：block_size=8 表示每个 block 包含 8x8=64 个 tokens
    """
    block_size: int = 8  # 2D block 边长（不是 token 数量）
    top_k: int = 8  # 选择的 2D block 数量
    query_block_size: int = 8  # 2D query block 边长（之前叫 query_chunk_size，现在是 2D 的）
    noise_block: bool = True
    lq_block: bool = True
    ref_block: bool = True
    share_across_heads: bool = True
    chunk_gather_size: int = 4
    idx_block_single: List[int] = field(default_factory=list)
    idx_block_double: List[int] = field(default_factory=list)
    idx_single_window: List[int] = field(default_factory=list)
    idx_double_window: List[int] = field(default_factory=list)


@dataclass
class IdPatchConfig:
    """ID Patch Attention 的配置。

    在指定层中，lq/ref 的 ID 区域只与对应的 ref/lq ID 区域做 cross-attention，
    非 ID 区域及其他 segment 交互保持 full attention。
    与 QuestConfig 互斥，不可同时使用。

    Version A（Expanded-KV Only）：
    - expand_ratio_lq:  被 cross-attend 的 lq patch 扩大倍数 (r_t)，结构侧
    - expand_ratio_ref: 被 cross-attend 的 ref patch 扩大倍数 (r_s)，细节侧  ← 主旋钮
    - expand_min_size:  扩大后 bbox 最小边长（token 数），对小脸兜底
    Version A'（Noise-Fixup）：
    - fixup_lqref: 是否保留原始 lq/ref 段 fix-up（默认 True）
    - fixup_noise: 是否对 noise(输出) 段人脸 query 做 fix-up（lq结构+ref细节）
    - noise_alpha: 残差注入强度，建议 0.4~0.6
    全默认值时与原实现完全一致。
    """
    idx_single_window: List[int] = field(default_factory=list)  # single block 中使用 id patch attention 的层
    idx_double_window: List[int] = field(default_factory=list)  # double block 中使用 id patch attention 的层
    expand_ratio_lq: float = 1.0
    expand_ratio_ref: float = 1.0
    expand_min_size: int = 0
    fixup_lqref: bool = True
    fixup_noise: bool = False
    noise_alpha: float = 0.5


class RefinerModel(object):
    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype = torch.float32,
                 **kwargs):
        self.device = device
        self.dtype = dtype
        self.transformer_path = kwargs.get('transformer_path', "")
        self.flux_model_path = kwargs.get('flux_model_path', "")
        self.refiner_convrot_quant = kwargs.get('refiner_convrot_quant', False)
        self.transformer_quant_path = kwargs.get('transformer_quant_path', "")
        self.rank = dist.get_rank()

        # Quest 相关配置
        self.use_quest = kwargs.get('use_quest', False)
        self.quest_config = None
        if self.use_quest:
            self.quest_config = QuestConfig(
                block_size=kwargs.get('quest_block_size', 16), # 2D KV block 边长, 每个block包含block_size X block_size个token
                top_k=kwargs.get('quest_top_k', 8), # 每个 query block 选择 top_k 个 KV block
                query_block_size=kwargs.get('query_block_size', 16), # 每个 query block 的边长，每个query block包含query_block_size X query_block_size个token
                noise_block=kwargs.get('quest_noise_block', True), # noise segment KV 是否参与分块
                lq_block=kwargs.get('quest_lq_block', True), # lq segment KV 是否参与分块
                ref_block=kwargs.get('quest_ref_block', True), # ref segment KV 是否参与分块
                share_across_heads=kwargs.get('share_across_heads', True),
                chunk_gather_size=kwargs.get('chunk_gather_size', 32),
                idx_block_single=kwargs.get('quest_idx_block_single', []), # 单流层full attention 层
                idx_block_double=kwargs.get('quest_idx_block_double', []), # 双流层full attention 层
                # idx_single_window=kwargs.get('quest_idx_single_window', list(np.arange(5))),
                # idx_double_window=kwargs.get('quest_idx_double_window', list(np.arange(20))),
                idx_single_window=kwargs.get('quest_idx_single_window', []), #单流层lq和ref用window attention的层
                idx_double_window=kwargs.get('quest_idx_double_window', []), #双流层lq和ref用window attention的层
            )
            if self.rank == 0:
                print(f"\n---Quest attention enabled with config: {self.quest_config}")
                print(f"   Note: block_size={self.quest_config.block_size} means {self.quest_config.block_size}x{self.quest_config.block_size}={self.quest_config.block_size**2} tokens per block")
                if self.quest_config.idx_single_window or self.quest_config.idx_double_window:
                    print(f"   Window attention for LQ/Ref: single_layers={self.quest_config.idx_single_window}, double_layers={self.quest_config.idx_double_window}")

        if self.rank == 0:
            print(f"\n---self.refiner_convrot_quant = {self.refiner_convrot_quant}")
        if not self.refiner_convrot_quant:
            if self.rank == 0:
                print(f"---load origin Refiner DiT")
            if self.transformer_path and os.path.exists(self.transformer_path):
                if self.rank == 0:
                    print(f"---loading transformer from:{self.transformer_path}\n")
                transformer = Flux2Transformer2DModel.from_pretrained(self.transformer_path)
            else:
                if self.rank == 0:
                    print(f"---loading transformer from:{self.flux_model_path}/transformer\n")
                transformer = Flux2Transformer2DModel.from_pretrained(self.flux_model_path, subfolder="transformer")
        else:
            sys.path.insert(0, str(Path(self.transformer_quant_path).parent))
            from self_quant_kernel import quant_weight, QuantLinearint8, QuantLinearint4

            if self.rank == 0:
                print(f"---load quant Refiner DiT")
                print(f"---self.transformer_quant_path = {self.transformer_quant_path}")
            transformer = torch.load(self.transformer_quant_path, weights_only=False, map_location=torch.device("cpu"))

        self.transformer = transformer.to(self.dtype).to(self.device)
        self.latent_channel = 128
        self.vae_scale_factor = kwargs.get('vae_scale_factor', 8)

        # ============ ID Patch Attention 配置 ============
        self.use_id_patch_attention = kwargs.get('use_id_patch_attention', False)
        self.id_patch_config = None
        if self.use_id_patch_attention:
            self.id_patch_config = IdPatchConfig(
                idx_single_window=kwargs.get('id_patch_idx_single_window', []),
                idx_double_window=kwargs.get('id_patch_idx_double_window', []),
                # ===== Version A (Expanded-KV Only) 旋钮 =====
                expand_ratio_lq=kwargs.get('id_patch_expand_ratio_lq', 1.0),
                expand_ratio_ref=kwargs.get('id_patch_expand_ratio_ref', 1.0),
                expand_min_size=kwargs.get('id_patch_expand_min_size', 0),
                # ===== Version A' (Noise-Fixup) 旋钮 =====
                fixup_lqref=kwargs.get('id_patch_fixup_lqref', True),
                fixup_noise=kwargs.get('id_patch_fixup_noise', False),
                noise_alpha=kwargs.get('id_patch_noise_alpha', 0.5),
            )
            if self.rank == 0:
                print(f"\n---ID Patch Attention enabled with config: {self.id_patch_config}")

    def transfer_torch_data(self, data):
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
        if isinstance(data, list):
            for i, _ in enumerate(data):
                data[i] = self.transfer_torch_data(data[i])
        return data

    def __call__(self, data):
        ori_device = data[0].device

        # 提取 id_patch_pairs（Python 对象，不需要 transfer_torch_data）
        id_patch_pairs = None
        if len(data) > 6 and isinstance(data[6], list):
            id_patch_pairs = data[6]
            data = data[:6]

        data = self.transfer_torch_data(data)
        patch_data, timestep, guidance, prompt_embeds, text_ids, patch_idx = data

        batch_size, patch_channel, height, width = patch_data.shape
        if patch_channel == self.latent_channel * 3:
            use_ref = True
        else:
            use_ref = False

        # ============ 计算 latent 空间的高度和宽度 ============
        latent_h = height
        latent_w = width

        unpacked_latents = patch_data[:, :self.latent_channel]
        packed_noisy_input = KleinLatentProcessor.pack_latents(unpacked_latents).to(self.device)
        latent_image_ids_0 = KleinLatentProcessor.prepare_latent_ids(unpacked_latents)

        model_input_1 = patch_data[:, self.latent_channel:self.latent_channel * 2]
        packed_ref_model_input_1 = KleinLatentProcessor.pack_latents(model_input_1)

        if use_ref:
            model_input_2 = patch_data[:, self.latent_channel * 2:self.latent_channel * 3]
            packed_ref_model_input_2 = KleinLatentProcessor.pack_latents(model_input_2)

            sample_condition_latents = [model_input_1, model_input_2]
            sample_image_ids = KleinLatentProcessor.prepare_image_ids_batch(
                sample_condition_latents,
                scale=10
            )

            latent_image_ids = torch.concat((latent_image_ids_0, sample_image_ids), dim=1).to(self.device)
            packed_ref_model_input = torch.concat((packed_ref_model_input_1, packed_ref_model_input_2), dim=1).to(self.device)
        else:
            sample_condition_latents = [model_input_1,]
            sample_image_ids = KleinLatentProcessor.prepare_image_ids_batch(
                sample_condition_latents,
                scale=10
            )
            latent_image_ids = torch.concat((latent_image_ids_0, sample_image_ids), dim=0).to(self.device)

            packed_ref_model_input = packed_ref_model_input_1.to(self.device)

        latent_model_input = torch.cat([packed_noisy_input, packed_ref_model_input], dim=1)

        # 计算各部分的序列长度
        seq_noise = packed_noisy_input.size(1)
        seq_lq = packed_ref_model_input_1.size(1)
        if use_ref:
            seq_ref = packed_ref_model_input_2.size(1)
        else:
            seq_ref = 0

        # 构建 transformer 调用参数
        transformer_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timestep / 1000,
            "guidance": None,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "return_dict": False,
        }

        # 如果启用 Quest attention，添加相关参数
        if self.use_quest and self.quest_config is not None:
            transformer_kwargs["quest_config"] = self.quest_config
            transformer_kwargs["seq_noise"] = seq_noise
            transformer_kwargs["seq_lq"] = seq_lq
            transformer_kwargs["seq_ref"] = seq_ref
            transformer_kwargs["latent_h"] = latent_h
            transformer_kwargs["latent_w"] = latent_w

        # 如果启用 ID Patch attention，添加相关参数
        if self.use_id_patch_attention and self.id_patch_config is not None:
            transformer_kwargs["id_patch_config"] = self.id_patch_config
            transformer_kwargs["id_patch_pairs"] = id_patch_pairs if id_patch_pairs else []
            transformer_kwargs["seq_noise"] = seq_noise
            transformer_kwargs["seq_lq"] = seq_lq
            transformer_kwargs["seq_ref"] = seq_ref
            transformer_kwargs["latent_h"] = latent_h
            transformer_kwargs["latent_w"] = latent_w

        noise_pred = self.transformer(**transformer_kwargs)[0]

        noise_pred = noise_pred[:, : packed_noisy_input.size(1)]

        noise_pred = KleinLatentProcessor.unpack_latents_with_ids(
            noise_pred, latent_image_ids_0.to(self.device)
        )

        return [noise_pred.to(ori_device), patch_idx]
