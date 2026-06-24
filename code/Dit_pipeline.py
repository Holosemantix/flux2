import inspect
import math
import os
import time
from typing import Optional, Union, List
import torch.distributed as dist

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from tqdm import tqdm

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from algorithms.pangu_i2i.refiner.model_interface.refine_model import RefinerModel
from algorithms.parallel.patch_utils import ImageSpliter
from algorithms.parallel.communications import gather_patches_result
from .klein_utils import KleinVAEProcessor, KleinLatentProcessor, retrieve_timesteps, compute_empirical_mu

try:
    from algorithms.pangu_i2i.refiner.archs.common_model_ckpt_dict import g_model_name_map
except ImportError:
    g_model_name_map = {}


class DitPipe:
    def __init__(self, cfg):
        self.cfg = cfg

        self.refiner_overlap = self.cfg.get('refiner_overlap', 8)
        self.encoder_overlap = self.cfg.get('encoder_overlap', 64)
        self.decoder_overlap = self.cfg.get('decoder_overlap', 8)
        self.patch_split_num = self.cfg.get('patch_split_num', 2)
        assert dist.get_world_size() == math.pow(self.patch_split_num, 2),\
            (f"patch_split_num * patch_split_num must be equal to world_size, patch_split_num is {self.patch_split_num},"
             f" but world_size is {dist.get_world_size()}.")

        model_path = cfg.get('model_path', '')
        self.dtype = cfg.get('dtype', torch.bfloat16)
        self.rank = dist.get_rank()
        self.device = cfg['device']

        self.flux_model_path = os.path.join(model_path, cfg['flux_model_path'])
        self.transformer_path = os.path.join(model_path, cfg['transformer_path'])
        self.refiner_convrot_quant = cfg['refiner_convrot_quant']
        self.transformer_quant_path = os.path.join(model_path, cfg['transformer_quant_path'])
        self.cfg_scale = cfg.get('cfg_scale', 1.0)
        self.vae = None
        self.transformer = None
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None
        self.neg_prompt_embeds = None
        self.neg_pooled_prompt_embeds = None
        self.neg_text_ids = None
        self.use_cfg = False
        self.noise_scheduler = None
        self.transformer_pipe = None

        self.transformer = None
        self.split_patch_tool: Union[ImageSpliter, None] = None

        self.num_inference_steps = self.cfg.get('step', 28)
        self.sigmas = np.linspace(1.0, 1 / self.num_inference_steps, self.num_inference_steps)

        self.load_modules()
        self.emb_dict = self.load_text_embs()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels))

        # ============ ID Patch Attention 配置 ============
        self.use_id_patch_attention = cfg.get('use_id_patch_attention', False)
        if self.use_id_patch_attention:
            assert self.patch_split_num == 1, (
                f"id_patch_attention only works in no_split mode (patch_split_num=1), "
                f"but got patch_split_num={self.patch_split_num}"
            )

        self.seed = cfg.get('seed', 3407)
        self.true_cfg_scale = cfg.get('true_cfg_scale', 1.0)
        self.colorfix_model = g_model_name_map.get(cfg.get('colorfix_type', 'WaveletRecon'), None)(device=self.device)
        self.colorfix_threshold = cfg.get('colorfix_threshold', 0.6)

    def create_timesteps(self, latents):
        image_seq_len = latents.shape[1]

        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.noise_scheduler,
            self.num_inference_steps,
            self.cfg['device'],
            sigmas=self.sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    def load_modules(self):
        from diffusers.models import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(
            self.flux_model_path,
            subfolder="vae",
        )
        self.vae = vae.to(self.device).to(self.dtype)

        dit_params = {}
        dit_params['flux_model_path'] = self.flux_model_path
        dit_params['transformer_path'] = self.transformer_path
        dit_params['dtype'] = self.dtype
        dit_params['ckpt_enable'] = True
        dit_params['model_name'] = 'transformer'
        dit_params['input_shapes'] = []
        dit_params['device'] = self.device
        dit_params['refiner_convrot_quant'] = self.refiner_convrot_quant
        dit_params['transformer_quant_path'] = self.transformer_quant_path

        # 注意力优化策略传参（互斥）
        self.use_id_patch = self.cfg.get('use_id_patch_attention', False)
        if self.use_id_patch:
            dit_params['use_quest'] = False
            dit_params['use_id_patch_attention'] = True

            dit_params['id_patch_idx_single_window'] = self.cfg.get('id_patch_idx_single_window', [])
            dit_params['id_patch_idx_double_window'] = self.cfg.get('id_patch_idx_double_window', [])

            dit_params['id_patch_expand_ratio_lq'] = self.cfg.get('id_patch_expand_ratio_lq', 1.0)
            dit_params['id_patch_expand_ratio_ref'] = self.cfg.get('id_patch_expand_ratio_ref', 1.0)
            dit_params['id_patch_expand_min_size'] = self.cfg.get('id_patch_expand_min_size', 0)
            dit_params['id_patch_fixup_lqref'] = self.cfg.get('id_patch_fixup_lqref', True)
            dit_params['id_patch_fixup_noise'] = self.cfg.get('id_patch_fixup_noise', False)
            dit_params['id_patch_noise_alpha'] = self.cfg.get('id_patch_noise_alpha', 0.5)

            # ===== Version B（Virtual ROI-QKV / persist）=====
            # 注意：以下每行末尾【不能】有逗号，否则值会变成单元素 tuple（之前的 bug）。
            dit_params['id_patch_roi_mode'] = self.cfg.get('id_patch_roi_mode', False)
            dit_params['id_patch_roi_size'] = self.cfg.get('id_patch_roi_size', 24)
            dit_params['id_patch_roi_pe_mode'] = self.cfg.get('id_patch_roi_pe_mode', 'pe2')
            dit_params['id_patch_roi_include_lq'] = self.cfg.get('id_patch_roi_include_lq', True)
            dit_params['id_patch_roi_persist'] = self.cfg.get('id_patch_roi_persist', False)
            dit_params['id_patch_roi_max_faces'] = self.cfg.get('id_patch_roi_max_faces', -1)
            dit_params['id_patch_roi_up_layer'] = self.cfg.get('id_patch_roi_up_layer', -1)
            dit_params['id_patch_roi_down_layer'] = self.cfg.get('id_patch_roi_down_layer', -1)

            # ===== Version A（真·高清 ref 重编码）=====
            dit_params['id_patch_roi_ref_reencode'] = self.cfg.get('id_patch_roi_ref_reencode', False)
            dit_params['id_patch_ref_crop_size'] = self.cfg.get('id_patch_ref_crop_size', 512)

        # 不需要 else，RefinerModel 默认 use_quest=True

        self.transformer = RefinerModel(**dit_params)  # Refiner DiT初始化

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.flux_model_path, subfolder="scheduler"
        )

        # ============ 加载 ID 匹配模型（YOLO + ResNet）============
        if self.use_id_patch:
            self._load_id_match_models()

    def load_text_embs(self):
        emb_dir = os.path.join(self.flux_model_path, 'ref_tag')
        emb_dict = {}
        task_emb_pair = {'general': 'task_general.pth',
                         'negative': 'task_general_cfg.pth'}
        for task_name, emb_name in task_emb_pair.items():
            emb_path = os.path.join(emb_dir, emb_name)
            if not os.path.exists(emb_path):
                continue
            prompt_emb_dict = torch.load(emb_path, map_location='cpu')
            cur_dict = {}
            for k, emb in prompt_emb_dict.items():
                cur_dict[k] = emb.to(self.dtype).to(self.device)
            emb_dict[task_name] = cur_dict
        if self.cfg_scale > 1.0 and os.path.exists(os.path.join(emb_dir, task_emb_pair.get('negative'))):
            self.use_cfg = True
        return emb_dict

    # ============ ID Patch Matching 方法 ============

    def _load_id_match_models(self):
        """加载 YOLO 检测模型和 ResNet ReID 模型。"""
        from ultralytics import YOLO
        from torchvision import models, transforms

        model_path = self.cfg.get('model_path', '')

        yolo_path = self.cfg.get('yolo_model_path', '')
        resnet_path = self.cfg.get('resnet_model_path', '')

        if self.rank == 0:
            print(f"\n---Loading YOLO model from: {yolo_path}")
        self.yolo_model = YOLO(yolo_path)

        if self.rank == 0:
            print(f"---Loading ResNet18 ReID model from: {resnet_path}")
        resnet = models.resnet18(weights=None)
        state_dict = torch.load(resnet_path, map_location='cpu')
        resnet.load_state_dict(state_dict)
        resnet.fc = torch.nn.Identity()
        self.reid_model = resnet.eval().to(self.device)

        self.reid_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.id_match_conf = self.cfg.get('id_match_conf_threshold', 0.15)
        self.id_match_dist = self.cfg.get('id_match_dist_threshold', 0.4)
        self.id_match_imgsz = self.cfg.get('id_match_imgsz', 3072)

        if self.rank == 0:
            print(f"---ID match models loaded. conf={self.id_match_conf}, "
                  f"dist_threshold={self.id_match_dist}, imgsz={self.id_match_imgsz}\n")

    @staticmethod
    def _tensor_to_pil(tensor):
        """将 [B, C, H, W] tensor（范围 -1~1）转为 PIL Image（取 batch 第一张）。"""
        from PIL import Image
        img = tensor[0].detach().cpu().float()
        img = (img + 1.0) / 2.0
        img = img.clamp(0, 1)
        img = (img * 255).to(torch.uint8)
        img = img.permute(1, 2, 0).numpy()
        return Image.fromarray(img, 'RGB')

    def _extract_reid_features(self, img_pil, bboxes):
        """提取图像中所有 bounding box 区域的 ReID 特征向量。"""
        import numpy as np
        features = []
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            crop = img_pil.crop((x1, y1, x2, y2))
            tensor = self.reid_transforms(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.reid_model(tensor)
            features.append(feat.cpu().numpy().flatten())
        return np.array(features)

    def _match_ids(self, image_lq, image_ref):
        """
        对 lq 和 ref 图像进行人物 ID 检测与匹配。

        Returns:
            id_patch_pairs: list of {'lq': (y1,x1,y2,x2), 'ref': (y1,x1,y2,x2)}，latent 坐标
            id_patch_pairs_pixel: list of {'lq': (x1,y1,x2,y2), 'ref': (x1,y1,x2,y2), 'dist': float}，像素坐标（xyxy 格式）
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        pil_lq = self._tensor_to_pil(image_lq)
        pil_ref = self._tensor_to_pil(image_ref)

        res_lq = self.yolo_model(
            pil_lq, classes=[0], imgsz=self.id_match_imgsz,
            conf=self.id_match_conf, verbose=False,
            # device=self.device,
            device='cpu',
        )
        res_ref = self.yolo_model(
            pil_ref, classes=[0], imgsz=self.id_match_imgsz,
            conf=self.id_match_conf, verbose=False,
            # device=self.device,
            device='cpu',
        )

        bboxes_lq = res_lq[0].boxes.xyxy.cpu().numpy()
        bboxes_ref = res_ref[0].boxes.xyxy.cpu().numpy()

        if len(bboxes_lq) == 0 or len(bboxes_ref) == 0:
            if self.rank == 0:
                print(f"---ID match: no detections (lq={len(bboxes_lq)}, ref={len(bboxes_ref)}), skipping")
            return [], []

        feat_lq = self._extract_reid_features(pil_lq, bboxes_lq)
        feat_ref = self._extract_reid_features(pil_ref, bboxes_ref)

        dist_matrix = cdist(feat_lq, feat_ref, metric='cosine')
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        scale = self.vae_scale_factor
        pixel_h, pixel_w = image_lq.shape[2], image_lq.shape[3]
        latent_h = pixel_h // scale
        latent_w = pixel_w // scale

        id_patch_pairs = []
        id_patch_pairs_pixel = []

        for idx_lq, idx_ref in zip(row_ind, col_ind):
            dist = dist_matrix[idx_lq, idx_ref]
            if dist > self.id_match_dist:
                continue

            lq_box = bboxes_lq[idx_lq]   # (x1, y1, x2, y2) 像素空间
            ref_box = bboxes_ref[idx_ref]

            # latent 坐标 (y1, x1, y2, x2)
            lq_latent = (
                max(0, int(lq_box[1]) // scale),
                max(0, int(lq_box[0]) // scale),
                min(latent_h, (int(lq_box[3]) + scale - 1) // scale),
                min(latent_w, (int(lq_box[2]) + scale - 1) // scale),
            )
            ref_latent = (
                max(0, int(ref_box[1]) // scale),
                max(0, int(ref_box[0]) // scale),
                min(latent_h, (int(ref_box[3]) + scale - 1) // scale),
                min(latent_w, (int(ref_box[2]) + scale - 1) // scale),
            )

            if lq_latent[0] >= lq_latent[2] or lq_latent[1] >= lq_latent[3]:
                continue
            if ref_latent[0] >= ref_latent[2] or ref_latent[1] >= ref_latent[3]:
                continue

            id_patch_pairs.append({
                'lq': lq_latent,
                'ref': ref_latent,
            })

            # 像素坐标保持 xyxy 格式，附带匹配距离
            id_patch_pairs_pixel.append({
                'lq': (int(lq_box[0]), int(lq_box[1]), int(lq_box[2]), int(lq_box[3])),
                'ref': (int(ref_box[0]), int(ref_box[1]), int(ref_box[2]), int(ref_box[3])),
                'dist': float(dist),
            })

        return id_patch_pairs, id_patch_pairs_pixel

    def _encode_ref_hr(self, ref_pixel, id_patch_pairs_pixel):
        """[Version A] 把（受 roi_max_faces 限制的）每个匹配 ID 的 ref 脸从像素空间 crop 出来、
        resize 到 ref_crop_size、VAE 编码成真·高清 latent。返回 list（与前 N 个 pair 对齐）。"""
        S = int(self.cfg.get('id_patch_ref_crop_size', 512))
        S = max(16, (S // 16) * 16)
        mf = self.cfg.get('id_patch_roi_max_faces', -1)
        pairs_px = id_patch_pairs_pixel if (mf is None or int(mf) < 0) else id_patch_pairs_pixel[:int(mf)]
        er = float(self.cfg.get('id_patch_expand_ratio_ref', 1.0))

        ref_dev = ref_pixel.to(self.device).to(self.dtype)
        _, _, Hpx, Wpx = ref_dev.shape
        ref_hr_latents = []
        for pp in pairs_px:
            x1, y1, x2, y2 = pp['ref']
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = (x2 - x1) * er, (y2 - y1) * er
            ex1 = max(0, int(cx - bw / 2)); ey1 = max(0, int(cy - bh / 2))
            ex2 = min(Wpx, int(cx + bw / 2)); ey2 = min(Hpx, int(cy + bh / 2))
            if ex2 - ex1 < 2 or ey2 - ey1 < 2:
                ref_hr_latents.append(None)
                continue
            crop = ref_dev[:, :, ey1:ey2, ex1:ex2].float()
            crop = F.interpolate(crop, size=(S, S), mode='bicubic', align_corners=False).clamp(-1, 1).to(self.dtype)
            ref_hr_latents.append(KleinVAEProcessor.encode(self.vae, crop))   # [1,128,S//16,S//16]
        if self.rank == 0:
            print(f"---Version A: re-encoded {sum(l is not None for l in ref_hr_latents)} ref crop(s) @ {S}px")
        return ref_hr_latents

    def _save_id_match_debug(self, image_lq, image_ref, id_patch_pairs_pixel, save_dir, prefix=""):
        """
        保存 ID 匹配的 debug 可视化：
          1. 一张 side-by-side 总览图，带彩色 bbox 和 ID 编号
          2. 每组匹配的 lq crop 和 ref crop 单独保存
        """
        from PIL import Image, ImageDraw, ImageFont

        if not id_patch_pairs_pixel:
            return

        os.makedirs(save_dir, exist_ok=True)

        pil_lq = self._tensor_to_pil(image_lq)
        pil_ref = self._tensor_to_pil(image_ref)

        # ---- 颜色列表（最多 20 组，循环使用）----
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
            (128, 255, 0), (0, 255, 128), (64, 224, 208), (255, 99, 71), (138, 43, 226),
            (50, 205, 50), (255, 215, 0), (220, 20, 60), (30, 144, 255), (255, 105, 180),
        ]

        # ---- 尝试加载字体 ----
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 24)
            except Exception:
                font = ImageFont.load_default()

        # ---- 1. 绘制带 bbox 的总览图 ----
        vis_lq = pil_lq.copy()
        vis_ref = pil_ref.copy()
        draw_lq = ImageDraw.Draw(vis_lq)
        draw_ref = ImageDraw.Draw(vis_ref)

        for pair_idx, pair in enumerate(id_patch_pairs_pixel):
            color = colors[pair_idx % len(colors)]
            dist = pair['dist']
            label = f"ID{pair_idx} d={dist:.3f}"

            lq_box = pair['lq']   # (x1, y1, x2, y2)
            ref_box = pair['ref']

            for thickness in range(4):
                draw_lq.rectangle(
                    [lq_box[0] - thickness, lq_box[1] - thickness,
                     lq_box[2] + thickness, lq_box[3] + thickness],
                    outline=color
                )
                draw_ref.rectangle(
                    [ref_box[0] - thickness, ref_box[1] - thickness,
                     ref_box[2] + thickness, ref_box[3] + thickness],
                    outline=color
                )

            draw_lq.text((lq_box[0] + 2, lq_box[1] + 2), label, fill=color, font=font)
            draw_ref.text((ref_box[0] + 2, ref_box[1] + 2), label, fill=color, font=font)

        # 拼接 side-by-side
        w_lq, h_lq = vis_lq.size
        w_ref, h_ref = vis_ref.size
        gap = 20
        total_w = w_lq + gap + w_ref
        total_h = max(h_lq, h_ref)
        canvas = Image.new('RGB', (total_w, total_h), (40, 40, 40))
        canvas.paste(vis_lq, (0, 0))
        canvas.paste(vis_ref, (w_lq + gap, 0))

        overview_path = os.path.join(save_dir, f"{prefix}id_match_overview.jpg")
        canvas.save(overview_path, quality=90)

        # ---- 2. 保存每组匹配的 crop ----
        for pair_idx, pair in enumerate(id_patch_pairs_pixel):
            lq_box = pair['lq']
            ref_box = pair['ref']
            dist = pair['dist']

            lq_crop = pil_lq.crop(lq_box)
            ref_crop = pil_ref.crop(ref_box)

            lq_crop_path = os.path.join(save_dir, f"{prefix}pair{pair_idx}_lq_d{dist:.3f}.jpg")
            ref_crop_path = os.path.join(save_dir, f"{prefix}pair{pair_idx}_ref_d{dist:.3f}.jpg")
            lq_crop.save(lq_crop_path, quality=95)
            ref_crop.save(ref_crop_path, quality=95)

            max_h = max(lq_crop.height, ref_crop.height)
            pair_canvas = Image.new('RGB', (lq_crop.width + 10 + ref_crop.width, max_h), (40, 40, 40))
            pair_canvas.paste(lq_crop, (0, 0))
            pair_canvas.paste(ref_crop, (lq_crop.width + 10, 0))
            pair_path = os.path.join(save_dir, f"{prefix}pair{pair_idx}_compare_d{dist:.3f}.jpg")
            pair_canvas.save(pair_path, quality=95)

        if self.rank == 0:
            print(f"---ID match debug saved to: {save_dir} ({len(id_patch_pairs_pixel)} pairs)")

    def get_prompt_embs(self, task_name='sr'):
        task_embs = self.emb_dict[task_name]
        text_ids = task_embs['text_ids'].to(self.dtype).to(self.device)

        prompt_embeds = task_embs['prompt_embeds']
        if self.use_cfg:
            neg_embs = self.emb_dict['negative']
            neg_prompt_embeds = neg_embs['neg_prompt_embeds']

            prompt_embeds = torch.cat([prompt_embeds, neg_prompt_embeds], dim=0)
        prompt_embeds = prompt_embeds.to(self.dtype).to(self.device)
        return prompt_embeds, text_ids

    def get_input_patches(self, x: torch.Tensor, out_channel: int, scale_factor: Union[int, float] = 1,
                          refiner_overlap: int = 8, align: int = 16):

        b, _, h, w = x.shape

        patch_size_w = math.ceil((w // self.patch_split_num + refiner_overlap // self.patch_split_num) / align) * align
        patch_size_h = math.ceil((h // self.patch_split_num + refiner_overlap // self.patch_split_num) / align) * align
        stride_w = w - patch_size_w
        stride_h = h - patch_size_h

        self.split_patch_tool = ImageSpliter(inp=x, patch_size_w=patch_size_w, patch_size_h=patch_size_h,
                                             stride_w=stride_w, stride_h=stride_h, scale_factor=scale_factor,
                                             out_channel=out_channel, overlap_mode='linear', offload=False, min_rule=True)

        patches, infos = [], []
        for idx in range(self.split_patch_tool.length):
            patch, info = next(self.split_patch_tool)
            patches.append(patch)
            infos.append(info)

        return patches, infos

    def merge_all_patches(self, patches_result: list, infos: list):
        for idx in range(len(patches_result)):
            patch_result, info = patches_result[idx], infos[idx]
            self.split_patch_tool.update(patch_result, info)

        patches_gather = self.split_patch_tool.gather()
        return patches_gather

    def dit_infer(self, image, ref=None, debug_save_dir=None, debug_prefix=""):
        '''
        image: [B, C, H, W], range (-1 ~ 1)
        ref: [B, C, H, W], range (-1 ~ 1), if ref is None, run blind sr mode.
        '''
        task_name = 'sr'
        image = image.to(self.device).to(self.dtype)
        no_split = (self.patch_split_num == 1)

        # ========== ID Matching ==========
        id_patch_pairs = None
        id_patch_pairs_pixel = None
        if self.use_id_patch_attention and ref is not None and no_split:
            ref_for_match = ref.to(self.device).to(self.dtype)
            id_patch_pairs, id_patch_pairs_pixel = self._match_ids(image, ref_for_match)
            if self.rank == 0:
                print(f"---ID matching: found {len(id_patch_pairs)} matched pair(s)")
                for i, pair in enumerate(id_patch_pairs):
                    lq_b = pair['lq']
                    ref_b = pair['ref']
                    pixel_pair = id_patch_pairs_pixel[i]
                    print(f"   Pair {i}: lq(y1={lq_b[0]},x1={lq_b[1]},y2={lq_b[2]},x2={lq_b[3]}), "
                        f"ref(y1={ref_b[0]},x1={ref_b[1]},y2={ref_b[2]},x2={ref_b[3]}), "
                        f"dist={pixel_pair['dist']:.4f}")

            # 保存 debug 可视化
            if debug_save_dir is not None and id_patch_pairs_pixel:
                self._save_id_match_debug(
                    image, ref_for_match, id_patch_pairs_pixel,
                    save_dir=debug_save_dir,
                    prefix=debug_prefix,
                )

        # ========== Version A: 高清 ref 脸重编码（真·高频源）==========
        ref_hr_latents = None
        if (self.use_id_patch_attention and ref is not None and no_split
                and self.cfg.get('id_patch_roi_ref_reencode', False)
                and id_patch_pairs and id_patch_pairs_pixel):
            ref_hr_latents = self._encode_ref_hr(ref, id_patch_pairs_pixel)

        # ========== 以下所有代码保持原样不变（除 input_data 多传 ref_hr_latents）==========
        if no_split:
            model_input_1 = KleinVAEProcessor.encode(self.vae, image)
        else:
            img_patches, img_infos = self.get_input_patches(image, out_channel=self.vae.config['latent_channels'] * 4,
                                                            scale_factor=1/16.0, refiner_overlap=self.encoder_overlap)
            model_input_1 = KleinVAEProcessor.encode(self.vae, img_patches[self.rank])
            patches_result = gather_patches_result(model_input_1, None)
            model_input_1 = self.merge_all_patches(patches_result, img_infos)

        generator = torch.Generator(f"cuda:{self.device}").manual_seed(self.seed)
        latents = torch.randn(model_input_1.shape, generator=generator, dtype=self.dtype, device=self.device)

        if ref is not None:
            ref = ref.to(self.device).to(self.dtype)

            if no_split:
                model_input_2 = KleinVAEProcessor.encode(self.vae, ref)
            else:
                ref_patches, ref_infos = self.get_input_patches(ref, self.vae.config['latent_channels'] * 4,
                                                                1 / 16.0,
                                                                refiner_overlap=self.encoder_overlap)
                model_input_2 = KleinVAEProcessor.encode(self.vae, ref_patches[self.rank])
                patches_result = gather_patches_result(model_input_2, None)
                model_input_2 = self.merge_all_patches(patches_result, ref_infos)

            ref_model_input = torch.concat((model_input_1, model_input_2), dim=1)
            task_name = 'general'
        else:
            ref_model_input = model_input_1

        timesteps, num_inference_steps = self.create_timesteps(latents)
        if self.use_cfg:
            latents = torch.cat([latents, latents], dim=0)
            ref_model_input = torch.cat([ref_model_input, ref_model_input], dim=0)

        prompt_embeds, text_ids = self.get_prompt_embs(task_name)
        guidance = torch.tensor([3.5], device=self.device).expand(latents.shape[0])

        for i, t in enumerate(tqdm(timesteps, desc=f'transformer task-{task_name}')):

            latent_model_input = latents
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            if no_split:
                dit_input = torch.cat((latent_model_input, ref_model_input), dim=1)
                input_data = [dit_input, timestep, guidance, prompt_embeds, text_ids, self.rank]

                if id_patch_pairs is not None:
                    input_data.append(id_patch_pairs)
                    if ref_hr_latents is not None:      # Version A: ref_hr 放在 id_patch_pairs 之后
                        input_data.append(ref_hr_latents)

                torch_npu.npu.synchronize()
                start_time = time.perf_counter()
                noise_pred = self.transformer(input_data)[0].to(latents.device)
                torch_npu.npu.synchronize()
                end_time = time.perf_counter()
                print(f"---Refiner DiT: step {i} cost time is(s): {end_time - start_time}, noise_pred:{noise_pred.shape}")
            else:
                patches, infos = self.get_input_patches(torch.cat((latent_model_input, ref_model_input), dim=1),
                                                        latent_model_input.shape[1],
                                                        refiner_overlap=self.refiner_overlap,
                                                        align=16)
                input_data = [patches[self.rank], timestep, guidance, prompt_embeds, text_ids, self.rank]
                torch_npu.npu.synchronize()
                start_time = time.perf_counter()
                patch_noise_pred = self.transformer(input_data)[0].to(latents.device)
                torch_npu.npu.synchronize()
                end_time = time.perf_counter()
                print(f"---Refiner DiT: step {i} cost time is(s): {end_time - start_time}, patch_noise_pred:{patch_noise_pred.shape}, patches:{patches[self.rank].shape}")
                patches_result = gather_patches_result(patch_noise_pred, None)
                noise_pred = self.merge_all_patches(patches_result, infos)

            latents_dtype = latents.dtype

            if self.use_cfg:
                noise_pred = noise_pred[1:2] + self.cfg_scale * (noise_pred[:1] - noise_pred[1:2])
                latents = self.noise_scheduler.step(noise_pred, t, latents[:1], return_dict=False)[0]
            else:
                latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if self.use_cfg and i < num_inference_steps - 1:
                latents = torch.cat([latents, latents], dim=0)

        if no_split:
            out = KleinVAEProcessor.decode(self.vae, latents)
        else:
            patches, infos = self.get_input_patches(latents, image.shape[1],
                                                    scale_factor=image.shape[-1] / latents.shape[-1],
                                                    refiner_overlap=self.decoder_overlap)
            latents = patches[self.rank]
            out = KleinVAEProcessor.decode(self.vae, latents)
            out_patches_gather = gather_patches_result(out, None)
            out = self.merge_all_patches(out_patches_gather, infos)

        return out

    @torch.no_grad()
    def forward(self, img_lr, img_ref=None, debug_save_dir=None, debug_prefix=""):
        img_lr = img_lr.to(self.device)
        out = self.dit_infer(img_lr, img_ref, debug_save_dir=debug_save_dir, debug_prefix=debug_prefix)

        # color trans
        color_fix_input = torch.cat((out, img_lr), dim=1)
        no_split = (self.patch_split_num == 1)

        if no_split:
            out = self.colorfix_model(color_fix_input)
            out = torch.clip(out * 0.5 + 0.5, 0, 1)
        else:
            patches, infos = self.get_input_patches(color_fix_input, out.shape[1])
            color_fix_input = patches[self.rank]

            out = self.colorfix_model(color_fix_input)
            out = torch.clip(out * 0.5 + 0.5, 0, 1)

            out_patches_gather = gather_patches_result(out, None)
            out = self.merge_all_patches(out_patches_gather, infos)

        return out
