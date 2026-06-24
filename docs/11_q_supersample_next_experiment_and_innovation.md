# 11 · 下一步：Q-only Supersampling ROI Attention + 创新性分析

> 目标：继续探索用户关心的路线——**不通过 crop ref concat / ref 像素重编码来提升清晰度，而是在原始 attention 内部做 ROI upscale**。本页基于 `docs/05/09/10` 的负结果，提出下一版最小可测实现。

---

## 1. 为什么不继续沿用 B 的插值式 ROI upscale

`docs/09` 已经说明，B-1 / B-1.5 的核心失败不是简单 bug，而是机制问题：

1. **插值上采不产生新高频**：native 小脸只有约 8–12 token，bilinear 到 P×P 只是低通放大。
2. **再下采回 native 会二次低通**：`上采 → attention → 下采` 的回写路径会把原本已有的锐度冲掉。
3. **persist 影子重复 token OOD**：同位置 native token + shadow token 同时在全注意力里演化，模型训练时没见过。
4. **单趟无后处理存在 native 解码上限**：最终仍从 ~8–12 token 解码，小脸上限约 128–192 px 的 crisp，而不可能接近 crop→1k 的完整高分辨率轨迹。

因此，下一步不再做“插值 token upscale”，而是把 upscale 从 **latent 内容空间** 转移到 **attention 查询/匹配空间**。

---

## 2. 新方案：Q-only Supersampling ROI Attention

核心思想：

> 不插值 Q/K/V 的内容，不生成 P×P 虚拟 latent，也不把虚拟 latent 下采样回 native。  
> 只把每个 native noise face query 复制成 `m×m` 个 **sub-query probes**，给它们不同的子 token RoPE 位置，让它们以更细粒度去匹配 native lq/ref K/V，然后把 attention output 聚合回原 native token。

数据流：

```text
native noise face q_i
  ├─ repeat 成 m×m 个 sub-query：内容向量相同，RoPE 子位置不同
  ├─ branch 1: attend native lq ROI K/V（结构 / 姿态 / 表情）
  ├─ branch 2: attend native ref ROI K/V（身份 / 纹理 / 细节）
  ├─ split-branch 融合，避免 lq/ref 在同一个 softmax 里互相稀释
  └─ mean/center 聚合回 q_i 的一个 native residual，按 noise_alpha 写回
```

这和旧 B 的关键差异：

| 设计点 | 旧 B：Virtual ROI-QKV | 新 B-2：Q-only supersampling |
|---|---|---|
| 是否插值 latent 内容 | 是，ROIAlign Q/K/V 到 P×P | 否，只 repeat q 内容 |
| 是否下采样虚拟 latent output | 是，P×P → native | 否，只聚合 sub-query attention output |
| 高频来源 | 插值后的 native token，本质无新高频 | native ref token 的原始 K/V + 更细 attention 匹配 |
| 主要风险 | 低通变糊、重复 token OOD | 效果可能弱，但不应因插值变糊 |
| 实验意义 | 已证伪插值式 ROI | 验证 attention-logit/query 分辨率是否有收益 |

---

## 3. 已推送的代码入口

新增 patch helper：

```bash
python tools/apply_roi_qsupersample_patch.py
```

它会修改：

- `code/transformer_flux2.py`
  - 新增 `roi_variant='q_supersample'` 路由；
  - 新增 `_q_supersample_roi_attention`；
  - 新增 `_make_subquery_pos_ids` / `_make_rect_pos_ids`；
  - 修正虚拟 token PE 端点：bbox 是半开区间 `[y1,y2)`，使用 token-center 坐标 `y1+0.5 → y2-0.5`；
  - 保留旧 B：`roi_variant='interpolate'` 时仍走原 `_virtual_roi_qkv_attention`。
- `code/refine_model.py`
  - `IdPatchConfig` 增加新字段；
  - 构造函数透传新字段；
  - `_ref_hr_pos_ids` 改为 token-center 坐标。
- `code/Dit_pipeline.py`
  - `load_modules()` 透传新字段。

新增字段：

```yaml
id_patch_roi_variant: "q_supersample"  # 新实验；旧 B 为 "interpolate"
id_patch_roi_subsample: 2              # m，每个 native token 生成 m×m 个子查询；先扫 {2,4}
id_patch_roi_agg_mode: "mean"          # mean / center
id_patch_roi_split_branches: true       # lq/ref 分支分开 softmax
id_patch_roi_detail_beta: 0.5           # split 下 ref 分支权重；扫 {0.3,0.5,0.7,1.0}
```

---

## 4. 推荐实验配置

先用单脸、少层、debug 模式确认机制：

```yaml
Dit:
  use_id_patch_attention: true
  patch_split_num: 1

  # 合法层，double 只有 0~7
  id_patch_idx_double_window: [1, 3, 5, 7]
  id_patch_idx_single_window: [1, 3]

  # 关闭旧路径，隔离 B-2
  id_patch_fixup_lqref: false
  id_patch_fixup_noise: false
  id_patch_roi_ref_reencode: false
  id_patch_roi_persist: false

  # 打开 ROI mode，但 variant 切到 q_supersample
  id_patch_roi_mode: true
  id_patch_roi_variant: "q_supersample"
  id_patch_roi_max_faces: 1

  # B-2 主参数
  id_patch_roi_subsample: 2
  id_patch_roi_pe_mode: "pe3"           # ref→target 坐标映射，先用 pe3
  id_patch_roi_include_lq: true
  id_patch_roi_split_branches: true
  id_patch_roi_detail_beta: 0.5
  id_patch_roi_agg_mode: "mean"
  id_patch_noise_alpha: 0.5

  # ROI 范围
  id_patch_expand_ratio_ref: 2.0
  id_patch_expand_ratio_lq: 1.5
  id_patch_expand_min_size: 0
```

首跑：

```bash
python tools/apply_roi_qsupersample_patch.py
python -m py_compile code/transformer_flux2.py code/refine_model.py code/Dit_pipeline.py
ROI_DEBUG=1 <your-run-command>
```

看日志是否出现：

```text
[roi-qsub] m=2 agg=mean split=True beta=0.50 noise=... q_sub=... ref_k=...
```

---

## 5. 实验矩阵

严格沿用 `docs/05` 的检验纪律：先跑 baseline×2 测噪声底，再看脸 crop，不要只看整图 pixel diff。

| Phase | 变量 | 取值 | 目的 |
|---|---|---|---|
| Q0 | baseline noise floor | 同配置两次 | 确认 NPU 噪声底 |
| Q1 | `roi_subsample` | {2, 4} | 子查询密度是否有效 |
| Q2 | `roi_detail_beta` | {0.3,0.5,0.7,1.0} | lq 结构 vs ref 细节平衡 |
| Q3 | `roi_split_branches` | true / false | 分支 softmax 是否比 concat KV 更稳 |
| Q4 | `roi_pe_mode` | pe1 / pe3 | ref absolute vs ref→target 坐标 |
| Q5 | layer set | 少层 / 中后层 / 全 active | 清晰度与结构破坏权衡 |

指标：

1. face crop 视觉对比；
2. 输出脸 vs ref 脸 ArcFace 相似度；
3. face crop Laplacian / sharpness；
4. outside face 区域是否退化；
5. 是否串脸 / 过度像 ref 表情。

---

## 6. 预期结果与决策

### 若 Q-only supersampling 有轻微但稳定提升
说明“attention 匹配分辨率”确实是一个独立旋钮，可以继续发展成：

- CRPA-style query-stride RoPE；
- query-adaptive ref patch selection；
- head/layer-wise gating；
- 少量 LoRA/adapter 训练，让模型学会利用 sub-query probes。

### 若仍无提升但不变糊
说明它比旧 B 更安全，但 native ref token 的细节容量 / 输出 native token 上限仍是瓶颈。此时 Q-only 可作为负结果，进一步证明：**不增加真高频源、不改变输出网格的 training-free attention 操作，上限很低。**

### 若变糊或结构坏
优先检查：

1. `noise_alpha` 是否过大，先降到 0.2/0.3；
2. `detail_beta=1.0` 是否过度 ref 化；
3. `pe3` 是否错配，改 `pe1`；
4. active 层是否太早，先只放中后层。

---

## 7. 创新性分析

### 7.1 和已有 region-adaptive / foveated 方法的区别

已有方法大多在 **token 数/分辨率本身** 上做 mixed-resolution：

- RALU 通过 region-adaptive latent upsampling 在不同 denoising 阶段切换局部 full-res 区域，并需要 noise-timestep rescheduling 稳定分辨率转换。
- Foveated Diffusion 根据 foveal mask 非均匀分配 token density，本质是混合分辨率 token 生成。
- HierEdit 走 region-aware hierarchical refinement，用局部窗口模型 refine 高分区域。

本方案不同：它不改变最终 token 网格，不做输出后处理，也不把 ROI 作为新高分图 crop 后单独跑；它只在 attention 内部增加 **query probe density**，用子查询的 RoPE 相位变化来提高匹配细粒度。这更接近“attention 内 foveation”，而不是“latent/image resolution foveation”。

### 7.2 和 CRPA / mixed-resolution RoPE 的关系

CRPA 指出 mixed-resolution DiT 的关键问题是 RoPE phase aliasing：不同分辨率网格混在一个 attention 里时，线性坐标插值会让同一物理距离对应不同相位增量，造成 blur/artifact。

Q-only supersampling 避免了旧 B 最危险的部分：不插值 V、不产生 P×P latent、不再下采样 latent output。它只让 Q 以子 token 位置去探测 native K/V；这为后续实现 CRPA-style query-stride RoPE 留出了清晰接口。

### 7.3 对图像编辑 / 合影超分的潜在贡献点

如果实验成立，创新点可以概括为：

> **Training-free, query-foveated attention for identity-preserving multi-reference face refinement.**  
> 在不裁剪重编码参考图、不改变输出分辨率、不训练模型的条件下，通过 ROI 内子查询采样与分支式 lq/ref attention，提高小脸 ID 细节迁移的匹配精度。

它的研究价值不在于一定超过 crop→1k，而在于清楚回答一个关键问题：

> 在固定 native 输出网格下，清晰度提升到底受限于“attention 匹配不够细”，还是受限于“没有真高频源 / 输出 token 容量不够”？

这个问题本身有论文价值，因为它能把多参考编辑中的 ROI attention、mixed-resolution DiT、foveated generation、training-free inference surgery 连接起来。

---

## 8. 本次 patch 的边界

- patch helper 已推送，但我没有在 NPU 环境实际运行；必须先 `py_compile` + `ROI_DEBUG=1` 首跑。
- B-2 是下一步验证代码，不保证一定提升清晰度；它的价值是避免已证伪的插值低通路径。
- 若目标是直接出效果，`docs/10` 的 ref_hr 真高清重编码更可能有效；若目标是坚持“不 crop ref concat”的研究问题，B-2 更适合作为下一步实验。
