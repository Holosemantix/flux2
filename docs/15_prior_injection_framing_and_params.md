# 15 · 从“ROI trick”到“先验知识注入扩散模型”：实验定位与参数建议

> 本文回应新的研究定位：不要只把方法描述成 ROI attention / 小脸 trick，而是把它上升到更本质的问题：**如何把先验知识在推理时更好地注入扩散模型，从而修复和提升生成结果**。这个问题同时适用于编辑、文生图后的局部修复、参考图增强、以及高分辨率生成/解码。

---

## 1. 更本质的研究问题

当前分支的实验不应只解释为：

```text
如何让合影小脸更清晰
```

更通用的问题是：

```text
当扩散模型生成结果局部不足时，如何在不训练或少训练的情况下，把外部先验注入到模型内部的 attention / denoising 过程里？
```

这里的“先验知识”包括：

| 先验类型 | 当前 refiner 场景中的对应 | 更通用场景 |
|---|---|---|
| 区域重要性先验 | 人脸 bbox / ROI | 任意需要修复的局部区域、用户 mask、saliency map |
| 身份先验 | ReID 匹配出的同 ID ref face | subject consistency、角色一致性、产品一致性 |
| 空间对齐先验 | ref/lq/noise ROI canonical coordinate | layout control、pose/landmark/box alignment |
| 结构/细节分离先验 | lq 负责结构，ref 负责身份/细节 | editing 中 structure preservation vs style/detail transfer |
| 置信度先验 | ReID distance、ref 清晰度、attention entropy | 多条件冲突时的可靠性加权 |
| 计算分配先验 | ROI 获得更多 query probes / attention 计算 | foveated generation、局部 refinement |

因此，创新点可以重写为：

> **Prior-injected diffusion inference:** 把区域、身份、空间对齐、结构/细节分离和置信度这些先验转成 attention 内的 query probes、K/V routing、RoPE coordinate maps 和 adaptive fusion gates。

---

## 2. 和 PiD / 高分辨率生成工作的关系

PiD（Pixel diffusion Decoder）说明了另一个方向：高分辨率生成不能只依赖 reconstruction-oriented VAE decoder，最好用更强的 pixel diffusion decoder，把 latent decoding 和 upsampling 统一为一个 generative module。

这对当前工作的启发是：

- PiD 解决的是 **latent → high-res pixels** 的 generative decoding / upsampling；
- 当前 ROI prior injection 解决的是 **denoising transformer 内部如何更好地使用参考和局部先验**；
- 二者互补，不冲突。

可以把高质量修复链路理解成两层：

```text
latent/DiT 层：通过 prior-injected attention 让 native latent 更正确、更像目标 ID
pixel decoder 层：通过 PiD 类 pixel diffusion decoder 把 latent 解码成更丰富的高频像素
```

如果只用 PiD，但 latent 里身份/结构已经错了，pixel decoder 可能会放大错误；如果只做 attention prior injection，但 decoder 仍是普通 VAE，native 小脸高频上限仍然受限。两者结合才是更完整的 high-res refinement 方向。

---

## 3. 当前 debug 的 token 数分析

给定这次 case 的第一张脸：

```text
lq face:  y1=133,x1=144,y2=139,x2=151 → 6×7 = 42 tokens
ref face: y1=132,x1=147,y2=139,x2=153 → 7×6 = 42 tokens
```

当前配置：

```yaml
id_patch_roi_subsample: 2
id_patch_expand_ratio_ref: 2.0
id_patch_expand_ratio_lq: 1.5
```

debug 输出：

```text
[roi-qsub] m=2 ... noise=6x7tok q_sub=(1, 168, 24, 128) ref_k=168
```

解释：

```text
native noise face tokens = 6×7 = 42
q_sub = 42 × m² = 42 × 4 = 168
ref_k ≈ 42 × r_ref² = 42 × 4 = 168
```

所以当前配置只是把 query 探测数和 ref K 数都提高到 168。

对比 crop resize：

| crop size | token 网格（/16） | token 数 |
|---|---:|---:|
| 512 px | 32×32 | 1024 |
| 1024 px | 64×64 | 4096 |

当前 `ref_k=168` 与 crop512 的 1024 token 相差约 6.1×；与 crop1024 的 4096 token 相差约 24.4×。

---

## 4. `expand_ratio_ref=2.0` 够吗？

### 4.1 对当前 6×7 小脸来说，不够接近 crop512

粗略公式：

```text
N_refK ≈ h_ref × w_ref × r_ref²
```

当前 `h_ref×w_ref=42`。

想接近 crop512 的 1024 token，需要：

```text
42 × r_ref² ≈ 1024
r_ref ≈ sqrt(1024/42) ≈ 4.94
```

也就是说，从 token 数量上看，`r_ref≈5` 才接近 crop512 的 K 数量。

但这不等于真的获得 crop512 的高频，因为：

```text
扩大 native bbox = 多拿原图附近的低密度 token
crop→512 = 把脸像素重新编码成 32×32 高密度 token
```

所以大 expand 只能扩大上下文和候选 K 范围，不能创造更高密度的 face token。

### 4.2 推荐 sweep

当前第一轮不建议直接跳到 `r_ref=5`，因为多人合影里脸挨得近，扩大太多会把邻脸/背景带进来。

建议按阶段扫：

```yaml
# Phase R1: 温和扩大
id_patch_expand_ratio_ref: 2.0 / 2.5 / 3.0
id_patch_expand_ratio_lq: 1.5
id_patch_roi_subsample: 2

# Phase R2: 对齐 + 大 ref 范围，只处理单脸
id_patch_roi_max_faces: 1
id_patch_expand_ratio_ref: 3.0 / 4.0
id_patch_expand_ratio_lq: 1.5 / 2.0
id_patch_roi_subsample: 4

# Phase R3: crop512 token 数量对照，只做诊断
id_patch_roi_max_faces: 1
id_patch_expand_ratio_ref: 5.0
id_patch_expand_ratio_lq: 2.0
id_patch_roi_subsample: 4
```

判断标准：

```text
如果 r_ref 从 2→3→4 提升 ArcFace/清晰度，但 r=5 串脸或背景污染，则说明“更多 ref 候选 K 有帮助但需 mask/landmark/segmentation 约束”。
如果 r_ref 增大无帮助，说明瓶颈更可能是真高频源或完整高密度 trajectory，而不是 K 数量。
```

### 4.3 建议加 `expand_min_size`

当前小脸只有 6×7 tokens，ratio 对不同脸大小的实际 token 数差异很大。可以用最小边长稳定：

```yaml
id_patch_expand_min_size: 16
```

或更激进：

```yaml
id_patch_expand_min_size: 24
```

但 `min_size=24` 在多人合影里很容易覆盖邻居，必须配合：

```yaml
id_patch_roi_max_faces: 1
ROI_DEBUG=1
```

先单脸诊断。

---

## 5. `roi_subsample={2,4}` 够吗？

`roi_subsample` 只增加 query probes，不增加 ref K 数。

对当前 6×7 face：

| m | q_sub token 数 | ref_k(r=2) |
|---:|---:|---:|
| 2 | 168 | 168 |
| 4 | 672 | 168 |

所以 `m=4` 会让 query 更密，但如果 ref K 仍只有 168，效果可能受 K 侧稀疏限制。

更合理的组合：

| 目标 | 建议 |
|---|---|
| 快速验证 qsub 是否跑通 | `m=2, r_ref=2` |
| 验证 query 密度是否有收益 | `m=4, r_ref=2 or 3` |
| 验证接近 crop512 的 K 数 | `m=4, r_ref=4 or 5, max_faces=1` |
| 降低串脸 | `r_ref≤3` + face/seg mask + adaptive beta |

---

## 6. 是否应该把 active layers 设成全部层？

上传的 `FLUX.2-klein-base-4B` transformer config 显示：

```json
num_layers = 5
num_single_layers = 20
num_attention_heads = 24
attention_head_dim = 128
axes_dims_rope = [32,32,32,32]
```

因此合法层下标是：

```text
double layers: 0,1,2,3,4
single layers: 0..19
```

如果你沿用旧配置：

```yaml
id_patch_idx_double_window: [1,3,5,7]
id_patch_idx_single_window: [1,3]
```

在 4B base 上实际只有：

```text
double: 1,3 生效；5,7 越界无效
single: 1,3 生效
```

也就是每 step 只在 4 层做 ROI qsub。

### 6.1 是否直接全层？

可以测试，但我不建议第一步就把所有层都作为主结论。原因：

- 早期层更负责全局结构/布局，强行注入 ref 可能造成脸型/表情漂移；
- 中后层更可能负责局部纹理和细节；
- 全层会显著增加计算量，也可能放大错误匹配。

建议按三档跑：

```yaml
# L1: 当前少层 baseline
id_patch_idx_double_window: [1, 3]
id_patch_idx_single_window: [1, 3]

# L2: 中后层增强（推荐下一步）
id_patch_idx_double_window: [2, 3, 4]
id_patch_idx_single_window: [6, 8, 10, 12, 14, 16, 18]

# L3: 全层压力测试
id_patch_idx_double_window: [0, 1, 2, 3, 4]
id_patch_idx_single_window: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

判定：

```text
如果 L2 > L1 且不漂移，说明层数不够是瓶颈之一。
如果 L3 更清晰但脸型/表情变坏，则说明需要 layer gating 或只用中后层。
如果 L1/L2/L3 都没明显变化，则问题不在层数，而在 K 侧密度/对齐/高频源。
```

---

## 7. 针对你当前 case 的下一组实验建议

先只处理 `roi_max_faces=1`，因为当前 debug 已显示第一张脸很小，且匹配列表里有相邻/重复区域风险。

### Group A：验证层数

```yaml
id_patch_roi_subsample: 2
id_patch_expand_ratio_ref: 2.0
id_patch_expand_ratio_lq: 1.5
id_patch_roi_detail_beta: 0.5

# A1
id_patch_idx_double_window: [1, 3]
id_patch_idx_single_window: [1, 3]

# A2
id_patch_idx_double_window: [2, 3, 4]
id_patch_idx_single_window: [6, 8, 10, 12, 14, 16, 18]

# A3
id_patch_idx_double_window: [0, 1, 2, 3, 4]
id_patch_idx_single_window: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

### Group B：验证 ref K 范围

固定 A2 层：

```yaml
id_patch_roi_subsample: 2
id_patch_expand_ratio_ref: 2.0 / 3.0 / 4.0
id_patch_expand_ratio_lq: 1.5
```

看 `ROI_DEBUG` 里的 `ref_k`：

```text
r_ref=2 → ref_k≈168
r_ref=3 → ref_k≈378
r_ref=4 → ref_k≈672
r_ref=5 → ref_k≈1050
```

### Group C：验证 query density

固定 A2 层 + 最佳 r_ref：

```yaml
id_patch_roi_subsample: 2 / 4
```

若 `m=4` 不提升，说明 query probe 多了但 K/V 或输出 native 容量仍然限制。

### Group D：验证 alignment

等代码支持 `pe3_all_canonical` 后：

```yaml
id_patch_roi_pe_mode: pe1 / pe3 / pe3_all_canonical
```

这是对应你提出的“crop 好可能因为更对齐”的关键实验。

---

## 8. 建议的研究 framing

这条线更适合写成：

```text
Prior Injection for Diffusion Refinement
```

而不是：

```text
ROI upscale trick
```

我们要研究的是：

```text
如何把区域重要性、身份匹配、空间对齐、结构/细节分离、置信度等先验，转化成扩散模型内部 attention 的 query probes、K/V routing、RoPE maps 和 adaptive gates。
```

这比单纯“文生图后处理”更通用：

- 文生图：生成后发现局部脸/手/文字不好，可以用 mask + prompt/reference prior 做局部 attention 修复；
- 图像编辑：需要保持结构但注入参考细节，可以用 structure/detail split branch；
- 高分辨率生成：可以用 PiD 类 pixel decoder 做 high-res decoding，用 prior-injected attention 改善 latent correctness；
- 多参考一致性：ID/产品/角色一致性可以通过匹配先验和 adaptive gate 控制。

---

## 9. 当前结论

对你这次 debug 来说：

```text
m=2, r_ref=2.0 还只是 168 q_sub / 168 ref_k。
```

这能验证代码路径，但距离 crop512 的 1024 tokens 还有明显差距。

我建议下一步不要只扫 `m={2,4}`，而是加上：

```yaml
id_patch_expand_ratio_ref: 2.0 / 3.0 / 4.0
id_patch_expand_min_size: 16
```

同时用 4B base 的合法层数重新设置：

```yaml
id_patch_idx_double_window: [2, 3, 4]
id_patch_idx_single_window: [6, 8, 10, 12, 14, 16, 18]
```

再做全层压力测试：

```yaml
id_patch_idx_double_window: [0, 1, 2, 3, 4]
id_patch_idx_single_window: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

如果这些都提升有限，就基本说明：**在 fixed native output grid 下，attention prior injection 可以改善匹配/身份约束，但真正清晰度可能需要 ref_hr 真高频源或 PiD 类 high-res pixel diffusion decoder。**
