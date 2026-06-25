# 14 · Q-only Supersampling / Canonical ROI Alignment 的创新性调研与定位

> 本文重新检查 `docs/11/12/13` 提出的方向：
> 1. Q-only Supersampling ROI Attention：只复制 native noise face query 成 `m×m` sub-query，用子 token RoPE 位置探测 native lq/ref K/V；
> 2. Canonical ROI Coordinate Alignment：不改变 stream id，只把 noise/lq/ref ROI 的 H/W 映射到 target-local face coordinate；
> 3. split-branch lq/ref attention + beta/gate：结构和细节分支分开，避免同一 softmax 互相稀释。
>
> 结论先行：**单个组件都能找到相邻工作，但“training-free + FLUX.2 multi-reference refiner + native output grid + Q-only sub-token RoPE probing + canonical ROI alignment + lq/ref split-branch”的组合没有找到直接等价方法。** 创新点应当按“组合创新 + 问题特化 + 负结果驱动的机制拆解”来表述，而不是声称“ROI attention / mixed-resolution / reference attention 本身全新”。

---

## 1. 方法的准确命名

建议内部和论文里暂称：

```text
Query-Foveated ROI Attention for Training-Free Multi-Reference Face Refinement
```

更具体：

```text
Q-only Sub-token RoPE Probing + Canonical Multi-stream ROI Alignment
```

不要叫 “latent upsampling”，因为它不提高 latent 内容分辨率，也不改变输出 token 网格。

---

## 2. 与已有方法族的对比

### 2.1 RALU / Region-Adaptive Latent Sampling

代表：

- RALU: *Upsample What Matters: Region-Adaptive Latent Sampling for Accelerated Diffusion Transformers*, arXiv 2507.08422.

RALU 的核心：

```text
低分辨率全局 denoising → ROI full-resolution latent upsampling → 全量 full-resolution refinement
```

并用 noise-timestep rescheduling 处理不同分辨率 latent 的噪声水平不匹配。

| 维度 | RALU | 当前 Q-only / canonical alignment |
|---|---|---|
| 主要目标 | 加速 DiT inference，同时保持质量 | 提升/验证多参考小脸 ID 细节迁移机制 |
| 是否改变 latent 分辨率 | 是 | 否 |
| 是否改变 denoising trajectory | 是，多阶段 mixed-resolution trajectory | 否，只改 attention 内 ROI query path |
| 是否需要 noise-timestep rescheduling | 是 | 当前不需要，因为没有 latent resolution transition |
| 是否对 V / latent output 做 upsample/downsample | 是 | 否 |
| 是否 reference-based ID refinement | 不是核心问题 | 是核心问题 |

**相似点**：都认为 ROI 应该获得更多计算。

**关键不同**：RALU 是 latent-resolution scheduling；当前方法是 query-side attention probing。当前方法不生成 full-res ROI latent，也不做 full-res refinement，因此不能直接叫 RALU 变体。

创新边界：

```text
不是 region-adaptive latent upsampling，而是 region-adaptive query-position supersampling。
```

---

### 2.2 Foveated Diffusion / mixed-resolution token density

代表：

- *Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation*, arXiv 2603.23491.

Foveated Diffusion 根据 gaze/fovea mask 给重要区域更高 token density，外围更低 token density，并且需要 post-training mixed-resolution model。

| 维度 | Foveated Diffusion | 当前方法 |
|---|---|---|
| 是否改变 token density | 是，fovea 高密度，periphery 低密度 | 不改变最终 token density |
| 是否训练/post-train | 是 | training-free |
| 目标 | 感知等价的高效生成 | 多参考 ID 小脸 refinement |
| 位置机制 | mixed-resolution token construction | sub-token RoPE probing + ROI canonical alignment |

**相似点**：都属于“foveated/ROI 获得更多计算”。

**关键不同**：Foveated Diffusion 的 fovea 是真实 token density 提高；当前只是 attention 内 query probes 增加，输出 grid 不变。

创新边界：

```text
attention-level foveation，而不是 token-level foveation。
```

---

### 2.3 CRPA / mixed-resolution RoPE 修复

代表：

- *One Attention, One Scale: Phase-Aligned Rotary Positional Embeddings for Mixed-Resolution Diffusion Transformer*, arXiv 2511.19778.

CRPA 解决的问题是 mixed-resolution DiT 中不同空间网格的 RoPE 相位采样率不一致，导致 attention collapse/blur/artifact。它通过把 Q/K 位置表达到 query stride 下，保持物理距离和相位增量一致。

| 维度 | CRPA | 当前方法 |
|---|---|---|
| 核心问题 | mixed-resolution RoPE phase aliasing | 小脸 ID refiner 中 query 匹配粒度和跨图 ROI 对齐 |
| 是否新增 query probes | 否 | 是 |
| 是否改 latent token resolution | 可服务于 mixed-resolution | 不改输出 token resolution |
| 是否 reference ID transfer | 不是重点 | 是重点 |
| 位置思想 | query-stride phase alignment | sub-token query RoPE + ref/ROI canonical H/W alignment |

**相似点**：都把 RoPE index map 当作可以 training-free 调整的关键控制面。

**关键不同**：CRPA 是修复不同分辨率网格混合时的相位一致性；当前方法用子 token RoPE 位置主动生成多个 query probes，并引入跨图 ROI canonical alignment。

创新边界：

```text
CRPA-like phase reasoning can作为后续理论支撑，但 Q-only sub-query probing 不是 CRPA 本身。
```

---

### 2.4 HierEdit / local-window high-res editing

代表：

- *HierEdit: Region-Aware Hierarchical Diffusion for Efficient High-Resolution Editing*, arXiv 2605.17294.

HierEdit 用低分 proxy 定位编辑区域，再用 Local-Window MMDiT refine 4K high-res 局部区域。

| 维度 | HierEdit | 当前方法 |
|---|---|---|
| 目标 | 高分辨率局部编辑加速 | 小脸 ID 细节迁移机制验证 |
| 是否局部窗口模型 | 是 | 否，仍在原 attention block 内做 surgery |
| 是否改变局部图像分辨率 | 是 | 否 |
| 是否需要额外局部模型/流程 | 是 | 不需要 |

**相似点**：都强调 edited/important region 的局部优先处理。

**关键不同**：HierEdit 是层级局部 high-res refinement；当前方法不引入局部模型、不输出局部 crop、不做后处理。

创新边界：

```text
不是 hierarchical local refinement，而是原模型 attention 内的 query-side局部增强。
```

---

### 2.5 Deformable Attention / 多采样点 attention

代表：

- Deformable DETR: *Deformable Transformers for End-to-End Object Detection*, arXiv 2010.04159.

Deformable DETR 的 attention 只 attend reference point 周围少量 key sampling points，解决视觉 transformer 在高分特征图上的收敛和效率问题。

| 维度 | Deformable Attention | 当前方法 |
|---|---|---|
| 多点采样 | 是，围绕 reference point 学 offsets | 是，但 sub-query 位置是规则子 token RoPE probes |
| 是否采样 feature/value | 是，采样多尺度 feature values | 否，不插值/采样 V，只读取 native K/V |
| 是否训练 learned offsets | 是 | training-free，无 learned offsets |
| 任务 | detection / segmentation 等判别任务 | diffusion refiner 生成任务 |
| 输出 | object query / detection features | native noise face token residual |

**相似点**：都把一个 query 的空间感受点拆成多个位置。

**关键不同**：Deformable Attention 是 learned sparse feature sampling；当前方法是 fixed sub-token RoPE query probing，不改 value，不训练。

创新边界：

```text
概念上接近“多位置 query probing”，但不是 deformable attention；更像 training-free sub-token RoPE probe。
```

---

### 2.6 Prompt-to-Prompt / Attend-and-Excite / attention editing

代表：

- Prompt-to-Prompt: cross-attention control for prompt-based editing, arXiv 2208.01626.
- Attend-and-Excite: on-the-fly attention guidance to reduce subject neglect, arXiv 2301.13826.

这些方法都操作 attention，但主要面向 text cross-attention 和 prompt semantic control。

| 维度 | Prompt-to-Prompt / Attend-and-Excite | 当前方法 |
|---|---|---|
| 操作对象 | text-image cross-attention map | image-stream self/joint attention 中的 noise/lq/ref ROI |
| 目标 | 保布局、增强文本概念、避免漏生成 | 多参考同 ID face detail transfer |
| 是否新增 sub-query RoPE probes | 否 | 是 |
| 是否多图 ROI 对齐 | 否 | 是 |

**相似点**：都是 test-time attention intervention。

**关键不同**：当前方法不是文本 token reweight，也不是 prompt semantic nursing；它直接在 image tokens 之间做 ID ROI attention。

---

### 2.7 MasaCtrl / mutual self-attention / reference feature injection

代表：

- MasaCtrl: tuning-free mutual self-attention for consistent image synthesis/editing, arXiv 2304.08465.

MasaCtrl 把 self-attention 改成 mutual self-attention，让 target query 查询 source image 的局部内容/纹理，支持一致性和非刚性编辑。

| 维度 | MasaCtrl | 当前方法 |
|---|---|---|
| 是否 training-free | 是 | 是 |
| 是否 reference/source feature attention | 是 | 是 |
| 是否 ID bbox / 多人隔离 | 不是核心 | 是核心 |
| 是否 sub-token RoPE probing | 否 | 是 |
| 是否 canonical ROI alignment | 不是主要机制 | 是后续关键变量 |
| 是否 split lq/ref 结构/细节 | 否 | 是 |

**相似点**：最接近的一类：training-free，改 self-attention，让 target 从 source 借局部外观。

**关键不同**：当前方法特化到 FLUX.2 多参考 refiner 的 `[noise,lq,ref]` 三路输入，使用 ID 匹配 bbox、wrong-ID hard block、sub-query RoPE probes、lq/ref 分支和 canonical face alignment。

创新边界：

```text
可以说继承了 mutual-attention/ref-attention 的思想，但新增了小脸 ID refinement 所需的 query foveation 与三路 ROI canonical alignment。
```

---

### 2.8 FRESCO / TokenFlow / correspondence-based consistency

代表：

- FRESCO: spatial-temporal correspondence for zero-shot video translation, arXiv 2403.12962.

这类方法显式建立 correspondence，增强跨帧/跨图一致性。

| 维度 | Correspondence methods | 当前方法 |
|---|---|---|
| 是否关心跨图/跨帧对应 | 是 | 是 |
| correspondence 来源 | optical flow / feature matching / attention correspondence | ID bbox + optional canonical ROI coordinates + attention entropy |
| 目标 | 视频/编辑一致性 | 多参考同 ID 小脸细节恢复 |
| 是否 sub-query RoPE | 否 | 是 |

**相似点**：都认为“对齐/对应关系”重要。

**关键不同**：当前方法用 RoPE coordinate remapping 和 ID bbox canonicalization 作为轻量 alignment prior，不是显式光流/匹配场。

---

### 2.9 IP-Adapter / InstantID / PhotoMaker / PuLID / DynamicID / InstantFamily

代表：

- IP-Adapter: decoupled cross-attention for image prompt adapters, arXiv 2308.06721.
- InstantID: IdentityNet + face/landmark conditions, arXiv 2401.07519.
- PhotoMaker: stacked ID embedding, arXiv 2312.04461.
- PuLID: tuning-free ID customization via contrastive alignment, arXiv 2404.16022.
- InstantFamily: masked attention for zero-shot multi-ID image generation, arXiv 2404.19427.
- DynamicID: Semantic-Activated Attention + identity-motion reconfiguration, arXiv 2503.06505.

| 维度 | ID adapter/customization methods | 当前方法 |
|---|---|---|
| 目标 | 生成指定 ID 的人物/多人图 | refiner 场景下修复已有合影小脸 |
| 是否训练 adapter / ID branch | 通常需要训练或预训练 ID 模块 | training-free，不新增权重 |
| ID 信息来源 | face encoder / ID embedding / landmark / adapter | 已在输入序列里的 lq/ref K/V + ReID bbox 匹配 |
| 是否多 ID 隔离 | 有些方法用 masked attention | 用 bbox pair 做 per-ID ROI hard block |
| 是否 sub-query RoPE probing | 否 | 是 |
| 是否 canonical ROI H/W alignment | 通常不是以 RoPE 坐标重映射形式 | 是一个核心变量 |

**相似点**：都关注 ID fidelity，多 ID 方法也会用 mask/attention 隔离。

**关键不同**：当前不是人脸 ID 生成 adapter，而是对已有 FLUX.2 refiner 的 image-token attention 做 inference-time surgery；不训练、不接人脸编码器作为新条件，不改变模型结构。

创新边界：

```text
不是新的 ID adapter，而是 training-free ID-aware attention routing/refinement for existing multi-reference refiner。
```

---

### 2.10 ControlNet / GLIGEN / BoxDiff / spatial control

代表：

- ControlNet: trainable spatial condition control, arXiv 2302.05543.
- GLIGEN: grounded text-to-image via trainable gated grounding layers, arXiv 2301.07093.
- BoxDiff: training-free box-constrained diffusion, arXiv 2307.10816.

| 维度 | Spatial control methods | 当前方法 |
|---|---|---|
| 控制输入 | box/mask/depth/pose/layout | ID bbox pair + lq/ref image tokens |
| 目标 | 物体在哪、布局遵守 | 同 ID 小脸细节恢复 |
| 是否训练 | ControlNet/GLIGEN 需要训练；BoxDiff training-free | 当前 training-free |
| 是否改 attention query resolution | 通常否 | 是 |
| 是否 reference detail transfer | 通常不是核心 | 是核心 |

**相似点**：都用空间条件约束生成。

**关键不同**：当前不是 layout-to-image 控制，而是已知 ID 匹配后的 ROI-level reference transfer。

---

### 2.11 High-resolution generation/editing: ScaleCrafter / DemoFusion / EditCrafter / MultiDiffusion

代表：

- ScaleCrafter: tuning-free higher-resolution generation, arXiv 2310.07702.
- DemoFusion: progressive upscaling, skip residual, dilated sampling, arXiv 2311.16973.
- EditCrafter: tuning-free high-res editing, arXiv 2604.10268.
- MultiDiffusion: fusing multiple diffusion paths, arXiv 2302.08113.

这些方法解决高分辨率画布、tile、panorama、high-res editing 的问题。

| 维度 | High-res/tile methods | 当前方法 |
|---|---|---|
| 目标 | 整图高分生成/编辑 | 合影小脸局部 ID refinement |
| 是否多路径/多尺度采样 | 通常是 | 否 |
| 是否输出高分局部 | 可以 | 否，输出 native grid |
| 是否 reference ID transfer | 通常不是核心 | 是 |
| 是否 sub-query RoPE probing | 否 | 是 |

**相似点**：都关心高分/局部细节。

**关键不同**：当前不走 tile/crop/multi-diffusion paths，只改原模型 attention。

---

### 2.12 Reference-based inpainting / detail-preserving attention

代表：

- HiFi-Inpaint: Shared Enhancement Attention + Detail-Aware Loss, arXiv 2603.02210.
- Patch-Adapter: patch-level attention for ultra-high-res inpainting, arXiv 2510.13419.

| 维度 | Reference inpainting/detail methods | 当前方法 |
|---|---|---|
| 目标 | 参考图细节保真，常面向 product/human-product | 多参考 face refiner |
| 是否训练 | 通常训练模块/损失 | training-free |
| 是否显式细节损失 | 是 | 否 |
| 是否 patch/ref attention | 是 | 是，但在已有 lq/ref/noise tokens 上做 |
| 是否 sub-token RoPE probes | 否 | 是 |

**相似点**：都认识到 reference detail preservation 需要更细 attention 或增强机制。

**关键不同**：当前没有训练 SEA/DAL/adapter，而是在推理 attention 中测试 query/probe/alignment 机制。

---

## 3. 真正可主张的创新点

### 创新点 A：Q-only sub-token RoPE probing

一句话：

> 不插值 latent value、不增加输出 token，只把 native ROI query 复制成多个子位置 RoPE probes，以提高 attention 匹配分辨率。

它区别于：

- RALU/Foveated：不提高 latent/token density；
- Deformable Attention：不学 offsets、不采样 V；
- CRPA：不是单纯修正 mixed-resolution RoPE，而是主动创建 query probes；
- B/per-layer/persist：不走 `Interp(V)` 和 `Downsample(O)`。

这是最明确的新点。

### 创新点 B：Multi-stream ROI canonical alignment while preserving stream T

一句话：

> 对同 ID 的 noise/lq/ref ROI 只在 H/W 上映射到 target-local face coordinate，T 仍保留 noise/lq/ref stream identity。

它区别于：

- crop→1k：显式裁剪+resize+完整重跑；
- CRPA：处理 mixed-resolution phase consistency，不专门处理多图 ID ROI 对齐；
- MasaCtrl/FRESCO：依赖 mutual attention/correspondence，不是 4D RoPE stream-preserving canonicalization。

这是第二个可主张点，尤其对应用户提出的“crop 好可能因为更对齐”。

### 创新点 C：三路 refiner 的结构/细节分支拆解

一句话：

> 在 `[noise,lq,ref]` refiner 中，把 lq 作为结构分支、ref 作为细节/身份分支，split softmax 后再用 beta/gate 融合。

它区别于：

- IP-Adapter 的 decoupled text/image attention：那里是训练 adapter 的 image prompt；
- InstantFamily 的 masked attention：多 ID generation，不是已有 lq/ref/noise refiner 的结构/细节拆分；
- A′ concat KV：lq/ref 同一 softmax 下互相稀释。

这个点单独不一定足够新，但和 A/B 结合后形成清楚机制。

### 创新点 D：负结果驱动的 crop-gain decomposition

该分支已经形成一条清晰实验逻辑：

```text
Expanded-KV 无效 → noise fixup 让输出响应但不提清晰度 → 插值式 ROI 变糊 → Q-only probing / canonical alignment 拆解 crop 收益
```

这本身是研究贡献：系统拆解 crop→1k 的收益来源：

```text
token density + spatial alignment + local dominance + full denoising trajectory
```

---

## 4. 不建议这样 claim

不建议 claim：

```text
我们首次提出 ROI attention。
```

因为 attention control / region attention / masked attention 已有大量工作。

不建议 claim：

```text
我们首次提出 foveated diffusion。
```

因为 Foveated Diffusion、RALU 等已经存在。

不建议 claim：

```text
我们首次提出 reference attention / ID preservation。
```

因为 MasaCtrl、IP-Adapter、InstantID、PhotoMaker、PuLID、InstantFamily 等已有。

不建议 claim：

```text
我们首次使用 RoPE coordinate remapping。
```

因为 CRPA 已经把 RoPE index map 作为 training-free 控制面系统化。

---

## 5. 建议的创新表述

比较稳的版本：

> We propose a training-free query-foveated ROI attention mechanism for multi-reference face refinement. Instead of upsampling latent values or changing the output grid, our method replicates each native face query into sub-token RoPE probes, attends to native lq/ref K/V in separated structure/detail branches, and aggregates the result back to the original native token. We further introduce stream-preserving canonical ROI coordinate alignment to test whether the gain of crop-based refinement comes from token density alone or also from cross-image alignment.

中文：

> 我们提出一种 training-free 的 query-foveated ROI attention，用于多参考合影小脸 refiner。它不插值 latent value、不改变输出网格，而是把每个 native face query 复制成多个子 token RoPE probe，在 lq 结构分支和 ref 细节分支中分别 attention，再聚合回原 native token。同时，我们提出保留 stream id 的 canonical ROI 坐标对齐，用来拆解 crop→1k 的收益到底来自 token 密度还是三路对齐。

---

## 6. 创新性等级判断

### 若只实现 Q-only supersampling

创新性：**中等**。

原因：query 多点 probing 和 attention surgery 有相邻工作，但“sub-token RoPE query probing，不插值 V，不改变 output grid”的形式比较有新意。

### 若再加 canonical ROI alignment 并做消融

创新性：**中高**。

原因：这会把问题从“又一个 ROI attention trick”提升为对 crop 收益来源的机制拆解，能明确回答 density vs alignment 的问题。

### 若再加 adaptive beta/gate 并证明多图稳定

创新性：**中高到较高**。

原因：这会形成完整系统：

```text
ID bbox matching → query-foveated probing → canonical alignment → structure/detail adaptive fusion
```

尤其如果能在 wide group photo 小脸上证明有效，和已有 ID adapter / high-res diffusion / foveated diffusion 的差异会更明显。

---

## 7. 需要验证才可站稳的关键实验

1. **Q-only vs interpolate B**：证明避免低通后至少不变糊。
2. **pe1 vs ref2target vs all-canonical**：证明 alignment 是独立贡献。
3. **fixed beta vs adaptive beta**：证明不同图片需要不同融合强度。
4. **crop aligned vs crop misaligned**：证明 crop→1k 的收益确实包含 alignment。
5. **no-ref / wrong-ref / matched-ref**：证明 ID ref 分支不是随机锐化或背景纹理注入。

---

## 8. 最终判断

当前方法的创新性不是来自单独一个关键词，而是来自一个特定组合：

```text
training-free
+ FLUX.2 multi-reference refiner
+ ID bbox matched ROI
+ fixed native output grid
+ Q-only sub-token RoPE probes
+ stream-preserving canonical ROI alignment
+ lq/ref structure-detail split attention
```

调研中没有找到完全等价的已有方法。最接近的是：

- MasaCtrl：training-free mutual self-attention/reference feature borrowing；
- CRPA：training-free RoPE index remapping for mixed resolution；
- Deformable Attention：query 多采样点；
- InstantFamily/IP-Adapter 系列：ID/reference attention 与多 ID 隔离；
- RALU/Foveated：ROI 获得更多计算/更高密度。

但它们分别缺少当前方法中的一个或多个核心约束：不改 output grid、不插值 V、不训练 adapter、保留 stream T 的 canonical face alignment、multi-reference refiner 的 lq/ref 结构-细节分离。

因此，建议把创新性定位为：

> **一种 training-free、面向多参考小脸 refiner 的 query-side foveated attention 与 canonical multi-stream ROI alignment 方法。**

而不是泛泛地说“提出 ROI upscale”。
