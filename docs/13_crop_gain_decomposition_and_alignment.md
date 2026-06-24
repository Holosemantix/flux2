# 13 · crop→1k 为什么好：密度、对齐、局部主导与完整去噪轨迹的拆解

> 用户新假设：crop 之所以效果好，可能不仅因为 noise/lq/ref 的 token 数量更多、密度更大，还因为三者在 crop 后更对齐了。  
> 这个判断很关键。后续实验不能只围绕“提 token 密度”做，还要把 **alignment** 单独作为变量拆出来。

---

## 1. crop→1k 带来的不止是 token 变多

把一张小脸 crop 出来 resize 到 1k 后再跑 refiner，至少同时改变了四件事：

| 因素 | crop→1k 中发生了什么 | 对结果的可能影响 |
|---|---|---|
| token density | 小脸从约 8–12 token 变成约 64×64 token | 表达容量、attention 匹配粒度、局部细节生成能力大幅提高 |
| spatial alignment | noise/lq/ref 的脸都被放到类似尺度、类似中心位置 | 同一局部语义更容易在 attention 里对应起来 |
| local dominance | 画面几乎只有脸，背景/身体/其他人被移除 | 注意力不再被整图背景和其他 ID 稀释 |
| full denoising trajectory | 高密度脸参与整个 denoising 过程，而不是单层/单步旁路 | 细节可以跨 step、跨 layer 逐步合成，而不是一次 attention 注入 |

因此，如果 crop→1k 变清晰，不能直接推出“只要在原图 attention 里把 ROI token 数变多就能复现”。它可能是四个因素叠加。

---

## 2. 对当前方案的启示

当前 B-2 Q-only supersampling 主要验证的是：

```text
attention query 的位置探测密度是否有帮助
```

但它没有完整解决：

```text
noise/lq/ref 三路是否在同一个局部规范坐标系里对齐
```

具体来说：

- noise/lq 是同图同空间，天然比较对齐；
- ref 来自另一张图，绝对位置、脸大小、姿态都可能不同；
- 如果只用 ref 的原始绝对 H/W 坐标，attention 可能在“同一个人但不同位置/尺度”的 token 间做困难匹配；
- crop→1k 通过裁剪和 resize，隐式把 ref/lq/noise 放到了更相似的尺度和中心，这等价于一种强 alignment prior。

因此，下一步除了 Q-only supersampling，还应该单独测试：

> 不增加 ref crop token、不做输出后处理，仅通过 RoPE 坐标重映射，把 noise/lq/ref ROI 映射到同一个 canonical face coordinate，是否能提升。

---

## 3. 新增假设：Canonical ROI Coordinate Alignment

### 3.1 核心想法

对同一个 ID 的三段 ROI：

```text
noise face ROI, lq face ROI, ref face ROI
```

不要让它们使用各自原图的绝对 H/W 坐标，而是在 ROI attention 内部把它们映射到同一个 canonical coordinate：

```text
u, v ∈ [0, 1]
```

例如，对任意 bbox `B=(y1,x1,y2,x2)` 内的 token：

```math
u = \frac{y - y_1 + 0.5}{y_2-y_1}, \quad
v = \frac{x - x_1 + 0.5}{x_2-x_1}
```

然后映射到 target/lq 脸框坐标：

```math
H^{canon} = y_1^{target} + u (y_2^{target}-y_1^{target})
```

```math
W^{canon} = x_1^{target} + v (x_2^{target}-x_1^{target})
```

stream id 仍保留：

```text
noise query: T = T_noise
lq key:      T = T_lq
ref key:     T = T_ref
```

也就是说，对齐只发生在 H/W，不改变 token 属于哪一张图。

### 3.2 和现有 pe3 的关系

现有 `pe3` 已经是 ref→target 的坐标映射雏形：

```text
ref key 的 T 保持 T_ref，H/W 映射到 target/lq 脸坐标
```

但后续可以更明确地区分三种对齐模式：

| 模式 | 说明 | 目的 |
|---|---|---|
| pe1 | 使用各自原图绝对坐标 | baseline / 看绝对位置是否重要 |
| pe3_ref2target | 只把 ref ROI 映射到 target ROI | 测 ref 对齐是否足够 |
| pe3_all_canonical | noise/lq/ref 都使用 target-local canonical H/W | 最大程度模拟 crop 的三路局部对齐 |

---

## 4. crop 效果拆解实验

下面实验用于回答：crop→1k 的收益到底来自哪里。

### E1：密度 vs 对齐

| 实验 | 操作 | 目的 |
|---|---|---|
| crop→1k aligned | 正常裁脸 resize 到 1k | 上限参考，密度+对齐+局部主导+完整轨迹全都有 |
| crop→1k misaligned | lq/ref crop 尺度一样，但人为平移/缩放 ref 脸位置 | 若明显变差，说明 alignment 是关键因素 |
| crop→1k same-density no-align | 保持 1k token 密度，但 ref 脸不居中或尺度不一致 | 分离“密度”和“对齐” |

判断：

```text
如果 aligned 明显好于 misaligned，则 crop 的收益不只是 token 多，alignment 也很重要。
```

### E2：原图 native token + 坐标对齐

在原图 native token 上，不增加 token，只改 ROI attention 的 RoPE H/W：

| 实验 | 操作 | 目的 |
|---|---|---|
| native + pe1 | 原始绝对坐标 | baseline |
| native + ref2target pe3 | ref ROI H/W 映射到 target ROI | 测 alignment 是否单独有效 |
| native + all-canonical pe3 | noise/lq/ref 都映射到 target-local canonical H/W | 尽量模拟 crop 对齐，但不增加 token |

判断：

```text
如果 native + pe3 提升 ArcFace/清晰度，说明 alignment 本身有收益；
如果仍无收益，说明 token density / full trajectory 可能更关键。
```

### E3：Q-only supersampling + canonical alignment

在 B-2 基础上组合：

```yaml
id_patch_roi_variant: "q_supersample"
id_patch_roi_pe_mode: "pe3_all_canonical"
id_patch_roi_subsample: 2 or 4
```

对比：

| 实验 | 变量 |
|---|---|
| qsub + pe1 | 只提高 query 探测密度，不做 ref 对齐 |
| qsub + pe3_ref2target | query 探测 + ref 对齐 |
| qsub + pe3_all_canonical | query 探测 + 三路 canonical 对齐 |

判断：

```text
如果 qsub+pe3_all_canonical > qsub+pe1，说明“密度探测”和“坐标对齐”有叠加价值。
```

### E4：完整轨迹 vs 单层注入

crop→1k 的另一个优势是高密度脸参与整个 denoising trajectory。当前 attention surgery 只在若干层里一次性注入。

对比：

| 实验 | 操作 | 目的 |
|---|---|---|
| qsub 少层 | 只在中后层启用 | 稳定性 baseline |
| qsub 多层 | 多个 double/single active 层 | 看是否需要跨层累积 |
| qsub 多 timestep | 每个 denoising step 都启用 | 当前默认会随每步调用，确认是否稳定 |
| crop→1k | 高密度完整轨迹 | 上限 |

判断：

```text
如果 qsub 多层仍不接近 crop→1k，说明完整高密度 denoising trajectory 是关键，而不是单层 attention 能完全补偿。
```

---

## 5. 对代码设计的建议

### 5.1 新增 pe mode

建议后续把 `id_patch_roi_pe_mode` 扩展为：

```yaml
id_patch_roi_pe_mode: "pe1"              # 绝对坐标
id_patch_roi_pe_mode: "pe3_ref2target"   # ref → target
id_patch_roi_pe_mode: "pe3_all_canonical"# noise/lq/ref 全部 target-local canonical
```

当前 `pe3` 可以视作 `pe3_ref2target`。

### 5.2 不要把 T 也 canonicalize

无论哪种 H/W 对齐，T 都不应该统一：

```text
noise query: T_noise
lq key:      T_lq
ref key:     T_ref
```

原因：T 负责区分 stream 身份；H/W 负责空间对齐。把 T 也统一会混淆输出和条件图。

### 5.3 alignment 和 beta gate 可以结合

如果 ref 与 target 的 alignment 置信度低，则降低 ref 分支 beta：

```math
\beta_f = \mathrm{clip}(\beta_0 + \lambda_1 C_{id} + \lambda_2 C_{sharp} + \lambda_3 C_{align}, \beta_{min}, \beta_{max})
```

其中 `C_align` 可以来自：

- lq/ref bbox aspect ratio 是否接近；
- face landmark/关键点几何是否接近；
- qsub attention entropy 是否低；
- ref2target 后 attention 是否更集中。

---

## 6. 研究判断

这个新假设把 crop→1k 的收益拆成：

```text
token density + spatial alignment + local dominance + full denoising trajectory
```

因此后续 paper/story 可以更清楚：

1. **Version A/A′** 证明：只扩大 KV 或直接 noise fixup 不够；
2. **B/per-layer/persist** 证明：插值式 token upscale 甚至会低通变糊；
3. **B-2 qsub** 验证：不插值 value，只提高 query 探测密度是否有收益；
4. **B-3 canonical alignment** 验证：crop 的隐式对齐是否是关键；
5. **ref_hr 重编码/crop→1k** 作为上限：真高频源 + 完整轨迹的效果。

如果 B-2/B-3 都提升有限，就可以有力地说明：

> 在 fixed native output grid 下，training-free attention surgery 可以改善匹配和身份约束，但真正清晰度主要受限于真高频源与完整高密度 denoising trajectory。

如果 B-3 明显提升，则说明：

> crop 的收益有相当部分来自三路 ROI 的 canonical alignment，而不仅是 token 数量增加。这会成为一个很有价值的 training-free 方向。
