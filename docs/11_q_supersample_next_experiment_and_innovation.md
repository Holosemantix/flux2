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

## 3. 数学形式化

### 3.1 原始 ROI noise token

设某个 ID 的 noise 人脸 ROI 在 native token 网格中有 `N_f` 个 token。第 `i` 个 native noise face token 的 query 记为：

```math
q_i \in \mathbb{R}^{H \times d}, \quad i=1,\dots,N_f
```

其中 `H` 是 attention head 数，`d` 是每个 head 的维度。普通 A′ 直接用这个 `q_i` attend 到局部 lq/ref K/V：

```math
z_i = \mathrm{Attn}(q_i, K_{lq}\oplus K_{ref}, V_{lq}\oplus V_{ref})
```

问题是：当小脸只有 `8–12` 个 native token 时，每个 `q_i` 的空间采样很粗，`q_i` 和 ref face token 的匹配粒度不足。

### 3.2 子查询：只复制 query 内容，不插值 value

令 `m = id_patch_roi_subsample`。对每个 native query `q_i`，生成 `m^2` 个 sub-query：

```math
\tilde{q}_{i,a,b}^{raw} = q_i, \quad a,b \in \{0,\dots,m-1\}
```

注意这里 **内容向量完全相同**，不是 bilinear interpolation。唯一变化是子 token 位置。

如果 native token 的网格坐标是 `(y_i, x_i)`，则第 `(a,b)` 个子查询的位置设为：

```math
\tilde{p}_{i,a,b}
=
\left(
T_{noise},
y_i + \frac{a+0.5}{m} - 0.5,
x_i + \frac{b+0.5}{m} - 0.5,
0
\right)
```

然后对 sub-query 重新施加 RoPE：

```math
\tilde{q}_{i,a,b} = \mathrm{RoPE}(\tilde{q}_{i,a,b}^{raw}, \tilde{p}_{i,a,b})
```

直观解释：每个 native token 内部放 `m×m` 个“探针”，探针不是新 latent，也不是新像素，只是在 attention logits 里用更细的相位位置去询问 ref/lq。

### 3.2.1 为什么只改变 RoPE 位置，而不改变 query 内容

这个设计的关键是把“内容是什么”和“从哪里看”分开。

- `q_i` 的内容向量来自当前 denoising 状态下的 native noise face token，里面包含模型此刻对这块脸的语义、结构、身份线索和噪声状态。如果对 `q_i` 本身做 bilinear 上采样、MLP 生成或插值变换，就等于构造了模型训练时没见过的新 latent query 内容，容易产生 OOD 行为。
- RoPE 位置只影响 attention score 里的相对相位关系。也就是说，改变 `\tilde{p}_{i,a,b}` 主要改变这个 query “以哪个子 token 位置去看 ref/lq K”，而不是强行创造新的 latent 内容。
- 小脸 native token 少的问题，未必首先是“内容向量不够多”，也可能是“一个粗 query 只能以一个相位位置匹配 ref”。把同一个 `q_i` 放到 token 内部多个子位置，相当于用多个相邻视角去搜索 ref/lq 里哪个 K 更匹配。
- 旧 B 的失败来自 `Interp(V)` 和 `Downsample(O)` 的低通链路；B-2 不插值 value，也不下采样虚拟 latent output，只对 attention logits 做更细采样，因此它是在验证“匹配分辨率”这个独立因素。

更形式化地说，普通 attention 的 score 是：

```math
A_{ij} = \frac{\langle \mathrm{RoPE}(q_i,p_i),\mathrm{RoPE}(k_j,p_j)\rangle}{\sqrt{d}}
```

B-2 不改变 `q_i` 的内容，只把一个 score 变成 `m^2` 个子位置 score：

```math
A_{i,a,b,j} = \frac{\langle \mathrm{RoPE}(q_i,\tilde{p}_{i,a,b}),\mathrm{RoPE}(k_j,p_j)\rangle}{\sqrt{d}}
```

然后在 `m^2` 个子位置上得到多个 attention output，再聚合回 native token。这样做的含义是：**提高 query 的位置探测密度，而不是提高 latent 内容分辨率**。

这也是它和 ref crop 重编码的根本区别：ref crop 重编码给模型新增真高频 K/V；B-2 不新增真高频源，只测试更细粒度的 query-position probing 是否能更好利用已有 native ref K/V。

### 3.3 native lq/ref K/V，不做 P×P 插值

lq 结构分支的 key/value 直接取 native lq ROI：

```math
K_l, V_l = K/V(\mathrm{ROI}_{lq}^{exp})
```

ref 细节分支的 key/value 直接取 native ref ROI：

```math
K_r, V_r = K/V(\mathrm{ROI}_{ref}^{exp})
```

其中 `exp` 表示外扩 bbox，例如 `expand_ratio_lq=1.5`、`expand_ratio_ref=2.0`。这些 K/V 是原始模型已经投影出的 native K/V，不做 ROIAlign 上采样，不做 latent value 插值。

对 ref 的位置可以用 `pe3` 映射到 target/lq 脸坐标系。若 ref bbox 坐标是 `(y_r,x_r)`，映射到 target bbox 的位置记作：

```math
p_{r \rightarrow t}
= \Phi_{r\rightarrow t}(p_r)
```

因此：

```math
K_r^{rope} = \mathrm{RoPE}(K_r, \Phi_{r\rightarrow t}(p_r))
```

### 3.4 split-branch attention：避免 lq/ref 互相稀释

如果把 lq 和 ref 直接 concat 到一个 softmax：

```math
\mathrm{Attn}(\tilde{q}, K_l \oplus K_r, V_l \oplus V_r)
```

ref 细节很容易被 lq 结构 token 稀释。B-2 默认用 split branch：

```math
s_{i,a,b}^{lq}
= \mathrm{Attn}(\tilde{q}_{i,a,b}, K_l, V_l)
```

```math
s_{i,a,b}^{ref}
= \mathrm{Attn}(\tilde{q}_{i,a,b}, K_r, V_r)
```

再用 `detail_beta = \beta` 融合：

```math
s_{i,a,b}
= (1-\beta)\,s_{i,a,b}^{lq} + \beta\,s_{i,a,b}^{ref}
```

直观解释：lq 分支负责结构、姿态和表情稳定；ref 分支负责身份和细节。分开 softmax 后，ref 分支不会在同一个归一化分母里被 lq token 吃掉权重。

### 3.5 聚合回 native token

对同一个 native token 的 `m^2` 个 sub-query 输出做聚合：

```math
\bar{s}_i
= \mathrm{Agg}_{a,b}(s_{i,a,b})
```

当前实现先用 mean：

```math
\bar{s}_i = \frac{1}{m^2}\sum_{a=0}^{m-1}\sum_{b=0}^{m-1}s_{i,a,b}
```

也预留了 `center` 聚合，即只取中心 sub-query。

最后残差写回原 native noise face token：

```math
o_i^{new}
= (1-\alpha)o_i^{base} + \alpha\bar{s}_i
```

其中：

- `o_i^{base}` 是原始 full attention 的输出；
- `\alpha = id_patch_noise_alpha`；
- 写回位置仍是原始 native face ROI，不新增 token、不改输出网格。

### 3.6 和旧 B 的数学区别

旧 B 是内容上采样：

```math
Q^{P\times P}, K^{P\times P}, V^{P\times P}
= \mathrm{Interp}(Q,K,V)
```

然后：

```math
O^{P\times P}=\mathrm{Attn}(Q^{P\times P},K^{P\times P},V^{P\times P})
```

最后：

```math
O^{native}=\mathrm{Downsample}(O^{P\times P})
```

这条路径有两次低通风险：`Interp` 和 `Downsample`。

B-2 是 query 探针上采样：

```math
\tilde{Q}=\mathrm{Repeat}(Q) + \mathrm{SubtokenRoPE}
```

```math
\bar{O}^{native}=\mathrm{Agg}(\mathrm{Attn}(\tilde{Q},K^{native},V^{native}))
```

没有 `Interp(V)`，也没有 `Downsample(O)`，因此不会因为 value 插值和 latent 下采样天然变糊。

---

## 4. 已推送的代码入口

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

## 5. 推荐实验配置

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

## 6. 实验矩阵

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

## 7. 预期结果与决策

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

## 8. 和 RALU 一样吗？

**不一样。** 二者都属于“区域自适应 / ROI 优先”的思路，但操作层级不同。

| 维度 | RALU | B-2 Q-only supersampling |
|---|---|---|
| 目标 | 加速 DiT 推理，同时保留画质 | 验证小脸 ID 细节迁移是否受 attention 匹配粒度限制 |
| 分辨率操作 | 改变 latent 采样分辨率：低分全局 → ROI full-res → 全量 full-res refinement | 不改变 latent 分辨率、不新增输出 token |
| 是否改变 denoising trajectory | 是，多阶段 mixed-resolution denoising | 否，只改选定 attention 层的 ROI query path |
| 是否需要噪声/时间步重调度 | RALU 需要 noise-timestep rescheduling 稳定分辨率切换 | 当前 B-2 不需要，因为没有 latent 分辨率切换 |
| 是否插值/上采 latent | 是，属于 latent sampling / upsampling 范畴 | 否，K/V/V-output 都保持 native；只增加 sub-query probes |
| 核心机制 | spatial mixed-resolution latent sampling | query-foveated attention / sub-token RoPE probing |

可以把二者的共同点概括为：

```text
都认为：全图同等分辨率/同等计算不是最优，重要 ROI 应该获得更多计算。
```

但不能说它们是同一个方法。RALU 是 **latent-resolution scheduling**；B-2 是 **attention-query supersampling**。

---

## 9. 创新性分析

### 9.1 和已有 region-adaptive / foveated 方法的区别

已有方法大多在 **token 数/分辨率本身** 上做 mixed-resolution：

- RALU 通过 region-adaptive latent upsampling 在不同 denoising 阶段切换局部 full-res 区域，并需要 noise-timestep rescheduling 稳定分辨率转换。
- Foveated Diffusion 根据 foveal mask 非均匀分配 token density，本质是混合分辨率 token 生成。
- HierEdit 走 region-aware hierarchical refinement，用局部窗口模型 refine 高分区域。

本方案不同：它不改变最终 token 网格，不做输出后处理，也不把 ROI 作为新高分图 crop 后单独跑；它只在 attention 内部增加 **query probe density**，用子查询的 RoPE 相位变化来提高匹配细粒度。这更接近“attention 内 foveation”，而不是“latent/image resolution foveation”。

### 9.2 和 CRPA / mixed-resolution RoPE 的关系

CRPA 指出 mixed-resolution DiT 的关键问题是 RoPE phase aliasing：不同分辨率网格混在一个 attention 里时，线性坐标插值会让同一物理距离对应不同相位增量，造成 blur/artifact。

Q-only supersampling 避免了旧 B 最危险的部分：不插值 V、不产生 P×P latent、不再下采样 latent output。它只让 Q 以子 token 位置去探测 native K/V；这为后续实现 CRPA-style query-stride RoPE 留出了清晰接口。

### 9.3 对图像编辑 / 合影超分的潜在贡献点

如果实验成立，创新点可以概括为：

> **Training-free, query-foveated attention for identity-preserving multi-reference face refinement.**  
> 在不裁剪重编码参考图、不改变输出分辨率、不训练模型的条件下，通过 ROI 内子查询采样与分支式 lq/ref attention，提高小脸 ID 细节迁移的匹配精度。

它的研究价值不在于一定超过 crop→1k，而在于清楚回答一个关键问题：

> 在固定 native 输出网格下，清晰度提升到底受限于“attention 匹配不够细”，还是受限于“没有真高频源 / 输出 token 容量不够”？

这个问题本身有论文价值，因为它能把多参考编辑中的 ROI attention、mixed-resolution DiT、foveated generation、training-free inference surgery 连接起来。

---

## 10. 本次 patch 的边界

- patch helper 已推送，但我没有在 NPU 环境实际运行；必须先 `py_compile` + `ROI_DEBUG=1` 首跑。
- B-2 是下一步验证代码，不保证一定提升清晰度；它的价值是避免已证伪的插值低通路径。
- 若目标是直接出效果，`docs/10` 的 ref_hr 真高清重编码更可能有效；若目标是坚持“不 crop ref concat”的研究问题，B-2 更适合作为下一步实验。
