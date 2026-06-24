# 12 · 为什么 sub-query RoPE 里有 T_noise，以及 split-branch 融合如何自适应

> 本文补充 `docs/11_q_supersample_next_experiment_and_innovation.md` 中两个容易误解的点：
> 1. 为什么 Q-only supersampling 的 sub-query 位置写成 `(T_noise, H, W, L)`，其中 `T_noise` 到底是什么；
> 2. split-branch 中固定 `detail_beta` 是否会不适配不同图片，应该如何改成更稳的自适应融合。

---

## 1. `T_noise` 不是 diffusion timestep，而是 stream/frame id

FLUX.2 refiner 当前输入是三段 image tokens：

```text
[noise, lq, ref]
```

每个 image token 的 RoPE 位置 id 是 4D：

```text
(T, H, W, L)
```

其中：

| 段 | 含义 | T |
|---|---|---|
| `noise` | 当前要预测/更新的输出 latent，也就是最终会 decode 出来的目标图 | 0 |
| `lq` | 待修复 / 低质量合影条件图 | 10 |
| `ref` | 高质量参考图 | 20 |

所以 `T_noise` 里的 `T` 不是 denoising step，也不是 scheduler timestep。它是 RoPE 的第 1 个坐标轴，用来区分 token 属于哪一个 image stream。

换句话说：

```text
T_noise = 0   表示“这是输出/noise stream 的 token”
T_lq    = 10  表示“这是 lq 条件 stream 的 token”
T_ref   = 20  表示“这是 ref 条件 stream 的 token”
```

---

## 2. 为什么 sub-query 必须保留 `T_noise`

Q-only supersampling 里的 sub-query 是从 native noise face query 复制出来的：

```math
\tilde{q}_{i,a,b}^{raw} = q_i
```

这个 `q_i` 的内容来自 noise/output stream。它代表模型当前对输出脸 token 的预测状态。

因此，sub-query 的 4D 位置应该是：

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

这里保留 `T_noise` 的原因有三点。

### 2.1 保持 query 的 stream 身份不变

sub-query 仍然是输出/noise token 的 query，不是 lq token，也不是 ref token。我们只是让它在同一个 native token cell 内以多个子位置去“看”参考信息。

如果把它的 T 改成 `T_lq` 或 `T_ref`，就等价于告诉模型：这个 query 来自条件图 stream，而不是输出 stream。这会破坏模型原本学到的 stream 关系。

因此：

```text
内容来自 noise → T 必须仍然是 T_noise
```

### 2.2 只细化空间位置，不改变跨图关系

Q-only supersampling 的目标是提高 H/W 上的探测密度，而不是重新定义 token 来自哪张图。

也就是说，我们只想改：

```text
H, W：在一个 native token 内部放多个子位置
```

不想改：

```text
T：这个 token 属于 output/noise stream
L：patch 内部通道/局部轴，仍为 0
```

所以 sub-query 的变化只发生在 H/W：

```math
(y_i,x_i) \rightarrow
\left(
y_i + \frac{a+0.5}{m} - 0.5,
x_i + \frac{b+0.5}{m} - 0.5
\right)
```

T 仍然固定为 `T_noise`。

### 2.3 保持模型已经学到的 noise-lq/ref 相对相位

RoPE attention score 可以写成：

```math
A_{ij}
=
\frac{\langle \mathrm{RoPE}(q_i,p_i),\mathrm{RoPE}(k_j,p_j)\rangle}{\sqrt{d}}
```

其中 `p_i=(T_i,H_i,W_i,L_i)`，`p_j=(T_j,H_j,W_j,L_j)`。

对 noise query attend ref key 时，模型看到的是一种固定的跨 stream 关系：

```text
T_query = 0, T_key = 20
```

对 noise query attend lq key 时，是：

```text
T_query = 0, T_key = 10
```

这些 T 差值参与 RoPE 相位，模型已经在训练/推理机制中适应这种关系。B-2 如果把 sub-query 的 T 改掉，就不只是细化空间位置，而是改变了跨 stream 相位关系，会引入额外变量。

因此 B-2 的原则是：

```text
只改 H/W 子位置，不改 T stream id。
```

---

## 3. 那 ref 的 `pe3` 又是怎么回事？

`pe3` 不是把 ref 变成 noise，也不是把 ref 的 T 改成 `T_noise`。

更合理的设计是：

```text
ref key 的 T 仍然是 T_ref = 20
ref key 的 H/W 可以映射到 target/lq 脸坐标系
```

即：

```math
p_{ref}^{pe3}
=
\left(
T_{ref},
\Phi_{r\rightarrow t}(H_{ref}),
\Phi_{r\rightarrow t}(W_{ref}),
0
\right)
```

这样做的含义是：

- T 轴保留“这是 ref stream”的身份；
- H/W 轴表示“这个 ref 脸局部位置对应到 target 脸局部坐标中的哪里”。

这比把 ref 绝对坐标直接拿来更适合同 ID face transfer，因为 ref 人脸在图像中的绝对位置可能和 lq/target 完全不同。

---

## 4. split-branch 固定 `detail_beta` 的问题

你的担心是对的：

```math
s_{i,a,b}
= (1-\beta)s_{i,a,b}^{lq} + \beta s_{i,a,b}^{ref}
```

如果 `beta` 固定，不同图片、不同人脸、不同层、不同 denoising step 下最优值很可能不同。

例如：

- ref 和 lq 姿态很接近时，`beta` 可以更大，多用 ref 细节；
- ref 和 lq 姿态差很大时，`beta` 太大会导致表情/脸型漂移；
- 小脸很糊但 ref 很清晰时，需要更强 ref 分支；
- ID 匹配不确定或有遮挡时，需要更保守，更多依赖 lq。

所以固定 `detail_beta` 更适合作为 **Phase-1 消融旋钮**，不是最终方案。

---

## 5. 推荐实验策略：先固定扫参，再做自适应 gate

### 5.1 为什么第一阶段仍然保留固定 beta

固定 beta 的优点是可解释、可控：

```yaml
id_patch_roi_detail_beta: 0.3 / 0.5 / 0.7 / 1.0
```

它能回答一个基础问题：

```text
ref 分支到底有没有带来稳定增益？
```

如果固定 beta 下完全没有收益，就没必要先做复杂自适应；如果某些图在 beta=0.7 有提升、某些图在 beta=0.3 更稳，就说明自适应 gate 有价值。

### 5.2 第二阶段：confidence-aware beta

可以把 beta 改成 per-face 的自适应权重：

```math
\beta_f = \sigma(w_1 C_{id} + w_2 C_{sharp} + w_3 C_{attn} - b)
```

training-free 版本不训练 `w`，可以用规则近似：

```math
\beta_f
=
\mathrm{clip}
\left(
\beta_0
+ \lambda_1 C_{id}
+ \lambda_2 C_{sharp}
+ \lambda_3 C_{attn},
\beta_{min},
\beta_{max}
\right)
```

其中：

| 符号 | 含义 | 直觉 |
|---|---|---|
| `C_id` | lq/ref ID 匹配置信度，例如 `1 - ReID distance` | 越像同一个人，越敢用 ref |
| `C_sharp` | ref face 比 lq face 清晰多少，例如 Laplacian 差 | ref 越清楚，越值得用 ref |
| `C_attn` | ref 分支 attention 是否集中，例如 attention entropy 低 | ref 匹配越集中，越可信 |
| `beta_min/max` | 安全边界，例如 0.2/0.8 | 防止完全抛弃 lq 或完全照搬 ref |

一个很实用的默认规则：

```text
if ReID 距离小、ref 比 lq 清楚、ref attention 集中 → beta 提高
if ReID 距离大、姿态差大、attention 分散 → beta 降低
```

### 5.3 第三阶段：token/head/layer 级 gate

更细粒度可以从 per-face beta 升级为 per-token 或 per-head gate：

```math
g_{i,h}
=
\sigma(\gamma (c_{i,h}^{ref} - c_{i,h}^{lq}))
```

然后：

```math
s_{i,a,b,h}
= (1-g_{i,h})s_{i,a,b,h}^{lq} + g_{i,h}s_{i,a,b,h}^{ref}
```

这里 `c` 可以来自 attention max probability、entropy、或 ref/lq 分支输出范数。这个版本更灵活，但也更容易引入不稳定，所以不建议第一版就上。

---

## 6. 对当前代码的建议

当前 patch helper 里先实现了固定 `detail_beta`，这是为了最小可测。

建议后续新增配置：

```yaml
id_patch_roi_beta_mode: "fixed"        # fixed / face_confidence / attn_confidence
id_patch_roi_beta_min: 0.2
id_patch_roi_beta_max: 0.8
id_patch_roi_beta_base: 0.5
```

实验顺序：

1. 固定 beta 扫 `{0.3, 0.5, 0.7, 1.0}`；
2. 如果不同图片最优 beta 差异明显，再加 `face_confidence`；
3. 如果 face-level 仍不够，再做 token/head-level gate。

这样能避免一开始把自适应做复杂后分不清到底是哪一部分起作用。

---

## 7. 汇报时可以这样解释

> `T_noise` 是 FLUX.2 多图输入里的 stream id，不是 diffusion timestep。sub-query 来自输出 noise token，所以 T 必须保留为 noise stream；我们只在 H/W 上做子 token RoPE 偏移，让同一个 query 以多个细粒度位置去探测 lq/ref。  
> split branch 的固定 beta 只是第一阶段可解释消融。不同图像确实需要不同融合强度，所以后续会用 ReID 置信度、ref/lq 清晰度差、attention entropy 做自适应 beta 或 gate。
