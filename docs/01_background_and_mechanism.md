# 01 · 背景与现有机制（先把 baseline 看准）

## 目标

在 **FLUX.2 多参考 refiner / 合影超分** 场景下，验证一个 **training-free**（不训练、不改权重、只改推理时 attention）的假设：

> 让同一个人物 ID 的「待修复图 bbox」和「参考图 bbox」做局部 cross-attention 已经能提升同 ID 超分质量；
> 如果在做 attention 时把被 attend 的一侧 bbox **人为扩大**（B → E），attention 完成后**只写回原始 bbox**，质量能否进一步提升。

这是整个实验的最小验证单元，对应方案里的 **Expanded-KV Only（版本 A）**。

## 模型输入（三路，固定不变）

DiT 的输入在 channel 维拼成 `[noise, lq, ref]`，VAE 编码后每段 128 通道：

| 段 | 含义 | 位置编码 T（frame/stream id） |
|----|------|------|
| `noise` | 被去噪的目标 latent（最终输出就是它）| T=0 |
| `lq` | 待修复图 / 低质量合影 | T=10 |
| `ref` | 高质量参考图 | T=20 |

- VAE：`vae.encode` 下采样 8× → `patchify_latents` 2×2 → **总 16×**；`pack_latents` 是纯 flatten（无额外空间压缩）。
- 所以 token 网格 = `(H_pixel//16, W_pixel//16)`，且 **`seq_lq == seq_ref == seq_noise == latent_h * latent_w`**。
- 位置编码是 4D `(T, H, W, L)`，RoPE `axes_dims_rope=(32,32,32,32)`；三段的 `(H,W)` 用真实空间坐标，靠 `T` 区分 stream。

## ID 匹配（已有）

`Dit_pipeline._match_ids()`：YOLO(person) 检测 lq/ref 的人 → ResNet18 ReID 特征 → 匈牙利匹配 → 阈值过滤，得到：

- `id_patch_pairs`：每对 `{'lq': (y1,x1,y2,x2), 'ref': (y1,x1,y2,x2)}`，**token 网格坐标**，`scale = vae_scale_factor = 2**len(block_out_channels) = 16`，与 token 网格逐格对齐。
- `id_patch_pairs_pixel`：像素坐标 + 匹配距离，仅用于 debug 可视化。

## 现有 ID Patch Attention 的真实行为（关键，别和直觉搞混）

`transformer_flux2._id_patch_attention()` 在 active 层（`idx_single_window` / `idx_double_window`）做的是：

1. **Step 1**：全序列 `[txt, noise, lq, ref]` 先做一遍 full attention，得到 `output`。
2. **Step 3 逐对覆盖写回**（只改 lq/ref 的 ID 区域 query 的输出）：
   - **lq 的 ID query**（`B_i^t`）→ attend 到 `[txt, noise, 整张 lq, 仅 B_i^s]`
   - **ref 的 ID query**（`B_i^s`）→ attend 到 `[txt, noise, 整张 ref, 仅 B_i^t]`

注意两点容易误解的地方：

- **lq 的人脸 query 仍然看得到「整张 lq」+ txt + noise**（全局上下文保留），它被限制的只是「对 ref 的可见范围」——只能看到匹配到的那一个 ref bbox，看不到 ref 里别的人/背景。这其实**已经实现了 wrong-ID hard block**。
- **`noise` 段（最终输出）不被 fix-up**，只走 full attention。ID 机制是通过「改善 lq/ref 表征 → 后续层 noise 全注意力读到更好的 lq」**间接**作用于输出。

### 由此推出的核心结论（决定 Version A 怎么做）

因为 `base_for_lq` 已经包含**整张 lq**，对 lq query 而言「扩大 target bbox」是 no-op。
**真正能影响目标人脸修复质量的唯一旋钮，是扩大 lq query 对 ref 的可见范围 `B_i^s → E_i^s`（即 `expand_ratio_ref`）。**
`expand_ratio_lq` 只影响 ref 段表征（ref query 看到的 lq patch），是次要项。

## 为什么 Version A 是零 PE 风险

`_id_patch_attention` 拿到的 `query/key/value` 是**已经 apply 过 RoPE** 的（processor 里先 `apply_rotary_emb` 再路由）。扩大 bbox 只是**多选了一些本来就存在的 token**，它们带的是自己真实的 `(T,H,W)` 位置。没有新建 token、没有改坐标、没有插值——这就是方案里的 **PE-0**。

因此 Version A 的唯一变量就是「被 attend 的 KV 集合大小」，干净。

## 与方案版本的对应

| 方案 | 本实验 |
|------|--------|
| 版本 A：Expanded-KV Only | ✅ 本次要做的 |
| 版本 B：Virtual ROI-QKV（ROIAlign 采样到 P×P + 自适应 PE）| 后续，视 A 结果再上 |
| PE-0 / PE-1 / PE-2 / PE-3 ... | A 只用 PE-0；其余 PE 策略属于版本 B |
| §7.2 wrong-ID hard block | 现有代码已内建（lq 只见匹配 ref bbox）|
| noise 段也做 fix-up | 记为 **A′**，A 出结果后再做 |
| 多 ID 泄漏（扩大后盖到邻近他人）| 记为后续挖掘点，A 暂不处理 |
