# 04 · 需要分析的点 & 判断逻辑

## 评价指标

### A. ID bbox 内（最重要）
合影通常没有 GT，所以以「身份一致性 + 清晰度」为主：

- **ArcFace 余弦相似度**：输出人脸 vs 参考图同 ID 人脸（越高越好）——核心指标。
- **人脸清晰度**：bbox 内 Laplacian variance / 高频能量（越高越锐，但要防过锐伪影）。
- **关键点稳定性**：landmark 抖动 / 五官是否变形。
- **人工主观**：发际线、脸型边缘、皮肤纹理、牙齿/眼睛细节，并排盲评。
- 若个别图有 GT/高清版：bbox 内 PSNR / SSIM / LPIPS / DISTS。

### B. 非目标区域 no-regression（必须看）
扩大 ROI 不能污染别处。检查 `Ω_outside = 全图 − ∪ B_i^t`：

- outside 区域 LPIPS / MSE（vs B1 输出）应基本不变；
- **其他人的脸有没有被带歪 / 串脸**（多 ID 场景重点）；
- 背景、色彩有没有漂移（注意 colorfix 在最后一步，别把 colorfix 的效果误判成 attention 的）。

### C. attention 诊断（可解释性，NPU 上需额外路径）
理想指标（方案 §14.3）：

```
A_correct = mean_attn(B_i^t → E_i^s)              # 关注对的 ref
A_leak    = Σ_{j≠i} mean_attn(B_i^t → E_j^s)      # 关注错的 ref
R_bind    = A_correct / (A_correct + A_leak + ε)
```

- **若视觉变好且 R_bind 升高 → 机制成立**（不是巧合）。
- 注意：NPU 走 `npu_fusion_attention` 拿不到 attention 权重。要算 R_bind，需在少量样本上临时切到 eager softmax 路径单独跑。属于独立工具，不阻塞主实验。

## 判断逻辑（决策树）

跑完 B0 / B1 / K2 / K3 / K4 后：

| 观察 | 结论 | 下一步 |
|------|------|--------|
| B1 > B0 | ID attention 有用（已知）| 继续 |
| **K* > B1**（尤其 K3/K4）| **扩大 KV 有额外增益，核心假设成立** | 精扫 r_s 甜点，固化为默认；再考虑 A′（noise 段）|
| K3 > K2 | source 扩大比对称扩大好 | 主打「source 扩得更大、target 中等」|
| K* ≈ B1（无差别）| 扩大无感 | 可能 bbox 已够大 / ref 周边信息无用 → 转去试 min_size（小脸）或直接上 A′ |
| K* < B1（变差）| 扩大引入了污染 | 八成是**多 ID 泄漏**（E_i^s 盖到邻居）或背景串入 → 提前做泄漏 mitigation |
| outside 退化明显 | 写回/索引有 bug，或泄漏严重 | 先查写回范围是否真的只在 B_i^t；再查多 ID 重叠 |

## 归因纪律

- 一次只动一个变量（expand 参数）。层、step、colorfix、seed 全锁死。
- 每个结论都要有 B1 对照；没有 B1 对照的「变好」不算数。
- 先确认 ID 匹配正确（看 `id_match_debug/`）再分析 attention 效果。

## 已知后续点（A 出结果后排期）

1. **A′：noise 段也做 fix-up**。当前输出（noise）只间接受益；让 noise 的 person-i tokens（bbox 同 lq）直接 attend 扩大后的 ref，预期更直接有效。这是 A 之后最该做的一步。
2. **多 ID 泄漏 mitigation**。扩大 `E_i^s` 时把其他 ID 的原始 bbox 从中挖掉（mask），或限制扩大不越过邻居中心。做成旋钮 `exclude_other_ids`。
3. **版本 B：Virtual ROI-QKV**。ROIAlign 把 E 采样到固定 P×P 当临时 KV/QKV，配 Compressed PE / Source-to-Target Canonical PE。只有 A（甚至 A′）证明「扩大有用」后再上，因为它有真实 PE 风险。

## 一句话的成功判据

> **K3 / K4（sample_ratio≈2、source 扩得更大）在 ID bbox 内 ArcFace 相似度 / 清晰度显著优于 B1，同时 outside 不退化、不串脸。**
> 成立即得到 training-free 结论：局部参考超分中，扩大被 attend 的 ref ROI 支撑、写回原 bbox，可在不训练下增强同 ID 细节恢复。
