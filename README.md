# flux2-id-roi-attention

FLUX.2 多参考 refiner / 合影超分场景下的 **training-free 实验**：验证「扩大同 ID 局部 attention 的 ROI、再写回原 bbox」能否在不训练的情况下提升同 ID 细节恢复质量。

本仓库含 **Version A（Expanded-KV Only）** 与 **Version A'（Noise-Fixup）** 的代码改动 + 实验文档 + 实测发现。不训练、不改权重，只在推理时改 attention。

> **当前结论(读 `docs/05`)**:Version A 在 wide 合影小脸上对输出脸**几乎无可见作用**,且效果埋在 NPU run-to-run 数值噪声(maxdiff~34)之下。已实现 **A'**(直接 steer 输出 noise 段,`[局部lq结构 + 扩大ref细节]` 残差注入)作为下一步,实验顺序见 `docs/06`。

## 这是什么

现有 pipeline 已经在特定层做「同 ID bbox cross-attention」：lq 的人脸区域只 attend 匹配到的那个 ref bbox。**Version A 的唯一改动**：做这步时把被 attend 的一侧 bbox 从 `B_i` 扩大到 `E_i`（用已 apply RoPE 的现有 token，零 PE 风险），attention 输出仍只写回原始 `B_i`。

主旋钮是 **`expand_ratio_ref`**（lq 对 ref 的可见范围）——见 `docs/01` 对「为什么是它」的推导。

## 目录

```
README.md
CHANGES_version_a.md          # ★ 精确改动清单（Version A + A'），合入真实仓按这个
code/
  transformer_flux2.py        # 改好的整份（含 A + A'，合入前请 diff）
  refine_model.py             # 改好的整份（含 A + A'）
docs/
  01_background_and_mechanism.md   # 现有机制 + 坐标系确认 + 关键认知校正
  02_version_a_design.md           # Version A 设计 + 正确性核对步骤
  03_experiment_plan.md            # Version A 实验步骤 + 矩阵 + cfg
  04_analysis_points.md            # 评价指标 + 决策树 + 后续点
  05_findings_and_verification.md  # ★ 实测发现 + 非确定性坑 + 检验纪律（必读）
  06_version_a_prime_design.md     # ★ A'(noise fix-up) 设计 + 下一版实验顺序
  07_parameter_reference.md        # ★ 所有参数含义 + Phase 1 配置逐行解释
  08_version_b_plan.md             # ★ Version B(提分辨率/细节容量) 实验计划
```

> `Dit_pipeline.py` 文件大、改动仅 3 行，未整份重放——补丁见 `CHANGES_version_a.md` 第 3 节。
> `code/` 里的整份文件是基于贴出的片段重建的，**合入真实仓前务必 diff**，确认改动只发生在 `CHANGES` 列出的位置。

## 安全性

所有改动在默认参数 `(expand_ratio_*=1.0, expand_min_size=0)` 下与现有实现**逐 bit 等价**，可安全合入；只有显式设置 expand 参数才会改变行为。

## 快速上手

1. 读 `docs/01`（看准 baseline，避免对机制的误解）。
2. 按 `CHANGES_version_a.md` 合入，跑 `docs/02` 的「坐标系核对 + baseline 对齐验证」。
3. 按 `docs/03` 的实验矩阵跑 B0 / B1 / K2 / K3 / K4。
4. 按 `docs/04` 的指标和决策树分析。

## 核心假设 & 成功判据

> sample_ratio≈2、source（ref）扩得比 target（lq）更大时，ID bbox 内 ArcFace 相似度 / 清晰度显著优于现有 baseline，且非目标区域不退化、不串脸。

成立 ⇒ training-free 结论：扩大被 attend 的 ref ROI 支撑、写回原 bbox，可在不训练下增强同 ID 细节恢复。

## 范围 / 进展

- **Version A（Expanded-KV Only）**：已实现并实测 → 小脸上无可见作用、被噪声淹没（`docs/05`）。
- **Version A'（Noise-Fixup）**：已实现并实测 → 输出脸**机械上响应了 ref**(结果与 baseline 有别)，但**清晰度未提升**。结论:路径有效，瓶颈在小脸的细节源/分辨率容量。
- **Version B（Virtual ROI-QKV · 单趟 · attention 内提分辨率 · 无 crop/无后处理）**：当前主攻方向。修正诊断:ref native token **确实带高频**(长焦小脸比 lq 清晰),A′ 没变清晰是因为 ~10 token 的注意力**粒度太粗**搬不动细节;crop+1k 证明 token 数上去后高频能迁移。解法:attention 里把人脸 ROI 用 ROIAlign 升到 P×P 虚拟 token、高密度下 attend、降采样回 native、残差融合。单趟、不 crop、不重编码、零后处理。phase/PE/风险见 `docs/08`。
- **多 ID 泄漏 mitigation**：已登记后续点(每脸独立 ROI 天然隔离)。
