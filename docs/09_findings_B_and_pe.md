# 09 · B 系列实测发现 + PE 排除 + 单趟下采上限

> 记录 Version B(Virtual ROI-QKV)的实测结论、为什么 persist 把脸搞糊、PE 是否是元凶、以及单趟无后处理的物理上限。结论:**插值类 ROI(per-layer / persist)对清晰度无效甚至变糊;不是 PE 问题;真细节必须靠 A(重编码 ref 像素)。**

## 实测结果
| 实验 | 配置要点 | 结果 |
|---|---|---|
| B-1 per-layer | ROIAlign 升 P×P→attend→降回,`P=24` | 与 baseline 无可见清晰度差异 |
| B-1.5 persist(3 影子) | 全程保持 P×P,末端降采回写,`P=48, α=0.6, max_faces=1` | **单脸完全模糊**(比 baseline 还糊)|
| crop→1k(对照/上限) | lq+ref 脸 crop→resize 1024 跑 refiner | **大幅变清晰** |

## persist 为什么把脸搞糊(不是 bug,是机制必然)
1. **插值上采**:影子 = native 脸(~10 token)bilinear 上采到 48×48 —— 低通,无新高频。
2. **插值下采**:collapse 再 bilinear 降回 ~10 token —— 又一次低通。
3. **强混合**:`α=0.6` 把"两次低通"的更糊版本占 60% 写回 noise 脸,**冲掉模型原有高频**。
4. **重复 token(OOD)**:影子与 native 脸**同位置**,模型没见过同位置重复 token,演化易跑偏。

→ 插值不产生细节 + 两次重采样反而抹掉细节 + 大比例混回 = 越混越糊。

## PE 不是元凶(用 crop→1k 反证)
- 用户提出的验证:"孤立这张脸的 noise/lq/ref 互相 attend + 不下采样,若不糊则非 PE"。
- **这恰恰就是 crop→1k**:孤立(crop 只含该脸)、高密度(1k=64×64 token)、不下采样(64×64 解码)、标准 PE → **结果清晰**。
- 按该判据:**不糊 → 非 PE**。标准网格 + 普通 RoPE 是锐的,所以 persist 的糊来自 in-pass 机制(下采+稀释+插值+单趟无跨步合成),**不是 PE**。
- 残留疑点:crop→1k 用**真编码 token + 标准 PE**,影子用**插值 token + 自定义 pe2**。零成本检查:扫 `roi_pe_mode ∈ {pe1,pe2,pe3}`,若同样糊 → 确认非 PE;若 pe1 明显更好 → 自定义 PE 有害。**A 用真编码 token(+可选 pe1)贴近已证清晰的 crop,PE 安全。**

## 单趟无后处理的物理上限(关键)
- **"不下采样"在固定输出网格里 = 在高分辨率解码这张脸 = 一个 crop 输出(=后处理)。**
- 所以"单趟+不后处理"**必须**把脸下采回 native ~10 token 网格 → **必然低通** → 清晰度注定不如 crop→1k。
- crop→1k 之所以猛:模型在 64×64 上跑**整条去噪轨迹**、逐步**合成**细节;in-pass 影子只在**一次前向的 block 内**高密度、且每步从 native 重新插值 → 无法跨步合成。

## 结论 & 下一步
1. **停掉插值类 ROI**(per-layer/persist 对清晰度无效或变糊)。`roi_persist` 默认关。
2. **PE 已排除**,不用再单独写"孤立 attend"验证(crop→1k 已验证)。
3. **上 A**:noise 脸 query **保持 native**(不上采→不引入低通糊)去 attend **从 ref 像素 crop 重编码的真·高清 token**(genuine 高频),残差注入。单趟、输出无后处理、native 尺寸下能做到 crisp(128px 级),这是单趟的上限。
4. A 需要 `Dit_pipeline.dit_infer` 做 crop+VAE 编码 ref 脸 → 作为附加高清 latent 段传入。
