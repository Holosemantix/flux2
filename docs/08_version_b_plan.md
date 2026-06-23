# 08 · Version B 实验计划（提高人脸分辨率 / 细节容量）

## 为什么需要 Version B（基于 A′ 的结果）
- **A′ 已证明 noise-fixup 路径机械上生效**:输出脸结果与 baseline 有区别 → 输出 query 确实 attend 了 ref 并被注入。
- **但清晰度没提升。** 根因:小脸只有 8~12 token。无论怎么 attend、attend 多大范围,参与的 ref 也是 ~10 token 的**低分辨率 latent**,里面**没有真正的高频细节可注入**;而且 10×10 的 attention 对应太粗,对不准五官。
- **Version A/A′ 只改"attend 的路径/范围",不改"分辨率/细节容量"。** 要提清晰度,必须:① 引入**真正更高分辨率的细节源**;② 让脸在**更高 token 密度**下做 attention;③ 再缩回原生脸 token 写回。这就是 Version B。

---

## ⚠️ Phase B-0:前提核查（几乎无代码，先做，省得白干）

清晰度的前提是**真有细节可搬**。先确认两件事:

**(a) ref 脸到底比 lq 脸清晰吗?**
把匹配到的 lq 脸和 ref 脸 crop 出来、放大到同尺寸并排看 + 算 Laplacian 方差。
- 若 ref 脸**并不比 lq 脸更清晰**(合影里 ref 往往也是小脸)→ **没有细节源,Version B 也救不了**,瓶颈在数据。需要换更高质量的 ref。
- 若 ref 脸确实更清晰 → 细节源存在,继续。

**(b) 最便宜的概念验证(零核心代码):喂高清 ref 脸 crop 给现有 A′。**
找一张**单人/单 ID** 测试图,把 ref **裁成脸部紧 crop 并放大**(让脸占满画面 → VAE 编码后脸占很多 token),作为 ref 喂进**现有 A′** 跑一遍。
- 若清晰度**明显提升** → "高分辨率 ref 细节"就是那个杠杆,值得做完整 Version B。
- 若仍不提升 → 瓶颈不在分辨率(可能模型在该 ROI 上无法合成细节 / OOD),Version B 性价比要重估。

> B-0 用最小代价回答"做 Version B 到底有没有希望",务必先做。

---

## 两条技术路线（区别:细节从哪来）

| 路线 | 做法 | 能否加清晰度 | 代价 |
|---|---|---|---|
| **B-interp** | ROIAlign 把脸 ROI 的 **latent** 上采样到 P×P | ✗ 多半不能(插值不含新高频),只让对应更细 | 低,PE 中等 |
| **B-reencode** | 把 ref 脸在**像素空间** crop → 单独 VAE 编码(脸填满画面 → 真·高细节 P×P token) | ✓ 这才是能加清晰度的 | pipeline 改动 + PE |

**主攻 B-reencode**;B-interp 仅作对照(验证"光提对应粒度够不够")。

---

## Phase 顺序

### Phase B-1：B-reencode 最小版（主攻清晰度）
- 对每个匹配 ID,用 ref 的(扩大)脸框在**像素空间** crop → resize 到能产出 **P×P latent** 的尺寸(P=24/32)→ VAE 编码 → **高细节 ref-face token**。
- 作为附加 KV 注入:**noise 脸 query**(先用原生脸 token)attend `[局部 lq(结构) ⊕ 高清 ref-face token(细节)]`,`noise_alpha` 残差写回原生脸 token。
- PE:先用 **PE-2**(把高清 ref-face token 的位置压回原 ref 脸框坐标范围,让模型当它"服务这张脸")。
- 判读:脸 crop + **ArcFace + Laplacian/清晰度 IQA**,对比 baseline / A′。**清晰度↑ 且身份不漂** = Version B 成立。

### Phase B-2：query 也上采样（高分辨率对应）
- 若 B-1 还不够锐:把 noise 脸 query 也 ROIAlign 到 **P×P** → 在高分辨率下 attend → **inverse-ROIAlign / bilinear scatter 回原生脸 token** → α 融合。
- 让脸在高分辨率下做 attention,五官对应更细。

### Phase B-3：PE 消融（虚拟 token 的位置编码）
- 对 P×P 虚拟 token 试三种:
  - **PE-1** 连续真实坐标;
  - **PE-2** 压回原框范围(你说的"放大后缩回"在 PE 上的对应);
  - **PE-3** ref→target 脸坐标对齐(把 ref 脸映射到 target 脸的坐标系,姿态差异大时最该试)。
- 选最稳、对齐最好、最不 OOD 的。

### Phase B-4：P / α / 扩大范围 / 多 ID
- `P ∈ {16, 24, 32}`(越大越细,但越贵、越 OOD);
- `noise_alpha ∈ {0.4, 0.6, 0.8}`;
- ref 扩大 `r_s ∈ {2.0, 2.5}`(控制 crop 进多少周边);
- 多 ID:每张脸**独立 crop+编码**,天然隔离,基本无串脸风险。

---

## 评估（沿用 docs/05 的检验纪律）
1. 先同配置跑两遍测**噪声底**;
2. **脸 crop** 比,不看整图;
3. 指标:**ArcFace(输出脸 vs ref 脸) + Laplacian/清晰度**,对照 **baseline / A′**;
4. 判定:清晰度↑、身份不漂、不串脸、outside 不退化 = Version B 成立。

---

## 实现提示（B-reencode 数据流，改动比 A/A′ 大）
1. 在 `Dit_pipeline._match_ids` 之后、进 transformer 前:对每个 ID 用 ref 脸框 crop 像素 → resize → `KleinVAEProcessor.encode` 得到 P×P 高清 face latent。
2. 把这些 face latent 作为**附加 condition 段**拼进序列(类似 ref,但是 zoom 的),位置 id 走 `KleinLatentProcessor.prepare_image_ids` 的**额外 stream**(如 T=30;PE-2/PE-3 则自定义坐标)。
3. 在 `_id_patch_attention` 里把这些 token 当 **detail KV** 喂给对应 ID 的 noise 脸 query(复用 A′ 的残差注入)。
4. 因为动了数据流 + PE,**强烈建议先做 B-0 核查、再 B-1 验证**,确认有希望后再投入 B-2/B-3。

> 等你 B-0 跑完(尤其 (b) 的高清 ref crop 概念验证),告诉我结果,我来写 B-1 的完整实现(crop+encode 注入 + PE-2 + 残差融合)。
