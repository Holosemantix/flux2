# 08 · Version B 实验计划（单趟 · attention 内高分辨率 ref 注入 · 无后处理）

## 目标约束（明确）
- **单趟前向,attention 内完成,输出不做任何 crop/贴回/后处理。**
- 通过在 attention 里给人脸 ROI 接入**真正的高分辨率细节源**来提升清晰度。

## 先把 A′ 没变清晰的真正原因说准
不是"输出从 ~80 token 解码"导致糊(VAE 从 80 token 重建一张 160px 的脸其实足够锐)。
**真正原因:序列里没有"高频来源"。** 全图 pass 里,ref 脸和 lq 脸**都是 ~80 个低分辨率 token**,ref 脸本身不比 lq 脸清晰 → noise 脸 query 再怎么 attend ref,**没有高频可搬**。
对比:crop→1024 之所以清晰,是那一刻脸有 ~4096 个**真·高细节** token。

→ 结论:**单趟方案能行,但必须把"细节源"换成高分辨率的 —— 用高清 ref 脸 crop 重编码出来的 token。**

## 核心做法
1. **输入/条件侧准备(不是输出后处理)**:对每个匹配 ID,把 **ref 脸**在像素空间 crop(带 margin)→ resize 到高分辨率(如 512/1024)→ `KleinVAEProcessor.encode` → 得到**高细节 ref-face token**(如 32×32 / 64×64)。这与"喂 ref 图"同类,属 conditioning。
2. **作为额外条件段拼进序列**(类似 ref,但是 zoom 的高清版),给它一段位置编码。
3. **attention 内注入(复用 A′)**:noise 脸 query(原生 bbox)attend `[局部 lq(结构) ⊕ 高清 ref-face token(细节)]`,`noise_alpha` 残差写回原生 noise 脸 token。
4. **输出**:原生 noise latent 正常解码,**无 crop/贴回/融合**。

> 这正是"在 attention 里扩大 ROI"——只是 ROI 的细节源用**高分辨率重编码 token**,而不是原生低分辨率 token。和 A′ 的唯一区别:**细节源从"native ref 脸(~80 低清 token)"换成"重编码 ref 脸 crop(高清 token)"**。

## 位置编码（新 token 必须有 PE）
高清 ref-face token 是新造的虚拟 token,要给位置 id:
- **PE-1** 连续真实坐标(各自网格);
- **PE-2** 压回原 ref 脸框坐标范围(你说的"放大后缩回",让模型把它当作"服务这张脸");
- **PE-3** 映射到 target(noise/lq)脸坐标系(姿态差异大时最该试)。
先用 **PE-2** 起步。

## 一个必须诚实交代的上限
输出脸在全图里仍是它的**原生像素尺寸**(脸占多大就多大),attention 注入能把它**锐化到 VAE 在该尺寸下的上限**(通常能比现在明显锐,因为终于有高频源了),但**不能让脸在输出里占据比网格更多的像素**。
- 若"原生尺寸下更锐"满足你 → 单趟方案达标。
- 若你需要脸在输出里**更大更锐**(超过原生尺寸)→ 那必须整体提高输出分辨率,单趟 attention 给不了(这是网格的硬约束,不是方法问题)。

---

## Phase 顺序（下一阶段就按这个）

### Phase B-1：高清 ref 注入 + A′ 注入（主实验）
- 对每个 ID:ref 脸 crop(margin 1.5×)→ resize 512 → 编码成高清 token。
- noise 脸 query attend `[局部 lq + 高清 ref token]`,`noise_alpha=0.6` 起步,PE 用 **PE-2**。
- 扫:**ref crop 分辨率 ∈ {256, 512, 1024}**(→ token 数 16²/32²/64²,越高细节源越强、越贵、越可能 OOD)。
- 判读:脸 crop + ArcFace + Laplacian,对照 baseline / A′ / 直接 crop-1024(上限参考)。**清晰度↑ = 单趟方案成立。**

### Phase B-2：PE 消融
- PE-1 / PE-2 / PE-3 比较;姿态差大时 PE-3(ref→target 对齐)是否更好、是否更稳不 OOD。

### Phase B-3：强度与范围
- `noise_alpha ∈ {0.4, 0.6, 0.8}`;ref crop margin / 分辨率;lq 结构侧 r_t。
- 看清晰度↑ 同时身份不漂、不串脸。

### Phase B-4（可选）：query 侧高分辨率对应
- 若 B-1 仍欠锐:把 noise 脸 query 也 ROIAlign 到 P×P 做 attention(对应更细),attend 完 bilinear 回原生 token。注意这**不改输出分辨率**(仍解码原生 token),只可能让注入更精准,属边际改进。

### Phase B-5：效率 / 多 ID
- 多脸的 ref crop 批量编码;每脸独立 token 段,天然隔离,基本无串脸。

---

## 评估（沿用 docs/05 纪律）
1. 先测噪声底;2. 脸 crop 比;3. **ArcFace + Laplacian/清晰度**,对照 **baseline / A′ / crop-1024(上限)**;4. 全图看身份/串脸/背景。

## 实现提示（比 A′ 多的就是"额外高清条件段 + PE"）
- 在 `Dit_pipeline.dit_infer` 里(匹配后、编码阶段):每个 ID crop+resize+`KleinVAEProcessor.encode` 出高清 face latent;pack 成 token,拼到 `ref_model_input` 之后形成新段。
- 位置 id:`prepare_image_ids` 给新段一个 stream(PE-1);PE-2/PE-3 自定义坐标。
- `id_patch_pairs` 里给每个 ID 多带一个"高清 ref-face token 的索引范围",`_id_patch_attention` 的 noise fixup 把 detail KV 从"native ref bbox"换成"这段高清 token"。
- 全程单趟,输出零后处理。

> 这条和 A′ 同源(都是 noise 段残差注入),改动集中在"加一段高清 ref 条件 + 给它 PE + detail KV 指过去"。配方/PE 选定后我来写完整实现。
