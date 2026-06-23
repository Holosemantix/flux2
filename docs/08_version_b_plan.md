# 08 · Version B 实验计划（Virtual ROI-QKV · 单趟 · attention 内提分辨率 · 无 crop/无后处理）

## 目标约束
- **单趟前向、attention 内完成、不 crop ref、不重编码、输出零后处理。**
- 用 ref **native token 里已有的高频**,只是在 attention 时把人脸 ROI 的 token 数升上去,让高频能被细粒度迁移到 lq/noise 脸。

## 诊断修正（之前错了）
- ❌ 旧说法:"ref native 脸 token 没有高频源"。
- ✅ 正确:推理时 lq/ref 都是 ~3k,**ref 的小人脸类似长焦,native latent token 里确实带高频,比 lq 脸清晰**。
- A′ 没变清晰的真正原因:**注意力粒度太粗**。脸只有 ~10 token,10↔10 的 query-key 对应太糙,搬不动细节。
- 证据:把两张脸 crop 出来 resize 到 ~1k(**token 数量大增**)后清晰度大幅提升 → 说明 **token 数上去,ref 已有的高频就能迁移到 lq**。
- → 结论:**不需要 crop/重编码 ref,只要在 attention 时把 ROI 的 token 升密度即可。**

## 核心做法:Virtual ROI-QKV
对每个匹配 ID,在选定层的 attention 里:
1. 把 **noise 脸 ROI**(native ~10 token)用 ROIAlign 升采样到固定 **P×P** 虚拟 query;
2. 把 **ref 脸 ROI**(native,带高频)+ 可选 **lq 脸 ROI**(结构)同样 ROIAlign 到 **P×P** 虚拟 key/value;
3. 在 **P×P 高密度**下做 attention(此时对应足够细,ref 高频能搬过去);
4. 把输出 **inverse-ROIAlign / bilinear 降采样回 native ~10 个 noise 脸 token**,`noise_alpha` 残差融合;
5. 输出仍是原生 noise latent 正常解码,**单趟、无 crop、无贴回**。

与 A′ 的关系:同样是 noise 段残差注入,区别是**把 attention 升到 P×P 高密度**(虚拟 token),而不是在 native ~10 token 上做。

## 位置编码（虚拟 token 必须有 PE）
P×P 虚拟 token 要重新给位置 id 并 apply RoPE:
- **PE-1** 连续真实坐标;
- **PE-2** 压回原脸框坐标范围("放大后缩回");
- **PE-3** ref→target 脸坐标对齐(姿态差大时)。
先 **PE-2** 起步。

## 唯一要盯住的经验性风险（设计实验时直接测,不预设结论）
ROIAlign 升采样 latent 是**插值**(带限),scatter 回 native 又是**降采样**——**要验证高频在"升采样→attention→降采样"这一来回里是否真的被搬进了 native noise token**,而不是被插值/降采样抹平。你的 crop+1k 证明了"token 多→能迁移",但那是**像素重编码**带来的高密度;latent ROIAlign 是否等效,**Phase B-1 直接用清晰度指标判**。若被抹平,退路是从更高密度特征 ROIAlign(后议),但**先按你的方向测 native ROIAlign**。

---

## Phase 顺序（下一阶段）

### Phase B-1：Virtual ROI-QKV 主实验
- noise 脸 + ref 脸(+lq 脸)ROIAlign 到 **P×P**,P×P 下 attend,降采样回 native,`noise_alpha=0.6` 残差,PE 用 **PE-2**。
- 扫 **P ∈ {16, 24, 32}**(越大对应越细、越贵、越可能 OOD)。
- 判读:脸 crop + **ArcFace + Laplacian/清晰度**,对照 **baseline / A′ / 直接 crop-1k(上限参考)**。清晰度↑ = 方向成立。

### Phase B-2：PE 消融
- PE-1 / PE-2 / PE-3 比较;姿态差大试 PE-3。

### Phase B-3：组成与强度
- KV 是否带 lq 结构 ROI(`[lq P×P + ref P×P]` vs 仅 `ref P×P`);`noise_alpha ∈ {0.4,0.6,0.8}`;ref/lq 的 ROI 扩大 r_s/r_t(ROIAlign 取多大原始区域)。

### Phase B-4：层 / 多 ID / 效率
- 在哪些层做 Virtual ROI-QKV(沿用 best layer set,先少层);多脸独立 ROI、天然隔离;ROIAlign 批量。

---

## 评估（沿用 docs/05 纪律）
1. 先测噪声底;2. 脸 crop 比;3. **ArcFace + Laplacian/清晰度**,对照 baseline / A′ / **crop-1k(上限)**;4. 全图看身份/串脸/背景。

## 实现提示（比 A′ 多的：ROIAlign 虚拟 token + PE + 降采样回写）
- 在 `_id_patch_attention`(或其上游)拿到 noise/lq/ref 段的 2D 网格布局(`latent_h × latent_w` 已知),对各脸 bbox 区域 reshape 成 2D 特征图 → ROIAlign 到 P×P。
- **RoPE 处理**:对虚拟 token **重新分配位置 id 并 apply RoPE**(PE-1/2/3);为干净起见,ROIAlign 最好作用在 **apply RoPE 之前**的 q/k/v 上,再对虚拟 token 加 RoPE。
- attend 后 inverse-ROIAlign(bilinear)回 native 脸 token 数,`(1-α)·全注意力 + α·此结果` 写回 noise 脸。
- 全程单趟,输出零后处理。

> 这条完全贴合"在 attention 里扩大 ROI、不要后处理"。你定一下 **P(先 24?)和 PE(先 PE-2?)**,我来写 Virtual ROI-QKV 的完整实现(ROIAlign + 虚拟 token PE + 降采样回写,复用 A′ 的残差融合)。
