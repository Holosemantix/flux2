# 08 · Version B 实验计划（区域高分辨率精修 + 合成回原图）

## 决定性发现（改变了 Version B 的方向）
- **实测:把同 ID 的 lq 脸和 ref 脸 crop 出来、resize 到 1024 跑 refiner,清晰度大幅提升。**
- 这同时验证了两件事:
  1. 瓶颈确实是**分辨率 / 细节容量**(脸占满画面 → 1024/16=64×64≈4096 token,有足够容量);
  2. refiner 在容量充足时**能**恢复出清晰的脸。

## 为什么"序列内 attention 注入(ROIAlign + scatter 回原生 token)"给不出清晰脸
最终图是把 **noise latent 解码**出来的。一张脸只占 **~10 个原生 noise token**。
- 无论 attention/guidance 多好,**输出脸是从这 ~10 token 解码的 → 天生糊**;
- "ROIAlign 到 P×P 做 attention,再 scatter 回 ~10 个原生 token"——**输出表征仍是 ~10 token,解码出来还是糊**。

> 结论:要清晰,**脸必须在高分辨率下被表征并解码**。光改 attention 不够。这就是为什么要走 crop→高分辨率精修→贴回。

---

## 主线:Region High-Res Refine + Composite（复用已验证可行的 refiner）

流程:
1. 检测 + 匹配(已有 YOLO+ReID)。
2. 对每个 ID:从 **lq** crop 脸(带 margin)+ 从 **ref** crop 同 ID 脸 → 各 resize 到 ~1024。
3. 跑**现有 refiner**:lq-crop 为输入、ref-crop 为参考 → 得到**高分辨率清晰脸**。
4. **合成回原图**:把清晰脸 resize 到该脸在全图中的像素尺寸 → 边界羽化 / 金字塔融合 → 颜色/光照对齐 → 贴回。
5. 多脸批量处理。

新代码主要在 **2/4/5(编排 + 合成)**,refiner 本身不改 —— 比 attention 方案更可靠、且已被你的实验证明能出清晰脸。

---

## Phase 顺序（下一阶段就按这个测）

### Phase B-1：锁定单脸精修配方（先不合成）
- 变量:crop margin(脸框外扩 1.3/1.6/2.0)、resize 目标(768/1024/1280)、长宽比处理(直接拉伸 vs pad 成方形)。
- ref:同 ID ref 脸 crop 作参考。
- 指标:单脸 1024 输出的 **ArcFace + Laplacian/清晰度**,量化"大幅提升"到底多大、最佳配方是什么。
- 产出:最佳单脸精修配方。

### Phase B-2：合成回原图（真正的难点——接缝/颜色）
- 把精修脸 resize 到全图中该脸的像素尺寸,贴回。
- 测融合:**羽化 alpha / Laplacian 金字塔 / Poisson**,消接缝。
- **颜色/光照对齐**:复用现有 colorfix(WaveletRecon)把精修脸的色彩匹配到周边;否则脸会和全图色调脱节。
- 指标:接缝可见度、色彩一致性、无双边/鬼影;贴回后 ArcFace 不掉。

### Phase B-3：作为第二趟集成进全流程
- Pass 1:现有全图 refine → 全局一致的底图。
- Pass 2:检测+匹配 → 每脸 crop + ref crop → 1024 精修 → 合成回 Pass 1 结果。
- **关键 ablation**:Pass 2 的 lq-crop 取自**原始 lq** 还是 **Pass 1 输出**?(Pass 1 已超分,用它 crop 再带 ref 精修可能更稳)——两种都测。
- 指标:全图质量、脸清晰、背景不变、无接缝。

### Phase B-4：鲁棒性 / 效率 / 边界情况
- 多脸(批量一次 forward)、相邻脸重叠、极小脸(放大到 1024 后能否撑住)、ref 与 lq 姿态/表情差异大(ref 帮忙还是帮倒忙 → 需要限制 ref 影响强度)、遮挡脸。
- 效率:所有脸 crop 拼 batch 一次跑。
- ref 影响强度旋钮:姿态差大时调低。

### Phase B-5（备选,仅当合成始终有接缝）:序列内高分辨率人脸 latent
- 给脸一个**专用高分辨率 latent 段**,和全图一起去噪、但**单独解码**再贴。比 crop-paste 复杂,只有当 B-2 的合成怎么都消不掉接缝时才考虑。

---

## 评估（沿用 docs/05 纪律）
1. 先测噪声底;2. 脸 crop 比;3. **ArcFace + Laplacian/清晰度**,对照 baseline / A′ / 直接 crop-1024(上限参考);4. 全图层面看接缝/颜色/背景。
- 单脸精修 B-1 的清晰度应接近"直接 crop-1024"的上限;B-2/B-3 的目标是**把这个清晰度无损地搬回全图、不留接缝**。

## 与之前 A / A′ 的关系
- A / A′ 是序列内 attention 方案,**受限于输出 token 分辨率,给不出清晰脸**(已被实验证实)。保留作对照/baseline。
- Version B 走**区域高分辨率精修 + 合成**,直接利用"脸占满画面就清晰"这一已验证事实。**这是当前主攻方向。**

> 下一步:先做 **Phase B-1**(单脸精修配方,量化清晰度上限与最佳 crop/resize),再啃 **B-2 合成**。B-2 的接缝/颜色对齐是成败关键。配方定了我可以帮你写 crop→batch refine→composite 的编排代码。
