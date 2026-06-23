# 05 · 实验发现 & 检验结论（Version A 阶段）

> 这份记录 Version A（Expanded-KV Only）在真实 refiner 上跑出来的结论,以及**踩过的检验坑**。
> 一句话:**Version A 在当前小脸合影上对输出脸几乎没有可见作用,而且效果埋在 NPU 的 run-to-run 数值噪声之下,根本测不出来。** 这直接推动了 A′（noise 段 fix-up）的设计。

---

## 1. 实验配置（务必和后续对齐）

| 项 | 值 |
|---|---|
| 模型 | klein 4b refiner,`checkpoint-17600`,**bf16,未量化**(`refiner_convrot_quant=False`) |
| 采样 | `step=28`,`FlowMatchEulerDiscreteScheduler`,`patch_split_num=1`(单卡 no_split) |
| 测试集 | `heying_test_ouput_org`(**wide 合影**),例 `case10_wide_center` |
| ID 匹配 | 人脸 YOLO `face_yolov8m-seg_60.pt` + ReID `resnet18`;`conf=0.15`,`dist_threshold=0.4`,`imgsz=2512→2528` |
| 匹配结果 | 单图检出并匹配 **12 对**人脸 |
| ID attention 生效层 | `idx_double_window=[1,3,5,7,9,11,13,15,17,19]`、`idx_single_window=[1,3]` |
| 扩大参数 | A 测了 `K4(r_t=1.5, r_s=2.0)`、`K3(r_t=1.5, r_s=2.5)`,`min_size=0` |

⚠️ **配置坑 1（生效层越界）**:模型只有 `num_layers=8` 个 double block(下标 0–7)和 `num_single_layers=48` 个 single block。`idx_double_window` 里的 9/11/…/19 **超出 double block 范围,不生效**。实际生效的是 double `{1,3,5,7}` + single `{1,3}` = **仅 6 层**。后续若想加强,要在合法范围里加层。

---

## 2. 人脸大小 & 扩大范围（关键尺度）

- 出图分辨率:**3584 × 2816 px**;latent token 网格:**224 × 176**(`seq=39424`,已验证 `seq_lq == latent_h*latent_w` 坐标对齐)。1 个 latent token = **16×16 px**。
- **人脸大小(token)**:wide 合影里每张脸只有 **~8–12 token 高、~6–12 token 宽**,即每脸约 **40–110 token**;换算像素约 **130–190 px**,在 3584px 宽的整图里极小。
- **扩大范围**:`r_t=1.5`(lq/结构)、`r_s=2.0`(ref/细节)。实测 `_expand_bbox` 例子:
  - ref `(131,123,141,130)`(70 tok)→ `(126,120,146,134)`(~280 tok),面积约 ×4;
  - 但**绝对量仍极小**:280 token / 39424 ≈ **0.7%**;在 lq query 的 KV 里(见下)占比 **~0.35%**。

---

## 3. 量化结果（K3/K4 vs B1）

整图像素差(`case10_wide_center`):

| 对比 | maxdiff | meandiff | 变化像素 | 最大差异位置 | 人脸 |
|---|---|---|---|---|---|
| K3(r_s=2.5) vs B1 | 48 | 0.36 | ~2.88M(≈28.6%) | **地砖** (3386,1236) | **未变化** |
| **B1 vs B1（同配置跑两遍）** | **34** | **0.36** | ~2.91M | **同一块地砖** | 未变化 |

---

## 4. 检验结论（⚠️ 当初没料到、必须记住的部分）

### 4.1 流程是**非确定性**的 —— 这是头号坑
同一配置跑两遍,maxdiff 34、弥散全图、最大点还在同一块地砖。**这和 K3-vs-B1 的 48 是同一量级。**

> 结论:**NPU + bf16 + `npu_fusion_attention` 不是逐 bit 可复现的。** run-to-run 噪声底 ≈ 34。Version A 的真实效果 ≤ 这个噪声底,**用整图 pixel diff 根本测不出来**。之前几轮"结果一模一样"的本质,就是参数效果淹没在数值噪声里。

### 4.2 由此沉淀的检验纪律（后续每个版本都按这个走）
1. **先测噪声底**:任何 A/B 前,**同一配置跑两遍求 maxdiff/meandiff**,确定 run-to-run 噪声底。低于这个底的差异一律不可信。
2. **整图 pixel diff 不可用**于弱效果;**改动必须强到明显超过噪声底**才有意义(A′ 的 α 残差注入就是为此)。
3. **看对地方**:效果在小脸上,整图缩略图看不见。**crop 到单张脸**对比,不要看全图。
4. **用对指标**:对脸 crop 算 **ArcFace 余弦相似度 / 人脸 IQA / Laplacian 清晰度**,对逐像素噪声远比 MSE 鲁棒。整图 MSE 在非确定流程下没意义。
5. **先确认机制在跑**:`npairs>0`、`_id_patch_attention` 有 `ENTER` 打印、扩大后 bbox 确实变大(本轮已确认:r_ref=2.0、框 70→280)。
6. **确认改的是被 import 的文件**:本项目实际 import 的是 `.../Pangu_I2I_Klein_patch_old/common/diffusers_klein/.../transformer_flux2.py`(本地可编辑版),不是 site-packages。
7. **配置自检**:active 层下标不能越界(见配置坑 1)。

### 4.3 为什么 Version A 对脸几乎无效（根因三条）
1. **稀释**:lq 人脸 query 的 KV = `[txt + 整张noise(39424) + 整张lq(39424) + ref_patch(280)]`。ref_patch 仅占 ~0.35%,softmax 几乎不分给它;70→280 是零头。
2. **没作用在输出上**:fix-up 只改 lq/ref(条件)段,**输出是 noise 段,只能间接受影响**。
3. **脸太小**:8–12 token 容量本就极低,Version A 只是"多选几个已有低分辨率 token",**不提高采样密度**,搬不进高频细节。

观察到的现象(差异弥散、集中在地砖等高频纹理、人脸不动)正是"微小扰动经扩散过程混沌传播 + 被数值噪声淹没"的样子,**不是定向的人脸细节迁移**。

---

## 5. 对下一版的指向
- 要让**输出脸**真正响应 ref → 必须直接作用在 **noise(输出) query** 上 = **A′**(见 `06_version_a_prime_design.md`),并用紧凑 base 去稀释、用 α 残差注入把效果顶到噪声底之上。
- 小脸的细节容量天花板 → 最终大概率需要 **Version B（ROIAlign 上采样 ROI）**。
- 评估改用 **脸 crop + ArcFace**,并永远先测噪声底。
