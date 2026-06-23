# 07 · 参数说明 & Phase 1 配置详解

> 这份把 ID-ROI attention 的**所有旋钮**讲清楚:每个参数是什么、作用在哪、调大调小有什么后果、彼此怎么配合;最后给 **Phase 1 的逐行配置解释**。
> 配置流:`yaml 的 Dit: 段` → `Dit_pipeline.load_modules` 透传 → `RefinerModel.__init__` 装进 `IdPatchConfig` → transformer 各 block 的 processor → `_id_patch_attention`。

---

## 0. 先理解机制里有哪些"段"和"框"

- 输入序列分 4 段:`txt`(文本)、`noise`(被去噪的**输出**)、`lq`(待修复图,条件)、`ref`(参考图,条件)。三个图像段**空间分辨率相同**,token 网格 = `(H//16, W//16)`,1 token = 16×16 像素。
- `noise` 与 `lq` **空间对齐**(同一网格,只是位置编码的 stream id 不同),所以**人脸 bbox 在 noise 段和 lq 段是同一个**。
- 每个匹配到的 ID 有一对人脸框:`lq` 框(=noise 框)和 `ref` 框,坐标是 `(y1,x1,y2,x2)`、token 单位。
- 「扩大」= 把框按比例放大、再 clip 到边界,选中**更多已存在的 token**(它们带各自真实的 RoPE,零 PE 改动)。

---

## 1. 全部参数一览

| yaml key（`Dit:` 下） | 类型 | 默认 | 作用 |
|---|---|---|---|
| `use_id_patch_attention` | bool | false | ID-ROI attention 总开关 |
| `id_patch_idx_double_window` | list[int] | [] | double block 中启用的层下标(**只有 0~7 有效**,模型 8 个 double block) |
| `id_patch_idx_single_window` | list[int] | [] | single block 中启用的层下标(0~47 有效) |
| `id_patch_expand_ratio_lq` | float | 1.0 | **r_t**:lq 框的扩大倍数(结构侧) |
| `id_patch_expand_ratio_ref` | float | 1.0 | **r_s**:ref 框的扩大倍数(细节侧) |
| `id_patch_expand_min_size` | int | 0 | 扩大后框的最小边长(token 数),对小脸兜底;0=关 |
| `id_patch_fixup_lqref` | bool | true | 是否保留 **Version A**(lq/ref 段 fix-up) |
| `id_patch_fixup_noise` | bool | false | 是否开 **Version A'**(noise 段 fix-up) |
| `id_patch_noise_alpha` | float | 0.5 | A' 的残差注入强度 α |

> 还有 ID 匹配相关的 `id_match_conf_threshold / id_match_dist_threshold / id_match_imgsz` 和 YOLO/ReID 路径,不在本文范围,保持你现有值即可。

---

## 2. 新参数详解（重点:同一个 ratio 在 A 和 A' 里含义不同）

### 2.0 关键澄清:scale 的是 query 还是 KV?和"分辨率"什么关系?

**`expand_ratio_*` 只放大被 attend 的 KV(=多选一些已存在的 token);query 和写回区永远是原始人脸框;token 的密度/分辨率不变,只是空间范围变大。**

| 参数 | Version A（`fixup_lqref`）| Version A'（`fixup_noise`）|
|---|---|---|
| `expand_ratio_ref`(r_s) | **ref 段 KV** 放大 → 被 **lq query** 看到 | **ref 段 KV(细节)** 放大 → 被 **noise query** 看到 |
| `expand_ratio_lq`(r_t) | **lq 段 KV** 放大 → 被 **ref query** 看到 | **lq 段 KV(结构)** 放大 → 被 **noise query** 看到 |
| **query / 写回区** | **永远原始框,从不放大** | **永远原始框(noise 人脸),从不放大** |

两个要点:

- ⚠️ 这里的"扩大" = **取更大的空间范围、选更多原有 token(密度不变)**,**不是把脸上采样到更高分辨率**。所以对 8~12 token 的小脸,它**没有增加细节容量** —— 这正是 Version A 对小脸近乎无效的根因之一。

- 「scale up 之后再 rescale 回原大小」有两种含义,别混:
  1. **空间范围意义**:现在就是这样 —— KV 取大范围,但 query/输出仍在原始框,**结果本来就写回原始大小**,不需要额外 rescale。
  2. **分辨率意义(才是对小脸有用的)= Version B**:用 ROIAlign 把人脸 ROI **重采样到固定 P×P(如 32×32)更高密度** → 在高分辨率下做 attention(此时脸有足够 token 承载细节) → 再**下采样 / scatter 回原来的 8~12 token** 写回。这才是真正的"放大分辨率 → attend → 缩回原大小"。
     - 代价:要给 P×P 这些**虚拟 token 安排位置编码**(连续坐标 PE-1 / **把扩大区压回原框坐标范围 = PE-2** / source→target 对齐 PE-3)→ 不再是零 PE 风险,属于 Version B。
     - 你说的"scale up 再 rescale 回去",在位置编码上正对应 **PE-2(Compressed PE)**:采样范围放大,但 PE 压回原框范围,让模型仍把这些 token 当作"服务于这张脸"。

### 2.1 `expand_ratio_lq`（r_t，结构侧）
框中心不变,宽高 × r_t,再 clip。`r_t=1.0` 即不扩。

它在两处被用到,**含义随版本不同**:
- **Version A（`fixup_lqref=true`）**:用于 **ref 段 query 的 fix-up** —— ref 人脸去 attend「扩大后的 lq 区域」。即 r_t 控制 ref 能看到多少 lq。
- **Version A'（`fixup_noise=true`）**:用于 **noise query 的结构来源** —— noise 人脸去 attend「扩大后的局部 lq 区域」拿结构。r_t 越大 = 给输出脸**更多周边结构上下文**(下巴/发际线位置/脖子衔接)。

调大:结构上下文更全,但太大会把别的脸/背景也算进结构。人脸建议 **1.0~1.5**。

### 2.2 `expand_ratio_ref`（r_s，细节侧，**主旋钮**）
同样是中心不变缩放。它在两处:
- **Version A**:lq 人脸去 attend「扩大后的 ref 区域」。r_s 控制 lq 能看到多少 ref。
- **Version A'**:noise 人脸去 attend「扩大后的 ref 区域」拿**细节**。r_s 越大 = 引入更多 ref 周边(发际线、脸型边缘、局部光照、纹理)。

调大:细节来源更丰富;但太大可能把**相邻另一个人的脸**也圈进去(合影里脸挨得近)→ 串脸风险。人脸建议 **2.0~2.5**,最多 3.0。

### 2.3 `expand_min_size`（小脸兜底，token）
扩大后边长 = `max(原边长 × ratio, min_size)`。当脸只有 8~12 token 时,即使 r_s=2 也才 ~20 token;设 `min_size=16` 能保证 ref 细节区至少 16×16 token。0=关。小脸场景试 **12 / 16**。

### 2.4 `fixup_lqref`（开关:Version A）
- `true`(默认):保留原始机制 —— lq 人脸 query 只 attend 匹配的 ref 框、ref 人脸 query 只 attend 匹配的 lq 框(改的是**条件段**,对输出只有间接、且被稀释的影响,实测近乎不可见)。
- `false`:关掉它,用于**单独测 A'**,让结论干净。

### 2.5 `fixup_noise`（开关:Version A'）
- `false`(默认):不动 noise 段,行为=Version A。
- `true`:对 **noise(输出) 人脸 query** 做 fix-up —— 它去 attend 一个**紧凑 KV** `[局部lq(结构, r_t) + 扩大ref(细节, r_s)]`(**不放整张图,避免把 ref 稀释掉**),再残差写回。这是**让输出脸真正响应 ref 的关键路径**。

### 2.6 `noise_alpha`（A' 的注入强度 α）
```
output[noise人脸] = (1 - α) · 全注意力输出  +  α · ID注意力输出
```
- `α=0`:无效果(=不开 A')。
- `α=1.0`:输出脸几乎**完全**由 [lq结构+ref细节] 注意力决定 → 效果最强(go/no-go 用)。
- 人脸质量甜点一般 **0.4~0.6**:既注入 ref 细节,又靠 (1-α) 的全注意力保住全局一致性、防接缝。
- α 太大:可能过锐、伪影、身份漂移;太小:看不出变化(尤其在非确定性噪声底之下)。

### 2.7 lq=结构 / ref=细节 的设计逻辑
lq 和 ref 是同一人但**表情/视角可能不同**,不能像素搬运。noise 与 lq 同位置 → 天然对 lq 高权重 = **拿正确的姿态/表情结构**;ref 在不同位置 → 靠**内容匹配软对齐**(QK 相似度自动找对应五官)= **注入细节**,从而容忍姿态差异。所以 r_t 管「结构上下文范围」,r_s 管「细节来源范围」,α 管「细节注入多少」。

---

## 3. Phase 1 配置详解（A' 的 go/no-go）

目的:用**最强**设置先回答"A' 到底能不能让输出脸动起来",而不是先纠结质量。

```yaml
Dit:
  use_id_patch_attention: true
  patch_split_num: 1                       # id patch 只在单卡 no_split 下生效

  id_patch_idx_double_window: [1, 3, 5, 7] # 只用合法层（double 仅 0~7）
  id_patch_idx_single_window: [1, 3]

  id_patch_fixup_lqref: false              # 关掉 Version A，单独测 A'，结论干净
  id_patch_fixup_noise: true               # 开 Version A'
  id_patch_noise_alpha: 1.0                # 最强注入：输出脸几乎完全由 ID 注意力决定

  id_patch_expand_ratio_lq: 1.5            # 结构：局部 lq 区域 = 1.5× 脸框
  id_patch_expand_ratio_ref: 2.0           # 细节：ref 区域 = 2.0× 脸框
  id_patch_expand_min_size: 0              # 先关，Phase 3 再试 12/16
```

**逐行为什么这么设:**

| 参数 | 值 | 原因 |
|---|---|---|
| `fixup_lqref` | **false** | Version A 已证明对脸无效;留着只会往结论里掺噪声。单独开 A' 才能干净判断 |
| `fixup_noise` | **true** | 启用 A'，这是本阶段要验证的东西 |
| `noise_alpha` | **1.0** | go/no-go:α 最大时若脸还不动,说明机制没用,不必再试小 α;若动了(哪怕过头/伪影)就说明机制 work，Phase 2 再调小 |
| `expand_ratio_lq` | **1.5** | 给输出脸一点周边结构上下文,又不至于圈进别人 |
| `expand_ratio_ref` | **2.0** | 引入 ref 发际线/脸型边缘等细节,串脸风险可控 |
| `expand_min_size` | **0** | 先不兜底,隔离变量;小脸增益留到 Phase 3 |
| 生效层 | `double[1,3,5,7]+single[1,3]` | 沿用已验证层,且去掉越界的 9~19 |

**怎么判读(评估方式,务必照做):**
1. **先测噪声底**:同一份 Phase 1 配置**跑两遍**,看脸 crop 的 maxdiff(确定 run-to-run 噪声底;NPU+bf16 不可复现)。
2. **看脸 crop,不看整图**:脸只有 ~150px,整图缩略看不见。
3. **算指标**:对脸 crop 算 **ArcFace 余弦(输出脸 vs ref 脸)** + Laplacian 清晰度,和 **B0(无ID)/B1(Version A)** 对照。
4. 判定:
   - 脸明显变化(更像 ref / 更清晰,且超过噪声底)→ **机制 work** → 进 Phase 2 调 α。
   - 脸几乎不变 → A' 也没把 ref 用起来 → 进 Phase 4 诊断(给 ref 加 bias / canonical PE)或直接上 Version B(小脸容量不足)。

---

## 4. 各 Phase 的配置速查

| Phase | fixup_lqref | fixup_noise | noise_alpha | r_t | r_s | min_size | 备注 |
|---|---|---|---|---|---|---|---|
| B0 基线 | — | — | — | — | — | — | `use_id_patch_attention:false` |
| B1（Version A） | true | false | — | 1.0 | 1.0 | 0 | 原始 ID-patch |
| **P1 go/no-go** | false | true | 1.0 | 1.5 | 2.0 | 0 | A' 最强 |
| P2 α 甜点 | false | true | 0.3/0.5/0.7 | 1.5 | 2.0 | 0 | 固定 r,扫 α |
| P3 细节范围 | false | true | 最优α | 1.0/1.5 | 2.0/2.5/3.0 | 0→16 | 扫扩大 |
| P4 组成 | false/true | true | 最优α | … | … | … | A' vs A+A';KV 去 lq 锚 |
| P5 | — | — | — | … | … | … | 大脸验证 + 小脸转 Version B |

> 改任何一组都只动表里一个变量;层/step/colorfix/seed 全锁死。
