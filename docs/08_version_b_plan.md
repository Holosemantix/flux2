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

## 关键设计轴:每层都下采 vs 跨层保持高分辨率（必须验证）
B-1 默认**每个 active 层都"上采→attend→下采回 native"**。隐患:上采是插值、下采是低通,**每层来回一次,高频反复被抹平、无法跨层累积**;脸在层间又回到 ~10 token,后一层拿不到前一层在高密度下长出来的细节。

**改进(你的提议):上采一次后,在连续多层里都保持 P×P 高分辨率,到指定层才下采回 native。**
- 让高频在高密度下**跨层累积/合成**(更接近 crop+1k:脸在网络里一直是高密度);
- **上采起始层 / 下采回写层做成可配置**:`roi_up_layer`(在此层把脸 token 升到 P×P 并开始保持)、`roi_down_layer`(在此层下采回 native)。中间这段脸都是 P×P。

代价/注意(比 B-1 的"层内旁路上下采"侵入性大):
- 跨层保持会**改变主序列长度**(脸的 ~10 token 临时替换成 P×P,多脸更长)→ 要处理跨层的序列长度、位置编码、回写映射;
- 其他 token 也会 attend 到这 P×P 脸 token(全局注意力变化),可能更好也可能更 OOD;
- 输出仍在 `roi_down_layer` 收回 native → **脸仍是原生尺寸**,但 native token 经多层高密度精修,能编码更锐的脸。

---

## Phase 顺序（下一阶段）

### Phase B-1：Virtual ROI-QKV 主实验
- noise 脸 + ref 脸(+lq 脸)ROIAlign 到 **P×P**,P×P 下 attend,降采样回 native,`noise_alpha=0.6` 残差,PE 用 **PE-2**。
- 扫 **P ∈ {16, 24, 32}**(越大对应越细、越贵、越可能 OOD)。
- 判读:脸 crop + **ArcFace + Laplacian/清晰度**,对照 **baseline / A′ / 直接 crop-1k(上限参考)**。清晰度↑ = 方向成立。

### Phase B-1.5：验证"跨层保持高分辨率 vs 每层下采"（你提的）
- 对比两种:
  - **V_perlayer**:每个 active 层都 上采→attend→下采(B-1 默认);
  - **V_persist**:在 `roi_up_layer` 上采一次,连续保持 P×P 到 `roi_down_layer` 才下采。
- 扫 `[roi_up_layer, roi_down_layer]` 跨度(短→长),`roi_down_layer` 在代码里可配。
- 若 **V_persist 明显更锐** → per-layer 下采确实在抹高频,后续以 persist 为主;
- 若两者接近 → per-layer 没损害,用更省的 B-1 即可。
- 这一步直接回答"下采到底影不影响"。

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

---

## ✅ 已实现（code/transformer_flux2.py + refine_model.py），下面是 B 系列实验参数

代码已落地(per-layer Virtual ROI-QKV)。`roi_mode=False` 时行为不变。合入需补 `Dit_pipeline` 的 7 行透传(见 `../CHANGES_version_a.md` 的 Version B 第 B-3 节)。

### cfg key（yaml `Dit:` 下）

| key | 默认 | 含义 |
|---|---|---|
| `id_patch_roi_mode` | false | **Version B 总开关**(开了就用虚拟 ROI 注入 noise,替代 native A′) |
| `id_patch_roi_size` | 24 | **P**,虚拟 ROI 边长(token);扫 {16,24,32} |
| `id_patch_roi_pe_mode` | "pe2" | 虚拟 token PE:`pe1`连续 / `pe2`压回原框 / `pe3`映射到 target 脸 |
| `id_patch_roi_include_lq` | true | KV 是否含 lq 结构 ROI(`[lq+ref]` vs 仅 `ref`) |
| `id_patch_noise_alpha` | 0.5 | 残差注入强度(沿用 A′ 的) |
| `id_patch_expand_ratio_ref` | 1.0→2.0 | ROIAlign 取多大 ref 区域(细节范围) |
| `id_patch_expand_ratio_lq` | 1.0→1.5 | ROIAlign 取多大 lq 区域(结构范围) |
| `id_patch_idx_double_window` / `_single_window` | — | 在哪些层做(沿用,double 仅 0~7) |
| `id_patch_roi_persist` | false | 跨层保持(**未接通**,置 true 仅告警回退 per-layer) |
| `id_patch_roi_up_layer` / `_down_layer` | -1 | persist 用(预留) |

### Phase B-1（主实验）cfg
```yaml
Dit:
  use_id_patch_attention: true
  patch_split_num: 1
  id_patch_idx_double_window: [1, 3, 5, 7]
  id_patch_idx_single_window: [1, 3]
  id_patch_fixup_lqref: false        # 隔离，单测 Version B 的 noise 注入
  id_patch_roi_mode: true            # 开 Version B
  id_patch_roi_size: 24              # P=24
  id_patch_roi_pe_mode: "pe2"
  id_patch_roi_include_lq: true
  id_patch_noise_alpha: 0.6
  id_patch_expand_ratio_ref: 2.0     # ROI 取 2× 脸框
  id_patch_expand_ratio_lq: 1.5
  id_patch_expand_min_size: 0
```

### 各 Phase 扫的参数
| Phase | 变量 | 取值 |
|---|---|---|
| B-1 | `roi_size` P | {16, 24, 32} |
| B-2 | `roi_pe_mode` | pe1 / pe2 / pe3 |
| B-3 | `noise_alpha` / `roi_include_lq` / `expand_ratio_ref` | α∈{0.4,0.6,0.8} / {true,false} / r_s∈{2.0,2.5} |
| B-4 | 生效层 / 多 ID | 加/减层;多脸自动批 |
| B-1.5 | `roi_persist`(待接通) | up/down layer 跨度 |

### 首跑务必
- `ROI_DEBUG=1` 看 `[roi]` 行的 `virt_q/virt_k` 形状是否 `[B, P², Hh, D]`、noise_face 尺寸是否合理。
- 先测噪声底(同配置跑两遍),再对照 baseline / A′ / **crop-1k(上限)** 用脸 crop + ArcFace/Laplacian 判定清晰度。
- 本机无 torch,代码仅过 py_compile;NPU 上若 `F.interpolate(bf16)` 或 `apply_rotary_emb` 广播报错,看 `ROI_DEBUG` 定位后告诉我。
