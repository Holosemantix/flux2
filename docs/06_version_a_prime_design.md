# 06 · Version A'（Noise-Fixup）设计 & 下一版实验顺序

## 为什么需要 A'
Version A 只改 lq/ref(条件)段,输出 noise 段只能间接受影响,且被整图 base 稀释 → 对脸几乎无效(见 `05`)。
**A' 直接作用在 noise(输出) query 上**,这才是让"输出脸"响应 ref 的路径。

## 核心思想:lq 给结构,ref 给细节
lq 和 ref 是同一 ID 但**表情/视角可能不同**,所以不能像素级搬运:
- **结构/姿态/表情** ← lq(noise 与 lq 空间对齐,本就该跟 lq);
- **身份细节/纹理/清晰度** ← ref(高质量),靠**内容匹配软对齐**,容忍姿态差异。

## 机制（已实现在 `_id_patch_attention`）
对每个 ID,noise 段人脸 query(bbox 与 lq 相同)attend 一个**紧凑** KV:

```
noise_face_query  →  K/V = [ 局部lq(扩大 r_t, 结构) ⊕ ref(扩大 r_s, 细节) ]      # 不放整图，避免稀释
output[noise_face] = (1 - noise_alpha) · 全注意力输出  +  noise_alpha · 上面的ID注意力输出
```

- 不放"整张 lq+noise"进 base → ref 终于拿得到注意力权重(解决稀释);
- 残差融合 `noise_alpha`:保住全局一致性,同时把 ID 细节顶到噪声底之上;
- query 只取原始 bbox、只写回原始 bbox → 不污染邻居(合影里脸挨得近)。

## 配置（cfg → 全链路已接通）

```yaml
# A' 开关与强度
id_patch_fixup_lqref: false     # 单独测 A' 时关掉原 lq/ref fixup（true=A 与 A' 叠加）
id_patch_fixup_noise: true      # 开启 noise 段 fix-up
id_patch_noise_alpha: 1.0       # 残差注入强度；Phase 1 先用 1.0 做 go/no-go
# 扩大（沿用）
id_patch_expand_ratio_lq: 1.5   # r_t，结构侧
id_patch_expand_ratio_ref: 2.0  # r_s，细节侧
id_patch_expand_min_size: 0
# 生效层（沿用，注意 double 只有 0~7 有效）
id_patch_idx_double_window: [1, 3, 5, 7]
id_patch_idx_single_window: [1, 3]
```

`fixup_noise=false` 时与现有行为完全一致;改动安全。

---

## 下一版实验顺序（严格按此走，每步只动一个变量）

> 评估方式统一:**先 B1×2 测噪声底 → 对脸 crop 看 → 算 ArcFace 余弦(输出脸 vs ref 脸)+ Laplacian 清晰度**。整图 pixel diff 作废。每组都和 **B0(无ID)/B1(Version A)** 对照。

### Phase 0 — 噪声底（已做,~34）
同配置跑两遍确认 run-to-run maxdiff。换图换分辨率要重测。

### Phase 1 — A' go/no-go（最关键的一步）
- 配置:`fixup_lqref=false, fixup_noise=true, noise_alpha=1.0, r_t=1.5, r_s=2.0`。
- α=1.0 = 输出脸几乎完全由 [lq结构+ref细节] 注意力决定,**效果必然远超噪声底**。
- 判读(看脸 crop):
  - 脸明显变化(更像 ref/更清晰,哪怕过头/有伪影)→ **机制 work**,进 Phase 2;
  - 脸仍几乎不变 → 注意力没把 ref 用起来(可能姿态差异大软对齐失败 / ref 权重不足)→ 跳 Phase 4 的诊断,或直接上 Version B。

### Phase 2 — α 调强度（质量甜点）
- `noise_alpha ∈ {0.3, 0.5, 0.7, 1.0}`,其余固定。
- 找"注入了 ref 细节、又不破坏结构/不串脸/无伪影"的 α。人脸 SR 一般 0.4~0.6。

### Phase 3 — 细节源扩大范围
- `r_s ∈ {2.0, 2.5, 3.0}`(ref/细节)× `r_t ∈ {1.0, 1.5}`(lq/结构),固定最优 α。
- 看 ArcFace↑ 和清晰度↑ 的同时,outside 不退化、不串脸。小脸可加 `expand_min_size ∈ {12,16}`。

### Phase 4 — 结构/细节组成 & 诊断
- A1:KV=`[lq + ref]`(当前默认);
- A2:KV=`[ref only]`(去掉 lq 锚,测纯细节注入是否更强或更乱);
- A3:`fixup_lqref=true, fixup_noise=true`(A 与 A' 叠加)。
- 若 Phase 1 卡住:在这里诊断姿态对齐——是否需要给 ref 加正 bias,或上 **Source-to-Target Canonical PE**(把 ref ROI 映射到 target 脸坐标系,属 Version B 的 PE 部分)。

### Phase 5 — 大脸验证 & 小脸转 Version B
- 找一张**大脸图**(单人/近景,脸占很多 token)跑 A',确认机制在容量充足时确实迁移细节;
- 你这批 **wide 合影小脸(8–12 token)** 容量天花板低,大概率需要 **Version B(ROIAlign 把脸 ROI 上采样到 P=16/24/32 → 局部 attention → scatter 回)** 才能真正加细节。Phase 1–4 验证清楚机制后再上 B。

---

## 需要补到 Dit_pipeline 的透传（A' 新增 3 行）
`load_modules()` 的 `if self.use_id_patch:` 块里追加:
```python
dit_params['id_patch_fixup_lqref'] = self.cfg.get('id_patch_fixup_lqref', True)
dit_params['id_patch_fixup_noise'] = self.cfg.get('id_patch_fixup_noise', False)
dit_params['id_patch_noise_alpha'] = self.cfg.get('id_patch_noise_alpha', 0.5)
```
详见 `../CHANGES_version_a.md` 的 A' 小节。
