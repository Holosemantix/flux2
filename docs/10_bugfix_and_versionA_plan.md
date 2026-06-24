# 10 · 关键 bug 修复 + 当前结论 + Version A 实现计划

## 🔴 关键 bug：Dit_pipeline `load_modules` roi 参数尾逗号
`load_modules` 里 Version B 的 8 行 `dit_params['id_patch_roi_*'] = self.cfg.get(...),` **每行尾部带逗号** → Python 把值变成 **1-tuple**。后果:

| 参数 | 实际值 | 后果 |
|---|---|---|
| `roi_size` | `(24,)` | "tuple vs int" 报错的真正源头(`_as_int` 只治标) |
| `roi_max_faces` | `(1,)` / 默认 `(False,)` | 同上;默认值还误写成 `False`(应 -1) |
| `roi_pe_mode` | `('pe2',)` | `== "pe2"` 恒 False → **永远走 pe1**;pe2/pe3 从未生效 |
| `roi_mode`/`roi_persist`/`roi_include_lq` | `(True,)`/`(False,)` | 非空 tuple **恒为真** → yaml 设 false 也关不掉 |

**修复:删掉这 8 行尾逗号,`roi_max_faces` 默认改 -1。** 见 `../CHANGES_version_a.md` Bugfix 段。

> 影响:之前 persist 的糊是在 **pe1** 下测的(以为是 pe2)。但核心结论不受影响(见下)。

## 当前结论(B 系列实测 + 分析)
1. **per-layer Virtual ROI-QKV(P=24)**:对清晰度无可见增益。
2. **persist 3-shadow(P=48, pe1)**:输出脸**完全模糊**(比 baseline 还糊)。
   - 原因:影子是 native 脸**插值上采**(无新高频)→ collapse **插值下采**(再低通)→ `α=0.6` 把更糊版本占 60% 写回,冲掉原有高频;加上"同位置重复 token"OOD。
3. **PE 不是元凶**:`crop→1k` 本身就是"孤立+高密度+不下采+标准 PE"且**清晰** → 按"不糊=非 PE"判据,排除 PE。
4. **单趟无后处理的物理上限**:输出脸从 native ~8–12 token(≈128–192px,token=16px)解码;"不下采样"= 在高分辨率解码该脸 = crop 输出(后处理)。所以单趟+不后处理**必须**下采回 native → 低通 → 注定不如 crop→1k。
5. **crop→1k 为何猛**:脸被**真实像素重编码**成 64×64 真·高细节 token,且模型在 64×64 上跑**整条去噪轨迹**逐步合成细节。插值的影子两者都没有。

→ **插值类 ROI(per-layer / persist)是死路**。唯一能在单趟内加真细节的,是给序列一个**真·高频源** = 把 ref 脸像素重编码成高密度 token = **Version A**。

## Version A 设计(真·高清 ref 重编码,单趟,无输出后处理)
核心:**noise 脸 query 保持 native(不上采、不重采样回写 → 不引入低通糊)**,去 attend **从 ref 像素 crop 重编码的真·高清 token**;残差注入。

数据流(三文件):

### (1) Dit_pipeline.dit_infer — 产出高清 ref 脸 latent
匹配后,若 `roi_ref_reencode`,对每个(受 `roi_max_faces` 限制的)ID:
- 用 `id_patch_pairs_pixel[i]['ref']`(像素 xyxy)从 `ref` 张量 crop 出脸;
- `F.interpolate` resize 到 `ref_crop_size`(如 512,需 16 整除)→ `clamp(-1,1)`;
- `KleinVAEProcessor.encode(self.vae, crop)` → `[1,128,S//16,S//16]`(真高细节);
- 收集成 list,`input_data.append(ref_hr_list)`(放在 id_patch_pairs 之后)。

### (2) refine_model.__call__ — 拼进序列 + 配 PE
- 取出 `ref_hr_list`(data[7]);
- 每个 `pack_latents` → `[1, gh*gw, 128]`,拼到 `latent_model_input` 尾部;
- 为每段建 4D 位置 id:把 `(gh,gw)` 网格**映射到 target(lq)脸框坐标**(pe3 思路,内容对齐)、stream T 用 ref(=20)或新流;拼到 `latent_image_ids`;
- 记录每个 ID 的 `ref_hr` 全局 token range `(start,end)`,写进 `id_patch_pairs[i]['ref_hr']`;
- 传 `roi_ref_reencode=True`。

### (3) transformer_flux2 — noise 脸 query attend 高清 ref
- ref_hr token 已在序列里 → 每层自动被 to_k/to_v 投影 + apply RoPE(用它们的 target-mapped ids);
- 在 active 层的 noise fixup:`noise 脸 query(native, 原始 bbox)` attend `[局部 lq(结构) + key[ref_hr_range](真高频细节)]`,`noise_alpha` 残差写回 noise 脸;
- **query 不重采样** → 无低通糊;**KV 是真高频** → 能把 native 脸推向 crisp;
- 输出 `noise_pred[:, :seq_noise]` 已自动丢弃 ref_hr 多余 token。

新配置:`id_patch_roi_ref_reencode: bool=False`、`id_patch_ref_crop_size: int=512`(扫 512/1024)。

### 为何这次不糊且能加细节
- 不上采 query → 没有"上采→下采"两次低通;
- KV 是**像素重编码的真高频**(非插值);
- PE 默认 **pe1/target-mapped**,贴近已证清晰的 crop→1k 设置 → PE 安全;
- 输出仍 native 尺寸(~128px 上限),但能 **crisp 不糊**——这是单趟无后处理的上限。

## 验收
先测噪声底 → 脸 crop → ArcFace + Laplacian,对照 baseline / B(插值) / **crop→1k(上限)**。
- A 明显比 B 锐、接近 crop→1k 的"native 尺寸版" → 成立。
- 若仍糊 → 检查 ref_hr 的 PE 对齐与是否真被 attend(`ROI_DEBUG`)。

## 顺序
1. **先修尾逗号 bug**(否则 roi_pe_mode/开关都不对)。
2. 实现 A(三文件)。
3. `roi_max_faces=1` 单脸先验证 A,再放开。
