# 02 · Version A 设计（Expanded-KV Only）

## 一句话

> **Query 用原始 bbox（B_i），写回范围不变；被 cross-attend 的一侧 KV 用扩大后的 bbox（E_i），扩大区域是已 apply RoPE 的现有 token（PE-0，零 PE 风险）。**

## 机制图

```
                    原实现                          Version A
lq ID query (B_i^t)  →  [txt, noise, 整张lq, B_i^s]   →  [txt, noise, 整张lq, E_i^s]   （ref 扩大）
ref ID query (B_i^s) →  [txt, noise, 整张ref, B_i^t]   →  [txt, noise, 整张ref, E_i^t]   （lq 扩大）
写回                 →  output[B_i^t], output[B_i^s]  →  不变（只写回原始 bbox）
```

- `E_i^s = expand(B_i^s, ratio=expand_ratio_ref)`，clip 到 token 网格边界。
- `E_i^t = expand(B_i^t, ratio=expand_ratio_lq)`。
- 因为 `base_for_lq` 已含整张 lq → **`expand_ratio_ref` 是影响目标修复质量的主旋钮**；`expand_ratio_lq` 次要。

## 参数

| cfg key | 含义 | 默认 | 说明 |
|---|---|---|---|
| `id_patch_expand_ratio_ref` | r_s，ref patch 扩大倍数 | 1.0 | **主旋钮**，建议先扫 1.5 / 2.0 / 2.5 |
| `id_patch_expand_ratio_lq`  | r_t，lq patch 扩大倍数 | 1.0 | 次要，1.0~1.5 |
| `id_patch_expand_min_size`  | 扩大后最小边长（token 数）| 0 | 小脸兜底，先 0，必要时 12/16 |
| `id_patch_idx_single_window` | single block 生效层 | [] | 沿用你已有 best layer set |
| `id_patch_idx_double_window` | double block 生效层 | [] | 同上 |

`ratio=1.0 & min_size=0` ⇒ 与现有实现逐 bit 等价（baseline）。

## 实现要点（已在 `code/` 里改好）

1. `_expand_bbox(bbox, ratio, min_size, latent_h, latent_w)`：中心不变缩放 + clip + 防空框。
2. `_id_patch_attention`：query 索引用原始 bbox，KV 索引用 `_expand_bbox` 后的 bbox。
3. 两个 processor 把 `id_patch_config.expand_*` 透传进去。
4. `IdPatchConfig`（refine_model.py + transformer_flux2.py 两份）加 3 个字段。
5. `Dit_pipeline.load_modules` 透传 3 个 cfg。

精确 diff 见 `../CHANGES_version_a.md`。

## 一次性正确性核对（强烈建议跑一次）

合入后、正式实验前，在 `_id_patch_attention` 入口临时打印一次（或在 `refine_model.__call__` 里）：

```python
# 坐标系核对：必须成立
assert seq_lq == latent_h * latent_w, (seq_lq, latent_h, latent_w)
# 每对 bbox 的 token 数（确认 bbox 落在 [0, latent_h)x[0, latent_w) 内、且非空）
for p in id_patch_pairs:
    ly1,lx1,ly2,lx2 = p['lq']; ry1,rx1,ry2,rx2 = p['ref']
    print('lq tokens', (ly2-ly1)*(lx2-lx1), 'ref tokens', (ry2-ry1)*(rx2-rx1),
          'exp_ref tokens', /* 用 _expand_bbox 后再算一遍 */ )
```

确认：
- `seq_lq == latent_h*latent_w`（坐标系对齐，已分析为真，跑一次坐实）；
- baseline（ratio=1.0）下，每对的 ref token 数 == 扩大前；ratio=2.0 下约 ×4（受边界 clip 影响）；
- 没有出现空框 / 越界。

## 与 baseline 对齐验证

先用 `ratio=1.0` 跑一张图，和**改动前的旧代码**输出做逐像素 diff（应当完全一致或仅浮点末位差异）。这一步过了，才说明合入没破坏 baseline，后续扩大实验的对比才可信。

## 显存 / 速度

每对、每个 active 层多两次小 attention。KV 长度随 `ratio²` 增长（被边界 clip），ID 数少时开销可忽略。若 ID 很多 + 层很多，注意 NFA 调用次数线性增加。
