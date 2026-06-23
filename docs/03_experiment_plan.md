# 03 · 实验步骤（Version A）

## 前置

1. 按 `../CHANGES_version_a.md` 把改动合入真实仓（或直接用 `code/` 里改好的两份文件 diff 后替换）。
2. 跑一次 `docs/02` 的「坐标系核对」+「baseline 对齐验证」（ratio=1.0 与旧代码逐像素一致）。**这两步不过不要往下走。**
3. 固定其它一切：同一组测试图、同一 `seed`、同一 step 数、同一 best layer set（`id_patch_idx_*_window`）、同一 colorfix 设置。每组实验**只动 expand 参数**。

测试集：`Group_Photo_03_18_org` 这类合影（已在 `test_refiner.py` 里）。每组跑全量、输出到独立目录，便于并排对比。

## 实验矩阵

保持 layer set 不变，只扫 expand 参数。bbox 格式 `(y1,x1,y2,x2)`，r_s = ref 扩大、r_t = lq 扩大。

| 实验 | `expand_ratio_lq` (r_t) | `expand_ratio_ref` (r_s) | `expand_min_size` | 目的 |
|------|------|------|------|------|
| **B1** baseline | 1.0 | 1.0 | 0 | 复现当前 ID patch attention（必须与旧代码一致）|
| K1 conservative | 1.5 | 1.5 | 0 | 轻微扩大 |
| **K2** main | 2.0 | 2.0 | 0 | 主实验，对称扩大 |
| K3 reference-heavy | 1.5 | 2.5 | 0 | source 扩更大（更看好，提供发际线/脸型/光照）|
| K4 首推 | 1.5 | 2.0 | 0 | 方案 §17 第一版配置 |
| K5 small-face | 1.5 | 2.0 | 16 | 开 min_size，看对小脸的额外增益 |

资源紧就先跑 **B1 / K2 / K3 / K4** 四组。

## 另设对照（强烈建议，定位增益来源）

| 实验 | 设置 | 说明 |
|------|------|------|
| **B0** | `use_id_patch_attention: false` | 完全不做 ID attention 的原模型，作为绝对底线 |

有了 B0 / B1 / K*，才能区分三件事：
1. ID attention 本身有没有用（B1 vs B0）；
2. 扩大 KV 有没有额外增益（K* vs B1）；
3. source 扩大 vs 对称扩大谁更好（K3 vs K2）。

## 关键 cfg 片段

```yaml
use_id_patch_attention: true
patch_split_num: 1                      # id patch 只在 no_split 下生效
id_patch_idx_single_window: [ ... ]     # 你已验证有效的层，保持不变
id_patch_idx_double_window: [ ... ]

# 本次唯一要变的：
id_patch_expand_ratio_lq:  1.5
id_patch_expand_ratio_ref: 2.0
id_patch_expand_min_size:  0

# ID 匹配（沿用现有）
id_match_conf_threshold: 0.15
id_match_dist_threshold: 0.4
id_match_imgsz: 3072
```

## 输出与归档

- 每组实验输出目录命名带参数，例如 `.../K3_rt1.5_rs2.5_min0/`。
- 保留 `id_match_debug/`（YOLO+ReID 匹配可视化），确认匹配本身没错——**匹配错了，attention 再好也白搭**。
- 每组至少留：最终结果图、`_lr`（lq）、`_ref`。

## 单图快速迭代

正式全量前，先挑 3~5 张「ID 匹配正确、人脸不大不小、退化明显」的图做快速扫描，确定 r_s 的大致甜点（1.5 / 2.0 / 2.5），再上全量。

## 层 / step 的后续可选维度（A 跑通后再碰）

- **层**：先严格沿用 best layer set。之后可试「只在中后层」（结构层少动、细节层多动）。
- **step**：少步数蒸馏模型，可试只在中间/后段开 ID attention。
- 这些都属于 A 之后的扩展，**第一轮不要和 expand 一起扫**，否则归因困难。
