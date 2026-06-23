# Version A 改动清单（Expanded-KV Only）

> 本文件列出把现有 ID Patch Attention 升级为「Version A：扩大被 cross-attend 一侧 KV」的**全部精确改动**。
> 共改 3 个文件、约 5 处。所有改动在默认参数 `(expand_ratio_*=1.0, expand_min_size=0)` 下与现有实现**逐 bit 等价**，因此可以安全合入而不改变 baseline 行为。
>
> 仓库里 `code/transformer_flux2.py` 和 `code/refine_model.py` 是改好的整份文件（基于你贴出的版本重建）。
> **强烈建议先 `diff` 你真实仓里的对应文件再合入**，确认改动只发生在下面列出的位置。

---

## 文件 1：`diffusers/models/transformers/transformer_flux2.py`

### 1.1 `IdPatchConfig` dataclass — 新增 3 个字段

```python
@dataclass
class IdPatchConfig:
    idx_single_window: List[int] = field(default_factory=list)
    idx_double_window: List[int] = field(default_factory=list)
    expand_ratio_lq: float = 1.0      # 新增：fix-up ref-query 时，被 attend 的 lq patch 扩大倍数 (r_t)
    expand_ratio_ref: float = 1.0     # 新增：fix-up lq-query  时，被 attend 的 ref patch 扩大倍数 (r_s) ← 主旋钮
    expand_min_size: int = 0          # 新增：扩大后 bbox 在 token 网格上的最小边长（token 数）
```

### 1.2 新增 helper `_expand_bbox`（放在 `_build_2d_rect_indices` 之后）

```python
def _expand_bbox(bbox, ratio, min_size, latent_h, latent_w):
    y1, x1, y2, x2 = bbox
    h, w = y2 - y1, x2 - x1
    cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
    new_h = max(h * ratio, float(min_size))
    new_w = max(w * ratio, float(min_size))
    ny1 = max(0, int(round(cy - new_h / 2.0)))
    nx1 = max(0, int(round(cx - new_w / 2.0)))
    ny2 = min(latent_h, int(round(cy + new_h / 2.0)))
    nx2 = min(latent_w, int(round(cx + new_w / 2.0)))
    ny2 = max(ny2, ny1 + 1); nx2 = max(nx2, nx1 + 1)
    ny2 = min(ny2, latent_h); nx2 = min(nx2, latent_w)
    return (ny1, nx1, ny2, nx2)
```

坐标系：`bbox=(y1,x1,y2,x2)` 是 token 网格坐标，与 `id_patch_pairs['lq'/'ref']` 一致。
clip 边界 `latent_h/latent_w` 与 `_build_2d_rect_indices` 用的 `latent_w` 同一坐标系（已确认 `seq_lq == latent_h*latent_w`）。

### 1.3 `_id_patch_attention` — 签名增 3 参数 + KV 用扩大 bbox

签名新增（带默认值，向后兼容）：

```python
def _id_patch_attention(query, key, value, ranges, id_patch_pairs, latent_h, latent_w,
                        expand_ratio_lq=1.0, expand_ratio_ref=1.0, expand_min_size=0,
                        backend=None, parallel_config=None):
```

Step 3 的 per-pair 循环内：**query 索引仍用原始 bbox，KV 索引改用扩大后 bbox**：

```python
for pair in id_patch_pairs:
    lq_bbox  = pair['lq']
    ref_bbox = pair['ref']

    # 新增：被 cross-attend 的一侧用扩大 bbox
    lq_bbox_exp  = _expand_bbox(lq_bbox,  expand_ratio_lq,  expand_min_size, latent_h, latent_w)
    ref_bbox_exp = _expand_bbox(ref_bbox, expand_ratio_ref, expand_min_size, latent_h, latent_w)

    # query 索引：原始 bbox（写回范围不变）
    lq_local_indices  = _build_2d_rect_indices(*lq_bbox,  latent_w, query.device)
    ref_local_indices = _build_2d_rect_indices(*ref_bbox, latent_w, query.device)
    # KV 索引：扩大 bbox
    lq_kv_indices  = _build_2d_rect_indices(*lq_bbox_exp,  latent_w, query.device)
    ref_kv_indices = _build_2d_rect_indices(*ref_bbox_exp, latent_w, query.device)

    # lq fix-up：ref_patch_key/value 改用 ref_kv_indices（原来是 ref_local_indices）
    if lq_local_indices.numel() > 0:
        lq_global_indices = lq_local_indices + lq_start
        lq_patch_query  = query[:, lq_global_indices]
        ref_patch_key   = ref_key[:,   ref_kv_indices]     # ← 扩大
        ref_patch_value = ref_value[:, ref_kv_indices]     # ← 扩大
        ... (其余不变)

    # ref fix-up：lq_patch_key/value 改用 lq_kv_indices（原来是 lq_local_indices）
    if ref_local_indices.numel() > 0:
        ref_global_indices = ref_local_indices + ref_start
        ref_patch_query = query[:, ref_global_indices]
        lq_patch_key    = lq_key[:,   lq_kv_indices]        # ← 扩大
        lq_patch_value  = lq_value[:, lq_kv_indices]        # ← 扩大
        ... (其余不变)
```

> 注意 `*lq_bbox` 解包成 `(y1, x1, y2, x2)`，`_build_2d_rect_indices` 形参顺序是 `(y1, x1, y2, x2, latent_w, device)`。

### 1.4 两个 Processor 的调用点 — 透传 3 个 ratio

`Flux2AttnProcessor.__call__` 和 `Flux2ParallelSelfAttnProcessor.__call__` 中调用 `_id_patch_attention` 处，新增三行：

```python
    expand_ratio_lq=getattr(id_patch_config, "expand_ratio_lq", 1.0),
    expand_ratio_ref=getattr(id_patch_config, "expand_ratio_ref", 1.0),
    expand_min_size=getattr(id_patch_config, "expand_min_size", 0),
```

---

## 文件 2：`algorithms/pangu_i2i/refiner/model_interface/refine_model.py`

### 2.1 `IdPatchConfig` dataclass — 同样新增 3 个字段

```python
@dataclass
class IdPatchConfig:
    idx_single_window: List[int] = field(default_factory=list)
    idx_double_window: List[int] = field(default_factory=list)
    expand_ratio_lq: float = 1.0
    expand_ratio_ref: float = 1.0
    expand_min_size: int = 0
```

> 实际被 transformer 用到的是 **这份**（`refine_model.py` 里实例化的）`IdPatchConfig`。
> `transformer_flux2.py` 里那份只作类型注解，但仍建议同步加字段（duck-typing 读取 `getattr`，缺字段也能回退默认值）。

### 2.2 `RefinerModel.__init__` 构造 `IdPatchConfig` 处 — 传入 3 个 kwargs

```python
self.id_patch_config = IdPatchConfig(
    idx_single_window=kwargs.get('id_patch_idx_single_window', []),
    idx_double_window=kwargs.get('id_patch_idx_double_window', []),
    expand_ratio_lq=kwargs.get('id_patch_expand_ratio_lq', 1.0),   # 新增
    expand_ratio_ref=kwargs.get('id_patch_expand_ratio_ref', 1.0), # 新增
    expand_min_size=kwargs.get('id_patch_expand_min_size', 0),     # 新增
)
```

---

## 文件 3：`algorithms/pangu_i2i/refiner/model_pipeline/Dit_pipeline.py`

### 3.1 `load_modules()` 里 `if self.use_id_patch:` 分支 — 透传 3 个 cfg

在已有的两行
```python
dit_params['id_patch_idx_single_window'] = self.cfg.get('id_patch_idx_single_window', [])
dit_params['id_patch_idx_double_window'] = self.cfg.get('id_patch_idx_double_window', [])
```
之后追加：
```python
dit_params['id_patch_expand_ratio_lq']  = self.cfg.get('id_patch_expand_ratio_lq', 1.0)
dit_params['id_patch_expand_ratio_ref'] = self.cfg.get('id_patch_expand_ratio_ref', 1.0)
dit_params['id_patch_expand_min_size']  = self.cfg.get('id_patch_expand_min_size', 0)
```

`test_refiner.py` 无需改动。

---

## 配置流（cfg → 生效）

```
cfg.yaml
  use_id_patch_attention: true
  id_patch_idx_single_window: [...]   # 沿用你已有的 best layer set
  id_patch_idx_double_window: [...]
  id_patch_expand_ratio_lq: 1.5       # r_t
  id_patch_expand_ratio_ref: 2.0      # r_s（主旋钮）
  id_patch_expand_min_size: 0         # 小脸兜底，先 0
        │
        ▼  Dit_pipeline.load_modules() 透传
   dit_params['id_patch_expand_*']
        │
        ▼  RefinerModel.__init__()
   IdPatchConfig(expand_ratio_lq=.., expand_ratio_ref=.., expand_min_size=..)
        │
        ▼  transformer_kwargs['id_patch_config']
   Flux2Transformer2DModel.forward → block_attention_kwargs
        │
        ▼  Flux2AttnProcessor / Flux2ParallelSelfAttnProcessor
   _id_patch_attention(..., expand_ratio_lq, expand_ratio_ref, expand_min_size)
        │
        ▼  _expand_bbox → 扩大后 bbox 的 token 索引 → 选已 apply RoPE 的现有 KV
```

## 等价性 / 安全性

- `expand_ratio_lq = expand_ratio_ref = 1.0` 且 `expand_min_size = 0` 时，`_expand_bbox` 返回原 bbox（中心不变、宽高×1、min 不触发），KV 索引 == 原 `*_local_indices`，输出与现有实现完全一致。
- 改动不新增 token、不改位置编码、不改主干 attention，只是把被选中的 KV 索引集合按比例扩大。属于 PE-0（零 PE 风险）。

---

# Version A' 改动清单（Noise-Fixup）

> 在 Version A 之上新增:对 **noise(输出) 段**人脸 query 做 fix-up，让输出直接 attend `[局部lq(结构) + 扩大ref(细节)]`，残差注入。
> 默认 `fixup_noise=False` 时与 Version A 行为一致。设计与实验顺序见 `docs/06`。

## A'-1. `transformer_flux2.py`

- **`IdPatchConfig` dataclass** 再加 3 个字段:
  ```python
  fixup_lqref: bool = True     # 是否保留原 lq/ref 段 fix-up
  fixup_noise: bool = False    # 是否对 noise 段做 fix-up
  noise_alpha: float = 0.5     # 残差注入强度
  ```
- **`_id_patch_attention`** 签名加 `fixup_lqref=True, fixup_noise=False, noise_alpha=0.5`;原 lq/ref fix-up 包进 `if fixup_lqref:`；循环内追加 noise fix-up 块:
  ```python
  if fixup_noise and lq_local_indices.numel() > 0 and noise_end > noise_start:
      noise_q_indices = lq_local_indices + noise_start          # noise 与 lq 共用 bbox
      noise_q = query[:, noise_q_indices]
      k_struct, v_struct = lq_key[:, lq_kv_indices],  lq_value[:, lq_kv_indices]   # 结构←lq(扩大 r_t)
      k_detail, v_detail = ref_key[:, ref_kv_indices], ref_value[:, ref_kv_indices] # 细节←ref(扩大 r_s)
      combined_k = torch.cat([k_struct, k_detail], dim=1)
      combined_v = torch.cat([v_struct, v_detail], dim=1)
      noise_out = _dispatch_attention(noise_q, combined_k, combined_v, num_heads=H, ...)
      output[:, noise_q_indices] = (1.0 - noise_alpha) * output[:, noise_q_indices] + noise_alpha * noise_out
  ```
- **两个 processor** 调用 `_id_patch_attention` 处各加 3 行:
  ```python
  fixup_lqref=getattr(id_patch_config, "fixup_lqref", True),
  fixup_noise=getattr(id_patch_config, "fixup_noise", False),
  noise_alpha=getattr(id_patch_config, "noise_alpha", 0.5),
  ```

## A'-2. `refine_model.py`
- `IdPatchConfig` 同样加 `fixup_lqref / fixup_noise / noise_alpha` 三字段。
- `RefinerModel.__init__` 构造时追加:
  ```python
  fixup_lqref=kwargs.get('id_patch_fixup_lqref', True),
  fixup_noise=kwargs.get('id_patch_fixup_noise', False),
  noise_alpha=kwargs.get('id_patch_noise_alpha', 0.5),
  ```

## A'-3. `Dit_pipeline.py`（`load_modules` 再加 3 行透传）
```python
dit_params['id_patch_fixup_lqref'] = self.cfg.get('id_patch_fixup_lqref', True)
dit_params['id_patch_fixup_noise'] = self.cfg.get('id_patch_fixup_noise', False)
dit_params['id_patch_noise_alpha'] = self.cfg.get('id_patch_noise_alpha', 0.5)
```

`code/transformer_flux2.py` 和 `code/refine_model.py` 已是含 A' 的整份文件。
