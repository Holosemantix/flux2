# Interactive Foveated Reference Editing (IFRE)

This branch introduces a unified framework for **reference-aware, interaction-driven, mixed-resolution image editing** built on FLUX.2-style diffusion/rectified-flow transformers.

---

## 1. Core Idea

We extend existing foveated diffusion and mixed-resolution generation into **reference-conditioned editing with explicit user interaction**.

Instead of only specifying *where high resolution is needed*, we also specify:

- where to **preserve reference content**
- where to **transform reference content**
- where to **ignore reference (new generation)**
- how different reference images contribute (multi-ref role routing)

This is formalized as a **Region–Reference Relation Graph**.

---

## 2. Key Objects

### 2.1 Tokens
- noise/output tokens: z_i
- reference tokens: r_j

Each token has:
- spatial position
- resolution level (LR/HR)
- area weight ΔA

---

## 3. Region-Reference Graph

We define:

G = (T, R, E)

Where:
- T: output regions
- R: reference regions (multi-image, multi-scale)
- E: edges = user/auto-defined relations

Each edge:

e = (T_a, R_b, τ, s)

- τ ∈ {preserve, transform, focus, no_ref}
- s: strength

---

## 4. Attention Formulation

Standard attention:

ℓ_ij = (q_i^T k_j) / sqrt(d)

Modified IFRE attention:

ℓ_ij = (q_i^T k_j) / sqrt(d) + λ log P_ij + g_τ + b_density

Where:
- P_ij: soft correspondence probability
- g_τ: relation-type bias
- b_density = log ΔA_j

Final attention:

A_ij = softmax(ℓ_ij)

---

## 5. Density-Aware Correspondence

We enforce:

Σ_j P_ij ≤ μ_i,  Σ_i P_ij ≤ ν_j

Where:
- μ_i: output token mass
- ν_j: reference token mass

This enables:
- one-to-many mapping
- many-to-one mapping
- unmatched (new content)

---

## 6. Interaction Types

Preserve:
- strong ref→output alignment

Transform:
- weak correspondence + text conditioning

Focus:
- increases output resolution budget

No-Reference:
- suppress ref attention

---

## 7. Reference Pyramid

Each reference is decomposed into:
- global LR image
- local HR crops (faces, text, logos)
- optional style/layout tokens

---

## 8. Training Strategy

Stage 1: Training-Free
- attention mass logging
- density bias measurement
- ref pyramid ablation
- causal intervention

Stage 2: LoRA
- interaction semantics
- preserve/transform/no-ref control
- multi-reference routing

Stage 3: full adaptation (optional)
- mixed-resolution training

---

## 9. Key Hypotheses

H1: token density biases attention mass
H2: reference pyramid improves fine detail editing
H3: interaction masks improve controllability
H4: no-reference improves strong editing robustness

---

## 10. Target Model

FLUX.2 / DiT-style architectures:
- multi-reference input
- RoPE positional encoding
- rectified-flow diffusion