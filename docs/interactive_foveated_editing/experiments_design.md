# IFRE Experimental Design

## 1. Overview

We design experiments to validate:

- token-count bias in attention
- effectiveness of reference pyramid
- value of interaction-based control
- causal correspondence in editing

---

## 2. Models

Primary target:
- FLUX.2 Klein 9B (base / KV)

Optional:
- FLUX.2 Klein 4B (fast ablations)

---

## 3. Experiment 1: Token Density Bias

### Goal
Test whether attention mass scales with token count.

### Setup
Same image, two encodings:
- full resize reference
- HR crop + LR global reference

### Metrics
For region R:

M(R) = Σ_i Σ_j∈R A_ij

Compare:
- M(HR crop)
- M(LR region)

### Hypothesis
Without correction:
- HR crop dominates attention

With correction:
- normalized mass invariant

---

## 4. Experiment 2: Reference Pyramid

### Conditions
1. Single resized reference
2. Multi-resolution pyramid
3. Pyramid + density correction
4. Pyramid + interaction graph

### Tasks
- text preservation
- face identity preservation
- logo preservation
- background editing

### Metrics
- OCR accuracy
- face similarity
- LPIPS (non-edited region)
- human preference

---

## 5. Experiment 3: Interaction Control

### Controls
- preserve
- transform
- focus
- no-reference

### Method
Convert user annotation into:

ℓ_ij += λ log P_ij

and suppression for no-ref regions.

### Hypothesis
Interaction improves:
- controllability
- edit consistency
- reduces ghost artifacts

---

## 6. Experiment 4: Causal Correspondence

### Intervention
- drop reference region
- swap reference patch
- mask KV cache

### Measure
CE(i,j) = ||x_i(full) - x_i(intervention)||

### Hypothesis
- high P_ij ⇒ high causal effect

---

## 7. Experiment 5: Resolution Allocation

Test whether:

R(i) = f(focus, preserve, transform)

improves efficiency.

Compare:
- uniform resolution
- foveated output resolution
- interaction-guided resolution

Metrics:
- latency
- FLOPs
- quality score

---

## 8. Ablation Study

We ablate:

- density correction
- reference pyramid
- interaction graph
- no-reference suppression

---

## 9. Expected Outcome

We expect:

- significant reduction in token-count bias
- improved small-text and face preservation
- better multi-reference compositionality
- more stable strong editing

---

## 10. Key Insight

Editing performance depends on:

1. reference resolution allocation
2. token density correction
3. explicit interaction structure
4. causal correspondence, not just attention