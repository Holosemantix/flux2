# IFRE Mathematical Formulation

## 1. Problem Definition

We model image editing as conditional generation:

\[
x_0 \sim p(x_0 | x_{ref}, c)
\]

Where:
- \(x_{ref}\): reference image(s)
- \(c\): text + interaction graph
- \(x_0\): target edited image

---

## 2. Tokenization

We define:

- noise/output tokens: \(z_i \in \mathbb{R}^d\)
- reference tokens: \(r_j \in \mathbb{R}^d\)

Each token has:

\[
z_i = (h_i, p_i, \Delta A_i)
\]

Where:
- \(h_i\): hidden feature
- \(p_i\): position
- \(\Delta A_i\): area weight

---

## 3. Region Graph

We define a directed bipartite graph:

\[
G = (T, R, E)
\]

Each edge:

\[
e_{ij} = (i, j, \tau_{ij}, s_{ij})
\]

Where:
- \(\tau_{ij} \in \{preserve, transform, focus, no\_ref\}\)
- \(s_{ij} \in [0,1]\)

---

## 4. Soft Correspondence

We define correspondence matrix:

\[
P_{ij} = \Pr(z_i \leftrightarrow r_j)
\]

Constraints:

\[
\sum_j P_{ij} \le \mu_i
\]

\[
\sum_i P_{ij} \le \nu_j
\]

Where:
- \(\mu_i\): output mass
- \(\nu_j = \Delta A_j \cdot r_j^{ref}\)

---

## 5. Attention Modification

Base attention:

\[
\ell_{ij} = \frac{q_i^T k_j}{\sqrt{d}}
\]

IFRE attention:

\[
\ell_{ij}^* = \frac{q_i^T k_j}{\sqrt{d}} + \lambda_t \log(P_{ij} + \epsilon) + g(\tau_{ij}) + \log \Delta A_j
\]

Softmax:

\[
A_{ij} = \text{softmax}_j(\ell_{ij}^*)
\]

---

## 6. Density Correction

To avoid token-count bias:

\[
\tilde{A}_{ij} = \frac{A_{ij}}{\Delta A_j}
\]

or equivalently:

\[
\ell_{ij} \leftarrow \ell_{ij} + \log \Delta A_j
\]

---

## 7. Resolution Mapping

Define resolution function:

\[
R(i) = f(r_i^{focus}, r_i^{preserve}, r_i^{new})
\]

Constraints:

- preserve → high ref resolution
- focus → high output resolution
- no_ref → low ref dependence

---

## 8. Covariance Under Upsampling

Let:

\[
z_h = U z_l
\]

Then:

\[
\text{Cov}(z_h) = \sigma^2 U U^T
\]

We define correction:

\[
z_h' = W z_h + \epsilon_{corr}
\]

such that:

\[
\text{Cov}(z_h') \approx I
\]

---

## 9. Causal Correspondence Test

Define intervention:

\[
CE(i,j) = || x_i^{full} - x_i^{drop(r_j)} ||
\]

If:

\[
CE(i,j) \propto P_{ij}
\]

then correspondence is causal.

---

## 10. Key Insight

Editing is not only generation:

\[
\text{Editing} = \text{Generation} + \text{Reference Routing} + \text{Resolution Allocation}
\]