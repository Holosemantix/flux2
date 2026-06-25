# Prior Injection for Diffusion Repair: high-level positioning and novelty check

## 0. One-sentence repositioning

The project should not be positioned merely as an ROI attention trick for small faces. A better high-level framing is:

> **Inference-time prior injection for diffusion repair/refinement:** inject structured external priors into a pretrained diffusion model's internal generation process, so that already-generated or partially-generated results can be repaired, sharpened, or made more consistent without retraining the base model.

In this framing, the current small-face multi-reference experiment is one concrete instance:

- target to repair: noisy/native face tokens in the output stream;
- prior source: low-quality structure image, reference identity image, detected face boxes, canonical face alignment, and attention confidence;
- injection interface: attention-space query/position/KV routing inside the pretrained refiner;
- goal: improve local identity/detail fidelity while preserving the global generated image.

This makes the contribution relevant not only to image editing, but also to text-to-image post-refinement, high-resolution generation, restoration, and generated-image improvement.

## 1. Why the higher-level framing is stronger

The earlier framing was: "How can we make ROI small faces clearer without crop-ref concat?" That is a valid engineering question, but too narrow.

The deeper question is:

> Given a pretrained diffusion model and additional prior knowledge about what a region should look like, where and how should that prior be injected so that the generation trajectory improves rather than collapses, blurs, or changes the wrong content?

This question is broader than editing. It applies to:

1. **Text-to-image repair**: after base generation, repair faces, hands, text, logos, object boundaries, or product details using detectors, OCR, segmentation, reference exemplars, or physical constraints.
2. **Image editing**: preserve structure from the source image while injecting new identity/style/detail priors.
3. **High-resolution generation**: after a low/mid-resolution latent generation, inject high-frequency priors during decoding or refinement.
4. **Restoration and inverse problems**: use known observations, masks, geometry, camera priors, identity priors, or physical priors as constraints.
5. **Multi-reference generation**: inject several priors selectively, preventing cross-ID or cross-object leakage.

The current branch provides a controlled testbed because small faces in group images are exactly where a global diffusion model lacks local capacity and where multiple priors can conflict.

## 2. A useful taxonomy: what kind of prior is injected, and where?

Existing methods can be organized by two axes.

### 2.1 Prior type

- **Text prior**: prompt, negative prompt, regional prompt.
- **Spatial/layout prior**: bbox, mask, depth, pose, segmentation, edge map.
- **Reference appearance prior**: image prompt, reference image tokens, ID embedding, style image.
- **Measurement/data prior**: known pixels, blurred observation, low-resolution image, sensor measurements.
- **Model/architectural prior**: high-resolution decoder, pixel diffusion decoder, foveated token allocation.
- **Geometry/canonicalization prior**: face alignment, object canonical frame, correspondence map.
- **Confidence prior**: attention entropy, ReID confidence, detector confidence, sharpness/quality scores.

### 2.2 Injection site

- **Input-level injection**: concatenate crop/ref image, add prompt, add ControlNet condition.
- **Adapter-level injection**: train or use an image/ID adapter, LoRA, ControlNet, T2I-Adapter.
- **Score/gradient-level injection**: posterior sampling, inverse-problem likelihood gradients, measurement consistency.
- **Latent-resolution injection**: regional latent upsampling, high-resolution latent stages.
- **Decoder-level injection**: replace VAE decoding with generative pixel diffusion decoding.
- **Attention-level injection**: modify query/key/value routing, masks, coordinates, positional encoding, attention logits, or memory.
- **Post-processing injection**: external face restoration, super-resolution, inpainting after generation.

Our current method lives in the **attention-level prior injection** family. The important difference is that it injects priors inside the generation/refinement model without changing the base weights, without introducing a new adapter, and without changing the final output token grid.

## 3. Current method as a prior-injection framework

For each ROI, define a prior package:

```text
P_f = {bbox_noise, bbox_lq, bbox_ref, stream ids, canonical map, quality/confidence scores}
```

The diffusion model already has native hidden states:

```text
H = [H_noise, H_lq, H_ref]
```

The current method injects `P_f` by changing the attention computation for selected ROI tokens:

```text
H_noise_face <- H_noise_face + alpha * Inject(H_noise_face, H_lq_face, H_ref_face, P_f)
```

The injection function is not a new neural network. It is a structured attention operation:

1. choose the target output tokens to repair;
2. preserve their raw query content;
3. create sub-token query probes by changing only their RoPE H/W positions;
4. preserve stream identity on the T axis;
5. route lq and ref memory through separate branches;
6. optionally map lq/ref H/W into a canonical target-local coordinate frame;
7. aggregate the resulting sub-query outputs back into the native output grid.

This can be generalized from faces to other ROI priors if we have a region detector and a correspondence/canonicalization rule.

## 4. Difference from PiD and why PiD is highly relevant

PiD, "Fast and High-Resolution Latent Decoding with Pixel Diffusion", addresses a key limitation of latent diffusion: the latent-to-pixel decoder is reconstruction-oriented and is not expressive enough to synthesize high-resolution details. PiD replaces ordinary decoding with conditional pixel diffusion, unifying decoding and upsampling. It can produce 4x/8x upscaled images and uses a sigma-aware adapter to inject noise-corrupted latents into a pixel diffusion backbone. It can decode 512x512 latents to 2048x2048 pixels quickly.

PiD is highly relevant because it supports the broader claim that **high-resolution generation is not just about sampling more latent tokens; the model also needs the right injection interface for high-frequency detail synthesis**.

However, our method is different:

| Aspect | PiD | This project |
|---|---|---|
| Main problem | Replace/upgrade latent decoding for high-res pixel synthesis | Inject structured priors into a pretrained refiner/generation process for ROI repair |
| Injection site | Decoder/pixel diffusion module | Internal attention of existing diffusion transformer/refiner |
| Training | Requires training/distilling a pixel diffusion decoder and adapter | Training-free attention surgery in current experiments |
| Prior source | Noise-corrupted latent condition | lq structure, ref identity/detail, ROI bbox, canonical alignment, confidence |
| Output grid | Produces high-res pixels via pixel-space denoising | Keeps native output token grid; modifies selected ROI hidden states |
| Scope | General high-res decoding/upsampling | Local prior-guided repair/refinement; potentially usable after T2I/editing generation |

The connection is strategic: PiD can be viewed as a powerful **decoder-level prior injection** mechanism, while this project explores **attention-level prior injection**. These are complementary, not redundant. In a future high-level system, PiD could provide global high-res decoding while ROI prior injection handles identity, text, object, or geometry-specific corrections inside the denoising/refinement process.

## 5. Difference from RALU / region-adaptive latent upsampling

RALU is close in spirit because it allocates more computation to important regions. It performs region-adaptive latent upsampling during denoising: low-resolution denoising captures global structure, artifact-prone regions are upsampled/refined at higher latent resolution, and timestep/noise scheduling is adjusted to handle resolution transitions.

Our method differs in the injection site and objective:

| Aspect | RALU | This project |
|---|---|---|
| High-level idea | Region-adaptive computation | Region-adaptive prior injection |
| What changes | Latent sampling/resolution schedule | Attention query/position/routing for ROI tokens |
| Token grid | ROI latent resolution changes | Final output token grid stays native |
| Key risk | Noise/timestep mismatch across resolutions | Prior mismatch, stream leakage, wrong alignment, over-injection |
| Prior source | Mainly model's own latent denoising process | Explicit lq/ref/ROI/canonical/confidence priors |
| Main use | Efficient high-res generation / reduce artifacts | Repair generated ROI using external structured prior |

Thus the method should not be called latent upsampling. A more accurate term is **prior-conditioned attention repair** or **query-foveated prior injection**.

## 6. Difference from CRPA / mixed-resolution RoPE repair

CRPA is important because it identifies a failure mode relevant to this branch: naive mixed-resolution RoPE can introduce phase mismatch, attention collapse, blur, or artifacts. CRPA's contribution is to align cross-resolution positional phases by expressing Q/K positions under compatible strides.

Our method uses a related observation but asks a different question:

- CRPA: how to make mixed-resolution attention geometrically valid?
- This project: how to use RoPE/attention coordinates as an injection interface for external priors?

Sub-token RoPE probes are therefore not just a mixed-resolution fix. They are a way to ask multiple localized questions from the same native output token:

```text
same semantic/noise query content + multiple local positions -> richer prior matching
```

If CRPA-style phase alignment is added, it becomes a correctness layer for our prior injection, not the core contribution by itself.

## 7. Difference from ControlNet / GLIGEN / BoxDiff / layout-control methods

These methods inject strong spatial or layout priors. They are important but operate at a different level.

| Family | Typical prior | Typical injection | Difference from this project |
|---|---|---|---|
| ControlNet/T2I-Adapter | depth, edge, pose, segmentation | trained side network / adapter | needs trained module; global condition pathway |
| GLIGEN / grounded generation | bbox + phrase/object grounding | gated grounding layers / trained components | primarily text-layout grounding, not post-hoc ROI repair |
| BoxDiff / training-free layout control | bbox constraints | attention/score guidance | controls object placement, not reference-detail repair |

Our method can be positioned as a complementary category: **attention-level prior injection for local refinement**, where the prior can be a detected ROI, a reference identity, an object canonical coordinate, OCR/text prior, or other structured knowledge.

## 8. Difference from ID adapters and reference-image methods

IP-Adapter, InstantID, PhotoMaker, PuLID, InstantFamily, and related methods use image prompts, ID embeddings, face encoders, landmarks, masks, or trained adapters to preserve identity.

Their prior is often learned into an adapter or represented as a compact embedding. This is powerful but different from our direction.

Our project is trying to answer:

> If the base refiner already contains lq/ref image tokens, can we inject identity/detail priors by changing inference-time attention routing and coordinate alignment rather than training a new adapter?

Key differences:

| Aspect | ID adapter/reference methods | This project |
|---|---|---|
| Prior representation | learned embedding or adapter tokens | existing lq/ref image tokens + bbox/canonical priors |
| Training | usually needs adapter training | current experiments are training-free |
| Injection | cross-attention adapter or ID branch | attention surgery inside existing refiner layers |
| Locality | often face-focused but may affect global generation | hard ROI writeback and per-ID isolation |
| Main risk | identity overfit, prompt conflict, adapter strength | wrong spatial alignment, stream leakage, beta/gate tuning |

The novelty is not "using reference images for identity". The novelty is **how** the reference prior is injected: region-specific, stream-aware, coordinate-canonicalized, and query-side.

## 9. Difference from inverse-problem posterior sampling

DPS and plug-and-play posterior sampling methods treat diffusion models as priors and inject measurement likelihood or constraints during sampling. This is the strongest high-level neighbor because it also asks how external knowledge can guide diffusion.

But the direction is different:

| Aspect | Posterior sampling / inverse problems | This project |
|---|---|---|
| External prior | measurement y and forward operator A | semantic/structural priors: bbox, ref identity, lq structure, alignment, confidence |
| Injection form | likelihood gradient, projection, posterior update | attention-space routing/coordinates/residual writeback |
| Mathematical goal | sample p(x | y) for inverse problem | improve selected generated regions while preserving global result |
| Domain | restoration, MRI, deblurring, phase retrieval, SR | editing/T2I post-repair/multi-reference refinement |

This suggests a high-level framing:

> We are extending the idea of posterior guidance from measurement-space constraints to **attention-space semantic priors**.

This is potentially broad and more fundamental than a face-specific trick.

## 10. Why crop works: prior injection view

The branch's crop-to-1k result can be reinterpreted through this framework. Crop works not only because it adds tokens. It also injects several implicit priors:

1. **Density prior**: face occupies more tokens.
2. **Alignment prior**: noise/lq/ref are placed in a more canonical frame.
3. **Local dominance prior**: background and other identities are removed, so attention is not diluted.
4. **Trajectory prior**: high-density face participates through the entire denoising/refinement process.
5. **Decoder prior**: if followed by stronger decoding/SR, high-frequency synthesis becomes easier.

Therefore, the correct research question is not "can we avoid crop?" but:

> Which prior injected by crop is actually responsible for quality improvement, and can we inject that prior directly inside the model without crop-ref concat?

The current experiments are a decomposition of crop's hidden priors:

- Expanded-KV tests larger context support.
- Noise-fixup tests direct output-stream injection.
- Virtual ROI-QKV tests token-density-style injection but reveals interpolation low-pass failure.
- Q-only supersampling tests query/probing density without V interpolation.
- Canonical alignment tests whether crop's hidden alignment prior is essential.
- Ref re-encode tests true high-frequency prior source.

This decomposition itself can be a research contribution.

## 11. More general method proposal

A high-level method could be named:

> **Prior-Conditioned Attention Repair (PCAR)**

or, more specific to the current implementation:

> **Query-Foveated Prior Injection (QFPI)**

General pipeline:

1. **Prior discovery**: detect regions and gather prior sources: reference images, lq image, masks, text/OCR, depth, pose, identity, sharpness, confidence.
2. **Prior canonicalization**: convert each prior source into a shared local coordinate frame while preserving stream identity.
3. **Prior routing**: decide which output tokens should receive which prior memory.
4. **Prior probing**: use query-side sub-token RoPE probes or other attention probes to query prior memory more finely.
5. **Prior fusion**: combine structure/detail/source memories with adaptive gates based on confidence.
6. **Safe writeback**: inject residuals only into target ROI tokens, with alpha/layer/timestep schedules.

This can cover:

- face repair from reference identity;
- hand repair from pose/mesh priors;
- text/logo repair from OCR/vector priors;
- product detail repair from reference product images;
- T2I post-generation local refinement;
- high-resolution decode/refinement combined with PiD-style decoders.

## 12. Stronger novelty statement

A safe and high-level novelty claim:

> We propose a training-free framework for injecting structured priors into pretrained diffusion models through attention-space routing, coordinate canonicalization, and query-side probing. Instead of adding new condition networks, changing the output resolution, or post-processing generated images, the method modifies how selected output tokens query prior memories inside the denoising/refinement process. The current multi-reference face refinement system instantiates this framework with lq/ref image-token priors, stream-preserving canonical ROI alignment, Q-only sub-token RoPE probes, and adaptive structure/detail fusion.

Shorter version:

> **Diffusion repair by attention-space prior injection.**

This is broader than face editing and can be connected to high-resolution T2I post-refinement.

## 13. What we can and cannot claim

### Can claim

- A training-free attention-space prior injection framework for local diffusion repair.
- Query-side foveated probing: increasing attention matching density without interpolating V or changing output grid.
- Stream-preserving canonical ROI alignment: inject alignment priors while preserving noise/lq/ref stream identities.
- Structure/detail prior decomposition: lq for structure, ref for detail/identity, with adaptive fusion.
- A decomposition of crop-gain into density, alignment, local dominance, trajectory, and high-frequency source priors.

### Should not claim

- First ROI attention.
- First reference-image identity injection.
- First foveated or high-resolution diffusion method.
- First RoPE coordinate manipulation.
- First prior-guided diffusion or inverse-problem guidance.

## 14. Experimental story that best supports the high-level claim

The strongest experimental story is not only final image quality. It is controlled decomposition:

1. **Baseline**: existing refiner.
2. **Expanded-KV**: context prior only.
3. **Noise-fixup**: direct output-stream injection.
4. **Virtual ROI-QKV**: density via interpolation; expected to fail or blur.
5. **Q-only probing**: query-density prior without V interpolation.
6. **Canonical alignment**: alignment prior isolated from density.
7. **Adaptive fusion**: confidence prior for lq/ref mixing.
8. **Ref re-encode / crop-to-1k / PiD-style decoder**: true high-frequency/decoder-level upper bounds.

If this sequence shows that alignment/gating/query probing helps while interpolation hurts, the paper/report can argue a general principle:

> Effective diffusion repair requires injecting the right prior at the right interface. Naively increasing token density by interpolation is weak; coordinate-canonicalized, stream-aware, confidence-gated attention injection is a more controllable route.

## 15. Relation to PiD in future work

PiD is a high-resolution decoder-level solution. Our method is an attention-level prior injection solution. A strong future combination is:

```text
base T2I/editing latent generation
  -> attention-space prior repair for semantic/local consistency
  -> PiD-style pixel diffusion decoder for high-frequency synthesis
```

This would separate two roles:

- attention-space prior injection decides **what** should be corrected and from which prior;
- pixel diffusion decoding decides **how** to synthesize high-resolution pixels.

This separation makes the proposed work a high-level method rather than a single face-enhancement trick.
