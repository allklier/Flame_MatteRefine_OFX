# ViTMatte and very thin hair strands: limits, pre- and post-processing (research, 2026-11-12)

Question: a user-drawn Focus Region becomes the unknown zone of a trimap (solid core of the existing
matte = foreground, outside the region = background). Inside lie hair locks of two 3-4 px strands with a
small gap, over skin, 15-20 px from the solid matte, where the existing matte is ~0. How far can
ViTMatte go, and what helps? Evidence: the paper, the authors' code, the Hugging Face configs used by
the ONNX files, and a synthetic experiment with ground truth (section 5). Synthetic data only.

## 1. What the model is (verified)

| Fact | Value | Source |
|---|---|---|
| Checkpoints in use | ONNX conversions of `hustvl/vitmatte-small-composition-1k` and `-base-composition-1k` (Xenova) | `ofx/ml/package/fetch.sh:23-24` |
| Backbone | ViTDet, hidden 384 (small), 12 blocks; window attention (window 14 tokens) in blocks 0,1,3,4,6,7,9,10; global attention in 2,5,8,11 | HF `config.json` of the small checkpoint |
| Patch size | 16 (VitDetConfig default, not overridden in the config) | transformers `configuration_vitdet.py:115` (v4.46.0) |
| Input | 4 channels: RGB + trimap, patch embedding widened with a zero-initialised 4th channel | paper Sec. 3.2, 3.5 |
| Detail Capture Module | 3 conv layers (3x3, BN, ReLU) on the 4-ch input at 1/2, 1/4, 1/8 (48/96/192 ch); fusion upsamples the 1/16 ViT map step by step and concatenates (256/128/64/32 ch) | paper Sec. 3.4; config `convstream_hidden_sizes`, `fusion_hidden_sizes` |
| Trimap encoding | 0 bg, 0.5 unknown, 1 fg; in training quantised (<85 -> 0, >=170 -> 1, else 0.5): the model never saw soft values | paper Eq. 1; repo `modeling/meta_arch/vitmatte.py:66-68` |
| Loss | L1 separately on unknown and known, + Laplacian + gradient penalty | paper Eq. 7-8 |
| Training data | Adobe Image Matting: 431 foregrounds x 100 COCO backgrounds (Composition-1k); 8-bit sRGB images /255 | paper Sec. 4.1; repo `data/dim_dataset.py:204` |
| Training trimaps | alpha resized to 640x640, fg / bg each eroded by an ellipse of random size 1-29 px, unknown = the rest, resized back | repo `data/dim_dataset.py:498-526` |
| Augmentation | random affine (rotation 30, scale 0.8-1.25, shear 10, flip), crop 512x512 centred on unknown pixels, HSV jitter, composite of two foregrounds | repo `data/dim_dataset.py:608`, paper Sec. 4.2 |
| Normalisation in training | ImageNet mean/std (0.485/0.456/0.406, 0.229/0.224/0.225) | repo `configs/common/model.py:40-41`, `modeling/meta_arch/vitmatte.py:63` |
| Normalisation in the HF processor (and in our plugin) | mean 0.5 / std 0.5 (`IMAGENET_STANDARD_*`) | HF `preprocessor_config.json`; transformers `utils/constants.py:3-4`; plugin `ofx/ml/src/edge_refine.cpp:94` |
| Padding | to a multiple of 32 | HF `size_divisibility` 32; repo `vitmatte.py:73-79` |
| Reported quality | Composition-1k SAD 21.46 (S) / 20.33 (B); D646 21.22 / 17.05 | paper Tables 1-2 |

**Finding: the HF / ONNX preprocessing (and so the plugin) uses mean 0.5 / std 0.5, the authors trained
with ImageNet statistics.** The difference is a contrast factor of about 2.2 and a small colour offset at the
model input. The HF conversion script only checks a 3x3 slice of a nearly-opaque corner
(`convert_vitmatte_to_hf.py:125-133`, values ~0.99), which would not reveal it. Measured below.

Consequences of the architecture for 3-4 px strands (reasoning, checked by the experiment):
- A 16 px patch holds a 3 px strand as a fraction of its pixels: the transformer "sees" the strand only as
  a texture inside the token; the position of the strand must come from the DCM's conv stream (1/2-1/8)
  and the final fusion stage at full resolution (32 channels, fed by the raw input).
- Training hair: the Adobe foregrounds are real photographs with fine hair at their native size, scaled
  0.8-1.25 (image sizes and strand widths in that set: UNVERIFIED, the dataset is on request only). A 2-4 px
  strand is therefore plausibly inside the training distribution in size; what is unusual is (a) strands far from
  the subject's edge in an otherwise clean background (the 640-px trimap erosion produces unknown bands
  hugging the true alpha's support, never a disconnected island), (b) a hand-drawn unknown zone shaped
  like the strands, and (c) skin-coloured backgrounds (COCO / VOC backgrounds are generic photos).
- Global attention in 4 of 12 blocks: the result depends on the tile's context (what else is in it).


## 2. Experiment (synthetic, ground truth)

Code: `exp/synth.py` (scene generator, variants, metrics), `exp/vm_run.cpp` (ONNX Runtime CUDA EP, one
session, raw float tiles, same session setup as `ofx/ml/src/ort_runner.cpp`), `exp/run_all.sh`,
`exp/analyse.py`, `exp/postproc.py`. Raw results: `results/round{1,2}_{small,base}.csv` (one row per
scene x variant). Models: the two ONNX checkpoints the plugin ships (section 1), CUDA, fp32.

- **Scenes** (60 = 4 looks x 5 widths x 3 gaps, 192 x 192 px, linear light): a solid dark mass on the
  left (alpha 1); two locks of two strands each (quadratic curves ~95 px long, constant width
  w = 1.5 / 2 / 3 / 4 / 6 px, edge-to-edge gap g = 1 / 2 / 3 px, 4x4 supersampled coverage); lock A starts
  inside the mass, lock B starts 15 px away from it. Looks: dark hair on smooth skin (`dark_skin`), on
  textured skin (`dark_texture`, 4x the fine texture), on skin in shadow (`dark_shadow`, background x0.15:
  hair 0.035 vs skin ~0.05, i.e. dark on dark), light-brown hair on skin (`light_skin`, low contrast). Low and
  fine background texture, hair colour variation, sensor-like noise; composited in linear light, then
  encoded (default: clamp 0-1, gamma 1/2.4 = what the plugin receives in Auto, decision 34).
- **Trimap** (default): fg = mass eroded 2 px; unknown = a "focus region" painted along each lock
  (circles of radius R = 12 px around its centre line: a ~24 px wide brush, like the user's 25 px shapes);
  bg elsewhere. The existing matte is 0 at the strands, as in the target case.
- **Metrics** (all on the same R = 12 zone, mass excluded): `cover` = mean alpha on true strand cores
  (truth > 0.9); `IoU` of alpha > 0.5; `SAD/100`; `gap alpha` = mean alpha on gap pixels (truth < 0.05);
  `gaps open` = share of those < 0.5; `leak` = mean alpha > 2 px from any strand. A 1 px gap leaves no pixel
  with truth < 0.05 after anti-aliasing, so gaps are measured at 2 and 3 px only.
- **Variants** (21 in round 1, 13 in round 2): normalisation (plugin/HF 0.5/0.5 vs the authors' ImageNet
  mean/std), encoding (gamma 2.4, sRGB, linear, ACEScct), bicubic upscale 2/3/4x with area downsample of
  the result, 8-way flip/rotate TTA (mean), unsharp mask (sigma 1, amount 1 or 2), local contrast stretch (the
  region's 1st-99th luminance percentile -> 0.05-0.95; "mild" = half of it), trimap radius 6 / 20, a box
  region, a soft (0..0.5) roto edge, the scene inside a 512 tile (reflect-padded context).
- Cost: 60 scenes x ~35 inferences per model, ~0.4 s of GPU per scene (small); the GPU was shared.

## 3. Results

**Normalisation (both models, mean over all 60 scenes):** cover / IoU / SAD/100 / leak

| model | 0.5/0.5 (plugin now) | ImageNet (authors) | ImageNet + unsharp 1 |
|---|---|---|---|
| small | 0.71 / 0.72 / 4.75 / 0.021 | 0.81 / 0.82 / 3.25 / 0.009 | **0.88 / 0.87 / 2.62 / 0.009** |
| base | 0.72 / 0.73 / 4.95 / 0.012 | 0.83 / 0.84 / 3.30 / 0.004 | **0.90 / 0.88 / 2.42 / 0.002** |

Round 2 reproduced round 1's baseline rows exactly (same seeds). Gaps kept open: small 0.97 -> 0.97 -> 0.95,
base 0.89 -> 0.84 -> 0.91.

**Boundary, small model** (cover / IoU / gaps open; columns w = 1.5, 2, 3, 4, 6 px, mean over gaps 1-3 px):

| look | plugin today (0.5/0.5, gamma 2.4) | ImageNet + unsharp |
|---|---|---|
| dark on skin | .71/.79/1 · .73/.81/1 · .80/.89/1 · .92/.94/1 · .93/.96/1 | .99/.94/1 · .98/.94/1 · .95/.95/1 · 1.0/.97/1 · 1.0/.98/1 |
| dark on textured skin | .62/.60/1 · .63/.71/1 · .76/.87/1 · .89/.92/1 · .95/.95/1 | .98/.90/1 · .97/.87/1 · .96/.93/1 · .99/.97/1 · 1.0/.97/1 |
| dark on dark (shadow) | .29/.03/1 · .38/.10/.99 · .45/.28/.99 · .56/.60/.88 · .75/.79/.50 | .59/.52/.93 · .58/.53/.89 · .61/.64/.81 · .68/.76/.85 · .82/.81/.59 |
| light hair on skin | .60/.62/1 · .68/.84/1 · .79/.91/1 · .84/.93/1 · .89/.95/.98 | .89/.87/1 · .90/.89/1 · .90/.95/1 · .94/.97/1 · .96/.97/1 |

Base model, same pipeline: dark on dark .77/.63/.84 (1.5 px) to .73/.76/.26 (6 px); the others >= .79 cover.
With the mild stretch the base model holds dark on dark at .76-.87 cover with all gaps open, but loses
light hair (cover .68-.81, IoU .54-.92).

**Where it fails (the boundary):**
- With enough contrast (dark hair on skin, textured or not; light hair on skin), the plugin's current
  preprocessing already finds 3-4 px strands (IoU .87-.94) but under-drives them (cover .76-.92: strands come
  out grey, ~0.7-0.8 alpha) and 1.5-2 px strands reach only .60-.84 IoU. With ImageNet normalisation +
  unsharp, every width down to 1.5 px is recovered (cover >= .89, IoU >= .87).
- Gaps of 2-3 px between two strands stay open on skin at every width, both models, every pipeline that does
  not stretch the contrast. 1 px gaps cannot be scored (see above); a visual check of the small model's
  1 px-gap predictions (w = 3 and 4 px): on skin the two strands stay visibly separate (clearer with ImageNet +
  unsharp), on dark-on-dark they merge into one soft blob.
- **Dark on dark (contrast ~1.5:1) is the real limit:** below 4 px the plugin today gets IoU .03-.28; and
  as strands widen the model fills the gap between them (gaps open .50 at 6 px; base model .10-.14 at 4-6 px).
  ImageNet + unsharp lifts it to IoU .52-.81; only a contrast stretch of the region recovers it (cover .87-.94).
- Leak (alpha > 2 px from any strand) is small on skin (< 0.01) and highest on dark-on-dark and light hair.

**What did not help** (small model, cover / SAD/100 vs baseline .71 / 4.75):
- **Upscaling before inference hurts:** 2x .67 / 4.73, 3x .61 / 5.28, 4x .58 / 5.43 (base model: .66 / .62 / .63).
  It only reduces gap alpha and leak (4x leak .002). With ImageNet normalisation 2x is also worse (.74 vs .81).
  Reading: the model was trained on strands at native size; enlarged, they look like thick, soft-edged
  structures that it treats as uncertain.
- **TTA (8 flips/rotations, mean):** .69 / 5.04 (base .66 / 5.36); averaging blurs thin strands. Not worth 8x.
- **Encodings:** sRGB ~ gamma 2.4 (.72 / 4.59); linear helps dark-on-skin but fails in shadow (.70 / 5.00);
  ACEScct-like log is worst (.62 / 6.79). Keep the display encoding (decision 34).
- **Trimap shape:** radius 6 / 12 / 20 and a box give the same cover (.69-.71); the tight one has the least
  leak in the reference zone but the most leak inside its own band on dark-on-dark (.145). A soft roto edge
  (values 0-0.5) changes nothing measurable (.71 / 4.62), though the model never saw soft values: harmless,
  but quantising to 0 / 0.5 / 1 is cleaner.
- **A bigger tile around the region** (512 with mirrored context): worse (.65 / 5.42, gap alpha .25); the
  context was artificial, but it shows the result depends on what else the tile contains (global attention
  in 4 of 12 blocks). Prefer a crop centred on the region, not a frame-aligned 512 tile.
- **Local contrast stretch** alone: the best on dark-on-dark (cover .49 -> .87) but worse on light hair (SAD
  4.50 -> 6.62) and it narrows gaps (small, with ImageNet: gaps open .78). Useful only when the region's
  contrast is low; the "mild" (half) version is the safer default for that case.

## 4. Post-processing (on the saved predictions, inside the region only)

small, ImageNet + unsharp (cover / IoU / SAD/100 / gap alpha / leak):

| rule | result |
|---|---|
| raw | .88 / .87 / 2.62 / .11 / .009 |
| cut < 0.05 to 0 | .88 / .87 / 2.40 / .10 / .004 |
| remove components < 12 px (at alpha > 0.2) | .88 / .87 / 2.26 / .08 / .002 |
| levels 0.05-0.8 -> 0-1 | .94 / .89 / 1.90 / .11 / .004 |
| components + levels | .94 / .89 / 1.83 / .09 / .002 |

- **Never lower the base matte:** out = base + w x (max(base, alpha) - base), w = the region's soft roto value.
  Nothing in the region can then remove existing matte, and the region's soft edge is the blend.
- **Small-component removal** (connected components of alpha > 0.2 smaller than ~12 px are zeroed): gap
  alpha -30 %, leak -75 %, no coverage loss. A connectivity-to-core rule would be wrong here: lock B does not
  touch the solid matte (15 px away) and would be deleted. An elongation test (bounding-box diagonal >= 10)
  gave identical numbers: the area test is enough.
- **Levels (gain):** ViTMatte under-drives thin strands (opaque strands come out ~0.8); levels 0.05-0.8
  lifts cover +0.06-0.11 and SAD -25-40 % without opening gaps. Caveat: the synthetic strands are fully opaque;
  real fine hair is partly translucent, so a fixed gain overshoots. Offer it as a user control (a "Strand
  Gain" / clip) with a mild default, not as a hidden constant.
- Low threshold (< 0.05 -> 0): cheap leak clean-up, -6-10 % SAD.

## 5. Alternatives (refine inside an unknown band; local)

| model | code licence | weights | thin strands | note |
|---|---|---|---|---|
| ViTMatte S/B (in use) | MIT | Composition-1k (decision 31) | see above | |
| MEMatte (AAAI 2025; ViTMatte + token routing for high resolution) | no LICENSE found (GitHub API: none) | Adobe-trained (UNVERIFIED) | aimed at high-res memory, not thin detail | treat as all rights reserved |
| MatteFormer (CVPR 2022) | Apache-2.0 (GitHub API) | Composition-1k: same Adobe question as ViTMatte, UNVERIFIED | older, below ViTMatte on Comp-1k (paper tables, not re-checked) | |
| AEMatter (2023) | **GPL-2.0** (GitHub API) | Adobe-trained (UNVERIFIED) | strong on Comp-1k per its paper (not re-checked) | copyleft: a problem to ship |
| FBA Matting | MIT code | **non-commercial**: README says the Adobe-trained models are only for non-commercial use | good fg colour + alpha | flag NC |
| GCA Matting | MIT | Adobe-trained (UNVERIFIED) | older | |
| MGMatting | custom (GitHub: NOASSERTION, not read) | UNVERIFIED | mask-guided (no trimap) | |
| Matting Anything (MAM) | MIT | UNVERIFIED | SAM-based, mask-guided | borderline scope |
| DiffMatte | no licence (shelved, decision 36) | Google Drive only | diffusion refinement | shelved |
| MatAnyone | **S-Lab License 1.0, non-commercial** (LICENSE read) | NC | video whole-subject matting | out of scope (17) and NC |

None is a clear thin-strand upgrade with a clean licence; the cheapest gains are in ViTMatte's input.

## 6. Ranked recommendations for the Focus Region (and Edge Refine)

1. **Normalise like the authors** ((x - ImageNet mean) / ImageNet std instead of (x - 0.5) / 0.5,
   `ofx/ml/src/edge_refine.cpp:94`). Cover +0.10-0.11, SAD -32-33 %, leak -57 % (small) / -67 % (base). Cost: one
   line, zero time. Must be re-checked on Edge Refine's own synthetic scenes (phase 0 / trimap_eval) before it
   changes B1's output; treat it as a model-table property (the ONNX files say 0.5/0.5 in their HF
   preprocessor config, which differs from the training code).
2. **Unsharp mask on the region's picture** (Gaussian sigma 1 px, amount 1, display-encoded, clamped):
   with 1, cover .88-.90, SAD -45-51 % vs today, gaps open .91-.95. Cost: a separable blur on a small crop.
3. **Post: never-lower blend + remove components < 12 px**: gap alpha -30 %, leak -75 %. Cheap CPU.
4. **A low-contrast switch:** when the region's luminance range (1st-99th percentile) is small, apply a
   mild contrast stretch (half-strength) before the model: dark-on-dark cover .61 -> .89 (small, ImageNet).
   Don't apply it on normal contrast (it hurts light hair and narrows gaps). Expose as Auto / Off.
5. **Optional user gain (levels 0.05-0.8)** on the region result: +0.06 cover, SAD -25 %; default mild.
6. Run the region as **one crop centred on the region** (multiple of 32, the region + ~32 px margin),
   not as frame-aligned 512 tiles; quantise the trimap to 0 / 0.5 / 1; region radius is not critical
   (6-20 px all fine). Model: small is enough (base: better 1.5-2 px on skin, worse gaps on dark-on-dark).
7. Don't: upscale (2-4x), TTA, log/linear encodings, bigger context tiles.

Limits of this study: synthetic strands are opaque, smooth curves of constant width over smooth
backgrounds; no motion blur, no compression, no real hair translucency or clumping; real footage decides.
Not tested: Lanczos / learned super-resolution (bicubic already hurt), denoising, multi-scale fusion
(the 2x result could be max-fused with 1x for gap safety; not measured), fp16.
