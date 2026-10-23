# ofx/ml/PLAN_FEATURES.md — how the rest of Matte Refine ML is built (2026-10-23)

Decision 36 (in `CLAUDE.md`) holds the choices; this file is the build plan. Everything below works on two
plain buffers (RGB picture, matte), so it is independent of how the node receives them (decision 35).

## Shared groundwork (before / beside the features)
| # | Piece | Notes |
|---|---|---|
| P1 | Generic runner protocol v2 + model table | today's protocol is ViTMatte-specific (4 channels in, 1 out). v2 carries any model: N channels in / out, size, a model id; the table (id, file, sha256, channels, normalisation, licence id, size) lives with the models. ViTMatte results stay bit-identical. |
| P2 | Model folder, licence gate, download step | decision 29/31; needed once there are three models. |
| P3 | Link plugin 1's ops (Fill Holes etc.) into the ML bundle | B3 reuses `fillHoles` instead of copying it. |
| P4 | More synthetic scenes with truth | hole inside the subject vs a real gap (B3), thin hair edge (B1/B2), a jittering sequence (B4). |

## Features, in order (decision 36)
1. **B1 polish:** Band px scaling with resolution, threaded trimap, hung-runner handling (see OPEN in `CLAUDE.md`).
2. **B3 Smart Fill Holes (BiRefNet-lite):** whole frame at about 1024 px, upscaled. Plugin 1's Fill Holes finds the holes;
   each is classified subject / real gap by the mean segmentation inside it. BiRefNet never supplies fill values. Params: Enable, Subject Threshold, Max Hole size, Keep Real Gaps. Cost 112 ms HD / 251 ms UHD.
   GPU memory: see decision 36.
3. **B4 Temporal Stabilise (classical):** optical flow t-1 / t / t+1 warps the neighbour input mattes; a weighted median only in
   the edge band, on the INPUT matte before the ML steps (ML cost unchanged). Needs neighbour frames (`setTemporalClipAccess`,
   used by LTX; UNVERIFIED for us in Flame). Params: Frames (1 or 2), Strength.
4. **B2 DiffMatte** as a model choice inside Edge Refine (same trimap, one column). Spike first: no ONNX conversion known;
   try to export encoder once + decoder with a fixed number of steps; compare with ViTMatte on the hair scene and real footage.
   Go / no-go on export success and a visible quality win. Slow is allowed behind a Quality switch. Licence of its training
   data: UNVERIFIED.

Chain order in the node: Temporal -> Smart Fill -> Edge Refine (one column each).

## Parallel work with agents (Opus, separate repo copies)
P1 alone first (runner / protocol files); P4 beside it (tools and tests only). After P1: B3, B4 core + flow, and the B2
export spike at the same time. Each ends with tests, mutation checks and a merge by the main session.

## Status 2026-10-23 (evening)
- P1 (runner protocol v2 + model table), P4 (scenes), the developer-key cleanup: merged.
- **B3 core merged** (`smart_fill.{h,cpp}`, tests `ml_smart_fill`, `ml_smart_fill_real`, eval tool). Evaluation (20 seeds, `gaps` scene):
  plugin 1 Fill Holes by size cannot separate (Max Hole 1 %: 20/100 real gaps kept; 0.2 %: only 31 % of defect pixels filled);
  segmentation alone or colour alone, and "Both" (a hole is filled only when both say subject, threshold 0.6, the default), all keep
  100/100 real gaps and fill 196/196 defects on the plain picture; on a low-contrast picture lite drops to 71 % hole accuracy, full BiRefNet
  stays at 100 % (about 2x the model time: HD 278 vs 161 ms in-process); on a picture with no colour contrast BiRefNet sees no subject and
  nothing is filled (safe failure). Real footage is the deciding test. Not in the core: Bridge Gaps, closed frame edges.
- **B2 spike done** (`research/diffmatte/B2_REPORT.md`, scripts and score tables; weights and ONNX files stay out of the repo): DiffMatte exports to
  ONNX with standard ops (opset 17, fixed 512 tile) and runs through our runner unchanged (same 4-channel input as ViTMatte). The diffusion steps
  add nothing: 1 step = 10 steps within 0.0005, and 10 steps cost 2.8-3.7x. DiffMatte ViT-B at 1 step has the lowest band error on 6 of 7 rows
  (hair_flat 0.073 vs ViTMatte-base 0.104; coarse 0.029 vs 0.031) at about the cost of ViTMatte-base (18.9 vs 18.5 ms per tile). Hair strands are
  not recovered by any model: about 80 % of the lost strand alpha lies outside even a 30 px band, so the TRIMAP RULE is the bottleneck (B1 polish).
  **Licence: no LICENSE file, weights have no stated licence: not clearable on today's facts (LICENCES.md).** A "slow Final" step switch is dropped.
  **SHELVED by the user (2026-10-23)**: no licence and Google-Drive-only weights; weights only from stable hosts like HuggingFace.
- **B4 core merged** (`flow.{h,cpp}`, `temporal.{h,cpp}`, test `ml_temporal`, tool `mr_temporal_eval`). Own pyramidal Lucas-Kanade flow weighted by the reference
  matte (band end-point error 0.27 px vs 1.75 plain), forward-backward confidence, weighted median of the warped neighbour mattes in the band (default band 6 px
  at 1080), majority vote for flickering holes; outside the band bit-identical. Synthetic sequence, 10 seeds: HD K=1 flicker -56 %, band error -26 %;
  K=2 -68 % / -35 %; no bandError regression in any case; the picture is required (flow from mattes alone: -7 %). Time on BFD: HD K=1 96 ms, K=2 185 ms;
  UHD K=1 186 ms; plus about 11 ms per neighbour fetch in Flame. Recommended default K=1, Median, strength 1. Fast motion degrades gracefully, never worse than the input: x6 (about 40 px/frame) -52 % flicker, x10 (about 65 px/frame) -49 %, the forward-backward check rejects 44 % of the
  +-2 reads there. Real footage, motion blur and occlusion not tested. **UNVERIFIED in Flame: fetchImage(t +- k) inside render(), getFramesNeeded, main-thread cost: LTX only fetches other frames from button presses.**
  Plugin integration comes after B3's; start with a Flame probe of neighbour fetch.
- **B3 integrated** (column, chain, Holes view, memory rule); 25/25 ML tests on akflame5, 13/13 ML OFF. End to end on `gaps` with the real runner and BiRefNet-lite: 100 %
  of defect pixels filled, 100 % of real-gap pixels kept. Next: Flame test, then the B4 neighbour-fetch probe, hair trimap (B1 polish), packaging (P2).
- **B4 integrated** (Temporal column; Flame answers pending: `TEMPORAL_TEST.txt`). **B1 polish done**: Outer px (default 0), runner hardening; see `CLAUDE.md` phase 1. Remaining before packaging: Flame tests of Smart Fill, Temporal, Outer px; then P2 (models / licence gate / installer) and release notes.
