# Model candidate for the Focus Region (Edge Recipes WP4)

Written 2026-11-12 by the WP4 agent. Library `mr_recipe_model` (`ofx/recipes/model/`), candidate (a) of decision 39.
The rules are in `model_candidate.h` (top comment); this file records the choices and the evaluation.

## What it does
Picture + base matte + region weight in; alpha + confidence out (`merge_fusion.h` `Candidate`). ViTMatte runs on one
crop centred on the region's box (multiple of 32, 32 px margin at 1080 lines, at most 1024 per side; a bigger box is
split into equal cells), through `RunnerClient::run(modelId, ...)` (`runner_client.h:83`; adapter `runner_crops.h`).
No ONNX Runtime in the library (C++14). Plugin 2's Edge Refine is unchanged (`edge_refine.cpp:94` still 0.5 / 0.5).

Order (the research's ranked list, `ofx/ml/research/vitmatte_thin_hair/README.md` section 6):
1. ImageNet mean / std, per call (`ModelCandidateParams::norm`; `Norm::Half` = plugin 2's).
2. Unsharp sigma 1 px, amount 1, on the display-encoded crop, clamped 0-1.
3. Model alpha restricted to the region, pieces < 12 px (area x s^2, 8-connected, alpha > 0.2) dropped, then never
   lower: `base + w (max(base, a) - base)`; bit for bit the base where w = 0.
4. Low-contrast switch (Auto / Off / On): the mild stretch when the region is dark and flat (rule below); unsharp is
   then skipped.
5. Strand Gain: `clamp((a - black) x gain)`, black 0 / gain 1 = identity (default).
6. `vitmatte-small` by default, `vitmatte-base` by `modelId`. No upscaling, no flip / rotation averaging.

Trimap: `regionTrimap` quantised as in training (< 85/255 -> 0, >= 170/255 -> 1, else 0.5), then base >= 0.95 forced to
foreground (`regionTrimap` alone makes the core unknown where the region covers it; the research kept the core known).

Low-contrast rule (chosen on the research scenes, region display luma = mean RGB over weight > 0): stretch when
p99 - p1 < 0.12 AND p99 < 0.40. Measured p1 / p99 (range): dark on dark 0.18-0.19 / 0.27-0.28 (0.08-0.09), light hair on
skin 0.43-0.46 / 0.57-0.60 (0.12-0.16), dark hair on skin 0.18-0.20 / 0.59-0.64 (0.40-0.44). The range alone does not
separate dark-on-dark from light hair (0.09 vs 0.12) and the stretch hurts light hair (research), hence the brightness
condition. The display ratio p99 / p1 (the "1.5:1" of the research) does not separate them either (1.43-1.49 vs 1.26-1.36).
Switched on in exactly the 15 dark-on-dark scenes of 60.

Confidence: `|2a - 1| x (0.5 + 0.5 (1 - |a - ridge|))`, ridge = `|L - G(L)| / (0.25 max(p99 - p1, 0.02))` clamped
(G: Gaussian sigma 2 px at 1080 lines); 0 outside the region.

Sizes scale with s = delivered height / 1080 (proxy included); the eval passes s = 1 (its scenes are 1080-line crops).

## Evaluation (real `mr_ml_runner`, CUDA, the GPU host; the research's 60 scenes, same seeds and metrics)
`tests/eval_scenes.py gen` writes the scenes, `mr_model_eval` runs the variants, `eval_scenes.py score` scores them in
the R = 12 region (mass excluded). Mean over 60 scenes: cover / IoU / SAD/100 / gaps open / leak.

| variant | small | base |
|---|---|---|
| plugin 2 Edge Refine, defaults (Band 10, Outer 0) | .00 / .00 / 12.99 / 1.00 / .002 | .00 / .00 / 12.99 / 1.00 / .001 |
| plugin 2 Edge Refine, Outer 100 | .59 / .55 / 9.19 / 1.00 / .104 | .64 / .61 / 7.67 / .89 / .067 |
| region crop, 0.5 / 0.5, no pre / post (research baseline) | .70 / .71 / 4.70 / .97 / .016 | .72 / .74 / 4.82 / .89 / .009 |
| + ImageNet | .80 / .81 / 3.26 / .97 / .007 | .84 / .84 / 3.24 / .84 / .003 |
| + unsharp | .88 / .86 / 2.61 / .96 / .007 | .91 / .88 / 2.39 / .92 / .001 |
| **full pipeline (default)** | **.94 / .91 / 1.91 / .98 / .005** | **.92 / .91 / 1.90 / 1.00 / .001** |
| full, unsharp also when stretched | .95 / .90 / 2.02 / .94 / .006 | .93 / .91 / 1.90 / 1.00 / .001 |
| full + Strand Gain (black .05, gain 1.33) | .99 / .91 / 1.45 / .98 / .002 | .97 / .92 / 1.44 / 1.00 / .000 |

Dark on dark, small: .48 / .35 / 8.27 (baseline) -> .89 / .82 / 3.09 (full); the others .91-.99 cover. Runtime: about 8 ms
per 192 px scene including the runner round trip. Leak = mean alpha > 2 px from any strand inside the region; outside the
region the output is the base bit for bit (tested). Limits: the research's (opaque synthetic strands, no motion blur or
compression); the low-contrast thresholds come from four looks only.
