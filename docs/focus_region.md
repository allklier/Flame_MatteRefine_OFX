# Focus Region (Matte Merge fusion node) — design, behaviour, tests, limits

Written 2026-11-12 by the Focus Region agent. Status: the parts that are already certain are BUILT (cores,
Matte Merge socket and controls, tests); candidate (b) and the ML trimap wiring are not. Decisions listed
under "For the main thread" are PROPOSED until recorded in `ofx/merge/CLAUDE.md`.

## Idea
A user-drawn soft roto, the **Focus Region**, is the only place a fusion step may change the matte. Matte
Merge becomes the fusion node: picture (front), base matte (source alpha), optional Detail matte, and an
optional **Focus Region** mask socket. Inside the region candidates may be combined; outside the output is
the base, bit for bit.

## What is built (files)
| File | What |
|---|---|
| `ofx/merge/src/merge_fusion.{h,cpp}` | `Candidate`, `FusionParams`, `FusionRule` + `defaultFusionRule`, `fuseCandidates`, `prepareRegion`, `snapInRegion`, `snapParamsForHeight` |
| `ofx/merge/src/matte_merge.{h,cpp}` | `FocusInput` and the region path in `merge(..., focus)`; `focus = nullptr` is the old path |
| `ofx/merge/src/merge_ui.{h,cpp}` | View "Region" (index 6, appended), `UiValues` region fields, `regionPx`, region footers |
| `ofx/merge/src/plugin_merge.cpp` | clip `FocusRegion` (mask, optional), column "Focus Region", render wiring |
| `ofx/src/ops/snap_to_image.{h,cpp}` | optional `weight` (last argument): `out = in + w (snapped - in)` |
| `ofx/src/ops/region_trimap.{h,cpp}` | `regionTrimap(base, region, ...)` for Edge Refine ML (not wired) |
| tests | `ofx/merge/tests/test_merge_fusion.cpp` (new), `ofx/tests/test_region_trimap.cpp` (new), `test_snap_to_image.cpp` (+ `testWeight`), `test_merge_ui.cpp`, `systest/merge_systest.cpp` |

## Merge with a region (`merge(..., const FocusInput* focus)`)
1. The automatic zone (Matte edge: Reach around the base edge minus solid core; Trimap: unknown band) is
   computed as before and used **only for the alignment** (the move is global; a small region alone had too
   little thin detail: the systest scene fell back to "no alignment").
2. The zone = pixels with region weight > 0; the Feather ramp of the core is not used (the weight is the ramp).
3. Optional fusion (only with extra candidates): candidate (a) = the aligned Detail matte, confidence = picture
   evidence on its thin structures when Evidence is On (1 on thick parts; nullptr = 1 when Off); the candidates
   are fused (`fuseCandidates`) and the fused map is the detail for connectivity, the thin-structure (top-hat)
   step, Bridge and its strand mask. Evidence is then not applied a second time. Without extra candidates
   (today: slot (b) empty) the Detail matte goes through the old steps unchanged.
4. Combine as before (Add Fine Detail / Zone Blend / Trimap fill), then
   `out = base + w (merged - base)` (w 0: base, w 1: merged, exact) and, with Source Matte Edge,
   `out = max(out, base)`: never below the base, so a Zone Blend in a region can only raise. Trimap keeps its fill.
5. Views: Zone = the region weight; Added and Rejected after the weight.

## Fusion rule (default; swap by passing another `FusionRule`)
Per pixel, gated value `g = alpha x confidence` (both clamped 0-1, NaN 0); on the two best:
- one candidate: `g`;
- agree (`|a1 - a2| <= agreeTol` 0.1): `max(a1, a2) x c`, `c = c1 + agreeBonus (1 - (1 - c1)(1 - c2) - c1)` (bonus 1);
- strong disagreement (`> disagreeGap` 0.4) and neither confidence `>= highConf` 0.7: `g1 (1 - penalty)` (0.5);
- else `g1`.
The base is not a candidate; the final step never lowers it. A zero candidate next to the Detail matte
leaves the result unchanged (tested), so adding candidate (b) cannot disturb pixels where it has nothing.

## Region preparation (`prepareRegion`)
Mask clamped to 0-1 (NaN 0). Grow: grey dilation (> 0) or erosion (< 0) with a (2r + 1) **square**,
r = round(|Grow px|). Feather: an **inward** ramp, `w x smoothstep((d - 0.5) / feather)`, d = distance to the
nearest pixel with weight 0: never reaches beyond the drawn (grown) mask. Grow 0 and Feather 0: the mask as is.

## Node controls (column 5 "Focus Region", 4 rows; the node now uses all 6 columns of one tab)
| Param id | Label | Default | Range | Notes |
|---|---|---|---|---|
| `regionGrow` | Region Grow % of height | 0 | -2 to 2 | % like Merge's other sizes (not px) |
| `regionFeather` | Region Feather % of height | 0.25 | 0 to 2 | inward |
| `regionSnap` | Snap to Picture | off | | Snap to Image inside the region, never lowers (Matte Edge) |
| `regionSnapStrength` | Snap Strength | 1 | 0 to 1 | weight = strength x region |
View: "Region" appended (index 6). Clip id `FocusRegion`, label "Focus Region", mask, optional, after Detail.
Snap sizes: plugin 1's defaults (radius 8, band 12 px) as 1080-line values, scaled with the height.

## Behaviour table
| Focus Region | Detail | Result | Footer |
|---|---|---|---|
| not connected | any | exactly the node without the feature (no new code runs) | as before |
| connected, not empty | connected | merge inside the region, base outside, optional snap | info + " Focus Region: N px." |
| connected | missing | base passes through (no candidate yet); View Region still shows the region | "Focus Region connected, but no candidate: connect a detail matte to the Detail input. The base matte passes through." (warning) |
| connected, all black | connected | base passes through | "The Focus Region is black everywhere: the base matte passes through." (warning) |
| connected, wrong size / not float / no image | any | base passes through | error footer |
| any | Source problems (no alpha, constant base) | as before (they win) | as before |

## Region trimap (`ofx/src/ops/region_trimap.h`, for Edge Refine ML later)
`L = 1` where base >= fgThreshold (0.95), else 0 (grey base pixels outside the region are background, by
design); `u` = region weight, raised to 1 within Band px of the region's 0.5 line; `out = L + u (0.5 - L)`.
Band in px of the delivered image: the caller scales it with the render scale (decision 6). In `mr_ops` so
plugin 2 and Merge can use it; not wired (plugin_ml.cpp belongs to another agent).

## Tests (2026-11-12, the local Linux test machine, share mode)
- Bit-identity of the old paths: FNV-1a hashes of `snapToImage` (2 settings) and `merge()` (Add, Zone Blend +
  Evidence, Trimap) on the 960x540 synthetic scene, built from the code before and after: identical.
- `test_snap_to_image` `testWeight`: w 0 / >1 / NaN / 1 exact, 0.25 exact, half frame, in place, stats.
  Mutations: blend from `in` in place, `<` -> `<=`, no NaN guard: all caught.
- `test_region_trimap` (smoke): known answers, soft values, band vs brute force, 2x frame + 2x band = 4x area
  (ratio 1.008). Mutations: 3 of 3 caught.
- `test_merge_fusion` (smoke, 0.13 s): rule known answers, fuse map, prepareRegion vs brute force, merge with
  region (0 px changed outside, 0 lowered, exact blend, region replaces the Reach zone, alignment equals the
  run without a region, Zone Blend raises only), fusion (zero candidate identical; strand candidate added
  inside only), snapInRegion. Mutations: 7 caught, 1 not (`wgt < 1` -> `<= 1`: base + (v - base) rounds back).
- `merge_systest`: clips (4, FocusRegion mask optional), layout 6 columns, region defaults; unconnected
  renders = `merge()` without a region (bit for bit) as before; connected = direct path bit for bit; View
  Region / Zone; Snap + Grow + Feather 0; Detail missing; empty region; disconnect again = default render.

## Limits / open
- Grow uses a square window (corners grow boxy at large values); Feather is inward only.
- Snap to Image assumes a scene-linear guide; Merge's picture arrives as Flame delivers it (converted once
  a mask clip is connected). Its epsilon is relative to brightness, so roughly exposure-invariant; check on
  real footage.
- Flame: the socket position and colour of a second mask clip, and whether Flame delivers it Alpha-only like
  Detail: UNVERIFIED.
- Candidate (b): slot only (`FocusInput::extra`). It must be in the picture's pixel grid (it is not aligned).
- With a region and Trimap source, the fill uses the trimap's prior inside the region; a region over a known
  area of the trimap re-fills it from the detail there.
