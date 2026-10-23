# ofx/src/ops/CLAUDE.md — the matte operations of Matte Refine

Pure functions on float mattes (w*h, 0-1, row 0 = top), no OFX types (root decision 3).
Plugin wiring, colour and Flame behaviour are in `ofx/CLAUDE.md`; goal, architecture and the
decision index in the root `CLAUDE.md`. Tags as there. Keep this file under ~250 lines.

## Decisions

### Operation 1: Fill Holes

- **7. Principle:** never lower a pixel, and never touch the background connected to the outside
  of the frame. That is why it doesn't damage the outer edge the way dilate/close does.
- **8. Params and defaults (DECIDED, user 2026-10-21).**
  Group "Fill Holes": **Enable** (on), **Threshold** (labelled "Solid Above", since Flame may
  not show hints; **0.95** (user, 2026-10-21, after Flame test 1: 0.97 made a keyed product
  shot solid; was 0.5): matte ≥ it is solid), **Max Hole %**
  (1.0 % of frame area; 0 = no limit; protects real gaps like an arm on a hip), **Bridge Gaps**
  (labelled "Bridge Gaps px"; 0 px: also fill holes whose solid edge has a gap up to about this
  width; the closing radius is half of it), **Rim** ("Rim px", 1 px: lift
  the soft inner rim around a filled hole). Group "Fill Holes Edges": **Left / Right / Top /
  Bottom Closed** (off: a closed frame edge counts as solid, for a subject cut by the frame).
- **9. Algorithm.** S = matte ≥ Threshold; closed frame edges act as a solid border.
  1. **C** = S closed by Bridge Gaps (dilate by r, then erode by r, Euclidean disc; r = 0 →
     C = S).
  2. **Outside** = flood fill (4-connected) over non-C pixels from the non-C pixels on the
     open frame edges. 4-connected background = 8-connected solid: a white edge that only
     touches diagonally still seals.
  3. **Candidates** R = not S and not Outside; label R's components (4-connected).
  4. **Fill** a component if it contains a pixel that is not C (a real hole, possibly reached
     through a bridged gap), or if it touches no Outside pixel (a small hole the closing
     swallowed). Don't fill a component that is all C and touches Outside: that is an outer
     notch the closing filled, i.e. the outside edge.
  5. Drop components larger than Max Hole % of the frame.
  6. **Rim**: grow the fill mask by Rim px into **solid pixels only** (never into Outside or a
     hole left open), then out = max(matte, mask). Without it a grey ring at about Threshold
     remains between the filled hole and the solid. Where the solid wall is thinner than Rim,
     the lift reaches the outer edge's soft pixels ≥ Threshold (accepted).
  Known limit: a hole smaller than the bridge width *and* open through a gap is indistinguishable
  from an outer notch and stays open. Bridged gaps fill up to the closing's surface at the gap's
  mouth (expected: that is what bridging means).
  **Implementation (rewritten 2026-10-23, output bit-identical, checked against a pasted copy of
  the old code on synth scenes and 60 random mattes):** pixel sets are horizontal runs found on
  threads and joined with union-find (4-connected). Bridge 0: one labelling of not-S; the
  components touching an open frame edge are the outside, every other one is a hole candidate
  (it has core pixels and never touches the outside, so the notch rule cannot trigger).
  Bridge > 0: the closing (dilate, invert, dilate), not-C runs for the outside, a per-pixel
  class byte, then the candidate runs with "has a core pixel" / "touches the outside" flags.
  Rim = a disc stamped from the fill's boundary straight into the output, exact because a
  pixel's nearest set pixel is always a boundary pixel; `dilate()` instead when boundary × disc
  exceeds the frame. Both paths are checked against brute-force references.
  **Measured** (`test_fill_holes`, UHD subject with 200 holes), akflame5 (Xeon, 48 threads):
  runs version 2.9 ms (Bridge 0), 11.8 ms (Bridge 8); before: 49–70 / 69–83 ms; first version
  (per-pixel BFS, full-frame distance transforms) 92 / 175. Bridge 8 is mostly the two `dilate()`
  calls of the closing (now threaded, 2026-10-23: UHD r=8 4.4 → 1.1 ms on BFD).
- **10.** Holes are filled with **1.0** (fully solid), not with neighbouring values.

### Operation 2: Solid Core

- **15. DECIDED (user, 2026-10-21): Solid Core, its own section after Fill Holes.** Found in
  Flame test 1: with Solid Above 0.97 the center filled, but a side face keyed semi-transparent
  (0.6–0.97) stayed grey. It is a broad soft band joined to the outer edge (row profile ×100:
  `2, 40, 85, 96, 97, 97, 98 … 99, 95, 77, 82, 98`), so no hole rule can tell it from a soft
  edge. Solid Core: every pixel farther than **Depth px** (6) from any pixel below **Edge At**
  (0.5) becomes 1.0, so the outer Depth px of edge are bit-identical and real gaps (below Edge
  At) keep a Depth px margin. Pixels beyond the frame don't count as edge (a subject cut by the
  frame cores up to it). **Enable** defaults off: it also solidifies legitimately
  semi-transparent areas deeper than Depth. On the test frame (2880×1620, dump t36): pixels
  below 0.9 more than 8 px inside the subject 3522 → 16, edge band unchanged, none darker.
  Measured: UHD 18 ms (BFD), 36 ms (akflame5). Fill Holes + Solid Core at UHD on akflame5 can
  exceed 100 ms with I/O; at 2880×1620 about 75 ms. Feather 0 (16) is this hard core, bit for bit.
  **Edge At min 0.01** (Flame test 3, 2026-10-21): at 0 nothing is below it, so the whole frame
  became core (4.2M px white). Same test: "stubborn holes" were grey slivers (0.45–0.9) with
  Solid Above lowered to 0.48, i.e. solid by definition; at 0.95 all filled, nothing outside
  the 0.5 contour changed. Lesson: when params "make no difference", check the threshold
  against the hole's values first (dump + log line).

- **16. DECIDED (user, 2026-10-22): soft core, Feather px** (default 2, 0 = the hard core of
  15). The hard core aliased: its edge is a pixel staircase, and where it crosses a grey area
  (t36's side face) it left a jagged step up to 0.47 between neighbours. Now out = max(in,
  smoothstep((D − Depth) / Feather)) where D > Depth; D = min over pixels q below Edge At of
  |p − q| − s(q), s = how far the Edge At contour lies beyond q's centre (per axis t by linear
  interpolation towards the above 4-neighbours, s = 1/√Σ1/t²). D ≤ the grid distance, so the
  outer Depth px stay untouched; never darker. **Why sub-pixel:** a ramp on the plain grid
  distance spread the step but its error stayed ~0.15 (the step itself); refined 0.02–0.08
  (test, three slants; visibly smoother on t36). `ops/morphology` `distanceWithin` stamps
  a disc of r + 1 on boundary pixels (exact); its fallback when stamping costs more than a
  frame (large Depth on a noisy matte; t36 hits it only at Edge At 0.95) is the plain grid
  distance, so there the edge is less smooth. Measured UHD: BFD 27 ms (hard 18), akflame5 37
  ms (hard 36), M1 Ultra 27 ms. Saved 0.1.1 nodes get Feather 2 (slightly softer core edge).

### Operation 3: Edge Refine (roadmap)

- **12. Roadmap, not designed** (user, 2026-10-21: needs its own conversation and research).
  Refine an existing matte's edges with a trimap-based matting model: trimap from the input
  matte (erode → solid, dilate → unknown band), the model solves the band. To research: models
  and their licences (ViTMatte, MatAnyone, Robust Video Matting, others), temporal stability,
  GPU need on Rocky and Apple Silicon, per-frame time and memory at HD/UHD (it must render inline,
  14), and how it runs from `render()` (in-process vs a resident helper). Order in the chain
  relative to Fill Holes: OPEN.

### Plugin 1 operations (2026-10-22, building; decisions 17-19)

Built by parallel agents from briefs (2026-10-22), merged and reviewed; per-op details are in the
headers and tests. **Speed work 2026-10-23 (Opus agents, every output bit-identical to the old code,
checked float for float with the old code pasted into the tests):** full chain, every op on at
defaults, synth "all", **UHD akflame5 275 → 82 ms, HD 43 ms** (BFD UHD 117): clean 6, fill 2, core
4, smooth 7, grow (+1 px) 6, width 6, snap 23, unmix 26 ms; the last two are the big ones now
(Snap's two passes, Edge Colour Unmix with Ignore Outer px). Decision 5's ~100 ms holds with all on.
All ops off by default.

- **A1+A2 Clean & Crush** (`clean_background`): islands = 8-connected parts ≥ Background Below
  (0.15, above haze, so haze can't glue islands to the subject), removed up to Max Island %
  (0.05) with their own fringe (never-rising path, 2 px, not next to a kept part); Crush =
  1 − Solid Core(1 − m) at the same level (Crush Beyond px 4, Feather 2). Known limit: a
  see-through area below 0.15 inside the subject (glass) is crushed. Its `crushSoft` stays: the
  shared `distanceWithin` is bit-identical but 1.6× slower here (frame-sized float buffer plus a
  separate crush pass); dynamic 32-row bands, row-clipped disc. UHD 40 → 5.5 ms (akflame5).
- **A3 Shrink / Grow** (`shrink_grow`): Grow (-Shrink) px; the profile at sd + Amount along the
  normal, 6 + |A| px rigid then faded; passes of ≤ 4 px; strands < 3 px (3×3 opening) kept;
  grow never lowers, shrink never raises. Fixed 3 px / 6 px thresholds don't scale yet.
- **A4 Contour Smooth** (`contour_smooth`): local quadratic fit per contour sample (Smooth px 30,
  Max Shift 1.5, Corner deg 20, Edge At 0.5). Safe (clean edges, discs, thin parts, hair
  bit-identical) but modest: synth chatter 0.101 → 0.091; creases < Corner soften ~0.6 px;
  skips noisy mattes. Judge on real chatter before more work.
- **A5 Edge Width** (`edge_width`): per side extent = 2 × area / level span, remapped to a linear
  ramp Edge Width px (2) through the 0.5 contour, Width Mix; levels kept, thin parts untouched,
  edges within 3 % bit-identical. A hard cut keeps its staircase (Contour Smooth runs first).
- **A6 Snap to Image** (`snap_to_image`): colour guided filter (He et al.) on 64 px tiles near the
  0.5 contour, ε scaled by brightness (exposure-invariant), a confidence κ keeps the matte where
  the picture is flat; Radius 8, Band 12, Detail 0.5, Edge From Picture 0.35. After Flame test
  1: the band also covers sloped grey matte (rings / dots at blocky tips gone) and two passes
  (outside tail about halved; coarse mean error 0.115 → 0.052; good matte max change 0.26).
  UHD 34-39 ms (loaded); `kPasses = 1` halves it but loses the tail gain.
- **A7 Foreground Colour** (`foreground_colour`): Germer et al. multi-level F/B solver, red-black
  SOR, full resolution only in the soft band + Extend px; alpha 1 bit-identical, alpha 0 = fg
  extended (no old background). fgcolor scene: 0.19 → 0.002. Shown by View Picture. Worst case
  (matte soft everywhere) 77 ms UHD.
- **26. DECIDED (user, 2026-10-23 test; built by an Opus agent): Edge Colour Unmix "Ignore Outer
  px".** The unmix trusts the matte: alpha-1 pixels pass through, and Extend spreads their
  colour. After Shrink/Grow or a hard Edge Width pushed the matte past the true edge, those
  pixels still hold the old background (green edge; a bigger Extend a bigger halo; user Flame
  test 2026-10-23, grown +4.8 px: greenness +0.53 inside the edge vs −0.04 clean). Now pixels
  at alpha ≥ 0.5 within N + 0.5 px of the 0.5 contour lose their data term at every solver
  level and take their colour from the interior (weight 1 solid neighbours, 0.05 soft ones);
  they are output like soft pixels. The deep interior stays the picture bit for bit; N = 0 is
  bit-identical to 0.2.0 (hash test, x86-64 only). A part narrower than ~2N has no core: it
  keeps N = 0 behaviour (else it came out black). Result, grown 4.8 px, N 6: −0.03 inside, ≤
  +0.01 outside. Cost on a clean matte, N 3: colour error in the soft band 0.0021 → 0.0056,
  recomposite mean 0.0006 → 0.0023; UHD 20 → 27 ms (N 6: 31, N 20: 48). Default **3** (covers
  a matte ~2 px too big); **the plugin adds Shrink / Grow's amount (when on and positive) to N**,
  so grow done in the node just works; N must be ≥ the growth + ~1 px otherwise (N 3 on a 4.8
  grow does nothing). Auto-detecting N from the picture was evaluated and rejected (real
  foreground changes colour at its edge too; it is alpha estimation, decision 17; would flicker).
  Known limit: a grown strand narrower than 2N keeps its green. Analyze Frame (A9) may suggest N.
- **A9 Analyze Frame** (`analyze_frame`; built by an Opus agent 2026-10-23; button in the Matte
  column, plugin.cpp `analyzeFrameButton`): analyses the input matte and sets Clean & Crush, Fill
  Holes and Solid Core only (not the edge or colour ops); a popup lists one line per decision and
  why. Refuses (changes nothing, says why) on a constant, empty, noisy (> 25 % of pixels in
  0.15-0.85, or > 10 % on an edge) or tiny matte. Solid Above: every threshold 0.50-0.99 is
  scored by one priority flood ("escape level" = the lowest wall to an open frame edge); 0.95
  is kept when it fills ≥ 90 % of the best, else the middle of the range that fills ≥ 97 %.
  Max Hole % sits between the largest hole and the smallest real gap (size jump ≥ 4×), the
  default kept when it already has 1.5× room; Closed edges only when holes were found (else the
  synth legs would change clean mattes); Bridge from a half-resolution trial; Solid Core on when
  grey below 0.9 remains > 8 px inside after Fill Holes, Depth = 2 × the ring where the grey
  share falls halfway (t36: 6); Clean & Crush when islands (below the largest size jump) or haze
  exist. It runs the real ops to verify (the "Left: N" in the notes is their count). Real
  frames (user's t36 and t44): Solid Above 0.95 on both, t36 Core Depth 6 (= the settled
  settings) plus Bottom Closed (fills 150 / 23 holes cut by the frame; nothing changes outside
  the subject); clean synth scenes: nothing changed. UHD worst case ~175 ms (BFD); a button, not
  per frame. Cannot tell: see-through parts (glass) from grey defects, a real small pocket from
  a hole without a size jump, an object from debris, frame-to-frame stability. "Learn from use"
  recording (roadmap) not built.
- **A8 views** (decision 11, indexes kept): 0 Picture (was Source; = clean fg when A7 is on), 1
  Show Changes (was Show Filled; lowered pixels green), 2 Matte, 3 Trimap (≥0.98 / edge / ≤0.02),
  4 Grey Inside (0.02-0.98 more than 4 px inside the 0.5 contour, red).

Original list: A1 **Clean Background** (remove islands below a size) + A2 **Background Crush** (everything N px
outside the edge → 0, soft, the mirror of Solid Core), A3 **Shrink / Grow** keeping the edge's
softness, A4 **Contour Smooth** (wiggle off the edge line, profile kept), A5 **Edge Width**
(one consistent falloff width), A6 **Snap to Image** (guided filter, edge band only, needs the
picture), A7 **Foreground Colour** (unmix the edge's colour, on RGB), A8 **QC views** (trimap,
grey inside, islands, changed), A9 **Analyze Frame** (sets the controls; Roadmap entry below).

- **20. Shared edge geometry** (`ops/edge_field`, `morphology` `distanceWithin`): the signed
  distance to a matte's `level` contour, sub-pixel, both sides. Near each contour pixel the
  contour is a tangent segment (±0.5 px) through its foot point: normal from central differences
  of the matte (exact on ramps), distance from the one-sided crossings (exact on 1 px edges).
  Decision 16's circle model overestimated by up to 0.29 px at 45°; the first segment version
  took its normal from the across-neighbours only and tilted up to 30° on slants. Measured (4 px
  soft disc): ≤ 0.09 px, mostly ≤ 0.005; straight edge 0.027 px; 1 px edges ~0.18 (the
  sampling limit). Each distance is capped at the grid distance, so decision 16's "outer Depth
  px untouched" holds strictly. `shiftAlongNormal` moves an edge keeping its profile by
  interpolating in distance space (bilinear in x/y is off by up to 0.25 on a 1 px edge). Budget:
  stamping up to 2 frames of writes (measured break-even with the distance transform at UHD).
  **Shared fast path (2026-10-23, `morphology.h` / `edge_field.h`, additive; old signatures keep
  their behaviour):** `Workers` (one thread pool per op call: a loop costs 0.015 ms instead of
  0.28 ms when threads start per loop), `parallelFor(Workers*, …)`, `dynamicFor` (uneven items),
  a raw-pointer `distanceWithin` overload with a frame budget, `stampRow` (stamping written so GCC
  vectorises it), threaded `dilate` with a word-skipping boundary scan; in `edge_field`:
  `signedEdgeDistanceExact` (12-frame budget: the edge ops need the exact distance at larger
  bands, where `distanceWithin`'s 2-frame budget falls back to the plain grid distance),
  `contourPixels`, `ContourIndex` (16×16 tiles replaces full-frame measure arrays: Edge Width's old
  array was 232 MB), `NearTiles` (band work visits only 32 px tiles near the contour), plus the
  helpers that lived in `shrink_grow.h` (`edgeNormal`, `profileAt`, `isContour`, `contourSlope`,
  `footMean`). Snap and Unmix still carry their own `parallelFor` copies. Equivalence tests:
  `ofx/tests/old_edge_ops.h` (the pre-speed code, git c739311) and `equivalence.h` (random
  mattes). Bit-identity verified with GCC on x86-64 and with clang on Apple Silicon (arm64, M1 Ultra, macOS 26:
  all 13 suites incl. every equivalence test; UHD Fill Holes 1.9 ms, Bridge 8 7.6 ms, Solid Core 6.3 ms). Left alone: scratch buffers kept between calls (~1-2 ms per op in page faults).
- **21. Synthetic mattes** (`ofx/tools/synth`, `mr_synth`): a figure from distance shapes (arm on
  hip enclosing a real ~1.0 % gap (just under the 1 % default), legs cut by the frame, hair strands) with per-scene defects
  and the clean truth: holes, specks, chatter, mixed_edge, eroded, coarse (+ picture), fgcolor
  (+ true foreground), all. Tests use them in memory; `mr_synth <dir> [WxH]` writes EXR (matte
  as grey RGB for the source alpha input, picture, truth) for evaluation and demos in Flame.
