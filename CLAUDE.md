# CLAUDE.md — Matte Refine OFX for Flame

Design knowledge for anyone changing this plugin, human or AI assistant: what was decided and
why, how Flame behaves as an OFX host, and what was tried and failed. Numbered decisions are
referenced from code comments ("decision 4"); keep the numbers stable. Machine-specific notes
(hosts, paths, accounts, status) go in the git-ignored `CLAUDE.local.md`.

This project follows the rules of the sister project **LTX Alpha Gen** (`~/ClaudeCode/LTX_OFX`,
its `CLAUDE.md` and `ofx/CLAUDE.md`) for design, operation, install and support. Where this file
is silent, those rules apply; where a rule there doesn't apply here, this file says so (decision 2).

Tags: **DECIDED** (maintainer approved), **PROPOSED** (recommended, not yet approved), **OPEN**
(needs a decision), **UNVERIFIED** (not yet confirmed against a real install or source).

## Goal

An OFX plugin for Autodesk Flame on **Rocky Linux 9** (and next **macOS on Apple Silicon**) that
holds a growing list of **matte refinement operations** in one node. Started 2026-10-21; the
first operation is needed for a project the same day.

| # | Operation | Kind | Status |
|---|---|---|---|
| 1 | **Fill Holes**: fill holes enclosed by solid matte, leave the outside edge untouched | algorithmic, in render | v0.1, building |
| 2 | **Solid Core**: everything deeper than N px inside the matte's edge becomes 1.0 | algorithmic, in render | v0.1, soft edge 0.1.2 (15, 16) |
| 3 | **Edge Refine**: refine the edges of an existing matte with a trimap-based model | ML, needs research | roadmap (decision 12) |

## Architecture

```
Flame ──OFX──> MatteRefine.ofx (one effect "Matte Refine", grouping "Matte Refine")
   render(): fetch source → matte = source alpha (4) → op 1 → op 2 … (enabled ones, fixed
             order, 3) → output RGBA: RGB = source (or the Show Filled view, 11), A = result
   ofx/src/ops/:   pure C++ on float mattes, no OFX types; unit-tested without a host
   ofx/src/plugin.cpp: params, layout, render, Collect Logs, version label
```

Source files: `ofx/src/plugin.cpp` (the effect), `ops/fill_holes.*` (op 1), `ops/solid_core.*`
(op 2), `ops/morphology.*` (dilate + distance transform, shared), `log.*`,
`user_config.*` (`~/.config/matte-refine/config.conf`, `log_level`), `env_info.*`,
`process.*` (tar for Collect Logs) and `platform.h` + `platform_linux.cpp` / `platform_macos.cpp`
(OS, CPU, RAM for the snapshot), the last five copied from LTX and trimmed. Tests: `ofx/tests/`.

Plugin id `com.janklier.matterefine`, version from CMake `MR_VERSION` (OFX version = its
major.minor), `third_party/openfx` = OpenFX `OFX_Release_1_4_TAG` (`include/` + `Support/`,
BSD), copied from LTX's vendored copy.

## Decisions

### Cross-cutting

- **1. DECIDED (user, 2026-10-21): one node, operations as sections.** A single "Matte Refine"
  effect; each operation is a param group with an **Enable** checkbox, applied in a fixed order.
  Consequence: the node's layout fills up (Flame: 6 columns of 5 rows per tab, then a second
  tab). All ops, the ML one included, render inline (14).
- **2. DECIDED (user, 2026-10-21): same principles as LTX Alpha Gen.** Carried over: C++14
  against the vendored OpenFX 1.4, libstdc++ dynamic, only the two OFX entry points exported,
  build on Rocky 9 (glibc ≤ 2.34, GCC 11), a macOS arm64 bundle (next, 13); Flame layout and
  `changedParam` rules (below); one log + **Collect Logs**; a Version label on the node;
  `packaging/install.sh` (sudo, refuses while Flame runs, `--uninstall`); release tarballs from
  `scripts/package_release*.sh`; ctest unit tests; a system test with a Flame-like host before
  releases; MPL 2.0. **Not** carried over (14): Process button, disk cache, markers, Flame
  Python hook. Setup / model download / licence gate only if the ML op's model needs them.
- **14. DECIDED (user, 2026-10-21): all matte processing is inline in `render()`; no cache, no
  Process button.** LTX needed its cache and detached jobs because its model is very heavy;
  matting models are light enough to run per frame. Inline keeps this a natural-feeling node
  (scrub, tweak, see the result; Batch Render just works). Applies to the ML op (12) too: it
  must fit a per-frame render budget, which constrains the model choice. Changing this needs an
  explicit maintainer decision: don't slide into a cache (or a "small" result cache) by accident.
- **3. An operation = a pure function + a param group.** `ofx/src/ops/<op>.{h,cpp}`: a params
  struct and `apply(const float* in, float* out, w, h, params)` on a dense float matte, no OFX
  types. `plugin.cpp` defines each op's group, reads its params, chains the enabled ops (output of
  one = input of the next). Adding an op = one ops file pair, one group, one line in the chain,
  its unit test. Order is fixed by the chain, not user-settable.

- **17. DECIDED (user, 2026-10-22): scope = refine, not replace.** The input is a good matte
  (LTX or elsewhere); every operation keeps it and fixes defects. Anything that makes a new
  matte from the picture (MatAnyone, VideoMaMa, GVM, SAM2Matting, SDMatte, keyers like
  CorridorKey) is out, however good. Built-in Flame ops (blur, plain erode/dilate, gamma, clip,
  combine) are not duplicated.
- **18. DECIDED (user, 2026-10-22): two plugins.** **Matte Refine** (this one) stays real-time
  and inline (14). A later **Matte Refine ML** holds the slow and ML options (roadmap B1-B4),
  LTX-style Process + cache, or inline where ~1 s/frame and Flame's cache suffice (Neat-like;
  whether Flame caches OFX results: UNVERIFIED). Decision 14 holds for this plugin.
- **19. DECIDED (user, 2026-10-22): UI = one node, two tabs, one operation per column.** An op =
  Enable + at most 4 controls, off by default; paired ops share a column. Tab 1 "Fill & Core":
  Matte (View, Version) · Clean & Crush · Fill Holes · Fill Holes Edges · Solid Core · Support
  (+ Analyze later). Tab 2 "Edge": Contour Smooth · Shrink / Grow · Edge Width · Snap to Image
  · Foreground Colour. Definition order = chain order (clean → fill → core → smooth → shrink/grow
  → edge width → snap → foreground colour). QC is View choices, not controls. Labels say what
  they do (hints may not show). Built 2026-10-22 (0.2.0); not yet seen in Flame.

### Matte in, matte out

- **4. DECIDED (user, 2026-10-22): picture on the front, matte on the source alpha input.** The
  matte is always the source clip's alpha; Matte From (Alpha / Luminance / Red, 2026-10-21) is
  removed, so the RGB is free for ops that need the picture (Snap to Image, Foreground Colour).
  **Verified** (probe build, Flame 2027 on akflame5, 2026-10-22): a filter's source shows in
  Flame as a front socket plus a "source alpha" socket; a matte there arrives as the source's
  alpha (RGBA, RGB = the front). With nothing on it, alpha is a constant 1.0, so the footer
  asks for the matte when the input matte is constant. An extra optional OFX clip (mask, General
  context) appeared as a third socket and received the matte too: redundant, removed. Output
  alpha = the refined matte; output RGB per View (11). Flame forces RGBA output; alpha is the
  node's matte output (LTX decision 48). Saved nodes that used Luminance now read alpha.
- **22. DECIDED (user, 2026-10-22, Flame test of 0.2.0): the matte has its own input; the
  picture is optional.** With the matte only on Source's alpha, an unconnected front left the
  node without size or colour info and it failed. Now: a "Matte" mask clip (Flame shows it as
  an extra socket, seen in the probe), Source optional (General context only: a filter's Source
  is mandatory), output RoD and PAR from the matte. Without a Matte input the matte is Source's
  alpha (decision 4's route still works). Snap to Image and Foreground Colour need the picture
  (same size as the matte), else they switch off with a footer note. Default View = Picture
  (the picture passes through, the refined matte without one). UNVERIFIED in Flame: the
  socket labels, colour-space info without a front input, and old filter-context nodes.
- **5. Render is synchronous and per frame** (14). Ops run in `render()` on Flame's main thread
  (all OFX actions do), so the algorithmic chain must stay well under ~100 ms at UHD. No
  temporal processing. Input outside 0–1 (float) is clamped to 0–1 before the ops.
- **6. Pixel-sized params scale with the OFX render scale** (proxies match); percentages don't.

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
  **Implementation:** outside = scanline span fill (a per-pixel BFS cost 2x); dilation (closing
  and rim) = a disc stamped on each boundary pixel of the mask, exact because a pixel's nearest
  set pixel is always a boundary pixel; when boundary × disc exceeds the frame's pixel count
  (large r on a noisy matte), the Felzenszwalb-Huttenlocher distance transform instead
  (columns in blocks of 16, threads). Both paths are checked against a brute-force reference.
  **Measured** (`test_fill_holes`, UHD subject with 200 holes): akflame5 (Xeon 658X, 48 threads)
  49–58 ms with Bridge 0, 69–78 ms with Bridge 8; BFD 36–40 / 55 ms. First version (per-pixel
  BFS, full-frame distance transforms) took 92 / 175 ms on akflame5.
- **10.** Holes are filled with **1.0** (fully solid), not with neighbouring values.
- **11. DECIDED (user, 2026-10-21): View** choice for the RGB output; alpha is always the result.
  **Matte** (default since Flame test 3: RGB = the refined matte, so both outputs agree), **Show
  Filled** (RGB = the input matte in grey, added fill (out − in) tinted magenta (1, 0, 1)),
  **Source** (RGB pass-through, for a front + matte chain). Why: with Matte From = Luminance the
  front input *is* the matte, so a pass-through RGB showed the unrefined matte next to the
  refined alpha (user confused in test 3). Indexes: 0 Source (was "Result" = source), 1 Show
  Filled, 2 Matte, so saved nodes keep their output. Render-only.

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
headers and tests. Measured UHD, akflame5, defaults: clean 40, fill 68, core 22, smooth 55,
grow 27, width 24, snap 17, fg 22 ms: **~275 ms with all on**, over decision 5's ~100 ms. All
off by default; 2-3 ops ~60-150 ms at UHD, ~1/4 at HD. Speed work is next (fill, smooth, clean).

- **A1+A2 Clean & Crush** (`clean_background`): islands = 8-connected parts ≥ Background Below
  (0.15, above haze, so haze can't glue islands to the subject), removed up to Max Island %
  (0.05) with their own fringe (never-rising path, 2 px, not next to a kept part); Crush =
  1 − Solid Core(1 − m) at the same level (Crush Beyond px 4, Feather 2). Known limit: a
  see-through area below 0.15 inside the subject (glass) is crushed. Its `crushSoft` duplicates
  the now threaded `distanceWithin`: replace with a call.
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
- **21. Synthetic mattes** (`ofx/tools/synth`, `mr_synth`): a figure from distance shapes (arm on
  hip enclosing a real 1.5 % gap, legs cut by the frame, hair strands) with per-scene defects
  and the clean truth: holes, specks, chatter, mixed_edge, eroded, coarse (+ picture), fgcolor
  (+ true foreground), all. Tests use them in memory; `mr_synth <dir> [WxH]` writes EXR (matte
  as grey RGB for the source alpha input, picture, truth) for evaluation and demos in Flame.

### Platforms

- **13. DECIDED (user, 2026-10-21): Rocky 9 today, macOS next.** The ops stay portable (no
  Linux-only calls). Built 2026-10-22 (0.1.2): `scripts/build_mac.sh` (as LTX's), bundle
  `Contents/MacOS/MatteRefine.ofx` (arm64) + generated `Info.plist`, exports via the vendored
  `osxSymbols`, links only libc++/libSystem; `CMAKE_OSX_DEPLOYMENT_TARGET` 15.0 (else the build
  Mac's version: minos 26); ctest 3/3 on an M1 Ultra. Install dir `/Library/OFX/Plugins`.
  NOT yet tried in Flame on a Mac; no release package script for it yet.

## Flame host behaviour (from LTX `ofx/CLAUDE.md`, verified there on Flame 2027.1/2027.2)

- Every OFX action, render included, runs on Flame's **main UI thread**. Flame renders the current
  frame on Batch load and on every frame change, revisit and param change. Frames are **32-bit
  float RGBA**; OFX time starts at 0.
- **Every plugin `setValue` comes straight back as `changedParam`**: reacting recursed and
  crashed Flame. Handle only `eChangeUserEdit`, use a re-entrancy guard, write labels only when
  they change.
- **Layout:** params flow into columns of 5 rows, up to 6 columns per tab, in definition order;
  each group starts a column; pages are ignored (a new tab only on overflow, named after the next
  OFX page); rows: choice 2, boolean 1, double 1, button 1, plain string 1 (no label), label
  string 1; text cut at ~30 characters. Don't use `setPageParamOrder` (throws). Int rows: UNVERIFIED.
- `getClipPreferences` → Alpha-only is ignored: output stays RGBA (image + matte outputs).
- Persistent messages show in the footer; `sendMessage` shows popups. Flame may not show param
  hints, so labels say what they do.
- **Doubles without `setDigits`/`setIncrement` showed as integers** (Flame test 1, 2026-10-21:
  Solid Above couldn't be set between 0 and 1). Now digits 3/2/1, increment 0.01/0.1/0.5
  (`ofxsParams.cpp:494-503`); whether Flame honours them: UNVERIFIED.

**Planned layout (tab 1):** Matte (View 2, Version label 1) · Fill Holes (Enable,
Threshold, Max Hole %, Bridge Gaps, Rim) · Fill Holes Edges (4 booleans) · Solid Core (Enable,
Depth px, Feather px, Edge At) · Support (Debug Logging, Collect Logs, Save Matte). Five columns; one left
before tab 2.

## Logging and support (LTX decision 39, scaled down)

- One log `~/.cache/matte-refine/logs/matte-refine.log` (rotated at plugin load, 10 MB × 5),
  LTX's line format (`YYYY-MM-DD HH:MM:SS.mmm LEVEL comp[pid/tid] node=<id> | msg`); environment
  snapshot at load (version + git, host, OS, glibc, CPU); INFO for user actions and param edits;
  DEBUG for renders with per-op timings. Never logged: image data, clip or project names.
- **Collect Logs** button → `~/matte-refine-support-<ts>.tar.gz`.
- **Save Matte** button → this frame's input matte as raw float32 (top row first) + a `.txt`
  header in `~/.cache/matte-refine/dumps/`, for exact analysis of a failing frame. Footage can
  be confidential: dumps stay on the machine; analyse them there or in a private scratch
  folder, never upload or commit them.
- DEBUG render line: settings, source channels, input matte min/mean/max, holes filled / too
  large / notches kept, % below threshold, % outside, per-op time.
- Version label on the node ("0.1.1 (git hash)") so a Flame test shows which build is loaded.

## Building, testing, releasing

- Linux release builds on the Rocky 9 build host (`HOST=<host> scripts/build_remote.sh`); local
  test builds on any Linux with `cmake -S . -B <build>` (not loadable in Flame: newer glibc).
  Mac: `HOST=<mac> REMOTE=<folder> CMAKE=<cmake> CMAKE_ARGS=... scripts/build_mac.sh` (on macOS
  26 pass `-DCMAKE_OSX_SYSROOT=<CLT>/SDKs/MacOSX.sdk` and `-stdlib++-isystem <SDK>/usr/include/c++/v1`
  in `CMAKE_CXX_FLAGS`, as LTX).
- **Unit tests (ctest, 3):** `solid_core`: core white, edge band and a gap's margin untouched
  (hard and soft), thin strip untouched, frame border not an edge, 30 random cases against brute
  force, hard and soft (both paths each), aliasing on slanted antialiased edges (soft < 0.09 off
  the ideal, hard > 0.14: the test sees the staircase), UHD timing; a halved-depth mutation
  fails it.
  `fill_holes` on synthetic mattes: simple hole, outer soft edge
  bit-identical, Max Hole % at the limit, diagonal seal, closed frame edges, bridged gap vs outer
  notch, a hole swallowed by the closing, rim distance, never lowers a pixel (noise), in place,
  48 random cases against a naive reference (both dilation paths), UHD timing (< 100 ms with
  Bridge 0). Mutations of the notch rule, the diagonal seal, the span fill and the disc each
  fail it. `load_bundle`: dlopen + plugin id. Render scale (6) is not unit-tested (plugin code).
- **System test:** NOT BUILT YET (after the first Flame test): a trimmed copy of LTX's
  Flame-like host (`systest/host`), fast tier only (no model): load/describe, defaults, render
  through the host, `setValue` recursion guard.
- **Installer:** `packaging/install.sh` puts `MatteRefine.ofx.bundle` in `/usr/OFX/Plugins`
  (macOS: `/Library/OFX/Plugins`, quarantine flag removed). No Flame Python hook.

## Roadmap

- Before a release package: the system test, `scripts/package_release.sh`, README for users.
- **Analyze Frame** (DECIDED for plugin 1, user 2026-10-22; was PROPOSED). A
  button that analyses the current frame and sets the controls, with a popup saying what it
  chose and why. No LLM: plain C++ statistics on the matte. Solid Above just below the
  interior's solid plateau, above the grey defects (the setting that went wrong in tests 1 and
  3); edge Closed switches for the frame edges the subject touches; Max Hole % at the largest
  size jump between defects and real gaps (0 if none); Solid Core only if grey remains deep
  inside after Fill Holes, Depth from where it starts; Bridge Gaps 0 unless holes leak through
  narrow gaps. ~2–3 h with tests; the saved dumps (tests 1 and 3) are the first cases.
  **Learn from use instead of up-front tuning:** each press records the frame's matte (as Save
  Matte), the analysis and the suggested values; the values the user settles on are recorded
  later (on the next Analyze, Save Matte, or when the node is destroyed). The pairs (suggested,
  settled) tune the heuristics over time. Footage is confidential: records stay on the
  machine (`~/.cache/matte-refine/analyze/`, size-capped), never uploaded; Collect Logs
  includes only the parameter records, not the mattes, unless the user adds them.

- **Matte Refine ML** (18), another day: B1 ViTMatte edge refine (trimap from the matte itself:
  core / grey band widened by N px / background; only the band re-solved, Mix + "only where
  grey"), B2 DiffMatte (MIT; same, slower: Process + cache), B3 segmentation-guided Fill Holes
  (BiRefNet, MIT: holes inside the subject are defects, outside are real gaps), B4 Temporal
  Stabilise (neighbour frames, motion-compensated, edge band and Fill Holes decisions; was A10).
  Watch: MatAnyone 2's matte quality evaluator (locates defects; checkpoint unreleased,
  non-commercial). Decision 12 becomes B1/B2.
- macOS: a Flame test and `package_release_mac.sh` (13).
- Not planned (14): a matte cache, a Process button.
- Later, if needed: user-settable op order; a temporal option for Fill Holes (a hole crossing
  Max Hole % between frames can flicker).

## Dead ends

- **Greyscale hole filling** (morphological reconstruction by erosion; each pixel raised to its
  lowest wall to the frame edge), tried 2026-10-21 on the test dump for the grey side face: no
  gain there (pixels below 0.9 inside: 3521 vs 3522; the face is joined to the edge, not
  walled), and it raised 340k background pixels (noise lifted to its neighbours). Solid Core
  (15) solved it.
- (LTX's Flame dead ends apply: `setPageParamOrder`, Alpha-only output, reacting to
  plugin-edit `changedParam`, background-thread param sets.)

## Conventions

- Verify every API, flag or config key against installed source or docs before using it, and
  cite where (file and line, or the command). Mark anything unconfirmed UNVERIFIED.
- Ask the maintainer rather than assume when a request has two readings.
- Testing has two layers: the unit tests, and a demonstration of each requirement (in Flame for
  anything host-related). Passing unit tests alone is not evidence.
- Keep this file current: record decisions with numbers and reasons, move items between
  PROPOSED/OPEN/DECIDED, add dead ends. Keep it under ~250 lines; split per module like LTX when
  it grows. Nothing machine- or person-specific: that goes in `CLAUDE.local.md`.
- Commit or push only when the maintainer asks.
