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
| 2 | **Edge Refine**: refine the edges of an existing matte with a trimap-based model | ML, needs research | roadmap (decision 12) |

## Architecture

```
Flame ──OFX──> MatteRefine.ofx (one effect "Matte Refine", grouping "Matte Refine")
   render(): fetch source → matte from Alpha/Luma/Red (4) → op 1 → op 2 … (enabled ones, fixed
             order, 3) → output RGBA: RGB = source (or the Show Filled view, 11), A = result
   ofx/src/ops/:   pure C++ on float mattes, no OFX types; unit-tested without a host
   ofx/src/plugin.cpp: params, layout, render, Collect Logs, version label
```

Source files: `ofx/src/plugin.cpp` (the effect), `ops/fill_holes.*` (op 1), `log.*`,
`user_config.*` (`~/.config/matte-refine/config.conf`, `log_level`), `env_info.*`,
`process.*` (tar for Collect Logs) and `platform.h` + `platform_linux.cpp` (OS, CPU, RAM for the
snapshot), the last five copied from LTX and trimmed. Tests: `ofx/tests/`.

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

### Matte in, matte out

- **4. DECIDED (user, 2026-10-21): Matte From** choice (Alpha / Luminance / Red), default Alpha.
  Output alpha = the refined matte; output RGB = the source RGB unchanged (pass-through), so the
  node drops into a front + matte chain. Flame hands 32-bit float RGBA and forces RGBA output;
  alpha is the node's matte output (LTX decision 48). Luminance = Rec.709 weights.
  UNVERIFIED (from LTX): that a matte on the node's matte socket arrives in the source alpha.
- **5. Render is synchronous and per frame** (14). Ops run in `render()` on Flame's main thread
  (all OFX actions do), so the algorithmic chain must stay well under ~100 ms at UHD. No
  temporal processing. Input outside 0–1 (float) is clamped to 0–1 before the ops.
- **6. Pixel-sized params scale with the OFX render scale** (proxies match); percentages don't.

### Operation 1: Fill Holes

- **7. Principle:** never lower a pixel, and never touch the background connected to the outside
  of the frame. That is why it doesn't damage the outer edge the way dilate/close does.
- **8. Params and defaults (DECIDED, user 2026-10-21).**
  Group "Fill Holes": **Enable** (on), **Threshold** (labelled "Solid Above", since Flame may
  not show hints; 0.5: matte ≥ it is solid), **Max Hole %**
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
- **11. DECIDED (user, 2026-10-21): View** choice: **Result** (RGB = source) / **Show Filled**
  (RGB = the input matte in grey, added fill (out − in) tinted magenta (1, 0, 1), alpha = result).
  Render-only: no effect on the alpha.

### Operation 2: Edge Refine (roadmap)

- **12. Roadmap, not designed** (user, 2026-10-21: needs its own conversation and research).
  Refine an existing matte's edges with a trimap-based matting model: trimap from the input
  matte (erode → solid, dilate → unknown band), the model solves the band. To research: models
  and their licences (ViTMatte, MatAnyone, Robust Video Matting, others), temporal stability,
  GPU need on Rocky and Apple Silicon, per-frame time and memory at HD/UHD (it must render inline,
  14), and how it runs from `render()` (in-process vs a resident helper). Order in the chain
  relative to Fill Holes: OPEN.

### Platforms

- **13. DECIDED (user, 2026-10-21): Rocky 9 today, macOS next.** The ops stay portable (no
  Linux-only calls); the Mac bundle, its build script and its test are a follow-up.

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

**Planned layout (tab 1):** Matte (Matte From 2, View 2, Version label 1) · Fill Holes (Enable,
Threshold, Max Hole %, Bridge Gaps, Rim) · Fill Holes Edges (4 booleans) · Support (Collect
Logs). Four columns; two left for future ops before tab 2.

## Logging and support (LTX decision 39, scaled down)

- One log `~/.cache/matte-refine/logs/matte-refine.log` (rotated at plugin load, 10 MB × 5),
  LTX's line format (`YYYY-MM-DD HH:MM:SS.mmm LEVEL comp[pid/tid] node=<id> | msg`); environment
  snapshot at load (version + git, host, OS, glibc, CPU); INFO for user actions and param edits;
  DEBUG for renders with per-op timings. Never logged: image data, clip or project names.
- **Collect Logs** button → `~/matte-refine-support-<ts>.tar.gz`.
- Version label on the node ("0.1.0 (git hash)") so a Flame test shows which build is loaded.

## Building, testing, releasing

- Linux release builds on the Rocky 9 build host (`HOST=<host> scripts/build_remote.sh`); local
  test builds on any Linux with `cmake -S . -B <build>` (not loadable in Flame: newer glibc).
- **Unit tests (ctest, 2):** `fill_holes` on synthetic mattes: simple hole, outer soft edge
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

- Operation 2, Edge Refine (decision 12). macOS bundle (13).
- Not planned (14): a matte cache, a Process button.
- Later, if needed: user-settable op order; a temporal option for Fill Holes (a hole crossing
  Max Hole % between frames can flicker).

## Dead ends

- (none yet; LTX's Flame dead ends apply: `setPageParamOrder`, Alpha-only output, reacting to
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
