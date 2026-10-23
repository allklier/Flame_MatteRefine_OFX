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

## Where things live (split 2026-10-23; each file stays under ~250 lines)

| File | Covers |
|---|---|
| `CLAUDE.md` (this) | goal, architecture, cross-cutting decisions, decision index, building / testing / releasing, roadmap, dead ends, conventions |
| `ofx/CLAUDE.md` | the plugin layer: inputs, colour and views, render rules, platforms, Flame host behaviour, layout, logging and support |
| `ofx/src/ops/CLAUDE.md` | the operations and their shared code: algorithms, params, measurements, synthetic mattes |
| `ofx/ml/CLAUDE.md` | Matte Refine ML, the second plugin: scope, plan, runtime, licences, risks (nothing built yet) |

Claude Code loads a nested `CLAUDE.md` when you work in its folder; this root file is always read,
so keep it short and put detail where it belongs.
Code comments cite sections by title ("Plugin 1 operations", "Flame host behaviour", "Logging and
support", "Synthetic mattes"): grep the three files for the title.

**Decision index** (numbers are stable; in `ofx/CLAUDE.md`: 4, 5, 6, 11, 13, 22, 23, 24, 25; in
`ofx/src/ops/CLAUDE.md`: 7-10, 12, 15, 16, 20, 21, 26; in `ofx/ml/CLAUDE.md`: 29, 30, 31; in this file: 1, 2, 3, 14, 17, 18, 19, 27, 28):

| # | Decision | In |
|---|---|---|
| 1 | one node, operations as sections | root |
| 2 | same principles as LTX Alpha Gen | root |
| 3 | an operation = a pure function + a param group | root |
| 4 | picture on the front, matte on the Matte input / source alpha | ofx |
| 5, 6 | render is synchronous per frame; pixel params scale with render scale | ofx |
| 7-10 | Fill Holes: principle, params, algorithm, fill value | ops |
| 11 | View choice for the RGB output | ofx |
| 12 | Edge Refine (ML): roadmap | ops |
| 13 | platforms: Rocky 9, macOS | ofx |
| 14 | all matte processing inline, no cache, no Process button | root |
| 15, 16 | Solid Core and its soft edge (Feather px) | ops |
| 17 | scope = refine, not replace | root |
| 18, 27, 28 | two plugins; one repo, two bundles; ML inline first | root |
| 19 | UI = one node, two tabs, one operation per column | root |
| 20, 21 | shared edge geometry; synthetic mattes | ops |
| 22 | the matte has its own input; the picture is optional | ofx |
| 23 | output colourspace tags; View on both tabs | ofx |
| 24 | Edge Colour Unmix errors on a non-linear picture | ofx |
| 25 | "Foreground Colour" is shown as "Edge Colour Unmix" | ofx |
| 26 | Edge Colour Unmix "Ignore Outer px" (grown mattes) | ops |
| 29, 30, 31 | ML runtime (ONNX Runtime, proposed); band-only tiled processing; ViTMatte for evaluation only (licence) | ml |

## Goal

An OFX plugin for Autodesk Flame on **Rocky Linux 9** and **macOS on Apple Silicon** that holds
a growing list of **matte refinement operations** in one node: it refines an existing good
matte, it does not replace it (17). Started 2026-10-21.

| Group | Operations | Status |
|---|---|---|
| Clean-up | Fill Holes, Solid Core, Clean & Crush | built (0.1-0.2) |
| Edge shape | Contour Smooth, Shrink / Grow, Edge Width | built (0.2) |
| Picture-aware | Snap to Image, Edge Colour Unmix | built (0.2) |
| Assist | Analyze Frame (sets Clean & Crush, Fill Holes, Solid Core) | built (0.3 dev) |
| ML refinement | ViTMatte / DiffMatte edge refine, segmentation-guided fill, temporal | another plugin (18), roadmap |

## Architecture

```
Flame ──OFX──> MatteRefine.ofx (one effect "Matte Refine", grouping "Matte Refine")
   render(): matte = Matte input, else source alpha (4, 22) → clean → fill holes → solid core
             → contour smooth → shrink/grow → edge width → snap → edge colour unmix (enabled
             ones, fixed order, 3, 19) → RGBA: alpha = result, RGB per View (11, 23)
   ofx/src/ops/:   pure C++ on float mattes, no OFX types; unit-tested without a host
   ofx/src/plugin.cpp: params, layout, render, colour tags, Collect Logs, version label
```

Source files: `ofx/src/plugin.cpp` (the effect), `ofx/src/ops/*` (one file pair per operation,
plus `morphology` and `edge_field`, shared), `ocio_encoding.*` (colourspace lookup in an OCIO
config, 24), `log.*`, `user_config.*` (`~/.config/matte-refine/config.conf`, `log_level`),
`env_info.*`, `process.*` (tar for Collect Logs), `platform.h` + `platform_linux.cpp` /
`platform_macos.cpp` (OS, CPU, RAM for the snapshot; the last five copied from LTX and trimmed).
Tests: `ofx/tests/`. Synthetic mattes: `ofx/tools/` (21).

Plugin id `com.janklier.matterefine`, version from CMake `MR_VERSION` (OFX version = its
major.minor), `third_party/openfx` = OpenFX 1.4 (`include/` + `Support/`, BSD, copied from LTX's
vendored copy) plus the two 1.5.1 colour headers (23, `VENDORED.txt`).

## Cross-cutting decisions

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
  and inline (14). A later **Matte Refine ML** holds the slow and ML options (B1-B4), in this
  repo (27), inline first, LTX-style worker + Process + cache only if the measurements need it
  (28; whether Flame caches OFX results: UNVERIFIED). Decision 14 holds for this plugin. Plan:
  `ofx/ml/CLAUDE.md`.
- **27. DECIDED (user, 2026-10-23): one repo and CMake project, two bundles** (option A):
  `MatteRefine.ofx.bundle` and `MatteRefineML.ofx.bundle`, sharing the static libraries;
  `ofx/ml/` for the second; CMake option `MR_BUILD_ML` (default OFF). Why not one bundle: a
  heavy runtime in the same binary could stop the light plugin from loading, and the ML one is
  an optional large install. Detail in `ofx/ml/CLAUDE.md`.
- **28. DECIDED (user, 2026-10-23): ML models inline in `render()` first**, an offline worker
  (as LTX Alpha Gen) only if Flame locks up too much or it is too slow; decided on measurements
  (phase 0 benchmark, phase 1 Flame spike), see `ofx/ml/CLAUDE.md`.
- **19. DECIDED (user, 2026-10-22): UI = one node, two tabs, one operation per column.** An op =
  Enable + at most 4 controls, off by default; paired ops share a column. Tab 1 "Fill & Core":
  Matte (View, Version) · Clean & Crush · Fill Holes · Fill Holes Edges · Solid Core · Support
  (+ Analyze later). Tab 2 "Edge": Matte (View, synced; 23) · Contour Smooth · Shrink / Grow ·
  Edge Width · Snap to Image · Edge Colour Unmix (25). Definition order = chain order (clean → fill → core → smooth → shrink/grow
  → edge width → snap → foreground colour). QC is View choices, not controls. Labels say what
  they do (hints may not show). Built 2026-10-22 (0.2.0); not yet seen in Flame.

## Building, testing, releasing

- Linux release builds on the Rocky 9 build host (`HOST=<host> scripts/build_remote.sh`); local
  test builds on any Linux with `cmake -S . -B <build>` (not loadable in Flame: newer glibc).
  Mac: `HOST=<mac> REMOTE=<folder> CMAKE=<cmake> CMAKE_ARGS=... scripts/build_mac.sh` (on macOS
  26 pass `-DCMAKE_OSX_SYSROOT=<CLT>/SDKs/MacOSX.sdk` and `-stdlib++-isystem <SDK>/usr/include/c++/v1`
  in `CMAKE_CXX_FLAGS`, as LTX).
- **Unit tests (ctest, 12 suites at 0.2.1):** `fill_holes`, `solid_core`, `edge_field`,
  `clean_background`, `contour_smooth`, `shrink_grow`, `edge_width`, `snap_to_image`,
  `foreground_colour`, `ocio_encoding`, `synth`, `load_bundle` (dlopen + plugin id). Every op's
  test has known-answer cases, a brute-force or independent reference, "untouched elsewhere"
  checks, the synthetic scenes against their truth (21), a UHD timing check, and a mutation
  check made when it was written. Run `ctest` in the build folder. Render scale (6) and the
  plugin layer (params, colour tags, views) are not unit-tested: they need Flame.
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

- **Matte Refine ML** (18, 27, 28): planned in `ofx/ml/CLAUDE.md` (B1 ViTMatte edge refine, B2
  DiffMatte, B3 segmentation-guided Fill Holes, B4 temporal). Next step there: phase 0, the
  benchmark tool, then a Flame spike. Decision 12 became B1/B2.
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
  PROPOSED/OPEN/DECIDED, add dead ends. Keep each CLAUDE.md under ~250 lines; split per area like LTX when
  one grows (split 2026-10-23). Nothing machine- or person-specific: that goes in `CLAUDE.local.md`.
- Commit or push only when the maintainer asks.
