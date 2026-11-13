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
| `ofx/merge/CLAUDE.md` | Matte Merge, the third plugin: picture + base matte (or trimap) + detail matte in, one combined matte out; core and plugin built (0.1.0), Flame test pending |
| `ofx/ml/CLAUDE.md` | Matte Refine ML, the second plugin: scope, plan, runtime, licences, risks; benchmark data in `ofx/ml/PHASE0_RESULTS.md`, licences in `ofx/ml/LICENCES.md`, packaging in `ofx/ml/PACKAGING.md` (plugin built, 0.1.0 dev) |

Claude Code loads a nested `CLAUDE.md` when you work in its folder; this root file is always read,
so keep it short and put detail where it belongs.
Code comments cite sections by title ("Plugin 1 operations", "Flame host behaviour", "Logging and
support", "Synthetic mattes"): grep the three files for the title.

**Decision index** (numbers are stable; in `ofx/CLAUDE.md`: 4, 5, 6, 11, 13, 22, 23, 24, 25; in
`ofx/src/ops/CLAUDE.md`: 7-10, 12, 15, 16, 20, 21, 26; in `ofx/ml/CLAUDE.md`: 29-36; in this file: 1, 2, 3, 14, 17, 18, 19, 27, 28, 37, 38):

| # | Decision | In |
|---|---|---|
| 1 | one node, operations as sections | root |
| 2 | same principles as LTX Alpha Gen | root |
| 3 | an operation = a pure function + a param group | root |
| 4 | picture on the front (mandatory), matte on the source alpha (filter; all three plugins) | ofx |
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
| 22 | superseded 2026-10-24: the picture is mandatory, matte on the source alpha, no Matte clip (as plugins 2 and 3) | ofx |
| 23 | output colourspace tags; View on both tabs | ofx |
| 24 | Edge Colour Unmix errors on a non-linear picture | ofx |
| 25 | "Foreground Colour" is shown as "Edge Colour Unmix" | ofx |
| 26 | Edge Colour Unmix "Ignore Outer px" (grown mattes) | ops |
| 29, 30, 31 | ML runtime (ONNX Runtime, proposed); band-only tiled processing; ViTMatte usable behind the licence gate, not flagged non-commercial (decided) | ml |
| 38 | packaging: install.sh (root) + setup.sh (per user, licence screen, downloads, hash gate in the node); two tarballs | root |
| 37 | third plugin Matte Merge: main matte + detail matte (LTX hair) -> combined; separate plugin, classical first, same repo | root |
| 39 | Focus Region (user roto) = the only place a fusion step may change the matte; Matte Merge is the fusion node; scoped exception to 17 | root |
| 40 | Jump Repair (ML plugin): removes 1-3 frame matte jumps, local and deterministic | root |
| 42 | scope filter: build for COMMON problems where manual roto is costly; leave rare edge cases to manual roto | root |
| 41 | fourth plugin "Edge Recipes": modular functions inside one plugin, a planner orders them per recipe, shared evidence | root |
| 36 | ML feature plan: B2 as a model choice, slow behind a Quality switch, BiRefNet 24 GB rule, B3 before B4 | ml |
| 35 | Matte Refine ML is a filter (picture on the front, matte in its alpha); supersedes 22 for ML only | ml |
| 34 | ML picture colour = Auto (Flame converts to Rec.709 display) / Bypass; no colour maths in the plugin | ml |
| 33 | Matte Refine ML on macOS deferred (CoreML via ORT too slow); the Mac ships plugin 1 only | ml |
| 32 | a synchronous runner process owns ORT / CUDA / models (Flame ships its own ORT); no GPU library in the .ofx | ml |

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
   render(): matte = Source alpha, picture = Source RGB (4) → clean → fill holes → solid core
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
- **37. DECIDED (user, 2026-10-23): a third plugin, Matte Merge: a main matte plus a detail matte in, one combined
  matte out.** The main matte (roto, a guided model, a keyer) keeps the subject and body edge; the detail matte (any source with
  better edge detail, e.g. LTX Alpha Gen hair) supplies that detail in a limited zone only (auto from the main matte's edge,
  or the grey zone of a trimap), registered first; output = the combined matte only, no colour handling (user, 2026-10-24). Why a
  separate plugin: a different workflow (two mattes); plugin 1 stays one matte in / one out, plugin 2 model-based. Same
  repo and CMake project as 27 (own bundle, option default OFF), classical first (no model, no runner). Still decision 17:
  the main matte stays the source. Method, inputs risk and plan: `ofx/merge/CLAUDE.md` (PROPOSED, to be validated).
- **38. DECIDED (user, 2026-10-24): packaging = `install.sh` (root: copies the bundles) + `setup.sh` (per user:
  licence screen, downloads, config).** Why: the models, the CUDA libraries and their licences belong to the
  artist who runs Flame, not to the admin who runs sudo; install-time downloads would make the licence acceptance
  implicit, and LTX's in-node Setup tab is for a 6 GB Python install, far more than ML needs (about 2.5 GB, all
  public). `setup.sh` (bash + curl, refuses root) shows `licences.txt` (ViTMatte MIT + Deep Image Matting
  attribution, BiRefNet MIT, ONNX Runtime MIT, NVIDIA CUDA / cuDNN terms), asks to accept, asks whether to also
  install the full models (default no), downloads from Hugging Face and PyPI with sha256 checks to
  `~/.local/share/matte-refine-ml/`, and writes `ml_model_dir`, `ml_cuda_libs` and `ml_licence_accepted` (the sha256
  of the shipped `licences.txt`, as LTX's gate, decision 38 there) into `~/.config/matte-refine/config.conf`. The node
  downloads nothing: in a release bundle (`Contents/Resources/licences.txt` present) it passes the matte through with
  a footer until the licence hash matches and the files exist; dev builds are not gated. Two tarballs: Matte Refine,
  and Matte Refine ML (bundle + runner + ONNX Runtime in `Contents/Resources/lib`, no models, no CUDA libraries).
  Weights only from stable hosts (memory rule). The gate covers the whole plugin (user, 2026-10-24); a "no" to the full models is
  remembered (`ml_full_declined`), dropdowns show "(not installed)". BUILT: how it works in `ofx/ml/PACKAGING.md`.
- **39. DECIDED (user, 2026-11-12; architecture PROPOSED, details open): Focus Region.** Real case: a good, tight
  matte misses thin hair locks (3-4 px strands, a small gap between two strands, 15-20 px from the solid matte, alpha ~0
  there). Loosening the whole matte costs elsewhere (same finding as the Matte Merge bridge pixels), so the fix is local:
  a user-drawn soft roto, the **Focus Region**, is the ONLY place a fusion step may change the matte (bit-identical outside).
  Inside it, several candidates are weighed together with evidence from the picture (not only matte data): (a) a matting-model
  result (ViTMatte in Edge Refine ML, region = unknown zone of its trimap, solid core = definite foreground), (b) a classical
  thin-line tracer on the picture that stops when confidence drops and connects open pieces, never wandering onto gloves or
  glasses; agreement raises confidence; Bridge and strand mask from Matte Merge close the gap to the solid matte; Snap to
  Image runs on the new pixels only (weight map). **Matte Merge is the fusion node** (fourth optional mask socket "Focus Region"; ML
  candidate arrives through its Detail socket), so the Mac keeps the classical candidate. **Scoped exception to 17:** matte may be
  created from the picture only inside a user region and only connected to the base matte; the base is never lowered.
  Research: strand tracing done (`docs/strand_tracing.md`: classical tracer with width / turn / hair-reference rules,
  prototype in Python only; open: keep unconnected pieces, one hair reference per shape or frame, tracer as a GATE on the
  model's alpha vs a generator, flicker near the confidence threshold); ViTMatte limits and pre-processing still running.
  GROUNDWORK BUILT and merged 2026-11-12 (uncommitted; not tried in Flame; `docs/focus_region.md`, test sheet
  `ofx/merge/package/FOCUS_REGION_TEST.txt`, whose MERGE_TEST.txt still says five columns): Matte Merge fourth socket
  `FocusRegion` (mask clip) with a 5th column (Region Grow % of height, Region Feather inward, Snap to Picture, Snap Strength;
  ids `regionGrow regionFeather regionSnap regionSnapStrength`; View "Region" index 6); unconnected = bit-identical (hashes);
  fusion core `ofx/merge/src/merge_fusion.*` (Candidate alpha + confidence, swappable rule, empty slot `FocusInput::extra`
  for the tracer); `snapToImage` takes an optional weight map (nullptr = old path); `ofx/src/ops/region_trimap.*` (not wired
  into Edge Refine ML). User rulings 2026-11-12 on the tracer: keep unconnected pieces unless they touch the roto's edge;
  the tracer is a GENERATOR (creates matte from the picture), marked as such: its own column "Rebuild Detail" (off by default,
  footer says it creates matte), own QC view, user guide split into "Refines" and "Rebuilds"; temporal steadiness by deciding
  strand presence over neighbour frames with the existing flow (plan), Temporal after the rebuild as polish (test first: thin
  strands may be erased); hair reference per frame with per-shape fallback (DECIDED, user); "packages" (DECIDED in principle, user): every function
  stays usable solo, and a package combines them as collaborators; the package layer does more than suggest values, it
  ORCHESTRATES the order of operations, which differs per recipe. Requirement ruling (user, 2026-11-12): separating the tiny gap
  between two strands of a lock is a NICE-TO-HAVE, not a must (a comp matte needs the pixels anyway; a gap too small to detect
  may stay filled) (design open: where it lives; leaning to a fourth node whose
  recipes are written in code and share evidence maps; not decided). **ViTMatte research done** (synthetic only, 60 scenes, 34
  variants, small and base models; `ofx/ml/research/vitmatte_thin_hair/README.md` with harness and result CSVs): (1) the
  plugin feeds the model `(x-0.5)/0.5` (`edge_refine.cpp:94`, as the HF / ONNX preprocessor config) while the authors trained with
  ImageNet mean/std (hustvl `configs/common/model.py:40-41`): ImageNet normalisation raises strand coverage .71 -> .81 (small),
  SAD -33 %; adding an unsharp of the region's picture (sigma 1, amount 1, display-encoded) gives .88 / SAD -45 %; (2)
  upscaling before inference and flip/rotate averaging HURT thin strands, linear/log encoding worse, region radius and soft
  trimap edge no effect; (3) real limit = dark hair on dark skin (contrast ~1.5:1; IoU .03-.28 today, .52-.81 with the
  changes, a half-strength contrast stretch of the region helps there); (4) post: never lower the base, drop pieces under
  12 px, a "Strand Gain" (levels) user control, one crop centred on the region instead of frame-aligned tiles, small model is
  enough; (5) no better licence-clean alternative model. The normalisation change ALSO changes Edge Refine's existing
  output: re-check against `trimap_eval` before switching (OPEN, not done). With a region and no Detail the base passes through with a footer. Still to do: tracer as candidate
  (b) in C++, tune fusion on real pairs, wire the region trimap into Edge Refine ML, align uses the automatic zone.
- **40. DECIDED (user, 2026-11-12): Jump Repair in Matte Refine ML.** A pixel group that appears or disappears for 1-3 frames
  while the neighbouring frames agree (outside the min/max of t-2..t+2, neighbours agree within a tolerance, blob at least a
  minimum size) is replaced by the neighbours' mean, feathered; everything else bit-identical. No model. BUILT and merged
  (2026-11-12, uncommitted; not tried in Flame): core `ofx/ml/src/jump_repair.*`, doc `docs/jump_repair.md`. Chain: matte -> Jump
  Repair -> Temporal -> Smart Fill -> Edge Refine (first: Temporal's median only partly removes a one-frame blob and Edge Refine
  would give a hallucination a convincing edge). Column "Jump Repair": Enable (off), Jump Above 0.5, Min Size % 0.0125, Span
  (whole frames 1-3, default 1), Motion Guards (default on); ids `jrEnable jrThreshold jrMinSize jrSpanFrames jrGuards` (the
  first draft's choice id `jrSpan` was retired unreleased); View "Jumps" at index 5; fixed constant Agree 0.15 and zone h/540.
  Guards (switched together by Motion Guards; user asked for the switch 2026-11-12) keep a blob instead of repairing it when it moved (shifted
  in t-1/t+1), is turning (back-and-forth edge) or, for spans of 2-3 frames, sits on a moving matte; reference confirmation stays on.
  Guards off on the real matte: 14 repaired, 2323 px; on: 2 repaired, 583 px. Real matte: 14 blobs, 2
  repaired, 12 kept (9 moved, 3 turning). Limits: cannot tell a one-frame object from a hallucination; constant-speed drift is
  kept as moving. HALO FIX (Flame test 2026-11-12 showed a faint outline left where a repaired blob's soft rim was): each repaired blob now
  grows by hysteresis, at most max(2, 16 h/1080) steps, joining pixels where the references agree within 0.15 and frame t exceeds
  their range by more than 0.02 (fixed, not tied to Jump Above); kept blobs are never entered; real crop, frame 1045: residual
  545 -> 0 px, decisions unchanged (2 repaired / 12 kept), 2 of 97 frames change (1045 and 1013, whose changed pixels 119 -> 258);
  limit: faint real matte on a hallucination's rim that both neighbours lack is removed too. Real-clip run in Flame: Jump
  Repair repaired exactly one blob (1045) at the default Min Size 0.0125 % (295 px at 2048x1152 vs about 30 px on the 424x562
  crop: Min Size as a share of frame area changes meaning with frame size; 1013's 62 px is below it; a pixel-based Min Size is a
  possible later change). OPEN: bump MRML_VERSION at release; Flame test sheet `ofx/ml/package/JUMP_REPAIR_TEST.txt` partly done. Motivation: a Silhouette-pulled matte
  had one-frame hallucinated blobs (24 of 97 frames flagged).
- **41. DECIDED (user, 2026-11-12; name provisional): a fourth plugin, "Edge Recipes".** Why: the order of operations differs per
  recipe, Flame's OFX interface is more restrictive than a node graph with a data pipe (the ComfyUI planner-node pattern worked
  well there), and the functions must cooperate on shared evidence. So the modularity is in the CODE (pure cores linked into
  the bundle, called in a recipe-defined order by a planner), not in separate nodes. Plugins 1-3 stay for solo use and keep
  sharing the same cores. First recipe: hair locks (decision 39: Jump Repair -> Focus Region fusion with the model and tracer
  candidates -> Snap in the region -> Temporal). Design doc: `docs/edge_recipes.md` (in progress). User rulings 2026-11-12: Edge Recipes installs the ENTIRE
  collection as one package (all methods incl. runner, ORT and model steps), Linux only for now, while the individual plugins
  stay installable alone (both must coexist); gating per step (ML steps only); a recipe sets the individual controls itself and
  may expose only 5-8 of them in the UI (every control still reachable, e.g. an advanced tab). UI (user, 2026-11-12; supersedes the
  generic-slot idea the same day): each recipe is its OWN OFX effect inside the one bundle, listed in Flame under the category
  "Edge Recipes" (precedents: BorisFX Sapphire, iNodes), with a FIXED UI of 5-8 controls plus an advanced column; switching
  recipe = delete and re-add the node. All effects share the planner, steps and evidence code and differ only by their descriptor
  (an alias). One bundle can export several plugin descriptors (`OfxGetNumberOfPlugins`, `third_party/openfx/include/ofxCore.h:229`);
  Flame's OFX menu is two levels (group, then effect; user), so every recipe uses the flat grouping "Edge Recipes" with the recipe name as the effect name; no nested grouping. Existing bundles use
  `setPluginGrouping("Matte Refine")` (`plugin.cpp:805`, `plugin_ml.cpp:1054`, `plugin_merge.cpp:588`). DESIGN WRITTEN (`docs/edge_recipes.md`, 536 lines; a throwaway planner/step
  spike with 8 checks passed in an agent scratch copy, not merged): `EdgeRecipes.ofx.bundle`, one effect per recipe, first
  "Hair Locks"; EACH recipe needs its OWN factory class (the support library keeps the plugin id in a static member per factory
  type, `ofxsImageEffect.h:206-220`; sharing one class would run every recipe as the last); evidence cache per render only, keyed
  (kind, matte version, params), never shared between nodes (decision 14 holds); fixed UI per recipe (tab 1: Matte, two recipe
  columns with 8 exposed controls, Plan, Model & Colour, Support; tab 2 "Advanced" holds every other control once); the bundle
  carries its own runner, ORT and the same `licences.txt` (one acceptance, one model folder); classical steps run without a
  model, only model steps are gated; packaging fixes needed: `install.sh --uninstall` removes all bundles whatever the package
  (`install.sh:6,17,53-56`), install must stop Edge Recipes' runner first, `setup.sh --remove` deletes shared models
  (`setup.sh:112`) and must warn when another ML bundle is installed; two runner processes if plugin 2 and Edge Recipes both
  render (about 1.1 GB RAM and 1.5 GB GPU each). DESIGN PROBLEM: Temporal takes neighbour values from the neighbours' RAW
  mattes (`render_chain.cpp:166-171`), so run last it would erase rebuilt strands: run Temporal outside the region only and keep
  strands steady with the presence vote. BUILD STATUS (2026-11-13, all merged into the main tree, uncommitted, not tried in Flame; docs: `docs/edge_recipes.md`,
  `docs/model_candidate.md`, `docs/strand_tracing.md`, `docs/jump_repair.md`, `docs/focus_region.md`): WP1 skeleton
  (`ofx/recipes/plugin`), WP2 core (`ofx/recipes/core`: steps, evidence, planner, hair-locks recipe), WP3 tracer (`ofx/recipes/tracer`; user
  rulings applied; the unanchored keep bar = Rebuild Min Confidence 0.45 plus an elongation rule `minElongation` 3.5: the real
  lock is 3.75, the synthetic mole 3.3, a thin margin), WP4 model candidate (`ofx/recipes/model`, synthetic strand coverage .70 ->
  .94 versus plugin 2's Edge Refine .00 at 15 px reach; ImageNet normalisation per call, plugin 2 unchanged), WP5 Temporal weight map +
  precomputed flows, WP6 strand vote (`ofx/recipes/vote`; Q3 A/B on synthetic: Temporal everywhere erases strands, coverage .667;
  Temporal outside the region + vote .930), WP7 Hair Locks wiring (candidates, fusion, Snap, weighted Temporal, views, gate;
  Plan strings; tracer decodes max(v,0)^2.4 for its internal measurement under Picture Colour Auto: an analysis-only exception to
  decision 34; with no region connected Temporal still runs on the whole matte). MERGED 2026-11-13: the vote integration (Hair Locks traces the neighbour frames with their own pictures and
  the region at t widened by the flow, shares one set of flows with Temporal; synthetic: strand flicker .0215 -> .0184, a spurious
  stroke 67 -> 0, lost locks' coverage .39-.57 -> .61-.70; UHD about 1.0-1.2 s per frame with everything on) and WP9 packaging
  (a fourth tarball `EdgeRecipes-<ER_VERSION>-linux-x86_64.tar.gz`; the Edge Recipes runner has its OWN build id "ER <ER_VERSION>
  <git>", separate from plugin 2's, so neither ever quits or replaces the other's runner; `install.sh` only touches the bundles
  shipped next to it; `setup.sh --remove` warns when another ML bundle is installed; 25 install tests; USERGUIDE section 7).
  Full suite 52/52 (ML+MERGE+RECIPES), default build 15/15. NOT done: the real Rocky build and trial install of the new package
  (hosts were off; commands in the WP9 report: rsync to a new build folder, `ORT_DIR=... scripts/package_release.sh`, trial install
  with temp dirs). WP8 end-to-end scores MERGED (`ofx/recipes/scoring`, tool `er_score`, summaries in `results/`;
  classical path only, synthetic, default setting: dark hair on skin coverage .93, on textured skin .91, dark on dark .26, LIGHT hair
  .03 (the tracer finds 0 pieces: those looks need the model, WP4 .89 on dark-on-dark); all contracts held on 3398 rows (never
  below the base, nothing outside the region, end frames normal). FINDINGS: Snap acted on the base's own edge with no candidate
  (distractor hit rate glove 1.0 versus .04 with Snap off; contradicts decision 39); the vote's flows dominate the cost (HD 465 ms
  plugin total, UHD 950 ms, vs the 170-230 ms estimate); Min Confidence has little leverage on synthetic data (needs real frames);
  Strand Gain only feeds the model. Draft Analyze rules (WP10): Temporal Amount from measured base jitter (< .31 -> 0 else 1, hit
  rate .99), Region Feather from region area (< 3.3 % -> 2 else 1); the rest needs real footage / the model. FIXES MERGED (2026-11-13): Snap now acts only on the pixels the fusion ADDED
  (growth 0 px, tuned on the distractor scenes: 3 px reaches a glove edge 3 px from a strand; glove hit rate 1.0 -> .036 = Snap
  off; Snap with no candidate is a bit-identical no-op; dark_skin SAD .131 -> .090); the vote's flows are computed on the tracer's
  crop (region box + 30 px margin, grown by the flow's support) when cheaper than whole-frame flows (cost rule on the grid area),
  so Temporal computes its own whole-frame flows again; vote 393 -> 81 ms at HD, recipe total HD 465 -> 334 ms, UHD 951 -> 777 ms
  on 2 cores (141 / 440 ms with 16 cores); Temporal's whole-frame flows are now the largest cost (188 ms HD), so the 170-230 ms
  HD estimate in docs/edge_recipes.md 5.2 holds only with more cores (option not built: the vote reusing Temporal's +-1 flows).
  Full suite 53/53. FIRST FLAME RESULTS (2026-11-13, user, Flame workstation, real clip; roto = Focus Region on the FOURTH socket; log shows nothing runs without it): t=63 good, right lock about 30 % short; t=94 grey, hard, poorly connected, hard step; t=96 worst; Snap made islands; the purple outside the region in Changes is Temporal by design (confirmed: Temporal Amount 0 removes it). Reproduced offline on 63/94/96 (tool `ofx/recipes/scoring/hl_probe.cpp`; the QC render is a 1:1 crop of the 2048x1152 frame at x=746, y=319; Region renders hold the roto in alpha). CAUSES: fusion output alpha x confidence (strand .85 -> .6); the 12 px tracer rule dropped a 10.5 px piece that bridges a lock to the matte; the matte join tried only the nearest solid pixel; Snap filter window (~8.5 px) vs 3-6 px strands pulls strands to .12-.25 and acts on every faint added pixel. MERGED (main tree, tests 23/23 targeted): `FusionParams::confFull` (Hair Locks 0.5: output alpha x min(1, conf/.5); Matte Merge 0 = unchanged); tracer `chainLength` (the length rule counts a chain, a chain attached to the matte is exempt: a CHANGE to user ruling 4, anchored strands only) and `coneJoin` (cheapest solid edge pixel within gapMax when the nearest fails); Snap options addedMin / raiseOnly / islandTouch (`dropSnapIslands`) in settings, default OFF, no UI. Real frames, lock coverage/reach (63 / 94 / 96): current .26/.46 / .21/.30 / 0 -> fixes only with Snap as now .43-.73 / .46-.62 / .56-.66 -> Snap off .96-1.0 on all. Synthetic (85 scenes): coverage .504 -> .554, SAD .512 -> .468, glove hit .024 unchanged, no contract violation. OPEN (user): Snap default (off, or limited: 0.05 min rise + raise only + drop islands: real frames best, synthetic false-in-region .012 -> .027). ROUND 2-3 (2026-11-13): hl_flicker patch MERGED (soft strand profile: hard-edge fraction .357 -> .142; neighbours traced with the region MOVED by the flow instead of widened: needs the user's OK against ruling Q7; presence blend `blendNear/blendFar`: flicker above Rebuild-off .029 -> .014 but synthetic false-in-region .027 -> .136, so a TEMPORARY switch `rbBlend` "Strand Blend" in Advanced; also a TEMPORARY `rgSnapMode` Limited/Off in Hair Locks 2). Dev bundle v4 built. USER VERDICT after the last retest: improvement but still temporal instability, halos and other issues; not usable on this production shot (the shot goes to manual roto). STOP (user, 2026-11-13): resume on a DIFFERENT, easier shot to refine and gain confidence in the technique under better conditions, then push on difficult shots. Assistant estimate on the hard shot: about 20-25 % usable within 1-2 more iterations (thin 3-6 px strands, synthetic-only tuning, new problems each round); offline harness `ofx/recipes/scoring/hl_probe.cpp` now reproduces real frames without a Flame round trip. Idea parked: an editable strand-guide output (traced centrelines to roto over). Still open: Snap default, rbBlend verdict, moved-region OK, remove the two temporary controls, WP10 Analyze. Rule 13
  leftover: `ofx/ml/PACKAGING.md` opening lines name hosts (scrub before public). Log component stays "ofx" (setLogComponent is process-wide), node tag
  `recipes:<recipe>:<nodeId>`.
  ANALYZE BUTTON (user question 2026-11-12: plugin 1's Analyze Frame was a big success, so Hair Locks should have the
  equivalent): PLANNED as WP10 after WP8, not in the design doc yet; a button that analyses the current frame (and +-2 for
  flicker) and sets the exposed controls with a popup saying what it chose and why, as plugin 1's; candidates: Region Feather
  from the region's size and softness, Rebuild Detail on/off and Min Confidence from the tracer's confidence distribution,
  Strand Gain from the region's picture contrast, Edge Model on/off from contrast and gate, Temporal Amount from measured
  flicker, Jump Repair Min Size from the jump blobs; ACCUMULATING (user, 2026-11-12): pressing Analyze on another frame does NOT start over, it refines: each press adds that frame's
  measurements to the node's analysis memory and the settings are recomputed from all analysed frames (robust statistics:
  medians / percentiles, outlier frames flagged); a **Reset Analysis** button clears the memory and is the way to start over
  (proposed: it also returns the controls Analyze sets to their defaults; to confirm); the memory must survive a save
  and reload (a small hidden string param in the Support group, which takes a layout slot: plan for it in WP7), the popup says how many
  frames are in it; heuristics calibrated on the WP8 scene scores, and the same local
  suggested-vs-settled record as plugin 1 (never uploaded). Work packages (1-5 parallel; critical path 1 -> 7 -> 8): 0a user runs plugin 2's
  neighbour-fetch test in Flame; 0b probe bundle (several effects in one binary, fourth socket, tab overflow); 1 multi-effect
  skeleton (test host must pick plugin n, `ofx_host.cpp:584-590`); 2 step interface + evidence + planner; 3 tracer C++ port; 4
  model step on one crop with ImageNet normalisation applied per call (plugin 2's Edge Refine output stays unchanged); 5 Temporal
  weight map + reuse flows; 6 presence vote; 7 Hair Locks UI; 8 end-to-end scores; 9 packaging. User rulings on the design questions (2026-11-12): the Edge Recipes package contains only its own bundle;
  two runner processes accepted for now; Temporal outside the region only plus the presence vote; no cross-render cache
  (decision 14); hidden `recipeRev` param from the start; widen the region by the flow for the vote (no neighbour-time mask
  fetch); keep `com.janklier.*` ids and the name EdgeRecipes; Strand Gain default 1.0. **FLAME PROBE RESULT (user, Flame workstation, 2026-11-12; `ofx/recipes/probe`, ids
  `com.janklier.edgerecipesprobe.alpha|beta|gamma`): VERIFIED** one binary exports several effects and Flame lists them under
  the group "Edge Recipes" (Alpha, Beta, Gamma); each node runs its own effect (own constant, tint, ids, labels in log and UI:
  the factory-per-recipe rule works); four sockets (1 red + 3 blue; outputs image + matte); Detail and Focus Region masks
  arrive Alpha-only ("1ch", full bounds); tab 1 with 6 columns plus a second tab "Advanced" with its columns in order, no extra
  tab; runtime text (a Plan string param and a checkbox label rewritten in render) updates live while scrubbing; Matte view
  correct. NOTES: Flame truncates labels at about 15 characters ("Probe Gamma Str"), so keep labels short; the node name field is
  also cut off; a footer message sent from the probe was not seen as a footer (the Plan text is); a string param shows as text with
  no label. SKIPPED by the user (Flame stores parameter values anyway): the saved Batch round trip (steps 6-7); the hidden node id surviving a reload stays UNVERIFIED, so nothing may depend on it. Hair Locks exposes: Region Feather %, Edge
  Model, Strand Gain, Rebuild Detail, Rebuild Min Confidence, Temporal Strength, Snap Strength, +1 (the first draft listed Jump
  Repair and Temporal on/off). Jump Repair is a WHOLE-matte clean-up (home: Matte Refine ML); in Hair Locks it is only an
  Advanced step, off by default; a later "Matte Cleanup" recipe (Jump Repair -> Temporal -> Smart Fill -> Edge Refine) could share
  evidence. Three temporal things: Jump Repair (single-frame jumps), Temporal (edge / hole steadiness, would erase thin strands),
  strand presence vote (hair locks only). Probe agent started for work packages 0a / 0b (Flame test sheets).
- **42. DECIDED (user, 2026-11-12): scope filter for new tools: common problems where manual roto is costly.** Every tool
  we add costs complexity, and chasing rare edge cases is a rabbit hole. Build something only when the problem recurs across
  shots AND a manual roto fix is expensive (complex shape, shifting shape through head turns with skew, needed on every frame).
  Leave to manual roto: problems that are a simple shape, hold only for a few frames, or occur rarely (example, 2026-11-12: a
  matte creeping over an ear for 3+ frames of a head turn: not built, roto an exclusion shape in Flame; no "exclude region" mode,
  no flow-anchored edge limit). Counter-example that passes the filter: thin hair locks over all frames (complex shifting
  shape, needed on every frame): decision 39. Apply this test before starting any new operation or recipe step.
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
- **System test:** a Flame-like host (`ofx/merge/systest/ofx_host.*`, copied from LTX) drives Matte Merge (`ofx/merge/systest`) and Matte Refine (`ofx/tests/refine_systest.cpp`, fast tier); plugin 2's ML tests use a scratch host. The full LTX-style one: a trimmed copy of LTX's
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
- **Backport the accumulating Analyze to Matte Refine (plugin 1)** (user, 2026-11-12), IF it works well in Edge Recipes
  (WP10, decision 41): today Analyze Frame is single-frame; the new behaviour keeps an analysis memory per node, each press on
  another frame refines instead of starting over (robust statistics over all analysed frames), a Reset Analysis button starts
  over, the memory survives save and reload. Decide after the Hair Locks Flame test; reuse the Edge Recipes code where possible.
- **LTX Alpha Gen: simplify its colour handling the same way** (user wish, 2026-10-23): Auto / Bypass through
  Flame's preferred-colourspace request instead of its Input colour dropdown (recorded in LTX's roadmap too;
  decision 34 in `ofx/ml/CLAUDE.md` has the verified mechanism).
- **Matte Merge** (37): built 0.1.0 (probe run in Flame, core, plugin, systest); next the Flame test (`ofx/merge/package/MERGE_TEST.txt`), package it with the others (a third tarball or inside one release).
- macOS (13, 33; user, 2026-10-24): Mac packages for **Matte Refine (plugin 1) and Matte Merge (plugin 3)**; Matte Refine ML stays
  Linux-only. To do: a Mac build of Matte Merge (bundle layout, lbm build, tests, a first timing), a Flame test of both on the Mac,
  `package_release_mac.sh` BUILT 2026-10-24 (`ofx/ml/PACKAGING.md`).
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
