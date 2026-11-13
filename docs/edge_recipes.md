# Edge Recipes, the fourth plugin: design and skeleton plan

Status: **PROPOSED** design (2026-11-12, decision 41: name provisional). Nothing here is built except the feasibility
spike in `ofx/recipes/spike/` (section 2.6). Tags as in the root `CLAUDE.md` (DECIDED / PROPOSED / OPEN / UNVERIFIED).
Every Flame behaviour claim cites `ofx/CLAUDE.md` "Flame host behaviour" or is marked UNVERIFIED. File:line citations
refer to the tree this was written against.

## 0. Summary

One bundle, "Edge Recipes", that exports **one OFX effect per recipe** (Flame menu: category "Edge Recipes", effect
name = the recipe). Each effect runs several existing pure cores in its recipe's order inside one `render()`, sharing
per-frame evidence (region weight, signed distances, ridge maps, flow) instead of recomputing it per step. All effects
share the same code (planner, steps, evidence cache) and differ only by their descriptor: plugin id suffix, params,
layout, defaults. A planner (deterministic C++, no LLM) turns the recipe, the node's controls and simple frame
statistics into a plan, runs it, and reports it (footer, log, Collect Logs). Plugins 1-3 keep their own nodes and keep linking the same cores; every
core stays callable solo (decision 3). First recipe, **Hair Locks**: Jump Repair -> Focus Region fusion (model candidate
+ tracer candidate + optional Detail) -> Snap to Image in the region -> Temporal (outside the strands) + strand presence
vote. The tracer is a **generator** ("Rebuild Detail", off by default, marked as such everywhere).

User rulings folded in (2026-11-12): (1) **packaging**: Edge Recipes is the whole collection in one package (every
method, the ML runner, ONNX Runtime and the model steps), **Linux only for now**; the solo plugins stay installable
instead of or next to it, and both must coexist (section 1.3); (2) **gating per step** (ML steps gated on licences and
models, classical steps always work); (3) **UI = one effect per recipe** (superseding an interim "generic slots"
design): recipes are build-time and rare, so each is its own effect with a FIXED UI of 5-8 exposed controls plus an
Advanced tab that reaches every control; switching recipe = delete and re-add the node (section 4); (4) separating the 1-2 px gap between the two strands of a lock is a **nice-to-have**, not a
requirement (the tracer follows a lock as one piece when the picture does not resolve the gap, `docs/strand_tracing.md`
section 6, and that is accepted).

## 1. Architecture

### 1.1 Bundle layout (Linux)

```
EdgeRecipes.ofx.bundle/Contents/
  Linux-x86-64/EdgeRecipes.ofx        C++14, libstdc++ only, no ORT / CUDA (decision 32), exports the 2 OFX entry points;
                                      OfxGetNumberOfPlugins() = number of recipes (section 4.1)
  Resources/mr_ml_runner              its own copy (C++17, owns ORT), built from the same source as plugin 2's
  Resources/lib/                      libonnxruntime + CUDA provider, as plugin 2 (PACKAGING.md:20-21, RUNPATH $ORIGIN/lib)
  Resources/licences.txt              makes it a "release bundle" for the gate (PACKAGING.md:35); the SAME text as plugin 2's
```

CMake: `option(MR_BUILD_RECIPES ... OFF)` and `add_subdirectory(ofx/recipes)` next to `ofx/ml` and `ofx/merge`
(decision 27's pattern, `CMakeLists.txt:31-35,195-199`); it requires `MR_BUILD_ML` (ORT_DIR) because the package always
carries the model steps. Source tree: `ofx/recipes/src/` (`plugin_recipes.cpp`, `recipe_ui.*` without OFX types like
`merge_ui.*`, `recipes/<recipe>.cpp` (one per recipe: steps, order, conditions, exposed controls, defaults), `planner.*`, `steps/*.cpp` adapters, `evidence.*`), `ofx/recipes/tests/`,
`ofx/recipes/package/`. No macOS flavour now; the code stays portable (decision 13) so a classical Mac build stays
possible later (it would first need the CMake move in WP1-later, see 1.2).

### 1.2 Cores linked, per step of Hair Locks

| Step | Cores (existing files) | Today's library | Note |
|---|---|---|---|
| Jump Repair | `ofx/ml/src/jump_repair.{h,cpp}`; neighbour plumbing `render_chain.{h,cpp}` (`collectNeighbours`, `neighbourOffsets`, `framesNeededRange`) | `mr_ml_jump`, `mr_ml_chain` | `mr_ml_chain` also links Smart Fill and the client; fine on Linux |
| Region weight | `ofx/merge/src/merge_fusion.{h,cpp}` `prepareRegion` | `mr_merge_core` | |
| Model candidate (ViTMatte in the region) | `ofx/ml/src/edge_refine.{h,cpp}` (`prepareTile`, `clampPicture`), `ofx/src/ops/region_trimap.{h,cpp}`, `runner_client.{h,cpp}` (`run(modelId, in, cIn, h, w, out, cOut, oh, ow, ...)`, `runner_client.h:83`), `model_table.*`, `gate.*` | `mr_ml_core`, `mr_ml_client`, `mr_ops` | needs a new region-crop prepare (one crop centred on the region, ImageNet normalisation, unsharp), WP4 |
| Tracer candidate (Rebuild Detail) | **new**: C++ port of the Python prototype (`docs/strand_tracing.md` section 3) | new `mr_tracer` | WP3 |
| Fusion | `merge_fusion.{h,cpp}` `fuseCandidates`, `defaultFusionRule`; `matte_merge.{h,cpp}` `merge(..., FocusInput*)` when a Detail matte is connected | `mr_merge_core` | without Detail: fuse + never-lower blend directly (both internal candidates are in the picture's grid, no registration) |
| Snap in the region | `ofx/src/ops/snap_to_image.{h,cpp}` (weight, `snap_to_image.h:56-62`), `merge_fusion` `snapInRegion`, `snapParamsForHeight` | `mr_ops`, `mr_merge_core` | |
| Temporal | `ofx/ml/src/temporal.{h,cpp}`, `flow.{h,cpp}` | `mr_ml_temporal` | needs an optional weight map (as Snap got) and an entry taking precomputed flows, WP5 |
| Strand presence vote | **new** (uses `flow.h` `denseFlow`, `temporal.h` `flowToNeighbour` / `flowAt`, the tracer) | new | WP6 |
| Other methods of the collection (Smart Fill, Edge Refine, plugin 1's ops) | `smart_fill.*`, `edge_refine.*`, `ofx/src/ops/*` | linked, available to later recipes | "all methods" (ruling 1): linked now, exposed by recipes as they are written |
| Shared | `ofx/src/ops/edge_field.*`, `morphology.*` (`Workers`), `ofx/src/log.*`, `user_config.*`, `env_info.*`, `process.*`, `platform_*` | `mr_ops`, support | |

Found while checking: every `ofx/ml` target sits behind `MR_BUILD_ML`, and `ofx/ml/CMakeLists.txt:4-9` stops with a
FATAL_ERROR without `ORT_DIR`, although `mr_ml_jump`, `mr_ml_temporal` and `mr_ml_core` are plain C++14
(`ofx/ml/CMakeLists.txt:13,175,187`). Not a problem for the Linux-only package; a future classical Mac build would first
move them into a CMake file without ORT (names kept, plugin 2 unchanged).

### 1.3 Packaging and coexistence (DECIDED by the user: the collection, Linux only)

Reading of the ruling used here: the **Edge Recipes bundle itself contains every method** (all cores linked, its own
runner and ORT), so installing Edge Recipes alone gives everything; the three solo bundles remain separate packages that
can be installed instead or as well. (The other reading, "the Edge Recipes tarball also installs the three solo
bundles", is Q1 below; the design works for both.)

**ML through its own runner (decision 32).** The `.ofx` never loads ORT or CUDA; the model step talks to its runner
through `RunnerClient` (C++14, `runner_client.h:1-3`). Edge Recipes carries **its own** `mr_ml_runner` + ORT in its
bundle instead of using plugin 2's, because a solo plugin 2 installed at another version would otherwise change the
runner under Edge Recipes: the handshake compares protocol and build id (`runner_client.cpp:116`) and a mismatching
runner is told to quit and replaced, so a borrowed runner would break on every independent update.

**Coexistence checklist (both installed on one machine):**

| Item | Today | Needed | Verdict |
|---|---|---|---|
| Plugin id / bundle name | `com.<owner>.matterefine`, `...matterefineml`, `...mattemerge` (`plugin.cpp:60`, `plugin_ml.cpp:60`, `plugin_merge.cpp:59`) | Edge Recipes: one id per recipe `com.<owner>.edgerecipes.<recipe>`, bundle `EdgeRecipes.ofx.bundle` | no collision |
| Two ORT copies | plugin 2's ORT in its bundle's `Resources/lib`, loaded only by its runner (RUNPATH `$ORIGIN/lib` only, `PACKAGING.md:10`) | Edge Recipes the same in its own bundle | no conflict: ORT is never loaded into Flame, each runner process loads its own copy (decision 32) |
| Runner processes | one per build id: socket `runner-<hash of build id>.sock` (`runner_ipc.cpp:114-119`) | Edge Recipes' build id differs (own version) | two runners may run at once when both plugins render in one session: no frame is computed twice, but each holds its sessions (about 1.1 GB RAM, +1.5 GB GPU per model, `ofx/ml/CLAUDE.md` phase 1); idle exit 5 min. Acceptable (Flame peaks ~45 GB VRAM on the test card with headroom, `ofx/ml/CLAUDE.md:205-206`); later improvement: one runner shared by build-independent protocol N |
| Model folder, CUDA libs | `setup.sh` writes `~/.local/share/matte-refine-ml/` and keys `ml_model_dir`, `ml_cuda_libs` (`PACKAGING.md:25-31`) | Edge Recipes reads the same keys | shared, nothing downloaded twice |
| Licence acceptance | key `ml_licence_accepted` = sha256 of the shipped `licences.txt` (`PACKAGING.md:29-30`) | ship the **identical** `licences.txt` (no new third-party component: the tracer is ours) | one acceptance covers both; if the texts ever differ, the later one re-asks (by design) |
| `setup.sh` | one script per ML package | ship the same script; it writes the same keys | either package's copy works |
| `setup.sh --remove` | deletes models, CUDA libs and downloads (`setup.sh:112`) and the `ml_*` keys | **change**: warn and ask when another ML-capable bundle is installed in the plugin folder | needed (else removing one breaks the other) |
| `install.sh --uninstall` | removes **all three** bundles regardless of which package it came from (`install.sh:6,17,53-56`) | **change**: remove only the bundles shipped next to the script (or `--uninstall <bundle>`); add `EdgeRecipes.ofx.bundle` to `BUNDLES` | needed |
| Runner stop before replace | stops plugin 2's runner only (`install.sh:44-49,55,73`) | also stop Edge Recipes' runner when replacing its bundle | needed |
| Config keys of its own | none | prefix `er_` if any are added (e.g. `er_runner`); shares `log_level` and the log file | no clash |
| Log, Collect Logs | one log `~/.cache/matte-refine/logs/matte-refine.log` (`ofx/CLAUDE.md:114`) | component "recipes" | shared, as plugin 2 shares it |

**Gating (DECIDED by the user: per step).** Only steps whose `needs()` include `kNeedModel` are gated. Release bundle
(`licences.txt` present) + licence hash mismatch or a model / CUDA library missing -> those steps do not run, the
classical steps do, footer "Model steps need setup.sh (licences and downloads); the other steps ran." No runner starts
while the gate is closed. Reuses `gate.h` unchanged (`checkGate`, `GateResult::open()`, `gate.h:42-52`); a selected
optional model that is missing turns off only its step, as plugin 2 does (`PACKAGING.md:37-39`). This differs from
plugin 2's whole-plugin gate (decision 38), which stays as it is.

## 2. The step interface

### 2.1 Inputs and outputs

```cpp
struct FrameInputs {            // read-only for one render; w*h or w*h*3, row 0 = top
  int w, h; double renderScale; double time;
  const float* rgb;             // the picture as delivered (Picture Colour Auto/Bypass, decision 34)
  const float* base;            // the matte = Source alpha (decision 4/35)
  const float* region;          // Focus Region mask or nullptr
  const float* detail;          // Detail mask or nullptr
  const Neighbours* nb;         // render_chain.h Neighbours: mattes (+ pictures) at t +- k; nullptr if none
};
struct FrameStats { long long regionPixels; double baseMean, greyFraction, regionContrast; /* ... */ };
class Step {
  const char* id() const;       // "jumpRepair", "modelCandidate", "rebuildDetail", "snapRegion", "temporal"
  StepKind kind() const;        // Refines | Rebuilds (UI marking, report)
  StepNeeds needs() const;      // flags Picture|Region|Detail|Model|Neighbours + neighbourReach; from settings only
  void run(const StepContext&, const float* in, float* out, StepReport*);
};
struct StepReport { id, ran, failed, note /* skip reason or error */, changed, ms, qc layers (optional) };
```

Outputs: the matte (w*h), a `StepReport` per step (for the footer, the DEBUG log line and Collect Logs), and optional QC
layers a step fills only when the View asks for them (as `MergeViews`, `matte_merge.h:90-95`: nullptr = not computed).
Candidate steps (model, tracer, Detail) do not change the matte themselves: they write **candidates** (alpha +
confidence) into the frame's evidence; the fusion step consumes them. That keeps "a step = matte in, matte out" for the
refining steps and lets the fusion rule see all candidates at once (`fuseCandidates`, `merge_fusion.h:55-56`).

### 2.2 Shared evidence cache (per render)

| Map | Key | Depends on | Computed by / for | Size |
|---|---|---|---|---|
| region weight | ("regionWeight", 0, grow/feather) | inputs | `prepareRegion`; all region steps | 1 plane |
| region bbox (+ margin) | ("regionBox", 0, ...) | inputs | crop for everything picture-heavy | 4 ints |
| luma / log-luma, log-chroma (crop) | ("logLuma", 0, sigma) | inputs | tracer, region contrast stat | 1-3 crop planes |
| ridge maps (crop): black top-hat, Hessian line measure + direction | ("ridge", 0, sigma) | inputs | tracer (centre); strand vote (neighbours: one per frame) | 3 crop planes per frame |
| unsharp region picture (crop) | ("unsharp", 0, sigma/amount) | inputs | model candidate | 3 crop planes |
| signed distance to 0.5 contour | ("sd0.5", matteVersion, band) | **current matte** | Snap band, Temporal band, trimap | 1 plane |
| flow centre -> neighbour k (+ FB confidence) | ("flow", 0, k, flowLines) | inputs (pictures, raw mattes) | Temporal, strand vote | 3 planes at <= 600 lines per k |
| candidates (alpha + conf) | ("cand:model" / "cand:tracer" / "cand:detail", 0) | inputs + settings | fusion | 2 crop planes each |

Rules: a key carries the matte version it was computed from; `0` = inputs only (stable for the whole render). The
planner bumps the version only when a step really changed the matte (it compares the buffers: about 37 MB per plane at
4096x2304, a memcmp of a few ms; a step's own "changed" count is not trusted), and drops matte-derived maps of older
versions. Inputs-only maps survive the whole render. Lifetime: one render of one effect instance; the storage is an
instance workspace (as `MergeWorkspace`, `matte_merge.h:132-137`) so buffers are reused, not reallocated, but **no
content survives into the next render** and nothing is shared between instances (two Hair Locks nodes in one Batch, or a
Hair Locks and a later recipe node, each hold their own; a process-wide cache would need locking across Flame's
renders and is a cache in the sense of decision 14). Reusing a neighbour's tracer result across renders would also be a
cache and needs an explicit decision (Q5).

Memory at 4096x2304 (9.44 M px; one float plane 37.7 MB): picture 113 MB + base 38 + region 38 + work planes (cur,
next, sd, out) 151 = about 340 MB before neighbours. Neighbours as plugin 2 holds them (matte + RGB, 4 planes = 151 MB
each): K = 1 -> +302 MB, K = 2 -> +604 MB; Jump Repair Span 1 needs t +- 2 (`docs/jump_repair.md:42-44`) but those
farther frames are fetched matte-only (+38 MB each). Flow at <= 600 lines (`temporal.h:148`): 2304 lines -> scale 4 ->
1024x576, 3 planes = 7 MB per neighbour. Crop-based maps (ridge, unsharp, candidates) are bounded by the region box: a
hair-locks region of a few 100k px costs a few MB. Peak at 4096x2304, K = 1, Span 1: about 0.8 GB; K = 2: about 1.1 GB.
Proxy / render scale: every plane scales with the delivered size (half-res proxy = 1/4); pixel params are scaled by the
render scale (decision 6), percentage sizes by the delivered height (Merge's rule, `matte_merge.h:78-87`).

### 2.3 Needs and frames needed

`needs()` is answered from the node's settings only (getFramesNeeded has no images). The plugin takes the union over the
**enabled** steps of the current recipe: flags OR, `neighbourReach` = max (Jump Repair: Span + 1; Temporal: K; strand
vote: its K). As plugin 2 (`plugin_ml.cpp:306-335`): all off -> nothing set (host default), else `[t - R, t + R]` clamped
to the clip range. The Focus Region clip also needs t +- K when the strand vote is on (the roto moves): set it on that
clip too (UNVERIFIED in Flame for a mask clip, Q7). Pixel-dependent conditions (an empty region, low contrast) are
decided at render and never shrink the fetch range (a wasted fetch is cheaper than a frame that cannot be judged).

### 2.4 Bit-identity rules

- A disabled step is not called; its input is passed on (no copy through the core).
- All steps off, or every step skipped: output alpha == Source alpha bit for bit (test: hash).
- Each step keeps decision 17's rule: out == in outside its zone (Jump Repair: outside repaired blobs' zones; region
  steps: where the region weight is 0; Temporal: outside its band and filled holes). The fusion's final blend is
  `base + w (max(base, fused) - base)` (`docs/focus_region.md` step 4): never below the base, exact at w = 0 and 1.
- A step that throws passes its input through (no partial write) and is reported failed; the chain goes on (as
  `render_chain.h:6-7`).

### 2.5 What a step adapter may not do

No OFX types, no param reads (the plugin turns controls into the core's params in `recipe_ui.*`, as `merge_ui.*`), no
threads of its own beyond the core's `Workers`, no static state, no I/O. A step that needs the picture in a specific
encoding gets it through the plugin's colour choice, never by converting.

### 2.6 Feasibility spike (done, scratch only)

`ofx/recipes/spike/recipe_core.{h,cpp}` + `test_recipe_core.cpp` (C++14, links only `mr_ops` sources): the interface
above with two real adapters (Solid Core, Snap to Image with the region weight), a probe step, a failing step, and the
planner. Built with `g++ -std=c++14 -O2` on the dev machine; **8 checks pass, 3 of 3 mutations caught** (always bump the
matte version; no pass-through on a throw; disabled steps counted in the needs union). Settled:
1. Existing cores fit behind the interface without changes (Snap's optional weight is exactly the region contract).
2. Recipe output == the solo cores chained by hand, bit for bit; all steps off == input bit for bit; Snap in a region
   leaves pixels with weight 0 bit-identical.
3. Evidence keyed by (kind, matte version, params): a map is computed once and shared while the matte is unchanged,
   recomputed after a real change, inputs-only maps (region weight) survive (computed 3, hit 1 in the probe recipe).
4. Skips with reasons ("no Focus Region", a frame-statistics condition) and a one-line plan summary
   (`hair locks v1: model(failed: runner not available) solidCore`) come out of the same loop.
Not settled by the spike: candidates as evidence (designed above, not coded), neighbour handling, cost.

## 3. The planner

**Recipe = an ordered list of (step, enabled-from-controls, condition).** Recipes live in code
(`ofx/recipes/src/recipes/<recipe>.cpp`), one definition per recipe; no user-editable order (decision 19's spirit:
order fixed by the chain).

Hair Locks v1 (order and conditions):

| # | Step | Kind | Runs when | Skips (reason in the report) |
|---|---|---|---|---|
| 1 | Jump Repair | Refines | enabled, t - 1 and t + 1 present | "first/last frame" (core passes through, `jump_repair.h:44`) |
| 2 | Region weight (evidence) | - | Focus Region connected and not empty | "no Focus Region" / "Focus Region empty": steps 3-6 skip, base passes through those steps |
| 3a | Model candidate | Refines | enabled, gate open, region non-empty | "needs setup.sh" / runner error (footer) |
| 3b | Tracer candidate | **Rebuilds** | Rebuild Detail enabled | - |
| 3c | Detail candidate | Refines | Detail connected | - |
| 4 | Strand presence vote | Rebuilds | 3b ran and Temporal Vote on and neighbours present | "no neighbours": tracer pieces kept by their own confidence |
| 5 | Fusion in the region | - | at least one candidate | "no candidate": base passes through (Merge's footer text, `focus_region.md` behaviour table) |
| 6 | Snap in the region | Refines | enabled | - |
| 7 | Temporal | Refines | enabled, neighbours present | weight = 1 - region (see below) |

Deterministic rules, examples (each is a pure function of controls + `FrameStats`, logged when it fires):
- **Low-contrast switch** (research rec. 4): region luminance range (1st-99th percentile) below a fixed threshold ->
  the model candidate gets the half-strength contrast stretch; else not. Exposed as Low Contrast Auto / Off.
- **Empty region** -> skip everything region-bound (no runner start, no tracer).
- **Region larger than X % of the frame** -> warn in the footer (the region is meant to be local; the model crop grows).
- **Jump Repair before anything temporal** (fixed by the recipe: `docs/jump_repair.md:34-40`).
- **Temporal vs. rebuilt strands** (found while designing, see 5.3): Temporal's neighbour values are the neighbours' raw
  mattes (`render_chain.cpp:166-171`), which lack the rebuilt strands, so a median would erase them. Hair Locks
  therefore runs Temporal with weight (1 - region) and handles strand steadiness by the vote (step 4). Reordering
  Temporal before the fusion is the alternative (then it smooths only the base); to be decided by test (Q3).

Reporting: (1) footer, one line: `Hair Locks v1: Jump Repair, Model, Rebuild (3 pieces, 1 dropped), Snap, Temporal`
plus warnings; a Rebuild step that ran adds "Rebuild Detail created matte from the picture." (2) log: one INFO line per
plan change, a DEBUG line per render with per-step ms, changed px and notes (LTX format, `ofx/CLAUDE.md` Logging); (3)
Collect Logs gets the plan lines (numbers only, never pixels); (4) View "Plan" is not proposed: the per-step views
(section 4.3) show what each did.

Each recipe is its own OFX effect (section 4), so the planner is chosen by the effect's descriptor, never by a param:
one `Planner` per effect instance, built in `createInstance` from the recipe's static definition. Recipe versioning and
the "adding a recipe" checklist are in sections 4.2 and 4.7 (they are descriptor matters now).

## 4. OFX / Flame layout: one effect per recipe (DECIDED by the user, 2026-11-12)

Recipes are build-time and added rarely, so **each recipe is its own OFX effect** inside the one `EdgeRecipes.ofx`
binary. Flame's OFX menu has two levels, plugin group then effect (user, 2026-11-12), so every recipe effect uses the
**flat grouping "Edge Recipes"** and the recipe name as the effect name ("Hair Locks"). Adding the node runs that
recipe with a FIXED UI designed for it; switching recipe = delete the node and add another. Under the hood all effects
share the same code and package; they differ only by their descriptor (plugin id suffix, params, layout, defaults).

### 4.1 Several plugin descriptors in one binary

- The host asks `OfxGetNumberOfPlugins()` and `OfxGetPlugin(nth)` (`third_party/openfx/include/ofxCore.h:221,229`).
  The vendored Support library implements both from the plugin's `OFX::Plugin::getPluginIDs(PluginFactoryArray&)`
  (`Support/Library/ofxsImageEffect.cpp:2980-3026`: `init()` loops over every factory, `OfxGetNumberOfPlugins` returns
  the array's size). The three bundles today push one static factory each (`ofx/src/plugin.cpp:1075-1078`,
  `plugin_merge.cpp:786`, `plugin_ml.cpp:1269`). N factories = N effects: the mechanism scales.
- **Catch, found in the Support lib:** `FactoryMainEntryHelper<FACTORY>` keeps the plugin's uid in a **static member
  per template type** and its static `mainEntry` dispatches by that uid (`Support/include/ofxsImageEffect.h:206,210-215,
  220`; the constructor asserts it runs once). N instances of ONE factory class would share one uid (the assert fires in
  a debug build; in a release build every effect dispatches to the last one). So each recipe needs its own factory TYPE:
  `template <class Recipe> class RecipeFactory : public OFX::PluginFactoryHelper<RecipeFactory<Recipe>>`, with one tag
  type per recipe (`struct HairLocks { static const RecipeDef& def(); };`) and `getPluginIDs` pushing one static
  `RecipeFactory<R>` per recipe. Everything else in the Support lib is keyed by the uid already (descriptor map
  `gEffectDescriptors[plugname]`, `ofxsImageEffect.cpp:165,2315,2393`; info map `plugInfoMap[newID]`, :2994).
- The load action is counted across the binary's plugins (`gLoadCount`, `ofxsImageEffect.cpp:1912-1972`): process-wide
  set-up (log rotation, environment snapshot) runs once per binary, not per recipe.
- `describe` / `describeInContext` / `createInstance` are per descriptor: one shared `RecipeEffect` class (an
  `OFX::ImageEffect`) takes the recipe's `RecipeDef` (steps, order, conditions, exposed controls, defaults, views,
  version) from its factory. Whether Flame lists every plugin of a multi-plugin binary: UNVERIFIED for our build (the
  user cites commercial precedents, Sapphire and iNodes); covered by the WP0 probe (4.7).

### 4.2 Plugin ids, versions, param ids, saved projects

- **Plugin id per recipe:** `com.<owner>.edgerecipes.<recipe>` (pattern of the existing ids, `plugin.cpp:60`,
  `plugin_ml.cpp:60`, `plugin_merge.cpp:59`), e.g. `...edgerecipes.hairlocks`. Never changed or reused after a release:
  a saved Batch binds a node to it. A retired recipe keeps its descriptor (marked "(old)" in its label) so old setups
  still load.
- **Version per recipe:** each descriptor carries its own OFX major.minor (`PluginFactoryHelper(id, major, minor)`,
  `plugin.cpp:797`): minor for compatible changes (params appended, behaviour tuned), major only for a deliberate break
  (a host treats a new major as another plugin: OFX convention; Flame's handling of two majors UNVERIFIED, avoid it).
  The Version label shows the bundle version and the recipe version ("ER 0.1.0, Hair Locks 1.0, <git>").
- **Param ids per recipe, append-only**, never renamed, choice indexes append-only (the rule all three plugins follow).
  Ids are per descriptor, so recipes may reuse names (`jrThreshold`, `tmStrength`, as plugin 2's) without clashing.
- **Behaviour pinning:** a hidden `recipeRev` param set in `createInstance` for new nodes (hidden params set there are
  saved with the setup, LTX `ofx/CLAUDE.md:233-234`; it sits inside the Support group, since a secret param takes a
  slot, `ofx/CLAUDE.md:97-99`) lets a recipe keep an old code path for old nodes when an output-changing tweak lands.
  Default: old nodes get the new behaviour (as plugin updates do today); pinning only on purpose (Q6).
- Defaults: a changed default affects new nodes; whether Flame stores values equal to the default in a setup is
  UNVERIFIED, so a default change is treated as a behaviour change (minor bump, release note).

### 4.3 The fixed UI of one recipe (Hair Locks v1)

Every control exists **exactly once**: the 5-8 exposed controls on tab 1, every other control on the Advanced tab. No
mirroring, no `setValue` cascades, so the changedParam recursion trap (`ofx/CLAUDE.md:90-92`) does not arise; nothing is
locked inside the recipe. Flame rules used (verified, `ofx/CLAUDE.md:93-99`, LTX `ofx/CLAUDE.md:222-227`): columns of 5
rows, 6 per tab, each group starts a column, pages ignored, a new tab only on overflow and named after the next OFX page;
choice 2 rows, bool / double / button / label string 1; a secret param still takes a slot. Tab 1 is therefore filled to 6
columns so the Advanced columns start on tab 2 named "Advanced" (the page defined before them).

Tab 1 "Hair Locks" (6 columns):

| Col | Group | Rows |
|---|---|---|
| 1 | **Matte** | View (choice) 2 · Version label 1 · (2 free) |
| 2 | **Hair Locks** | Jump Repair (on) · Region Feather % (0.25) · Edge Model (on) · Strand Gain (mild) · **Rebuild Detail (creates matte)** (off) |
| 3 | **Hair Locks** (cont.) | Rebuild Min Confidence (0.45) · Temporal (on) · Temporal Strength (1) · (2 free) |
| 4 | **Plan** | 5 label strings, rewritten per render (label string values update at runtime: verified, LTX `plugin.cpp:888-889`): plan line, "Rebuild Detail ON: creates matte", model gate state, notes |
| 5 | **Model & Colour** | Model (Small / Base) 2 · Picture Colour (Auto / Bypass, decision 34) 2 · (1 free) |
| 6 | **Support** | Debug Logging · Collect Logs · Save Matte · hidden `nodeId` · hidden `recipeRev` |

Tab 2 "Advanced" (chain order; each column = one step's remaining controls, decision 19: at most 4 + Enable):
Jump Repair (Jump Above 0.5, Min Size % 0.0125, Span 1, Motion Guards on) · Focus Region (Region Grow % 0, Snap to Picture
on, Snap Strength 1) · Edge Model (Low Contrast Auto / Off 2) · Rebuild Detail (Keep Loose Pieces on, Temporal Vote on,
Min Piece px 12 at 1080) · Temporal (Frames +-1 / +-2 2, Band px 6). 8 exposed controls = the ruling's upper bound; the
exposed set is a design choice per recipe, revisited after the first Flame test.

Labels say what they do (hints may not show, `ofx/CLAUDE.md:101-102`); doubles get digits / increments
(`ofx/CLAUDE.md:103-105`, honoured: UNVERIFIED). Ids: tab 1 `jrEnable rgFeather emEnable emGain rbEnable rbMinConf
tmEnable tmStrength view mdModel picColour`, Advanced `jrThreshold jrMinSize jrSpanFrames jrGuards rgGrow rgSnap
rgSnapStrength emLowContrast rbKeepLoose rbVote rbMinPiece tmFrames tmBand`, support as the other plugins.

Views (QC, decision 19; index append-only): 0 Result Matte · 1 Picture · 2 Changes (plugin 1's colours,
`render_chain.h:194-204`) · 3 Region · 4 Jumps (plugin 2's) · 5 Candidates (model red, tracer green, Detail blue, base
grey) · 6 **Rebuilt** (only what Rebuild Detail added; dropped pieces dim) · 7 Trimap (the model's). Output alpha is
always the result; colour tags as decision 23. The generator is marked in its label, the Plan line, the footer ("Rebuild
Detail created matte from the picture.") and View Rebuilt; the user guide splits "Refines" / "Rebuilds".

### 4.4 Sockets

Filter context (decisions 4, 35): Source = picture on the front (mandatory) + base matte on the source alpha; **Detail**
(optional mask, as Matte Merge, `plugin_merge.cpp:624-631`): an external detail matte (LTX hair, a keyer) as a third
candidate; **FocusRegion** (optional mask, `plugin_merge.cpp:634-638`). Clip names are stored in setups: never rename.
Sockets are per descriptor, so a later recipe declares only what it needs. Four = what Matte Merge declares; Flame showed
the first three as one red + two blue (`ofx/merge/CLAUDE.md:74`), the fourth's position and colour are UNVERIFIED
(`docs/focus_region.md:99-100`). Connecting any mask clip makes Flame deliver the picture converted and tag the Source
"Matte" (verified for Merge and ML, `ofx/merge/CLAUDE.md:65-73`); the model step keeps Picture Colour Auto / Bypass, to
be re-checked with masks connected (UNVERIFIED).

### 4.5 Shared code: cost and lifetime

- **Static initialisation:** one binary, one set of statics for all recipe effects: the factories, the log (rotated
  once at load), the user config. Steps and the planner hold no static mutable state (2.5), so effects cannot disturb
  each other.
- **Per instance:** one `Planner` (built in `createInstance` from the recipe's `RecipeDef`), one evidence workspace, one
  `RunnerClient` (plugin 2's pattern: one connection per node, `ofx/ml/CLAUDE.md:66-67`). The evidence lives for one
  render and is never shared across instances (2.2). Memory: a workspace keeps its frame-sized buffers between renders
  (page-fault saving, as `MergeWorkspace`); neighbour frames are freed after each render, so an idle node keeps about
  4-6 planes (150-230 MB at 4096x2304, 35-50 MB at HD). Many Hair Locks nodes in one Batch multiply that: acceptable
  at HD, worth a "trim after render" switch if UHD Batches show pressure.
- **Runner:** one per user, machine and build, shared by every node and every Flame process (`ofx/ml/CLAUDE.md:65-67`);
  model calls FIFO, one at a time; measured two clients 206-215 ms per HD frame each (`ofx/ml/CLAUDE.md:160-162`). All
  Edge Recipes effects share Edge Recipes' runner (same build id); plugin 2's own runner is separate (1.3).
- **Threading:** every action runs on Flame's main thread (`ofx/CLAUDE.md:87-89`), so renders of several nodes in one
  Flame process are serial anyway; `eRenderInstanceSafe` as the other plugins (`plugin_ml.cpp:1068`).

### 4.6 Gating per effect and per step (DECIDED: per step)

Each `RecipeDef` lists its steps with `needs()`; steps with `kNeedModel` are gated (1.3). A recipe also states whether
it **requires** a model: Hair Locks does not (the tracer and Detail candidates work without one), so with the gate
closed it runs its classical steps and the footer says "Edge Model needs setup.sh; the other steps ran." A future recipe
whose every candidate is a model would pass the matte through with that footer (its effect stays listed: hiding it
would break saved setups). At describe, the exposed model toggle's label can carry "(needs setup)" as plugin 2 sets
dropdown labels at describe (`PACKAGING.md:39-41`; shown in Flame: UNVERIFIED, `PACKAGING.md:48-49`).

### 4.7 Adding a recipe (checklist) and its tests

1. `ofx/recipes/src/recipes/<name>.cpp`: a tag type, its `RecipeDef` (steps + order + conditions, exposed 5-8 controls,
   Advanced columns, defaults, views, sockets, `requiresModel`, version 1.0), plugin id `...edgerecipes.<name>`.
2. One line in `getPluginIDs` (its `RecipeFactory<Tag>`).
3. Any new step: an adapter over a pure core with its own unit test (decision 3); never a param read inside a step.
4. Tests: layout check of its `RecipeDef` (rows per column <= 5, columns per tab <= 6, tab 1 full so Advanced starts on
   tab 2, every control exactly once, ids unique and append-only against a committed id list); planner tests (all off ==
   input bit for bit, equivalence with the hand-chained cores, region bit-identity, each condition); system test through
   the host by plugin index (WP1 extends the host, which today takes plugin 0 only, `ofx_host.cpp:584-590`);
   `test_load_bundle` expects the new count and id.
5. Flame test sheet section; user guide section under "Refines" / "Rebuilds"; release note; decision recorded in
   `CLAUDE.md`.

### 4.8 Coexistence with plugins 1-3 (the user may install them too)

Different bundle names and plugin ids (no collision), different menu group ("Matte Refine" for plugins 1-3,
`plugin.cpp:805`, `plugin_ml.cpp:1054`, `plugin_merge.cpp:588`; "Edge Recipes" here), separate params and setups, one
shared log (component "recipes"), shared model folder and licence acceptance, two possible runner processes and the
install / uninstall changes listed in 1.3.

### 4.9 Flame probe (WP0, before WP7)

A probe bundle `ofx/recipes/probe/` (as Merge's probe, own ids, CMake option default OFF), two or three descriptors
with grouping "Edge Recipes", logging every action. Questions: (1) Flame lists every plugin of the binary under "Edge
Recipes" with the effect names; (2) a saved Batch with two probe effects reloads each to the right descriptor; (3) the
fourth socket's position and colour; (4) tab 1 filled to 6 columns pushes the next page's columns to a tab named after
it; (5) label strings rewritten per render (re-check of LTX's finding in this layout); (6) the Version label per
descriptor; (7) delete + re-add keeps nothing from the old node (expected). Fallback if (1) fails: one bundle per recipe
(`EdgeRecipes<Recipe>.ofx.bundle`, one descriptor each, same static libraries; the runner and ORT stay in one
"EdgeRecipes" bundle that the others find through a config key `er_runner`, as plugin 2's `ml_runner` override,
`ofx/ml/CLAUDE.md:157-159`).

## 5. Rendering

### 5.1 Threading and frames

As the other plugins: `setHostFrameThreading(false)`, `eRenderInstanceSafe`, temporal clip access on the effect and
Source (`plugin_ml.cpp:1059-1068,1091`); every action runs on Flame's main thread (`ofx/CLAUDE.md:87-89`), so the
whole recipe is one synchronous render (decisions 5, 14, 28). Inside: one `Workers` pool per render shared by all steps
(`morphology.h:36-46`); the model step's runner call is the only out-of-process work. Neighbours are fetched once per
render (`collectNeighbours`, `render_chain.h:105-112`) and shared by Jump Repair, the vote and Temporal (plugin 2
already shares one fetch between two steps, `docs/jump_repair.md:42-44`); `fetchImage(t +- k)` inside `render()`,
`getFramesNeeded` being honoured, and the fetch cost are **UNVERIFIED in Flame** (`ofx/ml/CLAUDE.md:153-154`), which
makes plugin 2's pending TEMPORAL_TEST the first gate for this plugin (WP0).

Flame caching: whether Flame caches OFX results between frames or on revisit is UNVERIFIED (`ofx/ml/CLAUDE.md:181-182`);
the design assumes no caching (each scrub re-renders) and does not rely on it.

### 5.2 Per-frame cost (estimates from existing measurements; Linux, warm)

| Step | HD (1920x1080) | UHD | Source |
|---|---|---|---|
| Neighbour fetch | about 11 ms each (2-4 per frame) | more | `ofx/ml/PLAN_FEATURES.md:52` (UNVERIFIED in a Flame log) |
| Jump Repair, Span 1 | < 20 ms | 20 ms at 2880x1620 (2 threads) | `docs/jump_repair.md:77` |
| Region prep + stats | < 5 ms | < 10 ms | estimate (one pass + feather on a box) |
| Model candidate, one crop | about 6 ms per 512 crop + 2 ms runner overhead; cold first frame 0.6-0.8 s | same (crop-bound) | `ofx/ml/CLAUDE.md:194,165-168` |
| Tracer, one region | a few ms per 50x50 region: **UNVERIFIED** (Python 0.3 s per 112x112) | crop-bound | `docs/strand_tracing.md:77-79` |
| Strand vote (tracer on 2K neighbours + warp) | 2K x tracer | | estimate |
| Fusion + blend | < 10 ms (crop) | | estimate from `fuseCandidates` being per pixel |
| Snap in region | <= 23 ms (whole-band cost at UHD; less if cropped) | 23 ms | `ofx/src/ops/CLAUDE.md:106-107` |
| Temporal K = 1 / 2 | 75-96 / 130-185 ms | 186-210 / 365 ms | `ofx/ml/CLAUDE.md:153`, `PLAN_FEATURES.md:51` |

Hair Locks HD, K = 1, all on: about 170-230 ms warm, Temporal and fetches dominating; UHD about 350-450 ms. Within
decision 28's proposed gate (warm HD within about 1 s, `ofx/ml/CLAUDE.md:179-181`), above plugin 1's 100 ms (this node
is not plugin 1). Sharing the flow between Temporal and the vote avoids the second flow (the largest shared saving).

### 5.3 First / last frame, proxy, render scale

- First / last frame: Jump Repair passes through when t - 1 or t + 1 is missing (`jump_repair.h:44`); Temporal and the
  vote use the neighbours that exist (`neighbourOffsets` clamps to the range, `render_chain.h:83-86`); the footer says
  "first frame: Jump Repair not judged".
- Render scale (decision 6): px params x `renderScale.x` (Band px, Snap radius/band as plugin 2 does,
  `plugin_ml.cpp:431-435`); % params by the delivered height (Region Grow / Feather, Merge's rule); Jump Repair's sizes
  are already area / height relative (`docs/jump_repair.md:22,28-30`). Tracer constants (step 1.5 px, widths 0.5-7 px,
  connect 12 px, min piece 12 px) are 1080-line values: scale by delivered height / 1080, with a floor of 1 px; at a
  quarter-res proxy strands of 2 px vanish below the sampling limit: the proxy result will differ (accepted, footer
  note "proxy: Rebuild Detail is approximate"). The model crop at proxy: same rule as Edge Refine (crop in delivered
  pixels). Behaviour in Flame proxy mode: UNVERIFIED (open for every plugin here).

## 6. Testing

Tiers (global rule 5B): **smoke** (< 60 s, one per module, marked): the planner's all-off hash, the equivalence check on
one small scene, the region bit-identity check, the tracer known-answer case; **targeted**: all tests of the changed
module plus direct dependents (e.g. a `flow.h` change runs `ml_temporal`, the vote and the recipe tests); **full**:
main thread only, at milestones, under the project's full-suite lock.

- Per-step unit tests exist for every reused core (`test_jump_repair`, `test_merge_fusion`, `test_snap_to_image`
  incl. `testWeight`, `test_region_trimap`, `test_temporal`, `test_edge_refine`); new: `test_tracer` (port of
  `strand_tracing.md` section 4 metrics; the Python prototype's numbers as the reference within a tolerance, plus
  brute-force references for the cross-profile), `test_strand_vote`, `test_region_crop` (model input: ImageNet
  normalisation, unsharp, quantised trimap, crop multiple of 32), `test_evidence` (the spike's checks).
- Recipe-level: (1) all steps off == input (hash); (2) independent steps == solo cores chained by hand, bit for bit
  (spike pattern); (3) region steps bit-identical where weight 0; (4) never below the base inside the region; (5) each
  planner condition exercised by a scene; (6) a fake runner (`ofx/ml/tests/fake_runner.cpp`) for the model step,
  including "runner fails -> others still run"; (7) needs union -> frames needed.
- Synthetic scenes (extend `ofx/tools/synth.*`, decision 21): a **hair-locks sequence**: coarse base + strands of 1.5-6
  px leaving the solid matte across a 15-25 px gap, a drawn region per frame, the distractors of the tracer research
  (glove edge, glasses band, moles, creases), dark-on-dark and light-hair looks, 9-20 frames of motion with one injected
  1-frame jump and a strand that fades in and out (vote test); truth alpha and per-strand ids. Reuse
  `hair_flat` / `hair_busy` and `makeSequence` / `makeJumpSequence`.
- System test: the Flame-like host (`ofx/merge/systest/ofx_host.*`, already used by Merge, Refine and ML): describe,
  layout (columns, rows, tabs), defaults, all-off render == Source alpha, Hair Locks render == the core path bit for bit,
  getFramesNeeded ranges, footers (no region, empty region, gate closed, first frame), Collect Logs.
- Flame test sheet `ofx/recipes/package/RECIPES_TEST.txt` (same style as `FOCUS_REGION_TEST.txt`): sockets and their
  colours, picture tag with masks connected, scrub with neighbours (log: fetch ms), Rebuild Detail on/off and the
  footer, View Rebuilt, Temporal erasing strands or not (A/B with region weight), first/last frame, proxy, Batch Render
  UHD, two nodes + plugin 2 in one session (two runners), gate closed.
- NDA: real footage pixels are never viewed by agents; real frames are used on the user's machine for derived numbers
  only (as `docs/strand_tracing.md` section 5, `docs/jump_repair.md` real matte numbers); dumps stay on the machine.

## 7. Build plan

Work packages, each sized for one agent session (Opus for algorithmic / correctness-critical ones, Sonnet / main thread
for wiring, builds, docs; memory rule). Linux only.

| WP | What | Depends on | Parallel with | Agent |
|---|---|---|---|---|
| 0a | Flame: neighbour fetch via plugin 2's pending TEMPORAL_TEST (`fetchImage(t +- k)` in render, getFramesNeeded, fetch cost) | - | all code WPs | user + main |
| 0b | Flame probe of multi-descriptor bundle, sockets, tabs (4.9) | - | 1-6 | Sonnet (probe) + user |
| 1 | Multi-effect skeleton: `RecipeFactory<Tag>` per recipe, shared `RecipeEffect`, `RecipeDef` table (params, columns, tabs, views, sockets), `getPluginIDs`, `MR_BUILD_RECIPES`, Version label, logging, Collect Logs; `test_load_bundle_recipes` (count + ids), `RecipeDef` layout test, host extended to pick plugin n | - | 2, 3, 4 | Sonnet/main, Opus review of the factory part |
| 2 | Step interface + evidence cache + planner (from the spike), adapters for Jump Repair, region weight, fusion, Snap in region; recipe tests (all-off, equivalence, region bit-identity, conditions) | - | 1, 3, 4 | Opus |
| 3 | Tracer C++ port (`mr_tracer`): detectors, cross-profile, tracing, connect, hair context per frame with per-shape fallback, capsule alpha, drop < 12 px, loose pieces kept unless touching the roto edge (strand-gap separation: nice-to-have, not a target); tests vs the Python prototype's metrics | - | 1, 2, 4 | Opus |
| 4 | Model candidate: one crop centred on the region (multiple of 32 + margin) through `RunnerClient::run(modelId, ...)`, ImageNet normalisation as a per-call / model-table property (plugin 2's Edge Refine unchanged until its `trimap_eval` re-check), unsharp sigma 1 / amount 1, trimap 0 / 0.5 / 1 via `regionTrimap`, low-contrast switch, Strand Gain, never-lower; fake-runner tests + one real-runner eval on synthetic scenes | - | 1, 2, 3 | Opus |
| 5 | Temporal: optional weight map in `stabilise()` (nullptr = bit-identical) and an entry taking precomputed flows (shared evidence) | - | 1-4 | Opus |
| 6 | Strand presence vote over neighbours (flow-warped pieces, confidence hysteresis) + a synthetic hair-locks sequence in `ofx/tools/synth` | 3, 5 | 7 | Opus |
| 7 | Hair Locks effect: its `RecipeDef` (tab 1 with 8 exposed controls, Advanced tab), footers, Plan labels, per-step gate, getFramesNeeded (union of needs), views, colour tags; `recipe_ui` unit tests; system test through the host | 1, 2, 0b (layout), 4 for the model toggle | 6 | Sonnet/main, Opus review of render wiring |
| 8 | Hair Locks end to end: scene scores (strand recovery, false alpha, distractors, flicker), Temporal-vs-strands A/B (Q3), timing at HD / UHD / 4096x2304 | 2-7 | - | Opus |
| 9 | Packaging: Edge Recipes tarball (bundle with runner + ORT + licences.txt identical to plugin 2's, setup.sh, install.sh, docs); `install.sh`: add the bundle, uninstall only what the package shipped, stop both runners; `setup.sh --remove` warns when another ML bundle is installed; `package_release.sh` entry; README / USERGUIDE "Refines" / "Rebuilds"; Flame test sheet `RECIPES_TEST.txt` | 7 | 8 | main |

Critical path: 1 -> 7 -> 8 with 2, 3, 4, 5 in parallel before 7; 0a must report before 6 and 8 are trusted; 0b before 7.

Risks: (1) neighbour fetch in Flame (cost, honoured, invalidation) still unverified; the temporal parts depend on it.
(2) Temporal erasing rebuilt strands (5.3; mitigated by the region weight, to be measured). (3) Tracer tuned on synthetic
scenes and one real frame (`strand_tracing.md` section 6); flicker near its confidence threshold. (4) The ImageNet
normalisation changes plugin 2's Edge Refine output if applied globally (keep it per call). (5) Picture conversion when
masks are connected. (6) Frame time above 300 ms at UHD with K = 2. (7) Two runners when plugin 2 and Edge Recipes render
in one session (memory). (8) Flame listing several effects of one binary (probe 0b; fallback one bundle per recipe).
(9) Per-instance workspaces with many nodes at UHD.

## 8. Naming

- Bundle `EdgeRecipes.ofx.bundle`, binary `EdgeRecipes.ofx` (name provisional, decision 41).
- Menu: grouping **"Edge Recipes"** (flat; Flame's OFX menu is plugin group then effect, user 2026-11-12), effect name =
  the recipe, first **"Hair Locks"**. Plugins 1-3 stay under "Matte Refine".
- Plugin ids: `com.<owner>.edgerecipes.<recipe>` with the same owner part as the existing ids; whether that owner part
  changes before the repo goes public is a separate decision (Q8). Never changed after a release.
- CMake `MR_BUILD_RECIPES`, `ER_VERSION` (bundle) + a version per recipe descriptor; log component "recipes"; config keys
  `er_*` if any.

## 9. Open questions for the user (ranked)

1. **"Installs everything"**: does the Edge Recipes package contain only the Edge Recipes bundle (which already has
   every method inside; assumed here), or also install the three solo bundles next to it?
2. **Hair Locks' 8 exposed controls** (4.3): right set and defaults (Jump Repair on, Edge Model on, Rebuild Detail
   off, Temporal on)?
3. **Temporal and the rebuilt strands**: Temporal outside the region only (recommended), before the fusion, or after
   with the risk of erasing strands (A/B on the synthetic sequence first, WP8)?
4. **Two runner processes** when plugin 2 and Edge Recipes render in one session: acceptable for now, or plan the shared
   runner (protocol accepting any build with the same protocol number) before the first release?
5. **Cross-render reuse** for the strand vote (tracing 2K neighbour frames each render): small in-memory per-instance
   reuse acceptable, or does decision 14 (no cache) hold here too?
6. **Behaviour pinning** with `recipeRev` (old nodes keep old code paths) or old nodes always follow the newest recipe?
7. **Region roto over time**: fetch the Focus Region at t +- K for the vote (temporal access on a mask clip, untested),
   or use the region at t widened by the flow?
8. Plugin id owner part and the final bundle name.
9. Strand Gain default (research: levels 0.05-0.8 helps opaque synthetic strands; real hair is translucent: mild).
