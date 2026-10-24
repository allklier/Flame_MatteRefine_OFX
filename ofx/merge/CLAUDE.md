# ofx/merge/CLAUDE.md — Matte Merge, the third plugin (core and plugin built 0.1.0; Flame test pending)

Started 2026-10-23 from a user idea (decision 37 in the root `CLAUDE.md`), revised 2026-10-24 after the user's review
(general, matte-only output, no colour handling). Tags as in the root file. The method is **PROPOSED**; decided: separate
plugin, classical first, this repo, general (not LTX-specific), output = the combined matte only.

## What it is
Two existing mattes in, one combined matte out. A **base matte** (any source: roto, a guided model, a keyer) is right about
the subject and the body edge but has poor edge detail. A **detail matte** (any source: LTX Alpha Gen, a keyer, another
model) has better detail at the edge. The node keeps the base where it is right and takes the detail matte's finer edge
information only in a limited zone, aligned first so that no errors are introduced. Hair is the first test case, not the
definition: fur, fuzzy cloth, grass and soft motion-blurred edges are the same problem. Still decision 17: the base matte
stays the source. Why a separate plugin (user): a different workflow (two mattes); plugin 1 stays one matte in / one out,
plugin 2 is model-based. **Atomic by design (user):** no colour handling, no Edge Colour Unmix (plugin 1 already has it;
the artist may have better colour methods). The node outputs the combined matte, nothing else.

## Proposed method
1. **Zone (where the detail may act).** Auto: the base matte's grey pixels plus *Reach px* around them, minus its solid
   core. Or **Trimap**: the base input is a trimap (values near 0 and 1 are known, anything between is unknown) and the zone
   is the unknown region. The user's idea: a trimap on the base, with the smart merge happening only in the grey zone. The
   base is bit-identical outside the zone. Detail not connected to the zone (blobs, other objects) is ignored.
2. **Registration** (user: important). The detail matte may be offset or slightly scaled (LTX pads to multiples of 32;
   other sources differ). Estimate the offset / small scale on the thick, confident parts where both mattes agree (their
   morphological openings), not on the fine detail; apply to the detail matte; QC view shows the residual. Reuses the
   pyramidal Lucas-Kanade flow from plugin 2's Temporal (`ofx/ml/src/flow.{h,cpp}`, would move to a shared library) or a
   simple translation + scale search. Align = Auto / Off; drop it if real pairs prove already aligned. A merge that is
   misregistered makes detail float off the edge, worse than no merge.
3. **Combine** inside the zone, two methods to evaluate:
   - **Add fine detail** (hair, fur): extract the thin structures of the detail matte (white top-hat: detail minus its
     opening at *Detail Width px*; thick areas such as the detail matte's own body edge drop out), screen-combine with the
     base, `1 - (1 - base)(1 - fine)`. Never lowers the base, so its body edge is preserved.
   - **Zone blend** (general edges, trimap): inside the zone take the better of the two per pixel, weighted by quality
     evidence (below), feathered over *Feather px* at the zone border. With a trimap base the unknown zone has no base
     information, so the detail wins there.
   *Strength* scales the detail's contribution.
4. **Quality evidence (optional, picture-driven), no colour space needed (user).** The picture is used only as a guide:
   local contrast (edge strength) and local saturation (chroma), each normalised by its local robust range, so the result
   depends on structure and not on the encoding (log or display). A matte's edge detail is trusted where the picture has
   an edge or fringe of the same structure; detail with no picture support (hallucinated strands) is rejected. No
   colourspace request, no conversion, Picture Colour Bypass semantics (decision 34 not needed). The picture is always connected
   (see Inputs), the evidence option can be off. The assumption to check: normalised local contrast and saturation are close enough
   to encoding-invariant; test the Log3G10 clip against a display-rendered copy and compare the decisions.
5. **Views (QC, not controls):** Result, Added (what the detail changed), Rejected, Registration residual, Zone.

Later options, only if wanted: a temporal pass on the detail layer (plugin 2's `stabilise`, LTX hair flickers), an ML
pass over the zone, symmetrical "best of two" with no designated base.

## Parameters (draft, decision 19: one group per column, at most 4 controls + Enable)
Zone: Source (Auto / Trimap), Reach px · Registration: Align (Auto / Off) · Combine: Method (Add fine detail / Zone blend),
Detail Width px, Strength, Feather px · Support: Picture Evidence (Off / On), View (Result, Added, Rejected, Residual, Zone).

## Inputs: the main risk
**DECIDED (user, 2026-10-24): the picture is mandatory**, on the front as in plugin 2 (decision 35), even when the
evidence option is off: Flame behaves better with image size and colourspace known (plugin 1's missing-front failure,
decision 22), so it settles size, PAR and tags architecturally. **The base matte (or trimap) is on the source alpha, the
detail matte is the one extra clip.** **DECIDED (user): the trimap is fed as the base input**, no third matte input. Plugin
2 dropped its separate Matte mask clip because it made Flame deliver the picture already converted and tag the source
"Matte" (decision 35); with no colour handling here this only matters if it changes the picture values the evidence uses.
**First step: a tiny logging probe plugin** testing (a) the base on the source alpha with the detail as a second clip
declared non-mask RGBA versus mask, (b) which socket each lands on, (c) whether the picture arrives raw, (d) the reported
colourspace tags. **Probe built 2026-10-24, not yet run in Flame:** `ofx/merge/probe/` (bundle
`MergeProbe.ofx.bundle`, id `com.janklier.mergeprobe`, CMake `MR_BUILD_MERGE_PROBE`, default OFF; variant by config key
`probe_detail_kind` = none / rgba / mask, read at load; test sheet `ofx/merge/probe/PROBE_TEST.txt`). Output: combined matte in alpha; RGB per a View like plugin 1 (decision 11), colour tags as decision 23.

**Probe result (Flame 2027.x on akflame5, 2026-10-24, variants rgba and mask; `none` not run):** (1) the extra optional
"Detail" clip is accepted in the filter context in both declarations and the mattes arrive intact (source alpha mean 0.3146,
Detail 0.3176, bounds/PAR as the picture). Declared **rgba** it arrives as RGBA with the matte copied into R, G, B (A = 1);
declared **mask** it arrives Alpha-only (1 channel). (2) **Same quirk as plugin 2:** with Detail unconnected the picture
arrives raw (LogC3 tag, rgb max 0.824, hash stable); the moment Detail is connected, in either declaration, Flame delivers
the picture converted (rgb max 2.95) and tags the Source "Matte". So connecting a second matte always converts the picture;
this is no reason to avoid the clip, because Matte Merge does no colour maths: the evidence step (normalised local contrast
and saturation) must work on whatever arrives. Option if that is not enough: request a fixed space through
`OfxImageClipPropPreferredColourspaces_Source` (decision 34), untested with a Detail clip. **DECIDED (user, 2026-10-24, after seeing the sockets): declare Detail as a mask clip.**
Sockets seen: mask variant = one red (front picture) + two blue (matte) sockets, the clean layout (first blue = base, goes with the picture; second blue = Detail, user-confirmed); the other two variants
showed four sockets, the third green (a colour clip), the fourth blue. Alpha-only is also 4x less data.

## Core built 2026-10-24 (`MR_BUILD_MERGE`, default OFF; 15 tests, 16 mutations all caught; no plugin yet)
Files: `src/matte_merge.{h,cpp}` (`merge(base, detail, picture|nullptr, w, h, params, out, views, info)`), `merge_register`,
`merge_evidence`, `merge_morph` (exact distance transform, fast min/max/opening), `tools/merge_synth` + `mm_synth` (24 EXR scenes
at 1920x1080 in `ofx/merge/data/`, git-ignored, 602 MB: hair_flat / hair_busy with simulated detail, trimap and log-picture
variants), tests `test_merge_core` (known answers, brute-force references), `test_merge_scenes` (scores; the limits are
regression limits from first measurements). Choices: zone = base grey pixels or 0.5 contour within Reach, minus solid core,
trimap = its unknown pixels, smoothstep Feather, bit-identical outside; detail kept only if connected to the base's 0.5+
pixels; Add = `base + f(1 - base)` with `f = max(0, detail - opening(detail))`; evidence judges the thin structure only, via
local correlation of change with luma V and chroma spread S (7x7, floor 1% of range, smoothstep 0.3-0.6, 9x9 smoothing).
Registration: translation + scale + uniform edge offset ("delta", else thickness differences read as scale) on the signed
distance fields of both mattes' openings, Gauss-Newton with Huber weights at 1/4 then 1/2 resolution; returns a
`sensitivity`. The flow code of plugin 2 was NOT reused (dense per-pixel, needs the ML library).
**Scores (2880x1620, Add, reach 380):** strand recovery 0.446 (align off 0.28; true move given 0.50); body edge changed
0.0001, nothing lowered, nothing outside the zone changed; blob 0; fake strands added 0.50 -> 0.12 (flat) / 0.28 (busy) with
evidence, which costs 15-28% of real strands; evidence decisions agree 99.6% / 96.7% between display and log pictures;
trimap fill error 0.056 (no align 0.10, prior ramp 0.40). Zone blend recovers 0.63 but takes the detail's thickened body edge
(overshoot 0.62, 646 px lowered): Add stays the default. **Targets not met:** registration 1.45 px (target 0.5) on the
coarse base (0.13 px with a uniform edge offset, 0.49 px trimap): ill-conditioned on a convex subject cut by the frame
(sensitivity ~9 px per px; ty, scale and delta trade off); dead ends: leaving out edges near fine detail, even harmonics
of the edge direction (2-9 px, worse). Fake-strand rejection on busy backgrounds is weak. Timing (BFD, 24 threads): merge
alone 63-79 ms (no align) / 91-112 (align) / 122-148 (+ evidence) / 187-193 ms (reach 380, all views): more than the
"well under 200 ms" for a chain; reusing scratch buffers would help. Reach and Detail Width are px and do not scale yet.
**Open for the user:** picture-based registration (strands in the picture sit at the true positions; would remove the
thickness confound but contradicts "not on the fine detail"); fall back to translation-only / off when sensitivity is
high, shown in QC; Reach and Detail Width defaults and scaling with frame height; Evidence default On?; trimap "dropped
detail = background" ok?; Residual view as signed px distance error?

**Decided by the user 2026-10-24 (answers to the core's open questions):** (1) registration may use the picture, and the picture
is required (already mandatory, connected); (2) when the registration estimate is unreliable (high `sensitivity`) fall back to
translation only; (3) Evidence default: see the plugin step (recommended Off until tested on real footage; the Rejected view
shows what it would drop); (4) Reach and Detail Width are set as a fraction of frame height, so they scale with resolution
(proposed defaults: Reach 6 %, Detail Width 0.5 %, Feather 0.25 % of height; the user has no opinion, tune on real pairs);
(5) in a trimap fill, detail that fails the connectivity test becomes background by default, with a **Keep Disconnected** switch (default off) that keeps it (decided 2026-10-24 after the scenario: blobs and invented strands versus detached real flyaways).

## Core v2 (2026-10-24, merged; 17/17 with MERGE ON, default build 13/13, 15 mutations caught)
- **Picture registration** (`merge_picreg`): the detail's top-hat (thin parts) is matched block by block (about 48 px at 1080,
  near the zone) against the picture's brightness and chroma, both high-passed; peak of the normalised cross-correlation
  (sub-pixel by parabola; the sign gives the strand's polarity); robust fit to translation + scale in 3 rounds. No colourspace
  assumption. Error at 2880x1620: 0.020 px (flat) / 0.17 px (busy), log picture the same; trimap 0.04 / 0.11 px; 960x540 busy
  up to 0.37 px. Dead ends: difference-of-Gaussians band (busy worse), larger blocks (worse at 540).
- **Fallback order** (thresholds: uncertainty <= 0.25 px, block RMS <= 0.35 px, >= 25 % of blocks agree): picture
  translation+scale, picture translation only, mattes translation+scale if sensitivity <= 4, then **no alignment**
  (`AlignModel::None`). DECIDED (user, 2026-10-24): a matte-only translation estimate was dropped as a last resort because
  it left up to 13.9 px error on scenes with real scale (the full model: 1.45 px); a predictable error beats a hidden drift.
  `MergeInfo.align` reports the model, uncertainty and agreement; the footer should say when there is no alignment.
- **Sizes as % of frame height** (`HeightSizes`, `scaleToHeight`; px defaults are the 1080 values 64.8 / 5.4 / 2.7): the same
  scene at 2160 and 1080 gives the same result (ratio 0.994). Below 1080 lines results fall off (the opening is whole px).
- **`keepDisconnected`** (trimap, default false) implemented and tested. **Workspace** (`MergeWorkspace`, one per instance) saves ~40 ms.
- **Scores (2880x1620, defaults Reach 97 px):** strand recovery 0.298, fake strands added 0.255 (evidence on: 0.094 / 0.143);
  Reach 380: 0.502 = the oracle with the true move (so registration is no longer a limit), fakes 0.536 (evidence 0.16 / 0.30).
  Synthetic strands are up to 360 px long, far longer than real hair, which is why the default reach recovers less.
- **Timing (2880x1620, 16 threads):** Add + align 54 ms, + evidence 80 ms, trimap 62 ms, Reach 380 + evidence + all views 109 ms.
- API for the plugin: `matte_merge.h` (`MergeParams`, `MergeInfo`, `MergeWorkspace`, `merge(...)`); the picture is `w*h*3` RGB
  (the plugin repacks Flame's RGBA).

## Plugin built 2026-10-24 (0.1.0, not yet seen in Flame)
Bundle `MatteMerge.ofx.bundle`, id `com.janklier.mattemerge`, CMake `MM_VERSION` 0.1.0 under `MR_BUILD_MERGE` (default OFF).
Files: `src/plugin_merge.cpp`, `src/merge_ui.{h,cpp}` (no OFX types: controls -> `MergeParams`, footer texts, output
colourspace answer; unit-tested), `tests/test_merge_ui.cpp`, `test_load_bundle_merge.cpp`, `check_exports.sh`, `systest/`
(a Flame-like host copied from LTX's `systest/host` and adapted: the first systest of this repo; renders through the plugin and
checks the alpha equals `merge()` pixel for pixel, views, trimap, pass-through footers, colourspace answers, Collect Logs),
test sheet `package/MERGE_TEST.txt`. 21/21 ctest with MERGE ON (13/13 default); `ctest -j` trips the timing limits of existing
tests (run serially). Filter context: Source mandatory (picture + base/trimap in its alpha), "Detail" optional mask clip;
output alpha = combined matte, RGB per View (Result Matte, Picture, Added, Rejected, Zone, Residual).
**Layout (one tab, 5 columns):** Matte (View, Version) · Zone (Source Matte Edge / Trimap, Reach %, Keep Disconnected) · Align
(Align Auto / Off) · Combine (Method Add Fine Detail / Zone Blend, Detail Width %, Strength, Feather %) · Support (Picture
Evidence Off / On, Debug Logging, Collect Logs, hidden node id). Param ids and choice indexes are append-only from now on.
Behaviour: Detail missing -> base passes through with "Connect the detail matte to the Detail input."; constant base alpha ->
footer asks for the base matte or trimap; wrong-size detail (not resampled), non-float input or an exception in `merge()` ->
base passes through with an error footer; footer shows the align model ("no alignment (could not register)" for None).
Colour as plugin 2 (View Picture -> Source's colourspace, other views -> "Matte" or raw). Added view: white = detail added, green =
Zone Blend lowered the base. Open for Flame: info footer persistence across renders (else send it every render), greying out
Keep Disconnected unless Source = Trimap (needs `setEnabled` in `changedParam`, unverified), proxy/render-scale behaviour,
UHD speed, the sockets as in the probe's mask variant.

**Strength gain (user, 2026-10-24, after the first Flame look):** Strength 0-1 = how much detail is taken; 1-2 = also a gain on
what is ADDED (only where the detail exceeds the reference; never lowers; result clamped to 1). Why: the simulated detail
matte (the truth blurred, moved, scaled) has weaker strands than the truth and the node can add at most what the detail has;
Strength was capped at 1. Called out in USERGUIDE section 6 (it also brightens invented strands). Also learned from that test:
in Trimap mode the zone is the grey band, so Reach does nothing; the test trimap's band is narrower than the synthetic strands.

**Mac (user, 2026-10-24):** Matte Merge ships on macOS (Apple Silicon) with plugin 1; no model or GPU part. BUILT and tested on lbm
(M1 Ultra, 20 cores, macOS 26.7.1, build area `/Volumes/AvidMedia/ClaudeCode/Projects/MatteRefine_OFX/build`, `scripts/build_mac.sh`
with `-DMR_BUILD_MERGE=ON`): `Contents/MacOS/MatteMerge.ofx` arm64 + `Info.plist`, ctest 19/19 (plugin 1's 13 + merge_core,
scenes, align, ui, load_bundle, systest; `merge_bundle_exports` is Linux-only). No portability fixes were needed. Timing at
2880x1620 (best of 3, reused workspace): Add + align 55 ms, + evidence 68, no align 39, Zone Blend 45, Trimap 50, Reach 380 + evidence
80, all views 89 ms: the same as Linux (BFD 24 threads: 54 / 80). Dev symlink: `sudo ln -s
/Volumes/AvidMedia/ClaudeCode/Projects/MatteRefine_OFX/build/MatteMerge.ofx.bundle /Library/OFX/Plugins/`. Not yet tried in Flame on the Mac.

## Reuse
Plugin 1's `mr_ops` (edge distance, morphology, Workers); plugin 2's flow; footers and logging; P4 synthetic scenes
(`ofx/tools/synth`): `hair_flat` / `hair_busy` give a coarse base and truth strands. A simulated detail matte = truth with
blur, shift / rescale, a thickened body edge, an extra blob and hallucinated strands. Also a trimap case (grey zone around a
hard body edge). Real tests: the Log3G10 kitchen clip with an LTX matte; confidential dumps stay on the machine.

## Plan
0. Layout probe in Flame. 1. Classical core + tests on synthetic scenes (scored on strand recovery, body-edge damage,
rejection of the blob and fake strands, registration error, bit-identity outside the zone; trimap case; mutation checks).
2. Plugin, columns, views, logging, third bundle (CMake option, default OFF like `MR_BUILD_ML`, decision 27 pattern).
3. Flame test with real pairs. 4. Optional temporal / ML extensions.

## Open questions
Which combine method is default (Add fine detail versus Zone blend; a trimap base makes it a plain fill of the unknown
zone from the aligned detail matte); whether registration is needed on real LTX output; how well normalised contrast / saturation
hold up across encodings; temporal handling of flickering detail; bundle name and plugin id (`com.janklier.mattemerge`
proposed); installer shared with the other two.
