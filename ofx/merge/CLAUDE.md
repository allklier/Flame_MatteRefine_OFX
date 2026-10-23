# ofx/merge/CLAUDE.md — Matte Merge, the third plugin (plan; nothing built yet)

Started 2026-10-23 from a user idea (decision 37 in the root `CLAUDE.md`). Tags as in the root file. Design below is
a **starting point (PROPOSED)** to be validated; only "separate plugin, classical first, in this repo" is decided.

## What it is
Two existing mattes in, one combined matte out: a **main matte** made by any means (roto, a guided model, a keyer) that is
right about the subject and the body edge, and a **detail matte** that has fine detail in some area (typically hair from
LTX Alpha Gen, which is hard to aim at one object). The node keeps the main matte and takes only hair / fine detail from
the detail matte, near the main matte's edge. This is decision 17 (refine, not replace): the main matte stays the source.
Why a separate plugin (user, 2026-10-23): a different workflow (two mattes), Matte Refine stays "one matte in, one
refined matte out" (plugin 1) and Matte Refine ML stays model-based (plugin 2).

## Proposed method
1. **Reach:** detail is considered only within N px of the main matte's edge: the main matte dilated by *Hair Reach px*
   minus its solid core. Anything outside, and anything not connected to that neighbourhood (other objects, floating blobs
   that LTX can produce), is ignored.
2. **Alignment:** the detail matte may be shifted or slightly rescaled (LTX pads to multiples of 32). Register it to the
   main matte with the matte-weighted pyramidal Lucas-Kanade flow built for B4 (`ofx/ml/src/flow.{h,cpp}`; would move to
   a shared library) on picture luma and the two mattes; Align = Auto / Off.
3. **Combine:** in the reach band add the detail where it exceeds the main matte, faded in from the core edge over
   *Feather px*; the main matte is untouched elsewhere (bit-identical outside the reach). *Detail Strength* scales the add.
   Modes to evaluate: **Add** (max-like, default) and **Replace** (detail wins inside the band).
4. **Picture support (optional):** keep added detail only where the picture supports it (local colour / contrast
   differs from the nearby background estimate), to reject hallucinated strands. Needs the picture.
5. **Views (QC, not controls):** Result, Added (what the detail contributed), Rejected, Alignment (flow / offset).
Later options: Edge Colour Unmix on the added hair (plugin 1's op), a temporal pass on the detail layer (B4's `stabilise`:
LTX hair flickers), an ML pass (ViTMatte) over the combined band.

## Parameters (draft, UI rules of decision 19: one group per column, at most 4 controls + Enable)
Hair Reach px · Detail Strength · Align (Auto / Off) · Picture Support (Off / On, threshold) · Feather px · Mode (Add / Replace).

## Inputs: the main risk
Needs a picture (front) plus two mattes. Matte Refine ML dropped its separate Matte mask clip because with it connected
Flame handed the picture over already converted and tagged the Source "Matte" (decision 35, `ofx/ml/PHASE0_RESULTS.md`
"Colour probe"); the filter layout (front + source alpha) is verified. For two mattes one extra clip is unavoidable.
**First step: a tiny logging plugin** that tests (a) the main matte on the source alpha plus a second clip declared
as a non-mask RGBA clip and as a mask clip, (b) which socket each lands on, (c) whether the picture arrives raw, (d) the
reported colourspace tags. Decide the layout on that. Open: is the picture required or optional (plain merges).

## Reuse
Plugin 1's `mr_ops` (edge distance, morphology, Workers); B4's flow; the ML plugin's footers / logging / colour tagging
(decisions 23, 34, 35 apply); P4 synthetic scenes (`ofx/tools/synth`): `hair_flat` / `hair_busy` give a coarse matte (main) and
truth strands. A simulated LTX detail matte = truth with blur, shift / rescale, an extra blob and hallucinated strands.
Real test: the Log3G10 kitchen clip (hair on the woman) with an LTX matte; the confidential matte dumps stay on the machine.

## Plan
0. Layout probe in Flame (above). 1. Classical core + tests on synthetic scenes mimicking LTX (shift, blur, extra blob,
hallucinated strands; scored on band error, strand recovery, rejection of the blob, bit-identity outside the reach); mutation
checks. 2. Plugin, columns, views, logging, build as a third bundle (CMake option, default OFF like `MR_BUILD_ML`, root
decision 27 pattern). 3. Flame test with real LTX mattes. 4. Optional ML / temporal / colour-unmix extensions.

## Open questions
Add or Replace as the default; picture required or optional; how often the registration is needed on real LTX output
(measure first, drop Align if LTX is already aligned); temporal handling of flickering hair; shipping: bundle name,
plugin id (`com.janklier.mattemerge` proposed), installer shared with the other two.
