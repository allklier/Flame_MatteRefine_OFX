# Plate Merge (Plate Tools bundle) — design, behaviour, tests, limits

Written 2026-11-27. Status: first release (Plate Tools 0.1.0). Core, plugin, unit tests and a host system test are built; the maintainer
tried it in Flame on Linux and found the result correct. Not tried on macOS. Root decision 43 (accepted 2026-11-27).

## The problem
Two plates of the same camera, aligned to within a pixel (< 1 px drift), from different takes. A person moves in each.
Plate B's person is comped over plate A; plate A's person must not peek out from under B's person, and both persons'
backgrounds exist in the *other* plate. Tight mattes of each person exist (LTX Alpha Gen, optionally cleaned in Matte
Refine); they can carry a faint haze tail, and soft shadows / defocus halos reach past a tight matte.
Per pixel: neither matte: plate A; only one matte: the plate that does not have a person there; both: B's person is on
top (split screen), or no clean pixel exists anywhere (clean background). For the split screen this reduces to plate B
over plate A through the union of the two (grown, feathered) mattes, with B's level matched to A's so the seam vanishes.

## Placement (decision 43): a new bundle "Plate Tools", first effect "Plate Merge"
- **Not Matte Refine (plugin 1):** one matte in / one matte out on one picture (decisions 1, 3, 17).
- **Not Matte Refine ML (plugin 2):** no model; plugin 2 is the model-based node and Linux-only.
- **Not Matte Merge (plugin 3):** atomic by design, its output is a combined *matte* and it does no colour work (decision
  37, user 2026-10-24). Plate Merge outputs a *picture*, takes two pictures and does a level match: a different workflow.
- **A new bundle, `PlateTools.ofx.bundle`**, with one effect per factory class (the pattern verified for Edge Recipes,
  decision 41: one binary, several effects, one factory class each because the Support library keeps the plugin id in a
  static member per factory type, `ofxsImageEffect.h` `PluginFactoryHelper`). Further two-plate / clean-plate operations
  (e.g. a temporal clean plate from one moving take, a plate-difference matte for a QC view) become further effects in
  this bundle without a new install. Same repo and CMake project (decision 27 pattern), option `MR_BUILD_PLATE` (default
  OFF, needs `MR_BUILD_MERGE` for the exact distance transform `merge::distanceSquared` and the systest host). Listed under
  the same Flame group "Matte Refine". Plugin id `com.janklier.platemerge`, version `PT_VERSION` 0.1.0.
- **Scope (decisions 17 and 42):** decision 17 is about mattes (refine, not replace); Plate Merge makes no matte from the
  picture and changes no matte: it consumes the artist's mattes and composites pictures. That is a scope EXTENSION of the
  package beyond matte refinement, hence a scoped exception the maintainer must approve. Decision 42's test: the problem
  recurs (split screens and take merges are common in comp), and the manual fix is costly (a moving person on every
  frame, plus the halo and level seam by hand), so it passes.

## Sockets (Flame layout UNVERIFIED, see the end)
| Clip | Kind | Carries |
|---|---|---|
| Source (front + source alpha) | mandatory, RGBA | plate A, its person's matte in the alpha |
| `PlateB` "Plate B" | optional, RGBA, not a mask | plate B, its person's matte in the alpha |
| `KeepA` "Keep A" | optional mask | protect roto: plate A is kept there, bit for bit |
| Output | RGBA | RGB = merged plate (or a QC view); alpha = Swap Matte or Residual (Matte Out) |

## Controls (one tab, three columns; ids are stored in setups, append only)
| Column | Control (id) | Default | Notes |
|---|---|---|---|
| Plate Merge | View (`view`) | Result | Result, Plate A, Plate B Matched, Swap Matte, Residual, Ring |
| | Matte Out (`matteOut`) | Swap Matte | what the alpha output carries: Swap Matte / Residual |
| | Version (label) | | |
| Merge | Mode (`mode`) | B Over A | B Over A / Clean Background |
| | Grow px (`growPx`) | 8 | full-resolution px, scaled by the render scale (decision 6) |
| | Feather px (`featherPx`) | 8 | idem; 0 = hard edge |
| | Level Match (`levelMatch`) | on | |
| Support | Debug Logging, Collect Logs, hidden node id | | as the other plugins |

## Algorithm (`ofx/plate/src/plate_merge.{h,cpp}`, pure C++14, no OFX types)
1. **Swap weight per matte:** `m'` = matte clamped to 0-1 (NaN 0), values below the haze floor 0.02 set to 0 (fixed); `d` =
   exact Euclidean distance to the nearest pixel with `m >= 0.5`; `ramp(d)` = 1 for `d <= Grow`, smoothstep down to 0 at
   `Grow + Feather`; weight = `max(m', ramp)`. Soft values of the matte are weights, not switches; Grow reaches past halos.
2. **Share of B**, with `a`, `b` the two weights and `k` Keep A (0 when unconnected):
   - B Over A: `s = (1 - (1 - a)(1 - b)) (1 - k)` (screen union);
   - Clean Background: `s = a (1 - b) (1 - k)` (B only where A's person is and B's is not).
   Residual = `a b` (both persons; no clean pixel anywhere).
3. **Level Match (optional):** ring = pixels with `a = b = 0`, `k = 0`, finite in both plates, within the ring width (2 % of
   the height, at least 4 px) of the swap region. Per channel, per cell of a coarse grid (cell = sigma / 4), the count and
   the sum of `log2(A / B)` over ring pixels where both values are above 1e-4, each sample clipped to +-2 stops (a mean of log
   ratios, not a ratio of means: a highlight would otherwise dominate, and a sub-pixel drift averages out). Both are blurred
   with a Gaussian (sigma 5 % of the height, at least 8 px), then `gain = (blurred sum + p g) / (blurred count + p)` with
   the whole-ring mean `g` as a prior of weight `p` = 10 % of a full cell: where the ring is thin or absent the gain falls
   back to the global one. Clamped to +-2 stops; bilinear from the grid. Skipped (gain 1) with fewer than 64 ring pixels.
   Scene-linear assumption: a gain is the right correction only in linear light.
4. **Output:** `out = A` where `s = 0` (bit for bit), `B'` where `s = 1`, `A + s (B' - A)` between, `B' = B x gain`. No clamp:
   values > 1 and < 0 pass. Float32 throughout.

**Temporal:** per frame, no temporal access (decision 5). The switch follows the mattes; Feather is the only softening. The
gain is a mean over thousands of ring pixels and stays steady (fixtures: 0.008 stops range over 7 frames of moving persons).
A flickering matte gives a flickering swap edge; clean the matte (Matte Refine / ML Temporal) first.
**Sub-pixel drift:** no registration. Grow covers the misalignment at the matte edge; inside the swap the drift is invisible
(B is a whole plate). At the feather band a drifted texture is cross-faded with itself (a mild softening over Feather px).

## Measured (synthetic fixtures, `ofx/plate/tests/test_plate_merge.cpp`, 2026-11-27)
Scenes 320x200, 7 frames: a static analytic texture around 18 % grey with a highlight to 6.0 and a patch of -0.002, two
moving soft persons with shadow halos past their tight mattes (0.35 at the edge, 1/e at 3.5 px), a 0.012 haze tail in each
matte, B 0.30 / 0.25 / 0.35 stops brighter (R G B) and drifted 0.5 px, a window reflection only plate A has (Keep A).
| Check | B Over A | Clean Background |
|---|---|---|
| pixels with `s = 0` equal to A bit for bit | 100 % (48.6k-54.8k per frame) | 100 % (57.0k-61.3k) |
| relative error vs truth, mean | 0.0075-0.0097 | 0.0031-0.0124 |
| relative error vs truth, p99 | 0.030-0.086 | 0.018-0.065 |
| seam step (mean log2 out / truth in the feather band) | <= 0.006 stops | <= 0.003 stops |
| gain error vs the true offset | <= 0.003 stops | same |
| gain range over 7 frames | 0.008 stops | same |
| Keep A pixels not equal to A | 0 | 0 |
The error floor is the 0.5 px drift itself: the background against itself half a pixel off gives mean 0.0142, p99 0.064
over the frame. With no drift and no offset: mean 0.0003, p99 0.007 (the halo tail beyond Grow). The highlight (6.18) survives.
Fault injections, each caught: plates swapped (p99 5.9), a clamp to 0-1 (highlight 1.0, 1040 pixels no longer
bit-exact), Feather 0 (swap jumps 0 -> 1 between neighbours; defaults <= 0.24), Level Match off (seam +0.15 stops, mean
error 0.18), Grow 0 (A's halo left: p99 0.29), Keep A missing (reflection lost: p99 0.36). Mutation checks on the core
(gain inverted, Clean `s = a`, ring including matte pixels, haze floor removed, linear ramp, Keep ignored, Keep in the ring,
union = max): all 8 caught.
**Timing:** core at UHD 130 ms (12 threads, best of 3, a 16-core desktop); plus the plugin's copies. Over decision 5's
~100 ms guide for a chain, acceptable for a single node; not optimised (the ring pass and the three distance transforms
are the cost).

## Limits
- Level Match assumes scene-linear plates. On log or display-encoded plates it applies a gain where an offset would be right;
  turn it off there or convert to linear first. The node requests no colour conversion.
- No registration: drift above ~1 px shows as a doubled texture in the feather band and a misplaced seam; stabilise or
  align the plates in Flame first.
- A shadow or reflection of a person that reaches further than Grow + Feather is not swapped (grow more, or include it in the
  matte). A shadow or reflection that should stay (another actor's reflection in plate A) needs Keep A.
- Different light between the takes that is not smooth over 5 % of the frame height (a moving shadow of a cloud edge,
  practical light flicker) is only partly matched.
- Residual (both persons): in B Over A, B's person covers it; in Clean Background no clean pixel exists, the output takes
  plate A there; use Matte Out Residual to drive an inpaint or a paint fix.

## UNVERIFIED in Flame (test sheet `ofx/plate/package/PLATE_MERGE_TEST.txt`)
1. The sockets: whether a non-mask optional RGBA clip shows as a front (green) socket plus its own alpha (blue) socket, as
   a Matte Merge probe variant with a non-mask clip suggested (four sockets, the third green, the fourth blue;
   `ofx/merge/CLAUDE.md` "Probe result"), and whether B's matte connected
   there arrives in the alpha of the PlateB image. If not, B's matte needs its own mask clip (an appended clip; ids stay).
2. Labels and order of the sockets (Plate A front, A's matte, Plate B, B's matte, Keep A).
3. Colour: Matte Merge's probe saw Flame deliver the Source converted and tagged "Matte" as soon as a second clip was
   connected. Plate Merge outputs a picture: check that the Result is in the same colourspace as plate A (A/B the node with
   plate A where neither matte is on: identical), and what the output tag says. If Flame converts, Level Match may run on a
   different encoding than expected.
4. Footer persistence, render scale (proxy) behaviour, UHD speed inside Flame, Batch Render.
5. Mac build (not attempted).
