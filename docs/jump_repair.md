# Jump Repair (Matte Refine ML): reference notes

Status: unit- and host-tested; one Flame test found the soft-rim outline (fixed by the growth below, **not yet
retested in Flame**). The algorithm is
described in full in `ofx/ml/src/jump_repair.h`; this file holds the reasons, the measurements and the limits.

## What it does

A matting model sometimes produces a patch of matte that appears (or a hole that opens) for 1-3
frames while the frames around it agree. Jump Repair finds those patches by comparing frame t with
its neighbours' mattes and puts the neighbours' value back, only there. No model, no picture, deterministic.
Decision 17 holds: every pixel outside a repaired patch's zone is bit-identical to the input.

## Node

Column "Jump Repair", between Matte and Temporal (definition order = chain order). With it, the first tab has
6 columns of 5 rows and is full.

| Control | id | Default | Notes |
|---|---|---|---|
| Enable | `jrEnable` | off | |
| Jump Above | `jrThreshold` | 0.5 | how far outside both references a pixel must lie |
| Min Size % | `jrMinSize` | 0.0125 | of the frame area: 30 px at 562 lines, 260 px in HD, 1040 px in UHD; resolution- and proxy-independent |
| Span | `jrSpanFrames` | 1 | a number, whole frames 1-3 (one row; it replaced the 2-row choice `jrSpan`, never released, so Motion Guards fits the column) |
| Motion Guards | `jrGuards` | on | on: the three guards below keep blobs that look like motion (red in View Jumps); off: every jump found is repaired |
| View option "Jumps" | `view` index 5 | | appended: Jump Repair's input in grey, raised magenta, lowered green (the Changes colours), blobs kept as motion red |

The column is Enable, Jump Above, Min Size %, Span and Motion Guards: 5 rows, so the tab stays at 6 columns of 5 rows (decision 19).
Fixed inside the core: Agree 0.15 (the references must agree within it), and the repair zone (full weight up to
max(1, h / 540) px, then a fade of the same width), and the hysteresis growth (at most max(2, 16 h / 1080)
steps, threshold 0.02). All follow the rendered frame's height, so a proxy render makes the same decisions.

## Soft rims: hysteresis growth

A Flame test showed a faint outline left where a repaired hallucination's soft rim had been. The flagged set
holds only pixels more than Jump Above (0.5) beyond the references; the blob's falloff (alpha 0.03-0.4) reached
beyond the 1 + 1 px repair zone. So after the decisions (they do not change), each repaired blob grows from its
flagged pixels, 8-connected, at most 16 steps at 1080 lines (8 at 540, 32 at 2160), into every pixel where:
- the references agree within Agree (a moving edge stops it), and
- frame t lies beyond their range in the blob's direction by more than 0.02 (the level a rim stops showing).

It never grows into a kept blob. The repair value and the fade then apply to the grown blob as before.
Why 16 px at 1080: the real rim reached 8 px beyond the repaired area on a 562-line crop (the same at 1080 is
about 15 px). Why 0.02: the outline counted above 0.02; a higher threshold left part of it. A fixed threshold
rather than a share of Jump Above: lowering Jump Above should not make the growth chase noise.

## Chain position: first

matte -> **Jump Repair** -> Temporal -> Smart Fill -> Edge Refine. Reasons:
- Temporal's median over +-K frames sees a one-frame blob as one vote in three and only partly removes
  it, and its hole rule can fill or keep it by majority; Edge Refine re-solves a band around a
  hallucinated blob and gives it a convincing edge. Removing the jump first stops both from amplifying it.
- It compares raw input mattes with raw input mattes (the neighbours are the host's frames).
- Off: nothing is fetched, nothing is computed, the chain is bit for bit the one before (tested in
  `test_render_chain` and `ml_systest`).

Neighbours: `getFramesNeeded` asks for t +- K with K = max(Temporal's K, Span + 1). The guards look one
frame beyond the references, which is why the default needs t +- 2. Temporal and Jump Repair share one fetch per
neighbour. Frames beyond Temporal's K are fetched without the picture (matte only), and Temporal ignores them.

## Why the guards (the only hard part)

A matte alone cannot always tell a hallucination from real fast motion. Each guard keeps a blob when the
motion explanation is likely (refine, never replace). Kept blobs show in red in View Jumps and are logged at
DEBUG with their numbers. The node's Motion Guards checkbox switches all three together (the per-pixel
reference confirmation stays on): off repairs every blob found, for shots where the user knows the red ones are
hallucinations.
- **Reference confirmation (per pixel):** the references must agree with the frames beyond them. Without
  it, a jump in t - 1 next to a moving edge made frame t look like a jump of the opposite kind (found on the
  synthetic clip: a false "removed" repair one frame after an added jump).
- **Moved (pass-through):** an object moving more than its own width per frame is flagged whole. A real
  one is found in t - 1 shifted by about -v and in t + 1 by about +v. A blob displaced the same way
  against both neighbours is a jump. An earlier version compared areas in a window and failed on real
  footage, where the edge motion nearby dominated the sums.
- **Turning:** the tip of an edge at the end of a back-and-forth motion overshoots both neighbours. It
  is kept when the matte around it grows towards t before the references and shrinks after them.
- **Not still (spans of 2-3 frames):** an arm swinging with a period of about 8 frames looks exactly like a
  static 2-3 frame blob. Longer spans are repaired only where the matte around them does not move at all.
  This is why Span defaults to 1 frame.

## Measurements

Synthetic (`synth::makeJumpSequence`, 20 frames: moving figure, static box, fast disc, swinging bar, 5
injected jumps of 1, 1, 1, 2 and 3 frames, one of them straddling the moving edge):
- All jump frames are repaired at the matching Span (3 / 5 / 8 of 8 for Span 1 / 2 / 3). Mean error left in a
  jump is under 0.001; the max is 0.20 (one feathered rim pixel).
- Nothing changed away from the jumps. The fast disc (18 frames) and the bar's 4 turning frames were
  kept, at 480x270, 960x540 and 1920x1080 with the same decisions; changed pixels scale with the area (x3.9 per step).
- With 1.5 px per-frame edge jitter, nothing extra changed.
- Soft-rim jumps (Gaussian falloff 2, 4, 6, 8 px at 960x540, Span 3): rim left above 0.02 without the growth 24,
  814, 1638, 2666 px; with it 0 in each case. The same blobs are repaired and kept (8 / 46). Nothing changed
  away from the jumps, no legit pixel was touched, and no subject pixel was lowered, including next to the
  moving figure's edge.
- Default sequence (2 px edges), Span 1: 919 px changed (873 before the growth took in the 1 px soft edge).
- Motion Guards off, Span 1: the same 25 blobs, all repaired (6015 px changed against 873 with the guards on). The fast
  disc and the bar's turning frames are then changed (4690 px): this is what the guards are for.
- Timing at 2880x1620 (one jump, the disc, the bar, 2 threads on the test machine): Span 1 about 20 ms, Span 3 about 40 ms.
  The blob step (the shift search) dominates. It is bounded per blob (at most 512 pixels sampled, 65 x 65 shifts).

Real matte sequence (97 frames, 424x562, alpha only, numbers only), defaults:
- 95 frames judged (the first and last cannot be).
- 14 blobs at or above min size in 14 frames. 2 repaired: a 62 px added blob (258 px changed with the growth,
  119 before it) and a 308 px added blob (1057 px changed, 464 before; 0.44 % of the frame). The rim left near it
  (both neighbours below 0.02, within 8 px of the repaired area): 545 px above 0.02 and 468 above 0.05 before the
  growth, 0 and 0 after. All other frames are bit-identical to the version without growth. 12 kept: 9 as Moved (consistent -v / +v shifts of
  4-20 px), 3 as Turning. No other frame changed. Mean 1.4 ms per frame, at most about 15 ms.
- Motion Guards off: 14 blobs repaired in 14 frames, 4164 px changed with the growth (2323 before it; on: 2 repaired,
  1315 px, 583 before it). Span 1-3: 4 repaired, 151 kept, at most about 12 ms per frame.
- A simple detector (a pixel outside the t +- 2 range, at least 30 px) flags 24 frames. Of its
  strongest, Jump Repair repairs 2 and keeps 10 as motion. One is no longer flagged after the reference
  confirmation. Which of the kept ones are real motion cannot be decided from the mattes; View Jumps in Flame decides.

## Limits

- Real motion that a matte cannot tell from a hallucination: an object appearing for one frame (a thrown
  object, a flash), or a fast object that changes shape between frames. The guards keep what looks like
  motion. What they miss shows in View Jumps, and Enable can be keyframed off for such frames.
- Pixels that a moving edge crosses within t +- 2 are not repaired (the references there disagree or are not
  confirmed), so a jump straddling a fast edge is repaired only outside the edge's motion band.
- A hallucination that drifts like an object at constant speed over 3+ frames is kept as Moved.
- Spans of 2-3 frames are repaired only on still surroundings.
- The growth takes in any connected pixel beyond the agreeing references by more than 0.02 within 16 px (at 1080),
  so faint real matte that sits exactly on a hallucination's rim and is missing from both neighbours goes too. A rim
  wider than the step limit, or one cut off by a moving edge, is left in part (the fade still softens it).
- The first and last frames of a clip, and any frame whose t - 1 or t + 1 the host does not return, pass
  through (footer and log say so).
- UNVERIFIED in Flame: getFramesNeeded honoured, fetchImage(t +- k) inside render, the fetch cost per neighbour,
  and cache invalidation when a neighbour frame changes (the same open points as Temporal).
