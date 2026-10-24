// Matte Merge, the classical core (ofx/merge/CLAUDE.md, decision 37): a base matte (right about the
// subject and its body edge) plus a detail matte (better fine edge detail) in, one combined matte
// out. Pure C++14 on w*h float mattes (0-1, row 0 = top), no OFX types, as plugin 1's ops.
//
// 1. Zone: Matte edge = within Reach px of the base's grey pixels (0.02-0.98) or its 0.5 contour,
//    minus its solid core (>= 0.98 and more than max(2, Feather) px from a non-solid pixel).
//    Trimap = the base's unknown pixels (0.02-0.98; known = near 0 / near 1). The output is the
//    base, bit for bit, outside the zone.
// 2. Align (Auto): the detail's offset and small scale, estimated on the picture (merge_picreg.h:
//    the detail's thin structures against the picture's local structure, started from the matte
//    estimate of merge_register.h), with fallbacks when that estimate is unreliable (AlignModel);
//    the detail is resampled (bilinear). Off: used as given. The detail is clamped to 0-1 first
//    (non-finite: 0).
// 3. Connected: detail pixels above 0.05 in the zone count only when connected (8-neighbours, within
//    the zone) to the base: zone pixels with base >= 0.5 (Matte edge) or next to the known
//    foreground (Trimap). Floating blobs, other objects and loose strands are dropped (Matte edge:
//    the base stays; Trimap: background, or the detail as it is with keepDisconnected).
// 4. Combine, weighted by the zone's Feather ramp, Strength and (optional) picture evidence:
//    Add fine detail: the detail's thin structures (white top-hat: detail minus its opening at
//    Detail Width px) screened over the base, base + f (1 - base): never lowers the base, thick
//    parts (the detail's own body edge) drop out. Bridge (Add with Matte edge, bridgePx > 0): the
//    detail's thick parts next to the base (its body edge, the dense hair roots) are lost that way
//    and the coarse base lacks them, so strands looked detached (a dark rim). Near the base's edge
//    the detail is therefore also taken as it is: f_bridge = b(d) max(0, detail - base) on kept
//    (connected) pixels, d = the distance (px) from the base's 0.5 line, on either side, taken as
//    the distance to the nearest 0.5-contour pixel (isContour) + 0.5, b(d) = 1 - smoothstep(d /
//    bridgePx) (1 at the line, 0 from bridgePx on), times a strand mask: 1 within bridgePx of a strand
//    (kept top-hat > 0.25), 0 from 2 bridgePx (smoothstep), so a hairless edge is never bridged.
//    Combined with the thin part by screen, x = 1 - (1 - fine)(1 - f_bridge); then, as without the
//    bridge, the zone ramp x Strength weight and then the gain curve: f = boost(ramp * strength * x),
//    out = base + f (1 - base). Still never lowers the base; where f_bridge = 0 the old expression is
//    used unchanged (bridge 0 = bit-identical). Evidence judges only the thin part, not the bridge
//    (a thick smooth area has no structure to compare). Zone blend: base + wt (detail - base): the detail
//    wins inside the zone (it may lower the base there). Trimap source (either method): the unknown
//    zone is filled from the detail; where the detail is not taken (weight < 1) the fill falls back
//    to a ramp from the known foreground (1) to the known background (0), the "prior".
// 5. Evidence (On, picture given): only the detail's thin structures (the top-hat above) are judged,
//    each kept in proportion to how well the picture shows the same structure (merge_evidence.h; no
//    colourspace maths). In a blend or trimap fill an unsupported thin structure falls back to the
//    detail's opening (its thick part) there; a thick or smooth change is never judged (it has no
//    structure to compare, and judging it rejected whole trimap fills in the first version).
#pragma once

#include <cstdint>
#include <vector>

#include "merge_register.h"

namespace mr {
namespace merge {

enum class Source { MatteEdge, Trimap };
enum class Method { AddFineDetail, ZoneBlend };

// The sizes in px; the plugin sets them from fractions of the frame height (HeightSizes,
// scaleToHeight). The px defaults are the HeightSizes defaults at 1080 lines.
struct MergeParams {
  Source source = Source::MatteEdge;
  double reachPx = 64.8;         // Matte edge: how far beyond the base's edge the detail may act
  Method method = Method::AddFineDetail;
  double detailWidthPx = 5.4;    // structures narrower than this are "fine detail" (opening 2r+1 >= it)
  double strength = 1.0;         // 0-1: how much of the detail is taken; 1-2: also a gain on what is added, a curve that lifts weak values most
  double featherPx = 2.7;        // ramp at the zone's border (0: hard)
  bool align = true;             // Align Auto (true) / Off
  bool evidence = false;         // Picture Evidence On / Off (needs the picture)
  // Trimap only: detail not connected to the known foreground is kept as the detail matte has it
  // (true; floating blobs and invented strands then stay too) instead of becoming background
  // (false, the default). Matte edge ignores it: unconnected detail never changes the base.
  bool keepDisconnected = false;
  // Add fine detail with Matte edge only: within this many px of the base's 0.5 line (either
  // side) the aligned detail is also taken as it is, not only its thin parts, where it is connected
  // and above the base, so strands join the head (step 4). 0 = off (the output is then bit for bit
  // that of the code before the bridge).
  double bridgePx = 5.4;
};

// Reach, Detail Width and Feather as percentages of the frame height (user decision 2026-10-24):
// the plugin's controls, so a setting means the same at any resolution.
struct HeightSizes {
  double reachPct = 6.0;
  double detailWidthPct = 0.5;
  double featherPct = 0.25;
  double bridgePct = 0.5;
};
// Sets p's reachPx, detailWidthPx, featherPx and bridgePx for a frame h px high (pct / 100 * h).
void scaleToHeight(MergeParams& p, const HeightSizes& s, int h);

// QC buffers (w*h each), each optional (nullptr = not computed).
struct MergeViews {
  float* added = nullptr;     // out - base (what the detail changed)
  float* rejected = nullptr;  // |result taking every detail pixel - out|: dropped as unconnected or unsupported
  float* zone = nullptr;      // the zone with its feather ramp, 0-1
  float* residual = nullptr;  // |opening(aligned detail) - opening(base)| in the zone: registration QC
};

// Which move was applied to the detail (MergeInfo::align), for the plugin's QC.
enum class AlignModel {
  Off,                 // Align Off
  None,                // no usable estimate: the detail is used as given
  PictureFull,         // picture: translation + scale (the normal case)
  PictureTranslation,  // picture: translation only (translation + scale was unreliable)
  MatteFull,           // the mattes' thick parts: translation + scale (the picture gave no usable detail)
};
const char* alignModelName(AlignModel m);

struct AlignInfo {
  AlignModel model = AlignModel::Off;
  // The reliability of the applied move: its estimated 1-sigma displacement error over the frame
  // (px). Picture models: from the scatter of the block shifts (merge_picreg.h); matte models:
  // sensitivity x RMS edge residual, a pessimistic bound (systematic edge offsets dominate); 0 for
  // Off / None.
  double uncertaintyPx = 0;
  // Picture models: the fraction of blocks with thin detail that agree with the move (0-1; low on
  // a busy background or with many invented strands); matte models and Off / None: 0.
  double agreement = 0;
  int blocks = 0, inliers = 0;   // picture blocks with thin detail / agreeing
  Registration matte;            // the matte-only estimate (the start, and the fallback)
};

struct MergeInfo {
  Registration reg;           // the applied move (identity when Align is Off or no estimate)
  AlignInfo align;
  long zonePixels = 0;
  long keptPixels = 0;        // zone pixels above 0.05 in the detail and connected
  long droppedPixels = 0;     // zone pixels above 0.05 in the detail and not connected
};

// Buffers kept between calls (one per plugin instance; not shared between concurrent calls): frame-
// sized scratch is reused instead of allocated and page-faulted on every render. Grows to the
// largest frame seen; contents are not meaningful between calls.
struct MergeWorkspace {
  std::vector<uint8_t> zone, mask, kept;
  std::vector<float> d2, d2b, ramp, prior, da, warped, opened, fine, ev, sig, morph, bridge;
  std::vector<int> stack;
};

// base, detail: w*h mattes; picture: w*h*3 RGB, required by the plugin (registration and
// evidence use it; nullptr only in tests: registration then on the mattes alone, evidence off).
// out: w*h, may not alias the inputs. ws: optional reusable buffers. Threads: one mr::Workers pool
// for the call.
void merge(const float* base, const float* detail, const float* picture, int w, int h, const MergeParams& p,
           float* out, const MergeViews& views = MergeViews(), MergeInfo* info = nullptr,
           MergeWorkspace* ws = nullptr);

}  // namespace merge
}  // namespace mr
