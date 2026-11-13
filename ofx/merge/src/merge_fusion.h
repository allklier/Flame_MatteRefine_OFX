// Focus Region and candidate fusion for Matte Merge (docs/focus_region.md). Pure C++14 on w*h float
// maps (row 0 = top), no OFX types.
//
// Focus Region: a user-drawn soft roto, the ONLY place a fusion step may change the matte. Its weight
// (prepareRegion) replaces Matte Merge's automatic zone (matte_merge.h FocusInput); the output is
// base + weight (merged - base): bit for bit the base where the weight is 0, never below the base
// (Matte edge source).
//
// Candidates: proposals for the matte inside the region, each an alpha map with a per-pixel
// confidence (0-1). (a) the external Detail matte, aligned, its confidence from the picture evidence
// on its thin structures (1 on thick parts and with Evidence off). (b) EMPTY SLOT: an internal
// thin-line candidate traced from the picture (research pending); it would be passed as an extra
// Candidate in the picture's pixel grid (not aligned). fuseCandidates combines them per pixel with a
// FusionRule; the fused map then goes through the existing merge steps (connectivity to the base,
// thin structures, Bridge with its strand mask) as the detail matte did, and the region weight.
//
// The default rule (defaultFusionRule), per pixel, on the two candidates with the highest gated
// value g = alpha x confidence (both clamped to 0-1, NaN = 0):
//   one candidate: g;
//   agreement (|a1 - a2| <= agreeTol): max(a1, a2) x c, c = c1 + agreeBonus ((1 - (1 - c1)(1 - c2)) - c1),
//     the confidences combined as independent evidence (agreeBonus 1) or not at all (0); never below g1;
//   strong disagreement (|a1 - a2| > disagreeGap) with neither confidence >= highConf: g1 (1 - penalty);
//   otherwise g1 (the best gated value wins).
// The base is not a candidate: the final step never lowers it (merge with a FocusInput).
#pragma once

#include <vector>

#include "ops/snap_to_image.h"

namespace mr {
class Workers;
namespace merge {

struct Candidate {
  const char* name = "";              // for logs and QC ("detail", later "thin lines")
  const float* alpha = nullptr;       // w*h, 0-1
  const float* confidence = nullptr;  // w*h, 0-1; nullptr = 1 everywhere
};

struct FusionParams {
  float agreeTol = 0.1f;     // |alpha difference| at or below: the two candidates agree
  float agreeBonus = 1.0f;   // 0-1: share of the combined confidence given on agreement
  float disagreeGap = 0.4f;  // |alpha difference| above: strong disagreement ...
  float highConf = 0.7f;     // ... penalised unless one confidence reaches this
  float penalty = 0.5f;      // 0-1: the winner's gated value is scaled by (1 - penalty)
  // > 0: the OUTPUT value is alpha x min(1, confidence / confFull) instead of alpha x confidence (the choice
  // between candidates still uses g = alpha x confidence). 0 = the gated value (Matte Merge). Hair Locks sets
  // it so a confident strand keeps its alpha instead of coming out grey (agent hl96, frame 1094).
  float confFull = 0.0f;
};

// One pixel: n candidates' alphas and confidences (already clamped to 0-1) -> the fused value (0-1).
// Swap the rule by passing another function to fuseCandidates.
using FusionRule = float (*)(const float* alpha, const float* conf, int n, const FusionParams& p);
float defaultFusionRule(const float* alpha, const float* conf, int n, const FusionParams& p);

// fused (w*h) = rule per pixel over the n candidates (at most 8); n = 0: 0 everywhere.
void fuseCandidates(const Candidate* c, int n, int w, int h, const FusionParams& p, float* fused,
                    FusionRule rule = defaultFusionRule, Workers* wk = nullptr);

// The region weight from the Focus Region mask (w*h, Flame's mask clip; clamped to 0-1, NaN = 0).
// growPx > 0: grey dilation, < 0: grey erosion, with a (2r + 1) square, r = round(|growPx|) (window
// clipped to the frame). featherPx > 0: an inward ramp, weight x smoothstep((d - 0.5) / featherPx),
// d = the distance (px) to the nearest pixel with weight 0; pixels with weight 0 stay 0, so the
// feather never reaches outside the (grown) mask. Both 0: the clamped mask, bit for bit.
void prepareRegion(const float* mask, int w, int h, double growPx, double featherPx, float* weight,
                   Workers* wk = nullptr);

// Snap new pixels to the picture (optional stage after the merge): Snap to Image (plugin 1's op) on
// the merged matte with the weight strength x region, then (neverLower) max(base, result). Where the
// region is 0 the matte is untouched (snap_to_image.h weight). matte: in / out. picture: w*h*3.
void snapInRegion(const float* base, float* matte, const float* picture, const float* region, int w, int h,
                  double strength, const SnapParams& sp, bool neverLower);

// Snap to Image's sizes for Matte Merge: plugin 1's defaults (radius 8, band 12 px) taken as the
// values at 1080 lines and scaled to h (radius at least 1 px), like Merge's other sizes.
SnapParams snapParamsForHeight(int h);

}  // namespace merge
}  // namespace mr
