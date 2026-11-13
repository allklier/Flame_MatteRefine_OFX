// Region trimap (Focus Region, docs/focus_region.md): a trimap for a matting model (Edge Refine ML's
// "unknown" zone) from the base matte and a user-drawn soft region. Pure C++14, no OFX types, as the
// other ops (decision 3); shared (mr_ops) so plugin 2 and Matte Merge can both use it. Not wired yet.
//
// Values: 0 = background, 1 = foreground, 0.5 = unknown.
//   known label L = 1 where base >= fgThreshold (the solid core), else 0 (background elsewhere: a grey
//                   base pixel outside the region's influence is background, by design);
//   u = the region weight (0-1; NaN / outside 0-1 clamped), raised to 1 within bandPx of the region's
//       0.5 line (pixels with weight >= 0.5) when bandPx > 0;
//   out = L + u (0.5 - L): u 0 -> L exactly, u 1 -> 0.5 exactly, a soft region gives intermediate
//         values (between the known label and unknown).
// bandPx is in pixels of the image as delivered: the caller scales it with the render scale
// (decision 6), as every px param.
#pragma once

namespace mr {

struct RegionTrimapParams {
  float fgThreshold = 0.95f;  // base >= this = foreground (outside the region's influence)
  double bandPx = 0.0;        // unknown also within this many px of the region's 0.5 line; 0 = off
};

// base, region: w*h floats (row 0 = top); out: w*h, may alias base (not region).
void regionTrimap(const float* base, const float* region, int w, int h, const RegionTrimapParams& p, float* out);

}  // namespace mr
