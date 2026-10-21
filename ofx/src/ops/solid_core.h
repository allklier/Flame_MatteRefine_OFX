// Operation 2, Solid Core (CLAUDE.md decision 15): every pixel more than Depth px inside the
// matte's Edge At contour becomes 1.0; the outer Depth px of edge are not touched.
#pragma once

namespace mr {

struct SolidCoreParams {
  double depthPx = 6.0;  // core starts this far inside the edge
  float edgeAt = 0.5f;   // the contour the depth is measured from
};

struct SolidCoreStats {
  long long pixelsChanged = 0;
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void solidCore(const float* in, float* out, int w, int h, const SolidCoreParams& p,
               SolidCoreStats* stats = nullptr);

}  // namespace mr
