// Operation 2, Solid Core (CLAUDE.md decisions 15, 16): every pixel more than Depth + Feather px
// inside the matte's Edge At contour becomes 1.0, with a smooth ramp over the Feather px before
// it; the outer Depth px of edge are not touched and no pixel gets darker.
#pragma once

namespace mr {

struct SolidCoreParams {
  double depthPx = 6.0;  // core starts this far inside the edge
  float edgeAt = 0.5f;   // the contour the depth is measured from
  double featherPx = 2.0;  // ramp from the input to 1.0 over this many px; 0 = a hard core
};

struct SolidCoreStats {
  long long pixelsChanged = 0;
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void solidCore(const float* in, float* out, int w, int h, const SolidCoreParams& p,
               SolidCoreStats* stats = nullptr);

}  // namespace mr
