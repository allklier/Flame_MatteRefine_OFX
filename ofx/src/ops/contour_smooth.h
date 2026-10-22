// Operation A4, Contour Smooth (CLAUDE.md "Plugin 1 operations"): takes the wiggle and chatter
// off the matte's Edge At contour line while keeping the edge's softness profile across it and
// the real shape along it (corners, creases, thin features). Only the edge band changes.
#pragma once

namespace mr {

struct ContourSmoothParams {
  double smoothPx = 30.0;   // wiggle wavelengths up to this (px) are taken off the contour
  double maxShiftPx = 1.5;  // the edge moves at most this far (px); larger deviations are shape
  float edgeAt = 0.5f;      // the contour that is smoothed
  double cornerDeg = 20.0;  // contour turning more than this is a corner: not averaged across
};

struct ContourSmoothStats {
  long long pixelsChanged = 0;
  long long contourSamples = 0;  // boundary pixels the contour was fitted on
  double largestShiftPx = 0.0;   // largest |edge move| applied
  bool skipped = false;          // contour everywhere (a noisy matte): left as it was
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void contourSmooth(const float* in, float* out, int w, int h, const ContourSmoothParams& p,
                   ContourSmoothStats* stats = nullptr);

}  // namespace mr
