// Operation 1, Fill Holes (CLAUDE.md decisions 7-10): fills holes enclosed by solid matte with
// 1.0. Never lowers a pixel and never touches background connected to an open frame edge.
#pragma once

namespace mr {

struct FillHolesParams {
  float threshold = 0.5f;       // matte >= threshold is solid
  double maxHolePercent = 1.0;  // holes larger than this % of the frame stay open; 0 = no limit
  double bridgePx = 0.0;        // also fill holes whose solid edge has a gap up to this wide
  double rimPx = 1.0;           // grow the fill this far into the solid's soft inner rim
  bool closedLeft = false, closedRight = false, closedTop = false, closedBottom = false;
};

struct FillHolesStats {
  int holesFilled = 0;
  int holesTooLarge = 0;
  long long pixelsChanged = 0;
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void fillHoles(const float* in, float* out, int w, int h, const FillHolesParams& p,
               FillHolesStats* stats = nullptr);

}  // namespace mr
