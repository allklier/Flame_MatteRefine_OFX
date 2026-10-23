// Operation A3, Shrink / Grow (CLAUDE.md "Plugin 1 operations"): moves the matte's 0.5 contour
// outward (grow) or inward (shrink) by Amount px, sub-pixel, while each part of the edge keeps
// its own falloff profile (a 1 px edge stays 1 px, an 8 px edge stays 8 px). Morphological, as
// moving every contour by Amount means: a gap or hole narrower than 2 x Amount closes on grow, a
// part narrower than 2 x Amount disappears on shrink. Grow never lowers a pixel, shrink never
// raises one. Strands (parts of the >= 0.5 area under ~3 px wide) are detail and keep their
// shape; strands below 0.5 have no contour of their own: away from an edge they are not touched.
#pragma once

namespace mr {

struct ShrinkGrowParams {
  double amountPx = 0.0;  // > 0 grows, < 0 shrinks; 0 = no change
};

struct ShrinkGrowStats {
  long long pixelsChanged = 0;
  int passes = 0;  // large amounts move in passes of up to 4 px
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void shrinkGrow(const float* in, float* out, int w, int h, const ShrinkGrowParams& p,
                ShrinkGrowStats* stats = nullptr);

}  // namespace mr
