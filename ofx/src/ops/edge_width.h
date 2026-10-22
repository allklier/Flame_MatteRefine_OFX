// Operation A5, Edge Width (CLAUDE.md "Plugin 1 operations"): gives the matte's edge one
// falloff width along the whole contour. Hard-cut (aliased) stretches become soft, overly soft
// ones tighter; the 0.5 contour stays where it is, and the levels on either side of the edge
// (a grey face behind it, hair beyond it) are kept: only the transition between them is
// retimed, into a linear ramp. Width = the ramp's full 0-100 % width (10-90 % = 0.8 x Width).
// Parts and gaps narrower than the target width keep their own profile (hair is not smeared or
// erased); an edge already within ~3 % of the target is not touched. A hard cut (or 1 px edge)
// gets the target ramp drawn from its levels, on the distance averaged along the edge over
// +-Width / 2 (up to 4 px): it follows the line, not the pixel staircase, except within ~4 px of
// the steps at Width 2 (residual up to ~0.24); a full de-aliasing is Contour Smooth's job.
#pragma once

namespace mr {

struct EdgeWidthParams {
  double widthPx = 2.0;  // target 0-100 % width of the edge; clamped to 0.5-16 px
  double amount = 1.0;   // 0 = the input, 1 = the target width; clamped to 0-1
};

struct EdgeWidthStats {
  long long pixelsChanged = 0;
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void edgeWidth(const float* in, float* out, int w, int h, const EdgeWidthParams& p,
               EdgeWidthStats* stats = nullptr);

}  // namespace mr
