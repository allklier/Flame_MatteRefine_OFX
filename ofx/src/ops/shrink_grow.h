// Operation A3, Shrink / Grow (CLAUDE.md "Plugin 1 operations"): moves the matte's 0.5 contour
// outward (grow) or inward (shrink) by Amount px, sub-pixel, while each part of the edge keeps
// its own falloff profile (a 1 px edge stays 1 px, an 8 px edge stays 8 px). Morphological, as
// moving every contour by Amount means: a gap or hole narrower than 2 x Amount closes on grow, a
// part narrower than 2 x Amount disappears on shrink. Grow never lowers a pixel, shrink never
// raises one. Strands (parts of the >= 0.5 area under ~3 px wide) are detail and keep their
// shape; strands below 0.5 have no contour of their own: away from an edge they are not touched.
#pragma once

#include <functional>
#include <memory>

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

// Edge helpers shared with Edge Width (ops/edge_width); candidates for ops/edge_field.

// fn(begin, end) over ranges of [0, n) on up to 32 threads, at least minPerThread items each.
void parallelFor(int n, int minPerThread, const std::function<void(int, int)>& fn);

// edge_field.h signedEdgeDistance, the same values, but threaded and without distanceWithin's
// switch to the plain grid distance at 2 frames of writes: at the bands these ops need (up to
// 15 px) a 1080 frame with hair already crosses it, and the edge would move by whole pixels.
// Each band of rows stamps the contour pixels within reach into its own rows. Only when the
// stamping would cost more than 12 frames of writes (a very noisy matte), signedEdgeDistance.
// w*h floats, allocated without the zero fill a std::vector does (~8 ms single-threaded at UHD).
std::unique_ptr<float[]> signedEdgeDistanceThreaded(const float* in, int w, int h, float level,
                                                    double band);

// The unit normal of sd at pixel (x, y), pointing inward (up the distance): central differences,
// or where they cancel (the middle of a thin part or gap) the steeper one-sided difference per
// axis. False where sd is flat (beyond its band).
bool edgeNormal(const float* sd, int w, int h, int x, int y, double& nx, double& ny);

// The matte's value at signed distance `target` along the normal through (x, y), read near
// (x, y) + (target - sd) * n as edge_field.h shiftAlongNormal does: of the pixels there, the two
// whose sd brackets target most tightly, interpolated linearly; beyond the profile, its last
// value. Two changes: the window is 3 px across the normal's main axis and 7 along the other
// (not 3x3), and a read past the middle of a thin part stops at the middle (its peak).
float profileAt(const float* in, const float* sd, int w, int h, int x, int y,
                double nx, double ny, double target);

// A pixel of the 0.5 contour: a 4-neighbour on the other side of 0.5.
bool isContour(const float* in, int w, int h, int x, int y);

// The profile's slope across the contour at contour pixel (x, y), per px of sd: from the
// 4-neighbour across 0.5 farthest in sd (exact on a linear ramp; on a hard step, the step
// itself). 0 if there is none.
double contourSlope(const float* in, const float* sd, int w, int h, int x, int y);

// The tent-weighted (1.5 px) mean of per-contour-pixel measures around (fx, fy) (a pixel's foot
// on the contour), so they vary smoothly along the edge: m holds k floats per pixel, read only
// at contour pixels of in; a first float < 0 means no measure. Returns the weight sum (0 = no
// measure nearby); mean gets k values.
double footMean(const float* in, const float* m, int k, int w, int h, double fx, double fy, double* mean);

}  // namespace mr
