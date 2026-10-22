// Sub-pixel edge geometry shared by the edge operations (decision 16, generalised to both sides
// of the edge): the signed distance to a matte's contour and resampling along the edge normal,
// which moves an edge while keeping its own softness profile.
#pragma once

#include <vector>

#include "ops/morphology.h"

namespace mr {

// Where the `level` contour runs near pixel i (side = in < level): per axis, t = where level is
// crossed towards the nearest 4-neighbour on the other side (linear interpolation, 0-1); the
// contour is the line through those crossings: normal ~ (1/tx, 1/ty), distance
// s = 1 / sqrt(sum 1 / t^2) (decision 16). All zero if no neighbour is on the other side.
ContourFoot contourFoot(const float* in, const Mask& side, int w, int h, float level, int i);

// Signed distance of each pixel to the matte's `level` contour: > 0 inside (in >= level), < 0
// outside (morphology.h distanceWithin with contourFoot). Exact where |d| <= band; clamped to +-(band + 1)
// beyond. Pixels beyond the frame are neither inside nor outside: the frame border is no edge.
std::vector<float> signedEdgeDistance(const float* in, int w, int h, float level, double band);

// Bilinear sample of a w*h image at (x, y), pixel centres at integers, clamped to the frame.
float sampleBilinear(const float* img, int w, int h, double x, double y);

// out(p) = in(p + shift(p) * n(p)), n = the unit gradient of sd (pointing inward): shift > 0
// moves the edge outward (grows), < 0 inward, and the profile across the edge travels with it.
// Where sd has no usable gradient (beyond its band, |grad| < 0.5) or shift is 0, out = in.
// out may not be in.
void shiftAlongNormal(const float* in, const std::vector<float>& sd,
                      const std::vector<float>& shift, int w, int h, float* out);

}  // namespace mr
