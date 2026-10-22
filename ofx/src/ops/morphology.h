// Binary morphology shared by the operations: masks are w*h bytes, row 0 = the top row.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace mr {

using Mask = std::vector<uint8_t>;

// Pixels within Euclidean distance r of a set pixel of m (m included). Pixels outside the
// frame count as unset.
Mask dilate(const Mask& m, int w, int h, double r);

// For each pixel p: min over set pixels q of |p - q| - offset(q), offset(q) in 0-1 (e.g. how far
// a contour lies beyond q's centre; called only for set pixels with an unset 4-neighbour, the
// others count with 0). Exact where it is at most r, larger than r elsewhere. When the exact
// search would cost more than a frame's worth of writes (a large r on a noisy matte), the plain
// distance (offsets ignored) instead. Pixels outside the frame count as unset.
std::vector<float> distanceWithin(const Mask& m, int w, int h, double r,
                                  const std::function<float(int)>& offset);

}  // namespace mr
