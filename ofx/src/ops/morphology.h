// Binary morphology shared by the operations: masks are w*h bytes, row 0 = the top row.
#pragma once

#include <cstdint>
#include <vector>

namespace mr {

using Mask = std::vector<uint8_t>;

// Pixels within Euclidean distance r of a set pixel of m (m included). Pixels outside the
// frame count as unset.
Mask dilate(const Mask& m, int w, int h, double r);

}  // namespace mr
