#include "ops/region_trimap.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ops/morphology.h"

namespace mr {

void regionTrimap(const float* base, const float* region, int w, int h, const RegionTrimapParams& p, float* out) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  if (n == 0) return;
  Mask band;
  if (p.bandPx > 0) {
    Mask line(n);
    for (size_t i = 0; i < n; ++i) line[i] = region[i] >= 0.5f;
    band = dilate(line, w, h, p.bandPx);
  }
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      const float label = base[i] >= p.fgThreshold ? 1.0f : 0.0f;
      float u = region[i] > 0.0f ? std::min(region[i], 1.0f) : 0.0f;  // NaN -> 0
      if (!band.empty() && band[i]) u = 1.0f;
      out[i] = u <= 0.0f ? label : u >= 1.0f ? 0.5f : label + u * (0.5f - label);
    }
  });
}

}  // namespace mr
