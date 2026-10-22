#include "ops/solid_core.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>

#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {

void solidCore(const float* in, float* out, int w, int h, const SolidCoreParams& p,
               SolidCoreStats* stats) {
  SolidCoreStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  // Near the edge = within Depth of a pixel below Edge At. Pixels beyond the frame don't
  // count: a subject cut by the frame has no edge there.
  Mask below(n);
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) below[i] = in[i] < p.edgeAt;
  });
  if (p.featherPx <= 0.0) {  // the hard core of decision 15
    const Mask nearEdge = dilate(below, w, h, p.depthPx);
    for (size_t i = 0; i < n; ++i) {
      const bool lift = !nearEdge[i] && in[i] < 1.0f;
      st.pixelsChanged += lift;
      out[i] = lift ? 1.0f : in[i];
    }
    if (stats) *stats = st;
    return;
  }
  // A soft core (decision 16): a binary one leaves a jagged step where its pixel staircase
  // crosses a grey area. D = the distance to the Edge At contour, refined below pixel spacing
  // (contourFoot in ops/edge_field).
  auto foot = [&](int i) { return contourFoot(in, below, w, h, p.edgeAt, i); };
  const std::vector<float> d = distanceWithin(below, w, h, p.depthPx + p.featherPx, foot);
  std::atomic<long long> changed{0};
  parallelFor(h, 16, [&](int y0, int y1) {
    long long c = 0;
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      if (d[i] <= p.depthPx || in[i] >= 1.0f) {  // the edge band, or already solid
        out[i] = in[i];
        continue;
      }
      // smoothstep from Depth to Depth + Feather.
      const double t = std::min(1.0, (d[i] - p.depthPx) / p.featherPx);
      const float o = std::max(in[i], float(t * t * (3.0 - 2.0 * t)));
      c += o != in[i];
      out[i] = o;
    }
    changed += c;
  });
  st.pixelsChanged = changed;
  if (stats) *stats = st;
}

}  // namespace mr
