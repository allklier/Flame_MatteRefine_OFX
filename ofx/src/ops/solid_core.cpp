#include "ops/solid_core.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "ops/morphology.h"

namespace mr {

void solidCore(const float* in, float* out, int w, int h, const SolidCoreParams& p,
               SolidCoreStats* stats) {
  SolidCoreStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  // Near the edge = within Depth of a pixel below Edge At. Pixels beyond the frame don't
  // count: a subject cut by the frame has no edge there.
  Mask below(n);
  for (size_t i = 0; i < n; ++i) below[i] = in[i] < p.edgeAt;
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
  // crosses a grey area. D = the distance to the Edge At contour, refined below pixel spacing:
  // on a below pixel next to an above one, the contour lies s px beyond its centre, s from
  // linear interpolation towards the above neighbours (per axis t; s = 1 / sqrt(sum 1 / t^2)).
  auto offset = [&](int i) {
    const int x = i % w, y = i / w;
    double inv = 0.0;
    for (int axis = 0; axis < 2; ++axis) {
      double t = 2.0;
      for (int sg = -1; sg <= 1; sg += 2) {
        const int nx = x + (axis ? 0 : sg), ny = y + (axis ? sg : 0);
        if (nx < 0 || ny < 0 || nx >= w || ny >= h || below[size_t(ny) * w + nx]) continue;
        const double a = in[i], b = in[size_t(ny) * w + nx];
        t = std::min(t, (p.edgeAt - a) / (b - a));
      }
      if (t <= 1.0) inv += 1.0 / (t * t);
    }
    return float(1.0 / std::sqrt(inv));  // inv > 0: i has an above 4-neighbour
  };
  const std::vector<float> d = distanceWithin(below, w, h, p.depthPx + p.featherPx, offset);
  for (size_t i = 0; i < n; ++i) {
    if (d[i] <= p.depthPx || in[i] >= 1.0f) {  // the edge band, or already solid
      out[i] = in[i];
      continue;
    }
    // smoothstep from Depth to Depth + Feather.
    const double t = std::min(1.0, (d[i] - p.depthPx) / p.featherPx);
    const float o = std::max(in[i], float(t * t * (3.0 - 2.0 * t)));
    st.pixelsChanged += o != in[i];
    out[i] = o;
  }
  if (stats) *stats = st;
}

}  // namespace mr
