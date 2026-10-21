#include "ops/solid_core.h"

#include <algorithm>

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
  const Mask nearEdge = dilate(below, w, h, p.depthPx);
  for (size_t i = 0; i < n; ++i) {
    const bool lift = !nearEdge[i] && in[i] < 1.0f;
    st.pixelsChanged += lift;
    out[i] = lift ? 1.0f : in[i];
  }
  if (stats) *stats = st;
}

}  // namespace mr
