#include "ops/fill_holes.h"

#include "ops/morphology.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace mr {

namespace {

// Morphological closing of s by a disc of radius r: dilate, then erode (outside the frame
// counts as dilated, so the erosion doesn't eat in from the frame edges).
Mask closeMask(const Mask& s, int w, int h, double r) {
  Mask notDilated = dilate(s, w, h, r);
  for (auto& v : notDilated) v = !v;
  const Mask eroded = dilate(notDilated, w, h, r);  // the erosion's complement
  Mask c(s.size());
  for (size_t i = 0; i < s.size(); ++i) c[i] = s[i] || !eroded[i];
  return c;
}

}  // namespace

void fillHoles(const float* in, float* out, int w, int h, const FillHolesParams& p,
               FillHolesStats* stats) {
  FillHolesStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  if (n == 0) {
    if (stats) *stats = st;
    return;
  }

  // Decision 9, step by step.
  Mask solid(n);
  for (size_t i = 0; i < n; ++i) {
    solid[i] = in[i] >= p.threshold;
    st.belowThreshold += !solid[i];
  }
  const double r = p.bridgePx / 2.0;  // a closing of radius r bridges gaps up to 2r wide
  const Mask closed = r > 0.0 ? closeMask(solid, w, h, r) : solid;

  // Outside: 4-connected flood over non-closed pixels from the open frame edges (scanline
  // span fill: one stack entry per run of pixels, not per pixel).
  Mask outside(n);
  auto open = [&](size_t i) { return !closed[i] && !outside[i]; };
  std::vector<std::pair<int, int>> stack;
  auto seed = [&](int x, int y) {
    if (open(size_t(y) * w + x)) stack.emplace_back(x, y);
  };
  if (!p.closedTop)
    for (int x = 0; x < w; ++x) seed(x, 0);
  if (!p.closedBottom)
    for (int x = 0; x < w; ++x) seed(x, h - 1);
  if (!p.closedLeft)
    for (int y = 0; y < h; ++y) seed(0, y);
  if (!p.closedRight)
    for (int y = 0; y < h; ++y) seed(w - 1, y);
  while (!stack.empty()) {
    const int x = stack.back().first, y = stack.back().second;
    stack.pop_back();
    const size_t row = size_t(y) * w;
    if (!open(row + x)) continue;
    int l = x, r = x;
    while (l > 0 && open(row + l - 1)) --l;
    while (r + 1 < w && open(row + r + 1)) ++r;
    std::fill(&outside[row + l], &outside[row + r] + 1, uint8_t(1));
    st.outsidePixels += r - l + 1;
    for (const int ny : {y - 1, y + 1}) {
      if (ny < 0 || ny >= h) continue;
      const size_t nrow = size_t(ny) * w;
      for (int k = l; k <= r; ++k)  // one seed per run of open pixels next to the span
        if (open(nrow + k) && (k == l || !open(nrow + k - 1))) stack.emplace_back(k, ny);
    }
  }

  // Candidates: not solid and not outside, as 4-connected components.
  const double maxArea = p.maxHolePercent > 0.0 ? p.maxHolePercent / 100.0 * double(n) : 0.0;
  Mask fill(n), seen(n);
  std::vector<int> comp;
  for (size_t start = 0; start < n; ++start) {
    if (solid[start] || outside[start] || seen[start]) continue;
    comp.clear();
    comp.push_back(int(start));
    seen[start] = 1;
    bool hasCore = false, touchesOutside = false;
    for (size_t head = 0; head < comp.size(); ++head) {
      const int i = comp[head], x = i % w, y = i / w;
      hasCore |= !closed[i];
      const int nb[4] = {x > 0 ? i - 1 : -1, x + 1 < w ? i + 1 : -1, y > 0 ? i - w : -1,
                         y + 1 < h ? i + w : -1};
      for (int j : nb) {
        if (j < 0) continue;
        if (outside[j]) {
          touchesOutside = true;
        } else if (!solid[j] && !seen[j]) {
          seen[j] = 1;
          comp.push_back(j);
        }
      }
    }
    // All closed and touching the outside: an outer notch the closing filled, i.e. the edge.
    if (!hasCore && touchesOutside) {
      ++st.notchesKept;
      continue;
    }
    if (maxArea > 0.0 && double(comp.size()) > maxArea) {
      ++st.holesTooLarge;
      continue;
    }
    ++st.holesFilled;
    for (int i : comp) fill[i] = 1;
  }

  // Rim: grow the fill into the solid's soft inner rim (solid pixels only, so never into the
  // outside or into a hole left open).
  if (p.rimPx > 0.0 && st.holesFilled > 0) {
    const Mask grown = dilate(fill, w, h, p.rimPx);
    for (size_t i = 0; i < n; ++i) fill[i] = fill[i] || (grown[i] && solid[i]);
  }

  for (size_t i = 0; i < n; ++i) {
    const bool lift = fill[i] && in[i] < 1.0f;
    st.pixelsChanged += lift;
    out[i] = lift ? 1.0f : in[i];
  }
  if (stats) *stats = st;
}

}  // namespace mr
