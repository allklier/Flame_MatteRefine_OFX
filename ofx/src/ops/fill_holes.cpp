#include "ops/fill_holes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

namespace mr {

namespace {

using Mask = std::vector<uint8_t>;
constexpr double kFar = 1e20;

// Runs fn(begin, end) over [0, n) on up to 32 threads, at least minPerThread items each.
template <class F>
void parallelFor(int n, int minPerThread, F fn) {
  const int t =
      std::max(1, std::min({int(std::thread::hardware_concurrency()), 32, n / minPerThread}));
  if (t == 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> pool;
  for (int i = 0; i < t; ++i) pool.emplace_back(fn, n * i / t, n * (i + 1) / t);
  for (auto& th : pool) th.join();
}

// 1-D squared distance transform of f (Felzenszwalb & Huttenlocher, "Distance Transforms of
// Sampled Functions", 2012). v, z: scratch of n and n + 1.
void edt1d(const double* f, double* d, int n, int* v, double* z) {
  int k = 0;
  v[0] = 0;
  z[0] = -kFar;
  z[1] = kFar;
  auto meet = [&](int q, int p) {
    return ((f[q] + double(q) * q) - (f[p] + double(p) * p)) / (2.0 * (q - p));
  };
  for (int q = 1; q < n; ++q) {
    // s > -kFar always (|f| <= kFar, q - p >= 1), so k never drops below 0.
    double s = meet(q, v[k]);
    while (s <= z[k]) s = meet(q, v[--k]);
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kFar;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    const double dq = q - v[k];
    d[q] = dq * dq + f[v[k]];
  }
}

// Squared Euclidean distance of every pixel to the nearest set pixel of m (>= kFar if none).
std::vector<float> squaredDistance(const Mask& m, int w, int h) {
  std::vector<float> g(size_t(w) * h);
  // Columns, kBlock at a time so the reads and writes walk along rows.
  constexpr int kBlock = 16;
  parallelFor((w + kBlock - 1) / kBlock, 2, [&](int b0, int b1) {
    std::vector<double> f(size_t(h) * kBlock), d(h), z(h + 1);
    std::vector<int> v(h);
    for (int b = b0; b < b1; ++b) {
      const int x0 = b * kBlock, nb = std::min(kBlock, w - x0);
      for (int y = 0; y < h; ++y)
        for (int k = 0; k < nb; ++k) f[size_t(k) * h + y] = m[size_t(y) * w + x0 + k] ? 0.0 : kFar;
      for (int k = 0; k < nb; ++k) {
        edt1d(&f[size_t(k) * h], d.data(), h, v.data(), z.data());
        std::copy(d.begin(), d.end(), &f[size_t(k) * h]);
      }
      for (int y = 0; y < h; ++y)
        for (int k = 0; k < nb; ++k) g[size_t(y) * w + x0 + k] = float(f[size_t(k) * h + y]);
    }
  });
  parallelFor(h, 32, [&](int y0, int y1) {  // rows, in place
    std::vector<double> f(w), d(w), z(w + 1);
    std::vector<int> v(w);
    for (int y = y0; y < y1; ++y) {
      float* row = &g[size_t(y) * w];
      std::copy(row, row + w, f.begin());
      edt1d(f.data(), d.data(), w, v.data(), z.data());
      for (int x = 0; x < w; ++x) row[x] = float(d[x]);
    }
  });
  return g;
}

// Pixels within Euclidean distance r of a set pixel of m (m included). The nearest set pixel
// of any other pixel is a boundary pixel (one with an unset 4-neighbour), so stamping a disc
// on each boundary pixel is exact; when that costs more than a frame's worth of writes (a
// large r on a noisy matte), the distance transform does it instead.
Mask dilate(const Mask& m, int w, int h, double r) {
  const int ri = int(std::floor(r));
  std::vector<std::pair<int, int>> disc;
  for (int dy = -ri; dy <= ri; ++dy)
    for (int dx = -ri; dx <= ri; ++dx)
      if ((dx || dy) && dx * dx + dy * dy <= r * r) disc.emplace_back(dx, dy);
  std::vector<int> boundary;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      if (m[i] && ((x > 0 && !m[i - 1]) || (x + 1 < w && !m[i + 1]) || (y > 0 && !m[i - w]) ||
                   (y + 1 < h && !m[i + w])))
        boundary.push_back(int(i));
    }
  Mask out = m;
  if (double(boundary.size()) * disc.size() > double(m.size())) {
    const std::vector<float> d = squaredDistance(m, w, h);
    const float r2 = float(r * r);
    for (size_t i = 0; i < m.size(); ++i) out[i] = d[i] <= r2;
    return out;
  }
  for (int i : boundary) {
    const int x = i % w, y = i / w;
    for (const auto& o : disc) {
      const int nx = x + o.first, ny = y + o.second;
      if (nx >= 0 && ny >= 0 && nx < w && ny < h) out[size_t(ny) * w + nx] = 1;
    }
  }
  return out;
}

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
  for (size_t i = 0; i < n; ++i) solid[i] = in[i] >= p.threshold;
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
    if (!hasCore && touchesOutside) continue;
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
