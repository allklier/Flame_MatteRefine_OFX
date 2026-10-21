// Binary morphology shared by the operations (decision 9 "Implementation").
#include "ops/morphology.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <utility>

namespace mr {

namespace {

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

}  // namespace

// The nearest set pixel of any other pixel is a boundary pixel (one with an unset 4-neighbour),
// so stamping a disc on each boundary pixel is exact; when that costs more than a frame's worth
// of writes (a large r on a noisy matte), the distance transform does it instead.
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

}  // namespace mr
