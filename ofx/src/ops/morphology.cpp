// Binary morphology shared by the operations (decision 9 "Implementation").
#include "ops/morphology.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <thread>
#include <utility>

namespace mr {

namespace {

constexpr double kFar = 1e20;

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

// Disc offsets within r (centre excluded) and the boundary pixels of m (set, with an unset
// 4-neighbour): the nearest set pixel of any other pixel is a boundary pixel, so stamping the
// disc on each boundary pixel is exact.
void discAndBoundary(const Mask& m, int w, int h, double r, std::vector<std::pair<int, int>>& disc,
                     std::vector<int>& boundary) {
  const int ri = int(std::floor(r));
  for (int dy = -ri; dy <= ri; ++dy)
    for (int dx = -ri; dx <= ri; ++dx)
      if ((dx || dy) && dx * dx + dy * dy <= r * r) disc.emplace_back(dx, dy);
  // Rows in bands on threads, joined in order: the list stays sorted by index.
  std::vector<std::vector<int>> part(64);
  std::atomic<int> nextPart{0};
  parallelFor(h, 32, [&](int y0, int y1) {
    std::vector<int>& out = part[size_t(nextPart++)];
    out.push_back(y0);  // the band's first row, to sort the parts
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (m[i] && ((x > 0 && !m[i - 1]) || (x + 1 < w && !m[i + 1]) || (y > 0 && !m[i - w]) ||
                     (y + 1 < h && !m[i + w])))
          out.push_back(int(i));
      }
  });
  part.resize(size_t(nextPart));
  std::sort(part.begin(), part.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
    return a[0] < b[0];
  });
  for (const auto& p : part) boundary.insert(boundary.end(), p.begin() + 1, p.end());
}

}  // namespace

// When stamping costs more than a frame's worth of writes (a large r on a noisy matte), the
// distance transform does it instead.
Mask dilate(const Mask& m, int w, int h, double r) {
  std::vector<std::pair<int, int>> disc;
  std::vector<int> boundary;
  discAndBoundary(m, w, h, r, disc, boundary);
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

// Same two paths as dilate. A foot point lies within 1 px of q and its segment within ~1.12 px,
// so a pixel within r of the contour is within r + 1.5 of the q it is measured from; stamping
// from the boundary is exact as above. Squared distances, one sqrt per pixel at the end.
std::vector<float> distanceWithin(const Mask& m, int w, int h, double r,
                                  const std::function<ContourFoot(int)>& foot) {
  std::vector<std::pair<int, int>> disc;
  std::vector<int> boundary;
  discAndBoundary(m, w, h, r + 1.5, disc, boundary);
  std::vector<float> d(m.size());
  // Measured at UHD (BFD, 2026-10-22): stamping costs ~15 ms per frame's worth of writes, the
  // distance transform 25-45 ms, so they break even near 2 frames.
  if (double(boundary.size()) * disc.size() > 2.0 * double(m.size())) {
    d = squaredDistance(m, w, h);
    for (float& v : d) v = std::sqrt(v);
    return d;
  }
  std::vector<ContourFoot> feet(boundary.size());
  parallelFor(int(boundary.size()), 256, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b) feet[size_t(b)] = foot(boundary[size_t(b)]);
  });
  // Row bands on threads; each band stamps from the boundary pixels within reach of its rows
  // and writes only its own rows.
  const int reach = int(std::floor(r + 1.5));
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) d[i] = m[i] ? 0.0f : float(kFar);
    const auto first = std::lower_bound(boundary.begin(), boundary.end(), std::max(y0 - reach, 0) * w);
    const auto last = std::lower_bound(boundary.begin(), boundary.end(), std::min(y1 + reach, h) * w);
    for (auto it = first; it != last; ++it) {
      const int i = *it, x = i % w, y = i / w;
      const ContourFoot& f = feet[size_t(it - boundary.begin())];
      const bool line = f.nx != 0.0f || f.ny != 0.0f;
      for (const auto& o : disc) {
        const int px = x + o.first, py = y + o.second;
        if (px < 0 || py < y0 || px >= w || py >= y1) continue;
        const float ox = float(o.first), oy = float(o.second), grid = ox * ox + oy * oy;
        float d2 = grid;
        if (line) {
          const float vx = ox - f.fx, vy = oy - f.fy;
          const float perp = vx * f.nx + vy * f.ny;
          const float along = std::max(0.0f, std::fabs(vy * f.nx - vx * f.ny) - 0.5f);
          d2 = std::min(grid, perp * perp + along * along);
        }
        float& v = d[size_t(py) * w + px];
        v = std::min(v, d2);
      }
    }
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i)
      if (d[i] < float(kFar)) d[i] = std::sqrt(d[i]);
  });
  return d;
}

}  // namespace mr
