// Binary morphology shared by the operations (decision 9 "Implementation").
#include "ops/morphology.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

namespace mr {

Workers::Workers() {
  const int t = std::max(1, std::min(int(std::thread::hardware_concurrency()), 32));
  for (int k = 1; k < t; ++k) threads_.emplace_back(&Workers::loop, this, k);
}

Workers::~Workers() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  wake_.notify_all();
  for (auto& th : threads_) th.join();
}

void Workers::run(int t, const std::function<void(int)>& job) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    job_ = &job;
    active_ = t;
    pending_ = t - 1;
    ++generation_;
  }
  wake_.notify_all();
  job(0);
  std::unique_lock<std::mutex> lock(mu_);
  done_.wait(lock, [&] { return pending_ == 0; });
}

// A thread joins a loop when it is one of its first t; run returns only when all of those have
// finished, so none can miss a loop or run one twice.
void Workers::loop(int k) {
  unsigned seen = 0;
  std::unique_lock<std::mutex> lock(mu_);
  for (;;) {
    wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
    if (stop_) return;
    seen = generation_;
    if (k >= active_) continue;
    const std::function<void(int)>* job = job_;
    lock.unlock();
    (*job)(k);
    lock.lock();
    if (--pending_ == 0) done_.notify_one();
  }
}

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

// Squared Euclidean distance of every pixel to the nearest set pixel of m (>= kFar if none),
// written to g (w*h).
void squaredDistance(const uint8_t* m, int w, int h, float* g, Workers* wk) {
  // Columns, kBlock at a time so the reads and writes walk along rows.
  constexpr int kBlock = 16;
  parallelFor(wk, (w + kBlock - 1) / kBlock, 2, [&](int b0, int b1) {
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
  parallelFor(wk, h, 32, [&](int y0, int y1) {  // rows, in place
    std::vector<double> f(w), d(w), z(w + 1);
    std::vector<int> v(w);
    for (int y = y0; y < y1; ++y) {
      float* row = &g[size_t(y) * w];
      std::copy(row, row + w, f.begin());
      edt1d(f.data(), d.data(), w, v.data(), z.data());
      for (int x = 0; x < w; ++x) row[x] = float(d[x]);
    }
  });
}

// Disc offsets within r (centre excluded) and the boundary pixels of m (set, with an unset
// 4-neighbour): the nearest set pixel of any other pixel is a boundary pixel, so stamping the
// disc on each boundary pixel is exact.
void discAndBoundary(const uint8_t* m, int w, int h, double r, std::vector<std::pair<int, int>>& disc,
                     std::vector<int>& boundary, Workers* wk) {
  const int ri = int(std::floor(r));
  for (int dy = -ri; dy <= ri; ++dy)
    for (int dx = -ri; dx <= ri; ++dx)
      if ((dx || dy) && dx * dx + dy * dy <= r * r) disc.emplace_back(dx, dy);
  // Rows in bands on threads, joined in order: the list stays sorted by index.
  std::vector<std::vector<int>> part(64);
  std::atomic<int> nextPart{0};
  parallelFor(wk, h, 32, [&](int y0, int y1) {
    std::vector<int>& out = part[size_t(nextPart++)];
    out.push_back(y0);  // the band's first row, to sort the parts
    for (int y = y0; y < y1; ++y) {
      auto test = [&](int x) {
        const size_t i = size_t(y) * w + x;
        if (m[i] && ((x > 0 && !m[i - 1]) || (x + 1 < w && !m[i + 1]) || (y > 0 && !m[i - w]) ||
                     (y + 1 < h && !m[i + w])))
          out.push_back(int(i));
      };
      // 8 pixels at a time are skipped where each equals its 4 neighbours (no set pixel there
      // has an unset one): most of a mask, so the scan is mostly word compares.
      const uint8_t* row = m + size_t(y) * w;
      const uint8_t* up = y > 0 ? row - w : row;
      const uint8_t* dn = y + 1 < h ? row + w : row;
      int x = 0;
      if (w > 0) test(x++);
      for (; x + 8 < w; x += 8) {
        uint64_t c, l, r, u, d;
        std::memcpy(&c, row + x, 8);
        std::memcpy(&l, row + x - 1, 8);
        std::memcpy(&r, row + x + 1, 8);
        std::memcpy(&u, up + x, 8);
        std::memcpy(&d, dn + x, 8);
        if (((c ^ l) | (c ^ r) | (c ^ u) | (c ^ d)) == 0) continue;
        for (int k = x; k < x + 8; ++k) test(k);
      }
      for (; x < w; ++x) test(x);
    }
  });
  part.resize(size_t(nextPart));
  std::sort(part.begin(), part.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
    return a[0] < b[0];
  });
  for (const auto& p : part) boundary.insert(boundary.end(), p.begin() + 1, p.end());
}

// The disc as a half-width per row offset dy = -ri..ri: the same pixels as the offset list
// (plus the centre, which stamps nothing: it is set, so its distance is already 0).
std::vector<int> discRows(const std::vector<std::pair<int, int>>& disc, int ri) {
  std::vector<int> half(size_t(2 * ri + 1), 0);
  for (const auto& o : disc) half[size_t(o.second + ri)] = std::max(half[size_t(o.second + ri)], std::abs(o.first));
  return half;
}

}  // namespace

// When stamping costs more than a frame's worth of writes (a large r on a noisy matte), the
// distance transform does it instead. Row bands on threads, each stamping its own rows.
Mask dilate(const Mask& m, int w, int h, double r) {
  std::vector<std::pair<int, int>> disc;
  std::vector<int> boundary;
  discAndBoundary(m.data(), w, h, r, disc, boundary, nullptr);
  Mask out(m.size());
  if (double(boundary.size()) * disc.size() > double(m.size())) {
    std::unique_ptr<float[]> d(new float[m.size()]);
    squaredDistance(m.data(), w, h, d.get(), nullptr);
    const float r2 = float(r * r);
    parallelFor(h, 16, [&](int y0, int y1) {
      for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) out[i] = d[i] <= r2;
    });
    return out;
  }
  const int ri = int(std::floor(r));
  const std::vector<int> half = discRows(disc, ri);
  parallelFor(h, 16, [&](int y0, int y1) {
    std::copy(m.begin() + ptrdiff_t(y0) * w, m.begin() + ptrdiff_t(y1) * w, out.begin() + ptrdiff_t(y0) * w);
    const auto first = std::lower_bound(boundary.begin(), boundary.end(), std::max(y0 - ri, 0) * w);
    const auto last = std::lower_bound(boundary.begin(), boundary.end(), std::min(y1 + ri, h) * w);
    for (auto it = first; it != last; ++it) {
      const int x = *it % w, y = *it / w;
      for (int py = std::max(y0, y - ri); py < std::min(y1, y + ri + 1); ++py) {
        const int hw = half[size_t(py - y + ri)];
        uint8_t* row = &out[size_t(py) * w];
        for (int px = std::max(0, x - hw); px <= std::min(w - 1, x + hw); ++px) row[px] = 1;
      }
    }
  });
  return out;
}

std::vector<float> distanceWithin(const Mask& m, int w, int h, double r,
                                  const std::function<ContourFoot(int)>& foot) {
  std::vector<float> d(m.size());
  distanceWithin(m.data(), w, h, r, foot, d.data());
  return d;
}

// Same two paths as dilate. A foot point lies within 1 px of q and its segment within ~1.12 px,
// so a pixel within r of the contour is within r + 1.5 of the q it is measured from; stamping
// from the boundary is exact as above. Squared distances, one sqrt per pixel at the end.
void distanceWithin(const uint8_t* m, int w, int h, double r, const std::function<ContourFoot(int)>& foot,
                    float* d, double budgetFrames, Workers* wk) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  std::vector<std::pair<int, int>> disc;
  std::vector<int> boundary;
  discAndBoundary(m, w, h, r + 1.5, disc, boundary, wk);
  // Measured at UHD (BFD, 2026-10-22): stamping costs ~15 ms per frame's worth of writes, the
  // distance transform 25-45 ms, so they break even near 2 frames.
  if (double(boundary.size()) * disc.size() > budgetFrames * double(n)) {
    squaredDistance(m, w, h, d, wk);
    parallelFor(wk, h, 16, [&](int y0, int y1) {
      for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) d[i] = std::sqrt(d[i]);
    });
    return;
  }
  std::vector<ContourFoot> feet(boundary.size());
  parallelFor(wk, int(boundary.size()), 256, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b) feet[size_t(b)] = foot(boundary[size_t(b)]);
  });
  // Row bands, handed out one at a time; each band stamps from the boundary pixels within reach
  // of its rows and writes only its own rows.
  const int reach = int(std::floor(r + 1.5));
  const std::vector<int> half = discRows(disc, reach);
  constexpr int kRows = 16;
  auto band = [&](int b) {
    const int y0 = b * kRows, y1 = std::min(h, y0 + kRows);
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) d[i] = m[i] ? 0.0f : float(kFar);
    const auto first = std::lower_bound(boundary.begin(), boundary.end(), std::max(y0 - reach, 0) * w);
    const auto last = std::lower_bound(boundary.begin(), boundary.end(), std::min(y1 + reach, h) * w);
    for (auto it = first; it != last; ++it) {
      const int x = *it % w, y = *it / w;
      const ContourFoot& f = feet[size_t(it - boundary.begin())];
      const bool line = f.nx != 0.0f || f.ny != 0.0f;
      for (int py = std::max(y0, y - reach); py < std::min(y1, y + reach + 1); ++py) {
        const int hw = half[size_t(py - y + reach)];
        stampRow(d + size_t(py) * w, std::max(0, x - hw), std::min(w - 1, x + hw), x, py - y, f, line);
      }
    }
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i)
      if (d[i] < float(kFar)) d[i] = std::sqrt(d[i]);
  };
  dynamicFor<int>(wk, (h + kRows - 1) / kRows, [&](int b, int&) { band(b); });
}

}  // namespace mr
