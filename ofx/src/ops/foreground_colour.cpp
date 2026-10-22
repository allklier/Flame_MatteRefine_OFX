// Operation A7, Foreground Colour: the multi-level foreground estimation of Germer, Uelwer,
// Conrad, Harmeling, "Fast Multi-Level Foreground Estimation" (ICPR 2020; pymatting's
// estimate_foreground_ml, MIT). Per pixel it solves the compositing equation I = a F + (1 - a) B
// for F and B jointly, in least squares, with a smoothness term tying F and B to their 4
// neighbours, weighted by reg + |a - a_neighbour|. Coarse to fine: the image and matte are
// resampled (nearest) to sizes w^(l/n) x h^(l/n), l = 0..n, n = ceil(log2(max(w, h))); each level
// starts from the previous level's F and B (nearest) and runs a few Gauss-Seidel sweeps (10 up
// to 32x32, 2 above). The coarse levels carry the subject's colour outward and the background's
// inward across wide soft areas cheaply.
// Changes from the paper, for speed and for scene-linear pictures:
// - Red-black sweep order instead of raster order, so a sweep runs on threads.
// - Full resolution solves only the soft edge (0 < a < 1) plus Extend px around it; elsewhere
//   F and B are known (a = 1: F = the picture, a = 0: B = the picture) or taken from the level
//   below. There it runs kFullIters over-relaxed sweeps (SOR, kOmega) instead of 2 plain ones
//   (fewer sweeps when the edge is more than a quarter of UHD, e.g. a hazy matte), and the
//   levels between kDenseMax pixels and full resolution are skipped. fgcolor scene, 1920x1080,
//   mean |F - true F| in the band: the paper 0.0072, 8 plain sweeps 0.0074, 8 SOR sweeps 0.0021
//   (converged: 0.0019); UHD 20 ms instead of 84 ms with every level.
// - F and B are clamped to [min(0, picture min), max(1, picture max)] of the pixels solved at
//   that level instead of [0, 1].
#include "ops/foreground_colour.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace mr {

namespace {

constexpr int kSmallLevel = 32;  // levels up to this size get kSmallIters sweeps
constexpr int kSmallIters = 10, kIters = 2;
constexpr int kFullIters = 8;     // full resolution sweeps
constexpr double kOmega = 1.7;   // full resolution over-relaxation
constexpr long long kDenseMax = 1 << 19;  // largest coarse level (pixels) solved
constexpr double kGradientWeight = 1.0;   // the paper's weight of |a - a_neighbour|

// Runs fn(begin, end) over [0, n) on up to 32 threads, at least minPerThread items each
// (as in ops/morphology).
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

// Waits until all n threads have arrived (C++14 has no std::barrier).
class Barrier {
 public:
  explicit Barrier(int n) : n_(n) {}
  void wait() {
    std::unique_lock<std::mutex> lock(m_);
    const long long gen = gen_;
    if (++count_ == n_) {
      count_ = 0;
      ++gen_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return gen_ != gen; });
    }
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  int n_, count_ = 0;
  long long gen_ = 0;
};

float clampAlpha(float a) { return a < 0.0f ? 0.0f : a > 1.0f ? 1.0f : a; }

// Per row, the [x0, x1) runs of pixels to visit.
using Spans = std::vector<std::vector<std::pair<int, int>>>;

struct Solver {
  const float* img;  // w*h*3
  const float* alpha;
  float* f;  // w*h*3, F and B
  float* b;
  int w, h;
  double reg, lo, hi;

  // One pixel's update: the 2x2 normal equations of the compositing equation plus the
  // neighbours' smoothness terms, solved per channel (pymatting's _estimate_fb_ml). Beyond the
  // frame the neighbour is the pixel itself.
  void update(int i, double omega) const {
    const int x = i % w, y = i / w;
    const int nb[4] = {x > 0 ? i - 1 : i, x < w - 1 ? i + 1 : i, y > 0 ? i - w : i,
                       y < h - 1 ? i + w : i};
    const double a0 = clampAlpha(alpha[i]), a1 = 1.0 - a0;
    double a00 = a0 * a0, a11 = a1 * a1;
    const double a01 = a0 * a1;
    double b0[3], b1[3];
    for (int c = 0; c < 3; ++c) {
      b0[c] = a0 * img[i * 3 + c];
      b1[c] = a1 * img[i * 3 + c];
    }
    for (int j : nb) {
      const double wt = reg + kGradientWeight * std::fabs(a0 - clampAlpha(alpha[j]));
      a00 += wt;
      a11 += wt;
      for (int c = 0; c < 3; ++c) {
        b0[c] += wt * f[j * 3 + c];
        b1[c] += wt * b[j * 3 + c];
      }
    }
    const double inv = 1.0 / (a00 * a11 - a01 * a01);  // > 0 when reg > 0
    for (int c = 0; c < 3; ++c) {
      const double nf = (a11 * b0[c] - a01 * b1[c]) * inv, nb = (a00 * b1[c] - a01 * b0[c]) * inv;
      f[i * 3 + c] = float(std::min(hi, std::max(lo, f[i * 3 + c] + omega * (nf - f[i * 3 + c]))));
      b[i * 3 + c] = float(std::min(hi, std::max(lo, b[i * 3 + c] + omega * (nb - b[i * 3 + c]))));
    }
  }

  // Red-black Gauss-Seidel over the spans (nullptr = every pixel): a pixel's 4 neighbours have
  // the other parity, so each half runs on threads. The threads live for the whole sweep (one
  // set of threads per half cost more than the work on a thin edge) and split the rows by
  // work; a barrier separates the halves.
  void sweep(const Spans* spans, int iters, double omega = 1.0) const {
    std::vector<long long> work(h + 1, 0);  // prefix sums of pixels per row
    for (int y = 0; y < h; ++y) {
      long long rowWork = spans ? 0 : w;
      if (spans)
        for (const auto& s : (*spans)[y]) rowWork += s.second - s.first;
      work[y + 1] = work[y] + rowWork;
    }
    const int t = int(std::max<long long>(
        1, std::min<long long>({(long long)std::thread::hardware_concurrency(), 32, work[h] / 4096})));
    Barrier barrier(t);
    auto body = [&](int k) {
      const int y0 = int(std::lower_bound(work.begin(), work.end(), work[h] * k / t) - work.begin());
      const int y1 = int(std::lower_bound(work.begin(), work.end(), work[h] * (k + 1) / t) - work.begin());
      for (int it = 0; it < iters; ++it)
        for (int parity = 0; parity < 2; ++parity) {
          for (int y = std::min(y0, h); y < std::min(y1, h); ++y) {
            if (!spans) {
              for (int x = (y + parity) & 1; x < w; x += 2) update(y * w + x, omega);
              continue;
            }
            for (const auto& s : (*spans)[y])
              for (int x = s.first + ((s.first + y + parity) & 1); x < s.second; x += 2)
                update(y * w + x, omega);
          }
          if (t > 1) barrier.wait();
        }
    };
    std::vector<std::thread> pool;
    for (int k = 1; k < t; ++k) pool.emplace_back(body, k);
    body(0);
    for (auto& th : pool) th.join();
  }
};

// Nearest resampling as pymatting's: source x = floor(x * sw / dw).
void resizeNearest(const float* src, int sw, int sh, int ch, float* dst, int dw, int dh) {
  parallelFor(dh, 64, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const int sy = std::min(int(int64_t(y) * sh / dh), sh - 1);
      for (int x = 0; x < dw; ++x) {
        const int sx = std::min(int(int64_t(x) * sw / dw), sw - 1);
        const float* s = src + (size_t(sy) * sw + sx) * ch;
        float* d = dst + (size_t(y) * dw + x) * ch;
        for (int c = 0; c < ch; ++c) d[c] = s[c];
      }
    }
  });
}

// Per row, the union of the runs of rows y - r .. y + r, each widened by r: the pixels within
// Chebyshev distance r of a run.
Spans grow(const Spans& runs, int w, int h, int r) {
  Spans out(h);
  parallelFor(h, 32, [&](int y0, int y1) {
    std::vector<std::pair<int, int>> all;
    for (int y = y0; y < y1; ++y) {
      all.clear();
      for (int yy = std::max(0, y - r); yy <= std::min(h - 1, y + r); ++yy)
        for (const auto& s : runs[yy])
          all.emplace_back(std::max(0, s.first - r), std::min(w, s.second + r));
      std::sort(all.begin(), all.end());
      for (const auto& s : all)
        if (!out[y].empty() && s.first <= out[y].back().second)
          out[y].back().second = std::max(out[y].back().second, s.second);
        else
          out[y].push_back(s);
    }
  });
  return out;
}

}  // namespace

void foregroundColour(const float* rgb, const float* alpha, float* fg, int w, int h,
                      const FgParams& p, FgStats* stats) {
  FgStats st;
  if (w <= 0 || h <= 0) {
    if (stats) *stats = st;
    return;
  }
  const size_t n = size_t(w) * h;
  const double reg = std::max(1e-9, p.smoothness);

  // Coarse levels 0 .. up to kDenseMax pixels, each solved whole.
  const int levels = std::max(1, int(std::ceil(std::log2(double(std::max(w, h))))));
  std::unique_ptr<float[]> prevF(new float[3]()), prevB(new float[3]());
  int pw = 1, ph = 1;
  for (int l = 0; l < levels; ++l) {
    const int lw = std::max(1, int(std::lround(std::pow(double(w), double(l) / levels))));
    const int lh = std::max(1, int(std::lround(std::pow(double(h), double(l) / levels))));
    const size_t ln = size_t(lw) * lh;
    if ((long long)ln > kDenseMax) break;
    std::unique_ptr<float[]> img(new float[ln * 3]), a(new float[ln]), f(new float[ln * 3]),
        b(new float[ln * 3]);
    resizeNearest(rgb, w, h, 3, img.get(), lw, lh);
    resizeNearest(alpha, w, h, 1, a.get(), lw, lh);
    resizeNearest(prevF.get(), pw, ph, 3, f.get(), lw, lh);
    resizeNearest(prevB.get(), pw, ph, 3, b.get(), lw, lh);
    double lo = 0.0, hi = 1.0;
    for (size_t k = 0; k < ln * 3; ++k) {
      lo = std::min(lo, double(img[k]));
      hi = std::max(hi, double(img[k]));
    }
    const Solver s{img.get(), a.get(), f.get(), b.get(), lw, lh, reg, lo, hi};
    s.sweep(nullptr, lw <= kSmallLevel && lh <= kSmallLevel ? kSmallIters : kIters);
    prevF.swap(f);
    prevB.swap(b);
    pw = lw;
    ph = lh;
    ++st.levels;
  }

  // Full resolution, F in fg. Every pixel starts from the picture where alpha = 1, else the
  // level below; the soft runs are collected on the way.
  auto coarse = [&](const float* buf, int x, int y) {
    const int cx = std::min(int(int64_t(x) * pw / w), pw - 1);
    const int cy = std::min(int(int64_t(y) * ph / h), ph - 1);
    return buf + (size_t(cy) * pw + cx) * 3;
  };
  Spans runs(h);
  std::vector<long long> rowEdge(h);
  parallelFor(h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      int start = -1;
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        const float a = clampAlpha(alpha[i]);
        const bool soft = a > 0.0f && a < 1.0f;
        if (soft && start < 0) start = x;
        if (!soft && start >= 0) {
          runs[y].emplace_back(start, x);
          rowEdge[y] += x - start;
          start = -1;
        }
        const float* src = a >= 1.0f ? rgb + i * 3 : coarse(prevF.get(), x, y);
        for (int c = 0; c < 3; ++c) fg[i * 3 + c] = src[c];
      }
      if (start >= 0) {
        runs[y].emplace_back(start, w);
        rowEdge[y] += w - start;
      }
    }
  });
  for (long long e : rowEdge) st.edgePixels += e;

  // Solve the edge plus Extend px; B is read there and one pixel further (the ring), so only
  // that is initialised: the picture where alpha = 0, else the level below.
  const int m = std::min(1000, std::max(0, int(std::ceil(p.extendPx))));
  const Spans solve = grow(runs, w, h, m), ring = grow(runs, w, h, m + 1);
  std::unique_ptr<float[]> bFull(new float[n * 3]);
  std::vector<float> rowLo(h, 0.0f), rowHi(h, 1.0f);
  parallelFor(h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (const auto& s : ring[y])
        for (int x = s.first; x < s.second; ++x) {
          const size_t i = size_t(y) * w + x;
          const float* src = clampAlpha(alpha[i]) <= 0.0f ? rgb + i * 3 : coarse(prevB.get(), x, y);
          for (int c = 0; c < 3; ++c) {
            bFull[i * 3 + c] = src[c];
            rowLo[y] = std::min(rowLo[y], rgb[i * 3 + c]);
            rowHi[y] = std::max(rowHi[y], rgb[i * 3 + c]);
          }
        }
  });
  for (int y = 0; y < h; ++y)
    for (const auto& s : solve[y]) st.solvedPixels += s.second - s.first;
  const Solver s{rgb, alpha, fg, bFull.get(), w, h, reg,
                 *std::min_element(rowLo.begin(), rowLo.end()),
                 *std::max_element(rowHi.begin(), rowHi.end())};
  // At most about kFullIters sweeps over a quarter of UHD: a matte soft everywhere (haze) gets
  // fewer.
  const long long budget = kFullIters * (1LL << 21);
  const long long iters = std::min<long long>(kFullIters, budget / std::max(1LL, st.solvedPixels));
  s.sweep(&solve, int(std::max<long long>(kIters, iters)), kOmega);
  ++st.levels;

  // Solid pixels keep the picture bit for bit; Mix blends towards it where alpha < 1.
  const float mix = std::min(1.0f, std::max(0.0f, p.mix));
  parallelFor(h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      for (const auto& sp : solve[y])
        for (int x = sp.first; x < sp.second; ++x) {
          const size_t i = size_t(y) * w + x;
          if (clampAlpha(alpha[i]) >= 1.0f)
            for (int c = 0; c < 3; ++c) fg[i * 3 + c] = rgb[i * 3 + c];
        }
      if (mix < 1.0f)
        for (size_t i = size_t(y) * w; i < size_t(y + 1) * w; ++i)
          if (clampAlpha(alpha[i]) < 1.0f)
            for (int c = 0; c < 3; ++c)
              fg[i * 3 + c] = rgb[i * 3 + c] + mix * (fg[i * 3 + c] - rgb[i * 3 + c]);
    }
  });
  if (stats) *stats = st;
}

}  // namespace mr
