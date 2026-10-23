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
// - Ignore Outer px (2026-10-23, a grown matte's green edge in a Flame test): the band of pixels
//   at or above 0.5 within N px inside the contour (outerBand) gets no data term at any level
//   (the alpha of a coarse pixel covering one is kept, its data dropped), so F and B there are
//   only the neighbours' weighted mean, with weight 1 for solid neighbours (alpha 1, ignored or
//   not) and kSoftLink for soft ones: colour flows in from the trusted interior, not from the
//   doubtful edge. The true alpha stays everywhere else (soft neighbours still link to these
//   pixels by |a - a_neighbour|). Kept rather than a choked alpha for the solver: that also
//   dropped the soft band's own data and cost 20x the error on a clean matte (0.045 vs 0.0021,
//   fgcolor 1080). Ignoring the band only at alpha 1 left its soft >= 0.5 pixels (pure green
//   after a grow) pulling F back to green (+0.31 greenness 0-2 px inside); plain gradient
//   weights let the soft edge leak in (+0.21). Measured, fgcolor 1080, grown 4.8 px, N 6:
//   greenness -0.03 inside, -0.005 / +0.008 just outside (clean -0.04; N 0: +0.53 / +0.29).
#include "ops/foreground_colour.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <new>
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
constexpr double kSoftLink = 0.05;  // an ignored pixel's weight of a soft neighbour (solid: 1)

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
  const uint8_t* ignored = nullptr;  // w*h, 1 = an ignored solid pixel (Ignore Outer px); or none

  // One pixel's update: the 2x2 normal equations of the compositing equation plus the
  // neighbours' smoothness terms, solved per channel (pymatting's _estimate_fb_ml). Beyond the
  // frame the neighbour is the pixel itself.
  void update(int i, double omega) const {
    const int x = i % w, y = i / w;
    const int nb[4] = {x > 0 ? i - 1 : i, x < w - 1 ? i + 1 : i, y > 0 ? i - w : i,
                       y < h - 1 ? i + w : i};
    const bool ign = ignored && ignored[i];
    const double a0 = ign ? 0.0 : clampAlpha(alpha[i]), a1 = ign ? 0.0 : 1.0 - a0;
    double a00 = a0 * a0, a11 = a1 * a1;
    const double a01 = a0 * a1;
    double b0[3], b1[3];
    for (int c = 0; c < 3; ++c) {
      b0[c] = a0 * img[i * 3 + c];
      b1[c] = a1 * img[i * 3 + c];
    }
    for (int j : nb) {
      // An ignored pixel takes its colours from the solid side (alpha = 1, ignored or not) and
      // only weakly from soft neighbours, whose data is the doubtful edge.
      const double wt = ign ? (clampAlpha(alpha[j]) >= 1.0f ? 1.0 : kSoftLink)
                            : reg + kGradientWeight * std::fabs(a0 - clampAlpha(alpha[j]));
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

// Calls visit(x, y, band) for every pixel whose centre is within rr of a seed's (seeds[y] = the
// seeds' x in row y): row bands on threads, each band visiting only its own rows, so visit may
// write those rows' pixels; band = 0 .. bands - 1.
template <class Visit>
void stampDiscs(const std::vector<std::vector<int>>& seeds, int w, int h, double rr, int bands,
                Visit visit) {
  const int reach = int(std::floor(rr));
  std::vector<int> half(2 * reach + 1);  // the disc's half width per row offset
  for (int dy = -reach; dy <= reach; ++dy) half[dy + reach] = int(std::floor(std::sqrt(rr * rr - dy * dy)));
  parallelFor(bands, 1, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b) {
      const int y0 = int(int64_t(h) * b / bands), y1 = int(int64_t(h) * (b + 1) / bands);
      for (int sy = std::max(0, y0 - reach); sy < std::min(h, y1 + reach); ++sy)
        for (int sx : seeds[sy])
          for (int y = std::max(y0, sy - reach); y < std::min(y1, sy + reach + 1); ++y) {
            const int hw = half[y - sy + reach];
            for (int x = std::max(0, sx - hw); x <= std::min(w - 1, sx + hw); ++x) visit(x, y, b);
          }
    }
  });
}

// Ignore Outer px. The band: pixels at or above 0.5 whose centre lies within r + 0.5 px of the
// centre of a pixel below 0.5 (only those with a 4-neighbour at or above 0.5 can be nearest),
// i.e. within about r px inside the 0.5 contour; pixels beyond the frame are no edge. The core:
// the pixels at or above 0.5 deeper than that. Only band pixels within r + 1.5 px of the core
// are ignored: their colour can be carried in from it. A part with no core (a strand or limb
// narrower than about 2r) keeps its own data, as at r = 0: with nothing deeper to take the
// colour from, ignoring it would leave its colour undetermined. The mask is calloc'd: pages
// away from the edge are never touched (zeroing 8 MB at UHD cost ~2 ms).
struct OuterBand {
  std::unique_ptr<uint8_t, void (*)(void*)> mask{nullptr, std::free};  // w*h, 1 = ignored
  std::vector<std::pair<int, int>> xy;  // the ignored pixels' coordinates
};

OuterBand outerBand(const float* alpha, int w, int h, double r) {
  OuterBand out;
  out.mask.reset(static_cast<uint8_t*>(std::calloc(size_t(w) * h, 1)));
  if (!out.mask) throw std::bad_alloc();
  uint8_t* mask = out.mask.get();  // 1 = band, 2 = band near the core (ignored)
  const int bands = std::max(1, std::min({int(std::thread::hardware_concurrency()), 32, h / 16}));
  auto solid = [&](size_t i) { return clampAlpha(alpha[i]) >= 0.5f; };
  // Seeds per row; each thread keeps the rows' above / at / below 0.5 flags of y - 1, y, y + 1
  // so every alpha is read once.
  std::vector<std::vector<int>> seeds(h);
  parallelFor(h, 16, [&](int y0, int y1) {
    std::vector<uint8_t> up[3];
    auto load = [&](std::vector<uint8_t>& row, int y) {
      row.assign(w, 0);
      if (y < 0 || y >= h) return;
      for (int x = 0; x < w; ++x) row[x] = solid(size_t(y) * w + x);
    };
    load(up[0], y0 - 1);
    load(up[1], y0);
    for (int y = y0; y < y1; ++y) {
      load(up[2], y + 1);
      const uint8_t *above = up[0].data(), *cur = up[1].data(), *below = up[2].data();
      for (int x = 0; x < w; ++x)
        if (!cur[x] && (above[x] || below[x] || (x > 0 && cur[x - 1]) || (x < w - 1 && cur[x + 1])))
          seeds[y].push_back(x);
      std::swap(up[0], up[1]);
      std::swap(up[1], up[2]);
    }
  });
  std::vector<std::vector<std::pair<int, int>>> found(bands);
  stampDiscs(seeds, w, h, r + 0.5, bands, [&](int x, int y, int b) {
    const size_t i = size_t(y) * w + x;
    if (!mask[i] && solid(i)) {
      mask[i] = 1;
      found[b].emplace_back(x, y);
    }
  });
  // The core's pixels next to the band: 4-neighbours of band pixels, at or above 0.5, not band
  // (found from the band, so the frame's other pages stay untouched).
  std::vector<std::vector<std::pair<int, int>>> near(bands);
  parallelFor(bands, 1, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b)
      for (const auto& q : found[b]) {
        const int nx[4] = {q.first - 1, q.first + 1, q.first, q.first};
        const int ny[4] = {q.second, q.second, q.second - 1, q.second + 1};
        for (int k = 0; k < 4; ++k) {
          if (nx[k] < 0 || ny[k] < 0 || nx[k] >= w || ny[k] >= h) continue;
          const size_t j = size_t(ny[k]) * w + nx[k];
          if (!mask[j] && solid(j)) near[b].emplace_back(ny[k], nx[k]);
        }
      }
  });
  std::vector<std::pair<int, int>> all;
  for (const auto& v : near) all.insert(all.end(), v.begin(), v.end());
  std::sort(all.begin(), all.end());
  all.erase(std::unique(all.begin(), all.end()), all.end());
  std::vector<std::vector<int>> coreSeeds(h);
  for (const auto& q : all) coreSeeds[q.first].push_back(q.second);
  stampDiscs(coreSeeds, w, h, r + 1.5, bands, [&](int x, int y, int) {
    uint8_t& m = mask[size_t(y) * w + x];
    if (m) m = 2;
  });
  for (const auto& f : found)
    for (const auto& q : f) {
      uint8_t& m = mask[size_t(q.second) * w + q.first];
      m = m == 2;
      if (m) out.xy.push_back(q);
    }
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

  // Ignore Outer px: the ignored pixels at full resolution (none when 0).
  OuterBand band;
  if (p.ignoreOuterPx > 0.0) {
    band = outerBand(alpha, w, h, std::min(1000.0, p.ignoreOuterPx));
    st.ignoredPixels = (long long)band.xy.size();
  }
  const uint8_t* ign = band.mask.get();

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
    // A coarse pixel at or above 0.5 is ignored when any full resolution pixel it covers is.
    std::vector<uint8_t> lign;
    if (ign) {
      lign.assign(ln, 0);
      std::vector<int> cx(w), cy(h);  // the level's pixel covering a full resolution column / row
      for (int x = 0; x < w; ++x) cx[x] = int(int64_t(x) * lw / w);
      for (int y = 0; y < h; ++y) cy[y] = int(int64_t(y) * lh / h) * lw;
      for (const auto& q : band.xy) {
        const size_t j = size_t(cy[q.second]) + cx[q.first];
        if (clampAlpha(a[j]) >= 0.5f) lign[j] = 1;
      }
    }
    const Solver s{img.get(), a.get(), f.get(), b.get(), lw, lh, reg, lo, hi, ign ? lign.data() : nullptr};
    s.sweep(nullptr, lw <= kSmallLevel && lh <= kSmallLevel ? kSmallIters : kIters);
    prevF.swap(f);
    prevB.swap(b);
    pw = lw;
    ph = lh;
    ++st.levels;
  }

  // Full resolution, F in fg. Every pixel starts from the picture where alpha = 1 (and not
  // ignored), else the level below; the runs of soft and ignored pixels are collected on the way.
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
        const bool known = a >= 1.0f && !(ign && ign[i]);
        const bool soft = (a > 0.0f && !known);
        if (soft && start < 0) start = x;
        if (!soft && start >= 0) {
          runs[y].emplace_back(start, x);
          start = -1;
        }
        rowEdge[y] += a > 0.0f && a < 1.0f;
        const float* src = known ? rgb + i * 3 : coarse(prevF.get(), x, y);
        for (int c = 0; c < 3; ++c) fg[i * 3 + c] = src[c];
      }
      if (start >= 0) runs[y].emplace_back(start, w);
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
                 *std::max_element(rowHi.begin(), rowHi.end()), ign};
  // At most about kFullIters sweeps over a quarter of UHD: a matte soft everywhere (haze) gets
  // fewer.
  const long long budget = kFullIters * (1LL << 21);
  const long long iters = std::min<long long>(kFullIters, budget / std::max(1LL, st.solvedPixels));
  s.sweep(&solve, int(std::max<long long>(kIters, iters)), kOmega);
  ++st.levels;

  // Solid pixels keep the picture bit for bit (ignored ones excepted); Mix blends towards it
  // elsewhere.
  const float mix = std::min(1.0f, std::max(0.0f, p.mix));
  parallelFor(h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      for (const auto& sp : solve[y])
        for (int x = sp.first; x < sp.second; ++x) {
          const size_t i = size_t(y) * w + x;
          if (clampAlpha(alpha[i]) >= 1.0f && !(ign && ign[i]))
            for (int c = 0; c < 3; ++c) fg[i * 3 + c] = rgb[i * 3 + c];
        }
      if (mix < 1.0f)
        for (size_t i = size_t(y) * w; i < size_t(y + 1) * w; ++i)
          if (clampAlpha(alpha[i]) < 1.0f || (ign && ign[i]))
            for (int c = 0; c < 3; ++c)
              fg[i * 3 + c] = rgb[i * 3 + c] + mix * (fg[i * 3 + c] - rgb[i * 3 + c]);
    }
  });
  if (stats) *stats = st;
}

}  // namespace mr
