#include "ops/snap_to_image.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace mr {

namespace {

constexpr int kTile = 64;            // work unit: only tiles near the 0.5 contour are filtered
constexpr float kGuideMax = 64.0f;   // brighter picture values (and inf) are clamped to this
constexpr double kDarkFloor = 0.02;  // epsilon's brightness never drops below this (shadow noise)
constexpr double kFar = 1e20;
constexpr float kGrey = 0.05f;      // grey: matte values in (kGrey, 1 - kGrey)
constexpr float kSlope = 0.04f;     // grey with a central difference above this is edge
constexpr int kPasses = 2;          // filter passes (decision in snap_to_image.h)

// Runs fn(begin, end) over [0, n) on up to 32 threads, at least minPerThread items each
// (copied from ops/morphology.cpp, where it is file-local).
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

// 1-D squared distance transform (Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled
// Functions", 2012), as in ops/morphology.cpp (file-local there). f, d: n values with stride;
// v, z: scratch of n and n + 1.
void edt1d(const double* f, double* d, int n, int* v, double* z) {
  int k = 0;
  v[0] = 0;
  z[0] = -kFar;
  z[1] = kFar;
  auto meet = [&](int q, int p) {
    return ((f[q] + double(q) * q) - (f[p] + double(p) * p)) / (2.0 * (q - p));
  };
  for (int q = 1; q < n; ++q) {
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

// Squared distance from each pixel of a rw x rh region to the nearest pixel with
// target[i] == want (>= kFar if none).
void squaredDistance(const uint8_t* target, uint8_t want, int rw, int rh, std::vector<double>& g) {
  g.resize(size_t(rw) * rh);
  const int m = std::max(rw, rh);
  std::vector<double> f(m), d(m), z(m + 1);
  std::vector<int> v(m);
  for (int x = 0; x < rw; ++x) {
    for (int y = 0; y < rh; ++y) f[y] = target[size_t(y) * rw + x] == want ? 0.0 : kFar;
    edt1d(f.data(), d.data(), rh, v.data(), z.data());
    for (int y = 0; y < rh; ++y) g[size_t(y) * rw + x] = d[y];
  }
  for (int y = 0; y < rh; ++y) {
    edt1d(&g[size_t(y) * rw], d.data(), rw, v.data(), z.data());
    std::copy(d.begin(), d.begin() + rw, g.begin() + size_t(y) * rw);
  }
}

// Summed-area table of C interleaved channels over a rw x rh region, in double (the
// covariances subtract nearly equal sums).
template <int C>
struct Sat {
  int rw = 0, rh = 0;
  std::vector<double> s;
  // v(x, y, out) fills the C values of region pixel (x, y).
  template <class V>
  void build(int w_, int h_, V v) {
    rw = w_;
    rh = h_;
    s.assign(size_t(rw + 1) * (rh + 1) * C, 0.0);
    double val[C];
    for (int y = 0; y < rh; ++y) {
      double row[C] = {};
      for (int x = 0; x < rw; ++x) {
        v(x, y, val);
        double* dst = &s[(size_t(y + 1) * (rw + 1) + x + 1) * C];
        const double* up = &s[(size_t(y) * (rw + 1) + x + 1) * C];
        for (int c = 0; c < C; ++c) {
          row[c] += val[c];
          dst[c] = up[c] + row[c];
        }
      }
    }
  }
  // Mean over region pixels [x0, x1] x [y0, y1], inclusive.
  void mean(int x0, int y0, int x1, int y1, double* out) const {
    const double inv = 1.0 / (double(x1 - x0 + 1) * (y1 - y0 + 1));
    const double* a = &s[(size_t(y0) * (rw + 1) + x0) * C];
    const double* b = &s[(size_t(y0) * (rw + 1) + x1 + 1) * C];
    const double* c = &s[(size_t(y1 + 1) * (rw + 1) + x0) * C];
    const double* d = &s[(size_t(y1 + 1) * (rw + 1) + x1 + 1) * C];
    for (int k = 0; k < C; ++k) out[k] = (d[k] - b[k] - c[k] + a[k]) * inv;
  }
};

inline float guideValue(float v) { return v >= 0.0f ? std::min(v, kGuideMax) : 0.0f; }  // NaN -> 0

}  // namespace

double snapEpsilon(double detail) {
  return std::pow(10.0, -1.0 - 3.0 * std::min(std::max(detail, 0.0), 1.0));
}

void snapToImage(const float* in, const float* rgb, float* out, int w, int h, const SnapParams& p,
                 SnapStats* stats) {
  SnapStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  // Pixels outside the band are copied; band pixels read the input from src, never from out
  // (out may be in, and tiles read each other's pixels).
  std::vector<float> copy;
  const float* src = in;
  if (out == in) {
    copy.assign(in, in + n);
    src = copy.data();
  } else if (n) {
    std::memcpy(out, in, n * sizeof(float));
  }
  const double band = p.bandPx;
  if (n == 0 || band <= 0.0) {
    if (stats) *stats = st;
    return;
  }

  const int tilesX = (w + kTile - 1) / kTile, tilesY = (h + kTile - 1) / kTile;
  const int reach = int(std::ceil(band)) + 1;  // > Band: the distance region's margin
  // tileWt[t]: the blend weight of tile t's pixels (empty: none > 0).
  std::vector<std::vector<float>> tileWt(size_t(tilesX) * tilesY);

  // The band: within Band px of an edge pixel of matte m: grey (kGrey to 1 - kGrey) and sloped
  // (not a flat grey area: a semi-transparent face or haze is no edge, and the contrast boost
  // would push it towards 0 or 1), or next to a 0.5 crossing. Anchoring it on the 0.5 contour alone left rings and dots where a blurred
  // matte hides a thin gap: only an island or a line of it dips below 0.5, so the band was a
  // disc or strip around that and cut the gap's snapped shape at its border (decision in
  // snap_to_image.h). The weight: d = grid distance to the nearest edge pixel, then a
  // smoothstep from 1 at Band / 2 to 0 at Band, so the filtered band meets the untouched matte
  // without a seam. Only tiles with edge pixels within reach are searched; the distance region
  // reaches beyond Band, so d is exact wherever it is below Band.
  auto addBand = [&](const float* m) {
    auto edge = [&](int x, int y) {
      const float v = m[size_t(y) * w + x];
      if (v > kGrey && v < 1.0f - kGrey) {
        const float gx = m[size_t(y) * w + std::min(x + 1, w - 1)] - m[size_t(y) * w + std::max(x - 1, 0)];
        const float gy = m[size_t(std::min(y + 1, h - 1)) * w + x] - m[size_t(std::max(y - 1, 0)) * w + x];
        if (std::fabs(gx) + std::fabs(gy) > kSlope) return true;
      }
      const bool a = v >= 0.5f;
      return (x > 0 && (m[size_t(y) * w + x - 1] >= 0.5f) != a) ||
             (x + 1 < w && (m[size_t(y) * w + x + 1] >= 0.5f) != a) ||
             (y > 0 && (m[size_t(y - 1) * w + x] >= 0.5f) != a) ||
             (y + 1 < h && (m[size_t(y + 1) * w + x] >= 0.5f) != a);
    };
    std::vector<long long> hits(size_t(tilesX) * tilesY, 0);
    parallelFor(tilesY, 1, [&](int ty0, int ty1) {
      for (int ty = ty0; ty < ty1; ++ty)
        for (int y = ty * kTile; y < std::min(h, (ty + 1) * kTile); ++y)
          for (int x = 0; x < w; ++x) hits[size_t(ty) * tilesX + x / kTile] += edge(x, y);
    });
    const int nb = (reach + kTile - 1) / kTile;
    std::vector<int> candidates;
    for (int ty = 0; ty < tilesY; ++ty)
      for (int tx = 0; tx < tilesX; ++tx) {
        long long a = 0;
        for (int y = std::max(0, ty - nb); y <= std::min(tilesY - 1, ty + nb); ++y)
          for (int x = std::max(0, tx - nb); x <= std::min(tilesX - 1, tx + nb); ++x)
            a += hits[size_t(y) * tilesX + x];
        if (a > 0) candidates.push_back(ty * tilesX + tx);
      }
    parallelFor(int(candidates.size()), 4, [&](int begin, int end) {
      std::vector<uint8_t> target;
      std::vector<double> dist;
      for (int ci = begin; ci < end; ++ci) {
        const int t = candidates[size_t(ci)];
        const int tx0 = (t % tilesX) * kTile, ty0 = (t / tilesX) * kTile;
        const int tx1 = std::min(w, tx0 + kTile), ty1 = std::min(h, ty0 + kTile);  // exclusive
        const int dx0 = std::max(0, tx0 - reach), dy0 = std::max(0, ty0 - reach);
        const int dw = std::min(w, tx1 + reach) - dx0, dh = std::min(h, ty1 + reach) - dy0;
        target.resize(size_t(dw) * dh);
        for (int y = 0; y < dh; ++y)
          for (int x = 0; x < dw; ++x) target[size_t(y) * dw + x] = edge(x + dx0, y + dy0);
        squaredDistance(target.data(), 1, dw, dh, dist);
        std::vector<float>& wt = tileWt[size_t(t)];
        for (int y = ty0; y < ty1; ++y)
          for (int x = tx0; x < tx1; ++x) {
            const double d = std::sqrt(dist[size_t(y - dy0) * dw + x - dx0]);
            const double u = std::min(1.0, (band - d) / (0.5 * band));
            if (u <= 0.0) continue;
            if (wt.empty()) wt.assign(size_t(kTile) * kTile, 0.0f);
            float& v = wt[size_t(y - ty0) * kTile + x - tx0];
            v = std::max(v, float(u * u * (3.0 - 2.0 * u)));
          }
      }
    });
  };

  const int r = std::max(1, int(std::lround(p.radiusPx)));
  const double eps = snapEpsilon(p.detail);
  // Edge From Picture: the matte is filtered after a contrast boost around 0.5. The guided
  // filter fits the matte as a linear function of the picture per window, so a matte softer
  // than the picture's edge comes out soft too (its fit has a slope < 1); a hard input lets the
  // picture's colours set the edge's softness (He's guided feathering), but it also overrides a
  // good matte's own soft detail (hair). 0 = the matte as is, 1 = nearly binary (slope 20).
  const double firm = 1.0 / (1.0 - 0.95 * std::min(std::max(p.fromPicture, 0.0), 1.0));

  // One pass: the guided filter on pin (contrast-boosted), blended into the source matte by
  // band weight and confidence, written to pout's band pixels.
  std::mutex statsLock;
  auto pass = [&](const float* pin, float* pout, bool last) {
  parallelFor(tilesX * tilesY, 16, [&](int begin, int end) {
    SnapStats ts;
    Sat<13> s1;  // g (3), p, g g^T (6), g p (3)
    Sat<5> s2;   // a (3), b, kappa
    std::vector<float> ab;
    for (int t = begin; t < end; ++t) {
      const std::vector<float>& wt = tileWt[size_t(t)];
      if (wt.empty()) continue;
      const int tx0 = (t % tilesX) * kTile, ty0 = (t / tilesX) * kTile;
      const int tx1 = std::min(w, tx0 + kTile), ty1 = std::min(h, ty0 + kTile);  // exclusive
      // Statistics region: the tile + 2r (a pixel's mean of a, b reaches r, each a's window r
      // more); coefficient region: the tile + r. Both clipped to the frame, so every window
      // clipped to the frame lies inside them (the paper's box filter normalises by the count).
      const int rx0 = std::max(0, tx0 - 2 * r), ry0 = std::max(0, ty0 - 2 * r);
      const int rx1 = std::min(w, tx1 + 2 * r), ry1 = std::min(h, ty1 + 2 * r);
      const int ax0 = std::max(0, tx0 - r), ay0 = std::max(0, ty0 - r);
      const int ax1 = std::min(w, tx1 + r), ay1 = std::min(h, ty1 + r);
      s1.build(rx1 - rx0, ry1 - ry0, [&](int x, int y, double* v) {
        const size_t i = size_t(y + ry0) * w + x + rx0;
        const double g0 = guideValue(rgb[i * 3]), g1 = guideValue(rgb[i * 3 + 1]),
                     g2 = guideValue(rgb[i * 3 + 2]);
        const double m = std::min(1.0, std::max(0.0, (pin[i] - 0.5) * firm + 0.5));
        v[0] = g0, v[1] = g1, v[2] = g2, v[3] = m;
        v[4] = g0 * g0, v[5] = g0 * g1, v[6] = g0 * g2, v[7] = g1 * g1, v[8] = g1 * g2, v[9] = g2 * g2;
        v[10] = g0 * m, v[11] = g1 * m, v[12] = g2 * m;
      });
      // Per window k: a = (Sigma + e U)^-1 cov(g, p), b = mean p - a . mean g (the paper's
      // eq. 14-15). e is Detail's epsilon times the window's mean brightness squared: scaling
      // the picture by c scales Sigma by c^2, so the result doesn't depend on exposure (the
      // guide stays scene-linear: a composite edge is linear in alpha only in linear light).
      const int aw = ax1 - ax0, ah = ay1 - ay0;
      ab.resize(size_t(aw) * ah * 5);
      for (int y = ay0; y < ay1; ++y)
        for (int x = ax0; x < ax1; ++x) {
          double m[13];
          s1.mean(std::max(0, x - r) - rx0, std::max(0, y - r) - ry0, std::min(w - 1, x + r) - rx0,
                  std::min(h - 1, y + r) - ry0, m);
          const double bright = std::max((m[0] + m[1] + m[2]) / 3.0, kDarkFloor);
          const double e = eps * bright * bright;
          const double s00 = m[4] - m[0] * m[0] + e, s01 = m[5] - m[0] * m[1], s02 = m[6] - m[0] * m[2];
          const double s11 = m[7] - m[1] * m[1] + e, s12 = m[8] - m[1] * m[2], s22 = m[9] - m[2] * m[2] + e;
          const double c0 = m[10] - m[0] * m[3], c1 = m[11] - m[1] * m[3], c2 = m[12] - m[2] * m[3];
          // Symmetric 3x3 inverse by cofactors (positive definite: Sigma >= 0, e > 0).
          const double i00 = s11 * s22 - s12 * s12, i01 = s02 * s12 - s01 * s22, i02 = s01 * s12 - s02 * s11;
          const double i11 = s00 * s22 - s02 * s02, i12 = s01 * s02 - s00 * s12, i22 = s00 * s11 - s01 * s01;
          const double det = s00 * i00 + s01 * i01 + s02 * i02;
          const double a0 = (i00 * c0 + i01 * c1 + i02 * c2) / det;
          const double a1 = (i01 * c0 + i11 * c1 + i12 * c2) / det;
          const double a2 = (i02 * c0 + i12 * c1 + i22 * c2) / det;
          float* o = &ab[(size_t(y - ay0) * aw + x - ax0) * 5];
          o[0] = float(a0), o[1] = float(a1), o[2] = float(a2);
          o[3] = float(m[3] - a0 * m[0] - a1 * m[1] - a2 * m[2]);
          // Confidence kappa: the window's picture contrast against e. Where the picture is flat
          // (no edge to snap to) a -> 0 and the plain filter returns the box-blurred matte, which
          // would soften a good edge; there the input is kept instead. Averaged over the windows
          // like a and b: per pixel's own window it jumps from ~1 to ~0 one radius beyond a
          // hard picture edge, a visible step.
          const double var = (s00 + s11 + s22 - 3.0 * e) / 3.0;
          o[4] = float(var / (var + e));
        }
      s2.build(aw, ah, [&](int x, int y, double* v) {
        const float* o = &ab[(size_t(y) * aw + x) * 5];
        v[0] = o[0], v[1] = o[1], v[2] = o[2], v[3] = o[3], v[4] = o[4];
      });
      for (int y = ty0; y < ty1; ++y)
        for (int x = tx0; x < tx1; ++x) {
          const double weight = wt[size_t(y - ty0) * kTile + x - tx0];
          if (weight <= 0.0) continue;
          const size_t i = size_t(y) * w + x;
          double m[5];  // q = mean a . g + mean b over the windows containing the pixel
          s2.mean(std::max(0, x - r) - ax0, std::max(0, y - r) - ay0, std::min(w - 1, x + r) - ax0,
                  std::min(h - 1, y + r) - ay0, m);
          const double q = m[0] * guideValue(rgb[i * 3]) + m[1] * guideValue(rgb[i * 3 + 1]) +
                           m[2] * guideValue(rgb[i * 3 + 2]) + m[3];
          const double k = m[4] * weight;
          const float o = float(std::min(1.0, std::max(0.0, src[i] + k * (q - src[i]))));
          if (last && o != src[i]) {
            ++ts.pixelsChanged;
            ts.maxChange = std::max(ts.maxChange, std::fabs(o - src[i]));
          }
          if (last) ++ts.bandPixels;
          pout[i] = o;
        }
    }
    std::lock_guard<std::mutex> lock(statsLock);
    st.bandPixels += ts.bandPixels;
    st.pixelsChanged += ts.pixelsChanged;
    st.maxChange = std::max(st.maxChange, ts.maxChange);
  });
  };
  // Pass k filters pass k-1's result (pass 0: the input) and blends it into the input: the
  // input's wrongly-foreground pixels, which pull a single pass's fit back and leave a tail
  // about Radius px wide outside the edge, are mostly gone after the first pass.
  addBand(src);
  std::vector<float> bufA, bufB;
  const float* prev = src;
  for (int k = 0; k + 1 < kPasses; ++k) {
    std::vector<float>& cur = k % 2 ? bufB : bufA;
    cur.assign(src, src + n);  // pixels outside the band: the input
    pass(prev, cur.data(), false);
    prev = cur.data();
  }
  pass(prev, out, true);
  if (stats) *stats = st;
}

}  // namespace mr
