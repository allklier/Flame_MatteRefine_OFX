// Plate Merge core (plate_merge.h, docs/plate_merge.md).
#include "plate_merge.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "merge_morph.h"
#include "ops/morphology.h"

namespace mr {
namespace plate {

namespace {

constexpr long kMinRing = 64;          // fewer ring pixels: no Level Match (gain 1)
constexpr double kMeanFloor = 1e-4;    // ring values at or below this give no log ratio
constexpr double kPriorShare = 0.1;    // where the local ring density is below ~10 %, the global gain wins

float clamp01(float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; }  // NaN -> 0

// Separable Gaussian on a gw x gh grid, zero beyond it (sums: numerator and count blur alike).
void blurGrid(std::vector<double>& g, int gw, int gh, double sigma) {
  if (sigma <= 0.0) return;
  const int r = std::max(1, int(std::ceil(3.0 * sigma)));
  std::vector<double> k(size_t(2 * r + 1));
  double sum = 0;
  for (int i = -r; i <= r; ++i) sum += (k[size_t(i + r)] = std::exp(-0.5 * i * i / (sigma * sigma)));
  for (auto& v : k) v /= sum;
  std::vector<double> t(g.size());
  for (int y = 0; y < gh; ++y)
    for (int x = 0; x < gw; ++x) {
      double acc = 0;
      for (int i = std::max(-r, -x); i <= std::min(r, gw - 1 - x); ++i) acc += k[size_t(i + r)] * g[size_t(y) * gw + x + i];
      t[size_t(y) * gw + x] = acc;
    }
  for (int y = 0; y < gh; ++y)
    for (int x = 0; x < gw; ++x) {
      double acc = 0;
      for (int i = std::max(-r, -y); i <= std::min(r, gh - 1 - y); ++i) acc += k[size_t(i + r)] * t[size_t(y + i) * gw + x];
      g[size_t(y) * gw + x] = acc;
    }
}

// The Level Match gain field (linear factors per channel) on a coarse grid, bilinear lookup.
struct GainField {
  int gw = 0, gh = 0, cell = 1;
  std::vector<float> g;  // gw*gh*3
  void at(int x, int y, float out[3]) const {
    const double fx = std::min(std::max((x + 0.5) / cell - 0.5, 0.0), double(gw - 1));
    const double fy = std::min(std::max((y + 0.5) / cell - 0.5, 0.0), double(gh - 1));
    const int x0 = int(fx), y0 = int(fy), x1 = std::min(x0 + 1, gw - 1), y1 = std::min(y0 + 1, gh - 1);
    const double tx = fx - x0, ty = fy - y0;
    for (int c = 0; c < 3; ++c) {
      const double v00 = g[(size_t(y0) * gw + x0) * 3 + c], v10 = g[(size_t(y0) * gw + x1) * 3 + c];
      const double v01 = g[(size_t(y1) * gw + x0) * 3 + c], v11 = g[(size_t(y1) * gw + x1) * 3 + c];
      out[c] = float((v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty);
    }
  }
};

}  // namespace

double autoRingPx(int h) { return std::max(4.0, std::round(0.02 * h)); }
double autoSigmaPx(int h) { return std::max(8.0, 0.05 * h); }

void swapWeight(const float* m, int w, int h, double growPx, double featherPx, float* out, Workers* wk) {
  const size_t n = size_t(w) * h;
  const double grow = std::max(0.0, growPx), feather = std::max(0.0, featherPx), reach = grow + feather;
  std::vector<uint8_t> core(n);
  for (size_t i = 0; i < n; ++i) {
    const float v = clamp01(m[i]);
    out[i] = v < kHazeFloor ? 0.0f : v;
    core[i] = v >= 0.5f;
  }
  std::vector<float> d2(n);
  merge::distanceSquared(core.data(), w, h, d2.data(), wk);
  parallelFor(wk, h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      const double d = std::sqrt(double(d2[i]));
      float ramp = 0.0f;
      if (d <= grow) {
        ramp = 1.0f;
      } else if (feather > 0.0 && d < reach) {
        const double s = (reach - d) / feather;
        ramp = float(s * s * (3.0 - 2.0 * s));
      }
      out[i] = std::max(out[i], ramp);
    }
  });
}

void mergePlates(const float* aRgb, const float* aMatte, const float* bRgb, const float* bMatte, const float* keepA,
                 int w, int h, const PlateParams& p, float* rgbOut, const PlateViews& views, PlateInfo* info) {
  if (w <= 0 || h <= 0 || !aRgb || !aMatte || !bRgb || !bMatte || !rgbOut)
    throw std::invalid_argument("mergePlates: bad size or missing buffer");
  const size_t n = size_t(w) * h;
  Workers wk;
  std::vector<float> a(n), b(n), s(n);
  swapWeight(aMatte, w, h, p.growPx, p.featherPx, a.data(), &wk);
  swapWeight(bMatte, w, h, p.growPx, p.featherPx, b.data(), &wk);
  const bool clean = p.mode == Mode::CleanBackground;
  std::vector<uint8_t> touched(n);  // a or b on: a person (or its halo) may be in that pixel
  long swapPx = 0, residualPx = 0;
  for (size_t i = 0; i < n; ++i) {
    const float k = keepA ? clamp01(keepA[i]) : 0.0f;
    const float u = clean ? a[i] * (1.0f - b[i]) : 1.0f - (1.0f - a[i]) * (1.0f - b[i]);
    s[i] = u * (1.0f - k);
    touched[i] = a[i] > 0.0f || b[i] > 0.0f;
    swapPx += s[i] > 0.0f;
    residualPx += a[i] * b[i] > 0.0f;
    if (views.swap) views.swap[i] = s[i];
    if (views.residual) views.residual[i] = a[i] * b[i];
  }

  // Level Match: B's level to A's on the ring (outside both swap weights and Keep A, within ringPx of them).
  const double ringPx = p.ringPx > 0 ? p.ringPx : autoRingPx(h);
  const double sigma = p.matchSigmaPx > 0 ? p.matchSigmaPx : autoSigmaPx(h);
  std::vector<float> d2(n);
  merge::distanceSquared(touched.data(), w, h, d2.data(), &wk);
  const int cell = std::max(1, int(std::lround(sigma / 4.0)));
  GainField field;
  field.cell = cell;
  field.gw = (w + cell - 1) / cell;
  field.gh = (h + cell - 1) / cell;
  const size_t gn = size_t(field.gw) * field.gh;
  // Per cell and channel: the count and the sum of log2(A / B) over ring pixels where both are above
  // kMeanFloor, each sample clipped to +-kMaxGainStops. A mean of log ratios, not a ratio of means: a
  // highlight would dominate the means, and a sub-pixel drift averages out in the log ratios.
  std::vector<double> sums[6];  // count rgb, sum rgb
  for (auto& v : sums) v.assign(gn, 0.0);
  double tot[6] = {0, 0, 0, 0, 0, 0};
  long ringPx2 = 0;
  const double r2 = ringPx * ringPx;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      if (touched[i] || d2[i] > r2 || (keepA && clamp01(keepA[i]) > 0.0f)) continue;
      const float* pa = aRgb + 3 * i;
      const float* pb = bRgb + 3 * i;
      if (!std::isfinite(pa[0] + pa[1] + pa[2] + pb[0] + pb[1] + pb[2])) continue;
      if (views.ring) views.ring[i] = 1.0f;
      ++ringPx2;
      const size_t g = size_t(y / cell) * field.gw + x / cell;
      for (int c = 0; c < 3; ++c) {
        if (!(pa[c] > kMeanFloor) || !(pb[c] > kMeanFloor)) continue;
        const double l = std::min(kMaxGainStops, std::max(-kMaxGainStops, std::log2(double(pa[c]) / pb[c])));
        sums[c][g] += 1.0;
        sums[3 + c][g] += l;
        tot[c] += 1.0;
        tot[3 + c] += l;
      }
    }
  if (views.ring)
    for (size_t i = 0; i < n; ++i)
      if (views.ring[i] != 1.0f) views.ring[i] = 0.0f;
  const bool match = p.levelMatch && ringPx2 >= kMinRing;
  double global[3] = {0, 0, 0};
  if (match) {
    for (int c = 0; c < 3; ++c) global[c] = tot[c] >= kMinRing ? tot[3 + c] / tot[c] : 0.0;
    for (auto& v : sums) blurGrid(v, field.gw, field.gh, sigma / cell);
    const double prior = kPriorShare * cell * cell;  // blurred counts are per-cell averages (kernel sum 1)
    field.g.resize(gn * 3);
    for (size_t g = 0; g < gn; ++g)
      for (int c = 0; c < 3; ++c) {
        const double st = (sums[3 + c][g] + prior * global[c]) / (sums[c][g] + prior);
        field.g[g * 3 + c] = float(std::exp2(std::min(kMaxGainStops, std::max(-kMaxGainStops, st))));
      }
  }

  parallelFor(&wk, h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        const float* pa = aRgb + 3 * i;
        const float* pb = bRgb + 3 * i;
        float* po = rgbOut + 3 * i;
        const float si = s[i];
        const bool needB = si > 0.0f || views.matchedB;
        float gain[3] = {1.0f, 1.0f, 1.0f};
        if (needB && match) field.at(x, y, gain);
        float bm[3] = {pb[0] * gain[0], pb[1] * gain[1], pb[2] * gain[2]};
        if (views.matchedB)
          for (int c = 0; c < 3; ++c) views.matchedB[3 * i + c] = bm[c];
        for (int c = 0; c < 3; ++c)
          po[c] = si <= 0.0f ? pa[c] : si >= 1.0f ? bm[c] : pa[c] + si * (bm[c] - pa[c]);
      }
  });
  if (info) {
    info->swapPixels = swapPx;
    info->residualPixels = residualPx;
    info->ringPixels = ringPx2;
    info->matched = match;
    for (int c = 0; c < 3; ++c) info->globalGainStops[c] = match ? global[c] : 0.0;
  }
}

}  // namespace plate
}  // namespace mr
