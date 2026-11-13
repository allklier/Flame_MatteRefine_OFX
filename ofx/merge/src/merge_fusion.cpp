#include "merge_fusion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "merge_morph.h"
#include "ops/morphology.h"

namespace mr {
namespace merge {

namespace {

float clamp01(float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; }  // NaN -> 0

constexpr int kMaxCandidates = 8;

}  // namespace

float defaultFusionRule(const float* alpha, const float* conf, int n, const FusionParams& p) {
  if (n <= 0) return 0.0f;
  // The two best gated values.
  int k1 = 0, k2 = -1;
  float g1 = alpha[0] * conf[0], g2 = -1.0f;
  for (int k = 1; k < n; ++k) {
    const float g = alpha[k] * conf[k];
    if (g > g1) {
      k2 = k1, g2 = g1;
      k1 = k, g1 = g;
    } else if (g > g2) {
      k2 = k, g2 = g;
    }
  }
  // The output scale of a confidence: c itself (gated), or the ramp min(1, c / confFull).
  auto out = [&](float c) { return p.confFull > 0.0f ? std::min(1.0f, c / p.confFull) : c; };
  const float v1 = alpha[k1] * out(conf[k1]);
  if (k2 < 0) return v1;
  const float a1 = alpha[k1], a2 = alpha[k2], c1 = conf[k1], c2 = conf[k2];
  const float d = std::fabs(a1 - a2);
  if (d <= p.agreeTol) {
    const float both = 1.0f - (1.0f - c1) * (1.0f - c2);
    const float c = c1 + clamp01(p.agreeBonus) * (both - c1);
    return std::max(v1, std::max(a1, a2) * out(c));
  }
  if (d > p.disagreeGap && std::max(c1, c2) < p.highConf) return v1 * (1.0f - clamp01(p.penalty));
  return v1;
}

void fuseCandidates(const Candidate* c, int n, int w, int h, const FusionParams& p, float* fused, FusionRule rule,
                    Workers* wk) {
  if (n > kMaxCandidates) throw std::invalid_argument("fuseCandidates: more than 8 candidates");
  const size_t total = size_t(std::max(w, 0)) * std::max(h, 0);
  if (n <= 0) {
    std::fill(fused, fused + total, 0.0f);
    return;
  }
  parallelFor(wk, h, 8, [&](int y0, int y1) {
    float a[kMaxCandidates], k[kMaxCandidates];
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      for (int j = 0; j < n; ++j) {
        a[j] = clamp01(c[j].alpha[i]);
        k[j] = c[j].confidence ? clamp01(c[j].confidence[i]) : 1.0f;
      }
      fused[i] = clamp01(rule(a, k, n, p));
    }
  });
}

void prepareRegion(const float* mask, int w, int h, double growPx, double featherPx, float* weight, Workers* wk) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  if (n == 0) return;
  const int r = int(std::lround(std::fabs(growPx)));
  if (r > 0) {
    std::vector<float> m(n);
    for (size_t i = 0; i < n; ++i) m[i] = clamp01(mask[i]);
    if (growPx > 0) maxFilter(m.data(), w, h, r, weight, wk);
    else minFilter(m.data(), w, h, r, weight, wk);
  } else {
    for (size_t i = 0; i < n; ++i) weight[i] = clamp01(mask[i]);
  }
  if (featherPx <= 0) return;
  // Inward ramp from the pixels with weight 0 (as matte_merge.cpp's zone feather).
  std::vector<uint8_t> zero(n);
  for (size_t i = 0; i < n; ++i) zero[i] = weight[i] <= 0.0f;
  std::vector<float> d2(n);
  distanceSquared(zero.data(), w, h, d2.data(), wk);
  parallelFor(wk, h, 8, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      if (weight[i] <= 0.0f || d2[i] >= kFar) continue;
      double t = (std::sqrt(double(d2[i])) - 0.5) / featherPx;
      t = std::min(1.0, std::max(0.0, t));
      const double s = t * t * (3 - 2 * t);
      if (s < 1.0) weight[i] = float(weight[i] * s);
    }
  });
}

SnapParams snapParamsForHeight(int h) {
  SnapParams sp;
  const double k = h / 1080.0;
  sp.radiusPx = std::max(1.0, sp.radiusPx * k);
  sp.bandPx = sp.bandPx * k;
  return sp;
}

void snapInRegion(const float* base, float* matte, const float* picture, const float* region, int w, int h,
                  double strength, const SnapParams& sp, bool neverLower) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  const float s = clamp01(float(strength));
  if (n == 0 || s <= 0.0f) return;
  std::vector<float> wt(n);
  for (size_t i = 0; i < n; ++i) wt[i] = s * clamp01(region[i]);
  snapToImage(matte, picture, matte, w, h, sp, nullptr, wt.data());
  if (neverLower)
    for (size_t i = 0; i < n; ++i) matte[i] = std::max(matte[i], base[i]);
}

}  // namespace merge
}  // namespace mr
