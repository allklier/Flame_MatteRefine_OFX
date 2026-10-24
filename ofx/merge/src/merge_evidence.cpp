#include "merge_evidence.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "ops/morphology.h"

namespace mr {
namespace merge {

namespace {

float finite(float v) { return std::isfinite(v) ? v : 0.0f; }

// 2-98 % range of v over every 7th pixel.
double spread(const std::vector<float>& v) {
  std::vector<float> s;
  s.reserve(v.size() / 7 + 1);
  for (size_t i = 0; i < v.size(); i += 7) s.push_back(v[i]);
  if (s.size() < 2) return 0.0;
  const size_t a = s.size() * 2 / 100, b = s.size() * 98 / 100;
  std::nth_element(s.begin(), s.begin() + a, s.end());
  const float lo = s[a];
  std::nth_element(s.begin(), s.begin() + b, s.end());
  return double(s[b]) - lo;
}

}  // namespace

void pictureEvidence(const float* rgb, const float* change, const uint8_t* candidate, int w, int h,
                     const EvidenceParams& p, float* weight, float* corr, Workers* wk) {
  const size_t n = size_t(w) * h;
  std::vector<float> V(n), S(n);
  parallelFor(wk, h, 8, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      const float r = finite(rgb[i * 3]), g = finite(rgb[i * 3 + 1]), b = finite(rgb[i * 3 + 2]);
      V[i] = (r + g + b) / 3.0f;
      S[i] = std::max(r, std::max(g, b)) - std::min(r, std::min(g, b));
    }
  });
  const double fv = 0.01 * spread(V), fs = 0.01 * spread(S);
  const double floorV = std::max(fv * fv, 1e-12), floorS = std::max(fs * fs, 1e-12);
  const int r = std::max(1, p.radius);
  const double lo = p.lo, span = std::max(1e-6, double(p.hi) - p.lo);
  parallelFor(wk, h, 4, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        weight[i] = 0.0f;
        if (corr) corr[i] = 0.0f;
        if (!candidate[i]) continue;
        double sa = 0, saa = 0, sv = 0, svv = 0, sav = 0, ss = 0, sss = 0, sas = 0;
        int cnt = 0;
        for (int yy = std::max(0, y - r); yy <= std::min(h - 1, y + r); ++yy)
          for (int xx = std::max(0, x - r); xx <= std::min(w - 1, x + r); ++xx) {
            const size_t j = size_t(yy) * w + xx;
            const double a = change[j], v = V[j], s = S[j];
            sa += a, saa += a * a, sv += v, svv += v * v, sav += a * v;
            ss += s, sss += s * s, sas += a * s;
            ++cnt;
          }
        const double ma = sa / cnt, va = saa / cnt - ma * ma;
        if (va <= 1e-8) continue;  // nothing changes in the window
        const double mv = sv / cnt, vv = svv / cnt - mv * mv, cav = sav / cnt - ma * mv;
        const double ms = ss / cnt, vs = sss / cnt - ms * ms, cas = sas / cnt - ma * ms;
        const double c = std::max(std::fabs(cav) / std::sqrt(va * std::max(vv, floorV)),
                                  std::fabs(cas) / std::sqrt(va * std::max(vs, floorS)));
        const double t = std::min(1.0, std::max(0.0, (c - lo) / span));
        weight[i] = float(t * t * (3 - 2 * t));
        if (corr) corr[i] = float(std::min(1.0, c));
      }
  });
  if (p.smoothRadius <= 0) return;
  std::vector<float> raw(weight, weight + n);
  const int sr = p.smoothRadius;
  parallelFor(wk, h, 4, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (!candidate[i]) continue;
        double s = 0;
        int cnt = 0;
        for (int yy = std::max(0, y - sr); yy <= std::min(h - 1, y + sr); ++yy)
          for (int xx = std::max(0, x - sr); xx <= std::min(w - 1, x + sr); ++xx) {
            const size_t j = size_t(yy) * w + xx;
            if (candidate[j]) s += raw[j], ++cnt;
          }
        weight[i] = float(s / cnt);
      }
  });
}

}  // namespace merge
}  // namespace mr
