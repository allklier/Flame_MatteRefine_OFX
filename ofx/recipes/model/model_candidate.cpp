// model_candidate.h. The research references: ofx/ml/research/vitmatte_thin_hair/exp/synth.py (pack, unsharp,
// local_stretch) and postproc.py (area_filter, levels).
#include "model_candidate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "ops/region_trimap.h"

namespace mr {
namespace recipe {

namespace {

using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ImageNet statistics of the authors' training (repo configs/common/model.py:40-41; research README section 1).
const float kImMean[3] = {0.485f, 0.456f, 0.406f};
const float kImStd[3] = {0.229f, 0.224f, 0.225f};

float clamp01(float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; }  // NaN -> 0

int roundUp32(int v) { return (v + 31) / 32 * 32; }

// numpy.percentile's default (linear interpolation between the sorted values).
double percentile(std::vector<float>& v, double q) {
  if (v.empty()) return 0;
  const double pos = q / 100.0 * double(v.size() - 1);
  const size_t lo = size_t(std::floor(pos)), hi = std::min(lo + 1, v.size() - 1);
  std::nth_element(v.begin(), v.begin() + lo, v.end());
  const double a = v[lo];
  if (hi == lo) return a;
  const double b = *std::min_element(v.begin() + lo + 1, v.end());
  return a + (pos - double(lo)) * (b - a);
}

void split(int a0, int a1, double margin, int maxSide, std::vector<std::pair<int, int>>* cells) {
  const int len = a1 - a0, m = int(std::ceil(margin));
  const int cellMax = std::max(maxSide - 2 * m, 32);
  const int n = std::max(1, (len + cellMax - 1) / cellMax);
  for (int i = 0; i < n; ++i) cells->push_back({a0 + int(int64_t(len) * i / n), a0 + int(int64_t(len) * (i + 1) / n)});
}

// The crop's start on one axis: centred on the cell, shifted into the frame when it fits.
int place(int c0, int c1, int size, int frame) {
  int x = c0 - (size - (c1 - c0)) / 2;
  if (size >= frame) return 0;
  return std::min(std::max(x, 0), frame - size);
}

}  // namespace

float quantiseTrimap(float v) {
  if (!(v >= 85.0f / 255.0f)) return 0.0f;  // NaN -> 0
  return v >= 170.0f / 255.0f ? 1.0f : 0.5f;
}

std::vector<CropRect> planCrops(int bx0, int by0, int bx1, int by1, int w, int h, double margin, int maxSide) {
  std::vector<CropRect> out;
  if (bx1 <= bx0 || by1 <= by0) return out;
  std::vector<std::pair<int, int>> xs, ys;
  split(bx0, bx1, margin, maxSide, &xs);
  split(by0, by1, margin, maxSide, &ys);
  const int m = int(std::ceil(margin));
  for (const auto& cy : ys)
    for (const auto& cx : xs) {
      CropRect r;
      r.cx0 = cx.first;
      r.cx1 = cx.second;
      r.cy0 = cy.first;
      r.cy1 = cy.second;
      r.size_w = std::min(std::min(roundUp32(r.cx1 - r.cx0 + 2 * m), roundUp32(w)), maxSide);
      r.size_h = std::min(std::min(roundUp32(r.cy1 - r.cy0 + 2 * m), roundUp32(h)), maxSide);
      r.x = place(r.cx0, r.cx1, r.size_w, w);
      r.y = place(r.cy0, r.cy1, r.size_h, h);
      out.push_back(r);
    }
  return out;
}

void gaussianBlur(const float* in, int w, int h, double sigma, float* out) {
  const size_t n = size_t(w) * h;
  if (sigma <= 0) {
    if (out != in) std::memcpy(out, in, n * sizeof(float));
    return;
  }
  const int r = int(std::ceil(4 * sigma));
  std::vector<float> k(2 * r + 1);
  double sum = 0;
  for (int i = -r; i <= r; ++i) sum += k[i + r] = float(std::exp(-0.5 * i * i / (sigma * sigma)));
  for (float& v : k) v = float(v / sum);
  std::vector<float> tmp(n);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float s = 0;
      for (int i = -r; i <= r; ++i) s += k[i + r] * in[size_t(y) * w + std::min(std::max(x + i, 0), w - 1)];
      tmp[size_t(y) * w + x] = s;
    }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float s = 0;
      for (int i = -r; i <= r; ++i) s += k[i + r] * tmp[size_t(std::min(std::max(y + i, 0), h - 1)) * w + x];
      out[size_t(y) * w + x] = s;
    }
}

int dropSmallPieces(float* a, int w, int h, float threshold, double minArea) {
  if (minArea <= 1) return 0;  // every piece has at least one pixel
  const size_t n = size_t(w) * h;
  std::vector<int> lab(n, 0), stack, members;
  int removed = 0, next = 0;
  for (size_t s = 0; s < n; ++s) {
    if (lab[s] || !(a[s] > threshold)) continue;
    lab[s] = ++next;
    stack.assign(1, int(s));
    members.clear();
    while (!stack.empty()) {
      const int i = stack.back();
      stack.pop_back();
      members.push_back(i);
      const int x = i % w, y = i / w;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const int j = yy * w + xx;
          if (!lab[j] && a[j] > threshold) {
            lab[j] = next;
            stack.push_back(j);
          }
        }
    }
    if (double(members.size()) < minArea) {
      for (int i : members) a[i] = 0.0f;
      ++removed;
    }
  }
  return removed;
}

void modelCandidate(const float* rgb, const float* base, const float* region, int w, int h,
                    const ModelCandidateParams& p, CropRunner& runner, float* alpha, float* confidence,
                    ModelCandidateStats* stats) {
  ModelCandidateStats st;
  const size_t n = size_t(w) * h;
  if (alpha != base) std::memcpy(alpha, base, n * sizeof(float));
  if (confidence) std::fill(confidence, confidence + n, 0.0f);

  // The box of weight > 0.
  int bx0 = w, by0 = h, bx1 = 0, by1 = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (clamp01(region[size_t(y) * w + x]) > 0.0f) {
        ++st.regionPixels;
        bx0 = std::min(bx0, x);
        bx1 = std::max(bx1, x + 1);
        by0 = std::min(by0, y);
        by1 = std::max(by1, y + 1);
      }
  if (st.regionPixels == 0) {
    if (stats) *stats = st;
    return;
  }
  st.bx0 = bx0, st.by0 = by0, st.bx1 = bx1, st.by1 = by1;
  const int bw = bx1 - bx0, bh = by1 - by0;
  st.scale = p.sizeScale > 0 ? p.sizeScale : h / 1080.0;
  st.margin = p.marginPx * st.scale;
  st.minPieceArea = p.minPiecePx > 0 ? std::max(1.0, p.minPiecePx * st.scale * st.scale) : 0.0;

  // Region luma statistics (display luma = mean of the clamped RGB, as the research's local_stretch).
  auto luma = [&](size_t i) {
    return (clamp01(rgb[i * 3]) + clamp01(rgb[i * 3 + 1]) + clamp01(rgb[i * 3 + 2])) / 3.0f;
  };
  {
    std::vector<float> lv;
    lv.reserve(size_t(st.regionPixels));
    for (int y = by0; y < by1; ++y)
      for (int x = bx0; x < bx1; ++x) {
        const size_t i = size_t(y) * w + x;
        if (clamp01(region[i]) > 0.0f) lv.push_back(luma(i));
      }
    st.lumaP1 = percentile(lv, 1);
    st.lumaP99 = percentile(lv, 99);
  }
  const double range = st.lumaP99 - st.lumaP1;
  st.stretched = p.lowContrast == LowContrast::On ||
                 (p.lowContrast == LowContrast::Auto && range < p.lowContrastMaxRange && st.lumaP99 < p.lowContrastMaxHigh);
  const bool sharpen = p.unsharpAmount > 0 && (!st.stretched || p.unsharpWhenStretched);

  // The model's alpha on the box (0 where no crop ran).
  std::vector<float> model(size_t(bw) * bh, 0.0f);
  st.crops = planCrops(bx0, by0, bx1, by1, w, h, st.margin, p.maxCropSide);
  RegionTrimapParams tp;
  tp.fgThreshold = p.fgThreshold;
  for (const CropRect& c : st.crops) {
    bool any = false;  // a cell without region weight is skipped
    for (int y = c.cy0; y < c.cy1 && !any; ++y)
      for (int x = c.cx0; x < c.cx1; ++x)
        if (clamp01(region[size_t(y) * w + x]) > 0.0f) {
          any = true;
          break;
        }
    if (!any) continue;
    const auto tp0 = Clock::now();
    const int S = c.size_w, T = c.size_h;
    const size_t plane = size_t(S) * T;
    std::vector<float> in(4 * plane), cb(plane), cr(plane);
    for (int ly = 0; ly < T; ++ly)
      for (int lx = 0; lx < S; ++lx) {
        const size_t i = size_t(std::min(c.y + ly, h - 1)) * w + std::min(c.x + lx, w - 1), o = size_t(ly) * S + lx;
        for (int ch = 0; ch < 3; ++ch) in[ch * plane + o] = clamp01(rgb[i * 3 + ch]);
        cb[o] = base[i];
        cr[o] = region[i];
      }
    if (st.stretched) {  // research local_stretch (0.05-0.95), half strength ("mild")
      const double lo = st.lumaP1, d = std::max(range, 1e-3);
      for (size_t i = 0; i < 3 * plane; ++i)
        in[i] = float(0.5 * in[i] + 0.5 * clamp01(float((in[i] - lo) / d * 0.9 + 0.05)));
    }
    if (sharpen) {
      std::vector<float> b(plane);
      for (int ch = 0; ch < 3; ++ch) {
        float* pl = &in[ch * plane];
        gaussianBlur(pl, S, T, p.unsharpSigma, b.data());
        for (size_t i = 0; i < plane; ++i) pl[i] = clamp01(float(pl[i] + p.unsharpAmount * (pl[i] - b[i])));
      }
    }
    for (int ch = 0; ch < 3; ++ch) {
      const float mean = p.norm == Norm::ImageNet ? kImMean[ch] : 0.5f;
      const float sd = p.norm == Norm::ImageNet ? kImStd[ch] : 0.5f;
      float* pl = &in[ch * plane];
      for (size_t i = 0; i < plane; ++i) pl[i] = (pl[i] - mean) / sd;
    }
    float* tri = &in[3 * plane];
    regionTrimap(cb.data(), cr.data(), S, T, tp, tri);
    for (size_t i = 0; i < plane; ++i) tri[i] = cb[i] >= p.fgThreshold ? 1.0f : quantiseTrimap(tri[i]);
    st.prepMs += msSince(tp0);

    const auto tr0 = Clock::now();
    std::vector<float> out(plane);
    runner.run(p.modelId, in.data(), 4, T, S, out.data(), 1, c.x, c.y);
    st.runMs += msSince(tr0);
    for (int y = c.cy0; y < c.cy1; ++y)
      for (int x = c.cx0; x < c.cx1; ++x)
        model[size_t(y - by0) * bw + (x - bx0)] = out[size_t(y - c.y) * S + (x - c.x)];
  }

  const auto tq0 = Clock::now();
  // Restrict to the region, clamp, drop small pieces, Strand Gain.
  for (int y = by0; y < by1; ++y)
    for (int x = bx0; x < bx1; ++x) {
      float& m = model[size_t(y - by0) * bw + (x - bx0)];
      m = clamp01(region[size_t(y) * w + x]) > 0.0f ? clamp01(m) : 0.0f;
    }
  if (st.minPieceArea > 0) st.piecesRemoved = dropSmallPieces(model.data(), bw, bh, p.pieceThreshold, st.minPieceArea);
  if (p.strandBlack != 0.0 || p.strandGain != 1.0)
    for (float& m : model) m = clamp01(float((m - p.strandBlack) * p.strandGain));

  // Never lower: alpha = base + w (max(base, a) - base).
  for (int y = by0; y < by1; ++y)
    for (int x = bx0; x < bx1; ++x) {
      const size_t i = size_t(y) * w + x;
      const float wt = clamp01(region[i]);
      if (wt <= 0.0f) continue;
      const float b = base[i], a = std::max(b, model[size_t(y - by0) * bw + (x - bx0)]);
      alpha[i] = wt >= 1.0f ? a : b + wt * (a - b);
      if (alpha[i] > b) ++st.pixelsRaised;
    }

  // Confidence on the box: certainty x (0.5 + 0.5 agreement with the picture's ridge evidence).
  if (confidence) {
    const double sig = p.ridgeSigmaPx * st.scale;
    const int pad = int(std::ceil(4 * sig));
    const int ex0 = std::max(bx0 - pad, 0), ey0 = std::max(by0 - pad, 0);
    const int ex1 = std::min(bx1 + pad, w), ey1 = std::min(by1 + pad, h), ew = ex1 - ex0, eh = ey1 - ey0;
    std::vector<float> L(size_t(ew) * eh), G(L.size());
    for (int y = ey0; y < ey1; ++y)
      for (int x = ex0; x < ex1; ++x) L[size_t(y - ey0) * ew + (x - ex0)] = luma(size_t(y) * w + x);
    gaussianBlur(L.data(), ew, eh, sig, G.data());
    const double scale = 0.25 * std::max(range, 0.02);
    for (int y = by0; y < by1; ++y)
      for (int x = bx0; x < bx1; ++x) {
        const size_t i = size_t(y) * w + x;
        if (clamp01(region[i]) <= 0.0f) continue;
        const size_t e = size_t(y - ey0) * ew + (x - ex0);
        const float ridge = clamp01(float(std::fabs(L[e] - G[e]) / scale));
        const float a = clamp01(alpha[i]);
        const float certainty = std::fabs(2.0f * a - 1.0f), agree = 1.0f - std::fabs(a - ridge);
        confidence[i] = certainty * (0.5f + 0.5f * agree);
      }
  }
  st.postMs = msSince(tq0);
  if (stats) *stats = st;
}

}  // namespace recipe
}  // namespace mr
