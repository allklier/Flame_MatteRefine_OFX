#include "ops/edge_width.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "ops/edge_field.h"

namespace mr {

namespace {

// Edges are measured up to this far each side of the contour: an input edge up to ~16 px wide
// is retimed whole, a softer one only over its inner 16 px.
constexpr double kBandPx = 10.0;

// What an inside contour pixel knows about its edge: how much it may be retimed (0-1; < 0: no
// usable measure), how far the transition reaches inside and outside (px), and the inside's
// share of the level span ((hi - 0.5) / (hi - lo); 0.5 for an edge from 0 to 1), and the levels.
// One per contour pixel; footMeanAt reads 7 floats.
struct EdgeInfo {
  float wt, eIn, eOut, fIn, hi, lo, room;
};

}  // namespace

// sd = the signed distance to the 0.5 contour. At each contour pixel: the levels the edge runs
// between (hi inside, lo outside), read just past the transition, and how far the transition
// reaches each side, e = 2 x the area between the profile and its level / the level's distance
// from 0.5: for a linear ramp exactly its extent (the profile read along the normal from many
// pixels, so a 1 px edge's clamped neighbours don't shorten it, as the slope between two pixels
// did). The target is the linear ramp from lo to hi, Width wide, through the same contour: each
// side's share of Width by its share of the level span. Each pixel in the band reads the input's
// profile at g(sd): the transition's e px each side mapped linearly onto the target's, beyond
// that the input as it was (its level where the old transition was wider). So the levels and
// the contour stay, and nothing beyond the transition changes.
void edgeWidth(const float* in, float* out, int w, int h, const EdgeWidthParams& p,
               EdgeWidthStats* stats) {
  EdgeWidthStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  const double width = std::min(std::max(p.widthPx, 0.5), 2.0 * (kBandPx - 2.0));
  const double half = width / 2;
  const double amount = std::min(std::max(p.amount, 0.0), 1.0);
  if (amount == 0.0 || n == 0) {  // nothing changes
    if (out != in) parallelFor(h, 16, [&](int y0, int y1) {
      std::copy(in + size_t(y0) * w, in + size_t(y1) * w, out + size_t(y0) * w);
    });
    if (stats) *stats = st;
    return;
  }
  Workers wk;
  const float* src = in;
  std::unique_ptr<uint8_t[]> below(new uint8_t[n]);
  parallelFor(&wk, h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) below[i] = src[i] < 0.5f;
  });
  const std::vector<int> contour = contourPixels(below.get(), w, h, &wk);
  std::unique_ptr<float[]> sdBuf(new float[n]);
  signedEdgeDistanceExact(src, below.get(), contour, w, h, 0.5f, kBandPx, sdBuf.get(), &wk);
  const float* sd = sdBuf.get();
  std::vector<EdgeInfo> info(contour.size());
  constexpr int kChunk = 64;  // contour pixels per item: their cost varies with the edge
  dynamicFor<int>(&wk, int((contour.size() + kChunk - 1) / kChunk), [&](int chunk, int&) {
    for (int c = chunk * kChunk; c < std::min(int(contour.size()), (chunk + 1) * kChunk); ++c) {
        const int x = contour[size_t(c)] % w, y = contour[size_t(c)] / w;
        const size_t i = size_t(y) * w + x;
        EdgeInfo& e = info[size_t(c)];
        e.wt = -1.0f;
        if (src[i] < 0.5f) continue;  // measured from the inside's contour pixels only (half the work)
        const double slope = contourSlope(src, sd, w, h, x, y);
        double nx, ny;
        if (slope <= 0.0 || !edgeNormal(sd, w, h, x, y, nx, ny)) continue;
        const double e0 = std::min(0.5 / slope, kBandPx - 2.0);
        const double hi = profileAt(src, sd, w, h, x, y, nx, ny, e0 + 1.0);
        const double lo = profileAt(src, sd, w, h, x, y, nx, ny, -(e0 + 1.0));
        if (hi - lo < 0.02) continue;
        // Midpoint sums in 8 steps over twice the slope's estimate + 1 px.
        const double reachOld = std::min(kBandPx - 2.0, 2.0 * e0 + 1.0), step = reachOld / 8;
        double areaIn = 0.0, areaOut = 0.0;
        for (double t = step / 2; t < reachOld; t += step) {
          const double a = profileAt(src, sd, w, h, x, y, nx, ny, t);
          const double b = profileAt(src, sd, w, h, x, y, nx, ny, -t);
          areaIn += (hi - std::min(std::max(a, 0.5), hi)) * step;
          areaOut += (std::min(std::max(b, lo), 0.5) - lo) * step;
        }
        e.eIn = hi - 0.5 > 0.01 ? float(std::min(2.0 * areaIn / (hi - 0.5), kBandPx - 2.0)) : 0.0f;
        e.eOut = 0.5 - lo > 0.01 ? float(std::min(2.0 * areaOut / (0.5 - lo), kBandPx - 2.0)) : 0.0f;
        e.fIn = float((hi - 0.5) / (hi - lo));
        e.hi = float(hi);
        e.lo = float(lo);
        // Thin parts and gaps: going Width / 2 + 1 px across from here, the distance should grow
        // by as much; past the middle of a strand (or gap) at r px it falls again: 2r - reach.
        // room = the widest edge that fits, 2 x (r - 0.5) (>= Width + 1 where nothing is in the
        // way). Widening is capped to it, so a wedge's edges narrow smoothly into its tip
        // (switching the change off there left a hard stub inside soft flanks) and a strand or
        // gap too thin to widen stays as it is. wt (full from half a pixel short of the ideal
        // reach: at a staircase corner the tilted normal measures ~0.5 px less) still keeps
        // narrowing off thin parts.
        const double reach = half + 1.0;
        const double rIn = sampleBilinear(sd, w, h, x + reach * nx, y + reach * ny) - sd[i];
        const double rOut = sd[i] - sampleBilinear(sd, w, h, x - reach * nx, y - reach * ny);
        e.wt = float(std::min(std::max(std::min(rIn, rOut) - half + 0.5, 0.0), 1.0));
        e.room = float(std::max(std::min(rIn, rOut) + reach - 1.0, 0.0));
    }
  });
  const ContourIndex index(contour, w, h);
  // Only pixels within kBandPx - 1 of the contour change (sd beyond the band is +-(band + 1)).
  const NearTiles near(contour, w, h, int(std::ceil(kBandPx + 1.5)) + 1);
  // In place, the new values are collected and written after: the profile is read around each
  // pixel.
  const bool inPlace = out == in;
  constexpr int kRows = 8;
  const int bands = (h + kRows - 1) / kRows;
  std::vector<std::vector<std::pair<int, float>>> parts(size_t(inPlace ? bands : 0));
  std::atomic<long long> changed(0);
  dynamicFor<int>(&wk, bands, [&](int b, int&) {
    long long c = 0;
    for (int y = b * kRows; y < std::min(h, (b + 1) * kRows); ++y)
      near.runs(y, [&](int x0, int x1, bool isNear) {
        if (!isNear) {
          if (!inPlace) std::copy(src + size_t(y) * w + x0, src + size_t(y) * w + x1, out + size_t(y) * w + x0);
          return;
        }
        for (int x = x0; x < x1; ++x) {
          const size_t i = size_t(y) * w + x;
          if (!inPlace) out[i] = src[i];
          const double s = sd[i];
          double nx, ny;
          if (std::fabs(s) > kBandPx - 1.0 || !edgeNormal(sd, w, h, x, y, nx, ny)) continue;
          // The edge's measures at this pixel's foot on the contour.
          double m[7];
          const auto measure = [&](int qx, int qy) -> const float* {
            const int k = index.at(qx, qy);
            return k < 0 ? nullptr : &info[size_t(k)].wt;
          };
          if (footMeanAt(7, w, h, x - s * nx, y - s * ny, m, measure) <= 0.0 || m[0] < 0.0) continue;
          const double eOld = s >= 0 ? m[1] : m[2], as = std::fabs(s), share = s >= 0 ? m[3] : 1.0 - m[3];
          double eNew = std::max(width * share, 0.05), wt = m[0];
          if (eNew > eOld) {  // widening: up to the room (less 0.1 px: a 2 px gap measured a hair
            // wider than it is), never narrower than it was
            eNew = std::max(std::min(eNew, (m[6] - 0.1) * share), eOld);
            wt = 1.0;
          }
          const double widthHere = eNew / std::max(share, 1e-6);
          // An edge already within ~3 % of the target is left as it is (bit for bit): retiming it
          // would only add the measurement's noise (up to 0.04 on a near-vertical 2 px edge).
          // Full from 7 % off, so there is no seam where an edge's width crosses the line.
          const double off = eOld > 0.0 ? std::fabs(std::log(eOld / eNew)) : 1.0;
          const double k = wt * std::min(std::max((off - 0.03) / 0.04, 0.0), 1.0);
          const double g = as <= eNew ? s * eOld / eNew : std::copysign(std::max(as, eOld), s);
          if (g == s || k <= 0.0) continue;
          // A transition under ~1 px (a hard cut, a 1 px edge) has no shape worth keeping, and
          // resampling it failed: its pixels lie ~0.5 px from the contour, except at a staircase
          // corner (~0.1), so the stretched step jumped by up to 0.46 there (a comb along a near-
          // vertical hard edge, saw teeth on slants). There the target ramp is drawn from the
          // levels instead, on sd averaged along the edge over +-Width / 2 (up to 4 px), so it
          // follows the line rather than the pixel staircase; from 1 px on, the resampled profile.
          const double synth = std::min(std::max((1.0 - eOld) / 0.5, 0.0), 1.0);
          double v = synth < 1.0 ? profileAt(src, sd, w, h, x, y, nx, ny, g) : 0.0;
          if (synth > 0.0) {
            const int reach = int(std::min(widthHere / 2, 4.0));
            double sum = s;
            for (int t = 1; t <= reach; ++t)
              sum += sampleBilinear(sd, w, h, x - t * ny, y + t * nx) + sampleBilinear(sd, w, h, x + t * ny, y - t * nx);
            const double sa = sum / (2 * reach + 1);
            const double ramp = sa >= 0 ? std::min(m[4], 0.5 + (m[4] - 0.5) * sa / eNew)
                                        : std::max(m[5], 0.5 + (0.5 - m[5]) * sa / eNew);
            v = synth * ramp + (1.0 - synth) * v;
          }
          const float o = float(src[i] + amount * k * (v - src[i]));
          c += o != src[i];
          if (inPlace)
            parts[size_t(b)].emplace_back(int(i), o);
          else
            out[i] = o;
        }
      });
    changed += c;
  });
  parallelFor(&wk, int(parts.size()), 1, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b)
      for (const auto& v : parts[size_t(b)]) out[size_t(v.first)] = v.second;
  });
  st.pixelsChanged = changed;
  if (stats) *stats = st;
}

}  // namespace mr
