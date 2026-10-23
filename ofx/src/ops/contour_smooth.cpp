#include "ops/contour_smooth.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {

namespace {

// How far beyond the contour an edge's soft profile may reach and still travel with it (a
// 12 px wide soft edge); pixels farther out are never touched.
constexpr double kProfilePx = 6.0;

// Gaussian sigma per px of Smooth wavelength: the quadratic fit's response (1 + x/2) e^(-x/2),
// x = (2 pi sigma / wavelength)^2, keeps 20 % of a wiggle at the Smooth wavelength, 0.5 % at half.
constexpr double kSigmaPerWavelength = 0.4;

// Features (or gaps) up to 2 x kThinPx wide are not moved, from 2 x kThickPx on fully.
constexpr double kThinPx = 1.5, kThickPx = 3.5;

// A fit with less support than kSupportNone of a straight contour's does not move its sample,
// from kSupportFull on fully (fitMove).
constexpr double kSupportNone = 0.3, kSupportFull = 0.6;

// Every move is shrunk by this (px), smaller ones become 0: the foot points' own scatter
// (+-0.04 px on a clean slant) is no wiggle, and resampling for it only adds noise.
constexpr float kDeadPx = 0.05f;

// How far the profile across an edge may turn back (alpha) and still count as one edge.
constexpr float kMonotone = 0.05f;

// A move more than kSpike px off the median of the moves within kMedianPx is replaced by that
// median (see contourSmooth).
constexpr float kSpike = 0.25f;
constexpr int kMedianPx = 3;

// A point on the contour: a boundary pixel's foot point and its outward unit normal.
struct Sample {
  float x, y, nx, ny;
  int pix;     // the boundary pixel it was measured at
  bool below;  // that pixel's side
};

// Samples bucketed in square cells, for the neighbours within a radius.
struct Grid {
  int cell, cw, ch;
  std::vector<int> start, items;
  Grid(const std::vector<Sample>& s, int w, int h, int cellPx)
      : cell(std::max(cellPx, 1)), cw(w / cell + 1), ch(h / cell + 1), start(size_t(cw) * ch + 1, 0) {
    for (const Sample& p : s) ++start[index(p.x, p.y) + 1];
    for (size_t c = 1; c < start.size(); ++c) start[c] += start[c - 1];
    items.resize(s.size());
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (size_t k = 0; k < s.size(); ++k) items[size_t(fill[index(s[k].x, s[k].y)]++)] = int(k);
  }
  size_t index(float x, float y) const {
    const int cx = std::min(std::max(int(x) / cell, 0), cw - 1), cy = std::min(std::max(int(y) / cell, 0), ch - 1);
    return size_t(cy) * cw + cx;
  }
  template <class F>
  void near(float x, float y, double r, F fn) const {
    const int x0 = std::max(int((x - r) / cell), 0), x1 = std::min(int((x + r) / cell), cw - 1);
    const int y0 = std::max(int((y - r) / cell), 0), y1 = std::min(int((y + r) / cell), ch - 1);
    for (int cy = y0; cy <= y1; ++cy)
      for (int cx = x0; cx <= x1; ++cx) {
        const size_t c = size_t(cy) * cw + cx;
        for (int k = start[c]; k < start[c + 1]; ++k) fn(items[size_t(k)]);
      }
  }
};

// Weighted least squares h(u) = a + b u + c u^2 from the sums of w u^k (k = 0-4), w h u^k
// (k = 0-2).
struct Fit {
  double m[5] = {0, 0, 0, 0, 0}, v[3] = {0, 0, 0};
  void add(double wt, double u, double hgt) {
    double p = wt;
    for (int e = 0; e < 5; ++e, p *= u) {
      m[e] += p;
      if (e < 3) v[e] += p * hgt;
    }
  }
  // The fit at u = 0: quadratic (Cramer), else linear (neighbours on one side only), else the
  // mean.
  double at0() const {
    if (m[0] <= 0.0) return 0.0;
    const double det = m[0] * (m[2] * m[4] - m[3] * m[3]) - m[1] * (m[1] * m[4] - m[3] * m[2]) +
                       m[2] * (m[1] * m[3] - m[2] * m[2]);
    if (det > 1e-6 * m[0] * m[0] * m[0])
      return (v[0] * (m[2] * m[4] - m[3] * m[3]) - m[1] * (v[1] * m[4] - m[3] * v[2]) +
              m[2] * (v[1] * m[3] - m[2] * v[2])) / det;
    const double d2 = m[0] * m[2] - m[1] * m[1];
    if (d2 > 1e-6 * m[0] * m[0]) return (v[0] * m[2] - m[1] * v[1]) / d2;
    return v[0] / m[0];
  }
};

// The smoothed contour at sample j, as an outward move of j's foot point along its normal (px):
// a weighted quadratic h(t) through the neighbours in j's tangent frame, so curves and slants
// are kept and only the wiggle is averaged away (a plain mean, like blurring the distance
// field, shrinks a 34 px arm by sigma^2 / 2r). Weights: Gaussian in t; samples whose normal turns more than the
// corner angle from j's are left out (a corner, the far side of a thin feature), tapered over
// the last 10 deg. Inside that all count fully: weighting by normal agreement (bilateral) kept
// the wiggle, which tilts the normal too. Also tried: a fit on each side of j alone, the one
// with the smaller residual winning (ENO / Kuwahara): it kept kinks but followed half a wiggle
// period, so the wiggle stayed.
double fitMove(const std::vector<Sample>& s, const Grid& grid, int j, double sigma, double cosCorner,
               double cosFull) {
  const Sample& o = s[size_t(j)];
  const double r = 2.5 * sigma;
  Fit fit;
  grid.near(o.x, o.y, r, [&](int k) {
    const Sample& q = s[size_t(k)];
    const double dot = o.nx * q.nx + o.ny * q.ny;
    if (dot <= cosCorner) return;
    const double dx = q.x - o.x, dy = q.y - o.y;
    const double t = -dx * o.ny + dy * o.nx, hgt = dx * o.nx + dy * o.ny;
    // A wiggle tilts the contour by up to ~30 deg; farther off the tangent is another contour.
    if (dx * dx + dy * dy > r * r || std::fabs(hgt) > 1.5 + 0.6 * std::fabs(t)) return;
    const double u = t / sigma;
    fit.add(std::exp(-0.5 * u * u) * std::min(1.0, (dot - cosCorner) / (cosFull - cosCorner)), u, hgt);
  });
  // Little support (a crease's apex, where both legs turn away; a tip): shape, not wiggle. A
  // straight contour has ~2 samples per px (both sides), sum of weights 2 sqrt(2 pi) sigma.
  const double support = fit.m[0] / (2.0 * std::sqrt(2.0 * M_PI) * sigma);
  const double trust = std::min(1.0, std::max(0.0, (support - kSupportNone) / (kSupportFull - kSupportNone)));
  // The fitted curve's height at t = 0 is its distance along j's own normal, the direction the
  // pixels move in (shiftAlongNormal); the distance across the fitted curve instead moved steep
  // wiggle flanks ~6 % short.
  return trust * fit.at0();
}

// The edge band, stored only in the 16 x 16 tiles it touches: full-frame buffers cost more
// than the whole smoothing (UHD, BFD: ~8 ms per fresh 33 MB buffer, 9 of them were 75 of 95
// ms).
struct Band {
  static constexpr int kTile = 16;
  int w, h, tw, th;
  std::vector<int> slot;  // per tile: its storage index, -1 = outside the band
  size_t size = 0;           // tiles in the band x kTile^2
  std::unique_ptr<float[]> d2;  // squared distance to the contour (pixels of the other side)
  std::unique_ptr<int[]> nearest;  // the sample it was measured from, -1 = none
  Band(int w, int h) : w(w), h(h), tw((w + kTile - 1) / kTile), th((h + kTile - 1) / kTile), slot(size_t(tw) * th, -1) {}
  int index(int x, int y) const {
    const int s = slot[size_t(y / kTile) * tw + x / kTile];
    return s < 0 ? -1 : s * kTile * kTile + (y % kTile) * kTile + x % kTile;
  }
};

}  // namespace

void contourSmooth(const float* in, float* out, int w, int h, const ContourSmoothParams& p,
                   ContourSmoothStats* stats) {
  ContourSmoothStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  if (n == 0 || p.smoothPx <= 0.0 || p.maxShiftPx <= 0.0) {
    if (out != in) std::copy(in, in + n, out);
    if (stats) *stats = st;
    return;
  }
  Workers wk;
  if (out != in) parallelFor(&wk, h, 16, [&](int y0, int y1) {
    std::copy(in + size_t(y0) * w, in + size_t(y1) * w, out + size_t(y0) * w);
  });

  // Contour samples on both sides of the Edge At contour (pixels beyond the frame don't count:
  // the frame border is no edge). Hair below Edge At has no contour and gives none.
  std::unique_ptr<uint8_t[]> belowBuf(new uint8_t[n]);
  const uint8_t* below = belowBuf.get();
  parallelFor(&wk, h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) belowBuf[i] = in[i] < p.edgeAt;
  });
  const std::vector<int> contour = contourPixels(below, w, h, &wk);
  std::vector<Sample> samples(contour.size());  // in scan order
  parallelFor(&wk, int(contour.size()), 256, [&](int c0, int c1) {
    for (int c = c0; c < c1; ++c) {
      const int i = contour[size_t(c)], x = i % w, y = i / w;
      const bool b = below[i];
      const ContourFoot f = contourFoot(in, below, w, h, p.edgeAt, i);
      const float sg = b ? -1.0f : 1.0f;  // outward: a below pixel's normal points inward
      samples[size_t(c)] = {x + f.fx, y + f.fy, sg * f.nx, sg * f.ny, i, b};
    }
  });
  st.contourSamples = (long long)samples.size();

  // The signed distance to the contour within the band, as signedEdgeDistance (decision 20)
  // computes it, plus which sample each pixel is nearest to: each sample stamps a disc onto the
  // pixels of the other side. Done here because signedEdgeDistance works on full frames
  // (~35 ms at UHD) and gives no nearest sample. Each row of tiles is stamped by one thread from
  // the samples within reach, in scan order: a strict minimum then keeps the lowest of tied
  // samples, the rule that makes the result independent of the stamping order.
  const double band = p.maxShiftPx + kProfilePx;
  const double r = band + 1.5;  // a foot point lies within 1.5 px of its pixel (morphology.h)
  const int ri = int(std::ceil(r));
  std::vector<int> half(size_t(2 * ri + 1), -1);  // the disc's half-width per row
  for (int dy = -ri; dy <= ri; ++dy)
    for (int dx = -ri; dx <= ri; ++dx)
      if (dx * dx + dy * dy <= r * r) half[size_t(dy + ri)] = std::max(half[size_t(dy + ri)], std::abs(dx));
  // A noisy matte (a contour at 1 in 16 pixels or more) has no line to smooth, and would cost
  // seconds.
  if (samples.size() > n / 16) {
    st.skipped = true;
    if (stats) *stats = st;
    return;
  }
  Band bd(w, h);
  int tiles = 0;
  for (const Sample& s : samples) {
    const int x = s.pix % w, y = s.pix / w, T = Band::kTile;
    for (int ty = std::max(y - ri, 0) / T; ty <= std::min(y + ri, h - 1) / T; ++ty)
      for (int tx = std::max(x - ri, 0) / T; tx <= std::min(x + ri, w - 1) / T; ++tx)
        if (bd.slot[size_t(ty) * bd.tw + tx] < 0) bd.slot[size_t(ty) * bd.tw + tx] = tiles++;
  }
  bd.size = size_t(tiles) * Band::kTile * Band::kTile;
  bd.d2.reset(new float[bd.size]);
  bd.nearest.reset(new int[bd.size]);
  dynamicFor<int>(&wk, bd.th, [&](int ty, int&) {
    constexpr int T = Band::kTile;
    const int y0 = ty * T, y1 = std::min(y0 + T, h);
    for (int tx = 0; tx < bd.tw; ++tx) {
      const int s = bd.slot[size_t(ty) * bd.tw + tx];
      if (s < 0) continue;
      std::fill(&bd.d2[size_t(s) * T * T], &bd.d2[size_t(s + 1) * T * T], 1e30f);
      std::fill(&bd.nearest[size_t(s) * T * T], &bd.nearest[size_t(s + 1) * T * T], -1);
    }
    auto first = std::lower_bound(samples.begin(), samples.end(), size_t(std::max(0, y0 - ri)) * w,
                                  [](const Sample& a, size_t v) { return size_t(a.pix) < v; });
    for (auto it = first; it != samples.end() && it->pix < std::min(h, y1 + ri) * w; ++it) {
      const int x = it->pix % w, y = it->pix / w, j = int(it - samples.begin());
      const float fx = it->x - x, fy = it->y - y, snx = it->nx, sny = it->ny;
      const uint8_t side = it->below;
      for (int py = std::max(y0, y - ri); py < std::min(y1, y + ri + 1); ++py) {
        const int hw = half[size_t(py - y + ri)];
        if (hw < 0) continue;
        const float oy = float(py - y);
        const uint8_t* bl = below + size_t(py) * w;
        const int xa = std::max(0, x - hw), xb = std::min(w - 1, x + hw);
        for (int tx0 = xa; tx0 <= xb;) {  // one tile's part of the span at a time
          const int tEnd = std::min(xb, (tx0 / T + 1) * T - 1);
          const size_t base = size_t(bd.slot[size_t(ty) * bd.tw + tx0 / T]) * T * T + size_t(py - y0) * T;
          float* d2 = &bd.d2[base];
          int* nr = &bd.nearest[base];
          for (int px = tx0; px <= tEnd; ++px) {
            if (bl[px] == side) continue;
            const float ox = float(px - x), vx = ox - fx, vy = oy - fy;
            const float perp = vx * snx + vy * sny;
            const float along = std::max(0.0f, std::fabs(vy * snx - vx * sny) - 0.5f);
            const float dd = std::min(ox * ox + oy * oy, perp * perp + along * along);
            const int k = px % T;
            if (dd < d2[k]) d2[k] = dd, nr[k] = j;
          }
          tx0 = tEnd + 1;
        }
      }
    }
  });
  // Signed distance, > 0 inside, +-(band + 1) beyond the band (as signedEdgeDistance).
  const float cap = float(band + 1.0);
  parallelFor(&wk, int(bd.size / 256), 16, [&](int k0, int k1) {
    for (size_t k = size_t(k0) * 256; k < size_t(k1) * 256; ++k)
      bd.d2[k] = bd.nearest[k] >= 0 ? std::min(std::sqrt(bd.d2[k]), cap) : cap;
  });
  auto sd = [&](int x, int y) {
    const int k = bd.index(x, y);
    const float d = k < 0 ? cap : bd.d2[size_t(k)];
    return below[size_t(y) * w + x] ? -d : d;
  };
  auto sdNear = [&](double x, double y) {
    const int xi = int(std::lround(x)), yi = int(std::lround(y));
    return xi < 0 || yi < 0 || xi >= w || yi >= h ? 0.0f : sd(xi, yi);
  };

  const double sigma = kSigmaPerWavelength * p.smoothPx;
  const Grid grid(samples, w, h, std::max(4, int(std::ceil(sigma))));
  const double corner = std::min(std::max(p.cornerDeg, 10.0), 179.0) * M_PI / 180.0;
  const double cosCorner = std::cos(corner), cosFull = std::cos(corner - 10.0 * M_PI / 180.0);
  std::vector<float> fitted(samples.size(), 0.0f);
  std::vector<uint8_t> kept(samples.size(), 1);  // thin or not one edge: never moves
  parallelFor(&wk, int(samples.size()), 256, [&](int j0, int j1) {
    for (int j = j0; j < j1; ++j) {
      const Sample& o = samples[size_t(j)];
      // Thin features (hair strands, narrow gaps) are left alone: their contour is too poorly
      // sampled to move, and a 1.5 px move erases them. Half-thickness = the deepest |sd| along
      // the normal on either side, within kThickPx.
      double inner = 0.0, outer = 0.0;
      for (double k = 0.5; k <= kThickPx; k += 0.5) {
        inner = std::max(inner, double(sdNear(o.x - k * o.nx, o.y - k * o.ny)));
        outer = std::max(outer, -double(sdNear(o.x + k * o.nx, o.y + k * o.ny)));
      }
      const double thick =
          std::min(1.0, std::max(0.0, (std::min(inner, outer) - kThinPx) / (kThickPx - 0.5 - kThinPx)));
      if (thick == 0.0) continue;
      // Nor is an edge with something else in its profile (a strand below Edge At touching it,
      // a hair root): rising again outside or falling again inside. Its foot points are tilted
      // by the neighbour, and moving it drags the strand along.
      bool monotone = true;
      float prevIn = sampleBilinear(in, w, h, o.x, o.y), prevOut = prevIn;
      for (double k = 1.0; k <= kThickPx + 0.5 && monotone; k += 1.0) {
        const float vi = sampleBilinear(in, w, h, o.x - k * o.nx, o.y - k * o.ny);
        const float vo = sampleBilinear(in, w, h, o.x + k * o.nx, o.y + k * o.ny);
        monotone = vi >= prevIn - kMonotone && vo <= prevOut + kMonotone;
        prevIn = vi;
        prevOut = vo;
      }
      if (!monotone) continue;
      kept[size_t(j)] = 0;
      const double m = thick * fitMove(samples, grid, j, sigma, cosCorner, cosFull);
      fitted[size_t(j)] = float(std::min(std::max(m, -p.maxShiftPx), p.maxShiftPx));
    }
  });
  // A sample right at a crease (the normal turns over 1-2 px there) can still see both legs and
  // moves alone: a spike against the median of the moves within kMedianPx (same side of the
  // corner, kept samples left out) is replaced by it. Only spikes: a plain median flattened a
  // steep wiggle's extremes.
  std::vector<float> move(samples.size());
  parallelFor(&wk, int(samples.size()), 1024, [&](int j0, int j1) {
    std::vector<float> near;
    for (int j = j0; j < j1; ++j) {
      const Sample& o = samples[size_t(j)];
      move[size_t(j)] = 0.0f;
      if (kept[size_t(j)]) continue;
      near.clear();
      grid.near(o.x, o.y, kMedianPx, [&](int k) {
        const Sample& q = samples[size_t(k)];
        if (!kept[size_t(k)] && std::hypot(q.x - o.x, q.y - o.y) <= kMedianPx &&
            o.nx * q.nx + o.ny * q.ny > cosCorner)
          near.push_back(fitted[size_t(k)]);
      });
      std::nth_element(near.begin(), near.begin() + near.size() / 2, near.end());
      const float med = near[near.size() / 2];
      const float m = std::fabs(fitted[size_t(j)] - med) > kSpike ? med : fitted[size_t(j)];
      move[size_t(j)] = std::copysign(std::max(0.0f, std::fabs(m) - kDeadPx), m);
    }
  });

  // Each pixel in the band moves with its nearest contour sample along the distance's normal:
  // out(p) = in(p + s n), bilinear. Not shiftAlongNormal's distance-space interpolation: its
  // choice of the tightest-bracketing pixel pair jumps between rows on slanted edges, which put
  // a 0.13 px row-to-row jitter (aliasing) into a smoothed slant; bilinear is smooth, exact on
  // the ramp and coverage-correct on a hard edge, and moved hair strands near the edge far less
  // (synth hair crop, truth as input: max change 0.38 -> 0.07).
  std::unique_ptr<float[]> res(new float[bd.size]);
  std::vector<int> tileList(size_t(bd.tw) * bd.th);
  for (size_t t = 0; t < tileList.size(); ++t) tileList[t] = bd.slot[t] < 0 ? -1 : int(t);
  tileList.erase(std::remove(tileList.begin(), tileList.end(), -1), tileList.end());
  std::vector<double> largest(tileList.size(), 0.0);
  std::vector<long long> changedIn(tileList.size(), 0);
  parallelFor(&wk, int(tileList.size()), 16, [&](int t0, int t1) {
    for (int t = t0; t < t1; ++t) {
      const int T = Band::kTile, bx = tileList[size_t(t)] % bd.tw * T, by = tileList[size_t(t)] / bd.tw * T;
      for (int y = by; y < std::min(by + T, h); ++y)
        for (int x = bx; x < std::min(bx + T, w); ++x) {
          const int k = bd.index(x, y);
          const size_t i = size_t(y) * w + x;
          res[size_t(k)] = in[i];
          if (bd.nearest[size_t(k)] < 0 || std::fabs(sd(x, y)) > band) continue;
          const float s = move[size_t(bd.nearest[size_t(k)])];
          if (s == 0.0f) continue;
          largest[size_t(t)] = std::max(largest[size_t(t)], double(std::fabs(s)));
          const int xl = x > 0 ? x - 1 : x, xr = x + 1 < w ? x + 1 : x;
          const int yu = y > 0 ? y - 1 : y, yd = y + 1 < h ? y + 1 : y;
          const double gx = xr > xl ? (sd(xr, y) - sd(xl, y)) / (xr - xl) : 0.0;
          const double gy = yd > yu ? (sd(x, yd) - sd(x, yu)) / (yd - yu) : 0.0;
          const double g = std::hypot(gx, gy);
          if (g < 0.5) continue;
          res[size_t(k)] = sampleBilinear(in, w, h, x + s * gx / g, y + s * gy / g);
        }
    }
  });
  parallelFor(&wk, int(tileList.size()), 16, [&](int t0, int t1) {  // after all reads of in
    for (int t = t0; t < t1; ++t) {
      const int T = Band::kTile, bx = tileList[size_t(t)] % bd.tw * T, by = tileList[size_t(t)] / bd.tw * T;
      for (int y = by; y < std::min(by + T, h); ++y)
        for (int x = bx; x < std::min(bx + T, w); ++x) {
          const float v = res[size_t(bd.index(x, y))];
          changedIn[size_t(t)] += v != in[size_t(y) * w + x];
          out[size_t(y) * w + x] = v;
        }
    }
  });
  for (size_t t = 0; t < tileList.size(); ++t) {
    st.largestShiftPx = std::max(st.largestShiftPx, largest[t]);
    st.pixelsChanged += changedIn[t];
  }
  if (stats) *stats = st;
}

}  // namespace mr
