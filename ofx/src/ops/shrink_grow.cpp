#include "ops/shrink_grow.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {

namespace {

// Profiles up to this far each side of the contour move rigidly (edges up to 12 px wide).
constexpr double kProfilePx = 6.0;
// Larger amounts move in passes: the band, and with it the cost, grows with the step.
constexpr double kMaxStepPx = 4.0;

// out = the min (erode) or max (dilate) of m over the (2r + 1)^2 box, separable, on threads;
// straight loops over a padded row so they vectorise (12-25 ms at UHD before). Beyond the
// frame the nearest pixel counts (the frame border is no edge). out may be m.
template <bool kErode>
void boxFilter(const uint8_t* m, uint8_t* out, int w, int h, int r) {
  std::unique_ptr<uint8_t[]> rowsBuf(new uint8_t[size_t(w) * h]);
  uint8_t* rows = rowsBuf.get();
  // Plain locals in the loops: a uint8_t store may alias anything reached through a pointer
  // (the lambda's captures), which kept the loops from vectorising.
  parallelFor(h, 16, [=](int y0, int y1) {
    const int W = w, R = r;
    std::vector<uint8_t> padBuf(size_t(W) + 2 * R);
    uint8_t* pad = padBuf.data();
    for (int y = y0; y < y1; ++y) {
      const uint8_t* in = m + size_t(y) * W;
      uint8_t* o = rows + size_t(y) * W;
      std::copy(in, in + W, pad + R);
      std::fill(pad, pad + R, in[0]);
      std::fill(pad + R + W, pad + 2 * R + W, in[W - 1]);
      std::copy(in, in + W, o);
      for (int d = 0; d <= 2 * R; ++d) {
        const uint8_t* q = pad + d;
        for (int x = 0; x < W; ++x) o[x] = kErode ? std::min(o[x], q[x]) : std::max(o[x], q[x]);
      }
    }
  });
  parallelFor(h, 16, [=](int y0, int y1) {
    const int W = w, H = h, R = r;
    for (int y = y0; y < y1; ++y) {
      uint8_t* o = out + size_t(y) * W;
      std::copy(rows + size_t(y) * W, rows + size_t(y) * W + W, o);
      for (int d = -R; d <= R; ++d) {
        const uint8_t* q = rows + size_t(std::min(std::max(y + d, 0), H - 1)) * W;
        for (int x = 0; x < W; ++x) o[x] = kErode ? std::min(o[x], q[x]) : std::max(o[x], q[x]);
      }
    }
  });
}

}  // namespace

void parallelFor(int n, int minPerThread, const std::function<void(int, int)>& fn) {
  const int t = std::max(1, std::min({int(std::thread::hardware_concurrency()), 32, n / minPerThread}));
  if (t == 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> pool;
  for (int i = 0; i < t; ++i) pool.emplace_back(fn, n * i / t, n * (i + 1) / t);
  for (auto& th : pool) th.join();
}

std::unique_ptr<float[]> signedEdgeDistanceThreaded(const float* in, int w, int h, float level,
                                                    double band) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  Mask below(n);
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) below[i] = in[i] < level;
  });
  // Contour pixels (a 4-neighbour on the other side), row-major, found in row bands on threads.
  constexpr int kRows = 64;
  const int bands = (h + kRows - 1) / kRows;
  std::vector<std::vector<int>> found(size_t(std::max(bands, 0)));
  parallelFor(bands, 1, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b)
      for (int y = b * kRows; y < std::min(h, (b + 1) * kRows); ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          if ((x > 0 && below[i - 1] != below[i]) || (x + 1 < w && below[i + 1] != below[i]) ||
              (y > 0 && below[i - w] != below[i]) || (y + 1 < h && below[i + w] != below[i]))
            found[b].push_back(int(i));
        }
  });
  std::vector<int> edge;
  for (const auto& f : found) edge.insert(edge.end(), f.begin(), f.end());
  // The disc of morphology.cpp distanceWithin (radius band + 1.5), as a half-width per row.
  const double r = band + 1.5;
  const int ri = int(std::floor(r));
  std::vector<int> half(size_t(2 * ri + 1));
  long long disc = 0;
  for (int dy = -ri; dy <= ri; ++dy) {
    half[size_t(dy + ri)] = int(std::floor(std::sqrt(std::max(0.0, r * r - double(dy) * dy))));
    disc += 2 * half[size_t(dy + ri)] + 1;
  }
  if (double(edge.size()) * double(disc) > 12.0 * double(n)) {
    const std::vector<float> sd = signedEdgeDistance(in, w, h, level, band);
    std::unique_ptr<float[]> out(new float[n]);
    std::copy(sd.begin(), sd.end(), out.get());
    return out;
  }
  std::vector<ContourFoot> feet(edge.size());
  parallelFor(int(edge.size()), 256, [&](int e0, int e1) {
    for (int e = e0; e < e1; ++e) feet[size_t(e)] = contourFoot(in, below, w, h, level, edge[size_t(e)]);
  });
  // Each pixel measures to the contour pixels on the other side only (signedEdgeDistance: above
  // pixels to the contour near below pixels, and the reverse); squared, as distanceWithin.
  constexpr float kFar = 1e20f;
  std::unique_ptr<float[]> d2(new float[n]);
  parallelFor(h, 16, [&](int y0, int y1) { std::fill(&d2[size_t(y0) * w], &d2[size_t(y1) * w], kFar); });
  parallelFor(bands, 1, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b) {
      const int y0 = b * kRows, y1 = std::min(h, (b + 1) * kRows);
      const auto first = std::lower_bound(edge.begin(), edge.end(), std::max(0, y0 - ri) * w);
      const auto last = std::lower_bound(edge.begin(), edge.end(), std::min(h, y1 + ri) * w);
      for (auto it = first; it != last; ++it) {
        const int q = *it, qx = q % w, qy = q / w;
        const bool qBelow = below[size_t(q)] != 0;
        const ContourFoot& f = feet[size_t(it - edge.begin())];
        const bool line = f.nx != 0.0f || f.ny != 0.0f;
        for (int py = std::max(y0, qy - ri); py < std::min(y1, qy + ri + 1); ++py) {
          const int hw = half[size_t(py - qy + ri)];
          for (int px = std::max(0, qx - hw); px <= std::min(w - 1, qx + hw); ++px) {
            const size_t j = size_t(py) * w + px;
            if ((below[j] != 0) == qBelow) continue;
            const float ox = float(px - qx), oy = float(py - qy), grid = ox * ox + oy * oy;
            float v = grid;
            if (line) {
              const float vx = ox - f.fx, vy = oy - f.fy;
              const float perp = vx * f.nx + vy * f.ny;
              const float along = std::max(0.0f, std::fabs(vy * f.nx - vx * f.ny) - 0.5f);
              v = std::min(grid, perp * perp + along * along);
            }
            d2[j] = std::min(d2[j], v);
          }
        }
      }
    }
  });
  const float cap = float(band + 1.0);
  parallelFor(h, 16, [&](int y0, int y1) {  // in place: d2 becomes sd
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
      const float d = d2[i] < kFar ? std::sqrt(d2[i]) : kFar;
      d2[i] = below[i] ? -std::min(d, cap) : std::min(d, cap);
    }
  });
  return d2;
}

bool edgeNormal(const float* sd, int w, int h, int x, int y, double& nx, double& ny) {
  const size_t i = size_t(y) * w + x;
  double g[2];
  for (int axis = 0; axis < 2; ++axis) {
    const bool lo = axis ? y > 0 : x > 0, hi = axis ? y + 1 < h : x + 1 < w;
    const size_t step = axis ? size_t(w) : 1;
    const double dl = lo ? sd[i] - sd[i - step] : 0.0, dr = hi ? sd[i + step] - sd[i] : 0.0;
    g[axis] = lo && hi ? (dl + dr) / 2 : dl + dr;
  }
  double len = std::hypot(g[0], g[1]);
  if (len < 0.5) {  // a ridge: the two sides cancel, either side's direction will do
    for (int axis = 0; axis < 2; ++axis) {
      const bool lo = axis ? y > 0 : x > 0, hi = axis ? y + 1 < h : x + 1 < w;
      const size_t step = axis ? size_t(w) : 1;
      const double dl = lo ? sd[i] - sd[i - step] : 0.0, dr = hi ? sd[i + step] - sd[i] : 0.0;
      g[axis] = std::fabs(dl) >= std::fabs(dr) ? dl : dr;
    }
    len = std::hypot(g[0], g[1]);
    if (len < 0.5) return false;
  }
  nx = g[0] / len;
  ny = g[1] / len;
  return true;
}

float profileAt(const float* in, const float* sd, int w, int h, int x, int y, double nx, double ny,
                double target) {
  const double s0 = sd[size_t(y) * w + x], move = target - s0;
  double px = x + move * nx, py = y + move * ny;
  // Past the middle of a thin part (or gap) the distance falls again, and the window would read
  // the far side or the next strand: then walk the normal in half pixels and read where the
  // distance reaches the target, or at the middle (the part's peak) if it never does.
  if (std::fabs(sampleBilinear(sd, w, h, px, py) - target) > 0.75) {
    const double dir = move > 0 ? 1.0 : -1.0;
    const int steps = int(std::ceil(2.0 * std::fabs(move)));
    double best = s0, bestT = 0.0;
    for (int k = 1; k <= steps; ++k) {
      const double t = std::min(0.5 * k, std::fabs(move));
      const double v = sampleBilinear(sd, w, h, x + dir * t * nx, y + dir * t * ny);
      if (dir * (v - best) > 0) best = v, bestT = t;
      if (dir * (v - target) >= 0) break;
    }
    if (dir * (best - target) < 0) target = best;
    px = x + dir * bestT * nx;
    py = y + dir * bestT * ny;
  }
  const int cx = int(std::lround(px)), cy = int(std::lround(py));
  // 3 px across the normal's main axis and 7 along the other: near an axis-aligned edge the 3x3
  // pixels' sd lie ~0.8 px apart, and linear interpolation across the kink at the end of a ramp
  // was off by up to 0.065 (3 px ramp); the rows along the edge fill in between (0.03).
  const bool alongY = std::fabs(nx) >= std::fabs(ny);
  const int rx = alongY ? 1 : 3, ry = alongY ? 3 : 1;
  double loD = -1e30, hiD = 1e30;
  float loV = 0.0f, hiV = 0.0f;
  for (int dy = -ry; dy <= ry; ++dy)
    for (int dx = -rx; dx <= rx; ++dx) {
      const int qx = std::min(std::max(cx + dx, 0), w - 1), qy = std::min(std::max(cy + dy, 0), h - 1);
      const size_t j = size_t(qy) * w + qx;
      if (sd[j] <= target && sd[j] > loD) loD = sd[j], loV = in[j];
      if (sd[j] >= target && sd[j] < hiD) hiD = sd[j], hiV = in[j];
    }
  if (loD > -1e29 && hiD < 1e29)
    return hiD > loD ? float(loV + (hiV - loV) * (target - loD) / (hiD - loD)) : loV;
  return loD > -1e29 ? loV : hiD < 1e29 ? hiV : in[size_t(y) * w + x];
}

bool isContour(const float* in, int w, int h, int x, int y) {
  const size_t i = size_t(y) * w + x;
  const bool a = in[i] >= 0.5f;
  return (x > 0 && (in[i - 1] >= 0.5f) != a) || (x + 1 < w && (in[i + 1] >= 0.5f) != a) ||
         (y > 0 && (in[i - w] >= 0.5f) != a) || (y + 1 < h && (in[i + w] >= 0.5f) != a);
}

double contourSlope(const float* in, const float* sd, int w, int h, int x, int y) {
  const size_t i = size_t(y) * w + x;
  const bool above = in[i] >= 0.5f;
  double bestDs = 0.0, slope = 0.0;
  const int nb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  for (const auto& o : nb) {
    const int qx = x + o[0], qy = y + o[1];
    if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
    const size_t j = size_t(qy) * w + qx;
    const double ds = std::fabs(double(sd[i]) - sd[j]);
    if ((in[j] >= 0.5f) != above && ds > bestDs) {
      bestDs = ds;
      slope = std::fabs(double(in[i]) - in[j]) / ds;
    }
  }
  return bestDs < 0.05 ? 0.0 : slope;
}

double footMean(const float* in, const float* m, int k, int w, int h, double fx, double fy, double* mean) {
  double sum = 0.0;
  for (int j = 0; j < k; ++j) mean[j] = 0.0;
  for (int qy = std::max(0, int(std::floor(fy)) - 1); qy <= std::min(h - 1, int(std::floor(fy)) + 2); ++qy)
    for (int qx = std::max(0, int(std::floor(fx)) - 1); qx <= std::min(w - 1, int(std::floor(fx)) + 2); ++qx) {
      const float* v = m + (size_t(qy) * w + qx) * k;
      if (!isContour(in, w, h, qx, qy) || v[0] < 0.0f) continue;
      const double t = std::max(0.0, 1.5 - std::fabs(qx - fx)) * std::max(0.0, 1.5 - std::fabs(qy - fy));
      sum += t;
      for (int j = 0; j < k; ++j) mean[j] += t * v[j];
    }
  if (sum > 0.0)
    for (int j = 0; j < k; ++j) mean[j] /= sum;
  return sum;
}

// Each pass: the body = the matte without its strands. Parts of the >= 0.5 area that an opening
// by the 3x3 box (radius 1.5) removes, under ~3 px wide, are strands: detail that keeps its
// shape (moving them too made every strand of hair above 0.5 a 2 x Amount wide band, and their
// contours pulled the head's edge into lumps at the roots). out(p) = the body's profile at
// sd(p) + step along the normal (edge_field shiftAlongNormal's resampling in distance space).
// The step is full within kProfile + |step| of the contour and tapers to 0 over |step| + 1 px
// beyond, so detail inside near the edge (a grey face) is carried along without a seam and the
// mapping stays monotone. Grow: max with the input (strands stay, nothing gets lower); shrink:
// min with the input, except where the matte is more than the body (strands, hair).
void shrinkGrow(const float* in, float* out, int w, int h, const ShrinkGrowParams& p,
                ShrinkGrowStats* stats) {
  ShrinkGrowStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  const double amount = p.amountPx;
  const int passes = amount == 0.0 ? 0 : int(std::ceil(std::fabs(amount) / kMaxStepPx));
  const double step = passes ? amount / passes : 0.0, a = std::fabs(step);
  const double rigid = kProfilePx + a, band = rigid + a + 1.0;
  // Each pass reads src and writes dst; the last one writes out unless out is in.
  const float* src = in;
  std::unique_ptr<float[]> prev, next, body(passes ? new float[n] : nullptr),
      falloffs(passes ? new float[n] : nullptr);
  std::unique_ptr<uint8_t[]> solid(passes ? new uint8_t[n] : nullptr), core(passes ? new uint8_t[n] : nullptr);
  for (int pass = 0; pass < passes; ++pass) {
    const bool last = pass + 1 == passes;
    if (!(last && out != in)) next.reset(new float[n]);
    float* dst = last && out != in ? out : next.get();
    // The opening: erode by the 3x3 box, dilate back; and 1 px more, since the opening also
    // takes single staircase pixels off a smooth contour (a strand keeps a stub that short).
    parallelFor(h, 16, [&](int y0, int y1) {
      for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) solid[i] = src[i] >= 0.5f;
    });
    boxFilter<true>(solid.get(), core.get(), w, h, 1);
    boxFilter<false>(core.get(), core.get(), w, h, 2);
    parallelFor(h, 16, [&](int y0, int y1) {
      for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) body[i] = solid[i] && !core[i] ? 0.0f : src[i];
    });
    const std::unique_ptr<float[]> sdBuf = signedEdgeDistanceThreaded(body.get(), w, h, 0.5f, band);
    const float* sd = sdBuf.get();
    // Outside the contour the body's profile ends where the edge's own falloff does (0.5 /
    // its slope at the contour, + 1 px) and only falls going out (no pixel more than any nearer
    // it along the normal): so hair below 0.5 next to the edge stays detail and is not carried
    // along (it was, as a second copy Amount px out). Below 0.5: sd stays as it is.
    parallelFor(h, 16, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x)
          if (isContour(body.get(), w, h, x, y)) {
            const double slope = contourSlope(body.get(), sd, w, h, x, y);
            falloffs[size_t(y) * w + x] = slope > 0.0 ? float(0.5 / slope) : -1.0f;
          }
    });
    std::unique_ptr<float[]> mono(new float[n]);
    parallelFor(h, 16, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          double v = body[i], nx, ny, falloff;
          if (v > 0.0 && sd[i] < 0.0 && sd[i] > -band && edgeNormal(sd, w, h, x, y, nx, ny)) {
            if (footMean(body.get(), falloffs.get(), 1, w, h, x - sd[i] * nx, y - sd[i] * ny, &falloff) > 0.0 &&
                -sd[i] > falloff + 1.0)
              v = 0.0;
            for (double t = 0.5; t < -sd[i] && v > 0.0; t += 0.5)
              v = std::min(v, double(sampleBilinear(body.get(), w, h, x + t * nx, y + t * ny)));
          }
          mono[i] = float(v);
        }
    });
    body.swap(mono);
    parallelFor(h, 16, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          dst[i] = src[i];
          const double s = sd[i], k = std::min(1.0, (band - std::fabs(s)) / (a + 1.0));
          double nx, ny;
          if (k <= 0.0 || !edgeNormal(sd, w, h, x, y, nx, ny)) continue;
          const float v = profileAt(body.get(), sd, w, h, x, y, nx, ny, s + step * k);
          dst[i] = step > 0 ? std::max(src[i], v) : std::min(src[i], std::max(v, src[i] - body[i]));
        }
    });
    if (!last) {
      prev.swap(next);
      src = prev.get();
    }
  }
  const float* result = passes == 0 ? in : out != in ? out : next.get();
  std::atomic<long long> changed(0);
  parallelFor(h, 16, [&](int y0, int y1) {
    long long c = 0;
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) c += result[i] != in[i];
    changed += c;
    if (result != out) std::copy(result + size_t(y0) * w, result + size_t(y1) * w, out + size_t(y0) * w);
  });
  st.pixelsChanged = changed;
  st.passes = passes;
  if (stats) *stats = st;
}

}  // namespace mr
