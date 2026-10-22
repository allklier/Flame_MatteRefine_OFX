// Snap to Image (operation A6): against a slow, direct reference of the colour guided filter,
// known answers, the band's bit-identity, the synth scenes vs their truth, UHD timing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "ops/snap_to_image.h"
#include "synth.h"

using mr::SnapParams;
using mr::SnapStats;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

struct Scene {
  int w, h;
  std::vector<float> matte, rgb;
  Scene(int w, int h) : w(w), h(h), matte(size_t(w) * h, 0.0f), rgb(size_t(w) * h * 3, 0.0f) {}
};

std::vector<float> run(const Scene& s, const SnapParams& p, SnapStats* st = nullptr) {
  std::vector<float> out(s.matte.size());
  mr::snapToImage(s.matte.data(), s.rgb.data(), out.data(), s.w, s.h, p, st);
  return out;
}

// Distance of each pixel to the band's anchor as the op defines it: the nearest edge pixel,
// i.e. grey (0.05 < m < 0.95) with |central differences| summed > 0.04, or with a 4-neighbour
// on the other side of 0.5; brute force.
float slope(const std::vector<float>& m, int w, int h, int x, int y) {
  auto at = [&](int qx, int qy) {
    return m[size_t(std::min(std::max(qy, 0), h - 1)) * w + std::min(std::max(qx, 0), w - 1)];
  };
  return std::fabs(at(x + 1, y) - at(x - 1, y)) + std::fabs(at(x, y + 1) - at(x, y - 1));
}

std::vector<double> bandDistance(const std::vector<float>& m, int w, int h) {
  auto edge = [&](int x, int y) {
    const float v = m[size_t(y) * w + x];
    if (v > 0.05f && v < 0.95f && slope(m, w, h, x, y) > 0.04f) return true;
    const int nx[4] = {x - 1, x + 1, x, x}, ny[4] = {y, y, y - 1, y + 1};
    for (int k = 0; k < 4; ++k)
      if (nx[k] >= 0 && ny[k] >= 0 && nx[k] < w && ny[k] < h &&
          (m[size_t(ny[k]) * w + nx[k]] >= 0.5f) != (v >= 0.5f))
        return true;
    return false;
  };
  std::vector<std::pair<int, int>> edges;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (edge(x, y)) edges.emplace_back(x, y);
  std::vector<double> d(m.size(), 1e9);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double best = 1e18;
      for (const auto& e : edges)
        best = std::min(best, double(e.first - x) * (e.first - x) + double(e.second - y) * (e.second - y));
      d[size_t(y) * w + x] = best < 1e17 ? std::sqrt(best) : 1e9;
    }
  return d;
}

// The same, searching only R px around each pixel (1e9 beyond): for the large synth frames.
std::vector<double> localBandDistance(const std::vector<float>& m, int w, int h, int R) {
  std::vector<uint8_t> edge(m.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const float v = m[size_t(y) * w + x];
      bool e = v > 0.05f && v < 0.95f && slope(m, w, h, x, y) > 0.04f;
      const int nx[4] = {x - 1, x + 1, x, x}, ny[4] = {y, y, y - 1, y + 1};
      for (int k = 0; k < 4 && !e; ++k)
        e = nx[k] >= 0 && ny[k] >= 0 && nx[k] < w && ny[k] < h &&
            (m[size_t(ny[k]) * w + nx[k]] >= 0.5f) != (v >= 0.5f);
      edge[size_t(y) * w + x] = e;
    }
  std::vector<double> d(m.size(), 1e9);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double best = 1e18;
      for (int qy = std::max(0, y - R); qy <= std::min(h - 1, y + R); ++qy)
        for (int qx = std::max(0, x - R); qx <= std::min(w - 1, x + R); ++qx)
          if (edge[size_t(qy) * w + qx])
            best = std::min(best, double(qx - x) * (qx - x) + double(qy - y) * (qy - y));
      if (best < 1e17) d[size_t(y) * w + x] = std::sqrt(best);
    }
  return d;
}

// Solves A x = b (3x3) by Gaussian elimination with partial pivoting.
void solve3(double A[3][3], double b[3], double x[3]) {
  for (int c = 0; c < 3; ++c) {
    int piv = c;
    for (int r = c + 1; r < 3; ++r)
      if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
    std::swap(A[c], A[piv]);
    std::swap(b[c], b[piv]);
    for (int r = c + 1; r < 3; ++r) {
      const double f = A[r][c] / A[c][c];
      for (int k = c; k < 3; ++k) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  for (int r = 2; r >= 0; --r) {
    double s = b[r];
    for (int k = r + 1; k < 3; ++k) s -= A[r][k] * x[k];
    x[r] = s / A[r][r];
  }
}

// The whole operation, written directly from the paper (He et al. 2013, eq. 14-16 and the colour
// version) with plain loops over every window: guide clamped to 0-64 (NaN 0), the matte's
// contrast boosted around 0.5, a_k and b_k per window clipped to the frame, q = the mean of a
// and b over the windows containing the pixel, kappa = the windows' confidence, likewise averaged,
// blended into the input by the band weight, clamped; two passes, the second filtering the
// first's result. plain: one pass without kappa (the paper's filter as is).
std::vector<float> referencePass(const Scene& s, const std::vector<float>& pin, const SnapParams& p,
                                 bool plain) {
  const int w = s.w, h = s.h, r = std::max(1, int(std::lround(p.radiusPx)));
  const double eps = mr::snapEpsilon(p.detail);
  const double firm = 1.0 / (1.0 - 0.95 * p.fromPicture);
  auto g = [&](int x, int y, int c) {
    const float v = s.rgb[(size_t(y) * w + x) * 3 + c];
    return v >= 0.0f ? double(std::min(v, 64.0f)) : 0.0;
  };
  auto pm = [&](int x, int y) {
    return std::min(1.0, std::max(0.0, (pin[size_t(y) * w + x] - 0.5) * firm + 0.5));
  };
  std::vector<double> A(size_t(w) * h * 5);
  for (int ky = 0; ky < h; ++ky)
    for (int kx = 0; kx < w; ++kx) {
      double mg[3] = {0, 0, 0}, mp = 0, cnt = 0;
      for (int y = std::max(0, ky - r); y <= std::min(h - 1, ky + r); ++y)
        for (int x = std::max(0, kx - r); x <= std::min(w - 1, kx + r); ++x) {
          for (int c = 0; c < 3; ++c) mg[c] += g(x, y, c);
          mp += pm(x, y);
          ++cnt;
        }
      for (int c = 0; c < 3; ++c) mg[c] /= cnt;
      mp /= cnt;
      double S[3][3] = {}, cov[3] = {};
      for (int y = std::max(0, ky - r); y <= std::min(h - 1, ky + r); ++y)
        for (int x = std::max(0, kx - r); x <= std::min(w - 1, kx + r); ++x)
          for (int i = 0; i < 3; ++i) {
            cov[i] += (g(x, y, i) - mg[i]) * (pm(x, y) - mp) / cnt;
            for (int j = 0; j < 3; ++j) S[i][j] += (g(x, y, i) - mg[i]) * (g(x, y, j) - mg[j]) / cnt;
          }
      const double bright = std::max((mg[0] + mg[1] + mg[2]) / 3.0, 0.02);
      const double e = eps * bright * bright;
      const double var = (S[0][0] + S[1][1] + S[2][2]) / 3.0;
      for (int i = 0; i < 3; ++i) S[i][i] += e;
      double a[3];
      solve3(S, cov, a);
      double* o = &A[(size_t(ky) * w + kx) * 5];
      o[4] = plain ? 1.0 : var / (var + e);
      o[0] = a[0], o[1] = a[1], o[2] = a[2], o[3] = mp - a[0] * mg[0] - a[1] * mg[1] - a[2] * mg[2];
    }
  const std::vector<double> d = bandDistance(s.matte, w, h);
  std::vector<float> out(s.matte);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const double u = std::min(1.0, (p.bandPx - d[i]) / (0.5 * p.bandPx));
      if (u <= 0.0) continue;
      const double wt = u * u * (3 - 2 * u);
      double m[5] = {0, 0, 0, 0, 0}, cnt = 0;
      for (int ky = std::max(0, y - r); ky <= std::min(h - 1, y + r); ++ky)
        for (int kx = std::max(0, x - r); kx <= std::min(w - 1, x + r); ++kx) {
          for (int c = 0; c < 5; ++c) m[c] += A[(size_t(ky) * w + kx) * 5 + c];
          ++cnt;
        }
      const double q = (m[0] * g(x, y, 0) + m[1] * g(x, y, 1) + m[2] * g(x, y, 2) + m[3]) / cnt;
      const double v = s.matte[i] + wt * m[4] / cnt * (q - s.matte[i]);
      out[i] = float(std::min(1.0, std::max(0.0, v)));
    }
  return out;
}

std::vector<float> reference(const Scene& s, const SnapParams& p, bool plain = false) {
  if (plain) return referencePass(s, s.matte, p, true);
  return referencePass(s, referencePass(s, s.matte, p, false), p, false);
}

// A picture with a curved colour edge, texture, noise, a highlight above 1, and a matte near
// that edge: soft, offset, blocky in places.
Scene testScene(int w, int h, unsigned seed) {
  Scene s(w, h);
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, 0.02);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const double sd = 0.35 * h - std::hypot(x - 0.5 * w, y - 0.55 * h);  // > 0 inside
      const double a = std::min(1.0, std::max(0.0, 0.5 + sd / 2.0));
      const double fg[3] = {0.7 + 0.1 * std::sin(x * 0.3), 0.4, 0.25 + (x > w / 2 && y < h / 3 ? 2.5 : 0.0)};
      const double bg[3] = {0.2, 0.35 + 0.05 * std::cos(y * 0.2), 0.6};
      for (int c = 0; c < 3; ++c) s.rgb[i * 3 + c] = float(a * fg[c] + (1 - a) * bg[c] + noise(rng));
      const double off = 2.0 + 1.5 * std::sin(y * 0.15);  // a wobbly offset outward
      s.matte[i] = float(std::min(1.0, std::max(0.0, 0.5 + (sd + off) / 6.0)));
    }
  return s;
}

void testAgainstReference() {
  struct Case {
    double r, band, detail, from;
  } cases[] = {{1, 2.5, 0.5, 0.0}, {3, 6, 0.0, 0.5}, {4, 8, 1.0, 1.0}, {7, 12, 0.5, 0.5},
               {2, 70, 0.7, 0.3}};  // a band wider than a 64 px tile
  unsigned seed = 1;
  for (const Case& c : cases) {
    const Scene s = testScene(150, 110, seed++);  // 3 x 2 tiles
    SnapParams p;
    p.radiusPx = c.r, p.bandPx = c.band, p.detail = c.detail, p.fromPicture = c.from;
    SnapStats st;
    const std::vector<float> got = run(s, p, &st), want = reference(s, p);
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, double(std::fabs(got[i] - want[i])));
    std::printf("reference r %.0f band %.1f detail %.1f from %.1f: max diff %.2g, %lld changed\n", c.r,
                c.band, c.detail, c.from, worst, st.pixelsChanged);
    CHECK(worst < 1e-4);
    CHECK(st.pixelsChanged > 100);
  }
}

// Every pixel Band px or more from the 0.5 contour is bit-identical; so is a matte without one.
void testUntouchedOutsideBand() {
  const Scene s = testScene(150, 110, 7);
  SnapParams p;
  p.bandPx = 6.0;
  const std::vector<float> out = run(s, p);
  const std::vector<double> d = bandDistance(s.matte, s.w, s.h);
  long long outside = 0, changed = 0, inBandChanged = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (d[i] >= p.bandPx) {
      ++outside;
      changed += out[i] != s.matte[i];
    } else {
      inBandChanged += out[i] != s.matte[i];
    }
  }
  CHECK(outside > 5000 && changed == 0);
  CHECK(inBandChanged > 500);
  for (float fill : {0.0f, 1.0f, 0.3f}) {  // no edge (flat grey is none): nothing to snap
    Scene flat = testScene(100, 80, 3);
    std::fill(flat.matte.begin(), flat.matte.end(), fill);
    SnapStats st;
    CHECK(run(flat, SnapParams(), &st) == flat.matte);
    CHECK(st.bandPixels == 0);
  }
  SnapParams off = p;  // Band 0 = off
  off.bandPx = 0.0;
  CHECK(run(s, off) == s.matte);
}

void testInPlace() {
  const Scene s = testScene(150, 110, 9);
  const std::vector<float> sep = run(s, SnapParams());
  std::vector<float> inplace = s.matte;
  mr::snapToImage(inplace.data(), s.rgb.data(), inplace.data(), s.w, s.h, SnapParams());
  CHECK(inplace == sep);
}

// Known answer: a picture edge at x = 40 (hard: pixels < 40 foreground), the matte's soft edge
// (as wide as its offset) `off` px to the right. The matte's 0.5 crossing moves onto the
// picture's edge for a slight offset, part of the way for a larger one: the filter fits the
// matte per window, so pixels the matte wrongly calls foreground pull the result back and leave
// a fading tail about Radius px wide on the background side.
void testSnapsToEdge() {
  for (double off : {2.0, 4.0}) {
    Scene s(100, 40);
    for (int y = 0; y < s.h; ++y)
      for (int x = 0; x < s.w; ++x) {
        const size_t i = size_t(y) * s.w + x;
        const bool fg = x < 40;
        s.rgb[i * 3] = fg ? 0.8f : 0.1f, s.rgb[i * 3 + 1] = fg ? 0.5f : 0.6f, s.rgb[i * 3 + 2] = 0.2f;
        s.matte[i] = float(std::min(1.0, std::max(0.0, 0.5 - (x + 0.5 - 40.0 - off) / off)));
      }
    auto crossing = [&](const std::vector<float>& m, int y) {  // where row y falls through 0.5
      for (int x = 1; x < s.w; ++x) {
        const float a = m[size_t(y) * s.w + x - 1], b = m[size_t(y) * s.w + x];
        if (a >= 0.5f && b < 0.5f) return x - 1 + 0.5 + (a - 0.5) / (a - b);
      }
      return -1.0;
    };
    for (double from : {0.0, 0.5, 1.0}) {
      SnapParams p;
      p.fromPicture = from;
      const std::vector<float> out = run(s, p);
      const double before = crossing(s.matte, 20), after = crossing(out, 20);
      std::printf("snap to edge, offset %.0f px, from picture %.1f: 0.5 crossing %.2f -> %.2f (picture "
                  "edge 40.0), x 38-41: %.3f %.3f | %.3f %.3f\n", off, from, before, after,
                  out[20 * s.w + 38], out[20 * s.w + 39], out[20 * s.w + 40], out[20 * s.w + 41]);
      CHECK(after - 40.0 < (off == 2.0 ? 0.6 : off - 1.0));
      CHECK(out[20 * s.w + 39] > 0.99f);  // the foreground side stays solid
    }
  }
}

// Where the picture has no edge (flat colour plus grain below Detail's threshold), the plain
// guided filter would return the box-blurred matte; the op keeps the matte instead.
void testFlatPictureKeepsMatte() {
  Scene s(120, 60);
  std::mt19937 rng(5);
  std::normal_distribution<double> grain(0.0, 0.005);
  for (int y = 0; y < s.h; ++y)
    for (int x = 0; x < s.w; ++x) {
      const size_t i = size_t(y) * s.w + x;
      for (int c = 0; c < 3; ++c) s.rgb[i * 3 + c] = float(0.4 + grain(rng));
      s.matte[i] = float(std::min(1.0, std::max(0.0, 0.5 - (x + 0.5 - 60.0) / 3.0)));
    }
  SnapStats st;
  run(s, SnapParams(), &st);
  const std::vector<float> plain = reference(s, SnapParams(), true);
  double plainMax = 0;
  for (size_t i = 0; i < plain.size(); ++i) plainMax = std::max(plainMax, double(std::fabs(plain[i] - s.matte[i])));
  std::printf("flat picture (grain 1.25 %%): max change %.4f over %lld band px (the plain filter: %.3f)\n",
              st.maxChange, st.bandPixels, plainMax);
  CHECK(st.bandPixels > 1000);
  CHECK(st.maxChange < 0.03f);
  CHECK(plainMax > 0.1);
}

// A good matte stays: the picture is exactly the alpha composite of two colours (in linear light),
// the matte is that alpha. With the matte's own softness (From Picture 0) it is kept almost
// exactly; taking softness from the picture rebuilds it from the colours, close to it.
void testGoodMatteKept() {
  Scene s(120, 120);
  for (int y = 0; y < s.h; ++y)
    for (int x = 0; x < s.w; ++x) {
      const size_t i = size_t(y) * s.w + x;
      const double sd = 40.0 - std::hypot(x - 60.0, y - 60.0);
      const double a = std::min(1.0, std::max(0.0, 0.5 + sd / 2.5));
      const double fg[3] = {0.9, 0.5, 0.3}, bg[3] = {0.1, 0.6, 0.15};
      for (int c = 0; c < 3; ++c) s.rgb[i * 3 + c] = float(a * fg[c] + (1 - a) * bg[c]);
      s.matte[i] = float(a);
    }
  for (double from : {0.0, 0.5, 1.0}) {
    SnapParams p;
    p.fromPicture = from;
    SnapStats st;
    run(s, p, &st);
    std::printf("good matte, from picture %.1f: max change %.4f\n", from, st.maxChange);
    CHECK(st.maxChange < (from == 0.0 ? 0.01f : 0.05f));
  }
}

// Acute gaps in a blocky matte (12 px block averages, as synth "coarse"): a wedge whose blurred
// tip stays above 0.5 except a thin dark line, and a thin gap the blocks hide entirely. A band
// anchored on the 0.5 contour alone was a strip around that line: the wedge's sides beyond it
// stayed untouched and grey, a ring/rim where the band ended (old code: 172 such pixels at
// Band 6, 454 at Band 4) and a step at the band's border inside the gap. Now the grey, sloped
// pixels anchor the band too.
void testWedge() {
  const int w = 240, h = 160;
  Scene s(w, h);
  std::vector<float> truth(size_t(w) * h);
  auto wedgeDepth = [](int x, int y) { return (x - 40) * 18.0 / 200 - std::fabs(y - 50.0); };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double thin = (x - 60) * 4.0 / 180 - std::fabs(y - 120.0);  // > 0 inside a gap
      const double a = std::min(1.0, std::max(0.0, 0.5 - std::max(wedgeDepth(x, y), thin)));
      const size_t i = size_t(y) * w + x;
      truth[i] = float(a);
      const double fg[3] = {0.7 + 0.1 * std::sin(x * 0.2 + y * 0.1), 0.45, 0.3}, bg[3] = {0.25, 0.4, 0.6};
      for (int c = 0; c < 3; ++c) s.rgb[i * 3 + c] = float(a * fg[c] + (1 - a) * bg[c]);
    }
  const int cell = 12, cw = (w + cell - 1) / cell, ch = (h + cell - 1) / cell;
  std::vector<double> blk(size_t(cw) * ch, 0.0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) blk[size_t(y / cell) * cw + x / cell] += truth[size_t(y) * w + x] / (cell * cell);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double fx = std::min(std::max((x + 0.5) / cell - 0.5, 0.0), cw - 1.0);
      const double fy = std::min(std::max((y + 0.5) / cell - 0.5, 0.0), ch - 1.0);
      const int x0 = int(fx), y0 = int(fy), x1 = std::min(x0 + 1, cw - 1), y1 = std::min(y0 + 1, ch - 1);
      const double tx = fx - x0, ty = fy - y0;
      s.matte[size_t(y) * w + x] = float((blk[size_t(y0) * cw + x0] * (1 - tx) + blk[size_t(y0) * cw + x1] * tx) * (1 - ty) +
                                         (blk[size_t(y1) * cw + x0] * (1 - tx) + blk[size_t(y1) * cw + x1] * tx) * ty);
    }
  for (double band : {4.0, 6.0, 12.0}) {
    SnapParams p;
    p.bandPx = band;
    const std::vector<float> out = run(s, p);
    long long untouched = 0;
    double stepIn = 0, stepOut = 0;  // largest vertical step strictly inside the wedge
    for (int y = 0; y + 1 < 100; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (wedgeDepth(x, y) > 1.5) untouched += out[i] == s.matte[i] && s.matte[i] > 0.2f;
        if (wedgeDepth(x, y) > 1.5 && wedgeDepth(x, y + 1) > 1.5) {
          stepIn = std::max(stepIn, double(std::fabs(s.matte[i + w] - s.matte[i])));
          stepOut = std::max(stepOut, double(std::fabs(out[i + w] - out[i])));
        }
      }
    std::printf("wedge, band %.0f: gap pixels left untouched though > 0.2: %lld; largest step inside "
                "the gap %.3f -> %.3f\n", band, untouched, stepIn, stepOut);
    CHECK(untouched == 0);
    CHECK(stepOut < 0.05);
  }
}

// Epsilon scales with the window's brightness: the same picture 5x brighter gives the same matte.
void testExposureInvariant() {
  Scene s = testScene(150, 110, 11);
  const std::vector<float> a = run(s, SnapParams());
  for (float& v : s.rgb) v *= 5.0f;
  for (float& v : s.rgb) v = std::min(v, 64.0f);  // stays under the guide's clamp
  const std::vector<float> b = run(s, SnapParams());
  double worst = 0;
  for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, double(std::fabs(a[i] - b[i])));
  std::printf("exposure x5: max diff %.2g\n", worst);
  CHECK(worst < 1e-3);
}

// NaN, inf and huge picture values don't spread: the output stays finite and in 0-1.
void testBadPicture() {
  Scene s = testScene(150, 110, 13);
  s.rgb[(size_t(50) * s.w + 40) * 3] = std::numeric_limits<float>::quiet_NaN();
  s.rgb[(size_t(60) * s.w + 100) * 3 + 1] = std::numeric_limits<float>::infinity();
  s.rgb[(size_t(30) * s.w + 75) * 3 + 2] = 1e30f;
  s.rgb[(size_t(80) * s.w + 30) * 3 + 2] = -5.0f;
  int bad = 0;
  for (float v : run(s, SnapParams())) bad += !(v >= 0.0f && v <= 1.0f);
  CHECK(bad == 0);
}

// The synth scenes against their truth (CLAUDE.md decision 21), at 1920x1080.
void testSynth() {
  const int w = 1920, h = 1080;
  const SnapParams p;
  {
    const synth::Scene s = synth::make("coarse", w, h);
    std::vector<float> out(s.matte.size());
    SnapStats st;
    mr::snapToImage(s.matte.data(), s.rgb.data(), out.data(), w, h, p, &st);
    double sumB = 0, sumA = 0, maxB = 0, maxA = 0;
    long long band = 0, outside = 0, outsideChanged = 0;
    // The band: pixels the op may change (weight > 0), from a local brute-force distance.
    const std::vector<double> bd = localBandDistance(s.matte, w, h, int(std::ceil(p.bandPx)) + 1);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (bd[i] < p.bandPx) {
          ++band;
          const double eb = std::fabs(s.matte[i] - s.truth[i]), ea = std::fabs(out[i] - s.truth[i]);
          sumB += eb, sumA += ea, maxB = std::max(maxB, eb), maxA = std::max(maxA, ea);
        } else {
          ++outside;
          outsideChanged += out[i] != s.matte[i];
        }
      }
    std::printf("coarse 1920x1080: band %lld px, error vs truth mean %.4f -> %.4f, max %.3f -> %.3f; "
                "outside the band %lld px, %lld changed\n",
                band, sumB / band, sumA / band, maxB, maxA, outside, outsideChanged);
    CHECK(band == st.bandPixels);
    CHECK(outsideChanged == 0);
    CHECK(sumA < 0.65 * sumB);
    // The tail outside the edge: what is left where the truth is background.
    double tailB = 0, tailA = 0;
    for (size_t i = 0; i < out.size(); ++i)
      if (s.truth[i] == 0.0f) tailB += s.matte[i], tailA += out[i];
    std::printf("coarse background tail (sum where the truth is 0): %.0f -> %.0f\n", tailB, tailA);
    CHECK(tailA < 0.6 * tailB);  // one pass (0.1.x): 0.68
  }
  {
    const synth::Scene s = synth::make("fgcolor", w, h);
    std::vector<float> out(s.matte.size());
    SnapStats st;
    mr::snapToImage(s.matte.data(), s.rgb.data(), out.data(), w, h, p, &st);
    double sum = 0;
    long long over = 0;
    for (size_t i = 0; i < out.size(); ++i) {
      const double c = std::fabs(out[i] - s.matte[i]);
      sum += c;
      over += c > 0.05;
    }
    std::printf("fgcolor 1920x1080 (good matte): %lld band px, %lld changed, mean change over the band "
                "%.4f, max %.3f, %lld px changed by > 0.05\n",
                st.bandPixels, st.pixelsChanged, sum / st.bandPixels, st.maxChange, over);
    CHECK(sum / st.bandPixels < 0.01);
  }
}

// No seam where the band ends: the change (out - in) of neighbouring pixels across the band's
// border differs little, on the coarse scene where the change inside is large.
void testNoSeam() {
  const int w = 960, h = 540;
  const synth::Scene s = synth::make("coarse", w, h);
  SnapParams p;
  p.bandPx = 6.0;
  std::vector<float> out(s.matte.size());
  mr::snapToImage(s.matte.data(), s.rgb.data(), out.data(), w, h, p);
  double worstBorder = 0, worstInside = 0;
  const std::vector<double> bd = localBandDistance(s.matte, w, h, 8);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x + 1 < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const double c0 = out[i] - s.matte[i], c1 = out[i + 1] - s.matte[i + 1];
      if (c0 == 0.0 && c1 == 0.0) continue;
      const double d0 = bd[i], d1 = bd[i + 1];
      const bool border = (d0 >= p.bandPx) != (d1 >= p.bandPx);
      (border ? worstBorder : worstInside) = std::max(border ? worstBorder : worstInside, std::fabs(c1 - c0));
    }
  std::printf("seam: largest step in the change across the band border %.4f (inside the band %.3f)\n",
              worstBorder, worstInside);
  CHECK(worstBorder < 0.05);
}

void testTiming() {
  const int w = 3840, h = 2160;
  for (const char* name : {"coarse", "fgcolor"}) {
  const synth::Scene s = synth::make(name, w, h);
  std::vector<float> out(s.matte.size());
  SnapStats st;
  double best = 1e9;
  for (int k = 0; k < 3; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    mr::snapToImage(s.matte.data(), s.rgb.data(), out.data(), w, h, SnapParams(), &st);
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::printf("UHD snap to image (defaults, %s scene): %.1f ms (best of 3), %lld band px\n", name, best,
              st.bandPixels);
  CHECK(st.pixelsChanged > 0);
  CHECK(best < 60.0);  // target ~30 ms; slack for a loaded machine
  }
}

}  // namespace

int main() {
  testAgainstReference();
  testUntouchedOutsideBand();
  testInPlace();
  testSnapsToEdge();
  testFlatPictureKeepsMatte();
  testGoodMatteKept();
  testWedge();
  testExposureInvariant();
  testBadPicture();
  testSynth();
  testNoSeam();
  testTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
