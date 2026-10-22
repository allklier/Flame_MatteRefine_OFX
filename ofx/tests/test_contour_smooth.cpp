// Contour Smooth (A4, CLAUDE.md "Plugin 1 operations"): wiggle off, profile, shape, thin
// features and everything away from the edge kept; against a brute-force reference; the synth
// "chatter" scene against its truth; UHD timing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "edge_reference.h"
#include "ops/contour_smooth.h"
#include "ops/edge_field.h"
#include "ops/morphology.h"
#include "synth.h"

using mr::ContourSmoothParams;
using mr::ContourSmoothStats;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

using Img = std::vector<float>;

Img run(const Img& in, int w, int h, const ContourSmoothParams& p = ContourSmoothParams(),
        ContourSmoothStats* st = nullptr) {
  Img out(in.size());
  mr::contourSmooth(in.data(), out.data(), w, h, p, st);
  return out;
}

double clamp01(double v) { return std::min(1.0, std::max(0.0, v)); }

double maxDiff(const Img& a, const Img& b) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
  return m;
}

// A straight edge at a slant (x = x0 + slope * y, inside to the left), 2 px soft, its line
// displaced by wiggle(y) px along the normal.
template <class Wiggle>
Img slantedEdge(int w, int h, double slope, Wiggle wiggle) {
  Img m(size_t(w) * h);
  const double nrm = std::hypot(1.0, slope);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      m[size_t(y) * w + x] = float(clamp01(0.5 + ((w * 0.4 + slope * y - x) / nrm + wiggle(y)) / 2.0));
  return m;
}

// The contour's x where row y crosses 0.5 (the edges here run top to bottom).
double crossing(const Img& m, int w, int y) {
  for (int x = 1; x < w; ++x) {
    const float a = m[size_t(y) * w + x - 1], b = m[size_t(y) * w + x];
    if (a >= 0.5f && b < 0.5f) return x - 1 + (a - 0.5) / (a - b);
  }
  return -1.0;
}

// Wiggle off and no aliasing: a slanted edge with a wiggle comes out close to the straight edge
// (the largest difference from the ideal, as test_solid_core coreValueError) and its contour
// line's row-to-row steps vs the ideal's (a staircase or jitter would show there).
// +-0.5 px at 24 px with the defaults; +-1 px at 12 px tilts the contour by up to 28 deg, more
// than the default Corner angle (20): it needs Corner 45.
void testWiggleRemoved() {
  const int w = 700, h = 300;
  for (int steep = 0; steep < 2; ++steep)
    for (double slope : {0.0, 0.11, 0.3, 0.81}) {
      const double lambda = steep ? 12.0 : 24.0, amp = steep ? 1.0 : 0.5;
      const Img in = slantedEdge(w, h, slope, [&](int y) { return amp * std::sin(2 * M_PI * y / lambda); });
      const Img ideal = slantedEdge(w, h, slope, [](int) { return 0.0; });
      ContourSmoothParams p;
      if (steep) p.cornerDeg = 45.0;
      const Img out = run(in, w, h, p);
      double before = 0, after = 0, lineBefore = 0, lineAfter = 0, stepBefore = 0, stepAfter = 0;
      for (int y = 30; y < h - 30; ++y) {
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          before = std::max(before, double(std::fabs(in[i] - ideal[i])));
          after = std::max(after, double(std::fabs(out[i] - ideal[i])));
        }
        const double c = crossing(ideal, w, y);
        lineBefore = std::max(lineBefore, std::fabs(crossing(in, w, y) - c));
        lineAfter = std::max(lineAfter, std::fabs(crossing(out, w, y) - c));
        const double dc = crossing(ideal, w, y + 1) - c;
        stepBefore = std::max(stepBefore, std::fabs(crossing(in, w, y + 1) - crossing(in, w, y) - dc));
        stepAfter = std::max(stepAfter, std::fabs(crossing(out, w, y + 1) - crossing(out, w, y) - dc));
      }
      std::printf("slope %.2f, wiggle +-%.1f px at %2.0f px%s: max error vs straight %.3f -> %.3f, line off by "
                  "%.2f -> %.2f px, row-to-row step error %.3f -> %.3f px\n", slope, amp, lambda,
                  steep ? " (Corner 45)" : "", before, after, lineBefore, lineAfter, stepBefore, stepAfter);
      CHECK(after < 0.65 * before);
      CHECK(lineAfter < (steep ? 0.65 : 0.5) * lineBefore);
      // OPEN: the row-to-row jitter is not reduced on slants (per-sample foot scatter reaches the
      // pixels); this only guards that it gets no worse.
      CHECK(stepAfter < 1.25 * stepBefore);
    }
}

// Clean edges are kept: straight slants, discs (a plain blur of the distance would shrink an
// r = 15 disc by sigma^2 / 2r ~ 4.8 px at the default sigma 12; the quadratic fit doesn't).
void testCleanKept() {
  const int w = 400, h = 400;
  for (double slope : {0.11, 0.3, 0.81}) {
    const Img in = slantedEdge(w, h, slope, [](int) { return 0.0; });
    const double d = maxDiff(run(in, w, h), in);
    std::printf("clean slant %.2f: max change %.4f\n", slope, d);
    CHECK(d < 0.01);
  }
  for (double rad : {15.0, 40.0, 120.0}) {
    Img in(size_t(w) * h);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        in[size_t(y) * w + x] = float(clamp01(0.5 + (rad - std::hypot(x - 200.3, y - 200.6)) / 2.0));
    const double d = maxDiff(run(in, w, h), in);
    std::printf("clean disc r %.0f: max change %.4f\n", rad, d);
    CHECK(d < (rad < 20 ? 0.06 : 0.03));
  }
}

// Corners are shape: an antialiased square, axis-aligned and rotated, with a 90 deg notch cut
// into one side; its corners (convex and concave) stay.
void testCornersKept() {
  const int w = 400, h = 400;
  for (double rot : {0.0, 0.5}) {
    Img in(size_t(w) * h);
    const double c = std::cos(rot), s = std::sin(rot);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double u = c * (x - 200.2) + s * (y - 199.7), v = -s * (x - 200.2) + c * (y - 199.7);
        double sd = 90 - std::max(std::fabs(u), std::fabs(v));                  // square, > 0 inside
        sd = std::min(sd, std::max(std::fabs(u) - 20, std::fabs(v - 90) - 25));  // notch
        in[size_t(y) * w + x] = float(clamp01(0.5 + sd / 2.0));
      }
    const double d = maxDiff(run(in, w, h), in);
    std::printf("square rotated %.1f rad with a notch: max change %.4f\n", rot, d);
    CHECK(d < 0.05);
  }
}

// Thin features don't move: a 1 px strand at 0.9 and a 2.5 px line (they have an Edge At
// contour), a 2 px gap in a solid; hair below Edge At has no contour at all. All bit-identical,
// as are the clean solid's straight edge (its moves are under the 0.05 px dead zone).
// Then a 0.4 strand running alongside the solid's edge, 2-4 px off: it tilts the edge's foot
// points by a few hundredths of a px, under the 0.05 px dead zone (before it, resampling for
// those moves changed the strand by up to 0.2). A strand that merges into the edge is part of
// its profile and is smoothed with it.
void testThinUntouched() {
  const int w = 300, h = 300;
  for (int touching = 0; touching < 2; ++touching) {
    Img in(size_t(w) * h, 0.0f);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double wig = std::sin(2 * M_PI * y / 12.0);
        auto line = [&](double cx, double half, double op) {
          return op * clamp01(0.5 + (half - std::fabs(x - cx - 0.2 * y - wig)));
        };
        // The 0.4 strand: far away, or running alongside the solid's edge, 2-4 px from it.
        const double strand = touching ? (x < 158 ? line(120, 0.7, 0.4) : 0.0) : line(15, 0.7, 0.4);
        double v = std::max({line(40, 0.5, 0.9), line(80, 1.25, 1.0), strand});
        if (x > 160) v = std::min(1.0, 1.0 - line(230, 1.0, 1.0));  // a solid with a 2 px gap
        in[size_t(y) * w + x] = float(v);
      }
    ContourSmoothStats st;
    const Img out = run(in, w, h, ContourSmoothParams(), &st);
    const double d = maxDiff(out, in);
    std::printf("thin features%s: %lld samples, %lld px changed, max change %.3f\n",
                touching ? ", a 0.4 strand touching an edge" : "", st.contourSamples, st.pixelsChanged, d);
    CHECK(st.contourSamples > 1000);
    CHECK(touching ? d < 0.03 : out == in);
  }
}

// A strand below Edge At inside the band of a moving edge moves with it (resampled along the
// edge's normal) and changes no more than that move makes it.
void testStrandNearMovingEdge() {
  const int w = 300, h = 300;
  Img in(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      // Straight above row 150 (equal distances down each column), wiggly below.
      const double edge = clamp01(0.5 + (150 + (y > 150 ? 0.4 * std::sin(2 * M_PI * y / 10.0) : 0.0) - x) / 2.0);
      const double strand = 0.4 * clamp01(1.0 - std::fabs(x - 153.5 - 0.03 * y));  // 3.5-12.5 px out
      in[size_t(y) * w + x] = float(std::max(edge, strand));
    }
  ContourSmoothStats st;
  const Img out = run(in, w, h, ContourSmoothParams(), &st);
  double strandChange = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 154; x < w; ++x) strandChange = std::max(strandChange, double(std::fabs(out[size_t(y) * w + x] - in[size_t(y) * w + x])));
  std::printf("strand near a moving edge: largest edge move %.2f px, strand changed by up to %.3f\n", st.largestShiftPx,
              strandChange);
  CHECK(st.largestShiftPx > 0.1);
  CHECK(strandChange < 0.15);
}

// Only the edge band changes: on the synth chatter scene every changed pixel lies within
// band + 1 px (Max Shift + 6 + 1) of a boundary pixel; a full frame and a noisy matte are left as
// they are; the frame border is no edge; out may be in.
void testUntouchedElsewhere() {
  const int w = 960, h = 540;
  const synth::Scene sc = synth::make("chatter", w, h);
  ContourSmoothParams p;
  const Img out = run(sc.matte, w, h, p);
  mr::Mask edge(sc.matte.size(), 0);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const bool b = sc.matte[i] < p.edgeAt;
      edge[i] = (x > 0 && (sc.matte[i - 1] < p.edgeAt) != b) || (x + 1 < w && (sc.matte[i + 1] < p.edgeAt) != b) ||
                (y > 0 && (sc.matte[i - w] < p.edgeAt) != b) || (y + 1 < h && (sc.matte[i + w] < p.edgeAt) != b);
    }
  const mr::Mask near = mr::dilate(edge, w, h, p.maxShiftPx + 6.0 + 1.0);
  int outside = 0, changed = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    changed += out[i] != sc.matte[i];
    outside += out[i] != sc.matte[i] && !near[i];
  }
  std::printf("chatter %dx%d: %d px changed, %d outside the band\n", w, h, changed, outside);
  CHECK(changed > 1000);
  CHECK(outside == 0);

  Img inPlace = sc.matte;
  mr::contourSmooth(inPlace.data(), inPlace.data(), w, h, p, nullptr);
  CHECK(inPlace == out);

  const Img full(size_t(50) * 40, 0.9f);
  CHECK(run(full, 50, 40) == full);

  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  Img noise(size_t(200) * 150);
  for (float& v : noise) v = u(rng);
  ContourSmoothStats st;
  CHECK(run(noise, 200, 150, p, &st) == noise);
  CHECK(st.skipped);

  // A vertical edge running into the top and bottom of the frame: kept up to the border.
  const int vw = 300, vh = 200;
  const Img vert = slantedEdge(vw, vh, 0.0, [](int) { return 0.0; });
  CHECK(maxDiff(run(vert, vw, vh), vert) < 1e-3);
}

// Brute-force reference: the same model written out without the tiles, the grid, the stripes
// or threads; Gaussian elimination for the fit; the shift applied on full frames. Float where the op decides in float.
namespace ref {

struct S {
  float x, y, nx, ny;
  int px, py;
  bool below;
};

Img contourSmooth(const Img& in, int w, int h, const ContourSmoothParams& p) {
  const float level = p.edgeAt;
  auto below = [&](int x, int y) { return in[size_t(y) * w + x] < level; };
  std::vector<S> s;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      bool edge = false;
      const int nb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
      for (const auto& o : nb) {
        const int qx = x + o[0], qy = y + o[1];
        if (qx >= 0 && qy >= 0 && qx < w && qy < h) edge |= below(qx, qy) != below(x, y);
      }
      if (!edge) continue;
      const ::ref::Foot f = ::ref::foot(in, w, h, level, x, y);
      const float sg = below(x, y) ? -1.0f : 1.0f;
      s.push_back({x + float(f.fx), y + float(f.fy), sg * float(f.nx), sg * float(f.ny), x, y, below(x, y)});
    }
  const double band = p.maxShiftPx + 6.0, r = band + 1.5;
  const float cap = float(band + 1.0);
  Img sd(in.size());
  std::vector<int> nearest(in.size(), -1);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float best = 1e30f;
      int bi = -1;
      for (size_t j = 0; j < s.size(); ++j) {
        const int ox = x - s[j].px, oy = y - s[j].py;
        if (s[j].below == below(x, y) || ox * ox + oy * oy > r * r) continue;
        const float vx = ox - (s[j].x - s[j].px), vy = oy - (s[j].y - s[j].py);
        const float perp = vx * s[j].nx + vy * s[j].ny;
        const float along = std::max(0.0f, std::fabs(vy * s[j].nx - vx * s[j].ny) - 0.5f);
        const float dd = std::min(float(ox * ox + oy * oy), perp * perp + along * along);
        if (dd < best) best = dd, bi = int(j);
      }
      const float d = bi >= 0 ? std::min(std::sqrt(best), cap) : cap;
      sd[size_t(y) * w + x] = below(x, y) ? -d : d;
      nearest[size_t(y) * w + x] = bi;
    }
  auto sdNear = [&](double x, double y) {
    const int xi = int(std::lround(x)), yi = int(std::lround(y));
    return xi < 0 || yi < 0 || xi >= w || yi >= h ? 0.0f : sd[size_t(yi) * w + xi];
  };
  const double sigma = 0.4 * p.smoothPx, R = 2.5 * sigma;
  const double corner = std::min(std::max(p.cornerDeg, 10.0), 179.0) * M_PI / 180.0;
  const double cosC = std::cos(corner), cosF = std::cos(corner - 10.0 * M_PI / 180.0);
  std::vector<float> fitted(s.size(), 0.0f);
  std::vector<int> kept(s.size(), 1);
  for (size_t j = 0; j < s.size(); ++j) {
    double inner = 0, outer = 0;
    for (double k = 0.5; k <= 3.5; k += 0.5) {
      inner = std::max(inner, double(sdNear(s[j].x - k * s[j].nx, s[j].y - k * s[j].ny)));
      outer = std::max(outer, -double(sdNear(s[j].x + k * s[j].nx, s[j].y + k * s[j].ny)));
    }
    const double thick = std::min(1.0, std::max(0.0, (std::min(inner, outer) - 1.5) / 1.5));
    if (thick == 0.0) continue;
    bool monotone = true;  // the profile across never turns back by more than 0.05
    for (int k = 1; k <= 4; ++k) {
      auto at = [&](double kk) { return mr::sampleBilinear(in.data(), w, h, s[j].x + kk * s[j].nx, s[j].y + kk * s[j].ny); };
      monotone &= at(-k) >= at(-(k - 1)) - 0.05f && at(k) <= at(k - 1) + 0.05f;
    }
    if (!monotone) continue;
    kept[j] = 0;
    // Normal equations A c = v for h = c0 + c1 u + c2 u^2.
    double A[3][4] = {{0}};
    for (const S& q : s) {
      const double dot = s[j].nx * q.nx + s[j].ny * q.ny;
      if (dot <= cosC) continue;
      const double dx = q.x - s[j].x, dy = q.y - s[j].y;
      const double t = -dx * s[j].ny + dy * s[j].nx, hgt = dx * s[j].nx + dy * s[j].ny;
      if (dx * dx + dy * dy > R * R || std::fabs(hgt) > 1.5 + 0.6 * std::fabs(t)) continue;
      const double u = t / sigma, wt = std::exp(-0.5 * u * u) * std::min(1.0, (dot - cosC) / (cosF - cosC));
      const double b[3] = {1, u, u * u};
      for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c) A[a][c] += wt * b[a] * b[c];
        A[a][3] += wt * b[a] * hgt;
      }
    }
    const double m0 = A[0][0];
    const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[1][2]) - A[0][1] * (A[0][1] * A[2][2] - A[1][2] * A[0][2]) +
                       A[0][2] * (A[0][1] * A[1][2] - A[1][1] * A[0][2]);
    const double d2 = A[0][0] * A[1][1] - A[0][1] * A[0][1];
    const int dim = det > 1e-6 * m0 * m0 * m0 ? 3 : d2 > 1e-6 * m0 * m0 ? 2 : 1;
    double M[3][4];
    for (int a = 0; a < dim; ++a) {
      for (int c = 0; c < dim; ++c) M[a][c] = A[a][c];
      M[a][dim] = A[a][3];
    }
    for (int col = 0; col < dim; ++col)  // Gauss-Jordan (the matrix is positive definite)
      for (int row = 0; row < dim; ++row) {
        if (row == col) continue;
        const double f = M[row][col] / M[col][col];
        for (int c = col; c <= dim; ++c) M[row][c] -= f * M[col][c];
      }
    const double a0 = m0 > 0 ? M[0][dim] / M[0][0] : 0.0;
    const double support = m0 / (2.0 * std::sqrt(2.0 * M_PI) * sigma);
    const double trust = std::min(1.0, std::max(0.0, (support - 0.3) / 0.3));
    const double mv = thick * trust * a0;
    fitted[j] = float(std::min(std::max(mv, -p.maxShiftPx), p.maxShiftPx));
  }
  std::vector<float> move(s.size());
  for (size_t j = 0; j < s.size(); ++j) {
    move[j] = 0.0f;
    if (kept[j]) continue;
    std::vector<float> near;
    for (size_t k = 0; k < s.size(); ++k)
      if (!kept[k] && std::hypot(s[k].x - s[j].x, s[k].y - s[j].y) <= 3 && s[j].nx * s[k].nx + s[j].ny * s[k].ny > cosC)
        near.push_back(fitted[k]);
    std::sort(near.begin(), near.end());
    const float med = near[near.size() / 2];
    const float m = std::fabs(fitted[j] - med) > 0.25f ? med : fitted[j];
    move[j] = m > 0.05f ? m - 0.05f : m < -0.05f ? m + 0.05f : 0.0f;
  }
  std::vector<float> shift(in.size(), 0.0f);
  for (size_t i = 0; i < in.size(); ++i)
    if (nearest[i] >= 0 && std::fabs(sd[i]) <= band) shift[i] = move[size_t(nearest[i])];
  Img out(in);  // p moved by its shift along sd's normal, read bilinearly
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      if (shift[i] == 0.0f) continue;
      const int xl = std::max(x - 1, 0), xr = std::min(x + 1, w - 1), yu = std::max(y - 1, 0), yd = std::min(y + 1, h - 1);
      const double gx = xr > xl ? (double(sd[size_t(y) * w + xr]) - sd[size_t(y) * w + xl]) / (xr - xl) : 0.0;
      const double gy = yd > yu ? (double(sd[size_t(yd) * w + x]) - sd[size_t(yu) * w + x]) / (yd - yu) : 0.0;
      const double g = std::hypot(gx, gy);
      if (g >= 0.5) out[i] = mr::sampleBilinear(in.data(), w, h, x + shift[i] * gx / g, y + shift[i] * gy / g);
    }
  return out;
}

}  // namespace ref

void testAgainstBruteForce() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  int bad = 0, changed = 0;
  long px = 0;
  for (int trial = 0; trial < 5; ++trial) {
    const int w = 280, h = 260;
    // A wiggly blob (radius 45-75 with 2-4 wiggle periods), a square corner and a thin strand.
    const double R = 45 + 30 * u(rng), amp = 0.5 + u(rng), k = 7 + int(10 * u(rng)), ph = 6 * u(rng);
    const double soft = 1.0 + 2.0 * u(rng);
    Img in(size_t(w) * h);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double a = std::atan2(y - 120.0, x - 120.0);
        double sd = R + amp * std::sin(k * a + ph) - std::hypot(x - 120.0, y - 120.0);
        sd = std::max(sd, 30 - std::max(std::fabs(x - 225.0), std::fabs(y - 215.0)));
        double v = clamp01(0.5 + sd / soft);
        v = std::max(v, 0.8 * clamp01(1.0 - std::fabs(x - 250 - 0.1 * y)));
        in[size_t(y) * w + x] = float(v);
      }
    ContourSmoothParams p;
    p.smoothPx = 15 + 20 * u(rng);
    p.maxShiftPx = 0.5 + 1.5 * u(rng);
    p.cornerDeg = 15 + 30 * u(rng);
    p.edgeAt = float(0.3 + 0.4 * u(rng));
    ContourSmoothStats st;
    const Img got = run(in, w, h, p, &st);
    const Img want = ref::contourSmooth(in, w, h, p);
    CHECK(!st.skipped);
    for (size_t i = 0; i < in.size(); ++i) {
      bad += std::fabs(got[i] - want[i]) > 1e-4f;
      changed += got[i] != in[i];
    }
    px += long(in.size());
  }
  std::printf("brute force: %ld px compared, %d changed, %d differ\n", px, changed, bad);
  CHECK(changed > 2000);
  CHECK(bad == 0);
}

// The synth chatter scene against its truth. Error in the edge band (soft in the truth or the
// input), on the wiggly left and the clean right; its high-frequency part (the error minus its
// band-normalised Gaussian blur, sigma 10: the low-frequency part of the synth's wiggle is
// shape to any local smoother); the truth as input (should stay); crops at corners and hair.
void testChatterScene() {
  const int w = 1920, h = 1080;
  const synth::Scene sc = synth::make("chatter", w, h);
  const Img out = run(sc.matte, w, h), clean = run(sc.truth, w, h);
  auto inBand = [&](size_t i) {
    return (sc.truth[i] > 0 && sc.truth[i] < 1) || (sc.matte[i] > 0 && sc.matte[i] < 1);
  };
  auto bandError = [&](const Img& img, int side, bool highFreq) {
    std::vector<double> e(img.size(), 0.0), m(img.size(), 0.0);
    for (size_t i = 0; i < img.size(); ++i)
      if (inBand(i)) e[i] = img[i] - sc.truth[i], m[i] = 1;
    std::vector<double> be(e.size(), 0.0), bm(e.size(), 1.0);
    if (highFreq) {
      auto blur = [&](const std::vector<double>& f) {  // separable, sigma 10, radius 30
        std::vector<double> g(f.size(), 0.0), o(f.size(), 0.0);
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            for (int d = -30; d <= 30; ++d)
              if (x + d >= 0 && x + d < w) g[size_t(y) * w + x] += std::exp(-d * d / 200.0) * f[size_t(y) * w + x + d];
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            for (int d = -30; d <= 30; ++d)
              if (y + d >= 0 && y + d < h) o[size_t(y) * w + x] += std::exp(-d * d / 200.0) * g[size_t(y + d) * w + x];
        return o;
      };
      be = blur(e);
      bm = blur(m);
    }
    double sum = 0;
    long cnt = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (!m[i] || (side < 0 ? x >= w / 2 - 4 : x <= w / 2 + 4)) continue;
        sum += std::fabs(e[i] - (highFreq ? be[i] / bm[i] : 0.0));
        ++cnt;
      }
    return sum / cnt;
  };
  const double l0 = bandError(sc.matte, -1, false), l1 = bandError(out, -1, false);
  const double r1 = bandError(out, 1, false), c1 = bandError(clean, 1, false);
  const double hl0 = bandError(sc.matte, -1, true), hl1 = bandError(out, -1, true);
  std::printf("chatter, mean |error| in the edge band: wiggly left %.4f -> %.4f (high-frequency %.4f -> %.4f), "
              "clean right 0.0000 -> %.4f\n", l0, l1, hl0, hl1, r1);
  std::printf("truth as input: mean |change| in the band (right) %.4f\n", c1);
  CHECK(l1 < 0.9 * l0);
  CHECK(hl1 < 0.9 * hl0);
  CHECK(r1 < 0.01 && c1 < 0.01);
  // Crops (design units: 1080 lines = 1 px, origin = frame centre), 96 px: max |error| vs
  // truth before -> after on the chatter input, and the max change with the truth as input.
  struct Crop {
    const char* name;
    double x, y;
  } crops[] = {{"left elbow", -450, 0},   {"left hand on hip", -145, 110}, {"left shoulder", -137, -209},
               {"left upper arm", -295, -85}, {"right elbow", 230, 60},   {"right hand", 255, 290},
               {"hair", 0, -430}};
  for (const Crop& c : crops) {
    const int x0 = int(w / 2 + c.x) - 48, y0 = int(h / 2 + c.y) - 48;
    double b = 0, a = 0, k = 0;
    for (int y = y0; y < y0 + 96; ++y)
      for (int x = x0; x < x0 + 96; ++x) {
        const size_t i = size_t(y) * w + x;
        b = std::max(b, double(std::fabs(sc.matte[i] - sc.truth[i])));
        a = std::max(a, double(std::fabs(out[i] - sc.truth[i])));
        k = std::max(k, double(std::fabs(clean[i] - sc.truth[i])));
      }
    std::printf("  %-17s max error %.3f -> %.3f; truth as input: max change %.3f\n", c.name, b, a, k);
    CHECK(k < 0.4);  // corners and creases: < 0.8 px of edge move with the 2 px profile
  }
}

void testTiming() {
  const int w = 3840, h = 2160;
  const synth::Scene sc = synth::make("chatter", w, h);
  Img out(sc.matte.size());
  ContourSmoothStats st;
  double best = 1e9;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    mr::contourSmooth(sc.matte.data(), out.data(), w, h, ContourSmoothParams(), &st);
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::printf("UHD contour smooth (chatter): %.1f ms, %lld samples, %lld px changed\n", best, st.contourSamples,
              st.pixelsChanged);
  CHECK(st.pixelsChanged > 0);
  CHECK(best < 100.0);
}

}  // namespace

int main() {
  testWiggleRemoved();
  testCleanKept();
  testCornersKept();
  testThinUntouched();
  testStrandNearMovingEdge();
  testUntouchedElsewhere();
  testAgainstBruteForce();
  testChatterScene();
  testTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
