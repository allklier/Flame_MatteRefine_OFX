// The shared edge geometry (ops/edge_field): signed distance against brute force and against
// exact answers on straight and round antialiased edges; resampling along the normal.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "edge_reference.h"
#include "ops/edge_field.h"
#include "equivalence.h"
#include "old_edge_ops.h"
#include "synth.h"

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

// Soft discs: alpha = clamp(0.5 + (r - dist) / soft), so the 0.5 contour is the circle r.
std::vector<float> discs(int w, int h, const std::vector<std::array<double, 4>>& ds) {
  std::vector<float> m(size_t(w) * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (const auto& d : ds) {
        const double a = 0.5 + (d[2] - std::hypot(x - d[0], y - d[1])) / d[3];
        m[size_t(y) * w + x] = std::max(m[size_t(y) * w + x], float(std::min(1.0, std::max(0.0, a))));
      }
  return m;
}

// Exact path only (images large enough for stamping, morphology.h); edge_reference.h is the
// model written out.
void testAgainstBruteForce() {
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  int bad = 0, cases = 0;
  for (int trial = 0; trial < 12; ++trial) {
    const int w = 200, h = 180;
    std::vector<std::array<double, 4>> ds;
    for (int k = 0; k < 3; ++k)
      ds.push_back({20 + u(rng) * 160, 20 + u(rng) * 140, 6 + u(rng) * 14, 0.5 + u(rng) * 4});
    const std::vector<float> in = discs(w, h, ds);
    const float level = float(0.3 + 0.4 * u(rng));
    const double band = 2.0 + trial % 4;
    const std::vector<float> sd = mr::signedEdgeDistance(in.data(), w, h, level, band);
    const std::vector<ref::Boundary> belowEdge = ref::boundary(in, w, h, level, true);
    const std::vector<ref::Boundary> aboveEdge = ref::boundary(in, w, h, level, false);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const bool inside = in[size_t(y) * w + x] >= level;
        const double d = ref::distance(inside ? belowEdge : aboveEdge, x, y);
        const double want = (inside ? 1 : -1) * std::min(d, band + 1.0);
        const double got = sd[size_t(y) * w + x];
        if (std::fabs(want) <= band) {
          ++cases;
          bad += std::fabs(got - want) > 1e-4;
        } else {
          bad += std::fabs(got) <= band;  // beyond the band: reported as beyond
        }
      }
  }
  std::printf("brute force: %d in-band pixels checked\n", cases);
  CHECK(cases > 5000);
  CHECK(bad == 0);
}

// A straight antialiased edge at a slant: the distance to the true contour line.
void testStraightEdge() {
  const int w = 600, h = 200;
  const double slope = 0.3, nrm = std::hypot(1.0, slope);
  std::vector<float> in(size_t(w) * h);
  auto signedLine = [&](int x, int y) { return (x - 100 - slope * y) / nrm; };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      in[size_t(y) * w + x] = float(std::min(1.0, std::max(0.0, 0.5 + signedLine(x, y) / 2.0)));
  const std::vector<float> sd = mr::signedEdgeDistance(in.data(), w, h, 0.5f, 8.0);
  double worst = 0;
  for (int y = 10; y < h - 10; ++y)
    for (int x = 0; x < w; ++x)
      if (std::fabs(signedLine(x, y)) < 7.5)
        worst = std::max(worst, std::fabs(sd[size_t(y) * w + x] - signedLine(x, y)));
  std::printf("straight edge: max |sd - true| %.3f px\n", worst);
  CHECK(worst < 0.1);
}

// Grow and shrink: a soft disc shifted along the normal is the same disc with radius r + k.
void testShiftDisc() {
  const int w = 400, h = 400;  // large enough for the exact path
  for (double soft : {1.0, 4.0})
    for (double k : {2.5, -3.0}) {
      const std::vector<float> in = discs(w, h, {{200, 200, 40, soft}});
      const std::vector<float> want = discs(w, h, {{200, 200, 40 + k, soft}});
      const std::vector<float> sd = mr::signedEdgeDistance(in.data(), w, h, 0.5f, 12.0);
      const std::vector<float> shift(in.size(), float(k));
      std::vector<float> out(in.size());
      mr::shiftAlongNormal(in.data(), sd, shift, w, h, out.data());
      double worst = 0;
      for (size_t i = 0; i < in.size(); ++i) worst = std::max(worst, double(std::fabs(out[i] - want[i])));
      std::printf("disc soft %.0f shift %+.1f: max error %.3f\n", soft, k, worst);
      // A 1 px edge is point-sampled: half a pixel off a sample its value is undetermined
      // (+-0.25 for any resampler). A 4 px edge is resolved.
      CHECK(worst < (soft < 2 ? 0.26 : 0.04));
    }
}

// No edge at the frame border: a full frame is far inside everywhere.
void testFrameBorder() {
  std::vector<float> in(50 * 40, 0.9f);
  const std::vector<float> sd = mr::signedEdgeDistance(in.data(), 50, 40, 0.5f, 5.0);
  CHECK(*std::min_element(sd.begin(), sd.end()) == 6.0f);
}

// Equivalence of the shared code with the code before the speed work (old_edge_ops.h), bit for
// bit: dilate and distanceWithin (both overloads, the stamping and the distance-transform
// paths), contourFoot (both overloads), signedEdgeDistanceThreaded and the moved edge helpers
// (edgeNormal, profileAt, isContour, contourSlope, footMean), and contourPixels against its
// definition; on the synthetic scenes at 1080 and on random mattes and sizes.
void testEquivalence() {
  long long cases = 0, bad = 0, exactFallback = 0, gridFallback = 0;
  auto same = [&](bool ok, const char* what, int w, int h) {
    ++cases;
    if (!ok) {
      ++bad;
      std::printf("  differs: %s %dx%d\n", what, w, h);
    }
  };
  auto check = [&](const std::vector<float>& in, int w, int h, float level, double r) {
    const size_t n = in.size();
    mr::Mask below(n);
    old_::Mask oldBelow(n);
    for (size_t i = 0; i < n; ++i) below[i] = oldBelow[i] = in[i] < level;
    same(mr::dilate(below, w, h, r) == old_::dilate(oldBelow, w, h, r), "dilate", w, h);
    // Feet: both overloads against the old function.
    bool feetSame = true;
    for (size_t i = 0; i < n; ++i) {
      const mr::ContourFoot a = mr::contourFoot(in.data(), below, w, h, level, int(i));
      const mr::ContourFoot b = mr::contourFoot(in.data(), below.data(), w, h, level, int(i));
      const old_::ContourFoot c = old_::contourFoot(in.data(), oldBelow, w, h, level, int(i));
      const float fa[4] = {a.fx, a.fy, a.nx, a.ny}, fb[4] = {b.fx, b.fy, b.nx, b.ny}, fc[4] = {c.fx, c.fy, c.nx, c.ny};
      feetSame = feetSame && eq::bitsDiffer(fa, fc, 4) == 0 && eq::bitsDiffer(fb, fc, 4) == 0;
    }
    same(feetSame, "contourFoot", w, h);
    auto foot = [&](int i) { return mr::contourFoot(in.data(), below, w, h, level, i); };
    auto oldFoot = [&](int i) { return old_::contourFoot(in.data(), oldBelow, w, h, level, i); };
    const std::vector<float> dOld = old_::distanceWithin(oldBelow, w, h, r, oldFoot);
    const std::vector<float> dNew = mr::distanceWithin(below, w, h, r, foot);
    std::vector<float> dRaw(n);
    mr::distanceWithin(below.data(), w, h, r, foot, dRaw.data());
    std::vector<float> dPool(n);
    {
      mr::Workers wk;
      mr::distanceWithin(below.data(), w, h, r, foot, dPool.data(), 2.0, &wk);
    }
    same(eq::bitsDiffer(dNew.data(), dOld.data(), n) == 0 && eq::bitsDiffer(dRaw.data(), dOld.data(), n) == 0 &&
             eq::bitsDiffer(dPool.data(), dOld.data(), n) == 0,
         "distanceWithin", w, h);
    const std::unique_ptr<float[]> sdNew = mr::signedEdgeDistanceThreaded(in.data(), w, h, level, r);
    const std::unique_ptr<float[]> sdOld = old_::sg::signedEdgeDistanceThreaded(in.data(), w, h, level, r);
    same(eq::bitsDiffer(sdNew.get(), sdOld.get(), n) == 0, "signedEdgeDistanceThreaded", w, h);
    // contourPixels: the pixels with a 4-neighbour on the other side, in scan order.
    std::vector<int> want;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if ((x > 0 && below[i - 1] != below[i]) || (x + 1 < w && below[i + 1] != below[i]) ||
            (y > 0 && below[i - w] != below[i]) || (y + 1 < h && below[i + w] != below[i]))
          want.push_back(int(i));
      }
    same(mr::contourPixels(below.data(), w, h) == want, "contourPixels", w, h);
    // Which paths ran: signedEdgeDistanceThreaded's switch at 12 frames, distanceWithin's at 2.
    const double R = r + 1.5;
    long long disc = 0, discNoCentre = 0, setBoundary = 0;
    for (int dy = -int(R); dy <= int(R); ++dy) {
      disc += 2 * int(std::floor(std::sqrt(std::max(0.0, R * R - double(dy) * dy)))) + 1;
      for (int dx = -int(R); dx <= int(R); ++dx) discNoCentre += (dx || dy) && dx * dx + dy * dy <= R * R;
    }
    for (int i : want) setBoundary += below[size_t(i)] != 0;
    exactFallback += double(want.size()) * double(disc) > 12.0 * double(n);
    gridFallback += double(setBoundary) * double(discNoCentre) > 2.0 * double(n);
    // The moved helpers, on the 0.5 contour's distance, at every pixel near it.
    const std::unique_ptr<float[]> sd = mr::signedEdgeDistanceThreaded(in.data(), w, h, 0.5f, 6.0);
    std::vector<float> measure(n * 2);
    for (size_t i = 0; i < n; ++i) measure[2 * i] = float(i % 7) - 1.0f, measure[2 * i + 1] = float(i % 13);
    bool helpersSame = true;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const float* s = sd.get();
        if (std::fabs(s[size_t(y) * w + x]) > 6.0f) continue;
        helpersSame = helpersSame && mr::isContour(in.data(), w, h, x, y) == old_::sg::isContour(in.data(), w, h, x, y);
        const double sa = mr::contourSlope(in.data(), s, w, h, x, y), sb = old_::sg::contourSlope(in.data(), s, w, h, x, y);
        double nx = 0, ny = 0, mx = 0, my = 0;
        const bool na = mr::edgeNormal(s, w, h, x, y, nx, ny), nb = old_::sg::edgeNormal(s, w, h, x, y, mx, my);
        helpersSame = helpersSame && sa == sb && na == nb && (!na || (nx == mx && ny == my));
        if (!na) continue;
        for (double t : {-7.5, -2.0, -0.3, 0.0, 0.7, 3.0, 9.0}) {
          const float pa = mr::profileAt(in.data(), s, w, h, x, y, nx, ny, t);
          const float pb = old_::sg::profileAt(in.data(), s, w, h, x, y, nx, ny, t);
          helpersSame = helpersSame && eq::bitsDiffer(&pa, &pb, 1) == 0;
        }
        double ma[2], mb[2];
        const double wa = mr::footMean(in.data(), measure.data(), 2, w, h, x - 0.7 * nx, y - 0.7 * ny, ma);
        const double wb = old_::sg::footMean(in.data(), measure.data(), 2, w, h, x - 0.7 * nx, y - 0.7 * ny, mb);
        helpersSame = helpersSame && wa == wb && (wa <= 0.0 || (ma[0] == mb[0] && ma[1] == mb[1]));
      }
    same(helpersSame, "edge helpers", w, h);
  };
  const char* scenes[] = {"all", "mixed_edge", "eroded", "chatter", "holes"};
  for (const char* name : scenes) {
    const synth::Scene s = synth::make(name, 1920, 1080);
    check(s.matte, 1920, 1080, 0.5f, 8.0);
    check(s.matte, 1920, 1080, 0.2f, 30.0);  // the distance-transform paths on noisy scenes
  }
  std::mt19937 rng(20261027);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int t = 0; t < 40; ++t) {
    const int w = t < 4 ? 1 + t * 3 : 20 + int(u(rng) * 380), h = t < 4 ? 40 - t * 9 : 10 + int(u(rng) * 290);
    const std::vector<float> m = eq::randomMatte(rng, w, h);
    check(m, w, h, float(0.05 + 0.9 * u(rng)), 0.5 + u(rng) * (t % 3 == 0 ? 40.0 : 12.0));
  }
  std::printf("equivalence with the old shared code: %lld checks, %lld differ (cases on the 12-frame switch: %lld, "
              "on distanceWithin's 2-frame switch: %lld)\n", cases, bad, exactFallback, gridFallback);
  CHECK(bad == 0);
}

}  // namespace

int main() {
  testAgainstBruteForce();
  testStraightEdge();
  testShiftDisc();
  testFrameBorder();
  testEquivalence();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
