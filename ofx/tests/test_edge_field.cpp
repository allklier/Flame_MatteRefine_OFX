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

}  // namespace

int main() {
  testAgainstBruteForce();
  testStraightEdge();
  testShiftDisc();
  testFrameBorder();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
