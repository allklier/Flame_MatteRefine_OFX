// Solid Core (decisions 15, 16) on synthetic mattes with known answers, plus UHD timing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "ops/solid_core.h"

using mr::SolidCoreParams;
using mr::SolidCoreStats;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

struct Img {
  int w, h;
  std::vector<float> v;
  Img(int w, int h, float fill = 0.0f) : w(w), h(h), v(size_t(w) * h, fill) {}
  float& at(int x, int y) { return v[size_t(y) * w + x]; }
  float at(int x, int y) const { return v[size_t(y) * w + x]; }
};

Img run(const Img& in, const SolidCoreParams& p, SolidCoreStats* st = nullptr) {
  Img out(in.w, in.h);
  mr::solidCore(in.v.data(), out.v.data(), in.w, in.h, p, st);
  return out;
}

// A grey (0.9) disc with a soft edge and a dark hole: the core turns white, the outer Depth px
// of edge and the hole with its own Depth px margin are untouched, hard or soft.
void testCoreAndEdge(double feather) {
  Img in(200, 200);
  for (int y = 0; y < 200; ++y)
    for (int x = 0; x < 200; ++x) {
      const double d = std::hypot(x - 100.0, y - 100.0);
      float v = float(0.9 * std::min(1.0, std::max(0.0, (80.0 - d) / 10.0)));  // 10 px ramp
      if (std::hypot(x - 130.0, y - 100.0) < 8.0) v = 0.1f;                 // a real gap
      in.at(x, y) = v;
    }
  SolidCoreParams p;
  p.depthPx = 6.0;
  p.featherPx = feather;
  const Img out = run(in, p);
  int edgeChanged = 0, gapChanged = 0, coreLeft = 0, lower = 0;
  for (int y = 0; y < 200; ++y)
    for (int x = 0; x < 200; ++x) {
      const float a = in.at(x, y), b = out.at(x, y);
      lower += b < a;
      // The 0.5 contour sits ~4.4 px inside the disc radius 80 (0.9 * (80 - d) / 10 = 0.5).
      const double d = std::hypot(x - 100.0, y - 100.0), dg = std::hypot(x - 130.0, y - 100.0);
      if (d > 80.0 - 4.5 - 6.0 && b != a) ++edgeChanged;
      if (dg < 8.0 + 6.0 - 1.0 && b != a) ++gapChanged;  // 1 px for the discrete disc
      if (d < 80.0 - 4.5 - 7.0 - feather && dg > 8.0 + 7.0 + feather && b != 1.0f) ++coreLeft;
    }
  CHECK(lower == 0);
  CHECK(edgeChanged == 0);
  CHECK(gapChanged == 0);
  CHECK(coreLeft == 0);
}

// A strip narrower than 2 x Depth has no core and is untouched.
void testThinStrip() {
  Img in(60, 60);
  for (int y = 0; y < 60; ++y)
    for (int x = 25; x < 35; ++x) in.at(x, y) = 0.8f;  // 10 px wide
  SolidCoreParams p;
  p.depthPx = 6.0;
  p.featherPx = 0.0;
  SolidCoreStats st;
  const Img out = run(in, p, &st);
  CHECK(st.pixelsChanged == 0);
  p.depthPx = 4.0;  // 10 px wide: the middle 2 px are more than 4 px from the 0.5 contour
  CHECK(run(in, p).at(29, 30) == 1.0f && run(in, p).at(25, 30) == 0.8f);
}

// The frame border is not an edge: a subject filling the frame cores right up to it.
void testFrameBorder() {
  Img in(40, 30, 0.9f);
  SolidCoreParams p;
  const Img out = run(in, p);
  CHECK(out.at(0, 0) == 1.0f && out.at(39, 29) == 1.0f);
}

// The contour offset of a below pixel (decision 16), written out independently of the op.
double contourOffset(const Img& in, int x, int y, float edgeAt) {
  if (in.at(x, y) >= edgeAt) return 0.0;
  double inv = 0.0;
  const int nb[2][2][2] = {{{-1, 0}, {1, 0}}, {{0, -1}, {0, 1}}};
  for (const auto& axis : nb) {
    double t = 2.0;
    for (const auto& o : axis) {
      const int nx = x + o[0], ny = y + o[1];
      if (nx < 0 || ny < 0 || nx >= in.w || ny >= in.h || in.at(nx, ny) < edgeAt) continue;
      t = std::min(t, (edgeAt - in.at(x, y)) / double(in.at(nx, ny) - in.at(x, y)));
    }
    if (t <= 1.0) inv += 1.0 / (t * t);
  }
  return inv > 0.0 ? 1.0 / std::sqrt(inv) : 0.0;
}

// Against brute force. Hard (Feather 0): a pixel becomes 1 iff no pixel below Edge At lies
// within Depth. Soft: D = min over below pixels q of |p - q| - offset(q); a pixel with grid
// distance <= Depth stays, else max(in, smoothstep((D - Depth) / Feather)). When stamping would
// cost more than a frame of writes (morphology.h), D = the plain grid distance.
void testAgainstBruteForce(double feather) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  int bad = 0, exactPath = 0, fallbackPath = 0;
  for (int trial = 0; trial < 30; ++trial) {
    Img in(31 + trial % 4, 27 + trial % 5);
    const float pBelow = trial < 15 ? 0.02f : 0.3f;  // sparse: disc stamps; dense: transform
    for (auto& v : in.v) v = u(rng) < pBelow ? u(rng) * 0.5f : 0.5f + u(rng) * 0.5f;
    SolidCoreParams p;
    p.depthPx = 0.5 + (trial % 6) * 1.3;
    p.featherPx = feather;
    const Img out = run(in, p);
    // Which path the op takes: boundary pixels x disc offsets (radius Depth + Feather + 1).
    const double rs = p.depthPx + feather + 1.0;
    long disc = 0, boundary = 0;
    for (int dy = -int(rs); dy <= int(rs); ++dy)
      for (int dx = -int(rs); dx <= int(rs); ++dx) disc += (dx || dy) && dx * dx + dy * dy <= rs * rs;
    for (int y = 0; y < in.h; ++y)
      for (int x = 0; x < in.w; ++x) {
        auto below = [&](int a, int b) { return in.at(a, b) < p.edgeAt; };
        boundary += below(x, y) && ((x > 0 && !below(x - 1, y)) || (x + 1 < in.w && !below(x + 1, y)) ||
                                    (y > 0 && !below(x, y - 1)) || (y + 1 < in.h && !below(x, y + 1)));
      }
    const bool exact = double(boundary) * disc <= double(in.w) * in.h;
    if (feather > 0) (exact ? exactPath : fallbackPath)++;
    for (int y = 0; y < in.h; ++y)
      for (int x = 0; x < in.w; ++x) {
        double g2 = 1e30, dRef = 1e30;
        for (int j = 0; j < in.w * in.h; ++j) {
          if (in.v[j] >= p.edgeAt) continue;
          const int qx = j % in.w, qy = j / in.w, dx = qx - x, dy = qy - y;
          g2 = std::min(g2, double(dx * dx + dy * dy));
          dRef = std::min(dRef, std::sqrt(double(dx * dx + dy * dy)) - contourOffset(in, qx, qy, p.edgeAt));
        }
        if (!exact) dRef = std::sqrt(g2);
        float want = in.at(x, y);
        if (feather <= 0) {
          if (g2 > p.depthPx * p.depthPx) want = 1.0f;
        } else if (dRef > p.depthPx) {
          const double t = std::min(1.0, (dRef - p.depthPx) / feather);
          want = std::max(want, float(t * t * (3.0 - 2.0 * t)));
        }
        bad += std::fabs(out.at(x, y) - want) > 1e-5f;
      }
  }
  CHECK(bad == 0);
  if (feather > 0) CHECK(exactPath > 0 && fallbackPath > 0);
}

// Aliasing (decision 16): a straight antialiased edge at a slant with a grey (0.85) face behind
// it; the largest difference from the ideal soft core, computed from the true distance to the
// contour. A hard core (or the grid distance) is off by up to the whole step.
double coreValueError(double feather, double slope) {
  const int w = 700, h = 200;  // wide enough for the exact path (as on real frames)
  const double nrm = std::hypot(1.0, slope), grey = 0.85, depth = 6.0;
  auto signedDist = [&](int x, int y) { return (x - 20 - slope * y) / nrm; };
  Img in(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      in.at(x, y) = float(grey * std::min(1.0, std::max(0.0, signedDist(x, y) + 0.5)));
  SolidCoreParams p;
  p.depthPx = depth;
  p.featherPx = feather;
  const Img out = run(in, p);
  const double contour = 0.5 / grey - 0.5;  // where the input crosses Edge At
  const double f = std::max(feather, 1e-9);
  double worst = 0;
  for (int y = 10; y < h - 10; ++y)
    for (int x = 0; x < w; ++x) {
      const double t = std::min(1.0, std::max(0.0, (signedDist(x, y) - contour - depth) / f));
      const double ideal = std::max(double(in.at(x, y)), t * t * (3.0 - 2.0 * t));
      worst = std::max(worst, std::fabs(out.at(x, y) - ideal));
    }
  return worst;
}

void testSmoothCore() {
  for (double slope : {0.11, 0.3, 0.81}) {
    const double hard = coreValueError(0.0, slope), soft = coreValueError(2.0, slope);
    std::printf("slope %.2f: max error vs ideal: hard %.3f, feather 2 %.3f\n", slope, hard, soft);
    CHECK(hard > 0.14);  // the test sees the staircase
    CHECK(soft < 0.09);
  }
}

void testTiming() {
  const int w = 3840, h = 2160;
  Img in(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) in.at(x, y) = std::hypot(x - 1920.0, y - 1080.0) < 900 ? 0.95f : 0.0f;
  SolidCoreParams p;
  p.depthPx = 8.0;
  Img out(w, h);
  SolidCoreStats st;
  const auto t0 = std::chrono::steady_clock::now();
  mr::solidCore(in.v.data(), out.v.data(), w, h, p, &st);
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("UHD solid core 8 px: %.1f ms, %lld px\n", ms, st.pixelsChanged);
  CHECK(st.pixelsChanged > 0);
  CHECK(ms < 100.0);
}

}  // namespace

int main() {
  testCoreAndEdge(0.0);
  testCoreAndEdge(2.0);
  testThinStrip();
  testFrameBorder();
  testAgainstBruteForce(0.0);
  testAgainstBruteForce(2.5);
  testSmoothCore();
  testTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
