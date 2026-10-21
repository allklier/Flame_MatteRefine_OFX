// Solid Core (decision 15) on synthetic mattes with known answers, plus UHD timing.
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
// of edge and the hole with its own Depth px margin are untouched.
void testCoreAndEdge() {
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
      if (d < 80.0 - 4.5 - 7.0 && dg > 8.0 + 7.0 && b != 1.0f) ++coreLeft;
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

// Against brute force: a pixel becomes 1 iff no pixel below Edge At lies within Depth.
void testAgainstBruteForce() {
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  int bad = 0;
  for (int trial = 0; trial < 30; ++trial) {
    Img in(31 + trial % 4, 27 + trial % 5);
    const float pBelow = trial < 15 ? 0.02f : 0.3f;  // sparse: disc stamps; dense: transform
    for (auto& v : in.v) v = u(rng) < pBelow ? u(rng) * 0.5f : 0.5f + u(rng) * 0.5f;
    SolidCoreParams p;
    p.depthPx = 0.5 + (trial % 6) * 1.3;
    const Img out = run(in, p);
    for (int y = 0; y < in.h; ++y)
      for (int x = 0; x < in.w; ++x) {
        bool near = false;
        for (int j = 0; j < in.w * in.h && !near; ++j) {
          const int dx = j % in.w - x, dy = j / in.w - y;
          near = in.v[j] < p.edgeAt && dx * dx + dy * dy <= p.depthPx * p.depthPx;
        }
        bad += out.at(x, y) != (near ? in.at(x, y) : 1.0f);
      }
  }
  CHECK(bad == 0);
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
  testCoreAndEdge();
  testThinStrip();
  testFrameBorder();
  testAgainstBruteForce();
  testTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
