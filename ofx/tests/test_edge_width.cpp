// Edge Width (A5) on mattes with known answers: slanted edges of several widths retimed to one,
// the contour and the levels either side kept, thin parts left alone, the frame border, the
// synthetic "mixed_edge" scene against its truth per region, "holes" (a grey face is no edge),
// and UHD timing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ops/edge_width.h"
#include "synth.h"

using mr::EdgeWidthParams;
using mr::EdgeWidthStats;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

double clamp01(double v) { return std::min(1.0, std::max(0.0, v)); }

std::vector<float> run(const std::vector<float>& in, int w, int h, double width, double amount = 1.0,
                       EdgeWidthStats* st = nullptr) {
  std::vector<float> out(in.size());
  EdgeWidthParams p;
  p.widthPx = width;
  p.amount = amount;
  mr::edgeWidth(in.data(), out.data(), w, h, p, st);
  return out;
}

// A straight edge at a slant, d = the signed distance to it (> 0 inside, left); soft = 0 is a
// hard cut (aliased).
std::vector<float> slanted(int w, int h, double slope, double soft, double level = 1.0) {
  const double nrm = std::hypot(1.0, slope);
  std::vector<float> m(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double d = (300 + slope * y - x) / nrm;
      m[size_t(y) * w + x] = float(level * (soft > 0 ? clamp01(0.5 + d / soft) : d >= 0 ? 1.0 : 0.0));
    }
  return m;
}

// Every antialiased input width (1, 4, 8 px) retimed to the target: the largest difference from
// the ideal ramp of the target width on the same contour, and pixels farther than both
// transitions + 1.5 px from it untouched.
void testSlanted() {
  const int w = 900, h = 200;  // wide enough for the exact distance path
  for (double slope : {0.11, 0.3, 0.81})
    for (double soft : {1.0, 4.0, 8.0})
      for (double width : {2.0, 4.0}) {
        const std::vector<float> in = slanted(w, h, slope, soft), out = run(in, w, h, width);
        const double nrm = std::hypot(1.0, slope);
        double worst = 0;
        int far = 0;
        for (int y = 10; y < h - 10; ++y)
          for (int x = 0; x < w; ++x) {
            const size_t i = size_t(y) * w + x;
            const double d = (300 + slope * y - x) / nrm;
            worst = std::max(worst, std::fabs(out[i] - clamp01(0.5 + d / width)));
            far += std::fabs(d) > std::max(soft, width) / 2 + 1.5 && out[i] != in[i];
          }
        std::printf("slant %.2f input %.0f px -> %.0f px: max error %.3f\n", slope, soft, width, worst);
        // A 1 px input is point-sampled (+-0.25 at a sample for any resampler, test_edge_field);
        // stretched to 4 px about 0.07 of that remains.
        CHECK(worst < 0.07);
        CHECK(far == 0);
      }
}

// Hard cuts (aliased, 0/1) widened: a point-sampled line at x = L. Axis-aligned (L = 300.5 + 0):
// every row the same profile and exactly the ideal ramp. Slanted: the largest difference from
// the ideal ramp on the true line (as test_solid_core coreValueError). Before the fix the
// stretched step jumped by up to 0.46 at each pixel step of the staircase (a comb along the
// near-vertical leg of "mixed_edge", saw teeth on slants): 0.24-0.36 here at Width 4-8. What
// remains is the staircase itself, averaged along the edge over +-Width / 2 (up to 4 px).
void testHardEdges() {
  const int w = 900, h = 200;
  for (double slope : {0.0, 0.05, 0.11, 0.3, 0.81})
    for (double width : {2.0, 4.0, 6.0, 8.0}) {
      const double nrm = std::hypot(1.0, slope);
      std::vector<float> in(size_t(w) * h);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) in[size_t(y) * w + x] = 300.5 + slope * y - x >= 0 ? 1.0f : 0.0f;
      const std::vector<float> out = run(in, w, h, width);
      double worst = 0;
      int rowsDiffer = 0;
      for (int y = 10; y < h - 10; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          worst = std::max(worst, std::fabs(out[i] - clamp01(0.5 + (300.5 + slope * y - x) / nrm / width)));
          rowsDiffer += slope == 0.0 && out[i] != out[size_t(10) * w + x];
        }
      std::printf("hard slant %.2f -> %.0f px: max error %.3f\n", slope, width, worst);
      if (slope == 0.0) CHECK(rowsDiffer == 0 && worst < 1e-6);
      else CHECK(worst < (width < 3 ? 0.25 : 0.11));
    }
}

// An acute hard wedge (a 0 gap in a solid, 25 deg, like the hip gap) widened to 6 px: towards
// the tip the gap gets too narrow for 6 px. Before, the thin-part rule switched the change off
// there and a hard stub stayed inside the soft flanks (steps of ~1 between neighbours); now the
// width is capped to the room, so it narrows smoothly into the tip. Largest step between
// 4-neighbours within 40 px of the tip (a 6 px flank alone steps ~1/6 per px; the last pixel
// of the tip, a sliver under 1 px, stays the input's own).
void testWedge() {
  const int w = 900, h = 200;
  const double ax = 200.5, ay = 100.25, half = 12.5 * M_PI / 180;
  std::vector<float> in(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      in[size_t(y) * w + x] = x > ax && std::fabs(std::atan2(y - ay, x - ax)) < half ? 0.0f : 1.0f;
  for (double width : {4.0, 6.0, 8.0}) {
    const std::vector<float> out = run(in, w, h, width);
    double step = 0;
    for (int y = 60; y < 140; ++y)
      for (int x = 170; x < 240; ++x) {
        const size_t i = size_t(y) * w + x;
        step = std::max({step, double(std::fabs(out[i] - out[i + 1])), double(std::fabs(out[i] - out[i + w]))});
      }
    std::printf("hard wedge 25 deg -> %.0f px: largest neighbour step near the tip %.3f\n", width, step);
    CHECK(step < (width < 5 ? 0.75 : 0.55));  // 1.0 before (a hard stub)
  }
}

// Amount blends: 0 = the input bit for bit, 0.5 = halfway.
void testAmount() {
  const int w = 900, h = 120;
  const std::vector<float> in = slanted(w, h, 0.3, 8.0);
  EdgeWidthStats st;
  CHECK(run(in, w, h, 2.0, 0.0, &st) == in && st.pixelsChanged == 0);
  const std::vector<float> full = run(in, w, h, 2.0), half = run(in, w, h, 2.0, 0.5);
  double worst = 0;
  for (size_t i = 0; i < in.size(); ++i) worst = std::max(worst, std::fabs(half[i] - (in[i] + 0.5 * (full[i] - in[i]))));
  CHECK(worst < 1e-5);
}

// The levels either side are kept: an edge from 0 to a grey 0.85 face stays 0 -> 0.85 (the face
// is not raised to 1), the face beyond the edge is untouched, the 0.5 contour stays.
void testLevels() {
  const int w = 900, h = 120;
  const std::vector<float> in = slanted(w, h, 0.3, 6.0, 0.85), out = run(in, w, h, 2.0);
  const double nrm = std::hypot(1.0, 0.3);
  double top = 0, worst = 0;
  int faceChanged = 0;
  for (int y = 10; y < h - 10; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const double d = (300 + 0.3 * y - x) / nrm, c = 6.0 * (0.5 / 0.85 - 0.5);  // contour at d = c
      top = std::max(top, double(out[i]));
      faceChanged += d > c + 2.0 + 1.5 && out[i] != in[i];
      // Ideal: the linear ramp 0 -> 0.85, 2 px wide, through 0.5 at d = c.
      worst = std::max(worst, std::fabs(out[i] - std::min(0.85, std::max(0.0, 0.5 + 0.85 * (d - c) / 2.0))));
    }
  std::printf("grey face 0.85, 6 px -> 2 px: max %.4f, max error %.3f\n", top, worst);
  CHECK(top <= 0.85f + 1e-6);
  CHECK(faceChanged == 0);
  CHECK(worst < 0.05);
}

// Thin parts and gaps narrower than the target width keep their own profile: a 1.5 px strand
// (peak 0.8) and a 2 px gap with Width 4; strands below 0.5 have no contour and are untouched.
void testThin() {
  const int w = 900, h = 100;
  std::vector<float> in(size_t(w) * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float& v = in[size_t(y) * w + x];
      v = float(0.8 * clamp01(1.0 - std::fabs(x - 100.0 - 0.2 * y) / 1.5));  // strand above 0.5
      v = std::max(v, float(0.4 * clamp01(1.0 - std::fabs(x - 200.0 - 0.1 * y) / 1.2)));  // below
      if (x > 400) v = float(1.0 - clamp01(1.0 - std::fabs(x - 600.0) / 2.0));  // a 2 px gap
    }
  const std::vector<float> out = run(in, w, h, 4.0);
  double worst = 0;
  for (size_t i = 0; i < in.size(); ++i)
    if (std::abs(int(i % w) - 400) > 6) worst = std::max(worst, std::fabs(double(out[i]) - in[i]));  // x = 400: a real edge
  std::printf("thin parts with Width 4: max change %.4f\n", worst);
  CHECK(worst < 1e-6);
}

// The frame border is no edge.
void testFrameBorder() {
  std::vector<float> full(60 * 40, 0.7f);
  CHECK(run(full, 60, 40, 2.0) == full);
  const int w = 800, h = 50;
  std::vector<float> half(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) half[size_t(y) * w + x] = float(clamp01(0.5 + (50.0 - x) / 8.0));
  // An 8 px ramp crossing 0.5 at x = 50, cut by the top and bottom: every row, the first and
  // last included, becomes the 2 px ramp clamp(0.5 + (50 - x) / 2).
  const std::vector<float> o = run(half, w, h, 2.0);
  double worst = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) worst = std::max(worst, std::fabs(o[size_t(y) * w + x] - clamp01(0.5 + (50.0 - x) / 2.0)));
  CHECK(worst < 0.01);
}

// Synthetic figure regions (synth.cpp), in 1080-line design units.
bool inHeadArea(double x, double y) { return std::fabs(x) < 260 && y > -560 && y < -200; }

// "mixed_edge" with Width 2: the truth has 2 px edges everywhere. Per region: mean and max
// error in the edge band (pixels where the input, the truth or the output is between 0 and 1),
// before and after. The hair area is left out (hair is the same in both).
void testMixedEdge() {
  const int w = 1920, h = 1080;
  const synth::Scene s = synth::make("mixed_edge", w, h);
  EdgeWidthStats st;
  const std::vector<float> out = run(s.matte, w, h, 2.0, 1.0, &st);
  auto region = [&](int x, int y) {  // synth.cpp mixedEdge: 0 hard, 1 head 1 px, 2 arm 8 px, 3 rest 2 px
    const double dx = x - w / 2.0, dy = y - h / 2.0;
    if (inHeadArea(dx, dy) && dy < -420) return -1;
    // The input switches profile abruptly at the regions' borders (a leg is half hard, half
    // 2 px): those seams are left out.
    if (std::fabs(dx + 40) < 8 || (dx > -40 && std::fabs(dy + 240) < 8) ||
        (dy > -150 && std::fabs(dx - 150) < 8) || (dx > 150 && std::fabs(dy + 150) < 8))
      return -1;
    return dx < -40 ? 0 : dy < -240 ? 1 : dx > 150 && dy > -150 ? 2 : 3;
  };
  const char* names[4] = {"hard left", "head 1 px", "arm 8 px", "rest 2 px"};
  double sum[4] = {0, 0, 0, 0}, before[4] = {0, 0, 0, 0}, worst[4] = {0, 0, 0, 0}, worstBefore[4] = {0, 0, 0, 0};
  int cnt[4] = {0, 0, 0, 0};
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int r = region(x, y);
      const size_t i = size_t(y) * w + x;
      auto soft = [](float v) { return v > 0.0f && v < 1.0f; };
      if (r < 0 || !(soft(s.matte[i]) || soft(s.truth[i]) || soft(out[i]))) continue;
      const double e = std::fabs(out[i] - s.truth[i]), b = std::fabs(s.matte[i] - s.truth[i]);
      sum[r] += e;
      before[r] += b;
      worst[r] = std::max(worst[r], e);
      worstBefore[r] = std::max(worstBefore[r], b);
      ++cnt[r];
    }
  for (int r = 0; r < 4; ++r) {
    std::printf("mixed_edge, Width 2, %-9s: %5d px, mean error %.4f (before %.4f), max %.3f (before %.3f)\n",
                names[r], cnt[r], sum[r] / cnt[r], before[r] / cnt[r], worst[r], worstBefore[r]);
    CHECK(sum[r] / cnt[r] < 0.6 * before[r] / cnt[r] || sum[r] / cnt[r] < 0.01);
  }
  // The hard left side keeps its pixel staircase as the contour (Contour Smooth's job).
  CHECK(sum[1] / cnt[1] < 0.03 && sum[2] / cnt[2] < 0.015 && sum[3] / cnt[3] < 0.01);
  CHECK(st.pixelsChanged > 0);
}

// "holes": the grey (0.85) side face is no edge: away from any contour it is untouched.
void testGreyFace() {
  const int w = 1920, h = 1080;
  const synth::Scene s = synth::make("holes", w, h);
  const std::vector<float> out = run(s.matte, w, h, 2.0);
  int face = 0, changed = 0;
  for (int y = 4; y < h - 4; ++y)
    for (int x = 4; x < w - 4; ++x) {
      const size_t i = size_t(y) * w + x;
      if (std::fabs(s.matte[i] - 0.85f) > 0.02f) continue;
      bool nearContour = false;  // within 4 px of a pixel below 0.5
      for (int dy = -4; dy <= 4 && !nearContour; ++dy)
        for (int dx = -4; dx <= 4; ++dx) nearContour |= s.matte[size_t(y + dy) * w + x + dx] < 0.5f;
      if (nearContour) continue;
      ++face;
      changed += out[i] != s.matte[i];
    }
  std::printf("holes, Width 2: grey face %d px away from contours, %d changed\n", face, changed);
  CHECK(face > 1000 && changed == 0);
}

void testInPlace() {
  const int w = 900, h = 100;
  std::vector<float> in = slanted(w, h, 0.3, 6.0);
  const std::vector<float> out = run(in, w, h, 2.0);
  EdgeWidthParams p;
  mr::edgeWidth(in.data(), in.data(), w, h, p);
  CHECK(in == out);
}

void testTiming() {
  const int w = 3840, h = 2160;
  const synth::Scene s = synth::make("all", w, h);
  std::vector<float> out(s.matte.size());
  EdgeWidthParams p;
  EdgeWidthStats st;
  const auto t0 = std::chrono::steady_clock::now();
  mr::edgeWidth(s.matte.data(), out.data(), w, h, p, &st);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("UHD edge width 2 px: %.1f ms, %lld px changed\n", ms, st.pixelsChanged);
  CHECK(st.pixelsChanged > 0);
  CHECK(ms < 60.0);  // indicative: other builds share this machine
}

}  // namespace

int main() {
  testSlanted();
  testHardEdges();
  testWedge();
  testAmount();
  testLevels();
  testThin();
  testFrameBorder();
  testMixedEdge();
  testGreyFace();
  testInPlace();
  testTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
