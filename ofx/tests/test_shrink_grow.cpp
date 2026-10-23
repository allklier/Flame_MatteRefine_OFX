// Shrink / Grow (A3) on mattes with known answers: soft discs and slanted edges moved by an
// exact amount, thin parts and gaps, holes, the frame border, hair, the synthetic "eroded" and
// "mixed_edge" scenes against their truth, and UHD timing.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "edge_reference.h"
#include "ops/edge_field.h"
#include "ops/shrink_grow.h"
#include "synth.h"
#include "old_edge_ops.h"
#include "equivalence.h"

using mr::ShrinkGrowParams;
using mr::ShrinkGrowStats;

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

// Soft discs: alpha = clamp(0.5 + (r - dist) / soft), so the 0.5 contour is the circle r.
std::vector<float> discs(int w, int h, const std::vector<std::array<double, 4>>& ds) {
  std::vector<float> m(size_t(w) * h, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (const auto& d : ds)
        m[size_t(y) * w + x] =
            std::max(m[size_t(y) * w + x], float(clamp01(0.5 + (d[2] - std::hypot(x - d[0], y - d[1])) / d[3])));
  return m;
}

std::vector<float> run(const std::vector<float>& in, int w, int h, double amount,
                       ShrinkGrowStats* st = nullptr) {
  std::vector<float> out(in.size());
  ShrinkGrowParams p;
  p.amountPx = amount;
  mr::shrinkGrow(in.data(), out.data(), w, h, p, st);
  return out;
}

// A disc moved by k is the disc of radius r + k with the same softness: hard (1 px), 4 px and
// 8 px edges, grow and shrink, one pass and several. Pixels farther than |k| + the edge's half
// width (+ 1.5 px: the distance model and resampling are good to ~0.1 px, not to 0) from the
// contour are untouched, bit for bit.
void testDiscs() {
  const int w = 700, h = 260;  // wide enough for the exact path (morphology.h)
  for (double soft : {1.0, 4.0, 8.0})
    for (double k : {3.0, -3.0, 2.5, -1.5, 7.0, -9.0}) {
      const std::vector<float> in = discs(w, h, {{130, 130, 60, soft}});
      const std::vector<float> want = discs(w, h, {{130, 130, 60 + k, soft}});
      ShrinkGrowStats st;
      const std::vector<float> out = run(in, w, h, k, &st);
      double worst = 0, sum = 0;
      int band = 0, farChanged = 0;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          const double e = std::fabs(out[i] - want[i]), d = std::hypot(x - 130.0, y - 130.0) - 60;
          worst = std::max(worst, e);
          if (std::fabs(d) < std::fabs(k) + soft) sum += e, ++band;
          if (std::fabs(d) > std::fabs(k) + soft / 2 + 1.5 && out[i] != in[i]) ++farChanged;
        }
      std::printf("disc soft %.0f amount %+.1f (%d passes): max error %.3f, mean in band %.4f\n",
                  soft, k, st.passes, worst, sum / band);
      // A 1 px edge is point-sampled: half a pixel off a sample its value is undetermined
      // (+-0.25 for any resampler, test_edge_field), and each further pass resamples that
      // again. Wider edges are resolved.
      CHECK(worst < (soft < 2 ? (st.passes > 1 ? 0.4 : 0.26) : st.passes > 1 ? 0.06 : 0.035));
      CHECK(sum / band < (soft < 2 ? 0.025 : 0.002));
      CHECK(farChanged == 0);
    }
}

// Aliasing: a straight antialiased edge at a slant, moved; the largest difference from the
// ideal moved edge. A staircase would show as an error of the order of the edge's step.
void testSlanted() {
  const int w = 900, h = 200;
  for (double slope : {0.11, 0.3, 0.81})
    for (double soft : {1.0, 3.0})
      for (double k : {2.5, -1.7}) {
        const double nrm = std::hypot(1.0, slope);
        auto dist = [&](int x, int y) { return (100 + slope * y - x) / nrm; };  // > 0 inside (left)
        std::vector<float> in(size_t(w) * h), want(in.size());
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x) {
            in[size_t(y) * w + x] = float(clamp01(0.5 + dist(x, y) / soft));
            want[size_t(y) * w + x] = float(clamp01(0.5 + (dist(x, y) + k) / soft));
          }
        const std::vector<float> out = run(in, w, h, k);
        double worst = 0;
        for (int y = 10; y < h - 10; ++y)
          for (int x = 0; x < w; ++x) worst = std::max(worst, double(std::fabs(out[size_t(y) * w + x] - want[size_t(y) * w + x])));
        std::printf("slant %.2f soft %.0f amount %+.1f: max error %.3f\n", slope, soft, k, worst);
        // edge_field shiftAlongNormal's 3x3 window: up to 0.065 on the 3 px ramp at slope 0.11.
        CHECK(worst < (soft < 2 ? 0.26 : 0.045));
      }
}

// Thin parts and gaps (morphological): a 3 px strip shrunk by 2 is gone, by 1 it is 1 px; a
// 3 px gap grown by 2 closes, by 1 it stays open; a hole shrinks by the amount and closes when
// narrower than twice it.
void testThinPartsGapsHoles() {
  const int w = 1000, h = 80;
  std::vector<float> strip(size_t(w) * h), gap(strip.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      strip[size_t(y) * w + x] = float(clamp01(0.5 + (1.5 - std::fabs(x - 60.0)) / 1.0));
      gap[size_t(y) * w + x] = 1.0f - strip[size_t(y) * w + x];
    }
  auto column = [&](const std::vector<float>& m, int x) { return m[size_t(h / 2) * w + x]; };
  std::vector<float> o = run(strip, w, h, -2.0);
  CHECK(*std::max_element(o.begin(), o.end()) < 0.02f);
  o = run(strip, w, h, -1.0);
  CHECK(column(o, 60) > 0.9f && column(o, 59) < 0.1f && column(o, 61) < 0.1f);
  o = run(gap, w, h, 2.0);
  CHECK(*std::min_element(o.begin(), o.end()) > 0.98f);
  o = run(gap, w, h, 1.0);
  CHECK(column(o, 60) < 0.1f && column(o, 59) > 0.9f && column(o, 61) > 0.9f);

  // A hole of radius 12 (2 px soft) in a solid: grow 3 -> radius 9; grow 13 -> closed.
  const int W = 400, H = 160;
  std::vector<float> hole = discs(W, H, {{80, 80, 12, 2}}), want = discs(W, H, {{80, 80, 9, 2}});
  for (float& v : hole) v = 1.0f - v;
  o = run(hole, W, H, 3.0);
  double worst = 0;
  for (size_t i = 0; i < o.size(); ++i) worst = std::max(worst, std::fabs(double(o[i]) - (1.0 - want[i])));
  std::printf("hole r 12 grown 3: max error %.3f\n", worst);
  CHECK(worst < 0.06);
  o = run(hole, W, H, 13.0);  // the profile at 1 px inside: 1.0
  CHECK(*std::min_element(o.begin(), o.end()) > 0.95f);
}

// The frame border is no edge: a full frame stays as it is, and a subject cut by the frame moves
// only along its real edge, the same in the first and last rows as in the middle.
void testFrameBorder() {
  std::vector<float> full(60 * 40, 0.9f);
  CHECK(run(full, 60, 40, 3.0) == full && run(full, 60, 40, -3.0) == full);
  const int w = 800, h = 50;
  std::vector<float> half(size_t(w) * h), want(half.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      half[size_t(y) * w + x] = float(clamp01(0.5 + (50.0 - x) / 2.0));
      want[size_t(y) * w + x] = float(clamp01(0.5 + (53.0 - x) / 2.0));
    }
  const std::vector<float> o = run(half, w, h, 3.0);
  double worst = 0;
  for (size_t i = 0; i < o.size(); ++i) worst = std::max(worst, std::fabs(double(o[i]) - want[i]));
  CHECK(worst < 0.01);
}

// Hair: strands below 0.5 have no contour. Away from an edge they are untouched by grow and
// shrink; a strand growing out of the edge is kept by grow (no pixel gets lower) and may be
// thinned by shrink (no pixel gets higher). In place gives the same as out of place.
void testHair() {
  const int w = 700, h = 200;
  std::vector<float> in = discs(w, h, {{80, 100, 40, 2}});
  auto strand = [&](double x0, double y0, double x1, double y1, double peak) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double vx = x1 - x0, vy = y1 - y0;
        const double t = clamp01(((x - x0) * vx + (y - y0) * vy) / (vx * vx + vy * vy));
        const double d = std::hypot(x - x0 - t * vx, y - y0 - t * vy);
        float& v = in[size_t(y) * w + x];
        v = std::max(v, float(peak * clamp01(1.0 - d / 1.2)));
      }
  };
  strand(170, 20, 220, 180, 0.4);  // far from the disc
  strand(150, 30, 175, 170, 0.3);
  strand(118, 100, 160, 110, 0.45);  // grows out of the disc's edge (x = 120)
  for (double k : {3.0, -3.0}) {
    const std::vector<float> out = run(in, w, h, k);
    int farChanged = 0, wrongWay = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (x >= 145 && out[i] != in[i]) ++farChanged;
        wrongWay += k > 0 ? out[i] < in[i] : out[i] > in[i];
      }
    CHECK(farChanged == 0);
    CHECK(wrongWay == 0);
    std::vector<float> inPlace = in;
    ShrinkGrowParams p;
    p.amountPx = k;
    mr::shrinkGrow(inPlace.data(), inPlace.data(), w, h, p);
    CHECK(inPlace == out);
  }
  // The attached strand survives grow beyond the grown edge (x = 123 .. 160; at x = 140 it
  // runs through y = 105.2).
  const std::vector<float> grown = run(in, w, h, 3.0);
  CHECK(in[size_t(105) * w + 140] > 0.3f && grown[size_t(105) * w + 140] >= in[size_t(105) * w + 140]);
  CHECK(run(in, w, h, 0.0) == in);
}

// Strands: parts of the >= 0.5 area under ~3 px wide are detail. A 2 px strand (peak 0.9) keeps
// its shape under grow and shrink (not a 2 x Amount wide band, not erased); a 3.5 px wide part
// is body and moves. Hair below 0.5 crossing the band, at a slant or nearly along the normal
// (where it looks like a long soft tail of the edge), is not carried outward as a
// second copy: grow gives nothing above max(input, the grown disc) more than 4 px off the grown
// contour. (At the root, where the strand raises the edge's own profile, that bump moves out
// with the edge: up to ~0.3 over a few pixels.)
void testStrands() {
  const int w = 700, h = 200;
  auto line = [&](std::vector<float>& m, double x0, double y0, double x1, double y1, double halfW, double peak) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double vx = x1 - x0, vy = y1 - y0;
        const double t = clamp01(((x - x0) * vx + (y - y0) * vy) / (vx * vx + vy * vy));
        const double d = std::hypot(x - x0 - t * vx, y - y0 - t * vy);
        float& v = m[size_t(y) * w + x];
        v = std::max(v, float(peak * clamp01(0.5 + (halfW - d))));
      }
  };
  std::vector<float> in(size_t(w) * h, 0.0f);
  line(in, 300, 20, 330, 180, 1.0, 0.9);   // a 2 px strand
  line(in, 450, 20, 480, 180, 1.75, 1.0);  // a 3.5 px wide part
  int strandGrown = 0, partGrown = 0;
  for (double k : {3.0, -3.0}) {
    const std::vector<float> out = run(in, w, h, k);
    double strandChange = 0;
    for (int y = 30; y < 170; ++y)
      for (int x = 280; x < 350; ++x) strandChange = std::max(strandChange, std::fabs(double(out[size_t(y) * w + x]) - in[size_t(y) * w + x]));
    CHECK(strandChange < 1e-6);
    for (int y = 30; y < 170; ++y)
      for (int x = 430; x < 500; ++x) (k > 0 ? partGrown : strandGrown) += out[size_t(y) * w + x] != in[size_t(y) * w + x];
  }
  CHECK(partGrown > 500 && strandGrown > 500);  // the 3.5 px part grows, and vanishes on shrink 3

  std::vector<float> hair = discs(w, h, {{100, 100, 40, 2}});
  const std::vector<float> want = discs(w, h, {{100, 100, 43, 2}});
  line(hair, 135, 85, 150, 30, 0.6, 0.4);  // leaves the edge at ~50 deg to the normal
  line(hair, 65, 85, 24, 56, 0.6, 0.4);    // ~12 deg: along the normal it looks like a soft tail
  const std::vector<float> out = run(hair, w, h, 3.0);
  double ghost = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      if (std::fabs(std::hypot(x - 100.0, y - 100.0) - 43.0) > 4.0)
        ghost = std::max(ghost, out[i] - std::max(double(hair[i]), double(want[i])));
    }
  std::printf("strands: hair below 0.5 at a slant, grow 3: largest rise above input and grown disc, > 4 px off the contour: %.3f\n", ghost);
  CHECK(ghost < 0.06);
}

// signedEdgeDistanceThreaded equals signedEdgeDistance bit for bit where the latter takes its
// exact path (a few large soft discs, wide band), and stays exact where it falls back (hair).
void testThreadedDistance() {
  const int w = 400, h = 700;
  const std::vector<float> in = discs(w, h, {{200, 250, 120, 3}, {150, 520, 60, 1}, {300, 255, 30, 6}});
  const std::vector<float> a = mr::signedEdgeDistance(in.data(), w, h, 0.5f, 9.0);
  const std::unique_ptr<float[]> b = mr::signedEdgeDistanceThreaded(in.data(), w, h, 0.5f, 9.0);
  CHECK(std::equal(a.begin(), a.end(), b.get()));
  // Many contours (sparse dark specks), where signedEdgeDistance takes the grid distance (more
  // than 2 frames of writes) but this one stays exact (less than 12): against the brute-force
  // model of edge_reference.h.
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  int bad = 0, cases = 0, gridDiffers = 0;
  for (int trial = 0; trial < 3; ++trial) {
    const int W = 300 + trial, H = 150;
    std::vector<float> m(size_t(W) * H);
    for (float& v : m) v = u(rng) < 0.03f ? u(rng) * 0.5f : 0.5f + u(rng) * 0.5f;
    const double band = 2.0;
    const std::unique_ptr<float[]> sd = mr::signedEdgeDistanceThreaded(m.data(), W, H, 0.5f, band);
    const std::vector<float> grid = mr::signedEdgeDistance(m.data(), W, H, 0.5f, band);
    const std::vector<ref::Boundary> belowEdge = ref::boundary(m, W, H, 0.5f, true);
    const std::vector<ref::Boundary> aboveEdge = ref::boundary(m, W, H, 0.5f, false);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const size_t i = size_t(y) * W + x;
        const bool inside = m[i] >= 0.5f;
        const double want = (inside ? 1 : -1) * std::min(ref::distance(inside ? belowEdge : aboveEdge, x, y), band + 1.0);
        if (std::fabs(want) <= band) {
          ++cases;
          bad += std::fabs(sd[i] - want) > 1e-4;
        } else {
          bad += std::fabs(sd[i]) <= band;  // beyond the band: reported as beyond
        }
        gridDiffers += std::fabs(grid[i] - want) > 1e-3;
      }
  }
  CHECK(gridDiffers > 1000);  // the case this is for
  std::printf("threaded distance vs brute force: %d px, %d off\n", cases, bad);
  CHECK(bad == 0);
}

// Synthetic figure regions (synth.cpp), in 1080-line design units.
bool inHeadArea(double x, double y) { return std::fabs(x) < 260 && y > -560 && y < -200; }

// "eroded": the matte 3 px too small with its profile kept; Grow 3 must bring it back. Hair is
// not eroded in the scene (it lies on top), so the head area is reported apart: grow carries
// the hair near the head outward with the edge.
void testEroded() {
  const int w = 1920, h = 1080;
  const synth::Scene s = synth::make("eroded", w, h);
  const std::vector<float> out = run(s.matte, w, h, 3.0);
  double worst[2] = {0, 0}, sum[2] = {0, 0}, before[2] = {0, 0};
  int cnt[2] = {0, 0}, off[2] = {0, 0};
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const float t = s.truth[i], m = s.matte[i];
      if ((t <= 0 || t >= 1) && (m <= 0 || m >= 1) && (out[i] <= 0 || out[i] >= 1)) continue;
      const int r = inHeadArea(x - w / 2.0, y - h / 2.0);
      const double e = std::fabs(out[i] - t);
      worst[r] = std::max(worst[r], e);
      off[r] += e > 0.1;
      sum[r] += e;
      before[r] += std::fabs(m - t);
      ++cnt[r];
    }
  for (int r = 0; r < 2; ++r)
    std::printf("eroded, grow 3, %s: edge band %d px, mean error %.4f (before %.4f), max %.3f, %d px off by > 0.1\n",
                r ? "head + hair" : "body", cnt[r], sum[r] / cnt[r], before[r] / cnt[r], worst[r], off[r]);
  CHECK(sum[0] / cnt[0] < 0.005 && off[0] < 10);  // off: a 1 px seam of exactly 0.5 in the hip gap
  CHECK(sum[1] / cnt[1] < before[1] / cnt[1] / 2);
}

// "mixed_edge": each region keeps its own edge width when grown: mean width = pixels strictly
// inside the 5-95 % range per straddling 4-neighbour pair (a length measure).
void testMixedEdge() {
  const int w = 1920, h = 1080;
  const synth::Scene s = synth::make("mixed_edge", w, h);
  const std::vector<float> out = run(s.matte, w, h, 3.0);
  auto region = [&](int x, int y) {  // synth.cpp mixedEdge: 0 hard, 1 head 1 px, 2 arm 8 px, 3 rest 2 px
    const double dx = x - w / 2.0, dy = y - h / 2.0;
    if (inHeadArea(dx, dy) && dy < -420) return -1;  // the hair
    return dx < -40 ? 0 : dy < -240 ? 1 : dx > 150 && dy > -150 ? 2 : 3;
  };
  const char* names[4] = {"hard left", "head 1 px", "arm 8 px", "rest 2 px"};
  for (const std::vector<float>* m : {&s.matte, &out}) {
    double soft[4] = {0, 0, 0, 0}, len[4] = {0, 0, 0, 0};
    for (int y = 0; y + 1 < h; ++y)
      for (int x = 0; x + 1 < w; ++x) {
        const int r = region(x, y);
        if (r < 0) continue;
        const float v = (*m)[size_t(y) * w + x];
        soft[r] += v > 0.05f && v < 0.95f;
        len[r] += ((*m)[size_t(y) * w + x + 1] >= 0.5f) != (v >= 0.5f);
        len[r] += ((*m)[size_t(y + 1) * w + x] >= 0.5f) != (v >= 0.5f);
      }
    std::printf("mixed_edge %s:", m == &out ? "grown 3" : "input  ");
    for (int r = 0; r < 4; ++r) std::printf("  %s %.2f", names[r], soft[r] / len[r]);
    std::printf("  (px of 5-95 %% per contour step)\n");
    static double widthBefore[4];
    for (int r = 0; r < 4; ++r) {
      const double wdt = soft[r] / len[r];
      if (m == &s.matte) {
        widthBefore[r] = wdt;
        continue;
      }
      if (r == 0) CHECK(wdt < 1.0);  // hard stays hard (a 1 px antialiased step at most)
      else CHECK(std::fabs(wdt / widthBefore[r] - 1.0) < 0.15);
    }
  }
}

void testTiming() {
  const int w = 3840, h = 2160;
  const synth::Scene s = synth::make("all", w, h);
  std::vector<float> out(s.matte.size());
  for (double k : {3.0, -3.0, 10.0}) {
    ShrinkGrowParams p;
    p.amountPx = k;
    ShrinkGrowStats st;
    const auto t0 = std::chrono::steady_clock::now();
    mr::shrinkGrow(s.matte.data(), out.data(), w, h, p, &st);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("UHD shrink/grow %+.0f px: %.1f ms, %d passes, %lld px changed\n", k, ms, st.passes, st.pixelsChanged);
    CHECK(st.pixelsChanged > 0);
    if (std::fabs(k) <= 4) CHECK(ms < 60.0);  // indicative: other builds share this machine
  }
}

// Equivalence with the code before the speed work (old_edge_ops.h): bit for bit, out of place
// and in place, on the synthetic scenes at 1080 and on random mattes, sizes and amounts.
void testEquivalence() {
  long long cases = 0, bad = 0, active = 0, skipped = 0;
  auto check = [&](const std::vector<float>& in, int w, int h, double amount) {
    mr::ShrinkGrowParams p;
    p.amountPx = amount;
    old_::sg::ShrinkGrowParams q;
    q.amountPx = amount;
    std::vector<float> a(in.size()), b(in.size()), c = in;
    mr::ShrinkGrowStats sa, sc;
    old_::sg::ShrinkGrowStats sb;
    mr::shrinkGrow(in.data(), a.data(), w, h, p, &sa);
    old_::sg::shrinkGrow(in.data(), b.data(), w, h, q, &sb);
    mr::shrinkGrow(c.data(), c.data(), w, h, p, &sc);  // in place
    const long long d = eq::bitsDiffer(a.data(), b.data(), a.size()) + eq::bitsDiffer(c.data(), b.data(), c.size());
    const bool same = d == 0 && sa.pixelsChanged == sb.pixelsChanged && sc.pixelsChanged == sb.pixelsChanged &&
                      sa.passes == sb.passes && sc.passes == sb.passes;
    if (!same) std::printf("  differs: %dx%d amount %.3f: %lld px\n", w, h, amount, d);
    ++cases;
    bad += !same;
    active += sb.pixelsChanged > 0;
  };
  const char* scenes[] = {"all", "mixed_edge", "eroded", "chatter", "holes"};
  for (const char* name : scenes) {
    const synth::Scene s = synth::make(name, 1920, 1080);
    for (double k : {3.0, -3.0, 10.0, -7.5, 0.6, 0.0}) check(s.matte, 1920, 1080, k);
  }
  std::mt19937 rng(20261023);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int t = 0; t < 40; ++t) {
    const int w = t < 4 ? 1 + t * 3 : 20 + int(u(rng) * 380), h = t < 4 ? 40 - t * 9 : 10 + int(u(rng) * 290);
    const std::vector<float> m = eq::randomMatte(rng, w, h);
    check(m, w, h, (u(rng) - 0.5) * 24.0);
  }
  std::printf("equivalence with the old shrink/grow: %lld cases (%lld change pixels), %lld differ\n", cases, active, bad);
  CHECK(bad == 0);
}

}  // namespace

int main() {
  testDiscs();
  testSlanted();
  testThinPartsGapsHoles();
  testFrameBorder();
  testHair();
  testStrands();
  testThreadedDistance();
  testEroded();
  testMixedEdge();
  testEquivalence();
  testTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
