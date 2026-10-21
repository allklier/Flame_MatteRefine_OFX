// Fill Holes (decisions 7-10) on synthetic mattes with known answers, plus UHD timing.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "ops/fill_holes.h"

using mr::FillHolesParams;
using mr::FillHolesStats;

namespace {

int failures = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

struct Img {
  int w, h;
  std::vector<float> v;
  Img(int w, int h, float fill = 0.0f) : w(w), h(h), v(size_t(w) * h, fill) {}
  float& at(int x, int y) { return v[size_t(y) * w + x]; }
  float at(int x, int y) const { return v[size_t(y) * w + x]; }
  void rect(int x0, int y0, int x1, int y1, float val) {  // inclusive
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) at(x, y) = val;
  }
  void disc(double cx, double cy, double r, float val) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) at(x, y) = val;
  }
};

Img run(const Img& in, const FillHolesParams& p, FillHolesStats* st = nullptr) {
  Img out(in.w, in.h);
  mr::fillHoles(in.v.data(), out.v.data(), in.w, in.h, p, st);
  return out;
}

// Pixels where a and b differ.
int diffs(const Img& a, const Img& b) {
  int n = 0;
  for (size_t i = 0; i < a.v.size(); ++i) n += a.v[i] != b.v[i];
  return n;
}

FillHolesParams noLimit() {
  FillHolesParams p;
  p.maxHolePercent = 0.0;
  p.rimPx = 0.0;
  return p;
}

void testSimpleHole() {
  Img in(100, 100);
  in.rect(20, 20, 79, 79, 1.0f);
  in.rect(45, 45, 54, 54, 0.0f);  // 10x10 hole
  FillHolesStats st;
  const Img out = run(in, noLimit(), &st);
  Img want = in;
  want.rect(45, 45, 54, 54, 1.0f);
  CHECK(diffs(out, want) == 0);
  CHECK(st.holesFilled == 1 && st.pixelsChanged == 100);
}

// The outer soft edge is bit-identical; only the enclosed hole changes.
void testOuterEdgeUntouched() {
  Img in(200, 200);
  for (int y = 0; y < 200; ++y)
    for (int x = 0; x < 200; ++x) {
      const double d = std::hypot(x - 100.0, y - 100.0);
      in.at(x, y) = float(std::min(1.0, std::max(0.0, (80.0 - d) / 6.0)));  // soft 6 px edge
    }
  in.disc(100, 100, 10, 0.0f);   // hard hole
  in.disc(130, 90, 4, 0.3f);     // grey hole below threshold
  FillHolesParams p = noLimit();
  p.rimPx = 2.0;
  const Img out = run(in, p);
  int changedOutsideHoles = 0, holeLeft = 0;
  for (int y = 0; y < 200; ++y)
    for (int x = 0; x < 200; ++x) {
      const double d1 = std::hypot(x - 100.0, y - 100.0), d2 = std::hypot(x - 130.0, y - 90.0);
      const bool nearHole = d1 <= 13.0 || d2 <= 7.0;
      if (!nearHole && out.at(x, y) != in.at(x, y)) ++changedOutsideHoles;
      if ((d1 <= 10.0 || d2 <= 4.0) && out.at(x, y) != 1.0f) ++holeLeft;
      CHECK(out.at(x, y) >= in.at(x, y));
    }
  CHECK(changedOutsideHoles == 0);
  CHECK(holeLeft == 0);
}

void testMaxHoleSize() {
  Img in(100, 100);
  in.rect(5, 5, 94, 94, 1.0f);
  in.rect(10, 10, 19, 19, 0.0f);  // 100 px = 1 % of the frame
  in.rect(40, 40, 69, 69, 0.0f);  // 900 px = 9 %
  FillHolesParams p = noLimit();
  p.maxHolePercent = 1.0;
  FillHolesStats st;
  Img out = run(in, p, &st);
  CHECK(out.at(15, 15) == 1.0f);  // exactly at the limit: filled
  CHECK(out.at(55, 55) == 0.0f);  // too large: open
  CHECK(st.holesFilled == 1 && st.holesTooLarge == 1);
  p.maxHolePercent = 0.0;  // no limit
  out = run(in, p);
  CHECK(out.at(55, 55) == 1.0f);
}

// A white edge that only touches diagonally still seals the hole (8-connected solid).
void testDiagonalSeal() {
  Img in(40, 40);
  // A diamond outline, 1 px, pixels touching only at corners.
  for (int i = 0; i <= 10; ++i) {
    in.at(20 + i, 10 + i) = 1.0f;
    in.at(20 - i, 10 + i) = 1.0f;
    in.at(20 + i, 30 - i) = 1.0f;
    in.at(20 - i, 30 - i) = 1.0f;
  }
  const Img out = run(in, noLimit());
  CHECK(out.at(20, 20) == 1.0f);
  CHECK(out.at(2, 2) == 0.0f);
}

// A hole open to the bottom frame edge fills only when that edge is closed.
void testClosedEdges() {
  Img in(60, 60);
  in.rect(10, 20, 49, 59, 1.0f);  // subject cut by the bottom edge
  in.rect(25, 40, 34, 59, 0.0f);  // gap reaching the bottom edge
  FillHolesParams p = noLimit();
  Img out = run(in, p);
  CHECK(out.at(30, 50) == 0.0f);
  p.closedBottom = true;
  out = run(in, p);
  CHECK(out.at(30, 50) == 1.0f && out.at(30, 59) == 1.0f);
  CHECK(out.at(5, 50) == 0.0f);  // still open outside the subject (left edge open)
  // All edges closed: the background is one enclosed area too; Max Hole % keeps it open.
  p.closedLeft = p.closedRight = p.closedTop = true;
  p.maxHolePercent = 10.0;  // the gap is 200 px (5.6 %), the background ~55 %
  out = run(in, p);
  CHECK(out.at(30, 50) == 1.0f && out.at(5, 50) == 0.0f);
}

// Bridge Gaps: a ring with a 3 px gap fills with Bridge 4; an outer notch of the same width stays.
void testBridgeGaps() {
  Img in(120, 120);
  in.disc(60, 60, 40, 1.0f);
  in.disc(60, 60, 25, 0.0f);       // hole r 25
  in.rect(59, 15, 61, 37, 0.0f);   // 3 px gap from the hole through the top of the ring
  in.rect(15, 59, 30, 61, 0.0f);   // 3 px wide notch into the left side, not reaching the hole
  FillHolesParams p = noLimit();
  Img out = run(in, p);
  CHECK(out.at(60, 60) == 0.0f);  // open through the gap
  p.bridgePx = 4.0;
  FillHolesStats st;
  out = run(in, p, &st);
  CHECK(out.at(60, 60) == 1.0f);
  CHECK(out.at(60, 36) == 1.0f);  // inside the gap
  CHECK(out.at(22, 60) == 0.0f);  // the notch is the outside edge: untouched
  CHECK(out.at(5, 5) == 0.0f);
  int leftChanged = 0;  // left of the hole (x < 35), incl. the notch
  for (int y = 0; y < 120; ++y)
    for (int x = 0; x < 34; ++x) leftChanged += out.at(x, y) != in.at(x, y);
  CHECK(leftChanged == 0);
  // A gap wider than the bridge stays open.
  p.bridgePx = 2.0;
  out = run(in, p);
  CHECK(out.at(60, 60) == 0.0f);
}

// A hole smaller than the bridge width (the closing swallows it) still fills.
void testSmallHoleWithBridge() {
  Img in(60, 60);
  in.rect(10, 10, 49, 49, 1.0f);
  in.rect(29, 29, 30, 30, 0.0f);  // 2x2 hole
  FillHolesParams p = noLimit();
  p.bridgePx = 10.0;
  const Img out = run(in, p);
  CHECK(out.at(29, 29) == 1.0f && out.at(30, 30) == 1.0f);
  CHECK(diffs(out, in) == 4);
}

// Rim lifts solid pixels within Rim px of a filled hole, not further, never outside.
void testRim() {
  Img in(80, 80);
  in.rect(10, 10, 69, 69, 1.0f);
  in.rect(30, 30, 49, 49, 0.7f);  // soft rim, above threshold
  in.rect(35, 35, 44, 44, 0.0f);  // hole
  in.rect(10, 10, 69, 11, 0.7f);  // soft outer edge rows (must stay)
  FillHolesParams p = noLimit();
  p.rimPx = 0.0;
  Img out = run(in, p);
  CHECK(out.at(40, 40) == 1.0f && out.at(34, 40) == 0.7f);
  p.rimPx = 2.0;
  out = run(in, p);
  CHECK(out.at(34, 40) == 1.0f && out.at(33, 40) == 1.0f);  // 1 and 2 px from the hole
  CHECK(out.at(32, 40) == 0.7f);                              // 3 px: kept
  CHECK(out.at(40, 10) == 0.7f && out.at(40, 11) == 0.7f);    // outer edge untouched
}

// Never lowers a pixel, on noise with all features on.
void testNeverLowers() {
  Img in(150, 100);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for (auto& v : in.v) v = u(rng) < 0.6f ? 1.0f : u(rng);
  FillHolesParams p;
  p.bridgePx = 3.0;
  p.rimPx = 1.5;
  p.closedBottom = true;
  const Img out = run(in, p);
  int lower = 0;
  for (size_t i = 0; i < in.v.size(); ++i) lower += out.v[i] < in.v[i];
  CHECK(lower == 0);
  // In place gives the same result.
  Img inPlace = in;
  mr::fillHoles(inPlace.v.data(), inPlace.v.data(), in.w, in.h, p);
  CHECK(diffs(inPlace, out) == 0);
}

// Bridge 0 against a naive reference (per-pixel BFS from the open edges, holes by size,
// brute-force rim distance) on random mattes: checks the span fill and the distance transform.
Img reference(const Img& in, const FillHolesParams& p) {
  const int w = in.w, h = in.h;
  std::vector<int> outside(size_t(w) * h, 0), label(size_t(w) * h, -1), q;
  auto solid = [&](int x, int y) { return in.at(x, y) >= p.threshold; };
  auto push = [&](int x, int y) {
    if (x >= 0 && y >= 0 && x < w && y < h && !solid(x, y) && !outside[size_t(y) * w + x]) {
      outside[size_t(y) * w + x] = 1;
      q.push_back(y * w + x);
    }
  };
  for (int x = 0; x < w; ++x) {
    if (!p.closedTop) push(x, 0);
    if (!p.closedBottom) push(x, h - 1);
  }
  for (int y = 0; y < h; ++y) {
    if (!p.closedLeft) push(0, y);
    if (!p.closedRight) push(w - 1, y);
  }
  for (size_t i = 0; i < q.size(); ++i) {
    const int x = q[i] % w, y = q[i] / w;
    push(x - 1, y), push(x + 1, y), push(x, y - 1), push(x, y + 1);
  }
  std::vector<int> fill(size_t(w) * h, 0);
  for (int s = 0; s < w * h; ++s) {
    if (solid(s % w, s / w) || outside[s] || label[s] >= 0) continue;
    std::vector<int> c = {s};
    label[s] = s;
    for (size_t i = 0; i < c.size(); ++i) {
      const int x = c[i] % w, y = c[i] / w;
      const int nx[4] = {x - 1, x + 1, x, x}, ny[4] = {y, y, y - 1, y + 1};
      for (int k = 0; k < 4; ++k) {
        if (nx[k] < 0 || ny[k] < 0 || nx[k] >= w || ny[k] >= h) continue;
        const int j = ny[k] * w + nx[k];
        if (!solid(nx[k], ny[k]) && !outside[j] && label[j] < 0) label[j] = s, c.push_back(j);
      }
    }
    if (p.maxHolePercent > 0 && double(c.size()) > p.maxHolePercent / 100.0 * w * h) continue;
    for (int j : c) fill[j] = 1;
  }
  Img out = in;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      bool f = fill[size_t(y) * w + x];
      for (int j = 0; !f && solid(x, y) && j < w * h; ++j)
        f = fill[j] && (j % w - x) * (j % w - x) + (j / w - y) * (j / w - y) <= p.rimPx * p.rimPx;
      if (f) out.at(x, y) = 1.0f;
    }
  return out;
}

void testAgainstReference() {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  int bad = 0;
  for (int trial = 0; trial < 40; ++trial) {
    Img in(37 + trial % 5, 23 + trial % 7);
    for (auto& v : in.v) v = u(rng) < 0.55f ? 1.0f : u(rng) * 0.8f;
    FillHolesParams p;
    p.maxHolePercent = trial % 3 == 0 ? 0.0 : 0.5 + trial % 4;
    p.rimPx = (trial % 4) * 0.75;
    p.closedLeft = trial & 1, p.closedTop = trial & 2, p.closedRight = trial & 4,
    p.closedBottom = trial & 8;
    bad += diffs(run(in, p), reference(in, p)) != 0;
  }
  // Sparse holes with soft rims: the rim dilation takes the disc-stamp path (noise above
  // takes the distance-transform fallback).
  for (int trial = 0; trial < 8; ++trial) {
    Img in(90, 60);
    in.rect(5, 5, 84, 54, 1.0f);
    for (int k = 0; k < 6; ++k) {
      const double cx = 12 + rng() % 66, cy = 12 + rng() % 36;
      in.disc(cx, cy, 4.5, 0.8f);
      in.disc(cx, cy, 1.0 + rng() % 3, 0.1f);
    }
    FillHolesParams p;
    p.maxHolePercent = 0.0;
    p.rimPx = 1.0 + trial * 0.6;
    bad += diffs(run(in, p), reference(in, p)) != 0;
  }
  CHECK(bad == 0);
}

void testEmpty() {
  Img in(1, 1, 0.0f);
  CHECK(run(in, FillHolesParams()).at(0, 0) == 0.0f);
  mr::fillHoles(nullptr, nullptr, 0, 0, FillHolesParams());
}

// A UHD matte: a large soft subject with 200 small holes. Decision 9 target: < 50 ms with
// Bridge 0; the check allows 100 ms (decision 5) for slower machines.
void testTiming() {
  const int w = 3840, h = 2160;
  Img in(w, h);
  in.disc(1920, 1080, 900, 1.0f);
  std::mt19937 rng(3);
  for (int i = 0; i < 200; ++i)
    in.disc(1920 + int(rng() % 1200) - 600, 1080 + int(rng() % 1200) - 600, 2 + rng() % 12, 0.0f);
  FillHolesParams p;
  for (double bridge : {0.0, 8.0}) {
    p.bridgePx = bridge;
    FillHolesStats st;
    Img out(w, h);
    const auto t0 = std::chrono::steady_clock::now();
    mr::fillHoles(in.v.data(), out.v.data(), w, h, p, &st);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("UHD bridge %.0f px: %.1f ms, %d holes filled, %lld px\n", bridge, ms,
                st.holesFilled, st.pixelsChanged);
    CHECK(st.holesFilled > 0);
    if (bridge == 0.0) CHECK(ms < 100.0);
  }
}

}  // namespace

int main() {
  testSimpleHole();
  testOuterEdgeUntouched();
  testMaxHoleSize();
  testDiagonalSeal();
  testClosedEdges();
  testBridgeGaps();
  testSmallHoleWithBridge();
  testRim();
  testNeverLowers();
  testAgainstReference();
  testEmpty();
  testTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
