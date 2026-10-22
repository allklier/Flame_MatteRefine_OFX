// Clean Background + Background Crush (CLAUDE.md "Plugin 1 operations", A1/A2): known answers,
// a brute-force island reference, Crush against 1 - Solid Core(1 - matte), the synthetic
// "specks" and "all" scenes against their truth, and UHD timing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "ops/clean_background.h"
#include "ops/morphology.h"
#include "ops/solid_core.h"
#include "synth.h"

using mr::CleanBackgroundParams;
using mr::CleanBackgroundStats;

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

Img run(const Img& in, const CleanBackgroundParams& p, CleanBackgroundStats* st = nullptr) {
  Img out(in.w, in.h);
  mr::cleanBackground(in.v.data(), out.v.data(), in.w, in.h, p, st);
  return out;
}

CleanBackgroundParams islandsOnly(double maxPercent) {
  CleanBackgroundParams p;
  p.maxIslandPercent = maxPercent;
  p.crushPx = 0.0;
  return p;
}

// A soft disc: v inside radius r, a linear ramp of `soft` px to 0 outside.
void disc(Img& im, double cx, double cy, double r, double soft, float v) {
  for (int y = 0; y < im.h; ++y)
    for (int x = 0; x < im.w; ++x) {
      const double d = std::hypot(x - cx, y - cy);
      const float a = float(v * std::min(1.0, std::max(0.0, (r + soft - d) / soft)));
      im.at(x, y) = std::max(im.at(x, y), a);
    }
}

// A subject, a small island, a larger object; haze 0.08 everywhere. The island goes to 0 with
// its fringe; subject, object and the haze away from the island are bit-identical.
void testIslands() {
  Img in(200, 100, 0.08f);
  disc(in, 50, 50, 30, 3, 1.0f);   // subject, 2827 px = 14 % of the frame
  disc(in, 120, 30, 3, 2, 0.9f);   // island, ~40 px
  disc(in, 160, 70, 12, 2, 1.0f);  // object, ~450 px = 2.3 %
  CleanBackgroundStats st;
  const Img out = run(in, islandsOnly(1.0), &st);
  CHECK(st.islandsRemoved == 1 && st.componentsKept == 2);
  int islandLeft = 0, otherChanged = 0;
  for (int y = 0; y < 100; ++y)
    for (int x = 0; x < 200; ++x) {
      const double di = std::hypot(x - 120.0, y - 30.0);
      if (di <= 5.0) islandLeft += out.at(x, y) != 0.0f;  // island + its ramp (r + soft)
      else if (di > 8.0) otherChanged += out.at(x, y) != in.at(x, y);
    }
  CHECK(islandLeft == 0);
  CHECK(otherChanged == 0);
  // The object goes too once the limit is above its size; 0 = off.
  CHECK(run(in, islandsOnly(4.0)).at(160, 70) == 0.0f);
  CHECK(run(in, islandsOnly(0.0)).v == in.v);
}

// Haze (0.1) joining an island to the subject does not glue them: components are taken above
// Background Below. The subject's own falloff, right next to the island's valley, stays.
void testHazeDoesNotGlue() {
  Img in(120, 60);
  disc(in, 30, 30, 20, 4, 1.0f);
  for (int x = 50; x < 100; ++x)
    for (int y = 25; y < 35; ++y) in.at(x, y) = std::max(in.at(x, y), 0.1f);  // a haze bridge
  disc(in, 70, 30, 2, 2, 0.8f);  // the island sits on the bridge, 16 px from the subject
  disc(in, 57, 30, 1, 2, 0.6f);  // a second one 3 px off the subject's falloff
  const Img out = run(in, islandsOnly(1.0));
  CHECK(out.at(70, 30) == 0.0f && out.at(57, 30) == 0.0f);
  int subjectChanged = 0;
  for (int y = 0; y < 60; ++y)
    for (int x = 0; x < 120; ++x)
      if (std::hypot(x - 30.0, y - 30.0) <= 24.0)  // the subject with its 4 px falloff
        subjectChanged += out.at(x, y) != in.at(x, y);
  CHECK(subjectChanged == 0);
  // Below the haze level the bridge would glue them: then nothing is an island.
  CleanBackgroundParams p = islandsOnly(1.0);
  p.backgroundBelow = 0.05f;
  CHECK(run(in, p).v == in.v);
}

// The fringe never climbs: an island in the faint haze (0.02) just off a subject's long soft
// falloff (20 px ramp, below 0.15 for its outer 3 px) clears its own fringe but not the rising
// falloff beyond the valley, even with Fringe reaching that far.
void testFringeNeverRises() {
  Img in(120, 60, 0.02f);
  disc(in, 40, 30, 20, 20, 1.0f);  // > 0 out to x = 80 on row 30
  disc(in, 83, 30, 1, 1, 0.9f);    // the island, 3 px off the falloff
  CleanBackgroundParams p = islandsOnly(1.0);
  p.fringePx = 3.0;
  const Img out = run(in, p);
  int falloffChanged = 0;
  for (int y = 0; y < 60; ++y)
    for (int x = 0; x < 120; ++x)
      if (std::hypot(x - 40.0, y - 30.0) <= 39.5) falloffChanged += out.at(x, y) != in.at(x, y);
  CHECK(out.at(83, 30) == 0.0f && out.at(81, 30) == 0.0f);
  CHECK(falloffChanged == 0);
}

// 8-connected: a diagonal 1 px strand (like hair) stays part of the subject.
void testDiagonalStrand() {
  Img in(80, 80);
  disc(in, 20, 20, 12, 1, 1.0f);
  for (int k = 28; k < 75; ++k) in.at(k, k) = std::max(in.at(k, k), 0.4f);
  CleanBackgroundStats st;
  const Img out = run(in, islandsOnly(5.0), &st);
  CHECK(st.islandsRemoved == 0);
  CHECK(out.v == in.v);
}

// Max Island % at the limit: a component of exactly the limit goes, one pixel more stays.
void testLimit() {
  Img in(100, 100);  // 1 % = 100 px
  for (int y = 10; y < 20; ++y)
    for (int x = 10; x < 20; ++x) in.at(x, y) = 1.0f;  // 100 px
  for (int y = 50; y < 60; ++y)
    for (int x = 50; x < 60; ++x) in.at(x, y) = 1.0f;
  in.at(60, 50) = 1.0f;  // 101 px
  const Img out = run(in, islandsOnly(1.0));
  CHECK(out.at(15, 15) == 0.0f && out.at(55, 55) == 1.0f);
}

// Islands against brute force: components by repeated label propagation, then the fringe
// rule (never rising, fringePx steps, not next to a kept component) written out plainly.
void testIslandsAgainstBruteForce() {
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  int bad = 0, removedTotal = 0, keptTotal = 0;
  for (int trial = 0; trial < 40; ++trial) {
    const int w = 30 + trial % 7, h = 24 + trial % 5;
    Img in(w, h);
    for (auto& v : in.v) v = u(rng) < 0.25f ? 0.15f + 0.85f * u(rng) : 0.2f * u(rng);
    CleanBackgroundParams p = islandsOnly(0.5 + (trial % 5) * 1.5);
    p.fringePx = trial % 3;
    const Img out = run(in, p);
    const int n = w * h;
    std::vector<int> label(n);
    for (int i = 0; i < n; ++i) label[i] = in.v[i] >= p.backgroundBelow ? i : -1;
    for (bool changed = true; changed;) {
      changed = false;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          if (label[i] >= 0 && label[j] >= 0 && std::abs(i % w - j % w) <= 1 &&
              std::abs(i / w - j / w) <= 1 && label[j] < label[i])
            label[i] = label[j], changed = true;
    }
    std::vector<int> area(n, 0);
    for (int i = 0; i < n; ++i)
      if (label[i] >= 0) ++area[label[i]];
    std::vector<int> state(n, 0);  // 1 removed, 2 kept
    for (int i = 0; i < n; ++i)
      if (label[i] >= 0) state[i] = area[label[i]] <= p.maxIslandPercent / 100.0 * n ? 1 : 2;
    std::vector<int> reach(n, -1);  // fringe step at which a pixel joined; 0 = the island
    for (int i = 0; i < n; ++i)
      if (state[i] == 1) reach[i] = 0;
    for (int s = 1; s <= p.fringePx; ++s)
      for (int j = 0; j < n; ++j) {
        if (state[j] != 0 || reach[j] >= 0 || in.v[j] <= 0.0f) continue;
        bool from = false, nearKept = false;
        for (int i = 0; i < n; ++i) {
          if (std::abs(i % w - j % w) > 1 || std::abs(i / w - j / w) > 1) continue;
          nearKept |= state[i] == 2;
          from |= reach[i] == s - 1 && in.v[j] <= in.v[i];
        }
        if (from && !nearKept) reach[j] = s;
      }
    for (int i = 0; i < n; ++i) {
      const float want = reach[i] >= 0 ? 0.0f : in.v[i];
      bad += out.v[i] != want;
      removedTotal += reach[i] >= 0;
      keptTotal += state[i] == 2;
    }
  }
  std::printf("islands vs brute force: %d px removed, %d kept, %d wrong\n", removedTotal,
              keptTotal, bad);
  CHECK(bad == 0 && removedTotal > 100 && keptTotal > 100);
}

// Crush = 1 - Solid Core(1 - matte) with the level mirrored, hard and soft, both distance paths.
void testCrushMirrorsSolidCore(double feather) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  double worst = 0.0;
  for (int trial = 0; trial < 30; ++trial) {
    Img in(31 + trial % 4, 27 + trial % 5);
    const float pAbove = trial < 15 ? 0.02f : 0.3f;  // sparse: disc stamps; dense: transform
    for (auto& v : in.v) v = u(rng) < pAbove ? 0.5f + 0.5f * u(rng) : 0.5f * u(rng);
    CleanBackgroundParams p;
    p.maxIslandPercent = 0.0;
    p.backgroundBelow = 0.5f;
    p.crushPx = 0.5 + (trial % 6) * 1.3;
    p.featherPx = feather;
    const Img out = run(in, p);
    Img inv(in.w, in.h), core(in.w, in.h);
    for (size_t i = 0; i < in.v.size(); ++i) inv.v[i] = 1.0f - in.v[i];
    mr::SolidCoreParams sp;
    sp.depthPx = p.crushPx;
    sp.featherPx = feather;
    sp.edgeAt = 1.0f - p.backgroundBelow;
    mr::solidCore(inv.v.data(), core.v.data(), in.w, in.h, sp);
    for (size_t i = 0; i < in.v.size(); ++i)
      worst = std::max(worst, double(std::fabs(out.v[i] - (1.0f - core.v[i]))));
  }
  std::printf("crush vs 1 - solidCore(1 - m), feather %.1f: max diff %.2g\n", feather, worst);
  CHECK(worst < 1e-6);
}

// The same on a full HD frame (the synthetic "all"), where the distance is stamped in bands on
// threads: the same as Solid Core's one call over the frame, up to the rounding of 1 - x
// (the contour crossings come out a few ulp apart).
void testCrushThreadedMirrorsSolidCore() {
  const int w = 1920, h = 1080;
  const synth::Scene s = synth::make("all", w, h);
  CleanBackgroundParams p;
  p.maxIslandPercent = 0.0;
  std::vector<float> out(s.matte.size()), inv(s.matte.size()), core(s.matte.size());
  mr::cleanBackground(s.matte.data(), out.data(), w, h, p);
  for (size_t i = 0; i < inv.size(); ++i) inv[i] = 1.0f - s.matte[i];
  mr::SolidCoreParams sp;
  sp.depthPx = p.crushPx;
  sp.featherPx = p.featherPx;
  sp.edgeAt = 1.0f - p.backgroundBelow;
  mr::solidCore(inv.data(), core.data(), w, h, sp);
  double worst = 0.0;
  for (size_t i = 0; i < out.size(); ++i)
    worst = std::max(worst, double(std::fabs(out[i] - (1.0f - core[i]))));
  std::printf("crush on threads vs 1 - solidCore(1 - m), HD \"all\": max diff %.2g\n", worst);
  CHECK(worst < 1e-5);
}

// Crush known answer: a soft disc in haze. The falloff within Crush px of the contour is
// bit-identical, haze beyond Crush + Feather is 0, nothing is brighter.
void testCrush() {
  Img in(160, 120);
  for (int y = 0; y < 120; ++y)
    for (int x = 0; x < 160; ++x) in.at(x, y) = 0.05f + 0.05f * float((x * 7 + y * 3) % 5) / 4.0f;
  disc(in, 80, 60, 30, 10, 1.0f);  // 10 px falloff: the 0.15 contour at r 38.5
  CleanBackgroundParams p;
  p.maxIslandPercent = 0.0;
  CleanBackgroundStats st;
  const Img out = run(in, p, &st);
  int nearChanged = 0, farLeft = 0, brighter = 0;
  for (int y = 0; y < 120; ++y)
    for (int x = 0; x < 160; ++x) {
      const double d = std::hypot(x - 80.0, y - 60.0) - 38.5;  // outside the contour, px
      brighter += out.at(x, y) > in.at(x, y);
      if (d <= p.crushPx - 0.1) nearChanged += out.at(x, y) != in.at(x, y);
      if (d > p.crushPx + p.featherPx + 0.3) farLeft += out.at(x, y) != 0.0f;
    }
  CHECK(st.pixelsCrushed > 10000);
  CHECK(brighter == 0 && nearChanged == 0 && farLeft == 0);
  p.crushPx = 0.0;  // off
  CHECK(run(in, p).v == in.v);
}

// In place, and both parts together never brighten and leave the subject alone (noise).
void testInPlaceAndInvariants() {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  Img in(120, 90);
  for (auto& v : in.v) v = u(rng) < 0.03f ? u(rng) : 0.12f * u(rng);
  disc(in, 60, 45, 20, 3, 1.0f);
  CleanBackgroundParams p;
  p.maxIslandPercent = 0.5;
  const Img out = run(in, p);
  Img same = in;
  mr::cleanBackground(same.v.data(), same.v.data(), in.w, in.h, p);
  CHECK(same.v == out.v);
  int brighter = 0, subject = 0;
  for (int y = 0; y < 90; ++y)
    for (int x = 0; x < 120; ++x) {
      brighter += out.at(x, y) > in.at(x, y);
      if (std::hypot(x - 60.0, y - 45.0) <= 22.0) subject += out.at(x, y) != in.at(x, y);
    }
  CHECK(brighter == 0 && subject == 0);
}

// The synthetic scenes against their truth, at the defaults (and Max Island % 0.2, which also
// removes the two large blobs).
void testSynth(const char* name, double maxIsland) {
  const int w = 1920, h = 1080;
  const synth::Scene s = synth::make(name, w, h);
  CleanBackgroundParams p;
  p.maxIslandPercent = maxIsland;
  CleanBackgroundStats st;
  std::vector<float> out(s.matte.size());
  mr::cleanBackground(s.matte.data(), out.data(), w, h, p, &st);
  // The two large blobs (synth.cpp: (-700, -380) r 25 and (620, 300) r 30, 1080-line units).
  auto inBlob = [&](int i) {
    const double x = i % w - w / 2.0, y = i / w - h / 2.0;
    return std::hypot(x + 700, y + 380) < 28 || std::hypot(x - 620, y - 300) < 33;
  };
  // Speck pixels = no truth, above the haze (<= 0.1) and not glued to the figure: some specks
  // touch a hair strand above Background Below, so they are part of the subject by definition,
  // and so is the soft falloff within 3 px of it (and of the truth: chatter).
  const size_t n = out.size();
  mr::Mask glued(n, 0);
  std::vector<int> queue;
  for (size_t i = 0; i < n; ++i)
    if (s.truth[i] >= p.backgroundBelow && s.matte[i] >= p.backgroundBelow) {
      glued[i] = 1;
      queue.push_back(int(i));
    }
  for (size_t head = 0; head < queue.size(); ++head) {
    const int i = queue[head], x = i % w, y = i / w;
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx, ny = y + dy, j = ny * w + nx;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h || glued[j] || s.matte[j] < p.backgroundBelow)
          continue;
        glued[j] = 1;
        queue.push_back(j);
      }
  }
  for (size_t i = 0; i < n; ++i) glued[i] = glued[i] || s.truth[i] > 0.0f;
  glued = mr::dilate(glued, w, h, 3.0);
  // Subject = truth > 0. Inside its holes (truth 1, input below Background Below) Crush acts by
  // design (the matte is background there; Fill Holes, next in the chain, fills them): counted
  // apart. Everywhere else in the subject nothing may change.
  long specks = 0, specksGone = 0, gluedPx = 0, blob = 0, blobGone = 0, subjectChanged = 0;
  long holeChanged = 0, brighter = 0;
  for (size_t i = 0; i < n; ++i) {
    brighter += out[i] > s.matte[i];
    if (s.truth[i] > 0.0f) {
      const bool hole = s.truth[i] == 1.0f && s.matte[i] < p.backgroundBelow;
      (hole ? holeChanged : subjectChanged) += out[i] != s.matte[i];
    }
    if (s.truth[i] == 0.0f && s.matte[i] > 0.1f) {
      if (inBlob(int(i))) {
        ++blob;
        blobGone += out[i] == 0.0f;
      } else if (glued[i]) {
        ++gluedPx;
      } else {
        ++specks;
        specksGone += out[i] == 0.0f;
      }
    }
  }
  // Haze beyond Crush + Feather px of the result's matte (an independent dilation): must be 0.
  mr::Mask m(out.size());
  for (size_t i = 0; i < out.size(); ++i) m[i] = out[i] >= p.backgroundBelow;
  const mr::Mask near = mr::dilate(m, w, h, p.crushPx + p.featherPx + 1.0);
  // Untouched elsewhere: a pixel within Crush px of the result's matte changes only by going to
  // 0 with a removed island (crushing needs a grid distance > Crush px).
  const mr::Mask falloff = mr::dilate(m, w, h, p.crushPx);
  long otherChanged = 0;
  for (size_t i = 0; i < out.size(); ++i) otherChanged += falloff[i] && out[i] != s.matte[i] && out[i] != 0.0f;
  long hazeFar = 0, hazeBefore = 0;
  double hazeSumIn = 0, hazeSumOut = 0;
  for (size_t i = 0; i < out.size(); ++i)
    if (s.truth[i] == 0.0f && s.matte[i] <= 0.1f) {
      hazeBefore += s.matte[i] > 0.0f;
      hazeFar += !near[i] && out[i] > 0.0f;
      hazeSumIn += s.matte[i];
      hazeSumOut += out[i];
    }
  std::printf("%-6s max island %.2f%%: islands %d removed, %d kept; specks %.1f%% removed (%ld of "
              "%ld px; %ld px near the figure, glued to hair, kept), blobs %.1f%% removed; haze px > 0 beyond "
              "crush %ld (of %ld), haze sum %.0f -> %.0f; subject px changed %ld (hole interiors "
              "%ld); brighter %ld\n",
              name, maxIsland, st.islandsRemoved, st.componentsKept, 100.0 * specksGone / specks,
              specksGone, specks, gluedPx, 100.0 * blobGone / std::max(blob, 1L), hazeFar,
              hazeBefore, hazeSumIn, hazeSumOut, subjectChanged, holeChanged, brighter);
  CHECK(subjectChanged == 0 && brighter == 0 && hazeFar == 0 && otherChanged == 0);
  CHECK(specksGone == specks);
  CHECK(maxIsland < 0.1 ? blobGone == 0 : blobGone == blob);
}

void testTiming() {
  const int w = 3840, h = 2160;
  const synth::Scene s = synth::make("all", w, h);
  CleanBackgroundParams p;
  std::vector<float> out(s.matte.size());
  CleanBackgroundStats st;
  double best = 1e9;
  for (int k = 0; k < 3; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    mr::cleanBackground(s.matte.data(), out.data(), w, h, p, &st);
    best = std::min(best, std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count());
  }
  std::printf("UHD clean + crush (\"all\", defaults): %.1f ms (best of 3), %d islands, %lld px "
              "removed, %lld crushed\n", best, st.islandsRemoved, st.pixelsRemoved, st.pixelsCrushed);
  CHECK(st.islandsRemoved > 0 && st.pixelsCrushed > 0);
  CHECK(best < 50.0);
}

}  // namespace

int main() {
  testIslands();
  testHazeDoesNotGlue();
  testFringeNeverRises();
  testDiagonalStrand();
  testLimit();
  testIslandsAgainstBruteForce();
  testCrushMirrorsSolidCore(0.0);
  testCrushMirrorsSolidCore(2.5);
  testCrushThreadedMirrorsSolidCore();
  testCrush();
  testInPlaceAndInvariants();
  testSynth("specks", CleanBackgroundParams().maxIslandPercent);
  testSynth("specks", 0.2);
  testSynth("all", CleanBackgroundParams().maxIslandPercent);
  testTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
