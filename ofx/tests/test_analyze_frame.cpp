// Analyze Frame (CLAUDE.md roadmap): the suggestions on the synthetic scenes against their
// truth (applied with the real ops, chain order), clean scenes left alone, degenerate frames,
// UHD timing, and the two saved Flame-test mattes when MR_DUMP_DIR points at them (skipped
// otherwise: the footage is confidential and never in the repo).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "ops/analyze_frame.h"
#include "ops/morphology.h"
#include "synth.h"

using mr::AnalysisResult;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

AnalysisResult analyze(const std::vector<float>& m, int w, int h) {
  AnalysisResult r;
  mr::analyzeFrame(m.data(), w, h, &r);
  return r;
}

// The suggested chain, as the plugin would run it: clean -> fill -> core.
std::vector<float> apply(const AnalysisResult& r, const std::vector<float>& m, int w, int h) {
  std::vector<float> o(m.size());
  for (size_t i = 0; i < m.size(); ++i) o[i] = std::min(1.0f, std::max(0.0f, m[i]));
  if (r.cleanEnable) mr::cleanBackground(o.data(), o.data(), w, h, r.clean);
  if (r.fillEnable) mr::fillHoles(o.data(), o.data(), w, h, r.fill);
  if (r.coreEnable) mr::solidCore(o.data(), o.data(), w, h, r.core);
  return o;
}

bool anyChanged(const AnalysisResult::Changed& c) {
  return c.cleanEnable || c.maxIsland || c.backgroundBelow || c.crushPx || c.fillEnable ||
         c.threshold || c.maxHole || c.bridge || c.closedLeft || c.closedRight || c.closedTop ||
         c.closedBottom || c.coreEnable || c.depth;
}

void printResult(const char* name, const AnalysisResult& r) {
  const auto& M = r.measured;
  std::printf("%s: %.1f ms | clean %d (island %.3f %%, below %.2f, crush %.0f) | fill %d (above %.2f, "
              "max hole %.3f %%, bridge %.0f, closed L%d R%d T%d B%d) | core %d (depth %.0f, "
              "feather %.0f, at %.2f)\n",
              name, M.ms, r.cleanEnable, r.clean.maxIslandPercent, r.clean.backgroundBelow,
              r.clean.crushPx, r.fillEnable, r.fill.threshold, r.fill.maxHolePercent, r.fill.bridgePx,
              r.fill.closedLeft, r.fill.closedRight, r.fill.closedTop, r.fill.closedBottom,
              r.coreEnable, r.core.depthPx, r.core.featherPx, r.core.edgeAt);
  std::printf("  holes %d (%lld px, largest %.4f %%), gaps %d (smallest %.3f %%), islands %d, "
              "objects %d, haze %.3f, deep grey %lld -> %lld\n",
              M.holes, M.holePixels, M.largestHolePercent, M.realGaps, M.smallestGapPercent,
              M.islands, M.objects, M.hazeLevel, M.deepGreyBefore, M.deepGreyAfter);
  for (const auto& n : r.notes) std::printf("  | %s\n", n.c_str());
}

// The input's background: below 0.5 and joined (4-connected) to a frame edge that is not
// Closed (a Closed edge counts as solid, so a grey pocket the subject's cut leaves at it is
// inside, decision 8).
mr::Mask outsideOf(const std::vector<float>& m, int w, int h, const mr::FillHolesParams& fp) {
  mr::Mask out(m.size(), 0);
  std::vector<int> stack;
  auto push = [&](int x, int y) {
    const size_t i = size_t(y) * w + x;
    if (!out[i] && m[i] < 0.5f) {
      out[i] = 1;
      stack.push_back(int(i));
    }
  };
  for (int x = 0; x < w; ++x) {
    if (!fp.closedTop) push(x, 0);
    if (!fp.closedBottom) push(x, h - 1);
  }
  for (int y = 0; y < h; ++y) {
    if (!fp.closedLeft) push(0, y);
    if (!fp.closedRight) push(w - 1, y);
  }
  while (!stack.empty()) {
    const int i = stack.back(), x = i % w, y = i / w;
    stack.pop_back();
    if (x > 0) push(x - 1, y);
    if (x + 1 < w) push(x + 1, y);
    if (y > 0) push(x, y - 1);
    if (y + 1 < h) push(x, y + 1);
  }
  return out;
}

struct Outcome {
  long long changedOutside = 0;  // changed pixels in the input's background
  long long changedNearEdge = 0; // changed pixels within 3 px of that background (the edge band)
  long long darker = 0;          // pixels lowered (allowed only by Clean & Crush)
  long long deepGreyIn = 0, deepGreyOut = 0;  // < 0.9 more than 8 px inside, before / after
};

Outcome measure(const std::vector<float>& in, const std::vector<float>& out, int w, int h,
                const mr::FillHolesParams& fp) {
  Outcome o;
  const mr::Mask outside = outsideOf(in, w, h, fp);
  const std::vector<float> d =
      mr::distanceWithin(outside, w, h, 9.0, [](int) { return mr::ContourFoot(); });
  for (size_t i = 0; i < in.size(); ++i) {
    const bool changed = out[i] != std::min(1.0f, std::max(0.0f, in[i]));
    o.changedOutside += changed && outside[i];
    o.changedNearEdge += changed && !outside[i] && d[i] <= 3.0f;
    o.darker += out[i] < in[i] && in[i] <= 1.0f;
    o.deepGreyIn += d[i] > 8.0f && in[i] < 0.9f;
    o.deepGreyOut += d[i] > 8.0f && out[i] < 0.9f;
  }
  std::printf("  outcome: changed outside %lld, in the 3 px edge band %lld, darker %lld, grey < 0.9 "
              "more than 8 px inside %lld -> %lld\n",
              o.changedOutside, o.changedNearEdge, o.darker, o.deepGreyIn, o.deepGreyOut);
  return o;
}

// Against the scene's truth: true-solid pixels left grey, true-background pixels made solid.
struct TruthErr {
  long long solidLeftGrey = 0, solidDefects = 0, backgroundMadeSolid = 0, backgroundLeft = 0;
};

// Defects count only beyond `band` px inside the true edge (Solid Core leaves its outer Depth px
// as they are on purpose, and the synthetic grey face runs right up to the edge).
TruthErr vsTruth(const synth::Scene& s, const std::vector<float>& out, double band) {
  TruthErr e;
  mr::Mask below(s.truth.size());
  for (size_t i = 0; i < below.size(); ++i) below[i] = s.truth[i] < 0.5f;
  const std::vector<float> d =
      mr::distanceWithin(below, s.w, s.h, band + 1.0, [](int) { return mr::ContourFoot(); });
  for (size_t i = 0; i < out.size(); ++i) {
    if (s.truth[i] >= 0.999f && s.matte[i] < 0.9f && d[i] > band) {
      ++e.solidDefects;
      e.solidLeftGrey += out[i] < 0.9f;
    }
    if (s.truth[i] <= 0.001f) {
      e.backgroundMadeSolid += out[i] >= 0.5f;
      e.backgroundLeft += out[i] > 0.02f;
    }
  }
  std::printf("  truth: defects in the solid %lld, still < 0.9 %lld; true background >= 0.5 %lld, "
              "> 0.02 %lld\n",
              e.solidDefects, e.solidLeftGrey, e.backgroundMadeSolid, e.backgroundLeft);
  return e;
}

void testHoles() {
  const synth::Scene s = synth::make("holes", 1920, 1080);
  const AnalysisResult r = analyze(s.matte, s.w, s.h);
  printResult("holes", r);
  CHECK(!r.noisy && !r.constantMatte && r.fillEnable);
  // Between the grey holes (centres 0.05-0.8) and the grey face (0.85) they sit in, so the holes
  // inside the face are enclosed; the default 0.95 would let those leak through the face.
  CHECK(r.fill.threshold > 0.6f && r.fill.threshold <= 0.85f && r.changed.threshold);
  CHECK(r.measured.holes >= 15 && r.measured.realGaps >= 1);
  // The arm-on-hip gap (~1 % of the frame here) stays open: Max Hole % below it.
  CHECK(r.fill.maxHolePercent < r.measured.smallestGapPercent / 1.5);
  CHECK(r.fill.maxHolePercent > r.measured.largestHolePercent * 1.5);
  CHECK(r.fill.closedBottom && !r.fill.closedTop && !r.fill.closedLeft && !r.fill.closedRight);
  CHECK(r.fill.bridgePx == 0.0);
  CHECK(r.coreEnable);  // the grey face is joined to the edge: Solid Core's job (decision 15)
  CHECK(r.core.depthPx >= 3 && r.core.depthPx <= 8);
  CHECK(!r.cleanEnable);
  const std::vector<float> out = apply(r, s.matte, s.w, s.h);
  const Outcome o = measure(s.matte, out, s.w, s.h, r.fill);
  const TruthErr e = vsTruth(s, out, r.core.depthPx + r.core.featherPx);
  CHECK(o.changedOutside == 0 && o.darker == 0);
  CHECK(e.solidLeftGrey <= e.solidDefects / 200);  // >= 99.5 % of the defects fixed
  CHECK(e.backgroundMadeSolid == 0);               // the real gaps stay open
}

// The legs are cut by the bottom of the frame; a grey hole in a leg is open to that edge, so
// only Bottom Closed fills it, while the gap between the legs (also enclosed then) stays open.
void testHoleAtBottomEdge() {
  synth::Scene s = synth::make("holes", 1920, 1080);
  const int cx = 870, r0 = 14;  // the left leg (design x -90) at the bottom row
  for (int y = s.h - r0; y < s.h; ++y)
    for (int x = cx - r0; x <= cx + r0; ++x)
      if ((x - cx) * (x - cx) + (y - (s.h - 1)) * (y - (s.h - 1)) <= r0 * r0)
        s.matte[size_t(y) * s.w + x] = std::min(s.matte[size_t(y) * s.w + x], 0.3f);
  const AnalysisResult r = analyze(s.matte, s.w, s.h);
  printResult("holes + a hole at the bottom edge", r);
  CHECK(r.fill.closedBottom);
  const std::vector<float> out = apply(r, s.matte, s.w, s.h);
  long long left = 0;
  for (int y = s.h - r0 + 2; y < s.h; ++y)
    for (int x = cx - 4; x <= cx + 4; ++x) left += out[size_t(y) * s.w + x] < 0.99f;
  std::printf("  bottom-edge hole: %lld of its centre pixels still < 0.99\n", left);
  CHECK(left == 0);
  const TruthErr e = vsTruth(s, out, r.core.depthPx + r.core.featherPx);
  CHECK(e.backgroundMadeSolid == 0);
}

// A grey hole in the torso joined to the outside by a 2 px channel: it fills only with Bridge
// Gaps (decision 9), so the analysis suggests the smaller bridge that does it.
void testLeakingHole() {
  synth::Scene s = synth::make("holes", 1920, 1080);
  const int cx = 870, cy = 345, r0 = 10;  // design (-90, -195), below the shoulder line (y 310)
  for (int y = cy - r0; y <= cy + r0; ++y)
    for (int x = cx - r0; x <= cx + r0; ++x)
      if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r0 * r0) s.matte[size_t(y) * s.w + x] = 0.2f;
  for (int y = 295; y < cy; ++y)
    for (int x = cx - 1; x <= cx; ++x) s.matte[size_t(y) * s.w + x] = std::min(s.matte[size_t(y) * s.w + x], 0.3f);
  const AnalysisResult r = analyze(s.matte, s.w, s.h);
  printResult("holes + a hole leaking through a 2 px gap", r);
  CHECK(r.changed.bridge && r.fill.bridgePx >= 2.0 && r.fill.bridgePx <= 8.0);
  const std::vector<float> out = apply(r, s.matte, s.w, s.h);
  long long left = 0;
  for (int y = cy - 5; y <= cy + 5; ++y)
    for (int x = cx - 5; x <= cx + 5; ++x) left += out[size_t(y) * s.w + x] < 0.99f;
  std::printf("  leaking hole: %lld of its centre pixels still < 0.99\n", left);
  CHECK(left == 0);
  const TruthErr e = vsTruth(s, out, r.core.depthPx + r.core.featherPx);
  CHECK(e.backgroundMadeSolid == 0);
}

void testSpecks() {
  const synth::Scene s = synth::make("specks", 1920, 1080);
  const AnalysisResult r = analyze(s.matte, s.w, s.h);
  printResult("specks", r);
  CHECK(r.cleanEnable && !r.noisy);
  // Rule: islands below the largest size jump go; the two large blobs (0.1 % and 0.14 % of
  // the frame, a 5x jump above the specks) count as separate objects and stay (the note says
  // how to remove them).
  CHECK(r.measured.islands >= 25 && r.measured.objects == 2);
  CHECK(r.clean.maxIslandPercent > 0.02 && r.clean.maxIslandPercent < 0.09);
  CHECK(r.measured.hazeLevel > 0.03);  // the haze is seen, and crushed
  CHECK(r.clean.crushPx >= 4.0);
  CHECK(!r.changed.threshold && !r.changed.maxHole && !r.coreEnable);
  const std::vector<float> out = apply(r, s.matte, s.w, s.h);
  const TruthErr e = vsTruth(s, out, 8.0);
  // What is left in the true background: the two kept blobs (~2000 + 2800 px plus fringe) and
  // the haze inside Crush Beyond px of the edge.
  long long blobs = 0;
  for (size_t i = 0; i < out.size(); ++i) blobs += s.truth[i] <= 0.001f && out[i] >= 0.5f;
  CHECK(blobs > 4000 && blobs < 6000);
  CHECK(e.backgroundLeft < 40000);  // from ~1.15 M hazy pixels
}

void testAll() {
  const synth::Scene s = synth::make("all", 1920, 1080);
  const AnalysisResult r = analyze(s.matte, s.w, s.h);
  printResult("all", r);
  CHECK(r.cleanEnable && r.fillEnable && r.coreEnable && r.fill.closedBottom);
  CHECK(r.fill.threshold > 0.6f && r.fill.threshold <= 0.85f);
  CHECK(r.measured.objects == 2);
  const std::vector<float> out = apply(r, s.matte, s.w, s.h);
  const TruthErr e = vsTruth(s, out, r.core.depthPx + r.core.featherPx);
  CHECK(e.solidLeftGrey <= e.solidDefects / 100);
  CHECK(e.backgroundMadeSolid < 6000);  // the two kept blobs only
}

// Clean mattes are left alone: no control changes and the default chain changes nothing.
void testCleanScenes() {
  for (const char* name : {"fgcolor", "eroded", "chatter", "mixed_edge"}) {
    const synth::Scene s = synth::make(name, 1920, 1080);
    const AnalysisResult r = analyze(s.matte, s.w, s.h);
    printResult(name, r);
    CHECK(!r.noisy && !r.cleanEnable && !r.coreEnable && r.fillEnable);
    CHECK(r.measured.holes == 0 && r.measured.islands == 0);
    AnalysisResult::Changed c = r.changed;
    if (std::string(name) == "mixed_edge") {
      // The one exception: its hard-cut arm makes the real gap 0.99 % of the frame, under the
      // default 1 %, so the defaults would fill it. The analysis protects it.
      CHECK(c.maxHole && r.fill.maxHolePercent < 0.9);
      c.maxHole = false;
    }
    CHECK(!anyChanged(c));
    const std::vector<float> out = apply(r, s.matte, s.w, s.h);
    long long changed = 0;
    for (size_t i = 0; i < out.size(); ++i) changed += out[i] != s.matte[i];
    std::printf("  pixels changed by the suggested chain: %lld\n", changed);
    CHECK(changed == 0);
  }
}

void testDegenerate() {
  struct Case {
    const char* name;
    int w, h;
    std::vector<float> m;
  };
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<float> noise(1920 * 1080), salt(1920 * 1080);
  for (auto& v : noise) v = u(rng);
  for (auto& v : salt) v = u(rng) < 0.5f ? 0.0f : 1.0f;
  std::vector<Case> cases = {
      {"constant 0.5", 640, 360, std::vector<float>(640 * 360, 0.5f)},
      {"all zero", 640, 360, std::vector<float>(640 * 360, 0.0f)},
      {"all one", 640, 360, std::vector<float>(640 * 360, 1.0f)},
      {"noise", 1920, 1080, noise},
      {"salt and pepper", 1920, 1080, salt},
      {"1x1", 1, 1, {0.7f}},
      {"3x3", 3, 3, {0, 1, 0, 1, 0.5f, 1, 0, 1, 0}},
  };
  for (const Case& c : cases) {
    const AnalysisResult r = analyze(c.m, c.w, c.h);
    std::printf("%s: constant %d, no subject %d, noisy %d, too small %d, %.1f ms\n", c.name,
                r.constantMatte, r.noSubject, r.noisy, r.tooSmall, r.measured.ms);
    for (const auto& n : r.notes) std::printf("  | %s\n", n.c_str());
    CHECK(r.constantMatte || r.noSubject || r.noisy || r.tooSmall);
    CHECK(!anyChanged(r.changed) && !r.cleanEnable && !r.coreEnable && r.fillEnable);
    CHECK(r.notes.size() == 1);
  }
  // A dot of subject on black: too little to read (no subject).
  std::vector<float> dot(640 * 360, 0.0f);
  dot[180 * 640 + 320] = 1.0f;
  const AnalysisResult r = analyze(dot, 640, 360);
  CHECK(r.noSubject && !anyChanged(r.changed));
}

void testTiming() {
  const synth::Scene s = synth::make("all", 3840, 2160);
  AnalysisResult r;
  double best = 1e9;
  for (int k = 0; k < 3; ++k) {
    mr::analyzeFrame(s.matte.data(), s.w, s.h, &r);
    best = std::min(best, r.measured.ms);
  }
  std::printf("UHD \"all\" (clean, fill and core all on): best of 3 %.1f ms\n", best);
  CHECK(best < 250.0);  // ~165 ms on BFD; a loose bound against regressions
}

std::vector<float> loadDump(const std::string& path, int w, int h) {
  std::vector<float> m(size_t(w) * h);
  std::ifstream f(path, std::ios::binary);
  if (!f.read(reinterpret_cast<char*>(m.data()), std::streamsize(m.size() * 4))) m.clear();
  return m;
}

// The saved Flame-test mattes (CLAUDE.md decisions 8, 15). Confidential: read from MR_DUMP_DIR.
void testDumps() {
  const char* dir = std::getenv("MR_DUMP_DIR");
  if (!dir) {
    std::printf("dumps: skipped (MR_DUMP_DIR not set)\n");
    return;
  }
  const int w = 2880, h = 1620;
  {
    const std::vector<float> m = loadDump(std::string(dir) + "/05a4e946_t36.f32", w, h);
    CHECK(!m.empty());
    if (m.empty()) return;
    const AnalysisResult r = analyze(m, w, h);
    printResult("t36 (product shot, grey side face)", r);
    // Settled in Flame: Solid Above 0.95, Solid Core Depth 6, Edge At 0.5, Feather 2.
    CHECK(r.fill.threshold >= 0.92f && r.fill.threshold <= 0.96f);
    CHECK(r.coreEnable && r.core.depthPx >= 4 && r.core.depthPx <= 8 && r.core.edgeAt == 0.5f);
    CHECK(!r.cleanEnable);
    const std::vector<float> out = apply(r, m, w, h);
    const Outcome o = measure(m, out, w, h, r.fill);
    CHECK(o.changedOutside == 0 && o.darker == 0);
    // The maintainer's settings, measured the same way: the analysis must do at least as well.
    AnalysisResult ref;
    ref.fill.threshold = 0.95f;
    ref.coreEnable = true;
    ref.core.depthPx = 6.0;
    ref.core.edgeAt = 0.5f;
    ref.core.featherPx = 2.0;
    std::printf("  the maintainer's settings (Solid Above 0.95, Solid Core 6 / 0.5 / 2, edges open):\n");
    const Outcome oref = measure(m, apply(ref, m, w, h), w, h, r.fill);
    CHECK(o.deepGreyOut <= oref.deepGreyOut + oref.deepGreyOut / 10);
    CHECK(r.measured.deepGreyAfter <= 30);  // its own measure (after fill): 3445 -> 16
  }
  {
    const std::vector<float> m = loadDump(std::string(dir) + "/05a4e946_t44.f32", w, h);
    CHECK(!m.empty());
    if (m.empty()) return;
    const AnalysisResult r = analyze(m, w, h);
    printResult("t44 (grey slivers)", r);
    // 0.48 left the grey slivers solid by definition; 0.95 filled them all.
    CHECK(r.fill.threshold >= 0.9f && r.fill.threshold <= 0.97f);
    CHECK(r.measured.holes >= 1 && !r.cleanEnable);
    const std::vector<float> out = apply(r, m, w, h);
    const Outcome o = measure(m, out, w, h, r.fill);
    CHECK(o.changedOutside == 0 && o.darker == 0);
    mr::FillHolesParams open;  // the same with every frame edge open, as the maintainer had it
    std::printf("  with every frame edge open:\n");
    CHECK(measure(m, out, w, h, open).changedOutside == 0);
    // The slivers: pixels below 0.9 that are not the input's background and not its edge band.
    long long sliversLeft = 0;
    const mr::Mask outside = outsideOf(m, w, h, r.fill);
    const std::vector<float> d =
        mr::distanceWithin(outside, w, h, 4.0, [](int) { return mr::ContourFoot(); });
    long long sliversIn = 0;
    for (size_t i = 0; i < m.size(); ++i)
      if (!outside[i] && d[i] > 3.0f) {
        sliversIn += m[i] < 0.9f;
        sliversLeft += out[i] < 0.9f;
      }
    std::printf("  grey < 0.9 inside (beyond the 3 px edge band): %lld -> %lld\n", sliversIn, sliversLeft);
    CHECK(sliversLeft <= sliversIn / 20);
  }
}

}  // namespace

int main() {
  testHoles();
  testHoleAtBottomEdge();
  testLeakingHole();
  testSpecks();
  testAll();
  testCleanScenes();
  testDegenerate();
  testTiming();
  testDumps();
  if (failures) {
    std::printf("%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
