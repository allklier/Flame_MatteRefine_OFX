// Plate Merge core (plate_merge.h, docs/plate_merge.md): known answers on hand-made pixels, the swap
// weight against brute force, then synthetic two-plate scenes with ground truth (static textured
// background with a highlight > 1 and negative values, two moving soft persons with shadow halos past
// their tight mattes, a haze tail in the mattes, an exposure offset and a 0.5 px drift between the plates,
// a window reflection that only plate A has) and fault injections (swapped plates, a clamp, a hard seam,
// no level match) that the checks must catch.
// Usage: test_plate_merge [smoke|scenes]   (no argument: both)
// Mutation checks made when written (2026-11-27), all caught (failures in brackets): gain inverted (16), Clean
// s = a (6), ring including matte pixels (61), haze floor removed (8), linear ramp (2), Keep A ignored (28),
// Keep A pixels in the ring (58), union = max(a, b) (2).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "plate_merge.h"

using namespace mr::plate;

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

// ---------------------------------------------------------------------------------------------
// Known answers.

// One pixel per case, grow 0 / feather 0, so each weight is the matte itself (0 or 1).
void testTruthTable() {
  const int w = 6, h = 1;
  //               neither  A only  B only  both   A+keep  B+keep
  const float ma[] = {0, 1, 0, 1, 1, 0};
  const float mb[] = {0, 0, 1, 1, 0, 1};
  const float kp[] = {0, 0, 0, 0, 1, 1};
  std::vector<float> A(w * 3), B(w * 3), out(w * 3), sw(w), res(w);
  for (int i = 0; i < w * 3; ++i) {
    A[i] = 0.1f + 0.01f * i;
    B[i] = 5.0f + 0.01f * i;
  }
  PlateParams p;
  p.growPx = 0;
  p.featherPx = 0;
  p.levelMatch = false;
  PlateViews v;
  v.swap = sw.data();
  v.residual = res.data();
  const int expectBoA[] = {0, 1, 1, 1, 0, 0};    // 1 = B
  const int expectClean[] = {0, 1, 0, 0, 0, 0};
  for (int mode = 0; mode < 2; ++mode) {
    p.mode = mode ? Mode::CleanBackground : Mode::BOverA;
    mergePlates(A.data(), ma, B.data(), mb, kp, w, h, p, out.data(), v);
    for (int i = 0; i < w; ++i) {
      const int e = mode ? expectClean[i] : expectBoA[i];
      for (int c = 0; c < 3; ++c) CHECK(out[3 * i + c] == (e ? B : A)[3 * i + c]);
      CHECK(sw[i] == float(e));
      CHECK(res[i] == ma[i] * mb[i]);
    }
  }
  // Without a Keep A matte (nullptr) the keep columns swap like the others.
  p.mode = Mode::BOverA;
  mergePlates(A.data(), ma, B.data(), mb, nullptr, w, h, p, out.data(), v);
  CHECK(sw[4] == 1.0f && sw[5] == 1.0f);
}

// Soft weights: B Over A is the screen union 1 - (1 - a)(1 - b), Clean is a (1 - b), out the linear mix.
void testSoftUnion() {
  const int w = 1, h = 1;
  const float ma[] = {0.4f}, mb[] = {0.3f}, A[] = {1, 2, 3}, B[] = {3, 2, 1};
  float out[3], sw;
  PlateParams p;
  p.growPx = 0;
  p.featherPx = 0;
  p.levelMatch = false;
  PlateViews v;
  v.swap = &sw;
  mergePlates(A, ma, B, mb, nullptr, w, h, p, out, v);
  CHECK(std::fabs(sw - (1 - 0.6f * 0.7f)) < 1e-6f);
  CHECK(std::fabs(out[0] - (1 + sw * 2)) < 1e-5f && std::fabs(out[2] - (3 - sw * 2)) < 1e-5f);
  p.mode = Mode::CleanBackground;
  mergePlates(A, ma, B, mb, nullptr, w, h, p, out, v);
  CHECK(std::fabs(sw - 0.4f * 0.7f) < 1e-6f);
}

// swapWeight against a brute-force reference: haze below the floor dropped, soft values kept, ramp
// from the 0.5 set.
void testSwapWeight() {
  const int w = 61, h = 47;
  std::vector<float> m(size_t(w) * h, 0.0f), out(m.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double d = std::hypot(x - 20.3, y - 22.7) - 6.0;  // disc radius 6, 2 px soft edge
      m[size_t(y) * w + x] = float(clamp01(0.5 - d / 2.0));
      if (x > 45) m[size_t(y) * w + x] = 0.015f;  // haze, below the floor
    }
  m[size_t(3) * w + 3] = std::nanf("");
  m[size_t(40) * w + 50] = 0.3f;  // a soft speck: kept as its own value, no ramp (not >= 0.5)
  const double grow = 3.0, feather = 4.0;
  swapWeight(m.data(), w, h, grow, feather, out.data());
  int bad = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double best = 1e30;
      for (int yy = 0; yy < h; ++yy)
        for (int xx = 0; xx < w; ++xx)
          if (m[size_t(yy) * w + xx] >= 0.5f) best = std::min(best, std::hypot(double(x - xx), double(y - yy)));
      double ramp = best <= grow ? 1.0 : 0.0;
      if (best > grow && best < grow + feather) {
        const double s = (grow + feather - best) / feather;
        ramp = s * s * (3 - 2 * s);
      }
      float mv = m[size_t(y) * w + x];
      mv = std::isnan(mv) ? 0.0f : float(clamp01(mv));
      if (mv < kHazeFloor) mv = 0.0f;
      const double ref = std::max(double(mv), ramp);
      if (std::fabs(out[size_t(y) * w + x] - ref) > 1e-5) ++bad;
    }
  CHECK(bad == 0);
  CHECK(out[size_t(20) * w + 50] == 0.0f);   // haze
  CHECK(out[size_t(40) * w + 50] == 0.3f);   // soft speck kept as is
  CHECK(out[size_t(3) * w + 3] == 0.0f);     // NaN
}

// ---------------------------------------------------------------------------------------------
// Synthetic two-plate scenes.

// The static background, scene-linear: smooth texture around 18 % grey, a highlight up to 6.0 and a
// patch of small negative values (camera black offset). Analytic, so a sub-pixel drift samples it exactly.
void background(double x, double y, float out[3]) {
  const double t = 0.9 * std::sin(0.11 * x + 0.3) * std::cos(0.083 * y) + 0.5 * std::sin(0.017 * x + 0.029 * y);
  const double hl = 6.0 * std::exp(-((x - 150) * (x - 150) + (y - 52) * (y - 52)) / (2 * 6.0 * 6.0));
  for (int c = 0; c < 3; ++c) out[c] = float(0.18 * std::exp2(t + 0.15 * c * std::sin(0.05 * y)) + hl);
  if (x > 290 && y > 170) out[0] = out[1] = out[2] = -0.002f;
}

struct Person {
  double cx, cy, rx, ry;
  float colour[3];
};

// Signed distance (px, approximate) outside an ellipse.
double ellipseDist(const Person& p, double x, double y) {
  const double nx = (x - p.cx) / p.rx, ny = (y - p.cy) / p.ry;
  return (std::sqrt(nx * nx + ny * ny) - 1.0) * std::min(p.rx, p.ry);
}

struct Scene {
  int w = 320, h = 200;
  std::vector<float> A, B, mA, mB, keep, gtClean, gtBoA;  // pictures w*h*3, mattes w*h
  std::vector<uint8_t> bInfluence;
};

struct SceneOpts {
  double driftX = 0.5, driftY = 0.0;  // B samples the background here (px)
  double exposure[3] = {0.3, 0.25, 0.35};  // B = 2^exposure x A's light (stops, per channel)
  bool reflection = true;
  bool keep = true;
  bool haze = true;
};

// Frame t: person A walks right, person B walks left; they overlap around t = 3.
Scene makeScene(int t, const SceneOpts& o) {
  Scene s;
  const int w = s.w, h = s.h;
  const size_t n = size_t(w) * h;
  s.A.resize(n * 3);
  s.B.resize(n * 3);
  s.gtClean.resize(n * 3);
  s.gtBoA.resize(n * 3);
  s.mA.resize(n);
  s.mB.resize(n);
  s.keep.assign(n, 0.0f);
  s.bInfluence.assign(n, 0);
  const Person pa{100.0 + 12 * t, 110.0, 28, 45, {0.55f, 0.32f, 0.22f}};
  const Person pb{230.0 - 10 * t, 100.0, 24, 50, {0.12f, 0.2f, 0.45f}};
  // The window (x 60-140, y 30-80): a reflection only plate A shows; the artist keeps it (Keep A). Person
  // A's grown matte reaches into it, so without Keep A the swap would replace the reflection.
  auto inWindow = [](int x, int y) { return x >= 60 && x < 140 && y >= 30 && y < 80; };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      float bgA[3], bgB[3];
      background(x, y, bgA);
      background(x + o.driftX, y + o.driftY, bgB);
      if (o.reflection && inWindow(x, y))
        for (int c = 0; c < 3; ++c) bgA[c] += float(0.08 * (1 + c) * (0.5 + 0.5 * std::sin(0.3 * x)));
      if (o.keep && inWindow(x, y)) s.keep[i] = 1.0f;
      // A person: soft 2 px edge (the tight matte), plus a shadow / defocus halo past it (not in the matte).
      auto person = [&](const Person& p, const float bg[3], float px[3], float& matte, float& halo) {
        const double d = ellipseDist(p, x, y);
        matte = float(clamp01(0.5 - d / 2.0));
        halo = d > 0 ? float(0.35 * std::exp(-(d / 3.5) * (d / 3.5))) : 0.35f;
        for (int c = 0; c < 3; ++c) px[c] = bg[c] * (1 - halo) * (1 - matte) + p.colour[c] * matte;
      };
      float a[3], b[3], b0[3], bg0[3], ha, hb, ma, mb;
      person(pa, bgA, a, ma, ha);
      person(pb, bgB, b, mb, hb);
      background(x, y, bg0);
      float mb0, hb0;
      person(pb, bg0, b0, mb0, hb0);  // B as it would be with no drift and A's exposure
      for (int c = 0; c < 3; ++c) {
        s.A[3 * i + c] = a[c];
        s.B[3 * i + c] = float(b[c] * std::exp2(o.exposure[c]));
        s.gtClean[3 * i + c] = bgA[c];  // A's clean background (incl. its reflection)
      }
      const bool infl = mb0 > 0 || hb0 > 1e-4f;
      s.bInfluence[i] = infl;
      for (int c = 0; c < 3; ++c) s.gtBoA[3 * i + c] = infl ? b0[c] : bgA[c];
      // The mattes: tight, plus a faint haze tail (below the floor) as LTX mattes carry.
      const float hzA = o.haze && std::fabs(ellipseDist(pa, x, y)) < 40 ? 0.012f : 0.0f;
      const float hzB = o.haze && std::fabs(ellipseDist(pb, x, y)) < 40 ? 0.012f : 0.0f;
      s.mA[i] = std::max(ma, hzA);
      s.mB[i] = std::max(mb, hzB);
    }
  return s;
}

struct Score {
  long exactPixels = 0, notExact = 0;  // s == 0 pixels equal to A bit for bit / not
  double p99 = 0, mean = 0;            // relative error vs the truth in the judged region
  double seamStops = 0;                // mean log2(out / truth) in the feather band (0 < s < 1)
  double maxSJump = 0;                 // largest step of s between 4-neighbours (seam hardness)
  long keepBad = 0;                    // keep == 1 pixels not equal to A
  long judged = 0;
  float outMaxNearHighlight = 0;
};

// Judges an output: bit-identity where s == 0, error against the truth where the truth is defined (Clean:
// outside the residual and Keep A; B Over A: everywhere outside Keep A), the seam band, Keep A.
Score score(const Scene& sc, const std::vector<float>& out, const std::vector<float>& s, const std::vector<float>& res,
            bool clean) {
  Score r;
  const int w = sc.w, h = sc.h;
  std::vector<double> errs;
  double band = 0;
  long bandN = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      if (s[i] == 0.0f) {
        if (std::memcmp(&out[3 * i], &sc.A[3 * i], 3 * sizeof(float)) == 0) ++r.exactPixels;
        else ++r.notExact;
      }
      if (sc.keep[i] == 1.0f && std::memcmp(&out[3 * i], &sc.A[3 * i], 3 * sizeof(float)) != 0) ++r.keepBad;
      // Seam hardness away from Keep A's own (hand-drawn, here hard) edge.
      if (x + 1 < w && sc.keep[i] == 0.0f && sc.keep[i + 1] == 0.0f)
        r.maxSJump = std::max(r.maxSJump, double(std::fabs(s[i + 1] - s[i])));
      if (y + 1 < h && sc.keep[i] == 0.0f && sc.keep[i + w] == 0.0f)
        r.maxSJump = std::max(r.maxSJump, double(std::fabs(s[i + w] - s[i])));
      if (std::hypot(x - 150.0, y - 52.0) < 4) r.outMaxNearHighlight = std::max(r.outMaxNearHighlight, out[3 * i + 1]);
      if (s[i] == 0.0f || sc.keep[i] > 0.0f || (clean && res[i] > 0.0f)) continue;
      const std::vector<float>& gt = clean ? sc.gtClean : sc.gtBoA;
      double lum = 0, lumGt = 0;
      for (int c = 0; c < 3; ++c) {
        errs.push_back(std::fabs(out[3 * i + c] - gt[3 * i + c]) / (std::fabs(gt[3 * i + c]) + 0.02));
        lum += out[3 * i + c];
        lumGt += gt[3 * i + c];
      }
      if (s[i] < 1.0f && lum > 0.01 && lumGt > 0.01) {
        band += std::log2(lum / lumGt);
        ++bandN;
      }
    }
  r.judged = long(errs.size() / 3);
  if (!errs.empty()) {
    double sum = 0;
    for (double e : errs) sum += e;
    r.mean = sum / errs.size();
    std::nth_element(errs.begin(), errs.begin() + errs.size() * 99 / 100, errs.end());
    r.p99 = errs[errs.size() * 99 / 100];
  }
  r.seamStops = bandN ? band / bandN : 0;
  return r;
}

// Limits (first measurements 2026-11-27, with margin; see docs/plate_merge.md "Measured"). The error floor
// is the 0.5 px drift itself (driftFloor(): B's background is sampled half a pixel off), not the merge.
constexpr double kP99Limit = 0.10, kMeanLimit = 0.015, kSeamLimit = 0.01;

// The error a perfect merge still has: the background half a pixel off against itself, over the frame.
void driftFloor() {
  std::vector<double> e;
  for (int y = 0; y < 200; ++y)
    for (int x = 0; x < 320; ++x) {
      float a[3], b[3];
      background(x, y, a);
      background(x + 0.5, y, b);
      for (int c = 0; c < 3; ++c) e.push_back(std::fabs(b[c] - a[c]) / (std::fabs(a[c]) + 0.02));
    }
  double sum = 0;
  for (double v : e) sum += v;
  std::nth_element(e.begin(), e.begin() + e.size() * 99 / 100, e.end());
  std::printf(" drift floor (0.5 px, whole frame): mean %.5f  p99 %.4f\n", sum / e.size(), e[e.size() * 99 / 100]);
}

struct Run {
  Score sc;
  PlateInfo info;
};

Run runScene(const Scene& sc, const PlateParams& p, bool swapPlates = false, bool clampOut = false) {
  const size_t n = size_t(sc.w) * sc.h;
  std::vector<float> out(n * 3), s(n), res(n);
  PlateViews v;
  v.swap = s.data();
  v.residual = res.data();
  Run r;
  if (swapPlates)
    mergePlates(sc.B.data(), sc.mA.data(), sc.A.data(), sc.mB.data(), sc.keep.data(), sc.w, sc.h, p, out.data(), v, &r.info);
  else
    mergePlates(sc.A.data(), sc.mA.data(), sc.B.data(), sc.mB.data(), sc.keep.data(), sc.w, sc.h, p, out.data(), v, &r.info);
  if (clampOut)
    for (auto& x : out) x = float(clamp01(x));
  r.sc = score(sc, out, s, res, p.mode == Mode::CleanBackground);
  return r;
}

void print(const char* name, const Run& r) {
  std::printf("  %-34s judged %6ld  p99 %.4f  mean %.5f  seam %+.5f st  sJump %.3f  exact %ld/%ld  keepBad %ld  "
              "gain %+.3f %+.3f %+.3f  ring %ld  hl %.2f\n",
              name, r.sc.judged, r.sc.p99, r.sc.mean, r.sc.seamStops, r.sc.maxSJump, r.sc.exactPixels,
              r.sc.exactPixels + r.sc.notExact, r.sc.keepBad, r.info.globalGainStops[0], r.info.globalGainStops[1],
              r.info.globalGainStops[2], r.info.ringPixels, r.sc.outMaxNearHighlight);
}

bool passes(const Score& s) {
  return s.notExact == 0 && s.keepBad == 0 && s.p99 <= kP99Limit && s.mean <= kMeanLimit &&
         std::fabs(s.seamStops) <= kSeamLimit;
}

// Both modes over a sequence: defaults pass; gain stable across frames (no flicker); the highlight > 1
// survives inside the swap region.
void testScenes() {
  driftFloor();
  SceneOpts o;
  PlateParams p;  // defaults: grow 8, feather 8, Level Match on
  for (int mode = 0; mode < 2; ++mode) {
    p.mode = mode ? Mode::CleanBackground : Mode::BOverA;
    std::printf(" mode %s\n", mode ? "Clean Background" : "B Over A");
    double g0[3] = {0, 0, 0}, g2[3] = {0, 0, 0};
    double gMin = 1e9, gMax = -1e9;
    const int frames = 7;
    for (int t = 0; t < frames; ++t) {
      const Scene sc = makeScene(t, o);
      const Run r = runScene(sc, p);
      char name[64];
      std::snprintf(name, sizeof name, "frame %d", t);
      print(name, r);
      CHECK(passes(r.sc));
      CHECK(r.info.matched);
      for (int c = 0; c < 3; ++c) {
        g0[c] += r.info.globalGainStops[c];
        g2[c] += r.info.globalGainStops[c] * r.info.globalGainStops[c];
        CHECK(std::fabs(r.info.globalGainStops[c] + o.exposure[c]) < 0.02);
      }
      gMin = std::min(gMin, r.info.globalGainStops[1]);
      gMax = std::max(gMax, r.info.globalGainStops[1]);
      if (t == 6) CHECK(r.sc.outMaxNearHighlight > 4.0f);  // B's person passes the highlight at t >= 5
    }
    std::printf("  gain over %d frames (G): range %.4f st\n", frames, gMax - gMin);
    CHECK(gMax - gMin < 0.01);
  }
  // Overlap frame in Clean mode has a residual (both persons).
  {
    p.mode = Mode::CleanBackground;
    const Run r = runScene(makeScene(3, o), p);
    CHECK(r.info.residualPixels > 500);
  }
  // No drift / no exposure: the merge is near exact (only the halos left beyond Grow).
  {
    SceneOpts z;
    z.driftX = 0;
    for (double& e : z.exposure) e = 0;
    p.mode = Mode::BOverA;
    const Run r = runScene(makeScene(2, z), p);
    print("no drift, no exposure", r);
    CHECK(r.sc.p99 < 0.01);
  }
}

// Fault injections: each must be caught by the checks above. Also the guard regions: halo needs Grow.
void testFaults() {
  SceneOpts o;
  const Scene sc = makeScene(4, o);
  PlateParams p;
  p.mode = Mode::CleanBackground;
  const Run good = runScene(sc, p);
  CHECK(passes(good.sc));
  // 1. Swapped plates (A and B exchanged, mattes not): the persons land in the wrong plate.
  const Run swapped = runScene(sc, p, true);
  print("FAULT swapped plates", swapped);
  CHECK(!passes(swapped.sc));
  // 2. A clamp to 0-1: the highlight and the negative patch no longer survive (bit-identity breaks).
  const Run clamped = runScene(sc, p, false, true);
  print("FAULT clamp 0-1", clamped);
  CHECK(!passes(clamped.sc));
  CHECK(clamped.sc.notExact > 0);
  // 3. Hard seam: feather 0 makes s jump 0 -> 1 between neighbours.
  PlateParams hard = p;
  hard.featherPx = 0;
  const Run hardRun = runScene(sc, hard);
  print("FAULT hard seam (feather 0)", hardRun);
  CHECK(hardRun.sc.maxSJump > 0.99);
  CHECK(good.sc.maxSJump < 0.35);  // feather 8: smoothstep max slope 1.5/8 = 0.19, plus the matte's own edge
  // 4. No Level Match with a 0.3 stop exposure offset: a seam step of about the offset.
  PlateParams nomatch = p;
  nomatch.levelMatch = false;
  const Run nm = runScene(sc, nomatch);
  print("FAULT no level match", nm);
  CHECK(!passes(nm.sc));
  CHECK(std::fabs(nm.sc.seamStops) > 0.1);
  // 5. Grow 0: the shadow halo past the tight matte stays (A's halo is copied into the result).
  PlateParams nogrow = p;
  nogrow.growPx = 0;
  nogrow.featherPx = 2;
  const Run ng = runScene(sc, nogrow);
  print("grow 0 (halo left)", ng);
  CHECK(!passes(ng.sc));
  // 6. Keep A not connected: the window reflection under the swap is lost (judged with keep zeroed).
  Scene nk = sc;
  std::fill(nk.keep.begin(), nk.keep.end(), 0.0f);
  const Run noKeep = runScene(nk, p);
  print("FAULT no Keep A (reflection judged)", noKeep);
  CHECK(!passes(noKeep.sc));
}

// Timing at UHD (reported; a generous limit only catches an accidental quadratic path).
void testTiming() {
  const int w = 3840, h = 2160;
  const size_t n = size_t(w) * h;
  std::vector<float> A(n * 3), B(n * 3), mA(n), mB(n), out(n * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      float bg[3];
      background(x * 0.1, y * 0.1, bg);
      for (int c = 0; c < 3; ++c) {
        A[3 * i + c] = bg[c];
        B[3 * i + c] = bg[c] * 1.2f;
      }
      mA[i] = float(clamp01(0.5 - (std::hypot(x - 1500.0, (y - 1100.0) * 0.5) - 400) / 2));
      mB[i] = float(clamp01(0.5 - (std::hypot(x - 2300.0, (y - 1000.0) * 0.5) - 400) / 2));
    }
  PlateParams p;
  p.growPx = 16;
  p.featherPx = 16;
  double best = 1e9;
  for (int k = 0; k < 3; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    mergePlates(A.data(), mA.data(), B.data(), mB.data(), nullptr, w, h, p, out.data());
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::printf(" UHD merge: %.1f ms (best of 3)\n", best);
  CHECK(best < 3000);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string what = argc > 1 ? argv[1] : "all";
  if (what == "smoke" || what == "all") {
    testTruthTable();
    testSoftUnion();
    testSwapWeight();
  }
  if (what == "scenes" || what == "all") {
    testScenes();
    testFaults();
    testTiming();
  }
  std::printf("test_plate_merge %s: %s (%d failures)\n", what.c_str(), failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
