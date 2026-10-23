// Foreground Colour (A7): known answers, a slow reference of the paper's solver, the synthetic
// "fgcolor" scene against its true foreground, and UHD timing. Ignore Outer px: unchanged at 0
// (output hashes of the version before it), a grown matte's green edge removed, the cost on a
// clean matte, the deep interior untouched, a thin feature, timing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "ops/edge_field.h"
#include "ops/edge_width.h"
#include "ops/foreground_colour.h"
#include "ops/shrink_grow.h"
#include "synth.h"

using mr::FgParams;
using mr::FgStats;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

std::vector<float> run(const std::vector<float>& rgb, const std::vector<float>& a, int w, int h,
                       const FgParams& p, FgStats* st = nullptr) {
  std::vector<float> fg(rgb.size(), -1.0f);
  mr::foregroundColour(rgb.data(), a.data(), fg.data(), w, h, p, st);
  return fg;
}

// The paper's solver written out plainly (Germer et al. 2020; pymatting _estimate_fb_ml): every
// level dense, raster-order Gauss-Seidel (redBlack = false) or red-black order, fullIters sweeps
// over-relaxed by omega at full resolution (the paper: 2, 1.0), clamp to [lo, hi]. Slow; small images only (no level
// skipped).
std::vector<float> reference(const std::vector<float>& img, const std::vector<float>& alpha, int w,
                             int h, double reg, bool redBlack, int fullIters, double omega) {
  double lo = 0, hi = 1;
  for (float v : img) lo = std::min(lo, double(v)), hi = std::max(hi, double(v));
  const int n = std::max(1, int(std::ceil(std::log2(double(std::max(w, h))))));
  std::vector<double> F(3, 0.0), B(3, 0.0);
  int pw = 1, ph = 1;
  for (int l = 0; l <= n; ++l) {
    const int lw = std::max(1, int(std::lround(std::pow(double(w), double(l) / n))));
    const int lh = std::max(1, int(std::lround(std::pow(double(h), double(l) / n))));
    std::vector<double> I(size_t(lw) * lh * 3), A(size_t(lw) * lh), f(I.size()), b(I.size());
    for (int y = 0; y < lh; ++y)
      for (int x = 0; x < lw; ++x) {
        const int sx = std::min(x * w / lw, w - 1), sy = std::min(y * h / lh, h - 1);
        const int px = std::min(x * pw / lw, pw - 1), py = std::min(y * ph / lh, ph - 1);
        A[y * lw + x] = alpha[sy * w + sx];
        for (int c = 0; c < 3; ++c) {
          I[(y * lw + x) * 3 + c] = img[(sy * w + sx) * 3 + c];
          f[(y * lw + x) * 3 + c] = F[(py * pw + px) * 3 + c];
          b[(y * lw + x) * 3 + c] = B[(py * pw + px) * 3 + c];
          // Full resolution starts from what is known (not in the paper): F = I where a = 1,
          // B = I where a = 0.
          if (l == n && A[y * lw + x] >= 1) f[(y * lw + x) * 3 + c] = I[(y * lw + x) * 3 + c];
          if (l == n && A[y * lw + x] <= 0) b[(y * lw + x) * 3 + c] = I[(y * lw + x) * 3 + c];
        }
      }
    const int iters = l == n ? fullIters : lw <= 32 && lh <= 32 ? 10 : 2;
    for (int it = 0; it < iters; ++it)
      for (int pass = 0; pass < (redBlack ? 2 : 1); ++pass)
        for (int y = 0; y < lh; ++y)
          for (int x = 0; x < lw; ++x) {
            if (redBlack && ((x + y) & 1) != pass) continue;
            const int i = y * lw + x;
            const double a0 = A[i], a1 = 1 - a0;
            double a00 = a0 * a0, a01 = a0 * a1, a11 = a1 * a1, b0[3], b1[3];
            for (int c = 0; c < 3; ++c) b0[c] = a0 * I[i * 3 + c], b1[c] = a1 * I[i * 3 + c];
            const int nx[4] = {x - 1, x + 1, x, x}, ny[4] = {y, y, y - 1, y + 1};
            for (int k = 0; k < 4; ++k) {
              const int j = std::min(std::max(ny[k], 0), lh - 1) * lw + std::min(std::max(nx[k], 0), lw - 1);
              const double g = reg + std::fabs(a0 - A[j]);
              a00 += g;
              a11 += g;
              for (int c = 0; c < 3; ++c) b0[c] += g * f[j * 3 + c], b1[c] += g * b[j * 3 + c];
            }
            const double det = a00 * a11 - a01 * a01;
            for (int c = 0; c < 3; ++c) {
              const double om = l == n ? omega : 1.0;
              const double nf = (a11 * b0[c] - a01 * b1[c]) / det, nb = (a00 * b1[c] - a01 * b0[c]) / det;
              f[i * 3 + c] = std::min(hi, std::max(lo, f[i * 3 + c] + om * (nf - f[i * 3 + c])));
              b[i * 3 + c] = std::min(hi, std::max(lo, b[i * 3 + c] + om * (nb - b[i * 3 + c])));
            }
          }
    F.swap(f);
    B.swap(b);
    pw = lw;
    ph = lh;
  }
  std::vector<float> out(F.size());
  for (size_t i = 0; i < out.size(); ++i) out[i] = alpha[i / 3] >= 1.0f ? img[i] : float(F[i]);
  return out;
}

// Mean |a - b| over the soft pixels (lo < alpha < hi), all channels.
double bandError(const std::vector<float>& a, const std::vector<float>& b,
                 const std::vector<float>& alpha, float lo = 0.02f, float hi = 0.98f) {
  double sum = 0;
  long long cnt = 0;
  for (size_t i = 0; i < alpha.size(); ++i)
    if (alpha[i] > lo && alpha[i] < hi)
      for (int c = 0; c < 3; ++c) sum += std::fabs(a[i * 3 + c] - b[i * 3 + c]), ++cnt;
  return cnt ? sum / cnt : 0.0;
}

// A constant foreground over a constant background with a soft ramp: F comes back almost
// exactly, also above 1 (scene-linear), and solid pixels are the picture bit for bit.
void testConstant() {
  const int w = 160, h = 90;
  const float F[3] = {2.5f, 0.4f, 0.2f}, B[3] = {0.1f, 0.7f, 0.1f};
  std::vector<float> a(size_t(w) * h), rgb(a.size() * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      // A slanted 12 px ramp.
      a[i] = std::min(1.0f, std::max(0.0f, float((x + 0.3 * y - 70) / 12.0)));
      for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = a[i] * F[c] + (1 - a[i]) * B[c];
    }
  const std::vector<float> fg = run(rgb, a, w, h, FgParams());
  double worst = 0, worstOutside = 0;
  int notExact = 0;
  for (size_t i = 0; i < a.size(); ++i)
    for (int c = 0; c < 3; ++c) {
      if (a[i] >= 1.0f) notExact += std::memcmp(&fg[i * 3 + c], &rgb[i * 3 + c], 4) != 0;
      else if (a[i] > 0.02f) worst = std::max(worst, double(std::fabs(fg[i * 3 + c] - F[c])));
      else worstOutside = std::max(worstOutside, double(std::fabs(fg[i * 3 + c] - F[c])));
    }
  std::printf("constant F: max error in the ramp %.5f, at alpha <= 0.02 %.5f\n", worst, worstOutside);
  CHECK(notExact == 0);
  CHECK(worst < 0.005);
  CHECK(worstOutside < 0.05);  // alpha = 0: the foreground extended, not the background
}

// Small random scenes: with Extend px covering the frame the solver equals the plain red-black
// reference; with the default it stays close; against the truth it is at least as good as the
// paper's own order and sweep count.
void testReference() {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (int t = 0; t < 6; ++t) {
    const int w = 40 + int(u(rng) * 60), h = 30 + int(u(rng) * 50);
    const double cx = w * (0.3 + 0.4 * u(rng)), cy = h * (0.3 + 0.4 * u(rng));
    const double rad = std::min(w, h) * (0.2 + 0.15 * u(rng)), soft = 1 + 6 * u(rng);
    std::vector<float> a(size_t(w) * h), rgb(a.size() * 3), tf(rgb.size());
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        a[i] = float(std::min(1.0, std::max(0.0, (rad - std::hypot(x - cx, y - cy)) / soft + 0.5)));
        for (int c = 0; c < 3; ++c) {
          tf[i * 3 + c] = float(0.3 + 0.5 * c / 2.0 + 0.2 * std::sin(0.2 * x + c));
          const double bg = 0.6 - 0.2 * c + 0.1 * std::cos(0.15 * y);
          rgb[i * 3 + c] = float(a[i] * tf[i * 3 + c] + (1 - a[i]) * bg);
        }
      }
    FgParams p;
    const std::vector<float> rbRef = reference(rgb, a, w, h, p.smoothness, true, 8, 1.7);
    const std::vector<float> paper = reference(rgb, a, w, h, p.smoothness, false, 2, 1.0);
    const std::vector<float> ours = run(rgb, a, w, h, p);
    p.extendPx = 250;
    const std::vector<float> whole = run(rgb, a, w, h, p);
    double maxWhole = 0;
    for (size_t i = 0; i < whole.size(); ++i) maxWhole = std::max(maxWhole, double(std::fabs(whole[i] - rbRef[i])));
    const double dOurs = bandError(ours, rbRef, a, 0.0f, 1.0f);
    const double eOurs = bandError(ours, tf, a), ePaper = bandError(paper, tf, a), eNaive = bandError(rgb, tf, a);
    std::printf("reference %dx%d soft %.1f: whole vs red-black %.2g, band-only vs it %.4f; "
                "error vs truth: ours %.4f, paper %.4f, picture %.4f\n",
                w, h, soft, maxWhole, dOurs, eOurs, ePaper, eNaive);
    CHECK(maxWhole < 1e-4);
    CHECK(dOurs < 0.01);
    CHECK(eOurs < ePaper + 0.001);
    CHECK(eOurs < 0.5 * eNaive);
  }
}

// Mix: 0 = the picture everywhere, 0.5 = halfway.
void testMix() {
  const synth::Scene s = synth::make("fgcolor", 240, 135);
  FgParams p;
  const std::vector<float> full = run(s.rgb, s.matte, s.w, s.h, p);
  p.mix = 0.0f;
  const std::vector<float> none = run(s.rgb, s.matte, s.w, s.h, p);
  p.mix = 0.5f;
  const std::vector<float> half = run(s.rgb, s.matte, s.w, s.h, p);
  int notPicture = 0, notHalf = 0;
  for (size_t i = 0; i < none.size(); ++i) {
    notPicture += none[i] != s.rgb[i];
    notHalf += std::fabs(half[i] - 0.5f * (full[i] + s.rgb[i])) > 1e-5f;
  }
  CHECK(notPicture == 0);
  CHECK(notHalf == 0);
}

// The "fgcolor" scene (soft hair over green): the foreground error in the soft band against
// the true foreground, and a recomposite over a new background against the true comp, both
// compared with using the picture as the foreground (no decontamination).
void testScene(int w, int h) {
  const synth::Scene s = synth::make("fgcolor", w, h);
  FgStats st;
  const std::vector<float> fg = run(s.rgb, s.matte, w, h, FgParams(), &st);
  auto compErr = [&](const std::vector<float>& f, double* worst) {
    double sum = 0;
    long long cnt = 0;
    *worst = 0;
    for (size_t i = 0; i < s.matte.size(); ++i) {
      const float a = s.matte[i];
      if (a <= 0.02f || a >= 0.98f) continue;
      for (int c = 0; c < 3; ++c) {
        // a F + (1 - a) N against the true comp over any new background N.
        const double e = std::fabs(a * (f[i * 3 + c] - s.fg[i * 3 + c]));
        sum += e;
        ++cnt;
        *worst = std::max(*worst, e);
      }
    }
    return sum / cnt;
  };
  int solidChanged = 0, greenAtZero = 0, nearZero = 0;
  for (size_t i = 0; i < s.matte.size(); ++i) {
    if (s.matte[i] >= 1.0f)
      for (int c = 0; c < 3; ++c) solidChanged += std::memcmp(&fg[i * 3 + c], &s.rgb[i * 3 + c], 4) != 0;
    if (s.matte[i] <= 0.0f) {  // the background's green must not come back where alpha = 0
      ++nearZero;
      greenAtZero += fg[i * 3 + 1] > fg[i * 3 + 0];
    }
  }
  double wOurs, wNaive;
  const double eOurs = bandError(fg, s.fg, s.matte), eNaive = bandError(s.rgb, s.fg, s.matte);
  const double cOurs = compErr(fg, &wOurs), cNaive = compErr(s.rgb, &wNaive);
  std::printf("fgcolor %dx%d: %lld edge px, %lld solved, %d levels; fg error in band: ours %.4f, "
              "picture %.4f; recomposite: ours mean %.4f max %.3f, picture mean %.4f max %.3f; "
              "alpha 0 greenish %d of %d\n",
              w, h, st.edgePixels, st.solvedPixels, st.levels, eOurs, eNaive, cOurs, wOurs, cNaive,
              wNaive, greenAtZero, nearZero);
  CHECK(solidChanged == 0);
  CHECK(eOurs < 0.06 * eNaive);
  CHECK(cOurs < 0.06 * cNaive);
  CHECK(greenAtZero == 0);
}

void testTiming() {
  const int w = 3840, h = 2160;
  const synth::Scene s = synth::make("fgcolor", w, h);
  std::vector<float> fg(s.rgb.size());
  FgStats st;
  double best = 1e9;
  for (int k = 0; k < 3; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    mr::foregroundColour(s.rgb.data(), s.matte.data(), fg.data(), w, h, FgParams(), &st);
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::printf("UHD fgcolor: %.1f ms (best of 3), %lld solved px\n", best, st.solvedPixels);
  CHECK(best < 60.0);  // target: well under 40 ms (decision 5); slack for a busy machine
  // Worst case: soft everywhere (haze over the whole frame), every pixel solved.
  std::vector<float> hazy(s.matte);
  for (float& v : hazy) v = std::min(0.99f, std::max(0.01f, v));
  const auto t0 = std::chrono::steady_clock::now();
  mr::foregroundColour(s.rgb.data(), hazy.data(), fg.data(), w, h, FgParams(), &st);
  std::printf("UHD all soft: %.1f ms, %lld solved px\n",
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
              st.solvedPixels);
}

// Degenerate sizes run and keep solid pixels; a frame without soft pixels is the picture where
// alpha = 1.
void testTiny() {
  const int sizes[][2] = {{1, 1}, {1, 7}, {9, 1}, {2, 2}, {33, 3}};
  for (const auto& sz : sizes) {
    const int w = sz[0], h = sz[1];
    std::vector<float> a(size_t(w) * h), rgb(a.size() * 3);
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = i % 3 == 0 ? 1.0f : i % 3 == 1 ? 0.5f : 0.0f;
      for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = 0.1f * float(c + 1) + 0.01f * float(i % 7);
    }
    FgParams p;
    p.ignoreOuterPx = sz[0] == 33 ? 2.0 : 0.0;
    const std::vector<float> fg = run(rgb, a, w, h, p);
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
      for (int c = 0; c < 3; ++c) {
        bad += !std::isfinite(fg[i * 3 + c]);
        if (a[i] >= 1.0f && p.ignoreOuterPx == 0) bad += fg[i * 3 + c] != rgb[i * 3 + c];
      }
    CHECK(bad == 0);
  }
}

uint64_t fnv(const std::vector<float>& v) {
  uint64_t hsh = 1469598103934665603ULL;
  const unsigned char* b = reinterpret_cast<const unsigned char*>(v.data());
  for (size_t i = 0; i < v.size() * 4; ++i) hsh = (hsh ^ b[i]) * 1099511628211ULL;
  return hsh;
}

// Ignore Outer px 0 is the previous version bit for bit: output hashes recorded from
// foreground_colour.cpp as in 0.2.0 (c739311, before Ignore Outer px) on x86-64 (GCC 13, BFD).
// A deliberate change to the solver must re-record them.
// Checked only where the synthetic inputs hash the same (libm) and on x86-64 (arm64 compilers
// contract to FMA, which changes the last bits); elsewhere reported as skipped.
void testUnchangedAtZero() {
  struct Case {
    const char* scene;
    int w, h;
    double extend;
    float mix;
    uint64_t in, out;
  };
  const Case cases[] = {
      {"fgcolor", 480, 270, 3, 1.0f, 0x251d06d47fe2ae30ULL, 0x291a79191d4c88b9ULL},
      {"fgcolor", 480, 270, 0, 1.0f, 0x251d06d47fe2ae30ULL, 0x2f798bcec4148eadULL},
      {"fgcolor", 480, 270, 20, 0.5f, 0x251d06d47fe2ae30ULL, 0xda7fe693d1eefee5ULL},
      {"mixed_edge", 320, 180, 3, 1.0f, 0x3b4f15e81198ff7fULL, 0x1c570aaa41388ac7ULL},
      {"all", 320, 180, 3, 1.0f, 0xb2c06132e72e3ef6ULL, 0x3bd5056c94d3d47fULL},
      {"fgcolor", 1920, 1080, 3, 1.0f, 0x8620855a75a0abf2ULL, 0xc3b98af23c9e944bULL},
  };
  int checked = 0, skipped = 0;
  for (const Case& c : cases) {
    const synth::Scene s = synth::make(c.scene, c.w, c.h);
    std::vector<float> in(s.rgb);
    in.insert(in.end(), s.matte.begin(), s.matte.end());
#if defined(__x86_64__)
    const bool comparable = fnv(in) == c.in;
#else
    const bool comparable = false;
#endif
    if (!comparable) {
      ++skipped;
      continue;
    }
    FgParams p;
    p.extendPx = c.extend;
    p.mix = c.mix;
    p.ignoreOuterPx = 0.0;
    FgStats st;
    const std::vector<float> fg = run(s.rgb, s.matte, c.w, c.h, p, &st);
    CHECK(fnv(fg) == c.out);
    CHECK(st.ignoredPixels == 0);
    ++checked;
  }
  std::printf("Ignore Outer 0 = previous version: %d scenes bit-identical, %d skipped (inputs differ "
              "or not x86-64)\n", checked, skipped);
}

// Mean greenness g - (r + b) / 2 and mean |F - true F| of fg in rings of the matte's signed
// distance (px, > 0 inside): [6, 2), [2, 0), [0, -2), [-2, -6).
struct Rings {
  double green[4] = {0, 0, 0, 0}, err[4] = {0, 0, 0, 0};
};
Rings rings(const synth::Scene& s, const std::vector<float>& m, const std::vector<float>& fg) {
  const std::vector<float> sd = mr::signedEdgeDistance(m.data(), s.w, s.h, 0.5f, 8);
  const double edge[5] = {6, 2, 0, -2, -6};
  Rings r;
  long long cnt[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < m.size(); ++i)
    for (int k = 0; k < 4; ++k)
      if (sd[i] <= edge[k] && sd[i] > edge[k + 1]) {
        r.green[k] += fg[i * 3 + 1] - 0.5 * (fg[i * 3] + fg[i * 3 + 2]);
        for (int c = 0; c < 3; ++c) r.err[k] += std::fabs(fg[i * 3 + c] - s.fg[i * 3 + c]) / 3;
        ++cnt[k];
      }
  for (int k = 0; k < 4; ++k) r.green[k] /= std::max(1LL, cnt[k]), r.err[k] /= std::max(1LL, cnt[k]);
  return r;
}

// The Flame test's case: the clean matte grown 4.8 px (and then a hard 0.5 px Edge Width), so
// the pixels between the true edge and the matte's are alpha 1 but green. With Ignore Outer px
// 6 (>= the growth + the true edge's 1 px half width) the greenness in each ring comes back
// near the clean matte's, and Extend 20 is no longer greener than Extend 0.
void testGrownMatte() {
  const int w = 1920, h = 1080;
  const synth::Scene s = synth::make("fgcolor", w, h);
  const char* ringName[4] = {"in 6-2", "in 2-0", "out 0-2", "out 2-6"};
  Rings clean[2];
  for (int e = 0; e < 2; ++e) {
    FgParams p;
    p.extendPx = e ? 20 : 0;
    clean[e] = rings(s, s.truth, run(s.rgb, s.truth, w, h, p));
  }
  for (int variant = 0; variant < 2; ++variant) {
    std::vector<float> m(s.truth);
    mr::ShrinkGrowParams g;
    g.amountPx = 4.8;
    mr::shrinkGrow(m.data(), m.data(), w, h, g);
    if (variant) {
      mr::EdgeWidthParams ew;
      ew.widthPx = 0.5;
      mr::edgeWidth(m.data(), m.data(), w, h, ew);
    }
    Rings fixed[2];
    for (int e = 0; e < 2; ++e) {
      FgParams p;
      p.extendPx = e ? 20 : 0;
      const Rings before = rings(s, m, run(s.rgb, m, w, h, p));
      p.ignoreOuterPx = 6;
      fixed[e] = rings(s, m, run(s.rgb, m, w, h, p));
      std::printf("grown 4.8%s, Extend %2d, greenness (clean / Ignore 0 / Ignore 6):",
                  variant ? " + Edge Width 0.5" : "", e ? 20 : 0);
      for (int k = 0; k < 4; ++k)
        std::printf("  %s %+.3f / %+.3f / %+.3f", ringName[k], clean[e].green[k], before.green[k],
                    fixed[e].green[k]);
      std::printf("\n");
      for (int k = 0; k < 4; ++k) {
        CHECK(before.green[k] > clean[e].green[k] + 0.15);  // the defect is there
        // Inside: within 0.02 of the clean matte; outside (the grown matte's soft band, pure
        // background in the picture, solved with its doubtful alpha): within 0.06.
        CHECK(fixed[e].green[k] < clean[e].green[k] + (k < 2 ? 0.02 : 0.06));
      }
    }
    for (int k = 0; k < 4; ++k) {
      CHECK(fixed[1].green[k] < fixed[0].green[k] + 0.03);  // Extend 20 not greener (was +0.05-0.11)
      CHECK(fixed[1].err[k] < fixed[0].err[k] + 0.002);     // nor further from the true colour
    }
  }
}

// The cost on a good matte: the error against the true foreground in the soft band (0.02-0.98
// of the clean matte) and in the recomposite, for Ignore Outer px 0, 3, 6, against the picture.
// Pixels more than N + 1 px inside stay the picture bit for bit, nothing is NaN, Mix 0 is the
// picture.
void testCleanMatteCost() {
  const int w = 1920, h = 1080;
  const synth::Scene s = synth::make("fgcolor", w, h);
  const std::vector<float> sd = mr::signedEdgeDistance(s.truth.data(), w, h, 0.5f, 12);
  const double eNaive = bandError(s.rgb, s.fg, s.truth);
  double e0 = 0;
  for (double n : {0.0, 3.0, 6.0}) {
    FgParams p;
    p.ignoreOuterPx = n;
    FgStats st;
    const std::vector<float> fg = run(s.rgb, s.truth, w, h, p, &st);
    double comp = 0, worst = 0;
    long long cnt = 0;
    int deepChanged = 0, nan = 0, deep = 0;
    for (size_t i = 0; i < s.truth.size(); ++i) {
      const float a = s.truth[i];
      for (int c = 0; c < 3; ++c) {
        nan += !std::isfinite(fg[i * 3 + c]);
        if (a > 0.02f && a < 0.98f) {
          const double e = std::fabs(a * (fg[i * 3 + c] - s.fg[i * 3 + c]));
          comp += e, ++cnt, worst = std::max(worst, e);
        }
      }
      if (a >= 1.0f && sd[i] > n + 1.0) {
        ++deep;
        deepChanged += std::memcmp(&fg[i * 3], &s.rgb[i * 3], 12) != 0;
      }
    }
    const double e = bandError(fg, s.fg, s.truth);
    if (n == 0) e0 = e;
    std::printf("clean matte, Ignore Outer %.0f: fg error in band %.4f (picture %.4f), recomposite "
                "mean %.4f max %.3f; %lld ignored px; deep interior changed %d of %d\n",
                n, e, eNaive, comp / cnt, worst, st.ignoredPixels, deepChanged, deep);
    CHECK(nan == 0);
    CHECK(deepChanged == 0);
    CHECK(e < (n == 0 ? 0.06 : 0.25) * eNaive);  // still far better than no unmixing
    CHECK(e < e0 + 0.002 + 0.0015 * n);          // measured: +0.005 at 3, +0.008 at 6
    p.mix = 0.0f;
    const std::vector<float> none = run(s.rgb, s.truth, w, h, p);
    int notPicture = 0;
    for (size_t i = 0; i < none.size(); ++i) notPicture += none[i] != s.rgb[i];
    CHECK(notPicture == 0);
  }
}

// Thin features: a bar 4 px wide (2 px solid core, 1 px ramps) and a diagonal one, alone over
// green, with Ignore Outer px 6. Nothing is more than 6 px inside, so there is no deeper colour
// to carry in: they keep their own data, i.e. the output is exactly that of Ignore Outer 0
// (ignoring them all left their colour undetermined: black, found by this test). The same bar
// joined to a disc (r 20) is ignored only near the joint (within 7.5 px of the disc's core) and
// keeps the true colour; nothing is NaN.
void testThinFeature() {
  const int w = 120, h = 80;
  const float F[3] = {0.8f, 0.3f, 0.2f}, B[3] = {0.1f, 0.7f, 0.1f};
  for (int joined = 0; joined < 2; ++joined) {
    std::vector<float> a(size_t(w) * h), rgb(a.size() * 3);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        const double dBar = 2.0 - std::fabs(x - 30.0);  // vertical, 4 px
        const double dDiag = 2.0 - std::fabs((x - 80.0) - 0.5 * (y - 40.0)) / std::hypot(1.0, 0.5);
        const double dDisc = joined ? 20.0 - std::hypot(x - 30.0, y - 60.0) : -1e9;
        a[i] = float(std::min(1.0, std::max(0.0, std::max({dBar, dDiag, dDisc}) / 2.0 + 0.5)));
        for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = a[i] * F[c] + (1 - a[i]) * B[c];
      }
    FgParams p;
    const std::vector<float> off = run(rgb, a, w, h, p);
    p.ignoreOuterPx = 6;
    FgStats st;
    const std::vector<float> fg = run(rgb, a, w, h, p, &st);
    double worst = 0;
    int bad = 0, differ = 0, ignoredFar = 0;
    for (size_t i = 0; i < a.size(); ++i)
      for (int c = 0; c < 3; ++c) {
        bad += !std::isfinite(fg[i * 3 + c]);
        differ += std::memcmp(&fg[i * 3 + c], &off[i * 3 + c], 4) != 0;
        if (a[i] >= 0.5f) worst = std::max(worst, double(std::fabs(fg[i * 3 + c] - F[c])));
        // Solid bar pixels far above the disc (its core starts at y = 33.5) are not ignored: the
        // picture bit for bit.
        if (int(i) / w < 20 && std::abs(int(i) % w - 30) <= 2 && a[i] >= 1.0f)
          ignoredFar += std::memcmp(&fg[i * 3 + c], &rgb[i * 3 + c], 4) != 0;
      }
    std::printf("thin bars%s, Ignore Outer 6: %lld ignored px, %d values differ from Ignore Outer 0, "
                "max |F - true F| at alpha >= 0.5: %.4f\n", joined ? " + disc" : "",
                st.ignoredPixels, differ, worst);
    CHECK(bad == 0);
    CHECK(worst < 0.05);
    if (!joined) CHECK(st.ignoredPixels == 0 && differ == 0);
    if (joined) CHECK(st.ignoredPixels > 0 && ignoredFar == 0);
  }
}

void testIgnoreTiming() {
  const int w = 3840, h = 2160;
  const synth::Scene s = synth::make("fgcolor", w, h);
  std::vector<float> fg(s.rgb.size());
  for (double n : {0.0, 3.0, 6.0, 10.0}) {
    FgParams p;
    p.ignoreOuterPx = n;
    FgStats st;
    double best = 1e9;
    for (int k = 0; k < 3; ++k) {
      const auto t0 = std::chrono::steady_clock::now();
      mr::foregroundColour(s.rgb.data(), s.matte.data(), fg.data(), w, h, p, &st);
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("UHD fgcolor, Ignore Outer %.0f: %.1f ms (best of 3), %lld ignored px\n", n, best,
                st.ignoredPixels);
    if (n == 3.0) CHECK(best < 60.0);  // target: under 40 ms; slack for a busy machine
  }
}

}  // namespace

int main() {
  testUnchangedAtZero();
  testGrownMatte();
  testCleanMatteCost();
  testThinFeature();
  testTiny();
  testConstant();
  testReference();
  testMix();
  testScene(480, 270);
  testScene(1920, 1080);
  testTiming();
  testIgnoreTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
