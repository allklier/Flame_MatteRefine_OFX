// Foreground Colour (A7): known answers, a slow reference of the paper's solver, the synthetic
// "fgcolor" scene against its true foreground, and UHD timing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ops/foreground_colour.h"
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
    const std::vector<float> fg = run(rgb, a, w, h, FgParams());
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
      for (int c = 0; c < 3; ++c) {
        bad += !std::isfinite(fg[i * 3 + c]);
        if (a[i] >= 1.0f) bad += fg[i * 3 + c] != rgb[i * 3 + c];
      }
    CHECK(bad == 0);
  }
}

}  // namespace

int main() {
  testTiny();
  testConstant();
  testReference();
  testMix();
  testScene(480, 270);
  testScene(1920, 1080);
  testTiming();
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
