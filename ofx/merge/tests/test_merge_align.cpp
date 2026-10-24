// Matte Merge alignment and sizing (user decisions 2026-10-24): registration on the picture
// (known moves, display and log pictures, a busy background as decoy), the fallbacks and the
// model reported in MergeInfo, sizes as fractions of the frame height (the same scene at two sizes
// gives the same result), and reusing a MergeWorkspace (bit-identical to a fresh call).
// Scenes: merge_synth.h's hair scenes at 960x540 (fast), the detail rebuilt here with known moves.
// Mutation checks (2026-10-24): each case names a code change that made it fail.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "matte_merge.h"
#include "merge_picreg.h"
#include "merge_synth.h"
#include "ops/edge_field.h"

using namespace mr::merge;
using mmsynth::MergeScene;

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

struct Move {
  double tx, ty, s;
};

// The detail matte for a known move A(p) = c + s (p - c) + t: detail(x) = d0(A^-1(x)), d0 = the
// truth body (2 px edge, no thickening) with the real strands and the scene's fake strands.
// keep(x, y) (optional): which strands to keep (in base coordinates).
template <class Keep>
std::vector<float> movedDetail(const MergeScene& m, const Move& mv, Keep keep) {
  const int w = m.w, h = m.h;
  std::vector<float> d0(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const double st = keep(x, y) ? std::max(m.strands[i], m.fake[i]) : 0.0f;
      d0[i] = float(1.0 - (1.0 - m.body[i]) * (1.0 - st));
    }
  const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
  std::vector<float> d(d0.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      d[size_t(y) * w + x] = mr::sampleBilinear(d0.data(), w, h, cx + (x - cx - mv.tx) / mv.s, cy + (y - cy - mv.ty) / mv.s);
  return d;
}
std::vector<float> movedDetail(const MergeScene& m, const Move& mv) {
  return movedDetail(m, mv, [](int, int) { return true; });
}

// Max distance between the estimated and the true detail position over the base's 0.5 contour.
double regError(const MergeScene& m, const Registration& r, const Move& mv) {
  const double cx = (m.w - 1) / 2.0, cy = (m.h - 1) / 2.0;
  double worst = 0;
  for (int y = 0; y < m.h; ++y)
    for (int x = 0; x < m.w; ++x) {
      if (!mr::isContour(m.base.data(), m.w, m.h, x, y)) continue;
      double qx, qy;
      r.apply(x, y, &qx, &qy);
      worst = std::max(worst, std::hypot(qx - (cx + mv.s * (x - cx) + mv.tx), qy - (cy + mv.s * (y - cy) + mv.ty)));
    }
  return worst;
}

MergeParams defaults(int h) {
  MergeParams p;
  scaleToHeight(p, HeightSizes(), h);
  return p;
}

MergeInfo runMerge(const MergeScene& m, const std::vector<float>& base, const std::vector<float>& det,
                   const float* pic, const MergeParams& p, std::vector<float>* outp = nullptr) {
  std::vector<float> out(base.size());
  MergeInfo info;
  merge(base.data(), det.data(), pic, m.w, m.h, p, out.data(), MergeViews(), &info);
  if (outp) outp->swap(out);
  return info;
}

void printAlign(const char* what, const MergeInfo& in, double err) {
  std::printf("  %-44s model %-22s err %.3f px | t (%.3f, %.3f) s %.5f | unc %.3f agree %.2f (%d/%d) | matte sens %.1f\n",
              what, alignModelName(in.align.model), err, in.reg.tx, in.reg.ty, in.reg.scale, in.align.uncertaintyPx,
              in.align.agreement, in.align.inliers, in.align.blocks, in.align.matte.sensitivity);
}

// Known moves on the picture, flat and busy background (the busy one is the decoy: textured, with
// thin stripes at strand scale), display and log encodings: the picture model, error < 0.5 px.
// Mutations: picreg's update "reg.tx += reg.scale * d[0]" -> "-=" (14 checks fail); the parabola's
// sub-pixel offset -> 0 (11 fail); the polarity sign forced to +1 (sgn = 1.0: 3 fail); the
// uncertainty set to 0 (the uncertainty check fails).
void testKnownMoves() {
  const Move moves[] = {{0, 0, 1}, {2.3, -1.7, 1}, {-4.1, 3.6, 1.008}, {3.3, 2.05, 0.993}};
  double worst = 0;
  for (bool busy : {false, true}) {
    const MergeScene m = mmsynth::makeMergeScene(960, 540, busy);
    std::printf("%s 960x540, picture registration:\n", m.name.c_str());
    for (const Move& mv : moves) {
      const std::vector<float> det = movedDetail(m, mv);
      for (int enc = 0; enc < 2; ++enc) {
        const MergeInfo in = runMerge(m, m.base, det, enc ? m.rgbLog.data() : m.rgbDisplay.data(), defaults(m.h));
        const double err = regError(m, in.reg, mv);
        char what[96];
        std::snprintf(what, sizeof what, "t (%.1f, %.1f) s %.3f, %s", mv.tx, mv.ty, mv.s, enc ? "log" : "display");
        printAlign(what, in, err);
        CHECK(in.align.model == AlignModel::PictureFull);
        // The 0.5 px target; flat 0.02-0.18 px measured, busy 0.12-0.37 px (at 540 lines the strands
        // are 0.5-1.5 px wide among stripes of the same scale; 0.07 px at 1620, test_merge_scenes).
        CHECK(err < (busy ? 0.5 : 0.25));
        CHECK(in.align.uncertaintyPx > 0 && in.align.uncertaintyPx < 0.25);
        worst = std::max(worst, err);
      }
    }
  }
  std::printf("  worst error %.3f px\n", worst);

  // Polarity: the picture inverted (strands darker than their surroundings): the same move.
  const MergeScene m = mmsynth::makeMergeScene(960, 540, false);
  const Move mv{2.3, -1.7, 1.004};
  std::vector<float> inv(m.rgbDisplay);
  for (float& v : inv) v = 1.0f - v;
  const MergeInfo in = runMerge(m, m.base, movedDetail(m, mv), inv.data(), defaults(m.h));
  const double err = regError(m, in.reg, mv);
  printAlign("inverted picture (dark strands)", in, err);
  CHECK(in.align.model == AlignModel::PictureFull && err < 0.25);
}

// Fallbacks. (a) The detail's strands have no counterpart in the picture (the picture of another
// seed, busy: chance matches only): no picture model; the hair base is ill-conditioned for scale
// (sensitivity ~9), so no alignment at all. (b) Thin detail only in one small window:
// scale is unreliable (short lever arm) -> picture translation only, still accurate for a pure move.
// (c) A picture with no structure and a well-conditioned shape: the mattes' full model.
// Mutations: acceptable() -> always true ((a) reports a picture model); kMaxSensitivity 4 -> 1e9
// ((a) reports MatteFull); picreg's np = 3 regardless of translationOnly ((b) is not accepted as
// translation only); merge_register's fixScale ignored ((a)'s scale is not 1).
void testFallbacks() {
  {
    const MergeScene m = mmsynth::makeMergeScene(960, 540, true, 1);
    const MergeScene other = mmsynth::makeMergeScene(960, 540, true, 5);
    const Move mv{2.3, -1.7, 1.0};
    const MergeInfo in = runMerge(m, m.base, movedDetail(m, mv), other.rgbDisplay.data(), defaults(m.h));
    printAlign("(a) picture of another scene (busy)", in, regError(m, in.reg, mv));
    CHECK(in.align.model == AlignModel::None);  // no alignment beats a wrong move (user, 2026-10-24)
    CHECK(in.reg.scale == 1.0 && in.reg.tx == 0.0 && in.reg.ty == 0.0);
    CHECK(in.align.matte.sensitivity > 4.0);
  }
  {
    const MergeScene m = mmsynth::makeMergeScene(960, 540, false);
    const Move mv{1.7, -2.4, 1.0};
    // Strands kept only within a 90 px window at the top of the head.
    auto keep = [&](int x, int y) { return std::abs(x - m.w / 2) < 45 && std::abs(y - int(0.42 * m.h)) < 45; };
    const MergeInfo in = runMerge(m, m.base, movedDetail(m, mv, keep), m.rgbDisplay.data(), defaults(m.h));
    const double err = regError(m, in.reg, mv);
    printAlign("(b) thin detail in one small window", in, err);
    CHECK(in.align.model == AlignModel::PictureTranslation);
    CHECK(in.reg.scale == 1.0);
    CHECK(err < 0.5);
  }
  {
    // Three boxes and a disc (test_merge_core's registration shape), a flat grey picture.
    const int w = 640, h = 400;
    auto clamp01 = [](double v) { return std::min(1.0, std::max(0.0, v)); };
    auto box = [&](double x, double y, double x0, double y0, double x1, double y1) {
      return float(clamp01(0.5 + std::min(std::min(x - x0, x1 - x), std::min(y - y0, y1 - y)) / 2.0));
    };
    auto shape = [&](double x, double y, double g) {
      float v = box(x, y, 80 - g, 60 - g, 260 + g, 300 + g);
      v = std::max(v, box(x, y, 380 - g, 90 - g, 560 + g, 170 + g));
      v = std::max(v, box(x, y, 330 - g, 250 - g, 600 + g, 340 + g));
      return std::max(v, float(clamp01(0.5 + (40 + g - std::hypot(x - 470, y - 210)) / 2.0)));
    };
    const Move mv{-3.2, 2.1, 1.01};
    const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
    std::vector<float> base(size_t(w) * h), det(base.size()), pic(base.size() * 3, 0.4f), out(base.size());
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        base[size_t(y) * w + x] = shape(x, y, 0);
        det[size_t(y) * w + x] = shape(cx + (x - cx - mv.tx) / mv.s, cy + (y - cy - mv.ty) / mv.s, 1.5);
      }
    MergeParams p = defaults(h);
    p.reachPx = 20;
    MergeInfo in;
    merge(base.data(), det.data(), pic.data(), w, h, p, out.data(), MergeViews(), &in);
    double worst = 0;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (mr::isContour(base.data(), w, h, x, y)) {
          double qx, qy;
          in.reg.apply(x, y, &qx, &qy);
          worst = std::max(worst, std::hypot(qx - (cx + mv.s * (x - cx) + mv.tx), qy - (cy + mv.s * (y - cy) + mv.ty)));
        }
    printAlign("(c) boxes, flat picture", in, worst);
    CHECK(in.align.model == AlignModel::MatteFull);
    CHECK(in.align.blocks == 0);
    CHECK(worst < 0.3);
  }
  {
    // Align Off reports Off and applies nothing.
    const MergeScene m = mmsynth::makeMergeScene(960, 540, false);
    MergeParams p = defaults(m.h);
    p.align = false;
    const MergeInfo in = runMerge(m, m.base, m.detail, m.rgbDisplay.data(), p);
    CHECK(in.align.model == AlignModel::Off && in.reg.tx == 0 && in.reg.scale == 1);
  }
}

// Sizes scale with the frame height: the same scene at 3840x2160 and 1920x1080 with the default
// HeightSizes gives the same result: the added alpha per frame area and the strand recovery agree
// within a few %, and the half-size output matches the full-size one box-reduced 2x2.
// Below 1080 lines the results fall off (measured: added alpha per area 0.0047 at 1080-2160, 0.0038
// at 810, 0.0022 at 540): Detail Width's opening is a whole number of px (2r + 1 >= width: 3 px
// for 2.7 at 540) and strands of 0.5-1.5 px are blurred into their neighbours; not a scaling error.
// Mutation: scaleToHeight with h -> 1080 (the 1080 px values at every size) fails all three.
void testScaleWithHeight() {
  struct R {
    double addedPerArea = 0, recovery = 0;
    std::vector<float> out;
  };
  auto run = [&](int w, int h) {
    const MergeScene m = mmsynth::makeMergeScene(w, h, false);
    R r;
    MergeInfo in = runMerge(m, m.base, m.detail, m.rgbDisplay.data(), defaults(h), &r.out);
    double add = 0, eOut = 0, eBase = 0;
    for (size_t i = 0; i < r.out.size(); ++i) {
      add += r.out[i] - m.base[i];
      if (m.strands[i] > 0.05f && m.body[i] < 0.5f) {
        eOut += std::fabs(r.out[i] - m.truth[i]);
        eBase += std::fabs(m.base[i] - m.truth[i]);
      }
    }
    r.addedPerArea = add / (double(w) * h);
    r.recovery = 1.0 - eOut / eBase;
    std::printf("  %dx%d: Reach %.1f px, Detail Width %.2f px, Feather %.2f px -> added alpha / area %.5f, strand recovery "
                "%.3f, model %s\n", w, h, defaults(h).reachPx, defaults(h).detailWidthPx, defaults(h).featherPx,
                r.addedPerArea, r.recovery, alignModelName(in.align.model));
    return r;
  };
  std::printf("scale with height (defaults Reach %.2f %%, Detail Width %.2f %%, Feather %.2f %%):\n", HeightSizes().reachPct,
              HeightSizes().detailWidthPct, HeightSizes().featherPct);
  const int W = 1920, H = 1080;
  const R full = run(2 * W, 2 * H), half = run(W, H);
  // The full-size output box-reduced to W x H against the half-size output.
  double diff = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t a = size_t(2 * y) * (2 * W) + 2 * x;
      const double red = (full.out[a] + full.out[a + 1] + full.out[a + 2 * W] + full.out[a + 2 * W + 1]) / 4;
      diff += std::fabs(red - half.out[size_t(y) * W + x]);
    }
  diff /= double(W) * H;
  const double ratio = half.addedPerArea / full.addedPerArea;
  std::printf("  added ratio half / full %.3f, recovery %.3f vs %.3f, mean |reduced full - half| %.5f\n", ratio,
              half.recovery, full.recovery, diff);
  CHECK(std::fabs(ratio - 1) < 0.03);                     // 0.994
  CHECK(std::fabs(half.recovery - full.recovery) < 0.02);  // 0.302 vs 0.297
  CHECK(diff < 0.003);                                     // 0.0014
  MergeParams p;
  scaleToHeight(p, HeightSizes{10, 1, 0.5}, 2000);
  CHECK(p.reachPx == 200 && p.detailWidthPx == 20 && p.featherPx == 10);
}

// A MergeWorkspace reused across calls of different sizes, sources and options gives the same
// output and views, bit for bit, as fresh calls. Mutation: the feather ramp left unwritten outside
// the zone (stale values from the previous call) -> differences.
void testWorkspace() {
  const MergeScene a = mmsynth::makeMergeScene(960, 540, false), b = mmsynth::makeMergeScene(640, 360, true);
  MergeWorkspace ws;
  int bad = 0;
  for (int pass = 0; pass < 2; ++pass)
    for (const MergeScene* m : {&a, &b, &a})
      for (int mode = 0; mode < 3; ++mode) {
        MergeParams p = defaults(m->h);
        p.evidence = mode == 1;
        if (mode == 2) p.source = Source::Trimap;
        const std::vector<float>& base = mode == 2 ? m->trimap : m->base;
        const size_t n = base.size();
        std::vector<float> o1(n), o2(n), v1(4 * n), v2(4 * n);
        const MergeViews mv1{&v1[0], &v1[n], &v1[2 * n], &v1[3 * n]}, mv2{&v2[0], &v2[n], &v2[2 * n], &v2[3 * n]};
        merge(base.data(), m->detail.data(), m->rgbDisplay.data(), m->w, m->h, p, o1.data(), mv1, nullptr, &ws);
        merge(base.data(), m->detail.data(), m->rgbDisplay.data(), m->w, m->h, p, o2.data(), mv2);
        bad += o1 != o2 || v1 != v2;
      }
  std::printf("workspace reuse: %d of 18 calls differ from a fresh call\n", bad);
  CHECK(bad == 0);
}

}  // namespace

int main() {
  testKnownMoves();
  testFallbacks();
  testScaleWithHeight();
  testWorkspace();
  std::printf(failures ? "FAILED (%d failure(s))\n" : "OK (0 failure(s))\n", failures);
  return failures ? 1 : 0;
}
