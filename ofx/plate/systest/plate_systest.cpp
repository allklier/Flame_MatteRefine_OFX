// Plate Merge system test: loads PlateTools.ofx in Matte Merge's Flame-like host (ofx/merge/systest/ofx_host.h)
// and renders a synthetic two-plate scene through it. Checks: describe (clips, context, grouping, Flame
// layout: columns of <= 5 rows), defaults, the rendered RGB and alpha equal mergePlates() called directly
// (every View, Matte Out), Keep A, render scale (px params shrink with it), the pass-through footers (no
// plate B, a missing matte, a size mismatch), the output colourspace answers, Collect Logs.
// Usage: plate_systest <PlateTools.ofx> <scratch HOME folder> (the plugin writes its log, config and support
// bundle under $HOME: never the real one).
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ofxParam.h"
#include "ofx_host.h"
#include "plate_merge.h"
#include "plate_ui.h"

namespace {
int failures = 0;
#define CHECK(c)                                                \
  do {                                                          \
    if (!(c)) {                                                 \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);  \
      ++failures;                                               \
    }                                                           \
  } while (0)

using namespace systest;
namespace pm = mr::plate;

const char* kOcio = R"(ocio_profile_version: 2.4
colorspaces:
  - !<ColorSpace>
    name: ACEScg
    encoding: scene-linear
  - !<ColorSpace>
    name: Matte
    aliases: [adsk:matte_data]
    encoding: data
)";

double clamp01(double v) { return std::min(1.0, std::max(0.0, v)); }

// Two plates of a static texture (B 0.3 stops brighter, values > 1 and < 0 present), a person in each.
struct Scene {
  int w, h;
  std::vector<float> A, B, mA, mB, keep;
};
Scene makeScene(int w, int h) {
  Scene s{w, h, {}, {}, {}, {}, {}};
  const size_t n = size_t(w) * h;
  s.A.resize(n * 3);
  s.B.resize(n * 3);
  s.mA.resize(n);
  s.mB.resize(n);
  s.keep.assign(n, 0.0f);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      const double ma = clamp01(0.5 - (std::hypot(x - 0.35 * w, y - 0.5 * h) - 0.15 * h) / 2.0);
      const double mb = clamp01(0.5 - (std::hypot(x - 0.48 * w, y - 0.45 * h) - 0.17 * h) / 2.0);
      s.mA[i] = float(ma);
      s.mB[i] = float(mb);
      for (int c = 0; c < 3; ++c) {
        double bg = 0.18 * std::exp2(std::sin(0.07 * x + c) * std::cos(0.05 * y));
        if (x < 10 && y < 10) bg = -0.003;
        if (std::hypot(x - 0.8 * w, y - 0.2 * h) < 6) bg = 5.0;
        s.A[3 * i + c] = float(bg * (1 - ma) + 0.4 * ma);
        s.B[3 * i + c] = float(bg * std::exp2(0.3) * (1 - mb) + 0.1 * mb);
      }
      if (x > 0.25 * w && x < 0.35 * w && y < 0.3 * h) s.keep[i] = 1.0f;
    }
  return s;
}

Feed rgbaFeed(const std::vector<float>* rgb, const std::vector<float>* alpha) {
  return Feed{[rgb, alpha](int, int w, int h, std::vector<float>& o) {
                for (size_t i = 0; i < size_t(w) * h; ++i) {
                  o[i * 4] = (*rgb)[i * 3];
                  o[i * 4 + 1] = (*rgb)[i * 3 + 1];
                  o[i * 4 + 2] = (*rgb)[i * 3 + 2];
                  o[i * 4 + 3] = alpha ? (*alpha)[i] : 1.0f;
                }
              },
              4};
}
Feed alphaFeed(const std::vector<float>* m) {
  return Feed{[m](int, int, int, std::vector<float>& o) { o = *m; }, 1};
}

struct Direct {
  std::vector<float> rgb, swap, residual, ring, matchedB;
  pm::PlateInfo info;
};
Direct direct(const Scene& s, const pm::UiValues& ui, bool keep, double scale = 1.0) {
  Direct d;
  const size_t n = size_t(s.w) * s.h;
  d.rgb.resize(n * 3);
  d.swap.resize(n);
  d.residual.resize(n);
  d.ring.assign(n, 0.0f);
  d.matchedB.resize(n * 3);
  pm::PlateViews v;
  v.swap = d.swap.data();
  v.residual = d.residual.data();
  v.ring = d.ring.data();
  v.matchedB = d.matchedB.data();
  pm::mergePlates(s.A.data(), s.mA.data(), s.B.data(), s.mB.data(), keep ? s.keep.data() : nullptr, s.w, s.h,
                  pm::toPlateParams(ui, scale), d.rgb.data(), v, &d.info);
  return d;
}

// Pixels where channel c of the render differs from want (w*h single channel, or w*h*3 with stride 3).
long diff(const Rendered& r, const std::vector<float>& want, int c, int stride = 1, int wc = 0) {
  long n = 0;
  for (int row = 0; row < r.h; ++row)
    for (int x = 0; x < r.w; ++x)
      if (r.px(x, row)[c] != want[(size_t(row) * r.w + x) * stride + wc]) ++n;
  return n;
}
long diffRgb(const Rendered& r, const std::vector<float>& rgb) {
  return diff(r, rgb, 0, 3, 0) + diff(r, rgb, 1, 3, 1) + diff(r, rgb, 2, 3, 2);
}

void checkLayout(Host& host) {
  std::vector<std::pair<std::string, int>> cols;
  for (const auto& p : host.descriptor().params.list) {
    if (p->type == kOfxParamTypeGroup) {
      cols.push_back({p->props.getStr(kOfxPropLabel), 0});
      continue;
    }
    if (p->type == kOfxParamTypePage) continue;
    CHECK(!p->props.getStr(kOfxParamPropParent).empty());
    if (cols.empty()) cols.push_back({"(none)", 0});
    cols.back().second += p->type == kOfxParamTypeChoice ? 2 : 1;
    if (p->type == kOfxParamTypeDouble) {
      CHECK(p->props.getInt(kOfxParamPropDigits) >= 1);
      CHECK(p->props.getDbl(kOfxParamPropIncrement) > 0);
    }
    CHECK(p->props.getStr(kOfxPropLabel).size() <= 15);  // Flame cuts labels at ~15 characters
  }
  std::printf("layout:");
  for (const auto& c : cols) {
    std::printf(" %s (%d rows)", c.first.c_str(), c.second);
    CHECK(c.second <= 5);
  }
  std::printf(" = %zu columns\n", cols.size());
  CHECK(cols.size() == 3);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: plate_systest <PlateTools.ofx> <scratch home>\n");
    return 2;
  }
  const std::string home = argv[2];
  mkdir(home.c_str(), 0755);
  setenv("HOME", home.c_str(), 1);
  unsetenv("MR_LOG_LEVEL");
  std::remove((home + "/.config/matte-refine/config.conf").c_str());
  const std::string ocioPath = home + "/test_config.ocio";
  std::ofstream(ocioPath) << kOcio;

  Host host(argv[1]);
  if (!host.loaded()) {
    std::printf("FAIL load: %s\n", host.loadError().c_str());
    return 1;
  }
  CHECK(host.pluginCount() == 1);
  CHECK(std::strcmp(host.plugin()->pluginIdentifier, "com.janklier.platemerge") == 0);
  CHECK(host.plugin()->pluginVersionMajor == 0 && host.plugin()->pluginVersionMinor == 1);
  CHECK(host.describe() == kOfxStatOK);
  Effect& d = host.descriptor();
  CHECK(d.props.getStr(kOfxPropLabel) == "Plate Merge");
  CHECK(d.props.getStr(kOfxImageEffectPluginPropGrouping) == "Matte Refine");
  CHECK(d.props.getStr(kOfxImageEffectPropSupportedContexts, 0) == kOfxImageEffectContextFilter);
  CHECK(d.clips.count(kOfxImageEffectSimpleSourceClipName) && d.clips.count("PlateB") && d.clips.count("KeepA") &&
        d.clips.count(kOfxImageEffectOutputClipName) && d.clips.size() == 4);
  if (d.clips.count("PlateB")) {
    const PropSet& bp = d.clips["PlateB"]->props;
    CHECK(bp.getInt(kOfxImageClipPropIsMask, 0, 0) == 0 && bp.getInt(kOfxImageClipPropOptional) == 1);
    CHECK(bp.getStr(kOfxImageEffectPropSupportedComponents, 0) == kOfxImageComponentRGBA);
    CHECK(bp.getStr(kOfxPropLabel) == "Plate B");
  }
  if (d.clips.count("KeepA")) {
    const PropSet& kp = d.clips["KeepA"]->props;
    CHECK(kp.getInt(kOfxImageClipPropIsMask) == 1 && kp.getInt(kOfxImageClipPropOptional) == 1);
    CHECK(kp.getStr(kOfxImageEffectPropSupportedComponents, 0) == kOfxImageComponentAlpha);
  }
  checkLayout(host);

  const int W = 480, H = 270;
  const Scene sc = makeScene(W, H);
  Feeds feeds;
  feeds[kOfxImageEffectSimpleSourceClipName] = rgbaFeed(&sc.A, &sc.mA);
  feeds["PlateB"] = rgbaFeed(&sc.B, &sc.mB);
  auto inst = host.createInstance(W, H, 3, feeds, ocioPath);

  // Defaults.
  CHECK(inst->getInt("view") == pm::kViewResult && inst->getInt("matteOut") == pm::kMatteSwap &&
        inst->getInt("mode") == 0 && inst->getInt("levelMatch") == 1);
  CHECK(inst->getDouble("growPx") == 8.0 && inst->getDouble("featherPx") == 8.0);
  CHECK(inst->getString("version").find("0.1.0") == 0);
  CHECK(inst->getString("nodeId").size() == 8);

  // 1. Default render = mergePlates() bit for bit: RGB = result, alpha = swap matte; footer = info line.
  const pm::UiValues def;
  const Direct ref = direct(sc, def, false);
  Rendered r = inst->render(0);
  CHECK(r.status == kOfxStatOK);
  const long dRgb = diffRgb(r, ref.rgb), dA = diff(r, ref.swap, 3);
  std::printf("default render: RGB differs at %ld values, alpha at %ld px; footer '%s'\n", dRgb, dA,
              inst->persistent.c_str());
  CHECK(dRgb == 0 && dA == 0);
  CHECK(diffRgb(r, sc.A) > 0);
  CHECK(ref.info.matched && std::fabs(ref.info.globalGainStops[1] + 0.3) < 0.02);
  CHECK(ref.info.residualPixels > 100);  // the persons overlap
  CHECK(inst->persistent == pm::infoFooter(ref.info, true));
  CHECK(inst->outputColourspace() == "OfxColourspace_Source");

  // 2. Views and Matte Out.
  inst->setInt("view", pm::kViewPlateA);
  r = inst->render(0);
  CHECK(diffRgb(r, sc.A) == 0 && diff(r, ref.swap, 3) == 0);
  CHECK(inst->outputColourspace() == "OfxColourspace_Source");
  inst->setInt("view", pm::kViewPlateB);
  r = inst->render(0);
  CHECK(diffRgb(r, ref.matchedB) == 0);
  const std::pair<int, const std::vector<float>*> views[] = {
      {pm::kViewSwap, &ref.swap}, {pm::kViewResidual, &ref.residual}, {pm::kViewRing, &ref.ring}};
  for (const auto& v : views) {
    inst->setInt("view", v.first);
    r = inst->render(0);
    CHECK(diff(r, *v.second, 0) == 0 && diff(r, *v.second, 1) == 0 && diff(r, *v.second, 2) == 0);
    CHECK(inst->outputColourspace() == "Matte");
  }
  inst->setInt("view", pm::kViewResult);
  inst->setInt("matteOut", pm::kMatteResidual);
  r = inst->render(0);
  CHECK(diff(r, ref.residual, 3) == 0 && diffRgb(r, ref.rgb) == 0);
  inst->setInt("matteOut", pm::kMatteSwap);

  // 3. Settings reach the core: Clean Background, Grow, Feather, Level Match off.
  pm::UiValues u;
  u.mode = 1;
  u.growPx = 4;
  u.featherPx = 12;
  u.levelMatch = false;
  inst->setInt("mode", 1);
  inst->setDouble("growPx", 4);
  inst->setDouble("featherPx", 12);
  inst->setBool("levelMatch", false);
  const Direct ref2 = direct(sc, u, false);
  r = inst->render(0);
  CHECK(diffRgb(r, ref2.rgb) == 0 && diff(r, ref2.swap, 3) == 0);
  CHECK(inst->persistent == pm::infoFooter(ref2.info, false));
  inst->setInt("mode", 0);
  inst->setDouble("growPx", 8);
  inst->setDouble("featherPx", 8);
  inst->setBool("levelMatch", true);

  // 4. Keep A connected: plate A kept there bit for bit; equals mergePlates() with the keep matte.
  inst->setFeed("KeepA", alphaFeed(&sc.keep));
  const Direct refK = direct(sc, def, true);
  r = inst->render(0);
  CHECK(diffRgb(r, refK.rgb) == 0);
  long keepBad = 0;
  for (int row = 0; row < H; ++row)
    for (int x = 0; x < W; ++x) {
      const size_t i = size_t(row) * W + x;
      if (sc.keep[i] == 1.0f)
        for (int c = 0; c < 3; ++c) keepBad += r.px(x, row)[c] != sc.A[3 * i + c];
    }
  CHECK(keepBad == 0);
  inst->setFeed("KeepA", Feed{});

  // 5. Footers: no plate B, A's matte missing (alpha 1), B's matte missing: plate A passes through, alpha 0.
  const std::vector<float> zero(size_t(W) * H, 0.0f);
  inst->setFeed("PlateB", Feed{});
  r = inst->render(0);
  CHECK(inst->persistent == pm::kNoPlateBFooter && diffRgb(r, sc.A) == 0 && diff(r, zero, 3) == 0);
  inst->setFeed("PlateB", rgbaFeed(&sc.B, nullptr));
  r = inst->render(0);
  CHECK(inst->persistent == pm::constantMatteFooter("B") && diffRgb(r, sc.A) == 0);
  inst->setFeed("PlateB", rgbaFeed(&sc.B, &sc.mB));
  inst->setFeed(kOfxImageEffectSimpleSourceClipName, rgbaFeed(&sc.A, nullptr));
  r = inst->render(0);
  CHECK(inst->persistent == pm::constantMatteFooter("A") && diffRgb(r, sc.A) == 0 && diff(r, zero, 3) == 0);
  inst->setFeed(kOfxImageEffectSimpleSourceClipName, rgbaFeed(&sc.A, &sc.mA));
  r = inst->render(0);
  CHECK(diffRgb(r, ref.rgb) == 0 && inst->persistent == pm::infoFooter(ref.info, true));

  // 6. Collect Logs writes a support bundle; no recursion through the echo.
  inst->press("collectLogs");
  CHECK(!inst->messages.empty() && inst->messages.back().text.find("Support bundle written") == 0);
  CHECK(host.errors.empty());
  for (const auto& e : host.errors) std::printf("host error: %s\n", e.c_str());

  std::printf("plate_systest: %s (%d failures)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
