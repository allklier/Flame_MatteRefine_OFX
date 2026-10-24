// Matte Merge system test: loads MatteMerge.ofx in the Flame-like host (ofx_host.h) and renders the
// synthetic merge scenes (tools/merge_synth.h) through it. Checks: describe (clips, contexts, Flame
// layout: columns of <= 5 rows, <= 6 columns), the rendered alpha equals calling merge() directly
// (every View's buffer too), the pass-through footers (no Detail, constant base), the output
// colourspace answers, the changedParam echo guard, Collect Logs and Debug Logging.
// Usage: merge_systest <MatteMerge.ofx> <scratch HOME folder> (the plugin writes its log, config
// and support bundle under $HOME: never the real one).
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "matte_merge.h"
#include "merge_synth.h"
#include "merge_ui.h"
#include "ofxParam.h"
#include "ofx_host.h"

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
namespace mm = mr::merge;

const char* kOcio = R"(ocio_profile_version: 2.4
colorspaces:
  - !<ColorSpace>
    name: ACEScg
    encoding: scene-linear
  - !<ColorSpace>
    name: Matte
    aliases: [adsk:matte_data]
    encoding: data
  - !<ColorSpace>
    name: Raw
    encoding: data
)";

// Pixels where the output's channel c differs from want (w*h, top-down); also counts unwritten (-1).
long diffChannel(const Rendered& r, const std::vector<float>& want, int c, double* maxErr = nullptr) {
  long n = 0;
  double m = 0;
  for (int row = 0; row < r.h; ++row)
    for (int x = 0; x < r.w; ++x) {
      const float v = r.px(x, row)[c], t = want[size_t(row) * r.w + x];
      if (v != t) {
        ++n;
        m = std::max(m, double(std::fabs(v - t)));
      }
    }
  if (maxErr) *maxErr = m;
  return n;
}

Feed rgbaFeed(const std::vector<float>* rgb, const std::vector<float>* alpha) {
  return Feed{[rgb, alpha](int, int w, int h, std::vector<float>& o) {
                for (size_t i = 0; i < size_t(w) * h; ++i) {
                  o[i * 4] = (*rgb)[i * 3];
                  o[i * 4 + 1] = (*rgb)[i * 3 + 1];
                  o[i * 4 + 2] = (*rgb)[i * 3 + 2];
                  o[i * 4 + 3] = (*alpha)[i];
                }
              },
              4};
}
Feed alphaFeed(const std::vector<float>* m) {
  return Feed{[m](int, int, int, std::vector<float>& o) { o = *m; }, 1};
}

// merge() as the plugin should call it, with the views.
struct Direct {
  std::vector<float> out, added, rejected, zone, residual;
  mm::MergeInfo info;
};
Direct direct(const std::vector<float>& base, const std::vector<float>& detail, const std::vector<float>& rgb, int w,
              int h, const mm::UiValues& ui) {
  Direct d;
  const size_t n = size_t(w) * h;
  d.out.resize(n);
  d.added.resize(n);
  d.rejected.resize(n);
  d.zone.resize(n);
  d.residual.resize(n);
  mm::MergeViews v;
  v.added = d.added.data();
  v.rejected = d.rejected.data();
  v.zone = d.zone.data();
  v.residual = d.residual.data();
  mm::merge(base.data(), detail.data(), rgb.data(), w, h, mm::toMergeParams(ui, h), d.out.data(), v, &d.info);
  return d;
}

std::string fileText(const std::string& p) {
  std::ifstream in(p);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void checkLayout(Host& host) {
  // Flame (ofx/CLAUDE.md "Flame host behaviour"): each group starts a column; rows: choice 2, other 1,
  // a secret param takes a slot too; <= 5 rows per column, <= 6 columns per tab.
  std::vector<std::pair<std::string, int>> cols;
  for (const auto& p : host.descriptor().params.list) {
    if (p->type == kOfxParamTypeGroup) {
      cols.push_back({p->props.getStr(kOfxPropLabel), 0});
      continue;
    }
    if (p->type == kOfxParamTypePage) continue;
    CHECK(!p->props.getStr(kOfxParamPropParent).empty());  // every param inside a group
    if (cols.empty()) cols.push_back({"(none)", 0});
    cols.back().second += p->type == kOfxParamTypeChoice ? 2 : 1;
    if (p->type == kOfxParamTypeDouble) {  // Flame needs digits / increment (ofx/CLAUDE.md)
      CHECK(p->props.getInt(kOfxParamPropDigits) >= 2);
      CHECK(p->props.getDbl(kOfxParamPropIncrement) > 0);
    }
    CHECK(p->props.getStr(kOfxPropLabel).size() <= 30);
  }
  std::printf("layout:");
  for (const auto& c : cols) {
    std::printf(" %s (%d rows)", c.first.c_str(), c.second);
    CHECK(c.second <= 5);
  }
  std::printf(" = %zu columns\n", cols.size());
  CHECK(cols.size() <= 6);
  CHECK(cols.size() == 5);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: merge_systest <MatteMerge.ofx> <scratch home>\n");
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
  CHECK(std::strcmp(host.plugin()->pluginIdentifier, "com.janklier.mattemerge") == 0);
  CHECK(host.plugin()->pluginVersionMajor == 0 && host.plugin()->pluginVersionMinor == 1);
  CHECK(host.describe() == kOfxStatOK);
  Effect& d = host.descriptor();
  CHECK(d.props.getStr(kOfxPropLabel) == "Matte Merge");
  CHECK(d.props.getStr(kOfxImageEffectPluginPropGrouping) == "Matte Refine");
  CHECK(d.props.getStr(kOfxImageEffectPropSupportedContexts, 0) == kOfxImageEffectContextFilter);
  CHECK(d.clips.count(kOfxImageEffectSimpleSourceClipName) && d.clips.count("Detail") &&
        d.clips.count(kOfxImageEffectOutputClipName) && d.clips.size() == 3);
  if (d.clips.count("Detail")) {
    const PropSet& dp = d.clips["Detail"]->props;
    CHECK(dp.getInt(kOfxImageClipPropIsMask) == 1 && dp.getInt(kOfxImageClipPropOptional) == 1);
    CHECK(dp.getStr(kOfxImageEffectPropSupportedComponents, 0) == kOfxImageComponentAlpha);
  }
  CHECK(d.clips[kOfxImageEffectSimpleSourceClipName]->props.getInt(kOfxImageClipPropOptional, 0, 0) == 0);
  checkLayout(host);

  const int W = 960, H = 540;
  const mmsynth::MergeScene sc = mmsynth::makeMergeScene(W, H, true);
  Feeds feeds;
  feeds[kOfxImageEffectSimpleSourceClipName] = rgbaFeed(&sc.rgbDisplay, &sc.base);
  feeds["Detail"] = alphaFeed(&sc.detail);
  auto inst = host.createInstance(W, H, 3, feeds, ocioPath);

  // Defaults (decision 19 / the user's: Reach 6 %, Detail Width 0.5 %, Strength 1, Feather 0.25 %).
  CHECK(inst->getInt("view") == mm::kViewResult && inst->getInt("zoneSource") == 0 && inst->getInt("align") == 0 &&
        inst->getInt("method") == 0 && inst->getInt("evidence") == 0 && inst->getInt("zoneKeepDisconnected") == 0);
  CHECK(inst->getDouble("zoneReach") == 6.0 && inst->getDouble("detailWidth") == 0.5 &&
        inst->getDouble("strength") == 1.0 && inst->getDouble("feather") == 0.25 &&
        inst->getDouble("zoneBridge") == 0.5);
  CHECK(inst->getString("version").find("0.1.0") == 0);
  CHECK(inst->getString("nodeId").size() == 8);

  // 1. Default render = merge() called directly, bit for bit; RGB = the matte; footer = the info line.
  const mm::UiValues def;
  const Direct ref = direct(sc.base, sc.detail, sc.rgbDisplay, W, H, def);
  Rendered r = inst->render(0);
  CHECK(r.status == kOfxStatOK);
  double maxErr = 0;
  const long dA = diffChannel(r, ref.out, 3, &maxErr);
  std::printf("default render: alpha differs from merge() at %ld px (max %.3g); align '%s', kept %ld dropped %ld; "
              "footer '%s'\n",
              dA, maxErr, mm::alignModelName(ref.info.align.model), ref.info.keptPixels, ref.info.droppedPixels,
              inst->persistent.c_str());
  CHECK(dA == 0);
  CHECK(diffChannel(r, ref.out, 0) == 0 && diffChannel(r, ref.out, 1) == 0 && diffChannel(r, ref.out, 2) == 0);
  CHECK(diffChannel(r, sc.base, 3) > 0);  // the merge did something
  CHECK(inst->persistent == mm::infoFooter(ref.info));
  CHECK(inst->outputColourspace() == "Matte");
  // A second render (the workspace reused) is the same.
  Rendered r2 = inst->render(1);
  CHECK(diffChannel(r2, ref.out, 3) == 0);

  // 2. Views: each one's RGB = merge()'s buffer; alpha always the result.
  inst->setInt("view", mm::kViewPicture);
  r = inst->render(0);
  {
    std::vector<float> ch[3];
    for (int c = 0; c < 3; ++c) {
      ch[c].resize(size_t(W) * H);
      for (size_t i = 0; i < ch[c].size(); ++i) ch[c][i] = sc.rgbDisplay[i * 3 + c];
    }
    CHECK(diffChannel(r, ch[0], 0) == 0 && diffChannel(r, ch[1], 1) == 0 && diffChannel(r, ch[2], 2) == 0);
    CHECK(diffChannel(r, ref.out, 3) == 0);
    CHECK(inst->outputColourspace() == "OfxColourspace_Source");
  }
  const std::pair<int, const std::vector<float>*> views[] = {
      {mm::kViewRejected, &ref.rejected}, {mm::kViewZone, &ref.zone}, {mm::kViewResidual, &ref.residual}};
  for (const auto& v : views) {
    inst->setInt("view", v.first);
    r = inst->render(0);
    const long n = diffChannel(r, *v.second, 0) + diffChannel(r, *v.second, 2);
    double sum = 0;
    for (float x : *v.second) sum += x;
    std::printf("view %s: %ld px differ (buffer sum %.1f)\n", mm::kViewNames[v.first], n, sum);
    CHECK(n == 0 && diffChannel(r, ref.out, 3) == 0);
    CHECK(inst->outputColourspace() == "Matte");
  }
  inst->setInt("view", mm::kViewAdded);
  r = inst->render(0);
  {
    std::vector<float> pos(ref.added.size()), mag(ref.added.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = std::max(ref.added[i], 0.0f), mag[i] = std::fabs(ref.added[i]);
    CHECK(diffChannel(r, pos, 0) == 0 && diffChannel(r, mag, 1) == 0 && diffChannel(r, pos, 2) == 0);
  }
  inst->setInt("view", mm::kViewResult);

  // 3. Other settings reach the core: Zone Blend, Align Off, Evidence On, Strength, Reach, Feather.
  mm::UiValues u;
  u.method = 1;
  u.align = 1;
  u.evidence = 1;
  u.strength = 0.6;
  u.reachPct = 3.0;
  u.detailWidthPct = 0.8;
  u.featherPct = 0.5;
  u.bridgePct = 1.2;
  inst->setDouble("zoneBridge", 1.2);
  inst->setInt("method", 1);
  inst->setInt("align", 1);
  inst->setInt("evidence", 1);
  inst->setDouble("strength", 0.6);
  inst->setDouble("zoneReach", 3.0);
  inst->setDouble("detailWidth", 0.8);
  inst->setDouble("feather", 0.5);
  const Direct ref2 = direct(sc.base, sc.detail, sc.rgbDisplay, W, H, u);
  r = inst->render(0);
  CHECK(diffChannel(r, ref2.out, 3) == 0);
  CHECK(diffChannel(r, ref.out, 3) > 0);
  CHECK(inst->persistent == mm::infoFooter(ref2.info) && inst->persistent.find("Align: off.") == 0);

  // 3b. Bridge (Add Fine Detail): 0 (off) and 1.2 % both equal merge(); the bridge changes the result.
  {
    inst->setInt("method", 0);
    u.method = 0;
    long changed = 0, diff0 = 0, diff1 = 0;
    std::vector<float> outs[2];
    const double pcts[2] = {0.0, 1.2};
    for (int k = 0; k < 2; ++k) {
      inst->setDouble("zoneBridge", pcts[k]);
      u.bridgePct = pcts[k];
      const Direct rb = direct(sc.base, sc.detail, sc.rgbDisplay, W, H, u);
      r = inst->render(0);
      (k ? diff1 : diff0) = diffChannel(r, rb.out, 3);
      outs[k] = rb.out;
    }
    for (size_t i = 0; i < outs[0].size(); ++i) changed += outs[0][i] != outs[1][i];
    std::printf("bridge: alpha differs from merge() at %ld px (off) / %ld px (1.2 %%); the bridge changed %ld px\n",
                diff0, diff1, changed);
    CHECK(diff0 == 0 && diff1 == 0 && changed > 0);
    inst->setInt("method", 1);  // back to the settings of ref2 (used again below)
    u.method = 1;
  }

  // 4. Trimap: the trimap on the source alpha, Source = Trimap (+ Keep Disconnected).
  auto tri = host.createInstance(W, H, 1,
                                 Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed(&sc.rgbDisplay, &sc.trimap)},
                                       {"Detail", alphaFeed(&sc.detail)}},
                                 ocioPath);
  tri->setInt("zoneSource", 1);
  mm::UiValues ut;
  ut.source = 1;
  const Direct ref3 = direct(sc.trimap, sc.detail, sc.rgbDisplay, W, H, ut);
  r = tri->render(0);
  CHECK(diffChannel(r, ref3.out, 3) == 0);
  tri->setBool("zoneKeepDisconnected", true);
  ut.keepDisconnected = true;
  const Direct ref4 = direct(sc.trimap, sc.detail, sc.rgbDisplay, W, H, ut);
  r = tri->render(0);
  CHECK(diffChannel(r, ref4.out, 3) == 0);
  std::printf("trimap: fill differs from merge() at %ld px; keep disconnected changes %ld px\n",
              diffChannel(r, ref4.out, 3), [&] {
                long n = 0;
                for (size_t i = 0; i < ref3.out.size(); ++i) n += ref3.out[i] != ref4.out[i];
                return n;
              }());

  // 5. A Detail delivered as RGBA (a host that ignores the mask declaration): its alpha is used.
  auto rgbaDetail = host.createInstance(
      W, H, 1,
      Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed(&sc.rgbDisplay, &sc.base)},
            {"Detail", rgbaFeed(&sc.rgbDisplay, &sc.detail)}},
      ocioPath);
  r = rgbaDetail->render(0);
  CHECK(diffChannel(r, ref.out, 3) == 0);

  // 6. Detail disconnected: the base passes through, the footer asks for the detail matte.
  inst->setFeed("Detail", Feed{});
  r = inst->render(0);
  CHECK(diffChannel(r, sc.base, 3) == 0 && diffChannel(r, sc.base, 0) == 0);
  CHECK(inst->persistent == mm::kNoDetailFooter);
  std::printf("no detail: footer '%s'\n", inst->persistent.c_str());
  inst->setFeed("Detail", alphaFeed(&sc.detail));
  r = inst->render(0);
  CHECK(diffChannel(r, ref2.out, 3) == 0 && inst->persistent == mm::infoFooter(ref2.info));

  // 7. Nothing on the source alpha (Flame: constant 1): the footer asks for the base; alpha = 1.
  const std::vector<float> ones(size_t(W) * H, 1.0f);
  auto noBase = host.createInstance(W, H, 1,
                                    Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed(&sc.rgbDisplay, &ones)},
                                          {"Detail", alphaFeed(&sc.detail)}},
                                    ocioPath);
  r = noBase->render(0);
  CHECK(diffChannel(r, ones, 3) == 0);
  CHECK(noBase->persistent == mm::constantBaseFooter(1.0f));
  std::printf("no base: footer '%s'\n", noBase->persistent.c_str());

  // 8. Support: Debug Logging writes the per-user config and turns on the DEBUG render line;
  // Collect Logs writes a bundle into $HOME; no echo recursion; no host errors.
  inst->setBool("debugLog", true);
  r = inst->render(2);
  const std::string cfg = fileText(home + "/.config/matte-refine/config.conf");
  CHECK(cfg.find("log_level=debug") != std::string::npos);
  const std::string log = fileText(home + "/.cache/matte-refine/logs/matte-refine.log");
  CHECK(log.find("Matte Merge 0.1.0") != std::string::npos);
  CHECK(log.find("DEBUG") != std::string::npos && log.find("render t=2 960x540") != std::string::npos);
  CHECK(log.find("changedParam method (user)") != std::string::npos);
  CHECK(log.find("colour: output colourspace answer 'Matte'") != std::string::npos);
  inst->press("collectLogs");
  CHECK(!inst->messages.empty() && inst->messages.back().text.find("Support bundle written to " + home) == 0);
  std::printf("collect logs: '%s'\n", inst->messages.empty() ? "(none)" : inst->messages.back().text.c_str());
  inst->setBool("debugLog", false);
  std::printf("echoes %d, max echo depth %d, host errors %zu\n", inst->echoes, inst->maxEchoDepth, host.errors.size());
  for (const auto& e : host.errors) std::printf("host error: %s\n", e.c_str());
  CHECK(host.errors.empty());
  CHECK(inst->maxEchoDepth <= 2);

  if (failures) {
    std::printf("merge_systest: %d FAILED\n", failures);
    return 1;
  }
  std::printf("merge_systest: all passed\n");
  return 0;
}
