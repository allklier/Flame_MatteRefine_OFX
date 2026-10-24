// Matte Refine (plugin 1) system test: loads MatteRefine.ofx in the Flame-like host of Matte Merge
// (ofx/merge/systest/ofx_host.*, from LTX Alpha Gen) and checks the filter layout (decision 4): a
// mandatory Source (picture on the front, matte in its alpha), no "Matte" clip; the rendered alpha
// equals running the ops directly on a synthetic matte; Snap to Image and Edge Colour Unmix use
// Source's RGB; the footers (RGB-only Source, constant alpha, non-linear picture for Edge Colour
// Unmix, decision 24); the output colourspace answers (23); Save Matte and Analyze Frame read Source.
// Usage: refine_systest <MatteRefine.ofx> <scratch HOME folder> (never the real HOME).
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "ofxColour.h"
#include "ofx_host.h"
#include "ops/fill_holes.h"
#include "ops/foreground_colour.h"
#include "ops/snap_to_image.h"
#include "synth.h"

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

const char* kOcio = R"(ocio_profile_version: 2.4
colorspaces:
  - !<ColorSpace>
    name: ACEScg
    encoding: scene-linear
  - !<ColorSpace>
    name: ARRI LogC3 (EI800)
    encoding: log
  - !<ColorSpace>
    name: Matte
    aliases: [adsk:matte_data]
    encoding: data
)";

// Pixels where the output's channel c differs from want (w*h, top-down).
long diffChannel(const Rendered& r, const std::vector<float>& want, int c, int stride = 1, int off = 0) {
  long n = 0;
  for (int row = 0; row < r.h; ++row)
    for (int x = 0; x < r.w; ++x) n += r.px(x, row)[c] != want[(size_t(row) * r.w + x) * stride + off];
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
Feed rgbFeed(const std::vector<float>* rgb) {
  return Feed{[rgb](int, int, int, std::vector<float>& o) { o = *rgb; }, 3};
}

std::string fileText(const std::string& p) {
  std::ifstream in(p);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void setSourceColourspace(Instance& inst, const std::string& cs) {
  inst.effect().clips[kOfxImageEffectSimpleSourceClipName]->props.setStr(kOfxImageClipPropColourspace, cs);
}

// The plugin's default Fill Holes (Solid Above 0.95, Max Hole 1 %, Bridge 0, Rim 1), the only op on by default.
std::vector<float> defaultChain(const std::vector<float>& matte, int w, int h) {
  mr::FillHolesParams p;
  p.threshold = float(0.95);
  p.maxHolePercent = 1.0;
  p.bridgePx = 0.0;
  p.rimPx = 1.0;
  std::vector<float> out(matte);
  mr::fillHoles(out.data(), out.data(), w, h, p);
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: refine_systest <MatteRefine.ofx> <scratch home>\n");
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
  CHECK(std::strcmp(host.plugin()->pluginIdentifier, "com.janklier.matterefine") == 0);
  CHECK(host.describe() == kOfxStatOK);
  Effect& d = host.descriptor();
  CHECK(d.props.getStr(kOfxPropLabel) == "Matte Refine");
  // 1. Describe: a filter; Source (mandatory, RGBA / RGB) + Output only, no "Matte" clip.
  CHECK(d.props.getStr(kOfxImageEffectPropSupportedContexts, 0) == kOfxImageEffectContextFilter);
  CHECK(d.props.props[kOfxImageEffectPropSupportedContexts].size() == 1);
  CHECK(d.clips.count("Matte") == 0);
  CHECK(d.clips.count(kOfxImageEffectSimpleSourceClipName) && d.clips.count(kOfxImageEffectOutputClipName) &&
        d.clips.size() == 2);
  if (d.clips.count(kOfxImageEffectSimpleSourceClipName)) {
    const PropSet& sp = d.clips[kOfxImageEffectSimpleSourceClipName]->props;
    CHECK(sp.getInt(kOfxImageClipPropOptional, 0, 0) == 0);
    CHECK(sp.getInt(kOfxImageClipPropIsMask, 0, 0) == 0);
    std::printf("describe: contexts %d, clips %zu, Source optional %d, components:",
                d.props.props[kOfxImageEffectPropSupportedContexts].size(), d.clips.size(),
                sp.getInt(kOfxImageClipPropOptional, 0, 0));
    const auto it = sp.props.find(kOfxImageEffectPropSupportedComponents);
    if (it != sp.props.end())
      for (const auto& c : it->second.s) std::printf(" %s", c.c_str());
    std::printf("\n");
  }

  const int W = 960, H = 540;
  const synth::Scene sc = synth::make("all", W, H);
  Feeds feeds;
  feeds[kOfxImageEffectSimpleSourceClipName] = rgbaFeed(&sc.rgb, &sc.matte);
  auto inst = host.createInstance(W, H, 3, feeds, ocioPath);
  setSourceColourspace(*inst, "ACEScg");
  CHECK(inst->getInt("view") == 0);  // Picture stays the default

  // 2. Default render: alpha = Fill Holes run directly on Source's alpha; RGB = Source's RGB; no footer.
  const std::vector<float> ref = defaultChain(sc.matte, W, H);
  Rendered r = inst->render(0);
  CHECK(r.status == kOfxStatOK);
  const long dA = diffChannel(r, ref, 3);
  long changed = 0;
  for (size_t i = 0; i < ref.size(); ++i) changed += ref[i] != sc.matte[i];
  std::printf("default render: alpha differs from fillHoles() at %ld px (fill changed %ld px); footer '%s'\n", dA,
              changed, inst->persistent.c_str());
  CHECK(dA == 0 && changed > 0);
  CHECK(diffChannel(r, sc.rgb, 0, 3, 0) == 0 && diffChannel(r, sc.rgb, 1, 3, 1) == 0 &&
        diffChannel(r, sc.rgb, 2, 3, 2) == 0);
  CHECK(inst->persistent.empty());
  CHECK(inst->outputColourspace() == "OfxColourspace_Source");
  inst->setInt("view", 2);  // Matte
  r = inst->render(0);
  CHECK(diffChannel(r, ref, 0) == 0 && diffChannel(r, ref, 3) == 0);
  CHECK(inst->outputColourspace() == "Matte");
  inst->setInt("view", 0);

  // 3. Snap to Image uses Source's RGB (no "needs the picture" footer).
  inst->setBool("snEnable", true);
  std::vector<float> snapRef(ref);
  mr::snapToImage(snapRef.data(), sc.rgb.data(), snapRef.data(), W, H, mr::SnapParams());
  r = inst->render(0);
  const long dSnap = diffChannel(r, snapRef, 3);
  std::printf("snap: alpha differs from snapToImage() at %ld px; footer '%s'\n", dSnap, inst->persistent.c_str());
  CHECK(dSnap == 0 && diffChannel(r, ref, 3) > 0 && inst->persistent.empty());
  inst->setBool("snEnable", false);

  // 4. Edge Colour Unmix: RGB = foregroundColour() on Source's RGB (Ignore Outer 3 px); linear picture = no footer.
  inst->setBool("fgEnable", true);
  mr::FgParams fp;
  fp.ignoreOuterPx = 3.0;
  std::vector<float> fg(sc.rgb.size());
  mr::foregroundColour(sc.rgb.data(), ref.data(), fg.data(), W, H, fp);
  r = inst->render(0);
  const long dFg = diffChannel(r, fg, 0, 3, 0) + diffChannel(r, fg, 1, 3, 1) + diffChannel(r, fg, 2, 3, 2);
  std::printf("edge colour unmix: RGB differs from foregroundColour() at %ld values; footer '%s'\n", dFg,
              inst->persistent.c_str());
  CHECK(dFg == 0 && diffChannel(r, ref, 3) == 0 && inst->persistent.empty());
  // Decision 24: a log picture is an error footer.
  setSourceColourspace(*inst, "ARRI LogC3 (EI800)");
  r = inst->render(0);
  std::printf("edge colour unmix, log picture: footer '%s'\n", inst->persistent.c_str());
  CHECK(inst->persistent.find("Edge Colour Unmix needs a scene-linear picture; the front input is ARRI LogC3") == 0);
  setSourceColourspace(*inst, "ACEScg");
  inst->setBool("fgEnable", false);
  r = inst->render(0);
  CHECK(inst->persistent.empty());

  // 5. Save Matte and Analyze Frame read Source's alpha.
  inst->press("saveMatte");
  CHECK(!inst->messages.empty() && inst->messages.back().text.find("Matte saved to ") == 0);
  if (!inst->messages.empty()) {
    const std::string& t = inst->messages.back().text;
    const std::string f32 = t.substr(15, t.find(".f32") + 4 - 15);
    std::ifstream raw(f32, std::ios::binary);
    std::vector<float> saved(size_t(W) * H);
    raw.read(reinterpret_cast<char*>(saved.data()), std::streamsize(saved.size() * sizeof(float)));
    CHECK(raw.good() && saved == sc.matte);
    const std::string txt = fileText(f32.substr(0, f32.size() - 4) + ".txt");
    CHECK(txt.find("matte_from=source alpha") != std::string::npos);
  }
  inst->press("analyzeFrame");
  CHECK(!inst->messages.empty() && inst->messages.back().text.find("Analyze Frame:") == 0);
  std::printf("analyze: '%s'\n", inst->messages.empty() ? "(none)" : inst->messages.back().text.c_str());

  // 6. RGB-only Source: no matte, footer points at the second socket.
  auto rgbOnly = host.createInstance(W, H, 1, Feeds{{kOfxImageEffectSimpleSourceClipName, rgbFeed(&sc.rgb)}}, ocioPath);
  r = rgbOnly->render(0);
  std::printf("rgb-only source: footer '%s'\n", rgbOnly->persistent.c_str());
  CHECK(r.status == kOfxStatOK);
  CHECK(rgbOnly->persistent ==
        "The picture has no alpha, so there is no matte: connect the matte to the node's second socket (source "
        "alpha).");

  // 7. Nothing on the source alpha (Flame: constant 1): alpha 1 passes through, footer.
  const std::vector<float> ones(size_t(W) * H, 1.0f);
  auto noMatte = host.createInstance(W, H, 1, Feeds{{kOfxImageEffectSimpleSourceClipName, rgbaFeed(&sc.rgb, &ones)}},
                                     ocioPath);
  r = noMatte->render(0);
  std::printf("constant alpha: footer '%s'\n", noMatte->persistent.c_str());
  CHECK(diffChannel(r, ones, 3) == 0);
  CHECK(noMatte->persistent ==
        "The input matte is constant 1.00: no matte on the source alpha. Connect the matte to the node's second "
        "socket (source alpha).");

  std::printf("echoes %d, max echo depth %d, host errors %zu\n", inst->echoes, inst->maxEchoDepth, host.errors.size());
  for (const auto& e : host.errors) std::printf("host error: %s\n", e.c_str());
  CHECK(host.errors.empty());
  CHECK(inst->maxEchoDepth <= 2);

  if (failures) {
    std::printf("refine_systest: %d FAILED\n", failures);
    return 1;
  }
  std::printf("refine_systest: all passed\n");
  return 0;
}
