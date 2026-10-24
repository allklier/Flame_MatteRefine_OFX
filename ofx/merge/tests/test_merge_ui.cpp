// Matte Merge plugin logic (merge_ui.h): control values -> MergeParams, footers, the output
// colourspace answer. No host needed.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "merge_ui.h"

namespace {
int failures = 0;
#define CHECK(c)                                                    \
  do {                                                              \
    if (!(c)) {                                                     \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
      ++failures;                                                   \
    }                                                               \
  } while (0)
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }
}  // namespace

using namespace mr::merge;

int main() {
  // The node's defaults are the core's (user decision 2026-10-24: Reach 6 %, Detail Width 0.5 %,
  // Feather 0.25 % of height; Add Fine Detail, Align Auto, Evidence Off, Keep Disconnected off).
  const UiValues ui;
  const HeightSizes hs;
  const MergeParams core;
  CHECK(near(ui.reachPct, hs.reachPct) && near(ui.detailWidthPct, hs.detailWidthPct) &&
        near(ui.featherPct, hs.featherPct));
  CHECK(near(ui.reachPct, 6.0) && near(ui.detailWidthPct, 0.5) && near(ui.featherPct, 0.25));
  CHECK(near(ui.bridgePct, hs.bridgePct) && near(ui.bridgePct, 0.5));  // Bridge (user, 2026-10-24)
  const MergeParams p1080 = toMergeParams(ui, 1080);
  CHECK(p1080.source == core.source && p1080.method == core.method && p1080.align == core.align &&
        p1080.evidence == core.evidence && p1080.keepDisconnected == core.keepDisconnected &&
        near(p1080.strength, core.strength));
  CHECK(near(p1080.reachPx, core.reachPx) && near(p1080.detailWidthPx, core.detailWidthPx) &&
        near(p1080.featherPx, core.featherPx) && near(p1080.bridgePx, core.bridgePx));  // 64.8 / 5.4 / 2.7 / 5.4 at 1080
  CHECK(near(p1080.bridgePx, 5.4));
  CHECK(p1080.source == Source::MatteEdge && p1080.method == Method::AddFineDetail && p1080.align &&
        !p1080.evidence && !p1080.keepDisconnected);

  // Sizes follow the delivered height: UHD doubles them, a half-scale proxy halves them (decision 6
  // is satisfied by the percentage: the images arrive at render scale).
  const MergeParams p2160 = toMergeParams(ui, 2160), p540 = toMergeParams(ui, 540);
  CHECK(near(p2160.reachPx, 129.6) && near(p2160.detailWidthPx, 10.8) && near(p2160.featherPx, 5.4));
  CHECK(near(p540.reachPx, 32.4) && near(p540.detailWidthPx, 2.7) && near(p540.featherPx, 1.35));
  CHECK(near(p2160.bridgePx, 10.8) && near(p540.bridgePx, 2.7));
  UiValues v;
  v.reachPct = 10;
  v.detailWidthPct = 1;
  v.featherPct = 0;
  v.strength = 0.4;
  v.bridgePct = 0;
  const MergeParams pv = toMergeParams(v, 1000);
  CHECK(near(pv.reachPx, 100) && near(pv.detailWidthPx, 10) && near(pv.featherPx, 0) && near(pv.strength, 0.4) &&
        pv.bridgePx == 0.0);
  v.bridgePct = 1.5;
  CHECK(near(toMergeParams(v, 1000).bridgePx, 15));

  // Choice indexes (append only): 1 = the second option of each.
  v = UiValues();
  v.source = 1;
  v.method = 1;
  v.align = 1;
  v.evidence = 1;
  v.keepDisconnected = true;
  const MergeParams pc = toMergeParams(v, 1080);
  CHECK(pc.source == Source::Trimap && pc.method == Method::ZoneBlend && !pc.align && pc.evidence &&
        pc.keepDisconnected);
  // Unknown indexes (a setup from a later version) fall back to the first option.
  v.source = v.method = v.align = v.evidence = 7;
  const MergeParams pu = toMergeParams(v, 1080);
  CHECK(pu.source == Source::MatteEdge && pu.method == Method::AddFineDetail && pu.align && !pu.evidence);
  v.source = v.method = v.align = v.evidence = -1;
  const MergeParams pn = toMergeParams(v, 1080);
  CHECK(pn.source == Source::MatteEdge && pn.method == Method::AddFineDetail && pn.align && !pn.evidence);

  // Option names and their order (stored as indexes in setups).
  CHECK(std::strcmp(kViewNames[kViewResult], "Result Matte") == 0 && std::strcmp(kViewNames[kViewPicture], "Picture") == 0 &&
        std::strcmp(kViewNames[kViewAdded], "Added") == 0 && std::strcmp(kViewNames[kViewRejected], "Rejected") == 0 &&
        std::strcmp(kViewNames[kViewZone], "Zone") == 0 && std::strcmp(kViewNames[kViewResidual], "Residual") == 0);
  CHECK(std::strcmp(kSourceNames[0], "Matte Edge") == 0 && std::strcmp(kSourceNames[1], "Trimap") == 0);
  CHECK(std::strcmp(kAlignNames[0], "Auto") == 0 && std::strcmp(kAlignNames[1], "Off") == 0);
  CHECK(std::strcmp(kMethodNames[0], "Add Fine Detail") == 0 && std::strcmp(kMethodNames[1], "Zone Blend") == 0);
  CHECK(std::strcmp(kEvidenceNames[0], "Off") == 0 && std::strcmp(kEvidenceNames[1], "On") == 0);
  CHECK(clampView(0) == kViewResult && clampView(5) == kViewResidual && clampView(6) == kViewResult &&
        clampView(-1) == kViewResult && clampView(3) == kViewRejected);

  // Footers.
  CHECK(std::string(kNoDetailFooter) == "Connect the detail matte to the Detail input.");
  CHECK(constantBaseFooter(1.0f).find("constant 1.00") != std::string::npos &&
        constantBaseFooter(1.0f).find("base matte or trimap") != std::string::npos);
  CHECK(sizeMismatchFooter(960, 540, 1920, 1080).find("960x540, the picture 1920x1080") != std::string::npos);
  CHECK(failedFooter("x") == "Matte Merge failed (x): the base matte passes through.");
  CHECK(shortCount(0) == "0" && shortCount(9999) == "9999" && shortCount(12345) == "12.3k" &&
        shortCount(999949) == "999.9k" && shortCount(1234567) == "1.23M");
  MergeInfo info;  // Align Off: no uncertainty
  info.align.model = AlignModel::Off;
  CHECK(infoFooter(info) == "Align: off. Detail: 0 px kept, 0 dropped.");
  info.align.model = AlignModel::None;
  info.keptPixels = 12345;
  info.droppedPixels = 678;
  CHECK(infoFooter(info) == "Align: no alignment (could not register). Detail: 12.3k px kept, 678 dropped.");
  info.align.model = AlignModel::PictureFull;
  info.align.uncertaintyPx = 0.123;
  CHECK(infoFooter(info) == "Align: picture: move + scale, +-0.12 px. Detail: 12.3k px kept, 678 dropped.");
  info.align.model = AlignModel::MatteFull;
  info.align.uncertaintyPx = 3.456;
  CHECK(infoFooter(info) == "Align: mattes: move + scale, +-3.46 px. Detail: 12.3k px kept, 678 dropped.");

  // Output colourspace (decision 23 adapted).
  const std::string ocio = "OfxImageEffectColourManagementOCIO";
  CHECK(outputColourspaceAnswer(true, true, ocio, true) == "OfxColourspace_Source");
  CHECK(outputColourspaceAnswer(true, true, "", false) == "OfxColourspace_Source");
  CHECK(outputColourspaceAnswer(false, true, ocio, true) == "Matte");
  CHECK(outputColourspaceAnswer(false, true, ocio, false) == "Raw");
  CHECK(outputColourspaceAnswer(true, false, ocio, true) == "Matte");  // no picture: the matte is on RGB
  CHECK(outputColourspaceAnswer(false, true, "OfxImageEffectColourManagementFull", false) == "Raw");
  CHECK(outputColourspaceAnswer(false, true, "OfxImageEffectColourManagementCore", false) == "Raw");
  CHECK(outputColourspaceAnswer(false, true, "OfxImageEffectColourManagementBasic", false) == "ofx_raw");
  CHECK(outputColourspaceAnswer(false, true, "", false).empty());

  if (failures) {
    std::printf("test_merge_ui: %d FAILED\n", failures);
    return 1;
  }
  std::printf("test_merge_ui: all passed\n");
  return 0;
}
