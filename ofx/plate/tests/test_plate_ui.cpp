// Plate Merge plugin logic without OFX (plate_ui.h): px controls scale with the render scale (decision 6),
// negative sizes clamp to 0, names and footers, the output colourspace answers (decision 23).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "ofxColour.h"
#include "plate_ui.h"

using namespace mr::plate;

namespace {
int failures = 0;
#define CHECK(c)                                                \
  do {                                                          \
    if (!(c)) {                                                 \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);  \
      ++failures;                                               \
    }                                                           \
  } while (0)
}  // namespace

int main() {
  UiValues ui;
  PlateParams p = toPlateParams(ui, 1.0);
  CHECK(p.mode == Mode::BOverA && p.growPx == 8.0 && p.featherPx == 8.0 && p.levelMatch);
  ui.mode = 1;
  ui.growPx = 10;
  ui.featherPx = -3;
  ui.levelMatch = false;
  p = toPlateParams(ui, 0.5);  // a half-resolution proxy
  CHECK(p.mode == Mode::CleanBackground && p.growPx == 5.0 && p.featherPx == 0.0 && !p.levelMatch);
  CHECK(toPlateParams(ui, 0.0).growPx == 10.0);  // a missing scale counts as 1
  CHECK(clampView(-1) == kViewResult && clampView(kViewCount) == kViewResult && clampView(kViewRing) == kViewRing);
  CHECK(viewIsPicture(kViewResult) && viewIsPicture(kViewPlateA) && viewIsPicture(kViewPlateB) &&
        !viewIsPicture(kViewSwap) && !viewIsPicture(kViewResidual) && !viewIsPicture(kViewRing));
  for (int i = 0; i < kViewCount; ++i) CHECK(std::strlen(kViewNames[i]) <= 16);
  CHECK(constantMatteFooter("B").find("plate B") != std::string::npos);
  CHECK(sizeMismatchFooter(10, 20, 30, 40).find("10x20") != std::string::npos);
  PlateInfo info;
  info.matched = true;
  info.globalGainStops[0] = -0.3;
  info.residualPixels = 12;
  CHECK(infoFooter(info, true).find("-0.30") != std::string::npos);
  info.matched = false;
  CHECK(infoFooter(info, true).find("skipped") != std::string::npos);
  CHECK(infoFooter(info, false).find("off") != std::string::npos);
  const std::string ocio = kOfxImageEffectColourManagementOCIO;
  CHECK(outputColourspaceAnswer(true, true, ocio, true) == "OfxColourspace_Source");
  CHECK(outputColourspaceAnswer(false, true, ocio, true) == "Matte");
  CHECK(outputColourspaceAnswer(false, true, ocio, false) == kOfxColourspaceRaw);
  CHECK(outputColourspaceAnswer(true, false, kOfxImageEffectColourManagementBasic, false) == kOfxColourspaceOfxRaw);
  CHECK(outputColourspaceAnswer(false, true, "", false).empty());
  std::printf("test_plate_ui: %s (%d failures)\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
