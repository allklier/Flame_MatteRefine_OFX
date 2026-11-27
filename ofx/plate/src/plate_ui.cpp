// plate_ui.h
#include "plate_ui.h"

#include <algorithm>
#include <cstdio>

#include "ofxColour.h"  // colour management style names and kOfxColourspaceRaw / OfxRaw (OpenFX 1.5)

namespace mr {
namespace plate {

const char* const kViewNames[kViewCount] = {"Result", "Plate A", "Plate B Matched", "Swap Matte", "Residual", "Ring"};
const char* const kMatteOutNames[2] = {"Swap Matte", "Residual"};
const char* const kModeNames[2] = {"B Over A", "Clean Background"};

PlateParams toPlateParams(const UiValues& ui, double renderScale) {
  const double s = renderScale > 0 ? renderScale : 1.0;
  PlateParams p;
  p.mode = ui.mode == 1 ? Mode::CleanBackground : Mode::BOverA;
  p.growPx = std::max(0.0, ui.growPx) * s;
  p.featherPx = std::max(0.0, ui.featherPx) * s;
  p.levelMatch = ui.levelMatch;
  return p;  // ring and gain smoothness: auto from the delivered (already scaled) height
}

const char* const kNoPlateBFooter = "Connect plate B (its matte in its alpha) to the Plate B input. Plate A passes through.";

std::string constantMatteFooter(const char* which) {
  return std::string("Plate ") + which + "'s matte is missing (its alpha is a constant 1): connect the matte of the "
         "person in plate " + which + " to its alpha socket. Plate A passes through.";
}

std::string sizeMismatchFooter(int bw, int bh, int w, int h) {
  char b[200];
  std::snprintf(b, sizeof b, "Plate B is %dx%d, plate A %dx%d: they must be the same size (not resampled). Plate A "
                "passes through.", bw, bh, w, h);
  return b;
}

std::string failedFooter(const std::string& why) { return "Plate Merge failed: " + why + ". Plate A passes through."; }

std::string infoFooter(const PlateInfo& info, bool levelMatchOn) {
  char b[200];
  if (!levelMatchOn) {
    std::snprintf(b, sizeof b, "Level Match off. Residual (both persons): %ld px.", info.residualPixels);
  } else if (!info.matched) {
    std::snprintf(b, sizeof b, "Level Match skipped: too few clean pixels around the swap. Residual: %ld px.",
                  info.residualPixels);
  } else {
    std::snprintf(b, sizeof b, "Level Match: B %+.2f %+.2f %+.2f stops (R G B). Residual (both persons): %ld px.",
                  info.globalGainStops[0], info.globalGainStops[1], info.globalGainStops[2], info.residualPixels);
  }
  return b;
}

std::string outputColourspaceAnswer(bool viewPicture, bool sourceConnected, const std::string& style,
                                    bool configHasMatte) {
  if (viewPicture && sourceConnected) return "OfxColourspace_Source";
  if (style == kOfxImageEffectColourManagementOCIO) return configHasMatte ? "Matte" : kOfxColourspaceRaw;
  if (style == kOfxImageEffectColourManagementFull || style == kOfxImageEffectColourManagementCore)
    return kOfxColourspaceRaw;
  if (style == kOfxImageEffectColourManagementBasic) return kOfxColourspaceOfxRaw;
  return "";
}

}  // namespace plate
}  // namespace mr
