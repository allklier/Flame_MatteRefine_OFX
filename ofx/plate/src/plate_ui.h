// Plate Merge's plugin logic without OFX types (unit-tested in tests/test_plate_ui.cpp): the controls
// mapped to PlateParams (px scaled by the render scale, decision 6), view / matte-output names, footer
// texts and the output colourspace answer (decision 23, as Matte Merge's merge_ui).
#pragma once

#include <string>

#include "plate_merge.h"

namespace mr {
namespace plate {

// View (choice indexes stored in setups: append only).
enum View { kViewResult = 0, kViewPlateA = 1, kViewPlateB = 2, kViewSwap = 3, kViewResidual = 4, kViewRing = 5 };
constexpr int kViewCount = 6;
extern const char* const kViewNames[kViewCount];
// What the node's matte (alpha) output carries.
enum MatteOut { kMatteSwap = 0, kMatteResidual = 1 };
extern const char* const kMatteOutNames[2];
extern const char* const kModeNames[2];

inline int clampView(int v) { return v < 0 || v >= kViewCount ? kViewResult : v; }
inline bool viewIsPicture(int v) { return v == kViewResult || v == kViewPlateA || v == kViewPlateB; }

// The node's controls as read in render().
struct UiValues {
  int mode = 0;          // 0 B Over A, 1 Clean Background
  double growPx = 8.0;   // at full resolution
  double featherPx = 8.0;
  bool levelMatch = true;
};

// renderScale: the OFX render scale (x); proxies shrink px sizes with it.
PlateParams toPlateParams(const UiValues& ui, double renderScale);

// Footers.
extern const char* const kNoPlateBFooter;
std::string constantMatteFooter(const char* which);  // "A" / "B": the matte socket left empty (alpha 1)
std::string sizeMismatchFooter(int bw, int bh, int w, int h);
std::string failedFooter(const std::string& why);
std::string infoFooter(const PlateInfo& info, bool levelMatchOn);

// GetOutputColourspace: picture views keep the source's colourspace; matte views are tagged as a matte.
std::string outputColourspaceAnswer(bool viewPicture, bool sourceConnected, const std::string& style,
                                    bool configHasMatte);

}  // namespace plate
}  // namespace mr
