// Matte Merge plugin logic without OFX types (ofx/merge/CLAUDE.md, decision 37): the control values
// -> MergeParams mapping, the footer texts and the output colourspace answer, so they are unit-tested
// without a host (tests/test_merge_ui.cpp). plugin_merge.cpp only reads params and calls these.
#pragma once

#include <string>

#include "matte_merge.h"

namespace mr {
namespace merge {

// Choice options are stored as indexes in setups: append only (new options at the end).
enum View { kViewResult = 0, kViewPicture = 1, kViewAdded = 2, kViewRejected = 3, kViewZone = 4, kViewResidual = 5 };
extern const char* const kViewNames[6];
extern const char* const kSourceNames[2];    // Source: Matte edge / Trimap (merge::Source order)
extern const char* const kAlignNames[2];     // Align: Auto / Off
extern const char* const kMethodNames[2];    // Method: Add fine detail / Zone blend (merge::Method order)
extern const char* const kEvidenceNames[2];  // Picture Evidence: Off / On

// The node's controls as read from the params (choice indexes as stored).
struct UiValues {
  int source = 0;               // kSourceNames
  double reachPct = 6.0;        // % of the frame height
  bool keepDisconnected = false;
  double bridgePct = 0.5;       // % of the frame height (Add Fine Detail, Matte Edge; 0 = off)
  int align = 0;                // kAlignNames: 0 Auto, 1 Off
  int method = 0;               // kMethodNames
  double detailWidthPct = 0.5;  // % of the frame height
  double strength = 1.0;
  double featherPct = 0.25;     // % of the frame height
  int evidence = 0;             // kEvidenceNames: 0 Off, 1 On
};

// The defaults of UiValues are the core's (HeightSizes, MergeParams); test_merge_ui checks that.
// h: the height of the images as delivered (already at render scale, so a proxy keeps the same
// fraction of the frame). Out-of-range choice indexes fall back to the first option.
MergeParams toMergeParams(const UiValues& v, int h);

// View index clamped to the known views (unknown -> Result).
int clampView(int view);

// Footers (one sentence each, plain language).
extern const char* const kNoDetailFooter;
std::string constantBaseFooter(float value);
std::string noAlphaFooter();
std::string sizeMismatchFooter(int dw, int dh, int w, int h);
std::string failedFooter(const std::string& what);
// The info footer after a merge: the alignment used and its uncertainty, and the detail pixels
// kept / dropped in the zone, e.g. "Align: picture: move + scale, +-0.12 px. Detail: 12.3k px kept,
// 1.2k dropped."
std::string infoFooter(const MergeInfo& info);
// 1234 -> "1234", 12345 -> "12.3k", 1234567 -> "1.23M".
std::string shortCount(long n);

// GetOutputColourspace (decision 23 adapted, as Matte Refine ML): View Picture with the Source
// connected -> "OfxColourspace_Source" (the picture passes through); any other View (the matte on
// RGB) -> "Matte" if the OCIO config defines it, else "Raw" (OCIO, Full, Core) / "ofx_raw" (Basic);
// "" = reply default (no colour management style).
std::string outputColourspaceAnswer(bool viewPicture, bool sourceConnected, const std::string& style,
                                    bool configHasMatte);

}  // namespace merge
}  // namespace mr
