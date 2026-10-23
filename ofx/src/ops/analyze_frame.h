// Analyze Frame (CLAUDE.md roadmap, DECIDED for plugin 1): reads one frame's input matte and
// suggests the Clean & Crush, Fill Holes and Solid Core controls, each with a plain-words note
// for the popup. Plain statistics on the matte, checked by running the operations themselves
// on it (clean -> fill -> core, the chain order). Not per frame: it runs on a button press.
#pragma once

#include <string>
#include <vector>

#include "ops/clean_background.h"
#include "ops/fill_holes.h"
#include "ops/solid_core.h"

namespace mr {

struct AnalysisResult {
  // Suggested settings. Each starts at the plugin's default (plugin.cpp), so a control the
  // analysis leaves alone holds its default and its `changed` flag stays false.
  bool cleanEnable = false;
  CleanBackgroundParams clean;  // maxIslandPercent, backgroundBelow, crushPx, featherPx
  bool fillEnable = true;
  FillHolesParams fill = defaultFill();  // threshold ("Solid Above"), maxHolePercent, bridgePx,
                                         // rimPx, closedLeft/Right/Top/Bottom
  bool coreEnable = false;
  SolidCoreParams core;  // depthPx, edgeAt, featherPx

  // Which controls differ from their defaults (what the popup lists as changed).
  struct Changed {
    bool cleanEnable = false, maxIsland = false, backgroundBelow = false, crushPx = false;
    bool fillEnable = false, threshold = false, maxHole = false, bridge = false;
    bool closedLeft = false, closedRight = false, closedTop = false, closedBottom = false;
    bool coreEnable = false, depth = false;
  } changed;

  // One line per decision, for the popup: what was chosen and why; "left alone" lines last.
  std::vector<std::string> notes;

  // Confidence. When any of these is set, nothing is changed and the notes say why.
  bool constantMatte = false;  // max - min below 0.01 (nothing connected, or an empty matte)
  bool noSubject = false;      // almost nothing (< 0.01 % of the frame) at or above 0.5
  bool noisy = false;          // grey or speckled everywhere: no reliable reading
  bool tooSmall = false;       // fewer than 64 pixels

  // Measurements (for the log and the tests), at the frame's own resolution.
  struct Measured {
    double subjectPercent = 0;      // pixels >= 0.5, % of the frame
    double greyPercent = 0;         // pixels in (0.15, 0.85), % of the frame
    int holes = 0;                  // defect holes found at the suggested Solid Above
    int realGaps = 0;               // enclosed areas judged real (kept open by Max Hole %)
    long long holePixels = 0;       // pixels the holes cover
    double largestHolePercent = 0;  // largest defect hole
    double smallestGapPercent = 0;  // smallest real gap (0: none)
    int islands = 0, objects = 0;   // components outside the subject: removed / kept
    double hazeLevel = 0;           // 99.9th percentile of the far background
    long long deepGreyBefore = 0;   // pixels < 0.9 more than 8 px inside, after clean + fill
    long long deepGreyAfter = 0;    // the same after Solid Core (= before if it is off)
    long long cleanChanged = 0, fillChanged = 0, coreChanged = 0;  // the ops' own counts
    double ms = 0;                  // wall time of analyzeFrame
  } measured;

  static FillHolesParams defaultFill() {
    FillHolesParams p;
    p.threshold = 0.95f;  // the plugin's default Solid Above (decision 8), not the struct's 0.5
    return p;
  }
};

// matte: w*h floats, row 0 = the top row (values outside 0-1 are clamped). Pixel-sized
// results are in this frame's pixels: the plugin converts by its render scale (decision 6).
void analyzeFrame(const float* matte, int w, int h, AnalysisResult* out);

}  // namespace mr
