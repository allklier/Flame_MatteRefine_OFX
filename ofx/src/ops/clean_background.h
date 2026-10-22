// Operations A1 Clean Background + A2 Background Crush (CLAUDE.md "Plugin 1 operations"), one
// column "Clean & Crush": remove small islands outside the subject, then send everything more
// than Crush px outside the matte to 0 (the mirror of Solid Core, decisions 15, 16). Never
// brightens a pixel; pixels that are neither in a removed island nor crushed are bit-identical.
#pragma once

namespace mr {

struct CleanBackgroundParams {
  // Pixels below this are background (haze, noise); the matte = pixels at or above it. Islands
  // are the 8-connected components of the matte, and Crush measures from its contour.
  float backgroundBelow = 0.15f;
  double maxIslandPercent = 0.05;  // remove components up to this % of the frame; 0 = off
  double crushPx = 4.0;            // crush starts this far outside the edge; 0 = off
  double featherPx = 2.0;          // ramp from the input to 0 over this many px; 0 = hard
  // Not a control: how far a removed island's own soft fringe (below backgroundBelow, falling
  // away from it) is cleared with it. Pixel-sized, scaled with the others (decision 6).
  double fringePx = 2.0;
};

struct CleanBackgroundStats {
  int islandsRemoved = 0;
  int componentsKept = 0;          // the subject and objects above the size limit
  long long pixelsRemoved = 0;     // island and fringe pixels set to 0
  long long pixelsCrushed = 0;
};

// in, out: w*h floats, row 0 = the top row. out may be in.
void cleanBackground(const float* in, float* out, int w, int h, const CleanBackgroundParams& p,
                     CleanBackgroundStats* stats = nullptr);

}  // namespace mr
