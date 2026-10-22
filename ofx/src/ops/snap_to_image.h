// Operation A6, Snap to Image (CLAUDE.md "Plugin 1 operations"): pulls a slightly-off or blocky
// matte edge onto the picture's edges with the colour guided filter (He, Sun, Tang, "Guided
// Image Filtering", PAMI 2013), in two passes (the second filters the first's result, which
// halves the tail a single pass leaves outside a misplaced edge), only within Band px of the
// matte's edge pixels: its 0.5 crossings and its sloped grey (0.05-0.95, not flat grey areas).
// Pixels Band px or more from them are bit-identical (decision 17).
#pragma once

namespace mr {

struct SnapParams {
  double radiusPx = 8.0;     // filter window radius (window = 2r + 1 square)
  double bandPx = 12.0;      // changes reach this far from the edge pixels, full strength to half of it
  double detail = 0.5;       // 0-1: how weak a picture edge the matte follows (epsilon, log scale)
  double fromPicture = 0.35;  // 0-1: 0 = keep the matte's own softness, 1 = take it from the picture
};

struct SnapStats {
  long long bandPixels = 0;  // pixels with a blend weight > 0
  long long pixelsChanged = 0;
  float maxChange = 0.0f;
};

// The filter's regularisation for a Detail value, relative to the window's mean brightness
// squared (snap_to_image.cpp): 0.1 at Detail 0 down to 1e-4 at Detail 1.
double snapEpsilon(double detail);

// in, out: w*h floats 0-1; rgb: w*h*3 floats, the picture, scene-linear (values may exceed 1);
// row 0 = the top row. out may be in. Output is clamped to 0-1.
void snapToImage(const float* in, const float* rgb, float* out, int w, int h, const SnapParams& p,
                 SnapStats* stats = nullptr);

}  // namespace mr
