// Operation A7, Foreground Colour (CLAUDE.md "Plugin 1 operations"): the clean foreground
// colour in the matte's soft edge, unmixed from the old background, so a comp of it over a new
// background with the matte has no halo. Changes RGB only, never the matte.
#pragma once

namespace mr {

struct FgParams {
  // Smoothness of the estimated colours (Germer et al.'s regularization): higher = flatter fg
  // and background colours, less detail from the picture in the edge.
  double smoothness = 0.005;
  double extendPx = 3.0;  // also solve this far (px) beyond the soft edge, both sides
  float mix = 1.0f;       // 0 = the picture, 1 = the estimated foreground
};

struct FgStats {
  long long edgePixels = 0;    // 0 < alpha < 1
  long long solvedPixels = 0;  // the edge plus Extend px, at full resolution
  int levels = 0;
};

// rgb: w*h*3 picture (scene-linear, may exceed 1); alpha: w*h matte, 0-1; fg: w*h*3 output, the
// un-premultiplied foreground colour, not the same buffer as rgb. Row 0 = the top row.
// alpha = 1: fg = rgb bit for bit. alpha = 0: the foreground colour extended outward (smooth
// colour of the nearby subject, not the old background), so filtering fg later can't pull the
// old background back into the edge.
void foregroundColour(const float* rgb, const float* alpha, float* fg, int w, int h,
                      const FgParams& p, FgStats* stats = nullptr);

}  // namespace mr
