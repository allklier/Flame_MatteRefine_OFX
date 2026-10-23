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
  // Choke ("Ignore Outer px"): pixels at or above 0.5 within this many px inside the 0.5 contour
  // are not trusted to be foreground (after a grow or a hard Edge Width the matte's edge lies
  // outside the subject, so they hold the old background). They give no colour data: their
  // foreground colour is carried in from the pixels deeper inside, and they are output like
  // soft pixels. Needs at least the growth + ~1 px. A part with nothing deeper (a strand or limb
  // narrower than about 2x this) keeps its own data, as at 0. 0 = off. Scales with render scale.
  double ignoreOuterPx = 0.0;
};

struct FgStats {
  long long edgePixels = 0;    // 0 < alpha < 1
  long long ignoredPixels = 0;  // pixels (>= 0.5) ignored by Ignore Outer px
  long long solvedPixels = 0;  // the edge (and the ignored pixels) plus Extend px, full resolution
  int levels = 0;
};

// rgb: w*h*3 picture (scene-linear, may exceed 1); alpha: w*h matte, 0-1; fg: w*h*3 output, the
// un-premultiplied foreground colour, not the same buffer as rgb. Row 0 = the top row.
// alpha = 1: fg = rgb bit for bit, except the pixels ignored by ignoreOuterPx. alpha = 0: the
// foreground colour extended outward (smooth colour of the nearby subject, not the old
// background), so filtering fg later can't pull the old background back into the edge.
void foregroundColour(const float* rgb, const float* alpha, float* fg, int w, int h,
                      const FgParams& p, FgStats* stats = nullptr);

}  // namespace mr
