// Matte Merge registration on the picture (ofx/merge/CLAUDE.md, user decision 2026-10-24: the
// picture is required and registration may use it). Fine detail (hair) in the picture sits at its
// true position, so matching the detail matte's thin structures against the picture's local
// structure finds the detail's move without the thickness confound of the matte-to-matte estimate
// (merge_register.h: a coarse base's edge offset varies with direction and biases the move).
//
// Signals: D = the detail matte's white top-hat (detail minus its opening: the thin structures),
// and the picture's brightness V = (R + G + B) / 3 and chroma S = max - min, as encoded. Each is
// band-passed (minus its Gaussian blur, sigma ~ 1.5 px at 1080 lines): only local structure is
// left. Per block (~48 px at 1080) holding enough thin detail, the normalised cross-correlation of
// the band-passed D (moved by the current estimate) with the band-passed V and S over integer
// shifts; the larger |NCC| of the two channels wins, its sign is the strand's polarity (lighter or
// darker than its surroundings), and a parabola gives the sub-pixel peak. NCC is normalised per
// block, so an affine change of V or S, and a monotone curve (log, display) that is near-affine
// over a block, leave it unchanged: no colourspace is assumed. Thin structures in the detail with
// no counterpart in the picture (hallucinated strands, a busy background's chance matches) give a
// weak or inconsistent peak and drop out in the robust fit (Tukey weights on the block shifts) of
// the global move (tx, ty, scale about the frame centre; or tx, ty only). Three rounds, each
// re-moving D by the estimate so far (sub-pixel peaks are least biased near zero shift).
#pragma once

#include <cstdint>
#include <vector>

#include "merge_register.h"

namespace mr {
class Workers;
namespace merge {

struct PictureFit {
  bool ok = false;           // enough consistent blocks for the model asked for
  Registration reg;          // the estimate (tx, ty, scale; delta 0), ok as above
  int blocks = 0;            // blocks with enough thin detail
  int inliers = 0;           // blocks agreeing with the fitted move (Tukey weight > 0)
  double rmsPx = 0;          // RMS of the inlier blocks' shift residuals (px)
  // Estimated 1-sigma displacement error of the move over the frame (px): sqrt of the largest
  // eigenvalue of the parameter covariance (block residual variance times the inverse normal
  // matrix), in the same (tx, ty, scale x half the larger frame side) units as the matte estimate.
  double uncertaintyPx = 0;
};

// detail: w*h matte clamped to 0-1, in detail coordinates; picture: w*h*3 RGB (any encoding).
// openRadius: as registerMattes (Detail Width). init: the starting move (identity or the matte
// estimate). translationOnly: scale held at init.scale (normally 1). searchPx: the largest shift
// searched around init in the first round. zone (optional, w*h bytes, base px): only blocks with a
// zone pixel are used. Each block's signals are computed on a small window around it, so the cost
// follows the detail near the zone, not the frame size.
PictureFit registerToPicture(const float* detail, const float* picture, int w, int h, int openRadius,
                             const Registration& init, bool translationOnly, int searchPx,
                             const uint8_t* zone = nullptr, Workers* wk = nullptr);

}  // namespace merge
}  // namespace mr
