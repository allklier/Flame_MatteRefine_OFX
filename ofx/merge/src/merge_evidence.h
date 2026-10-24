// Matte Merge picture evidence (ofx/merge/CLAUDE.md "Proposed method" 4): does the picture show the
// structure a matte change proposes? No colourspace: the picture is read as encoded.
//
// Two picture signals per pixel: brightness V = (R + G + B) / 3 (local contrast) and chroma
// S = max(R, G, B) - min(R, G, B) (local saturation), as encoded. Evidence for a proposed change c
// (the thin detail to add, or the detail-minus-base difference) at pixel p = the larger of
// |corr(c, V)| and |corr(c, S)| over the (2r + 1)^2 window at p. A correlation divides the local
// covariance by both local spreads, so it is the local contrast normalised by its local range: an
// affine change of V or S (exposure, a log or display curve that is near-linear over a window) leaves
// it unchanged. A real strand changes the picture where the matte changes (|corr| near 1 on a flat
// background); a hallucinated strand does not (|corr| near 0, or chance level on a textured
// background). The spread of V or S has a floor (1 % of its frame-wide 2-98 % range) so a flat
// picture gives no evidence rather than noise divided by nothing.
#pragma once

#include <cstdint>

namespace mr {
class Workers;
namespace merge {

struct EvidenceParams {
  int radius = 3;      // window half-size, px
  float lo = 0.30f;    // |corr| at or below: no support (weight 0)
  float hi = 0.60f;    // |corr| at or above: full support (weight 1); smoothstep between
  // The weights are then averaged over the candidates in the (2 s + 1)^2 window: a per-pixel decision
  // chopped strands into dashes on a busy background (2026-10-24, hair_busy). 0 = off.
  int smoothRadius = 4;
};

// rgb: w*h*3 picture (any encoding; non-finite values read as 0). change: w*h, the proposed change.
// candidate: w*h bytes, where to evaluate. weight (w*h): 0-1 at candidates, 0 elsewhere. corr
// (optional, w*h): the |corr| behind it (0 elsewhere).
void pictureEvidence(const float* rgb, const float* change, const uint8_t* candidate, int w, int h,
                     const EvidenceParams& p, float* weight, float* corr = nullptr, Workers* wk = nullptr);

}  // namespace merge
}  // namespace mr
