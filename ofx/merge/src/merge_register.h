// Matte Merge registration (ofx/merge/CLAUDE.md "Proposed method" 2): the offset and small scale
// of the detail matte relative to the base, estimated on the thick, confident parts of both.
// Since 2026-10-24 the merge registers on the picture (merge_picreg.h); this estimate is its start
// (1/4 resolution, refine = false) and the fallback when the picture shows no usable detail.
//
// Method: the signed distance fields of both mattes' openings (thin detail removed), matched by
// Gauss-Newton over (tx, ty, scale about the frame centre, delta) with robust (Huber, then cut)
// weights, on a 1/4-resolution level first, then refined at 1/2 resolution (2x2 box + [1 2 1] blur
// per level; full resolution only for frames under 256 px). delta is a uniform edge offset:
// the detail matte may be fatter or thinner than the base everywhere (a model's soft edge, an eroded
// roto, a trimap's known-foreground line drawn inside the subject), which would otherwise bias the
// scale. Why not plugin 2's dense Lucas-Kanade flow (ofx/ml/src/flow.*): it estimates a per-pixel
// field on pictures, needs the ML library, and its window sums would follow the detail's edge offset
// as motion; four global parameters on the distance field are what the merge needs and are cheap.
// Known limit: a single circle cannot tell scale from a uniform offset (they are the same change);
// shapes with edges at several distances from the frame centre can.
#pragma once

#include <vector>

namespace mr {
class Workers;
namespace merge {

struct Registration {
  bool ok = false;             // false: too few usable edge pixels; the identity is returned
  double tx = 0, ty = 0;       // px
  double scale = 1;            // about (cx, cy)
  double delta = 0;            // detail's edge offset relative to the base (px, > 0: detail fatter)
  double cx = 0, cy = 0;       // the frame centre, px
  double rmsPx = 0;            // RMS residual of the inlier edge pixels after alignment (px)
  int samples = 0;             // edge pixels used at the refinement level
  // Conditioning: the contour displacement (px) a systematic edge-offset error of 1 px RMS can cause
  // (sqrt(N * largest eigenvalue of the (tx, ty, scale) covariance per unit residual)). Small (~1-3):
  // the shape pins the move down; large: translation, scale and the edge offset trade off (a convex
  // subject cut by the frame), and a base whose edge offset varies along its contour biases the move.
  double sensitivity = 0;
  // The detail-matte position that lines up with base pixel (x, y).
  void apply(double x, double y, double* qx, double* qy) const {
    *qx = cx + scale * (x - cx) + tx;
    *qy = cy + scale * (y - cy) + ty;
  }
};

// base, detail: w*h mattes. baseLevel: the base contour that bounds its confident part (0.5 for a
// matte; between the unknown grey and 1 for a trimap). openRadius: the opening (square half-size)
// that removes the thin detail before matching. Searches shifts up to ~60 px and scales up to 5 %.
// fixScale: scale held at 1 (translation and the edge offset delta only; sensitivity stays 0).
// delta is still fitted then: it is a nuisance parameter of the fit, never applied to the detail,
// and holding it at 0 would turn a fatter detail matte into a move (its top edge moves up when
// the subject is cut by the bottom of the frame). refine = false: the 1/4-resolution level only
// (frames >= 256 px; a start for the picture estimate, ~1-2 px, at about a third of the cost).
Registration registerMattes(const float* base, const float* detail, int w, int h, float baseLevel,
                            int openRadius, Workers* wk = nullptr, bool fixScale = false, bool refine = true);

// out(p) = detail at reg.apply(p), bilinear, clamped to the frame. out may not be detail.
void warpDetail(const float* detail, int w, int h, const Registration& reg, float* out, Workers* wk = nullptr);

}  // namespace merge
}  // namespace mr
