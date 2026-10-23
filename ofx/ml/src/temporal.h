// Temporal Stabilise (B4, ofx/ml/PLAN_FEATURES.md): calms a matte's frame-to-frame flicker in its
// edge band, and flickering holes, with motion-compensated neighbour frames. Classical, no model;
// pure C++14 (no OFX, no ONNX Runtime). The centre frame's matte stays the source (decision 17):
// outside the region below the output is bit-identical to it.
//
// Per neighbour n (|offset| <= frames): flow centre -> n and n -> centre (flow.h) on luma at a
// reduced resolution (at most flowLines lines: HD and UHD at 540), each weighted by its own
// frame's matte, so it measures the subject's motion, also on the background side of the edge
// (where the matte edge moves with the subject). Confidence per pixel: the forward-backward check
// |F(p) + B(p + F(p))|, in px of the flow grid: exp(-e^2 / (2 * 0.5^2)), 0 beyond fbMaxPx.
// Neighbour matte value: the bilinear read at p + F(p) (full resolution); none where that is
// outside the frame. Weight: base (1 / 0.8 / 0.6 for the centre / offset 1 / offset 2) x confidence.
//
// The region: (1) the edge band: |sd| < bandPx, sd = signed distance to the centre matte's 0.5
// contour (ops/edge_field), at full weight to 2/3 of bandPx, falling linearly to 0 at bandPx;
// (2) flickering holes: a 4-connected region of centre matte < 0.5 not connected to the frame
// border (a hole, as Fill Holes defines one) is filled when the neighbours that see it filled
// outweigh the centre plus the neighbours that see it open (a neighbour sees it filled when its
// read covers at least half of the hole confidently and its mean there is >= 0.5): its pixels
// then get full weight. Holes are only filled, never punched into the centre: a hole in the
// neighbours but not in the centre stays solid (outside the band nothing is lowered).
// In the region: c = centre value, m = the weighted median (Combine::Mean: mean) of c and the
// neighbour values (with a single neighbour read, as at the ends with K = 1: the weighted mean, a
// median of two being always the centre); out = c + strength * regionWeight * (m - c).
// Without a picture (rgb nullptr) the flow runs on the mattes: it follows the matte's own
// jitter and barely stabilises (evaluated); the picture is the intended input.
#pragma once

#include "flow.h"

namespace mr {
class Workers;
namespace ml {

struct TemporalFrame {
  const float* rgb = nullptr;    // w*h*3 display-referred picture (clamped to 0-1 for the flow);
                                 // nullptr: the flow runs on the mattes
  const float* matte = nullptr;  // w*h, 0-1, row 0 = top
};

enum class Combine { Median, Mean };

struct StabiliseParams {
  int frames = 1;          // K: neighbours used on each side (frames farther from the centre: ignored)
  double strength = 1.0;   // 0 = the centre matte, 1 = the combined value
  double bandPx = 6.0;     // half-width of the edge band, px at the frame's resolution
  bool fillHoles = true;   // the flickering-hole rule (2)
  Combine combine = Combine::Median;
  bool subjectWeighted = true;  // flow data term weighted by the matte (false: plain LK)
  int flowLines = 600;     // the flow runs at full / 2^k so that it has at most this many lines
  double fbMaxPx = 1.0;    // forward-backward error beyond which a neighbour pixel is rejected
  FlowParams flow;
};

struct StabiliseStats {
  double flowMs = 0, regionMs = 0, combineMs = 0, totalMs = 0;
  int neighbours = 0, flowScale = 1;
  long long regionPixels = 0, pixelsChanged = 0;
  int holes = 0, holesFilled = 0;
  double rejectedFraction = 0;  // neighbour reads in the region rejected (FB check or outside)
};

// One neighbour's flow from the centre, on the flow grid (scale s: grid pixel j covers frame pixels
// [j s, (j + 1) s)), and the forward-backward confidence 0-1 there.
struct NeighbourFlow {
  int scale = 1;
  Flow fwd;
  std::vector<float> conf;
};
NeighbourFlow flowToNeighbour(const TemporalFrame& centre, const TemporalFrame& nb, int w, int h,
                              const StabiliseParams& p, Workers* wk = nullptr);
// The flow-grid scale for a frame of h lines (a power of two).
int flowScale(int h, int flowLines);
// The flow at frame pixel (x, y) in frame px.
void flowAt(const NeighbourFlow& f, double x, double y, double* dx, double* dy);

// frames: count consecutive frames, frames[centre] the one to stabilise; count may be 1 (out =
// the centre matte) or less than 2 K + 1 at the sequence ends. out: w*h floats; may be the
// centre matte (in place gives the same result).
void stabilise(const TemporalFrame* frames, int count, int centre, int w, int h,
               const StabiliseParams& p, float* out, StabiliseStats* stats = nullptr);

}  // namespace ml
}  // namespace mr
