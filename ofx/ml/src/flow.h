// Dense optical flow for Temporal Stabilise (B4, ofx/ml/PLAN_FEATURES.md): pyramidal, iterative,
// dense Lucas-Kanade with a per-pixel weight on the data term. Pure C++14, no OpenCV.
//
// Why this method: the matte's edge is the motion boundary (subject over background), and the
// stabiliser needs the SUBJECT's motion there, on both sides of the edge (the matte edge moves
// with the subject even where the pixel shows background). Lucas-Kanade's window sums take a
// weight per pixel for free: weighting them by the reference frame's matte makes every window
// measure the subject's motion only, extended a window's reach into the background side of the
// band. A global-smoothness method (Horn-Schunck) or a patch search (DIS) would need the same
// weighting bolted on; LK gets it in the sums. Textureless areas: the increment is damped
// (Tikhonov, `lambda` per unit of weight), so they keep the coarser level's estimate, which saw a
// wider area. Every pixel's result depends only on the input values (no thread-dependent sums):
// deterministic at any thread count.
#pragma once

#include <vector>

namespace mr {
class Workers;  // ops/morphology.h
namespace ml {

struct FlowParams {
  int levels = 5;            // pyramid levels at most (stops before a side gets below 16 px)
  int iterations = 2;        // per level (more changed nothing on the sequence scene)
  int radius = 8;            // window half-width, px of each level (8: reaches across the band)
  float damping = 0.05f;     // damping relative to the window's gradient energy (trace)
  float lambda = 1e-7f;      // damping floor per unit of weight (luma^2 / px^2): textureless areas
  float matteGain = 0.0f;    // weight of the matte as a second channel (0: the picture only)
};

// A flow field: b(p + (u, v)(p)) ~ a(p), px of its own grid.
struct Flow {
  int w = 0, h = 0;
  std::vector<float> u, v;
};

// Dense flow from image a to image b (w*h each). ma, mb: the mattes as a second channel (used when
// p.matteGain > 0; may be nullptr otherwise). weight: w*h weights of a's data term (nullptr = 1).
Flow denseFlow(const float* a, const float* b, const float* ma, const float* mb, const float* weight,
               int w, int h, const FlowParams& p, Workers* wk = nullptr);

// in (w*h) at 1 / s (s a power of two), halved s times by a [1 3 3 1] / 8 binomial: out is
// ow*oh, ow = w halved (rounding up) log2(s) times; grid pixel j is centred on frame px j s + (s - 1) / 2.
void downsample(const float* in, int w, int h, int s, float* out, Workers* wk = nullptr);
// Luma (Rec.709 weights) of rgb (w*h*3, each channel clamped to 0-1, NaN -> 0), downsampled.
void luma(const float* rgb, int w, int h, int s, float* out, Workers* wk = nullptr);

// Bilinear read of a w*h field at (x, y), pixel centres at integers, clamped to the field.
float sampleField(const float* f, int w, int h, float x, float y);

}  // namespace ml
}  // namespace mr
