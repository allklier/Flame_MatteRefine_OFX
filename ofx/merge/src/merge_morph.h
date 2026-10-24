// Matte Merge (ofx/merge/CLAUDE.md): full-frame helpers the merge needs and plugin 1's ops/morphology
// does not have: a Euclidean distance transform for large radii (Reach px reaches 100+ px, where
// mr::dilate's stamping costs more than a frame) and grey-level min / max filters for openings.
// Pure C++14 on w*h buffers, row 0 = top, threaded on mr::Workers; deterministic.
#pragma once

#include <cstdint>
#include <vector>

namespace mr {
class Workers;  // ops/morphology.h
namespace merge {

// Squared Euclidean distance (px^2, pixel centres) from each pixel to the nearest pixel with
// set[i] != 0; set pixels get 0. Pixels beyond the frame do not exist (the frame border is not a
// set pixel). No set pixel at all: every value is kFar. Felzenszwalb & Huttenlocher's exact
// lower-envelope transform, columns then rows.
constexpr float kFar = 1e20f;
// scratch (optional): reused buffers (grown to w*h floats) instead of an allocation per call.
void distanceSquared(const uint8_t* set, int w, int h, float* d2, Workers* wk = nullptr,
                     std::vector<float>* scratch = nullptr);

// Grey-level erosion (min) / dilation (max) with a (2r+1) x (2r+1) square, the window clipped to
// the frame (pixels beyond it are ignored, so a subject cut by the frame keeps its cut edge). r = 0:
// a copy. van Herk / Gil-Werman, O(1) per pixel. out may not be in.
void minFilter(const float* in, int w, int h, int r, float* out, Workers* wk = nullptr,
               std::vector<float>* scratch = nullptr);
void maxFilter(const float* in, int w, int h, int r, float* out, Workers* wk = nullptr,
               std::vector<float>* scratch = nullptr);

// Grey-level opening (erosion then dilation, square of half-size r): removes bright structures
// narrower than 2r + 1 px and keeps thick areas and their edge profile. out may not be in.
// scratch: grown to 2 w*h floats.
void opening(const float* in, int w, int h, int r, float* out, Workers* wk = nullptr,
             std::vector<float>* scratch = nullptr);

}  // namespace merge
}  // namespace mr
