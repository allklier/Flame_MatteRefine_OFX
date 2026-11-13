// Model candidate for the Focus Region (Edge Recipes WP4, decision 39 candidate (a), docs/model_candidate.md).
// ViTMatte on ONE crop centred on the region (not plugin 2's frame-aligned tiles), with the input and output
// handling the thin-hair research found best (ofx/ml/research/vitmatte_thin_hair/README.md section 6):
//   trimap   regionTrimap (ops/region_trimap.h), quantised as in training (< 1/3 -> 0, >= 2/3 -> 1, else 0.5),
//            then the solid core (base >= fgThreshold) forced to foreground (the research setup: the region is
//            the unknown zone, the core stays known);
//   picture  as delivered (display encoding, Flame's Auto colour, decision 34), clamped to 0-1;
//            low-contrast switch: when the region is dark AND flat (display luma, mean of RGB, over pixels with
//            weight > 0: p99 < lowContrastMaxHigh and p99 - p1 < lowContrastMaxRange), the crop gets the mild
//            (half-strength) stretch of the research: 0.5 x + 0.5 clamp((x - p1) / (p99 - p1) x 0.9 + 0.05);
//            unsharp mask (Gaussian sigma 1 px, amount 1, clamped 0-1), skipped on a stretched crop (see
//            unsharpWhenStretched);
//            normalisation per call: ImageNet mean / std (the authors' training) by default, or plugin 2's
//            0.5 / 0.5 (Norm::Half). Plugin 2's Edge Refine (ofx/ml/src/edge_refine.cpp) is NOT changed;
//   post     model alpha clamped 0-1 (NaN 0), restricted to the region (weight > 0, else 0); connected pieces
//            (8-connected, alpha > pieceThreshold) smaller than minPiecePx x s^2 pixels zeroed (s = sizeScale, see
//            below; a piece need not touch the matte: a lock can start 15 px away); Strand Gain
//            a = clamp((a - strandBlack) x strandGain, 0, 1) (black 0 + gain 1 = identity); then never lower:
//            alpha = base + w (max(base, a) - base) (w >= 1: max(base, a) exactly; w = 0: base bit for bit).
// No upscaling, no flip / rotation averaging (both hurt thin strands in the research).
//
// Confidence (feeds merge_fusion.h Candidate): 0 where the weight is 0, else
//   certainty x (0.5 + 0.5 agreement),  certainty = |2 a - 1|,  agreement = 1 - |a - ridge|,
//   ridge = clamp(|L - G(L)| / (0.25 x max(p99 - p1, 0.02)), 0, 1), L = display luma of the (un-stretched,
//   un-sharpened) picture, G = Gaussian blur with sigma ridgeSigmaPx x s; a = the result alpha above.
// So a model value at 0.5 has confidence 0; a strand (a 1) on a picture ridge 1; a strand on a flat picture or
// nothing (a 0) on a strong ridge 0.5.
//
// Crops: the bounding box of weight > 0, plus margin = marginPx x s on every side, rounded up to a multiple of 32
// (the model's divisibility), capped at the frame rounded up to 32 per axis; centred on the box, then shifted to
// stay inside the frame; pixels past the frame repeat the edge. A box wider or taller than maxCropSide - 2 margin
// is split into equal cells (each with its own margin; a cell without region weight is skipped); every pixel of
// the box is taken from the crop of its cell (no blending between cells).
//
// Sizes: s = sizeScale, 0 = auto = delivered height / 1080 (the research's sizes are 1080-line values; a proxy
// delivers fewer lines, so the render scale is included, decision 6). margin and the ridge sigma scale with s,
// the piece area with s^2 (floor 1 px); the unsharp sigma stays 1 px (the model works at pixel level).
// Pure C++14, no OFX, no ONNX Runtime: the model is behind CropRunner (runner_crops.h adapts RunnerClient).
#pragma once

#include <string>
#include <vector>

namespace mr {
namespace recipe {

// One model call: in = [cIn, h, w] float32 (NCHW without N), out = [cOut, h, w]; x, y = the crop's place in the
// frame (may be negative only if the frame is smaller than the crop: never, see above; information only).
// Throws std::exception on failure.
class CropRunner {
 public:
  virtual ~CropRunner() {}
  virtual void run(const std::string& modelId, const float* in, int cIn, int h, int w, float* out, int cOut, int x,
                   int y) = 0;
};

enum class Norm { ImageNet, Half };  // (x - mean) / std: ImageNet 0.485/0.456/0.406, 0.229/0.224/0.225; Half 0.5/0.5
enum class LowContrast { Auto, Off, On };

struct ModelCandidateParams {
  std::string modelId = "vitmatte-small";  // "vitmatte-base" optional
  Norm norm = Norm::ImageNet;
  float fgThreshold = 0.95f;  // base >= this = known foreground (regionTrimap's rule)
  double unsharpSigma = 1.0, unsharpAmount = 1.0;  // amount 0 = off
  LowContrast lowContrast = LowContrast::Auto;
  double lowContrastMaxRange = 0.12, lowContrastMaxHigh = 0.40;  // Auto: p99 - p1 below AND p99 below
  bool unsharpWhenStretched = false;  // the research's best dark-on-dark run was the stretch alone
  double strandBlack = 0.0, strandGain = 1.0;  // Strand Gain (levels); defaults = identity
  float pieceThreshold = 0.2f;
  double minPiecePx = 12;  // area at 1080 lines; 0 = no piece filter
  double marginPx = 32;    // at 1080 lines
  double ridgeSigmaPx = 2; // at 1080 lines
  double sizeScale = 0;    // 0 = delivered height / 1080
  int maxCropSide = 1024;  // the model table's maxSide for ViTMatte (model_table.cpp)
};

struct CropRect {
  int x, y, size_w, size_h;  // the crop in frame pixels (may reach past the frame: edge repeated)
  int cx0, cy0, cx1, cy1;    // the cell it writes (frame pixels, end exclusive)
};

struct ModelCandidateStats {
  long long regionPixels = 0;  // weight > 0
  int bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;  // bounding box of weight > 0 (end exclusive)
  std::vector<CropRect> crops;  // the crops run
  double lumaP1 = 0, lumaP99 = 0;
  bool stretched = false;
  int piecesRemoved = 0;
  long long pixelsRaised = 0;  // alpha > base
  double scale = 1, margin = 0, minPieceArea = 0;
  double prepMs = 0, runMs = 0, postMs = 0;
};

// rgb w*h*3 (display encoding, any range), base and region w*h (row 0 = top). alpha (w*h) and confidence (w*h,
// may be nullptr) are written for the whole frame: alpha = base bit for bit where the region weight is 0,
// confidence 0 there. An empty region: alpha = base, no model call. Exceptions from the runner propagate
// (alpha / confidence are then undefined).
void modelCandidate(const float* rgb, const float* base, const float* region, int w, int h,
                    const ModelCandidateParams& p, CropRunner& runner, float* alpha, float* confidence,
                    ModelCandidateStats* stats = nullptr);

// --- pieces (tested on their own) ---------------------------------------------------------------------------
// The crops for a box [bx0, bx1) x [by0, by1) in a w x h frame (see the top).
std::vector<CropRect> planCrops(int bx0, int by0, int bx1, int by1, int w, int h, double margin, int maxSide);
// Training quantisation of a trimap value: < 85/255 -> 0, >= 170/255 -> 1, else 0.5.
float quantiseTrimap(float v);
// Separable Gaussian (radius ceil(4 sigma), edges clamped) of a w*h plane; in place allowed (out may be in).
void gaussianBlur(const float* in, int w, int h, double sigma, float* out);
// Zeroes the 8-connected pieces of a > threshold with fewer than minArea pixels (a w*h, in place). Returns how many.
int dropSmallPieces(float* a, int w, int h, float threshold, double minArea);

}  // namespace recipe
}  // namespace mr
