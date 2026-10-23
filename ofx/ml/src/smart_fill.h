// Smart Fill Holes (B3, ofx/ml/PLAN_FEATURES.md): plugin 1's Fill Holes decides by hole size (Max
// Hole %) and cannot tell a small defect from a small real gap. Here every hole is classified on
// its own: a defect (subject: fill it) or a real gap (keep it). Pure C++14, no OFX and no ONNX
// Runtime types; the segmentation model is behind SegRunner, so the tests use fakes.
//
// Holes (the same definition as Fill Holes, ofx/src/ops/CLAUDE.md decision 9 with Bridge Gaps 0
// and all frame edges open): the 4-connected components of matte < solidAbove that do not touch
// the frame edge. fill_holes.h does not expose its components, so they are found here again
// (runs + union-find, as fill_holes.cpp).
//
// Scores per hole, 0 = real gap .. 1 = subject, over the hole's core (its pixels below 0.5; all
// its pixels if it has none: the soft rim up to Solid Above belongs to the solid side):
//  - Segmentation: a statistic (mean / median / min) of the segmentation mask over the hole's
//    pixels. BiRefNet closes real gaps (PHASE0_RESULTS.md), so it is never used as a fill value,
//    only as a vote.
//  - Colour (no model): the mean picture colour inside the hole C_h against a subject reference
//    C_s (mean over the ring of pixels within ringPx of the hole with matte >= max(0.9,
//    solidAbove)) and a background reference C_b (mean over clear background pixels, matte <
//    0.05 and connected to the frame edge, in the 32 px grid cells that the hole's bounding
//    box grown by bgReachPx touches, the reach doubled until they hold at least 16 such pixels
//    or cover the frame). score = |C_h - C_b| / (|C_h - C_s| +
//    |C_h - C_b|) (Euclidean RGB, picture clamped 0-1); no reference: 0.5.
//  - Both: min(segmentation, colour) (fill only when both say subject).
// A hole is filled when score >= subjectThreshold. Fill (decision 10): 1.0, then the rim (decision
// 9 step 6): solid pixels (>= solidAbove) within rimPx of a filled hole are lifted to 1.0. Every
// other pixel is bit-identical to the input (decision 17).
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace mr {
namespace ml {

// The segmentation model: rgb w*h*3 (interleaved, row 0 = top, display-referred, any range) ->
// mask w*h in [0, 1] (1 = subject), full resolution. Throws std::exception on failure.
class SegRunner {
 public:
  virtual ~SegRunner() {}
  virtual void segment(const float* rgb, int w, int h, float* mask) = 0;
};

enum class ScoreMode { Segmentation, Colour, Both };
enum class SegStatistic { Mean, Median, Min };

struct SmartFillParams {
  float solidAbove = 0.95f;        // matte >= it is solid (Fill Holes "Solid Above", decision 8)
  double subjectThreshold = 0.6;   // score >= it: a defect, filled (0.6: eval_smart_fill, low-contrast picture)
  double maxHoleFraction = 0.1;    // never fill a hole larger than this fraction of the frame; 0 = no limit
  bool keepRealGaps = true;        // false: fill every hole up to maxHoleFraction, no scoring
  ScoreMode mode = ScoreMode::Both;
  SegStatistic segStat = SegStatistic::Mean;
  double rimPx = 1.0;              // as Fill Holes "Rim px"; 0 = no rim
  double ringPx = 4.0;             // colour: subject ring width around the hole
  double bgReachPx = 64.0;         // colour: background window around the hole's bounding box
};

enum class HoleDecision { Filled, KeptRealGap, KeptTooLarge, NotScored };

struct HoleInfo {
  long long area = 0;
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;           // bounding box, inclusive
  int sx = 0, sy = 0;                           // its first pixel in raster order (a flood fill seed)
  float segMean = -1, segMedian = -1, segMin = -1;  // -1 = not computed
  float colour = -1;                            // -1 = not computed
  float score = -1;                             // what was compared with the threshold
  HoleDecision decision = HoleDecision::NotScored;
};

struct SmartFillStats {
  int holesFound = 0, holesFilled = 0, holesKept = 0, holesTooLarge = 0;
  long long pixelsChanged = 0;
  bool segmented = false;  // the SegRunner was called
  bool aborted = false;
  double labelMs = 0, segMs = 0, scoreMs = 0, writeMs = 0, totalMs = 0;
  std::vector<HoleInfo> holes;  // in the order found (top to bottom by first pixel)
};

// matte w*h (0-1, row 0 = top), rgb w*h*3 interleaved; out w*h (may be matte). The runner is
// called at most once, only when a hole needs a segmentation score. abort() is asked before and
// after the runner call: when it says yes, out = the input and stats->aborted. Exceptions from the
// runner propagate (out is then undefined).
void smartFill(const float* matte, const float* rgb, int w, int h, const SmartFillParams& p, SegRunner& runner,
               float* out, SmartFillStats* stats = nullptr,
               const std::function<bool()>& abort = std::function<bool()>());

// --- BiRefNet pre / post processing (model table row "birefnet-lite" / "birefnet") ----------------
// The model's preprocessor_config.json: resize to 1024 x 1024 with PIL bilinear (resample 2),
// rescale to 0-1, ImageNet mean / std; its README: output_image -> sigmoid -> resize back.
constexpr int kBiRefNetSize = 1024;
// rgb w*h*3 (clamped to 0-1 first, NaN -> 0) -> input [3, 1024, 1024] planar, normalised.
void birefnetPrepare(const float* rgb, int w, int h, float* input);
// logits [1, 1024, 1024] -> mask w*h: sigmoid, then resized to w x h.
void birefnetPostprocess(const float* logits, int w, int h, float* mask);
// Separable resampling as PIL's BILINEAR: a triangle filter, widened by the scale when shrinking
// (antialiased), plain bilinear when enlarging; pixel centres at +0.5, weights renormalised at
// the frame edge (= the edge pixel repeated when enlarging). src sw*sh*ch
// interleaved -> dst dw*dh*ch.
void resampleBilinear(const float* src, int sw, int sh, int ch, float* dst, int dw, int dh);

const char* decisionName(HoleDecision d);

}  // namespace ml
}  // namespace mr
