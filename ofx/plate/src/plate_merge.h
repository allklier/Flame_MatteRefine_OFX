// Plate Merge (docs/plate_merge.md, root decision 43): two aligned plates of the same camera, each
// with a tight matte of the person in it, merged into one plate. Pure C++14 on dense float buffers,
// row 0 = top, no OFX types (decision 3). Pictures are w*h*3 RGB (any float values: > 1 and < 0
// pass unchanged), mattes w*h (clamped to 0-1, NaN = 0).
//
// Per pixel: a = the swap weight of plate A's matte (haze floor, grown, feathered outward), b = the
// same for B, k = the Keep A protect matte (0 if none). The share of plate B in the output:
//   B Over A:         s = (1 - (1 - a)(1 - b)) (1 - k)   (B over A through the union: B's person on top,
//                                                      A's person replaced by B's clean background)
//   Clean Background: s = a (1 - b) (1 - k)             (both persons removed where the other plate is clean)
// out = A where s == 0 (bit for bit), B' where s == 1, A + s (B' - A) between. B' = B times a smooth
// per-channel gain that matches B's level to A on a ring of pixels around the swap region where
// neither matte is on (Level Match; scene-linear assumption). Residual = a b: both persons there, no
// clean pixel in either plate (covered by B's person in B Over A; left for an inpaint otherwise).
#pragma once

#include <vector>

namespace mr {
class Workers;  // ops/morphology.h
namespace plate {

enum class Mode { BOverA = 0, CleanBackground = 1 };

// Matte values below this are haze (an LTX matte's faint tail), not the person: ignored. Fixed.
constexpr float kHazeFloor = 0.02f;
// Level Match clamps its gain to +-this many stops (a larger difference is not a level mismatch).
constexpr double kMaxGainStops = 2.0;

struct PlateParams {
  Mode mode = Mode::BOverA;
  double growPx = 8.0;     // swap region reaches this far beyond each matte's 0.5 contour (halo, shadow)
  double featherPx = 8.0;  // then fades to 0 over this many px (smoothstep); 0 = hard edge
  bool levelMatch = true;
  double ringPx = 0.0;      // ring width for Level Match; 0 = auto (2 % of the height, at least 4 px)
  double matchSigmaPx = 0;  // spatial smoothness of the gain; 0 = auto (5 % of the height, at least 8 px)
};

struct PlateInfo {
  long swapPixels = 0;      // s > 0
  long residualPixels = 0;  // a b > 0
  long ringPixels = 0;
  bool matched = false;               // Level Match ran (on, and the ring had enough pixels)
  double globalGainStops[3] = {0, 0, 0};  // log2(mean A / mean B) on the whole ring, per channel
};

// Optional outputs (nullptr = not wanted), w*h each.
struct PlateViews {
  float* swap = nullptr;      // s
  float* residual = nullptr;  // a b
  float* ring = nullptr;      // 1 on the Level Match ring, else 0
  float* matchedB = nullptr;  // w*h*3: plate B after Level Match (B itself where s == 0)
};

// Swap weight of one matte: max(m', ramp(d)), m' = m with values below kHazeFloor set to 0, d = the
// Euclidean distance (px) to the nearest pixel with m >= 0.5, ramp = 1 for d <= grow, then a
// smoothstep down to 0 at grow + feather. Exactly 0 beyond grow + feather and wherever m' = 0 there.
void swapWeight(const float* m, int w, int h, double growPx, double featherPx, float* out, Workers* wk = nullptr);

// The merge. keepA may be nullptr. rgbOut (w*h*3) may not alias the inputs. Throws
// std::invalid_argument on a bad size or a null required pointer.
void mergePlates(const float* aRgb, const float* aMatte, const float* bRgb, const float* bMatte, const float* keepA,
                 int w, int h, const PlateParams& p, float* rgbOut, const PlateViews& views = PlateViews(),
                 PlateInfo* info = nullptr);

// Auto values of PlateParams for a frame height (render-scaled).
double autoRingPx(int h);
double autoSigmaPx(int h);

}  // namespace plate
}  // namespace mr
