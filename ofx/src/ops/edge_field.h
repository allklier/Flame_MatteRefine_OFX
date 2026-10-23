// Sub-pixel edge geometry shared by the edge operations (decision 16, generalised to both sides
// of the edge): the signed distance to a matte's contour and resampling along the edge normal,
// which moves an edge while keeping its own softness profile.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "ops/morphology.h"

namespace mr {

// Where the `level` contour runs near pixel i (side = in < level): per axis, t = where level is
// crossed towards the nearest 4-neighbour on the other side (linear interpolation, 0-1); the
// contour is the line through those crossings: normal ~ (1/tx, 1/ty), distance
// s = 1 / sqrt(sum 1 / t^2) (decision 16). All zero if no neighbour is on the other side.
ContourFoot contourFoot(const float* in, const Mask& side, int w, int h, float level, int i);
// The same, side given as w*h bytes.
ContourFoot contourFoot(const float* in, const uint8_t* side, int w, int h, float level, int i);

// Signed distance of each pixel to the matte's `level` contour: > 0 inside (in >= level), < 0
// outside (morphology.h distanceWithin with contourFoot). Exact where |d| <= band; clamped to +-(band + 1)
// beyond. Pixels beyond the frame are neither inside nor outside: the frame border is no edge.
std::vector<float> signedEdgeDistance(const float* in, int w, int h, float level, double band);

// Bilinear sample of a w*h image at (x, y), pixel centres at integers, clamped to the frame.
inline float sampleBilinear(const float* img, int w, int h, double x, double y) {
  x = std::min(std::max(x, 0.0), double(w - 1));
  y = std::min(std::max(y, 0.0), double(h - 1));
  const int x0 = std::min(int(x), w - 2 < 0 ? 0 : w - 2), y0 = std::min(int(y), h - 2 < 0 ? 0 : h - 2);
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const double fx = x - x0, fy = y - y0;
  const double top = img[size_t(y0) * w + x0] * (1 - fx) + img[size_t(y0) * w + x1] * fx;
  const double bot = img[size_t(y1) * w + x0] * (1 - fx) + img[size_t(y1) * w + x1] * fx;
  return float(top * (1 - fy) + bot * fy);
}

// out(p) = in(p + shift(p) * n(p)), n = the unit gradient of sd (pointing inward): shift > 0
// moves the edge outward (grows), < 0 inward, and the profile across the edge travels with it.
// Where sd has no usable gradient (beyond its band, |grad| < 0.5) or shift is 0, out = in.
// out may not be in.
void shiftAlongNormal(const float* in, const std::vector<float>& sd,
                      const std::vector<float>& shift, int w, int h, float* out);

// ---- The fast path of the edge operations (Shrink / Grow, Edge Width, Contour Smooth) ----

// The contour pixels of a side mask (w*h bytes, 0 / 1): pixels with a 4-neighbour in the frame
// on the other side, either side, row-major.
std::vector<int> contourPixels(const uint8_t* side, int w, int h, Workers* wk = nullptr);

// signedEdgeDistance's values without distanceWithin's switch to the plain grid distance at 2
// frames of writes: at the bands the edge operations need (up to 15 px) a 1080 frame with hair
// already crosses it, and the edge would move by whole pixels. Each band of rows stamps the
// contour pixels within reach into its own rows. Only when the stamping would cost more than 12
// frames of writes (a very noisy matte), signedEdgeDistance. below = in < level, contour =
// contourPixels(below); sd: w*h floats, all written.
void signedEdgeDistanceExact(const float* in, const uint8_t* below, const std::vector<int>& contour,
                             int w, int h, float level, double band, float* sd, Workers* wk = nullptr);

// signedEdgeDistanceExact in a new buffer (w*h floats).
std::unique_ptr<float[]> signedEdgeDistanceThreaded(const float* in, int w, int h, float level,
                                                    double band);

// The unit normal of sd at pixel (x, y), pointing inward (up the distance): central differences,
// or where they cancel (the middle of a thin part or gap) the steeper one-sided difference per
// axis. False where sd is flat (beyond its band).
inline bool edgeNormal(const float* sd, int w, int h, int x, int y, double& nx, double& ny) {
  const size_t i = size_t(y) * w + x;
  double g[2];
  for (int axis = 0; axis < 2; ++axis) {
    const bool lo = axis ? y > 0 : x > 0, hi = axis ? y + 1 < h : x + 1 < w;
    const size_t step = axis ? size_t(w) : 1;
    const double dl = lo ? sd[i] - sd[i - step] : 0.0, dr = hi ? sd[i + step] - sd[i] : 0.0;
    g[axis] = lo && hi ? (dl + dr) / 2 : dl + dr;
  }
  double len = std::hypot(g[0], g[1]);
  if (len < 0.5) {  // a ridge: the two sides cancel, either side's direction will do
    for (int axis = 0; axis < 2; ++axis) {
      const bool lo = axis ? y > 0 : x > 0, hi = axis ? y + 1 < h : x + 1 < w;
      const size_t step = axis ? size_t(w) : 1;
      const double dl = lo ? sd[i] - sd[i - step] : 0.0, dr = hi ? sd[i + step] - sd[i] : 0.0;
      g[axis] = std::fabs(dl) >= std::fabs(dr) ? dl : dr;
    }
    len = std::hypot(g[0], g[1]);
    if (len < 0.5) return false;
  }
  nx = g[0] / len;
  ny = g[1] / len;
  return true;
}

// The matte's value at signed distance `target` along the normal through (x, y), read near
// (x, y) + (target - sd) * n as shiftAlongNormal does: of the pixels there, the two whose sd
// brackets target most tightly, interpolated linearly; beyond the profile, its last value. Two
// changes: the window is 3 px across the normal's main axis and 7 along the other (not 3x3),
// and a read past the middle of a thin part stops at the middle (its peak).
float profileAt(const float* in, const float* sd, int w, int h, int x, int y,
                double nx, double ny, double target);

// A pixel of the 0.5 contour: a 4-neighbour on the other side of 0.5.
inline bool isContour(const float* in, int w, int h, int x, int y) {
  const size_t i = size_t(y) * w + x;
  const bool a = in[i] >= 0.5f;
  return (x > 0 && (in[i - 1] >= 0.5f) != a) || (x + 1 < w && (in[i + 1] >= 0.5f) != a) ||
         (y > 0 && (in[i - w] >= 0.5f) != a) || (y + 1 < h && (in[i + w] >= 0.5f) != a);
}

// The profile's slope across the contour at contour pixel (x, y), per px of sd: from the
// 4-neighbour across 0.5 farthest in sd (exact on a linear ramp; on a hard step, the step
// itself). 0 if there is none.
double contourSlope(const float* in, const float* sd, int w, int h, int x, int y);

// The tent-weighted (1.5 px) mean of per-contour-pixel measures around (fx, fy) (a pixel's foot
// on the contour), so they vary smoothly along the edge. measure(qx, qy) = the k floats of
// contour pixel (qx, qy), nullptr if it is none; a first float < 0 means no measure. Returns
// the weight sum (0 = no measure nearby); mean gets k values.
template <class Measure>
double footMeanAt(int k, int w, int h, double fx, double fy, double* mean, Measure measure) {
  double sum = 0.0;
  for (int j = 0; j < k; ++j) mean[j] = 0.0;
  for (int qy = std::max(0, int(std::floor(fy)) - 1); qy <= std::min(h - 1, int(std::floor(fy)) + 2); ++qy)
    for (int qx = std::max(0, int(std::floor(fx)) - 1); qx <= std::min(w - 1, int(std::floor(fx)) + 2); ++qx) {
      const float* v = measure(qx, qy);
      if (!v || v[0] < 0.0f) continue;
      const double t = std::max(0.0, 1.5 - std::fabs(qx - fx)) * std::max(0.0, 1.5 - std::fabs(qy - fy));
      sum += t;
      for (int j = 0; j < k; ++j) mean[j] += t * v[j];
    }
  if (sum > 0.0)
    for (int j = 0; j < k; ++j) mean[j] /= sum;
  return sum;
}

// footMeanAt with the measures in a w*h*k array m, read only at contour pixels of in (isContour).
double footMean(const float* in, const float* m, int k, int w, int h, double fx, double fy, double* mean);

// Pixel -> its index in a contour pixel list (-1: not in it), in O(1), with memory only for the
// 16 x 16 tiles the contour crosses: per-contour-pixel measures can then be stored densely
// (a w*h array of them touched only at the contour costs page faults all over the frame).
class ContourIndex {
 public:
  ContourIndex(const std::vector<int>& contour, int w, int h);
  int at(int x, int y) const {
    const int s = slot_[size_t(y >> 4) * tw_ + (x >> 4)];
    return s < 0 ? -1 : index_[(size_t(s) << 8) + ((y & 15) << 4) + (x & 15)];
  }

 private:
  int tw_;
  std::vector<int> slot_, index_;
};

// The tiles (kTile px square) within `reach` px along each axis of a listed pixel: where a
// computation confined to the edge band has to look; elsewhere it can take its default.
class NearTiles {
 public:
  static constexpr int kTile = 32;
  NearTiles(const std::vector<int>& pixels, int w, int h, int reach);
  // fn(x0, x1, near) for row y's runs of near / far tiles, left to right, x1 exclusive.
  template <class F>
  void runs(int y, F fn) const {
    const uint8_t* row = &near_[size_t(y / kTile) * tw_];
    for (int t = 0; t < tw_;) {
      int e = t + 1;
      while (e < tw_ && row[e] == row[t]) ++e;
      fn(t * kTile, std::min(e * kTile, w_), row[t] != 0);
      t = e;
    }
  }

 private:
  int w_, tw_;
  std::vector<uint8_t> near_;
};

}  // namespace mr
