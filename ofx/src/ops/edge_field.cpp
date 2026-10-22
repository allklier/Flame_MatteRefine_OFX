#include "ops/edge_field.h"

#include <algorithm>
#include <cmath>

namespace mr {

// Direction from central differences of the matte (exact on a ramp, whichever neighbours are
// across; using only the across neighbours tilted the normal by up to 30 deg on slants);
// distance from the one-sided crossings (exact on a hard 1 px edge), each across axis giving
// t * |n_axis|, weighted by |n_axis|.
ContourFoot contourFoot(const float* in, const Mask& side, int w, int h, float level, int i) {
  const int x = i % w, y = i / w;
  const double a = in[i];
  double g[2], t[2] = {-1.0, -1.0}, across[2] = {0.0, 0.0};
  for (int axis = 0; axis < 2; ++axis) {
    const int lo = axis ? (y > 0 ? i - w : i) : (x > 0 ? i - 1 : i);
    const int hi = axis ? (y + 1 < h ? i + w : i) : (x + 1 < w ? i + 1 : i);
    const int span = (hi - lo) / (axis ? w : 1);
    g[axis] = span > 0 ? (in[hi] - in[lo]) / span : 0.0;
    for (int sg = -1; sg <= 1; sg += 2) {
      const int nx = x + (axis ? 0 : sg), ny = y + (axis ? sg : 0);
      if (nx < 0 || ny < 0 || nx >= w || ny >= h || side[size_t(ny) * w + nx] == side[size_t(i)])
        continue;
      const double b = in[size_t(ny) * w + nx];
      const double tc = (level - a) / (b - a);  // 0-1: level lies between a and b
      if (t[axis] < 0.0 || tc < t[axis]) t[axis] = tc, across[axis] = sg;
    }
  }
  ContourFoot f;
  if (across[0] == 0.0 && across[1] == 0.0) return f;
  double len = std::hypot(g[0], g[1]);
  if (len == 0.0) {  // flat around the centre (a 1 px line): the across neighbours' directions
    g[0] = across[0];
    g[1] = across[1];
    len = std::hypot(g[0], g[1]);
  }
  // n points away from this pixel's side: up the matte for a below pixel, down for an above one.
  const double sign = side[size_t(i)] ? 1.0 : -1.0;
  const double n[2] = {sign * g[0] / len, sign * g[1] / len};
  double sum = 0.0, wsum = 0.0;
  for (int axis = 0; axis < 2; ++axis)
    if (across[axis] != 0.0 && across[axis] * n[axis] > 0.0) {
      const double wa = std::fabs(n[axis]);
      sum += t[axis] * wa * wa;
      wsum += wa;
    }
  const double s = wsum > 0.0 ? std::min(1.0, sum / wsum) : 0.0;
  f.nx = float(n[0]);
  f.ny = float(n[1]);
  f.fx = float(n[0] * s);
  f.fy = float(n[1] * s);
  return f;
}

std::vector<float> signedEdgeDistance(const float* in, int w, int h, float level, double band) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  Mask below(n), above(n);
  for (size_t i = 0; i < n; ++i) {
    below[i] = in[i] < level;
    above[i] = !below[i];
  }
  // Inside pixels measure to the contour near the below pixels, and the reverse.
  auto foot = [&](int i) { return contourFoot(in, below, w, h, level, i); };
  const std::vector<float> dIn = distanceWithin(below, w, h, band, foot);
  const std::vector<float> dOut = distanceWithin(above, w, h, band, foot);
  const float cap = float(band + 1.0);
  std::vector<float> sd(n);
  for (size_t i = 0; i < n; ++i) sd[i] = above[i] ? std::min(dIn[i], cap) : -std::min(dOut[i], cap);
  return sd;
}

float sampleBilinear(const float* img, int w, int h, double x, double y) {
  x = std::min(std::max(x, 0.0), double(w - 1));
  y = std::min(std::max(y, 0.0), double(h - 1));
  const int x0 = std::min(int(x), w - 2 < 0 ? 0 : w - 2), y0 = std::min(int(y), h - 2 < 0 ? 0 : h - 2);
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const double fx = x - x0, fy = y - y0;
  const double top = img[size_t(y0) * w + x0] * (1 - fx) + img[size_t(y0) * w + x1] * fx;
  const double bot = img[size_t(y1) * w + x0] * (1 - fx) + img[size_t(y1) * w + x1] * fx;
  return float(top * (1 - fy) + bot * fy);
}

// The value at p + shift * n is read in distance space: the matte depends mainly on the
// distance to the contour, which sd knows to ~0.05 px, while bilinear sampling of a 1 px edge
// (a ramp with kinks) is off by up to 0.25. Of the 3x3 pixels around the sample point, the two
// whose sd brackets the target sd(p) + shift most tightly are interpolated linearly.
void shiftAlongNormal(const float* in, const std::vector<float>& sd,
                      const std::vector<float>& shift, int w, int h, float* out) {
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = size_t(y) * w + x;
      out[i] = in[i];
      if (shift[i] == 0.0f) continue;
      // Central differences, one-sided at the frame border.
      const int xl = x > 0 ? x - 1 : x, xr = x + 1 < w ? x + 1 : x;
      const int yu = y > 0 ? y - 1 : y, yd = y + 1 < h ? y + 1 : y;
      const double gx = xr > xl ? (sd[size_t(y) * w + xr] - sd[size_t(y) * w + xl]) / (xr - xl) : 0.0;
      const double gy = yd > yu ? (sd[size_t(yd) * w + x] - sd[size_t(yu) * w + x]) / (yd - yu) : 0.0;
      const double g = std::hypot(gx, gy);
      if (g < 0.5) continue;
      const double target = sd[i] + shift[i];
      const int cx = int(std::lround(x + shift[i] * gx / g)), cy = int(std::lround(y + shift[i] * gy / g));
      double loD = -1e30, hiD = 1e30;
      float loV = 0.0f, hiV = 0.0f;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int qx = std::min(std::max(cx + dx, 0), w - 1), qy = std::min(std::max(cy + dy, 0), h - 1);
          const size_t j = size_t(qy) * w + qx;
          if (sd[j] <= target && sd[j] > loD) loD = sd[j], loV = in[j];
          if (sd[j] >= target && sd[j] < hiD) hiD = sd[j], hiV = in[j];
        }
      if (loD > -1e29 && hiD < 1e29)
        out[i] = hiD > loD ? float(loV + (hiV - loV) * (target - loD) / (hiD - loD)) : loV;
      else
        out[i] = loD > -1e29 ? loV : hiD < 1e29 ? hiV : in[i];  // beyond the profile: flat
    }
}

}  // namespace mr
