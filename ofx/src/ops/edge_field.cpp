#include "ops/edge_field.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace mr {

namespace {

// Direction from central differences of the matte (exact on a ramp, whichever neighbours are
// across; using only the across neighbours tilted the normal by up to 30 deg on slants);
// distance from the one-sided crossings (exact on a hard 1 px edge), each across axis giving
// t * |n_axis|, weighted by |n_axis|.
template <class Side>
ContourFoot footAt(const float* in, const Side& side, int w, int h, float level, int i) {
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

}  // namespace

ContourFoot contourFoot(const float* in, const Mask& side, int w, int h, float level, int i) {
  return footAt(in, side, w, h, level, i);
}

ContourFoot contourFoot(const float* in, const uint8_t* side, int w, int h, float level, int i) {
  return footAt(in, side, w, h, level, i);
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

// Row bands on threads. Most of a matte is far from its contour: 8 pixels at a time are
// skipped when each equals its 4 neighbours (the masks hold 0 / 1, so a word compare does it).
std::vector<int> contourPixels(const uint8_t* s, int w, int h, Workers* wk) {
  constexpr int kRows = 32;
  const int bands = std::max(0, (h + kRows - 1) / kRows);
  std::vector<std::vector<int>> found(static_cast<size_t>(bands));
  parallelFor(wk, bands, 1, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b) {
      std::vector<int>& out = found[size_t(b)];
      for (int y = b * kRows; y < std::min(h, (b + 1) * kRows); ++y) {
        const uint8_t* row = s + size_t(y) * w;
        const uint8_t* up = y > 0 ? row - w : row;  // beyond the frame: the pixel itself (no edge)
        const uint8_t* dn = y + 1 < h ? row + w : row;
        auto test = [&](int x) {
          const uint8_t c = row[x];
          if ((x > 0 && row[x - 1] != c) || (x + 1 < w && row[x + 1] != c) || up[x] != c || dn[x] != c)
            out.push_back(int(size_t(y) * w + x));
        };
        int x = 0;
        if (w > 0) test(x++);
        for (; x + 8 < w; x += 8) {
          uint64_t c, l, r, u, d;
          std::memcpy(&c, row + x, 8);
          std::memcpy(&l, row + x - 1, 8);
          std::memcpy(&r, row + x + 1, 8);
          std::memcpy(&u, up + x, 8);
          std::memcpy(&d, dn + x, 8);
          if (((c ^ l) | (c ^ r) | (c ^ u) | (c ^ d)) == 0) continue;
          for (int k = x; k < x + 8; ++k) test(k);
        }
        for (; x < w; ++x) test(x);
      }
    }
  });
  size_t total = 0;
  for (const auto& f : found) total += f.size();
  std::vector<int> edge;
  edge.reserve(total);
  for (const auto& f : found) edge.insert(edge.end(), f.begin(), f.end());
  return edge;
}

// Each pixel measures to the contour pixels on the other side only (signedEdgeDistance: above
// pixels to the contour near below pixels, and the reverse); squared, as distanceWithin. A band
// stamps below and above contour pixels into two buffers, every pixel of the disc (straight
// loops that vectorise), and each pixel then reads the buffer of the other side: the same
// minimum over the same contour pixels as testing the side per pixel.
void signedEdgeDistanceExact(const float* in, const uint8_t* below, const std::vector<int>& edge,
                             int w, int h, float level, double band, float* sd, Workers* wk) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  // The disc of morphology.cpp distanceWithin (radius band + 1.5), as a half-width per row.
  const double r = band + 1.5;
  const int ri = int(std::floor(r));
  std::vector<int> half(size_t(2 * ri + 1));
  long long disc = 0;
  for (int dy = -ri; dy <= ri; ++dy) {
    half[size_t(dy + ri)] = int(std::floor(std::sqrt(std::max(0.0, r * r - double(dy) * dy))));
    disc += 2 * half[size_t(dy + ri)] + 1;
  }
  if (double(edge.size()) * double(disc) > 12.0 * double(n)) {
    const std::vector<float> s = signedEdgeDistance(in, w, h, level, band);
    parallelFor(wk, h, 16, [&](int y0, int y1) {
      std::copy(s.begin() + ptrdiff_t(y0) * w, s.begin() + ptrdiff_t(y1) * w, sd + size_t(y0) * w);
    });
    return;
  }
  std::vector<ContourFoot> feet(edge.size());
  parallelFor(wk, int(edge.size()), 256, [&](int e0, int e1) {
    for (int e = e0; e < e1; ++e) feet[size_t(e)] = contourFoot(in, below, w, h, level, edge[size_t(e)]);
  });
  constexpr float kFar = 1e20f;
  const float cap = float(band + 1.0);
  constexpr int kRows = 16;
  dynamicFor<std::vector<float>>(wk, (h + kRows - 1) / kRows, [&](int b, std::vector<float>& acc) {
    const int y0 = b * kRows, y1 = std::min(h, y0 + kRows);
    const size_t rows = size_t(y1 - y0) * w;
    acc.resize(size_t(2 * kRows) * w);
    float* fromBelow = acc.data();  // the distance for above pixels
    float* fromAbove = fromBelow + size_t(kRows) * w;
    std::fill(fromBelow, fromBelow + rows, kFar);
    std::fill(fromAbove, fromAbove + rows, kFar);
    const auto first = std::lower_bound(edge.begin(), edge.end(), std::max(0, y0 - ri) * w);
    const auto last = std::lower_bound(edge.begin(), edge.end(), std::min(h, y1 + ri) * w);
    for (auto it = first; it != last; ++it) {
      const int q = *it, qx = q % w, qy = q / w;
      float* acc0 = below[size_t(q)] ? fromBelow : fromAbove;
      const ContourFoot& f = feet[size_t(it - edge.begin())];
      const bool line = f.nx != 0.0f || f.ny != 0.0f;
      for (int py = std::max(y0, qy - ri); py < std::min(y1, qy + ri + 1); ++py) {
        const int hw = half[size_t(py - qy + ri)];
        stampRow(acc0 + size_t(py - y0) * w, std::max(0, qx - hw), std::min(w - 1, qx + hw), qx, py - qy, f, line);
      }
    }
    const uint8_t* bl = below + size_t(y0) * w;
    float* o = sd + size_t(y0) * w;
    for (size_t k = 0; k < rows; ++k) {
      const float d2 = bl[k] ? fromAbove[k] : fromBelow[k];
      const float d = d2 < kFar ? std::sqrt(d2) : kFar;
      o[k] = bl[k] ? -std::min(d, cap) : std::min(d, cap);
    }
  });
}

std::unique_ptr<float[]> signedEdgeDistanceThreaded(const float* in, int w, int h, float level,
                                                    double band) {
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  std::unique_ptr<uint8_t[]> below(new uint8_t[n]);
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) below[i] = in[i] < level;
  });
  std::unique_ptr<float[]> sd(new float[n]);
  signedEdgeDistanceExact(in, below.get(), contourPixels(below.get(), w, h), w, h, level, band, sd.get());
  return sd;
}

float profileAt(const float* in, const float* sd, int w, int h, int x, int y, double nx, double ny,
                double target) {
  const double s0 = sd[size_t(y) * w + x], move = target - s0;
  double px = x + move * nx, py = y + move * ny;
  // Past the middle of a thin part (or gap) the distance falls again, and the window would read
  // the far side or the next strand: then walk the normal in half pixels and read where the
  // distance reaches the target, or at the middle (the part's peak) if it never does.
  if (std::fabs(sampleBilinear(sd, w, h, px, py) - target) > 0.75) {
    const double dir = move > 0 ? 1.0 : -1.0;
    const int steps = int(std::ceil(2.0 * std::fabs(move)));
    double best = s0, bestT = 0.0;
    for (int k = 1; k <= steps; ++k) {
      const double t = std::min(0.5 * k, std::fabs(move));
      const double v = sampleBilinear(sd, w, h, x + dir * t * nx, y + dir * t * ny);
      if (dir * (v - best) > 0) best = v, bestT = t;
      if (dir * (v - target) >= 0) break;
    }
    if (dir * (best - target) < 0) target = best;
    px = x + dir * bestT * nx;
    py = y + dir * bestT * ny;
  }
  const int cx = int(std::lround(px)), cy = int(std::lround(py));
  // 3 px across the normal's main axis and 7 along the other: near an axis-aligned edge the 3x3
  // pixels' sd lie ~0.8 px apart, and linear interpolation across the kink at the end of a ramp
  // was off by up to 0.065 (3 px ramp); the rows along the edge fill in between (0.03).
  const bool alongY = std::fabs(nx) >= std::fabs(ny);
  const int rx = alongY ? 1 : 3, ry = alongY ? 3 : 1;
  double loD = -1e30, hiD = 1e30;
  float loV = 0.0f, hiV = 0.0f;
  for (int dy = -ry; dy <= ry; ++dy)
    for (int dx = -rx; dx <= rx; ++dx) {
      const int qx = std::min(std::max(cx + dx, 0), w - 1), qy = std::min(std::max(cy + dy, 0), h - 1);
      const size_t j = size_t(qy) * w + qx;
      if (sd[j] <= target && sd[j] > loD) loD = sd[j], loV = in[j];
      if (sd[j] >= target && sd[j] < hiD) hiD = sd[j], hiV = in[j];
    }
  if (loD > -1e29 && hiD < 1e29)
    return hiD > loD ? float(loV + (hiV - loV) * (target - loD) / (hiD - loD)) : loV;
  return loD > -1e29 ? loV : hiD < 1e29 ? hiV : in[size_t(y) * w + x];
}

double contourSlope(const float* in, const float* sd, int w, int h, int x, int y) {
  const size_t i = size_t(y) * w + x;
  const bool above = in[i] >= 0.5f;
  double bestDs = 0.0, slope = 0.0;
  const int nb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  for (const auto& o : nb) {
    const int qx = x + o[0], qy = y + o[1];
    if (qx < 0 || qy < 0 || qx >= w || qy >= h) continue;
    const size_t j = size_t(qy) * w + qx;
    const double ds = std::fabs(double(sd[i]) - sd[j]);
    if ((in[j] >= 0.5f) != above && ds > bestDs) {
      bestDs = ds;
      slope = std::fabs(double(in[i]) - in[j]) / ds;
    }
  }
  return bestDs < 0.05 ? 0.0 : slope;
}

double footMean(const float* in, const float* m, int k, int w, int h, double fx, double fy, double* mean) {
  return footMeanAt(k, w, h, fx, fy, mean, [&](int qx, int qy) -> const float* {
    return isContour(in, w, h, qx, qy) ? m + (size_t(qy) * w + qx) * k : nullptr;
  });
}

ContourIndex::ContourIndex(const std::vector<int>& contour, int w, int h)
    : tw_((w + 15) >> 4), slot_(size_t(tw_) * ((h + 15) >> 4), -1) {
  int tiles = 0;
  for (int p : contour) {
    int& s = slot_[size_t((p / w) >> 4) * tw_ + ((p % w) >> 4)];
    if (s < 0) s = tiles++;
  }
  index_.assign(size_t(tiles) << 8, -1);
  for (size_t c = 0; c < contour.size(); ++c) {
    const int x = contour[c] % w, y = contour[c] / w;
    index_[(size_t(slot_[size_t(y >> 4) * tw_ + (x >> 4)]) << 8) + ((y & 15) << 4) + (x & 15)] = int(c);
  }
}

NearTiles::NearTiles(const std::vector<int>& pixels, int w, int h, int reach)
    : w_(w), tw_((w + kTile - 1) / kTile) {
  const int th = (h + kTile - 1) / kTile, rt = (std::max(reach, 0) + kTile - 1) / kTile;
  std::vector<uint8_t> hit(size_t(tw_) * th, 0), across(hit.size(), 0);
  for (int p : pixels) hit[size_t(p / w / kTile) * tw_ + (p % w) / kTile] = 1;
  for (int ty = 0; ty < th; ++ty)
    for (int tx = 0; tx < tw_; ++tx)
      if (hit[size_t(ty) * tw_ + tx])
        for (int x = std::max(0, tx - rt); x <= std::min(tw_ - 1, tx + rt); ++x) across[size_t(ty) * tw_ + x] = 1;
  near_.assign(hit.size(), 0);
  for (int ty = 0; ty < th; ++ty)
    for (int tx = 0; tx < tw_; ++tx)
      if (across[size_t(ty) * tw_ + tx])
        for (int y = std::max(0, ty - rt); y <= std::min(th - 1, ty + rt); ++y) near_[size_t(y) * tw_ + tx] = 1;
}

}  // namespace mr
