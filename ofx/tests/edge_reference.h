// Brute-force reference of the sub-pixel edge model (ops/edge_field contourFoot, morphology
// distanceWithin), written out independently for the unit tests: slow, obvious, no shortcuts.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace ref {

struct Foot {
  double fx = 0, fy = 0, nx = 0, ny = 0;
};

// The contour near pixel (x, y): normal from central differences, pointing to the other side;
// distance = the one-sided crossings towards across 4-neighbours, t * |n_axis| weighted by |n_axis|.
inline Foot foot(const std::vector<float>& in, int w, int h, float level, int x, int y) {
  auto at = [&](int ax, int ay) { return double(in[size_t(ay) * w + ax]); };
  const bool below = at(x, y) < level;
  const double a = at(x, y);
  double g[2], t[2] = {-1, -1}, across[2] = {0, 0};
  for (int axis = 0; axis < 2; ++axis) {
    const int lx = axis ? x : std::max(x - 1, 0), ly = axis ? std::max(y - 1, 0) : y;
    const int hx = axis ? x : std::min(x + 1, w - 1), hy = axis ? std::min(y + 1, h - 1) : y;
    const int span = (hx - lx) + (hy - ly);
    g[axis] = span > 0 ? (at(hx, hy) - at(lx, ly)) / span : 0.0;
    for (int sg = -1; sg <= 1; sg += 2) {
      const int nx = x + (axis ? 0 : sg), ny = y + (axis ? sg : 0);
      if (nx < 0 || ny < 0 || nx >= w || ny >= h || (at(nx, ny) < level) == below) continue;
      const double tc = (level - a) / (at(nx, ny) - a);
      if (t[axis] < 0 || tc < t[axis]) t[axis] = tc, across[axis] = sg;
    }
  }
  Foot f;
  if (across[0] == 0 && across[1] == 0) return f;
  double len = std::hypot(g[0], g[1]);
  if (len == 0) g[0] = across[0], g[1] = across[1], len = std::hypot(g[0], g[1]);
  const double sign = below ? 1.0 : -1.0;
  f.nx = sign * g[0] / len;
  f.ny = sign * g[1] / len;
  const double n[2] = {f.nx, f.ny};
  double sum = 0, wsum = 0;
  for (int axis = 0; axis < 2; ++axis)
    if (across[axis] != 0 && across[axis] * n[axis] > 0) {
      sum += t[axis] * n[axis] * n[axis];
      wsum += std::fabs(n[axis]);
    }
  const double s = wsum > 0 ? std::min(1.0, sum / wsum) : 0.0;
  f.fx = f.nx * s;
  f.fy = f.ny * s;
  return f;
}

// A pixel on the given side with a 4-neighbour on the other side, and its foot.
struct Boundary {
  int x, y;
  Foot f;
};

inline std::vector<Boundary> boundary(const std::vector<float>& in, int w, int h, float level,
                                      bool belowSide) {
  std::vector<Boundary> out;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const bool below = in[size_t(y) * w + x] < level;
      if (below != belowSide) continue;
      bool edge = false;
      const int nb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
      for (const auto& o : nb) {
        const int nx = x + o[0], ny = y + o[1];
        if (nx >= 0 && ny >= 0 && nx < w && ny < h) edge |= (in[size_t(ny) * w + nx] < level) != below;
      }
      if (edge) out.push_back({x, y, foot(in, w, h, level, x, y)});
    }
  return out;
}

// Distance from pixel (x, y) to the contour modelled at the boundary pixels: a tangent segment
// (+-0.5 px) through each foot point, capped at the grid distance to that pixel.
inline double distance(const std::vector<Boundary>& bs, int x, int y) {
  double best = 1e30;
  for (const Boundary& b : bs) {
    const double ox = x - b.x, oy = y - b.y, grid = ox * ox + oy * oy;
    double d2 = grid;
    if (b.f.nx != 0 || b.f.ny != 0) {
      const double vx = ox - b.f.fx, vy = oy - b.f.fy;
      const double perp = vx * b.f.nx + vy * b.f.ny;
      const double along = std::max(0.0, std::fabs(vy * b.f.nx - vx * b.f.ny) - 0.5);
      d2 = std::min(grid, perp * perp + along * along);
    }
    best = std::min(best, d2);
  }
  return std::sqrt(best);
}

}  // namespace ref
