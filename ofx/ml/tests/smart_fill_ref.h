// Test helpers for Smart Fill (B3): an independent brute-force hole finder (per-pixel 4-connected
// flood fill, as decision 9 with Bridge Gaps 0), blur and closing for fake segmentations.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ops/morphology.h"

namespace sfref {

struct RefHole {
  std::vector<int> px;  // pixel indexes
  long long area = 0;
};

// Holes in raster order of their first pixel (the order smartFill reports them).
inline std::vector<RefHole> findHoles(const float* m, int w, int h, float thr, bool eightConnected = false) {
  const size_t n = size_t(w) * h;
  std::vector<int> comp(n, -1);
  std::vector<RefHole> holes;
  std::vector<int> stack, members;
  for (size_t s = 0; s < n; ++s) {
    if (m[s] >= thr || comp[s] >= 0) continue;
    members.clear();
    bool edge = false;
    comp[s] = 1;
    stack.push_back(int(s));
    while (!stack.empty()) {
      const int i = stack.back(), x = i % w, y = i / w;
      stack.pop_back();
      members.push_back(i);
      if (x == 0 || y == 0 || x == w - 1 || y == h - 1) edge = true;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy) continue;
          if (!eightConnected && dx && dy) continue;
          const int nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const size_t j = size_t(ny) * w + nx;
          if (m[j] >= thr || comp[j] >= 0) continue;
          comp[j] = 1;
          stack.push_back(int(j));
        }
    }
    if (edge) continue;
    RefHole r;
    r.px = members;
    std::sort(r.px.begin(), r.px.end());
    r.area = (long long)r.px.size();
    holes.push_back(std::move(r));
  }
  return holes;
}

inline void blur(std::vector<float>& v, int w, int h, double sigma) {
  if (sigma <= 0) return;
  const int r = std::max(1, int(std::ceil(3 * sigma)));
  std::vector<double> k(size_t(2 * r + 1));
  double sum = 0;
  for (int j = -r; j <= r; ++j) sum += k[size_t(j + r)] = std::exp(-0.5 * j * j / (sigma * sigma));
  for (double& e : k) e /= sum;
  std::vector<float> tmp(v.size());
  for (int pass = 0; pass < 2; ++pass) {
    const int n = pass ? h : w, lines = pass ? w : h;
    mr::parallelFor(lines, 16, [&](int l0, int l1) {
      for (int l = l0; l < l1; ++l)
        for (int i = 0; i < n; ++i) {
          double acc = 0;
          for (int j = -r; j <= r; ++j) {
            const int s = std::min(std::max(i + j, 0), n - 1);
            acc += k[size_t(j + r)] * (pass ? v[size_t(s) * w + l] : v[size_t(l) * w + s]);
          }
          (pass ? tmp[size_t(i) * w + l] : tmp[size_t(l) * w + i]) = float(acc);
        }
    });
    v.swap(tmp);
  }
}

// The truth subject (>= 0.5), closed by a disc of radius r px (gaps narrower than ~2r close: the
// known BiRefNet failure), blurred by sigma: a fake segmentation in [0, 1].
inline std::vector<float> fakeSegmentation(const std::vector<float>& truth, int w, int h, double closeR, double sigma) {
  mr::Mask s(truth.size());
  for (size_t i = 0; i < s.size(); ++i) s[i] = truth[i] >= 0.5f;
  if (closeR > 0) {
    mr::Mask d = mr::dilate(s, w, h, closeR);
    for (auto& b : d) b = !b;
    mr::Mask e = mr::dilate(d, w, h, closeR);
    for (size_t i = 0; i < s.size(); ++i) s[i] = !e[i];
  }
  std::vector<float> f(truth.size());
  for (size_t i = 0; i < f.size(); ++i) f[i] = s[i] ? 1.0f : 0.0f;
  blur(f, w, h, sigma);
  return f;
}

}  // namespace sfref
