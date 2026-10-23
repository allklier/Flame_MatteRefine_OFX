// Helpers for the equivalence tests (the current operations against old_edge_ops.h): random
// mattes with what the edge code has to handle (soft and hard edges, ties on axis-aligned edges,
// thin strands, holes, haze, noise, parts cut by the frame), and a bitwise compare.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace eq {

// Pixels whose float bits differ (NaN-safe, -0 vs +0 counts).
inline long long bitsDiffer(const float* a, const float* b, size_t n) {
  long long d = 0;
  for (size_t i = 0; i < n; ++i) d += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return d;
}

inline std::vector<float> randomMatte(std::mt19937& rng, int w, int h) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<float> m(size_t(w) * h, 0.0f);
  const double haze = u(rng) < 0.5 ? 0.0 : 0.1 * u(rng);
  for (float& v : m) v = float(haze);
  auto put = [&](int x, int y, double a, bool hole) {
    float& v = m[size_t(y) * w + x];
    v = hole ? std::min(v, float(1.0 - a)) : std::max(v, float(a));
  };
  const int shapes = 1 + int(u(rng) * 6);
  for (int k = 0; k < shapes; ++k) {
    const double cx = u(rng) * w, cy = u(rng) * h, r = 1.0 + u(rng) * 0.45 * std::min(w, h);
    const double soft = u(rng) < 0.3 ? 0.0 : 0.3 + u(rng) * 8.0;  // 0 = hard
    const bool rect = u(rng) < 0.3, hole = k > 0 && u(rng) < 0.25;
    const double level = u(rng) < 0.7 ? 1.0 : 0.5 + 0.5 * u(rng);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double d = rect ? std::max(std::fabs(x - cx), std::fabs(y - cy)) - r : std::hypot(x - cx, y - cy) - r;
        const double a = soft == 0.0 ? (d <= 0.0 ? 1.0 : 0.0) : std::min(1.0, std::max(0.0, 0.5 - d / soft));
        if (a > 0.0) put(x, y, a * level, hole);
      }
  }
  const int strands = int(u(rng) * 8);
  for (int k = 0; k < strands; ++k) {  // hair: thin lines, some above 0.5
    const double x0 = u(rng) * w, y0 = u(rng) * h, ang = u(rng) * 6.2832, len = 5 + u(rng) * std::max(w, h);
    const double wd = 0.4 + u(rng) * 2.0, a = 0.2 + 0.8 * u(rng);
    for (double t = 0; t < len; t += 0.5) {
      const int x = int(x0 + t * std::cos(ang)), y = int(y0 + t * std::sin(ang));
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          if (x + dx >= 0 && y + dy >= 0 && x + dx < w && y + dy < h && std::hypot(dx, dy) <= wd)
            put(x + dx, y + dy, a, false);
    }
  }
  if (u(rng) < 0.4) {  // noise and specks
    const double amp = 0.08 * u(rng), specks = 0.01 * u(rng);
    for (float& v : m) {
      v = float(std::min(1.0, std::max(0.0, v + amp * (u(rng) - 0.5))));
      if (u(rng) < specks) v = float(u(rng));
    }
  }
  if (u(rng) < 0.2)  // exact levels: ties
    for (float& v : m) v = std::round(v * 4.0f) / 4.0f;
  return m;
}

}  // namespace eq
