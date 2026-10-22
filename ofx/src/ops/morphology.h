// Binary morphology shared by the operations: masks are w*h bytes, row 0 = the top row.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace mr {

using Mask = std::vector<uint8_t>;

// Runs fn(begin, end) over [0, n) on up to 32 threads, at least minPerThread items each.
template <class F>
void parallelFor(int n, int minPerThread, F fn) {
  const int t =
      std::max(1, std::min({int(std::thread::hardware_concurrency()), 32, n / minPerThread}));
  if (t == 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> pool;
  for (int i = 0; i < t; ++i) pool.emplace_back(fn, n * i / t, n * (i + 1) / t);
  for (auto& th : pool) th.join();
}

// Pixels within Euclidean distance r of a set pixel of m (m included). Pixels outside the
// frame count as unset.
Mask dilate(const Mask& m, int w, int h, double r);

// Where the contour runs near a set pixel q, relative to q's centre (px): the foot point (fx, fy)
// and the unit normal (nx, ny) pointing away from q's side. All zero = the contour is q itself.
struct ContourFoot {
  float fx = 0.0f, fy = 0.0f, nx = 0.0f, ny = 0.0f;
};

// For each unset pixel p: the distance to the contour, modelled near each set pixel q with an
// unset 4-neighbour as a tangent segment (+-0.5 px) through q's foot point, and capped at the
// plain distance |p - q| (so never more than the grid distance). Set pixels get 0. foot(q) is
// called only for those boundary pixels. Exact (to the model) where it is at most r, larger
// than r elsewhere. When the exact search would cost more than 2 frames' worth of writes (a
// large r on a noisy matte), the plain distance (feet ignored) instead. Pixels outside the
// frame count as unset.
std::vector<float> distanceWithin(const Mask& m, int w, int h, double r,
                                  const std::function<ContourFoot(int)>& foot);

}  // namespace mr
