// Binary morphology shared by the operations: masks are w*h bytes, row 0 = the top row.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
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

// Threads for the parallel loops of one operation call: started once and handed each loop,
// joined when it goes out of scope. Starting threads costs ~0.3 ms per loop at 24 threads
// (BFD, 2026-10-23), and an edge operation runs 5-15 loops per pass. As many threads as
// parallelFor uses; nothing outlives the call (no threads left behind in the host).
class Workers {
 public:
  Workers();
  ~Workers();
  Workers(const Workers&) = delete;
  Workers& operator=(const Workers&) = delete;

  // As mr::parallelFor, the same split of [0, n), on these threads.
  template <class F>
  void parallelFor(int n, int minPerThread, F fn) {
    const int t = std::max(1, std::min(size(), n / minPerThread));
    if (t == 1) {
      fn(0, n);
      return;
    }
    run(t, [&](int k) { fn(n * k / t, n * (k + 1) / t); });
  }

  int size() const { return int(threads_.size()) + 1; }

 private:
  void run(int t, const std::function<void(int)>& job);  // job(k), k = 0 (here) to t - 1
  void loop(int k);
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable wake_, done_;
  const std::function<void(int)>* job_ = nullptr;
  int active_ = 0, pending_ = 0;
  unsigned generation_ = 0;
  bool stop_ = false;
};

// parallelFor on wk's threads, or on fresh ones without.
template <class F>
void parallelFor(Workers* wk, int n, int minPerThread, F fn) {
  if (wk)
    wk->parallelFor(n, minPerThread, fn);
  else
    parallelFor(n, minPerThread, fn);
}

// fn(i, scratch) for each i in [0, n), handed out one at a time to whichever thread is free:
// for items of uneven cost that write disjoint outputs. scratch: one default-constructed S per
// thread, kept across its items.
template <class S, class F>
void dynamicFor(Workers* wk, int n, F fn) {
  const int threads = wk ? wk->size() : std::min(int(std::thread::hardware_concurrency()), 32);
  const int t = std::max(1, std::min(threads, n));
  std::atomic<int> next{0};
  parallelFor(wk, t, 1, [&](int, int) {
    S scratch;
    for (int i = next++; i < n; i = next++) fn(i, scratch);
  });
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

// The same values written to d (w*h floats), for a mask given as w*h bytes, with the switch to
// the plain distance at budgetFrames frames' worth of writes (2 above), on wk's threads if given.
void distanceWithin(const uint8_t* m, int w, int h, double r,
                    const std::function<ContourFoot(int)>& foot, float* d,
                    double budgetFrames = 2.0, Workers* wk = nullptr);

// The stamping kernel of distanceWithin: acc[x] = min(acc[x], the squared distance from pixel
// (x, oy + qy) to the contour near boundary pixel (qx, qy) with foot f) for x in [x0, x1].
// line = f has a normal (else the plain distance). Straight loops, so they vectorise.
inline void stampRow(float* acc, int x0, int x1, int qx, int oyInt, const ContourFoot& f, bool line) {
  const float oy = float(oyInt);
  if (!line) {
    for (int x = x0; x <= x1; ++x) {
      const float ox = float(x - qx), grid = ox * ox + oy * oy;
      const float a = acc[x];
      acc[x] = grid < a ? grid : a;
    }
    return;
  }
  // The same values as std::min / std::max, written so GCC 13 vectorises the loop: value
  // selects (b < a ? b : a), and max(0, off) as (off + |off|) / 2, exact (off > 0: doubled and
  // halved; off <= 0: off + |off| = +0). A select there left a branch in the loop.
  const float fx = f.fx, fy = f.fy, nx = f.nx, ny = f.ny;
  for (int x = x0; x <= x1; ++x) {
    const float ox = float(x - qx), grid = ox * ox + oy * oy;
    const float vx = ox - fx, vy = oy - fy;
    const float perp = vx * nx + vy * ny;
    const float off = std::fabs(vy * nx - vx * ny) - 0.5f;
    const float along = (off + std::fabs(off)) * 0.5f;
    const float seg = perp * perp + along * along;
    const float v = seg < grid ? seg : grid;
    const float a = acc[x];
    acc[x] = v < a ? v : a;
  }
}

}  // namespace mr
