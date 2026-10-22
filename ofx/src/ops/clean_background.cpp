#include "ops/clean_background.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {

namespace {

enum : uint8_t { kNone = 0, kRemoved = 1, kKept = 2 };

// A1: the pixels to clear: the islands (8-connected components of `above` up to maxArea
// pixels) and their own fringe. Components are labelled as runs per row with union-find: a
// per-pixel flood over a UHD subject cost ~25 ms.
std::vector<int> findIslands(const float* in, const Mask& above, int w, int h,
                             const CleanBackgroundParams& p, CleanBackgroundStats& st) {
  const size_t n = above.size();
  struct Run {
    int y, x0, x1;
  };
  std::vector<Run> runs;
  std::vector<size_t> rowStart(size_t(h) + 1);
  for (int y = 0; y < h; ++y) {
    rowStart[y] = runs.size();
    const uint8_t* row = &above[size_t(y) * w];
    for (int x = 0; x < w; ++x) {
      if (!row[x]) continue;
      const int x0 = x;
      while (x + 1 < w && row[x + 1]) ++x;
      runs.push_back({y, x0, x});
    }
  }
  rowStart[h] = runs.size();
  std::vector<size_t> parent(runs.size());
  std::iota(parent.begin(), parent.end(), size_t(0));
  auto find = [&](size_t a) {
    while (parent[a] != a) a = parent[a] = parent[parent[a]];
    return a;
  };
  for (int y = 1; y < h; ++y) {  // runs of adjacent rows that touch, diagonally included
    size_t a = rowStart[y - 1], b = rowStart[y];
    while (a < rowStart[y] && b < rowStart[y + 1]) {
      if (runs[a].x0 <= runs[b].x1 + 1 && runs[b].x0 <= runs[a].x1 + 1) {
        const size_t ra = find(a), rb = find(b);
        if (ra != rb) parent[std::max(ra, rb)] = std::min(ra, rb);
      }
      (runs[a].x1 < runs[b].x1 ? a : b)++;
    }
  }
  std::vector<long long> area(runs.size(), 0);
  for (size_t r = 0; r < runs.size(); ++r) area[find(r)] += runs[r].x1 - runs[r].x0 + 1;
  const double maxArea = p.maxIslandPercent / 100.0 * double(n);
  for (size_t r = 0; r < runs.size(); ++r)
    if (find(r) == r) (double(area[r]) <= maxArea ? st.islandsRemoved : st.componentsKept)++;

  Mask state(n, kNone);
  std::vector<int> cleared;
  for (size_t r = 0; r < runs.size(); ++r) {
    const bool island = double(area[find(r)]) <= maxArea;
    const size_t row = size_t(runs[r].y) * w;
    std::fill(&state[row + runs[r].x0], &state[row + runs[r].x1] + 1, island ? kRemoved : kKept);
    if (island)
      for (int x = runs[r].x0; x <= runs[r].x1; ++x) cleared.push_back(int(row + x));
  }
  // The fringe: background pixels reached from a removed island by a path that never rises
  // (the island's soft falloff, not a neighbour's rising one), up to fringePx steps, and never
  // next to a kept component (that is its own falloff).
  auto nextToKept = [&](int x, int y) {
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx, ny = y + dy;
        if (nx >= 0 && ny >= 0 && nx < w && ny < h && state[size_t(ny) * w + nx] == kKept)
          return true;
      }
    return false;
  };
  size_t begin = 0;  // cleared[begin, end) = the previous step's pixels
  for (int step = 1; step <= p.fringePx && begin < cleared.size(); ++step) {
    const size_t end = cleared.size();
    for (size_t k = begin; k < end; ++k) {
      const int i = cleared[k], x = i % w, y = i / w;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const int j = ny * w + nx;
          if (state[j] != kNone || above[j] || in[j] <= 0.0f || in[j] > in[i] ||
              nextToKept(nx, ny))
            continue;
          state[j] = kRemoved;
          cleared.push_back(j);
        }
    }
    begin = end;
  }
  return cleared;
}

// A2, soft: morphology.h distanceWithin's exact (stamping) path, on threads, fused with the
// crush. Its stamping runs on one thread (~30 ms at UHD on the synthetic figure), so here each
// thread owns a band of rows: it finds the band's boundary pixels and their feet, then stamps
// every boundary pixel that reaches its rows into a band-local distance and crushes them. Same
// arithmetic and path choice as distanceWithin, so the same floats (checked against Solid
// Core); when the budget picks the distance transform (already threaded), distanceWithin
// itself. The threading belongs in morphology (Solid Core would gain too); it lives here
// because this op may not change morphology.
long long crushSoft(float* out, const Mask& above, int w, int h, const CleanBackgroundParams& p) {
  const size_t n = above.size();
  const double r = p.crushPx + p.featherPx, rs = r + 1.5;
  // Pixels at distance > Crush go towards 0 (smoothstep over Feather); 0 stays.
  auto crush = [&](const float* d, size_t i0, size_t i1) {
    long long changed = 0;
    for (size_t i = i0; i < i1; ++i) {
      const float dist = d[i - i0];
      if (dist <= p.crushPx || out[i] <= 0.0f) continue;  // the outer falloff, or already 0
      const double t = std::min(1.0, (dist - p.crushPx) / p.featherPx);
      const float o = std::min(out[i], float(1.0 - t * t * (3.0 - 2.0 * t)));
      changed += o != out[i];
      out[i] = o;
    }
    return changed;
  };
  // Sub-pixel distance to the contour from outside (decision 16): contourFoot takes the
  // below side, as signedEdgeDistance's outer distance does.
  Mask below(n);
  for (size_t i = 0; i < n; ++i) below[i] = !above[i];
  auto foot = [&](int i) { return contourFoot(out, below, w, h, p.backgroundBelow, i); };
  const int ri = int(std::floor(rs));
  std::vector<std::pair<int, int>> disc;
  for (int dy = -ri; dy <= ri; ++dy)
    for (int dx = -ri; dx <= ri; ++dx)
      if ((dx || dy) && dx * dx + dy * dy <= rs * rs) disc.emplace_back(dx, dy);
  const int threads =
      std::max(1, std::min({int(std::thread::hardware_concurrency()), 16, h / 64}));
  auto run = [&](const std::function<void(int, int, int)>& fn) {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(fn, t, h * t / threads, h * (t + 1) / threads);
    for (auto& th : pool) th.join();
  };
  // Phase 1: each band's boundary pixels (set, with an unset 4-neighbour) in row order, and
  // their feet. Nothing is written to out yet.
  std::vector<std::vector<int>> bandBoundary(threads);
  std::vector<std::vector<ContourFoot>> bandFeet(threads);
  run([&](int t, int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        if (above[i] && ((x > 0 && !above[i - 1]) || (x + 1 < w && !above[i + 1]) ||
                         (y > 0 && !above[i - w]) || (y + 1 < h && !above[i + w]))) {
          bandBoundary[t].push_back(int(i));
          bandFeet[t].push_back(foot(int(i)));
        }
      }
  });
  std::vector<int> boundary;
  std::vector<ContourFoot> feet;
  for (int t = 0; t < threads; ++t) {
    boundary.insert(boundary.end(), bandBoundary[t].begin(), bandBoundary[t].end());
    feet.insert(feet.end(), bandFeet[t].begin(), bandFeet[t].end());
  }
  if (threads == 1 || double(boundary.size()) * disc.size() > 2.0 * double(n)) {
    const std::vector<float> d = distanceWithin(above, w, h, r, foot);
    return crush(d.data(), 0, n);
  }
  // Phase 2: stamp and crush, each band its own rows only.
  std::vector<long long> changed(threads, 0);
  run([&](int t, int y0, int y1) {
    std::vector<float> d(size_t(y1 - y0) * w);
    const size_t i0 = size_t(y0) * w;
    for (size_t i = i0; i < size_t(y1) * w; ++i) d[i - i0] = above[i] ? 0.0f : 1e20f;
    const auto lo = std::lower_bound(boundary.begin(), boundary.end(), std::max(0, y0 - ri) * w);
    const auto hi = std::lower_bound(boundary.begin(), boundary.end(), std::min(h, y1 + ri) * w);
    for (auto it = lo; it != hi; ++it) {
      const int x = *it % w, y = *it / w;
      const ContourFoot& f = feet[size_t(it - boundary.begin())];
      const bool line = f.nx != 0.0f || f.ny != 0.0f;
      for (const auto& o : disc) {
        const int px = x + o.first, py = y + o.second;
        if (px < 0 || py < y0 || px >= w || py >= y1) continue;
        const float ox = float(o.first), oy = float(o.second), grid = ox * ox + oy * oy;
        float d2 = grid;
        if (line) {
          const float vx = ox - f.fx, vy = oy - f.fy;
          const float perp = vx * f.nx + vy * f.ny;
          const float along = std::max(0.0f, std::fabs(vy * f.nx - vx * f.ny) - 0.5f);
          d2 = std::min(grid, perp * perp + along * along);
        }
        float& v = d[size_t(py) * w + px - i0];
        v = std::min(v, d2);
      }
    }
    for (float& v : d) v = v < 1e20f ? std::sqrt(v) : v;
    changed[t] = crush(d.data(), i0, size_t(y1) * w);
  });
  return std::accumulate(changed.begin(), changed.end(), 0LL);
}

}  // namespace

void cleanBackground(const float* in, float* out, int w, int h, const CleanBackgroundParams& p,
                     CleanBackgroundStats* stats) {
  CleanBackgroundStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  Mask above(n);
  for (size_t i = 0; i < n; ++i) above[i] = in[i] >= p.backgroundBelow;
  if (out != in) std::copy(in, in + n, out);

  // A1 Clean Background: removed islands go to 0, fringe included.
  if (p.maxIslandPercent > 0.0 && n > 0)
    for (int i : findIslands(in, above, w, h, p, st)) {
      st.pixelsRemoved += out[i] > 0.0f;
      out[i] = 0.0f;
      above[i] = 0;  // Crush measures from what is left
    }

  // A2 Background Crush = 1 - Solid Core(1 - matte) with the level mirrored (decisions 15, 16):
  // everything more than Crush px outside the backgroundBelow contour goes to 0, with a
  // smoothstep over Feather px; the outer Crush px of falloff are untouched. Works on out (the
  // matte after A1).
  if (p.crushPx > 0.0 && n > 0) {
    if (p.featherPx > 0.0) {
      st.pixelsCrushed = crushSoft(out, above, w, h, p);
    } else {  // hard, as decision 15
      const Mask nearEdge = dilate(above, w, h, p.crushPx);
      for (size_t i = 0; i < n; ++i) {
        const bool crush = !nearEdge[i] && out[i] > 0.0f;
        st.pixelsCrushed += crush;
        if (crush) out[i] = 0.0f;
      }
    }
  }
  if (stats) *stats = st;
}

}  // namespace mr
