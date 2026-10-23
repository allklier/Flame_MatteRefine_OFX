#include "ops/clean_background.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {

namespace {

enum : uint8_t { kNone = 0, kRemoved = 1, kKept = 2 };

struct Run {
  int y, x0, x1;
};

// The matte (in >= backgroundBelow) as a mask and as runs per row, rows in bands on threads;
// out = in on the way when they differ.
void aboveAndRuns(const float* in, float* out, int w, int h, float level, Mask& above,
                  std::vector<Run>& runs, std::vector<size_t>& rowStart) {
  rowStart.assign(size_t(h) + 1, 0);
  std::vector<std::pair<int, std::vector<Run>>> parts;
  std::mutex mu;
  parallelFor(h, 16, [&](int y0, int y1) {
    std::vector<Run> local;
    for (int y = y0; y < y1; ++y) {
      const size_t row0 = size_t(y) * w, before = local.size();
      const float* src = in + row0;
      uint8_t* row = &above[row0];
      for (int x = 0; x < w; ++x) row[x] = src[x] >= level;
      if (out != in) std::copy(src, src + w, out + row0);
      for (int x = 0; x < w; ++x) {
        if (!row[x]) continue;
        const int x0 = x;
        while (x + 1 < w && row[x + 1]) ++x;
        local.push_back({y, x0, x});
      }
      rowStart[size_t(y) + 1] = local.size() - before;  // a count for now
    }
    std::lock_guard<std::mutex> lock(mu);
    parts.emplace_back(y0, std::move(local));
  });
  for (int y = 0; y < h; ++y) rowStart[size_t(y) + 1] += rowStart[size_t(y)];
  runs.resize(rowStart[size_t(h)]);
  for (const auto& part : parts)
    std::copy(part.second.begin(), part.second.end(),
              runs.begin() + std::ptrdiff_t(rowStart[size_t(part.first)]));
}

// A1: the pixels to clear: the islands (8-connected components of `above` up to maxArea
// pixels) and their own fringe. Components are labelled as runs per row with union-find: a
// per-pixel flood over a UHD subject cost ~25 ms.
std::vector<int> findIslands(const float* in, const Mask& above, const std::vector<Run>& runs,
                             const std::vector<size_t>& rowStart, int w, int h,
                             const CleanBackgroundParams& p, CleanBackgroundStats& st) {
  const size_t n = above.size();
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

  for (size_t r = 0; r < runs.size(); ++r) parent[r] = find(r);  // flat: read on threads
  // The cleared list's order doesn't matter: each fringe step's set doesn't depend on it.
  std::unique_ptr<uint8_t[]> stateBuf(new uint8_t[n]);
  uint8_t* state = stateBuf.get();
  std::vector<int> cleared;
  parallelFor(h, 16, [&](int y0, int y1) {
    std::fill(state + size_t(y0) * w, state + size_t(y1) * w, uint8_t(kNone));
    for (size_t r = rowStart[size_t(y0)]; r < rowStart[size_t(y1)]; ++r) {
      const bool island = double(area[parent[r]]) <= maxArea;
      const size_t row = size_t(runs[r].y) * w;
      std::fill(&state[row + runs[r].x0], &state[row + runs[r].x1] + 1, island ? kRemoved : kKept);
    }
  });
  for (size_t r = 0; r < runs.size(); ++r)
    if (double(area[parent[r]]) <= maxArea) {
      const size_t row = size_t(runs[r].y) * w;
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

// A2, soft: morphology.h distanceWithin's exact (stamping) path, fused with the crush, so no
// frame-sized distance buffer: the rows go in bands of kBand to threads as they free up (the
// boundary is uneven over the frame); each band finds the boundary pixels within reach and
// stamps them into a band-local distance, only the disc's rows inside the band, then crushes.
// Same disc, boundary, per-offset arithmetic and path choice as distanceWithin, so the same
// floats (a minimum doesn't depend on the order); when its budget picks the distance transform,
// distanceWithin itself (checked against Solid Core and against the previous version).
long long crushSoft(float* out, Mask& above, int w, int h, const CleanBackgroundParams& p) {
  const size_t n = above.size();
  const double r = p.crushPx + p.featherPx, rs = r + 1.5;
  // Pixels at distance > Crush go towards 0 (smoothstep over Feather); 0 stays. d2 = the
  // squared distance (1e20 = none within reach).
  auto crush = [&](const float* d2, bool squared, size_t i0, size_t i1) {
    long long changed = 0;
    for (size_t i = i0; i < i1; ++i) {
      if (out[i] <= 0.0f) continue;  // already 0
      const float v = d2[i - i0];
      const float dist = squared && v < 1e20f ? std::sqrt(v) : v;
      if (dist <= p.crushPx) continue;  // the outer falloff
      const double t = std::min(1.0, (dist - p.crushPx) / p.featherPx);
      const float o = std::min(out[i], float(1.0 - t * t * (3.0 - 2.0 * t)));
      changed += o != out[i];
      out[i] = o;
    }
    return changed;
  };
  // Sub-pixel distance to the contour from outside (decision 16): contourFoot takes the
  // below side, as signedEdgeDistance's outer distance does. `above` is inverted in place
  // into it (no second frame-sized mask; the caller doesn't use it after this).
  Mask& below = above;
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) below[i] = !below[i];
  });
  auto foot = [&](int i) { return contourFoot(out, below, w, h, p.backgroundBelow, i); };
  // The disc (centre excluded) as one span of dx per dy: |dx| <= span[dy + ri].
  const int ri = int(std::floor(rs));
  std::vector<int> span(size_t(2 * ri + 1), -1);
  size_t discSize = 0;
  for (int dy = -ri; dy <= ri; ++dy)
    for (int dx = -ri; dx <= ri; ++dx)
      if (dx * dx + dy * dy <= rs * rs) {
        span[size_t(dy + ri)] = std::max(span[size_t(dy + ri)], dx);
        discSize += dx || dy;
      }
  // Boundary pixels (set, with an unset 4-neighbour) in row order, and their feet. Nothing is
  // written to out yet.
  constexpr int kBand = 32;
  const int bands = (h + kBand - 1) / kBand;
  std::vector<std::vector<int>> bandBoundary(static_cast<size_t>(bands));
  std::vector<std::vector<ContourFoot>> bandFeet(static_cast<size_t>(bands));
  parallelFor(bands, 1, [&](int b0, int b1) {
    for (int b = b0; b < b1; ++b)
      for (int y = b * kBand; y < std::min(h, (b + 1) * kBand); ++y)
        for (int x = 0; x < w; ++x) {
          const size_t i = size_t(y) * w + x;
          if (!below[i] && ((x > 0 && below[i - 1]) || (x + 1 < w && below[i + 1]) ||
                            (y > 0 && below[i - w]) || (y + 1 < h && below[i + w]))) {
            bandBoundary[size_t(b)].push_back(int(i));
            bandFeet[size_t(b)].push_back(foot(int(i)));
          }
        }
  });
  std::vector<int> boundary;
  std::vector<ContourFoot> feet;
  for (int b = 0; b < bands; ++b) {
    boundary.insert(boundary.end(), bandBoundary[size_t(b)].begin(), bandBoundary[size_t(b)].end());
    feet.insert(feet.end(), bandFeet[size_t(b)].begin(), bandFeet[size_t(b)].end());
  }
  if (double(boundary.size()) * discSize > 2.0 * double(n)) {
    Mask aboveCopy(n);
    for (size_t i = 0; i < n; ++i) aboveCopy[i] = !below[i];
    const std::vector<float> d = distanceWithin(aboveCopy, w, h, r, foot);
    std::atomic<long long> changed{0};
    parallelFor(h, 16, [&](int y0, int y1) {
      changed += crush(d.data() + size_t(y0) * w, false, size_t(y0) * w, size_t(y1) * w);
    });
    return changed;
  }
  // Stamp and crush, each band its own rows only.
  std::atomic<int> nextBand{0};
  std::atomic<long long> changed{0};
  parallelFor(bands, 1, [&](int, int) {
    std::vector<float> d(size_t(kBand) * w);
    long long c = 0;
    for (int b; (b = nextBand++) < bands;) {
      const int y0 = b * kBand, y1 = std::min(h, y0 + kBand);
      const size_t i0 = size_t(y0) * w;
      for (size_t i = i0; i < size_t(y1) * w; ++i) d[i - i0] = below[i] ? 1e20f : 0.0f;
      const auto lo = std::lower_bound(boundary.begin(), boundary.end(), std::max(0, y0 - ri) * w);
      const auto hi = std::lower_bound(boundary.begin(), boundary.end(), std::min(h, y1 + ri) * w);
      for (auto it = lo; it != hi; ++it) {
        const int x = *it % w, y = *it / w;
        const ContourFoot& f = feet[size_t(it - boundary.begin())];
        const bool line = f.nx != 0.0f || f.ny != 0.0f;
        for (int dy = std::max(-ri, y0 - y); dy <= std::min(ri, y1 - 1 - y); ++dy) {
          const int sx = span[size_t(dy + ri)];
          float* drow = &d[size_t(y + dy - y0) * w];
          for (int dx = std::max(-sx, -x); dx <= std::min(sx, w - 1 - x); ++dx) {
            if (!dx && !dy) continue;
            const float ox = float(dx), oy = float(dy), grid = ox * ox + oy * oy;
            float d2 = grid;
            if (line) {
              const float vx = ox - f.fx, vy = oy - f.fy;
              const float perp = vx * f.nx + vy * f.ny;
              const float along = std::max(0.0f, std::fabs(vy * f.nx - vx * f.ny) - 0.5f);
              d2 = std::min(grid, perp * perp + along * along);
            }
            float& v = drow[x + dx];
            v = std::min(v, d2);
          }
        }
      }
      c += crush(d.data(), true, i0, size_t(y1) * w);
    }
    changed += c;
  });
  return changed;
}

}  // namespace

void cleanBackground(const float* in, float* out, int w, int h, const CleanBackgroundParams& p,
                     CleanBackgroundStats* stats) {
  CleanBackgroundStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  Mask above(n);
  std::vector<Run> runs;
  std::vector<size_t> rowStart;
  if (n > 0) aboveAndRuns(in, out, w, h, p.backgroundBelow, above, runs, rowStart);

  // A1 Clean Background: removed islands go to 0, fringe included.
  if (p.maxIslandPercent > 0.0 && n > 0)
    for (int i : findIslands(in, above, runs, rowStart, w, h, p, st)) {
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
      std::atomic<long long> crushed{0};
      parallelFor(h, 16, [&](int y0, int y1) {
        long long c = 0;
        for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
          const bool crush = !nearEdge[i] && out[i] > 0.0f;
          c += crush;
          if (crush) out[i] = 0.0f;
        }
        crushed += c;
      });
      st.pixelsCrushed = crushed;
    }
  }
  if (stats) *stats = st;
}

}  // namespace mr
