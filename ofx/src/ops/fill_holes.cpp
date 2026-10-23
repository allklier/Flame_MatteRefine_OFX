#include "ops/fill_holes.h"

#include "ops/morphology.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

namespace mr {

namespace {

// Morphological closing of s by a disc of radius r: dilate, then erode (outside the frame
// counts as dilated, so the erosion doesn't eat in from the frame edges). Returns the
// erosion's complement e: closed = s || !e, so "not closed" = !s && e.
Mask closeComplement(const Mask& s, int w, int h, double r) {
  Mask notDilated = dilate(s, w, h, r);
  parallelFor(h, 16, [&](int y0, int y1) {
    for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) notDilated[i] = !notDilated[i];
  });
  return dilate(notDilated, w, h, r);
}

// Horizontal runs of a pixel set, row by row; components are found on runs, not pixels
// (a per-pixel flood and labelling over UHD cost ~20 ms on one thread).
struct Run {
  int x0, x1;  // inclusive
};
struct Runs {
  std::vector<Run> run;
  std::vector<size_t> rowStart;  // h + 1: row y's runs are [rowStart[y], rowStart[y + 1])
};

// The runs of inSet(i) (i = y * w + x), rows in bands on threads.
template <class InSet>
Runs findRuns(int w, int h, InSet inSet) {
  Runs r;
  r.rowStart.assign(size_t(h) + 1, 0);
  std::vector<std::pair<int, std::vector<Run>>> parts;
  std::mutex mu;
  parallelFor(h, 16, [&](int y0, int y1) {
    std::vector<Run> local;
    for (int y = y0; y < y1; ++y) {
      const size_t row = size_t(y) * w, before = local.size();
      for (int x = 0; x < w; ++x) {
        if (!inSet(row + x)) continue;
        const int x0 = x;
        while (x + 1 < w && inSet(row + x + 1)) ++x;
        local.push_back({x0, x});
      }
      r.rowStart[size_t(y) + 1] = local.size() - before;  // a count for now
    }
    std::lock_guard<std::mutex> lock(mu);
    parts.emplace_back(y0, std::move(local));
  });
  for (int y = 0; y < h; ++y) r.rowStart[size_t(y) + 1] += r.rowStart[size_t(y)];
  r.run.resize(r.rowStart[size_t(h)]);
  for (const auto& part : parts)
    std::copy(part.second.begin(), part.second.end(),
              r.run.begin() + std::ptrdiff_t(r.rowStart[size_t(part.first)]));
  return r;
}

// 4-connected components of the runs (runs of adjacent rows that overlap). Returns each run's
// component root (a run index).
std::vector<int> labelRuns(const Runs& r, int h) {
  std::vector<int> parent(r.run.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int a) {
    while (parent[size_t(a)] != a) a = parent[size_t(a)] = parent[size_t(parent[size_t(a)])];
    return a;
  };
  for (int y = 1; y < h; ++y) {
    size_t a = r.rowStart[size_t(y) - 1], b = r.rowStart[size_t(y)];
    const size_t aEnd = r.rowStart[size_t(y)], bEnd = r.rowStart[size_t(y) + 1];
    while (a < aEnd && b < bEnd) {
      const Run &ra = r.run[a], &rb = r.run[b];
      if (ra.x0 <= rb.x1 && rb.x0 <= ra.x1) {
        const int pa = find(int(a)), pb = find(int(b));
        if (pa != pb) parent[size_t(std::max(pa, pb))] = std::min(pa, pb);
      }
      (ra.x1 < rb.x1 ? a : b)++;
    }
  }
  for (size_t k = 0; k < parent.size(); ++k) parent[k] = find(int(k));
  return parent;
}

// Components (by root) that touch an open frame edge: the outside (decision 9, step 2).
std::vector<uint8_t> touchesOpenEdge(const Runs& r, const std::vector<int>& root, int w, int h,
                                     const FillHolesParams& p) {
  std::vector<uint8_t> out(r.run.size(), 0);
  for (int y = 0; y < h; ++y)
    for (size_t k = r.rowStart[size_t(y)]; k < r.rowStart[size_t(y) + 1]; ++k)
      if ((y == 0 && !p.closedTop) || (y == h - 1 && !p.closedBottom) ||
          (r.run[k].x0 == 0 && !p.closedLeft) || (r.run[k].x1 == w - 1 && !p.closedRight))
        out[size_t(root[k])] = 1;
  return out;
}

struct FreeDeleter {
  void operator()(uint8_t* p) const { std::free(p); }
};

}  // namespace

void fillHoles(const float* in, float* out, int w, int h, const FillHolesParams& p,
               FillHolesStats* stats) {
  FillHolesStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  if (n == 0) {
    if (stats) *stats = st;
    return;
  }
  const float thr = p.threshold;
  auto isSolid = [&](size_t i) { return in[i] >= thr; };
  const double maxArea = p.maxHolePercent > 0.0 ? p.maxHolePercent / 100.0 * double(n) : 0.0;

  // Decision 9, step by step. S = solid; C = S closed by Bridge Gaps (C = S without a bridge).
  // Outside = the 4-connected components of not-C that touch an open frame edge; the
  // candidates R = not S and not Outside, as 4-connected components. Each component is
  // decided as a whole, so the order they are found in doesn't matter.
  const double r = p.bridgePx / 2.0;  // a closing of radius r bridges gaps up to 2r wide
  Runs cand;                // runs of R (without a bridge: of not S, the outside included)
  std::vector<int> root;    // each run's component
  std::vector<uint8_t> hasCore, touchesOutside;  // per run
  std::vector<uint8_t> outsideRoot;              // without a bridge: components of the outside
  if (r <= 0.0) {
    // Not C = not S: the components of not S are the outside or a candidate each; a candidate
    // has only not-C pixels (core) and touches no outside pixel (it would be part of it).
    cand = findRuns(w, h, [&](size_t i) { return !isSolid(i); });
    for (const Run& rr : cand.run) st.belowThreshold += rr.x1 - rr.x0 + 1;
    root = labelRuns(cand, h);
    outsideRoot = touchesOpenEdge(cand, root, w, h, p);
    for (size_t k = 0; k < cand.run.size(); ++k)
      if (outsideRoot[size_t(root[k])]) st.outsidePixels += cand.run[k].x1 - cand.run[k].x0 + 1;
    hasCore.assign(cand.run.size(), 1);
    touchesOutside.assign(cand.run.size(), 0);
  } else {
    Mask solid(n);
    std::atomic<long long> below{0};
    parallelFor(h, 16, [&](int y0, int y1) {
      long long b = 0;
      for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) {
        solid[i] = isSolid(i);
        b += !solid[i];
      }
      below += b;
    });
    st.belowThreshold = below;
    const Mask e = closeComplement(solid, w, h, r);
    const Runs notC = findRuns(w, h, [&](size_t i) { return !solid[i] && e[i]; });
    const std::vector<int> notCRoot = labelRuns(notC, h);
    const std::vector<uint8_t> notCOutside = touchesOpenEdge(notC, notCRoot, w, h, p);
    // Per pixel: 0 = S, 1 = C and not S, 2 = not C and not Outside (core), 3 = Outside.
    std::unique_ptr<uint8_t[]> cls(new uint8_t[n]);
    std::atomic<long long> outsidePx{0};
    parallelFor(h, 16, [&](int y0, int y1) {
      long long o = 0;
      for (int y = y0; y < y1; ++y) {
        const size_t row = size_t(y) * w;
        for (int x = 0; x < w; ++x) cls[row + x] = solid[row + x] ? 0 : e[row + x] ? 2 : 1;
        for (size_t k = notC.rowStart[size_t(y)]; k < notC.rowStart[size_t(y) + 1]; ++k)
          if (notCOutside[size_t(notCRoot[k])]) {
            std::fill(&cls[row + notC.run[k].x0], &cls[row + notC.run[k].x1] + 1, uint8_t(3));
            o += notC.run[k].x1 - notC.run[k].x0 + 1;
          }
      }
      outsidePx += o;
    });
    st.outsidePixels = outsidePx;
    // R's runs, each with whether it holds a core pixel and touches an outside 4-neighbour.
    cand = findRuns(w, h, [&](size_t i) { return cls[i] == 1 || cls[i] == 2; });
    hasCore.assign(cand.run.size(), 0);
    touchesOutside.assign(cand.run.size(), 0);
    parallelFor(h, 16, [&](int y0, int y1) {
      for (int y = y0; y < y1; ++y) {
        const uint8_t* row = &cls[size_t(y) * w];
        for (size_t k = cand.rowStart[size_t(y)]; k < cand.rowStart[size_t(y) + 1]; ++k) {
          const Run& rr = cand.run[k];
          uint8_t c = 0, t = 0;
          if (rr.x0 > 0 && row[rr.x0 - 1] == 3) t = 1;
          if (rr.x1 + 1 < w && row[rr.x1 + 1] == 3) t = 1;
          for (int x = rr.x0; x <= rr.x1; ++x) {
            c |= row[x] == 2;
            if (y > 0 && row[x - w] == 3) t = 1;
            if (y + 1 < h && row[x + w] == 3) t = 1;
          }
          hasCore[k] = c;
          touchesOutside[k] = t;
        }
      }
    });
    root = labelRuns(cand, h);
  }

  // Decide each candidate component (decision 9, steps 4 and 5).
  const size_t nr = cand.run.size();
  std::vector<long long> area(nr, 0);
  std::vector<uint8_t> compCore(nr, 0), compTouch(nr, 0), isRoot(nr, 0);
  for (size_t k = 0; k < nr; ++k) {
    const size_t c = size_t(root[k]);
    isRoot[c] = 1;
    area[c] += cand.run[k].x1 - cand.run[k].x0 + 1;
    compCore[c] |= hasCore[k];
    compTouch[c] |= touchesOutside[k];
  }
  std::vector<uint8_t> fillComp(nr, 0);
  for (size_t c = 0; c < nr; ++c) {
    if (!isRoot[c] || (!outsideRoot.empty() && outsideRoot[c])) continue;  // or the outside
    // All closed and touching the outside: an outer notch the closing filled, i.e. the edge.
    if (!compCore[c] && compTouch[c]) {
      ++st.notchesKept;
      continue;
    }
    if (maxArea > 0.0 && double(area[c]) > maxArea) {
      ++st.holesTooLarge;
      continue;
    }
    ++st.holesFilled;
    fillComp[c] = 1;
  }

  // Output: the input, filled holes lifted to 1.0 (decision 10), then the rim.
  if (out != in)
    parallelFor(h, 16, [&](int y0, int y1) {
      std::copy(in + size_t(y0) * w, in + size_t(y1) * w, out + size_t(y0) * w);
    });
  if (st.holesFilled == 0) {
    if (stats) *stats = st;
    return;
  }
  const bool rim = p.rimPx > 0.0;
  // The fill mask, only needed by the rim; calloc leaves untouched pages unmapped.
  std::unique_ptr<uint8_t, FreeDeleter> fill(
      rim ? static_cast<uint8_t*>(std::calloc(n, 1)) : nullptr);
  std::atomic<long long> changed{0};
  parallelFor(h, 16, [&](int y0, int y1) {
    long long c = 0;
    for (int y = y0; y < y1; ++y)
      for (size_t k = cand.rowStart[size_t(y)]; k < cand.rowStart[size_t(y) + 1]; ++k) {
        if (!fillComp[size_t(root[k])]) continue;
        const size_t row = size_t(y) * w;
        for (int x = cand.run[k].x0; x <= cand.run[k].x1; ++x) {
          // in[i] < 1 is out[i] < 1 here: out holds the input, and each pixel is visited once.
          if (out[row + x] < 1.0f) {
            out[row + x] = 1.0f;
            ++c;
          }
          if (rim) fill.get()[row + x] = 1;
        }
      }
    changed += c;
  });
  st.pixelsChanged = changed;

  // Rim: grow the fill into the solid's soft inner rim (solid pixels only, so never into the
  // outside or into a hole left open). The same disc, boundary and path choice as dilate():
  // stamp the disc on the fill's boundary pixels, or (when that costs more than a frame) the
  // distance transform via dilate() itself. A filled pixel already holds 1.0, so testing it
  // as solid after an in-place lift changes nothing.
  if (rim) {
    const uint8_t* fm = fill.get();
    std::vector<std::pair<int, int>> disc;
    const int ri = int(std::floor(p.rimPx));
    for (int dy = -ri; dy <= ri; ++dy)
      for (int dx = -ri; dx <= ri; ++dx)
        if ((dx || dy) && dx * dx + dy * dy <= p.rimPx * p.rimPx) disc.emplace_back(dx, dy);
    std::vector<int> boundary;
    for (int y = 0; y < h; ++y)
      for (size_t k = cand.rowStart[size_t(y)]; k < cand.rowStart[size_t(y) + 1]; ++k) {
        if (!fillComp[size_t(root[k])]) continue;
        for (int x = cand.run[k].x0; x <= cand.run[k].x1; ++x) {
          const size_t i = size_t(y) * w + x;
          if ((x > 0 && !fm[i - 1]) || (x + 1 < w && !fm[i + 1]) || (y > 0 && !fm[i - w]) ||
              (y + 1 < h && !fm[i + w]))
            boundary.push_back(int(i));
        }
      }
    long long c = 0;
    auto lift = [&](size_t j) {
      if (in[j] >= thr && out[j] < 1.0f) {
        out[j] = 1.0f;
        ++c;
      }
    };
    if (double(boundary.size()) * disc.size() > double(n)) {
      const Mask grown = dilate(Mask(fm, fm + n), w, h, p.rimPx);
      for (size_t i = 0; i < n; ++i)
        if (grown[i]) lift(i);
    } else {
      for (int i : boundary) {
        const int x = i % w, y = i / w;
        for (const auto& o : disc) {
          const int nx = x + o.first, ny = y + o.second;
          if (nx >= 0 && ny >= 0 && nx < w && ny < h) lift(size_t(ny) * w + nx);
        }
      }
    }
    st.pixelsChanged += c;
  }
  if (stats) *stats = st;
}

}  // namespace mr
