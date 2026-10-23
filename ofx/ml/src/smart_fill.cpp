#include "smart_fill.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>

#include "edge_refine.h"   // clampPicture
#include "ops/morphology.h"

namespace mr {
namespace ml {

namespace {

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Run {
  int y, x0, x1;  // inclusive
};

// The 4-connected components of the not-solid pixels, as runs (as fill_holes.cpp: runs of a row,
// joined with the overlapping runs of the row above by union-find).
struct Components {
  std::vector<Run> run;
  std::vector<int> root;  // per run: its component's root run
  std::vector<uint8_t> outside;  // per root: touches the frame edge
};

Components label(const float* m, int w, int h, float thr) {
  Components c;
  std::vector<size_t> rowStart(size_t(h) + 1, 0);
  for (int y = 0; y < h; ++y) {
    const float* row = m + size_t(y) * w;
    for (int x = 0; x < w; ++x) {
      if (row[x] >= thr) continue;  // NaN is not solid (as Fill Holes)
      const int x0 = x;
      while (x + 1 < w && !(row[x + 1] >= thr)) ++x;
      c.run.push_back({y, x0, x});
    }
    rowStart[size_t(y) + 1] = c.run.size();
  }
  std::vector<int>& parent = c.root;
  parent.resize(c.run.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int a) {
    while (parent[size_t(a)] != a) a = parent[size_t(a)] = parent[size_t(parent[size_t(a)])];
    return a;
  };
  for (int y = 1; y < h; ++y) {
    size_t a = rowStart[size_t(y) - 1], b = rowStart[size_t(y)];
    const size_t aEnd = rowStart[size_t(y)], bEnd = rowStart[size_t(y) + 1];
    while (a < aEnd && b < bEnd) {
      const Run &ra = c.run[a], &rb = c.run[b];
      if (ra.x0 <= rb.x1 && rb.x0 <= ra.x1) {  // overlapping columns: 4-connected
        const int pa = find(int(a)), pb = find(int(b));
        if (pa != pb) parent[size_t(std::max(pa, pb))] = std::min(pa, pb);
      }
      (ra.x1 < rb.x1 ? a : b)++;
    }
  }
  for (size_t k = 0; k < parent.size(); ++k) parent[k] = find(int(k));
  c.outside.assign(c.run.size(), 0);
  for (size_t k = 0; k < c.run.size(); ++k) {
    const Run& r = c.run[k];
    if (r.y == 0 || r.y == h - 1 || r.x0 == 0 || r.x1 == w - 1) c.outside[size_t(c.root[k])] = 1;
  }
  return c;
}

// A hole's pixels as a mask on its bounding box grown by `margin`, clipped to the frame.
struct Crop {
  int x0, y0, w, h;
  Mask m;
};
Crop holeCrop(const Components& c, const std::vector<int>& runs, const HoleInfo& hi, int margin, int w, int h) {
  Crop k;
  k.x0 = std::max(hi.x0 - margin, 0);
  k.y0 = std::max(hi.y0 - margin, 0);
  k.w = std::min(hi.x1 + margin, w - 1) - k.x0 + 1;
  k.h = std::min(hi.y1 + margin, h - 1) - k.y0 + 1;
  k.m.assign(size_t(k.w) * k.h, 0);
  for (int r : runs) {
    const Run& rr = c.run[size_t(r)];
    uint8_t* row = &k.m[size_t(rr.y - k.y0) * k.w];
    std::fill(row + (rr.x0 - k.x0), row + (rr.x1 - k.x0) + 1, uint8_t(1));
  }
  return k;
}

float dist3(const double* a, const double* b) {
  return float(std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])));
}

}  // namespace

const char* decisionName(HoleDecision d) {
  switch (d) {
    case HoleDecision::Filled: return "filled";
    case HoleDecision::KeptRealGap: return "kept (real gap)";
    case HoleDecision::KeptTooLarge: return "kept (too large)";
    case HoleDecision::NotScored: break;
  }
  return "not scored";
}

void smartFill(const float* matte, const float* rgb, int w, int h, const SmartFillParams& p, SegRunner& runner,
               float* out, SmartFillStats* stats, const std::function<bool()>& abort) {
  const double t0 = nowMs();
  SmartFillStats st;
  const size_t n = size_t(std::max(w, 0)) * size_t(std::max(h, 0));
  auto finish = [&]() {
    st.totalMs = nowMs() - t0;
    if (stats) *stats = std::move(st);
  };
  auto passThrough = [&]() {
    if (out != matte) std::copy(matte, matte + n, out);
  };
  if (n == 0) return finish();
  const float thr = p.solidAbove;

  // 1. The holes.
  const Components c = label(matte, w, h, thr);
  std::vector<int> holeOf(c.run.size(), -1);  // per root: hole index
  std::vector<std::vector<int>> holeRuns;
  for (size_t k = 0; k < c.run.size(); ++k) {
    const int r = c.root[k];
    if (c.outside[size_t(r)]) continue;
    if (holeOf[size_t(r)] < 0) {
      holeOf[size_t(r)] = int(holeRuns.size());
      holeRuns.emplace_back();
      HoleInfo hi;
      hi.x0 = c.run[k].x0, hi.x1 = c.run[k].x1, hi.y0 = hi.y1 = c.run[k].y;
      hi.sx = c.run[k].x0, hi.sy = c.run[k].y;  // runs are in raster order: the first one is the first pixel
      st.holes.push_back(hi);
    }
    const int id = holeOf[size_t(r)];
    HoleInfo& hi = st.holes[size_t(id)];
    holeRuns[size_t(id)].push_back(int(k));
    hi.area += c.run[k].x1 - c.run[k].x0 + 1;
    hi.x0 = std::min(hi.x0, c.run[k].x0);
    hi.x1 = std::max(hi.x1, c.run[k].x1);
    hi.y1 = std::max(hi.y1, c.run[k].y);
  }
  st.holesFound = int(st.holes.size());
  st.labelMs = nowMs() - t0;

  // 2. Which holes need a score.
  const double maxArea = p.maxHoleFraction * double(n);
  bool needScore = false;
  for (HoleInfo& hi : st.holes) {
    if (p.maxHoleFraction > 0.0 && double(hi.area) > maxArea) {
      hi.decision = HoleDecision::KeptTooLarge;
      ++st.holesTooLarge;
    } else if (!p.keepRealGaps) {
      hi.decision = HoleDecision::Filled;
    } else {
      needScore = true;
    }
  }
  const bool useSeg = needScore && p.mode != ScoreMode::Colour;
  const bool useColour = needScore && p.mode != ScoreMode::Segmentation;

  // 3. The segmentation (one model call per frame).
  std::vector<float> seg;
  if (useSeg) {
    if (abort && abort()) {
      st.aborted = true;
      passThrough();
      return finish();
    }
    const double ts = nowMs();
    seg.assign(n, 0.0f);
    runner.segment(rgb, w, h, seg.data());
    st.segmented = true;
    st.segMs = nowMs() - ts;
    if (abort && abort()) {
      st.aborted = true;
      passThrough();
      return finish();
    }
  }

  // 4. Scores and decisions.
  const double tsc = nowMs();
  // Clear background (colour): matte < 0.05 and in a component touching the frame edge.
  Mask outsideMask;
  if (useColour) {
    outsideMask.assign(n, 0);
    for (size_t k = 0; k < c.run.size(); ++k)
      if (c.outside[size_t(c.root[k])]) {
        const Run& r = c.run[k];
        std::fill(&outsideMask[size_t(r.y) * w + r.x0], &outsideMask[size_t(r.y) * w + r.x1] + 1, uint8_t(1));
      }
  }
  // Its sums per cell of kCell x kCell px, filled when a hole's window first touches the cell.
  const int kCell = 32, cw = (w + kCell - 1) / kCell, chh = (h + kCell - 1) / kCell;
  struct Cell {
    double sum[3];
    long long n;
    bool done;
  };
  std::vector<Cell> cells(useColour ? size_t(cw) * chh : 0, Cell{{0, 0, 0}, 0, false});
  auto cell = [&](int cx, int cy) -> const Cell& {
    Cell& e = cells[size_t(cy) * cw + cx];
    if (!e.done) {
      e.done = true;
      for (int y = cy * kCell; y < std::min((cy + 1) * kCell, h); ++y)
        for (int x = cx * kCell; x < std::min((cx + 1) * kCell, w); ++x) {
          const size_t i = size_t(y) * w + x;
          if (!outsideMask[i] || !(matte[i] < 0.05f)) continue;
          for (int k = 0; k < 3; ++k) e.sum[k] += clampPicture(rgb[i * 3 + k]);
          ++e.n;
        }
    }
    return e;
  };
  const float ringMin = std::max(0.9f, thr);
  const int ringI = int(std::ceil(std::max(p.ringPx, 0.0))) + 1;
  const int reach = int(std::ceil(std::max(p.bgReachPx, 0.0)));
  std::vector<float> vals;
  for (size_t id = 0; id < st.holes.size(); ++id) {
    HoleInfo& hi = st.holes[id];
    if (hi.decision != HoleDecision::NotScored) continue;
    const std::vector<int>& runs = holeRuns[id];
    // The pixels scored: the hole's core (matte < 0.5) if it has one, else all its pixels. The
    // soft rim between 0.5 and Solid Above belongs to the solid side and shows its colour.
    const float coreBelow = std::min(0.5f, thr);
    bool hasCore = false;
    for (size_t q = 0; q < runs.size() && !hasCore; ++q) {
      const Run& rr = c.run[size_t(runs[q])];
      for (int x = rr.x0; x <= rr.x1 && !hasCore; ++x) hasCore = matte[size_t(rr.y) * w + x] < coreBelow;
    }
    auto scored = [&](size_t i) { return !hasCore || matte[i] < coreBelow; };
    if (useSeg) {
      vals.clear();
      double sum = 0;
      for (int r : runs) {
        const Run& rr = c.run[size_t(r)];
        for (int x = rr.x0; x <= rr.x1; ++x) {
          if (!scored(size_t(rr.y) * w + x)) continue;
          const float v = std::min(std::max(seg[size_t(rr.y) * w + x], 0.0f), 1.0f);
          vals.push_back(v);
          sum += v;
        }
      }
      hi.segMean = float(sum / double(vals.size()));
      hi.segMin = *std::min_element(vals.begin(), vals.end());
      std::nth_element(vals.begin(), vals.begin() + std::ptrdiff_t(vals.size() / 2), vals.end());
      hi.segMedian = vals[vals.size() / 2];
    }
    if (useColour) {
      double ch[3] = {0, 0, 0}, cs[3] = {0, 0, 0}, cb[3] = {0, 0, 0};
      long long nh = 0, ns = 0, nb = 0;
      for (int r : runs) {
        const Run& rr = c.run[size_t(r)];
        for (int x = rr.x0; x <= rr.x1; ++x) {
          if (!scored(size_t(rr.y) * w + x)) continue;
          for (int k = 0; k < 3; ++k) ch[k] += clampPicture(rgb[(size_t(rr.y) * w + x) * 3 + k]);
          ++nh;
        }
      }
      // Subject ring: within ringPx of the hole, solid.
      const Crop crop = holeCrop(c, runs, hi, ringI, w, h);
      const Mask grown = dilate(crop.m, crop.w, crop.h, p.ringPx);
      for (int y = 0; y < crop.h; ++y)
        for (int x = 0; x < crop.w; ++x) {
          const size_t l = size_t(y) * crop.w + x, i = size_t(y + crop.y0) * w + (x + crop.x0);
          if (grown[l] && !crop.m[l] && matte[i] >= ringMin) {
            for (int k = 0; k < 3; ++k) cs[k] += clampPicture(rgb[i * 3 + k]);
            ++ns;
          }
        }
      // Background: clear pixels in the cells the window (bounding box + reach) touches; the
      // reach doubles until they hold 16 (or the window is the frame).
      for (int rch = std::max(reach, 1);; rch *= 2) {
        const int cx0 = std::max(hi.x0 - rch, 0) / kCell, cx1 = std::min(hi.x1 + rch, w - 1) / kCell;
        const int cy0 = std::max(hi.y0 - rch, 0) / kCell, cy1 = std::min(hi.y1 + rch, h - 1) / kCell;
        std::fill(cb, cb + 3, 0.0);
        nb = 0;
        for (int cy = cy0; cy <= cy1; ++cy)
          for (int cx = cx0; cx <= cx1; ++cx) {
            const Cell& e = cell(cx, cy);
            for (int k = 0; k < 3; ++k) cb[k] += e.sum[k];
            nb += e.n;
          }
        if (nb >= 16 || (cx0 == 0 && cy0 == 0 && cx1 == cw - 1 && cy1 == chh - 1)) break;
      }
      if (nb)
        for (double& v : cb) v /= double(nb);
      if (ns == 0 || nb == 0 || nh == 0) {
        hi.colour = 0.5f;
      } else {
        for (int k = 0; k < 3; ++k) ch[k] /= double(nh), cs[k] /= double(ns);
        const float ds = dist3(ch, cs), db = dist3(ch, cb);
        hi.colour = ds + db > 1e-6f ? db / (ds + db) : 0.5f;
      }
    }
    const float segScore = !useSeg ? -1.0f
                           : p.segStat == SegStatistic::Mean ? hi.segMean
                           : p.segStat == SegStatistic::Median ? hi.segMedian
                                                               : hi.segMin;
    hi.score = p.mode == ScoreMode::Segmentation ? segScore
               : p.mode == ScoreMode::Colour     ? hi.colour
                                                 : std::min(segScore, hi.colour);
    hi.decision = double(hi.score) >= p.subjectThreshold ? HoleDecision::Filled : HoleDecision::KeptRealGap;
  }
  st.scoreMs = nowMs() - tsc;

  // 5. Output: the input, filled holes at 1.0 (decision 10), then the rim (decision 9, step 6).
  const double tw = nowMs();
  passThrough();
  long long changed = 0;
  const int rimI = int(std::ceil(std::max(p.rimPx, 0.0))) + 1;
  for (size_t id = 0; id < st.holes.size(); ++id) {
    const HoleInfo& hi = st.holes[id];
    if (hi.decision == HoleDecision::Filled) {
      ++st.holesFilled;
    } else {
      if (hi.decision == HoleDecision::KeptRealGap) ++st.holesKept;
      continue;
    }
    for (int r : holeRuns[id]) {
      const Run& rr = c.run[size_t(r)];
      for (int x = rr.x0; x <= rr.x1; ++x) {
        const size_t i = size_t(rr.y) * w + x;
        if (out[i] != 1.0f) {
          out[i] = 1.0f;
          ++changed;
        }
      }
    }
    if (p.rimPx > 0.0) {
      const Crop crop = holeCrop(c, holeRuns[id], hi, rimI, w, h);
      const Mask grown = dilate(crop.m, crop.w, crop.h, p.rimPx);
      for (int y = 0; y < crop.h; ++y)
        for (int x = 0; x < crop.w; ++x) {
          const size_t l = size_t(y) * crop.w + x, i = size_t(y + crop.y0) * w + (x + crop.x0);
          // matte, not out: out may be matte, and a lifted pixel is >= thr anyway.
          if (grown[l] && !crop.m[l] && matte[i] >= thr && out[i] != 1.0f) {
            out[i] = 1.0f;
            ++changed;
          }
        }
    }
  }
  st.pixelsChanged = changed;
  st.writeMs = nowMs() - tw;
  finish();
}

// --- BiRefNet pre / post ---------------------------------------------------------------------------

namespace {

struct Taps {
  std::vector<int> start, count;
  std::vector<float> weight;  // count[i] weights from start[i]; packed at offset[i]
  std::vector<size_t> offset;
};

// PIL's ImagingResample coefficients for the bilinear (triangle) filter.
Taps taps(int src, int dst) {
  Taps t;
  const double scale = double(src) / dst, fscale = std::max(scale, 1.0), support = fscale, ss = 1.0 / fscale;
  for (int i = 0; i < dst; ++i) {
    const double centre = (i + 0.5) * scale;
    const int lo = std::max(int(centre - support + 0.5), 0), hi = std::min(int(centre + support + 0.5), src);
    t.start.push_back(lo);
    t.offset.push_back(t.weight.size());
    double sum = 0;
    std::vector<double> wv;
    for (int j = lo; j < hi; ++j) {
      const double d = std::fabs((j - centre + 0.5) * ss);
      wv.push_back(d < 1.0 ? 1.0 - d : 0.0);
      sum += wv.back();
    }
    for (double v : wv) t.weight.push_back(float(sum > 0 ? v / sum : 0.0));
    t.count.push_back(hi - lo);
  }
  return t;
}

}  // namespace

void resampleBilinear(const float* src, int sw, int sh, int ch, float* dst, int dw, int dh) {
  const Taps tx = taps(sw, dw), ty = taps(sh, dh);
  std::vector<float> tmp(size_t(dw) * sh * ch);
  parallelFor(sh, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < dw; ++x)
        for (int c = 0; c < ch; ++c) {
          double acc = 0;
          for (int k = 0; k < tx.count[size_t(x)]; ++k)
            acc += double(tx.weight[tx.offset[size_t(x)] + k]) *
                   src[(size_t(y) * sw + size_t(tx.start[size_t(x)] + k)) * ch + c];
          tmp[(size_t(y) * dw + x) * ch + c] = float(acc);
        }
  });
  parallelFor(dh, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < dw; ++x)
        for (int c = 0; c < ch; ++c) {
          double acc = 0;
          for (int k = 0; k < ty.count[size_t(y)]; ++k)
            acc += double(ty.weight[ty.offset[size_t(y)] + k]) *
                   tmp[(size_t(ty.start[size_t(y)] + k) * dw + x) * ch + c];
          dst[(size_t(y) * dw + x) * ch + c] = float(acc);
        }
  });
}

void birefnetPrepare(const float* rgb, int w, int h, float* input) {
  const int S = kBiRefNetSize;
  std::vector<float> disp(size_t(w) * h * 3), small(size_t(S) * S * 3);
  for (size_t i = 0; i < disp.size(); ++i) disp[i] = clampPicture(rgb[i]);
  resampleBilinear(disp.data(), w, h, 3, small.data(), S, S);
  const float mean[3] = {0.485f, 0.456f, 0.406f}, sdv[3] = {0.229f, 0.224f, 0.225f};
  for (size_t i = 0; i < size_t(S) * S; ++i)
    for (int c = 0; c < 3; ++c) input[size_t(c) * S * S + i] = (small[i * 3 + c] - mean[c]) / sdv[c];
}

void birefnetPostprocess(const float* logits, int w, int h, float* mask) {
  const int S = kBiRefNetSize;
  std::vector<float> prob(size_t(S) * S);
  for (size_t i = 0; i < prob.size(); ++i) prob[i] = 1.0f / (1.0f + std::exp(-logits[i]));
  resampleBilinear(prob.data(), S, S, 1, mask, w, h);
}

}  // namespace ml
}  // namespace mr
