// Temporal Stabilise (B4): see temporal.h.
#include "temporal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>

#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {
namespace ml {

namespace {
using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

double baseWeight(int offset) {
  const int d = std::abs(offset);
  return d == 0 ? 1.0 : d == 1 ? 0.8 : 0.6;
}

// Holes of the centre matte: 4-connected regions of m < 0.5 not connected to the frame border.
// Run-based: the low runs of each row (rows in parallel), joined with the overlapping runs of the
// row above (union-find, serial over runs, not pixels). A per-pixel flood fill took ~90 ms at
// UHD (most of the frame is background). Returns the number of holes; holeRuns: (row, x0, x1
// exclusive, hole index) of every hole run; label: w*h, -1 = not a hole, else the hole index.
struct Run {
  int y, x0, x1, hole;
};
int labelHoles(const float* m, int w, int h, std::vector<int>& label, std::vector<Run>& holeRuns, Workers* wk) {
  std::vector<std::vector<Run>> rows(h);
  parallelFor(wk, h, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
      const float* r = m + size_t(y) * w;
      for (int x = 0; x < w;) {
        if (r[x] >= 0.5f) {
          ++x;
          continue;
        }
        const int s = x;
        while (x < w && !(r[x] >= 0.5f)) ++x;  // NaN counts as low
        rows[y].push_back({y, s, x, -1});
      }
    }
  });
  std::vector<int> first(h + 1, 0);
  for (int y = 0; y < h; ++y) first[y + 1] = first[y] + int(rows[y].size());
  std::vector<int> parent(first[h]);
  for (int i = 0; i < first[h]; ++i) parent[i] = i;
  auto find = [&](int i) {
    while (parent[i] != i) i = parent[i] = parent[parent[i]];
    return i;
  };
  for (int y = 1; y < h; ++y) {
    const std::vector<Run>& up = rows[y - 1];
    const std::vector<Run>& cur = rows[y];
    size_t j = 0;
    for (size_t i = 0; i < cur.size(); ++i) {
      while (j < up.size() && up[j].x1 <= cur[i].x0) ++j;
      for (size_t k = j; k < up.size() && up[k].x0 < cur[i].x1; ++k) {
        const int a = find(first[y] + int(i)), b = find(first[y - 1] + int(k));
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
      }
    }
  }
  std::vector<uint8_t> border(first[h], 0);
  for (int y = 0; y < h; ++y)
    for (size_t i = 0; i < rows[y].size(); ++i) {
      const Run& r = rows[y][i];
      if (y == 0 || y == h - 1 || r.x0 == 0 || r.x1 == w) border[find(first[y] + int(i))] = 1;
    }
  std::vector<int> index(first[h], -1);
  int holes = 0;
  holeRuns.clear();
  for (int y = 0; y < h; ++y)
    for (size_t i = 0; i < rows[y].size(); ++i) {
      const int root = find(first[y] + int(i));
      if (border[root]) continue;
      if (index[root] < 0) index[root] = holes++;
      Run r = rows[y][i];
      r.hole = index[root];
      holeRuns.push_back(r);
    }
  label.assign(size_t(w) * h, -1);
  for (const Run& r : holeRuns) std::fill(&label[size_t(r.y) * w + r.x0], &label[size_t(r.y) * w + r.x1], r.hole);
  return holes;
}

struct Neighbour {
  const float* matte;
  double base;
  NeighbourFlow flow;
};

// Neighbour k's value and weight at frame pixel (x, y); weight 0 = no read.
inline void readNeighbour(const Neighbour& nb, int w, int h, int x, int y, float* v, double* wt) {
  double dx, dy;
  flowAt(nb.flow, x, y, &dx, &dy);
  const double qx = x + dx, qy = y + dy;
  if (qx < 0 || qy < 0 || qx > w - 1 || qy > h - 1) {
    *wt = 0;
    return;
  }
  const int s = nb.flow.scale;
  const float gx = float((x - (s - 1) * 0.5) / s), gy = float((y - (s - 1) * 0.5) / s);
  const double c = sampleField(nb.flow.conf.data(), nb.flow.fwd.w, nb.flow.fwd.h, gx, gy);
  *wt = nb.base * c;
  *v = sampleBilinear(nb.matte, w, h, qx, qy);
}
}  // namespace

int flowScale(int h, int flowLines) {
  int s = 1;
  while (h > s * std::max(flowLines, 16)) s *= 2;
  return s;
}

void flowAt(const NeighbourFlow& f, double x, double y, double* dx, double* dy) {
  const int s = f.scale;
  const float gx = float((x - (s - 1) * 0.5) / s), gy = float((y - (s - 1) * 0.5) / s);
  *dx = s * double(sampleField(f.fwd.u.data(), f.fwd.w, f.fwd.h, gx, gy));
  *dy = s * double(sampleField(f.fwd.v.data(), f.fwd.w, f.fwd.h, gx, gy));
}

namespace {
// A frame on the flow grid: luma (or the matte), the matte, the data-term weight.
struct Grid {
  int w = 0, h = 0, s = 1;
  std::vector<float> l, m, wt;
};
Grid makeGrid(const TemporalFrame& f, bool usePicture, int w, int h, const StabiliseParams& p, Workers* wk) {
  Grid g;
  g.s = flowScale(h, p.flowLines);
  g.w = w;
  g.h = h;
  for (int k = g.s; k > 1; k /= 2) g.w = (g.w + 1) / 2, g.h = (g.h + 1) / 2;
  const size_t on = size_t(g.w) * g.h;
  g.m.resize(on);
  downsample(f.matte, w, h, g.s, g.m.data(), wk);
  if (usePicture) {
    g.l.resize(on);
    luma(f.rgb, w, h, g.s, g.l.data(), wk);
  } else {
    g.l = g.m;
  }
  if (p.subjectWeighted) {
    g.wt.resize(on);
    for (size_t i = 0; i < on; ++i) g.wt[i] = std::min(std::max(g.m[i], 0.0f), 1.0f);
  }
  return g;
}

NeighbourFlow flowBetween(const Grid& a, const Grid& b, const StabiliseParams& p, Workers* wk) {
  NeighbourFlow r;
  r.scale = a.s;
  const int ow = a.w, oh = a.h;
  const size_t on = size_t(ow) * oh;
  r.fwd = denseFlow(a.l.data(), b.l.data(), a.m.data(), b.m.data(), a.wt.empty() ? nullptr : a.wt.data(), ow, oh,
                    p.flow, wk);
  const Flow bwd = denseFlow(b.l.data(), a.l.data(), b.m.data(), a.m.data(), b.wt.empty() ? nullptr : b.wt.data(),
                             ow, oh, p.flow, wk);
  r.conf.assign(on, 0.0f);
  const float maxE = float(p.fbMaxPx);
  parallelFor(wk, oh, 16, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < ow; ++x) {
        const size_t i = size_t(y) * ow + x;
        const float qx = x + r.fwd.u[i], qy = y + r.fwd.v[i];
        if (qx < 0 || qy < 0 || qx > ow - 1 || qy > oh - 1) continue;
        const float ex = r.fwd.u[i] + sampleField(bwd.u.data(), ow, oh, qx, qy);
        const float ey = r.fwd.v[i] + sampleField(bwd.v.data(), ow, oh, qx, qy);
        const float e2 = ex * ex + ey * ey;
        if (e2 <= maxE * maxE) r.conf[i] = std::exp(-e2 / (2 * 0.25f));
      }
  });
  return r;
}
}  // namespace

NeighbourFlow flowToNeighbour(const TemporalFrame& c, const TemporalFrame& nb, int w, int h,
                              const StabiliseParams& p, Workers* wk) {
  const bool pic = c.rgb && nb.rgb;
  return flowBetween(makeGrid(c, pic, w, h, p, wk), makeGrid(nb, pic, w, h, p, wk), p, wk);
}

void stabilise(const TemporalFrame* frames, int count, int centre, int w, int h, const StabiliseParams& p,
               float* out, StabiliseStats* stats) {
  StabiliseStats st;
  const auto t0 = Clock::now();
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  const float* cm = frames[centre].matte;
  if (out != cm && n) std::memcpy(out, cm, n * sizeof(float));
  const int K = std::max(p.frames, 0);
  const double strength = std::min(std::max(p.strength, 0.0), 1.0);
  std::vector<Neighbour> nbs;
  for (int i = 0; i < count; ++i)
    if (i != centre && std::abs(i - centre) <= K) nbs.push_back({frames[i].matte, baseWeight(i - centre), {}});
  st.neighbours = int(nbs.size());
  if (nbs.empty() || n == 0 || strength == 0.0) {
    st.totalMs = msSince(t0);
    if (stats) *stats = st;
    return;
  }
  Workers wk;
  {
    const auto tf = Clock::now();
    bool pic = true;  // the picture only if every frame used has one
    for (int i = 0; i < count; ++i)
      if (std::abs(i - centre) <= K) pic = pic && frames[i].rgb;
    const Grid gc = makeGrid(frames[centre], pic, w, h, p, &wk);
    for (int i = 0, k = 0; i < count; ++i)
      if (i != centre && std::abs(i - centre) <= K)
        nbs[k++].flow = flowBetween(gc, makeGrid(frames[i], pic, w, h, p, &wk), p, &wk);
    st.flowScale = nbs[0].flow.scale;
    st.flowMs = msSince(tf);
  }

  // The region: the band weight from the signed distance, and the holes to fill.
  const auto tr = Clock::now();
  const double band = std::max(p.bandPx, 0.5);
  std::unique_ptr<float[]> sd = signedEdgeDistanceThreaded(cm, w, h, 0.5f, band);
  std::vector<int> label;
  std::vector<uint8_t> fillHole;
  if (p.fillHoles) {
    std::vector<Run> holeRuns;
    st.holes = labelHoles(cm, w, h, label, holeRuns, &wk);
    // Per hole and neighbour: pixels, confident reads, the sum of the confident values.
    const size_t H = size_t(st.holes), N = nbs.size();
    std::vector<long long> pixels(H, 0), confident(H * N, 0);
    std::vector<double> sum(H * N, 0.0);
    for (const Run& r : holeRuns)
      for (int x = r.x0; x < r.x1; ++x) {
        const int l = r.hole;
        ++pixels[l];
        for (size_t k = 0; k < N; ++k) {
          float v = 0;
          double wt;
          readNeighbour(nbs[k], w, h, x, r.y, &v, &wt);
          if (wt >= 0.5 * nbs[k].base) {
            ++confident[l * N + k];
            sum[l * N + k] += v;
          }
        }
      }
    fillHole.assign(H, 0);
    for (size_t l = 0; l < H; ++l) {
      double filled = 0, open = baseWeight(0);
      for (size_t k = 0; k < N; ++k) {
        const long long c = confident[l * N + k];
        if (2 * c < pixels[l]) continue;  // abstains: too little of the hole seen confidently
        (sum[l * N + k] / c >= 0.5 ? filled : open) += nbs[k].base;
      }
      fillHole[l] = filled > open;
      st.holesFilled += fillHole[l];
    }
  }
  const double full = band * 2.0 / 3.0;
  auto regionWeight = [&](size_t i) {
    double r = 0;
    const double d = std::fabs(sd[i]);
    if (d < band) r = d <= full ? 1.0 : (band - d) / (band - full);
    if (!label.empty() && label[i] >= 0 && fillHole[label[i]]) r = 1.0;
    return r;
  };
  st.regionMs = msSince(tr);

  // The combine, per row in parallel; each pixel's value depends only on the inputs.
  const auto tc = Clock::now();
  const int rows = h;
  std::vector<long long> region(rows, 0), changed(rows, 0), reads(rows, 0), rejected(rows, 0);
  parallelFor(&wk, rows, 8, [&](int y0, int y1) {
    for (int y = y0; y < y1; ++y)
      for (int x = 0; x < w; ++x) {
        const size_t i = size_t(y) * w + x;
        const double r = regionWeight(i);
        if (r <= 0) continue;
        ++region[y];
        float vals[8];
        double wts[8];
        int m = 0;
        vals[m] = cm[i];
        wts[m++] = baseWeight(0);
        for (const Neighbour& nb : nbs) {
          float v = 0;
          double wt;
          readNeighbour(nb, w, h, x, y, &v, &wt);
          ++reads[y];
          if (wt <= 0) {
            ++rejected[y];
            continue;
          }
          vals[m] = v;
          wts[m++] = wt;
        }
        if (m == 1) continue;
        double combined;
        // A weighted median of two values is the centre (its weight is the larger): with a single
        // neighbour read (sequence ends at K = 1, or the others rejected), the weighted mean.
        if (p.combine == Combine::Mean || m == 2) {
          double s = 0, ws = 0;
          for (int j = 0; j < m; ++j) s += wts[j] * vals[j], ws += wts[j];
          combined = s / ws;
        } else {  // weighted median: the first value (ascending) where the weight reaches half
          for (int a = 1; a < m; ++a)
            for (int b = a; b > 0 && vals[b] < vals[b - 1]; --b) std::swap(vals[b], vals[b - 1]), std::swap(wts[b], wts[b - 1]);
          double total = 0;
          for (int j = 0; j < m; ++j) total += wts[j];
          double cum = 0;
          int j = 0;
          for (; j < m - 1; ++j) {
            cum += wts[j];
            if (cum >= total / 2) break;
          }
          combined = vals[j];
        }
        const float c = cm[i];
        const float o = float(c + strength * r * (combined - c));
        out[i] = o;
        changed[y] += o != c;
      }
  });
  for (int y = 0; y < rows; ++y) {
    st.regionPixels += region[y];
    st.pixelsChanged += changed[y];
  }
  long long rd = 0, rj = 0;
  for (int y = 0; y < rows; ++y) rd += reads[y], rj += rejected[y];
  st.rejectedFraction = rd ? double(rj) / rd : 0.0;
  st.combineMs = msSince(tc);
  st.totalMs = msSince(t0);
  if (stats) *stats = st;
}

}  // namespace ml
}  // namespace mr
