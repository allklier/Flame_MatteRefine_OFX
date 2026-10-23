// Edge Refine (B1): see edge_refine.h. Moved out of mr_ml_bench (phase 0) so the tool and the
// plugin share one implementation.
#include "edge_refine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {
namespace ml {

namespace {
using Clock = std::chrono::steady_clock;
double msSince(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

std::vector<int> positions(int n, int tile, int stride) {
  std::vector<int> p;
  for (int x = 0;; x += stride) {
    if (x + tile >= n) {
      p.push_back(std::max(n - tile, 0));
      break;
    }
    p.push_back(x);
  }
  return p;
}
}  // namespace

float clampPicture(float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; }

std::vector<float> trimapFromDistance(const std::vector<float>& sd, double band, double outer) {
  const double reach = band + std::max(outer, 0.0);
  std::vector<float> t(sd.size());
  for (size_t i = 0; i < sd.size(); ++i) t[i] = sd[i] > band ? 1.0f : sd[i] < -reach ? 0.0f : 0.5f;
  return t;
}

void buildTrimap(const float* matte, int w, int h, double band, std::vector<float>* sd,
                 std::vector<float>* trimap, double outer) {
  outer = std::max(outer, 0.0);
  *sd = signedEdgeDistance(matte, w, h, 0.5f, band + 2);
  if (outer > 0) {
    // The reach beyond band + 2 from a second, wider distance: within band + 2 px of the edge the
    // values stay those of the symmetric band (the wider one may switch to the grid distance, off
    // by up to ~1 px, which would move the inside's fade with Outer px).
    const std::vector<float> far = signedEdgeDistance(matte, w, h, 0.5f, band + outer + 2);
    for (size_t i = 0; i < sd->size(); ++i)
      if (std::fabs((*sd)[i]) > band + 2) (*sd)[i] = far[i];
  }
  *trimap = trimapFromDistance(*sd, band, outer);
}

std::vector<Tile> selectTiles(const std::vector<float>& tri, int w, int h, int tile, int overlap) {
  std::vector<Tile> out;
  const int stride = std::max(tile - overlap, 1);
  for (int y : positions(h, tile, stride))
    for (int x : positions(w, tile, stride)) {
      bool any = false;
      for (int yy = y; yy < std::min(y + tile, h) && !any; ++yy)
        for (int xx = x; xx < std::min(x + tile, w); ++xx)
          if (tri[size_t(yy) * w + xx] == 0.5f) {
            any = true;
            break;
          }
      if (any) out.push_back({x, y});
    }
  return out;
}

float tileWeight(const Tile& t, int tile, int overlap, int w, int h, int lx, int ly) {
  auto ramp = [&](int d, bool frameEdge) {
    return frameEdge || overlap <= 0 ? 1.0f : std::min(1.0f, (d + 0.5f) / overlap);
  };
  return ramp(lx, t.x == 0) * ramp(tile - 1 - lx, t.x + tile >= w) * ramp(ly, t.y == 0) *
         ramp(tile - 1 - ly, t.y + tile >= h);
}

void prepareTile(const float* rgb, const float* tri, int w, int h, const Tile& t, int T, float* input,
                 Workers* wk) {
  const size_t plane = size_t(T) * T;
  // Rows in parallel (from when a pow per pixel was ~20 % of a frame); every value is computed
  // exactly as in a single-threaded loop, so the result is bit-identical to it.
  parallelFor(wk, T, 32, [&](int y0, int y1) {
    for (int ly = y0; ly < y1; ++ly)
      for (int lx = 0; lx < T; ++lx) {
        const size_t i = size_t(std::min(t.y + ly, h - 1)) * w + std::min(t.x + lx, w - 1);
        const size_t o = size_t(ly) * T + lx;
        for (int c = 0; c < 3; ++c) input[c * plane + o] = (clampPicture(rgb[i * 3 + c]) - 0.5f) / 0.5f;
        input[3 * plane + o] = tri[i];
      }
  });
}

void accumulateTile(const float* alpha, const Tile& t, int T, int overlap, int w, int h, float* acc,
                    float* wsum) {
  for (int ly = 0; ly < T && t.y + ly < h; ++ly)
    for (int lx = 0; lx < T && t.x + lx < w; ++lx) {
      const float wt = tileWeight(t, T, overlap, w, h, lx, ly);
      const size_t i = size_t(t.y + ly) * w + t.x + lx;
      acc[i] += wt * std::min(std::max(alpha[size_t(ly) * T + lx], 0.0f), 1.0f);
      wsum[i] += wt;
    }
}

// Both sides fade over the same width, band / 2, at the end of their reach: inside 1 up to band / 2,
// 0 at band; outside 1 up to band + outer - band / 2, 0 at band + outer. outer = 0: the symmetric
// rule, bit for bit (reach = band exactly).
double blendWeight(double sd, double band, double outer) {
  const double reach = sd >= 0 ? band : band + std::max(outer, 0.0), d = std::fabs(sd), fade = band / 2;
  if (d >= reach) return 0.0;
  return d <= reach - fade ? 1.0 : (reach - d) / fade;
}

long long blendBack(const float* in, const float* ml, const float* sd, size_t n, double band,
                    double mix, bool onlyGrey, float* out, double outer) {
  long long changed = 0;
  for (size_t i = 0; i < n; ++i) {
    const float v = in[i];
    out[i] = v;
    const double b = blendWeight(sd[i], band, outer);
    if (b <= 0.0 || ml[i] < 0) continue;  // decision 17: beyond the band bit-identical
    if (onlyGrey && (v <= 0.02f || v >= 0.98f)) continue;
    const double wt = b * mix;
    out[i] = wt >= 1.0 ? ml[i] : float(v + wt * (ml[i] - v));
    changed += out[i] != v;
  }
  return changed;
}

void refineEdge(const float* matte, const float* rgb, float* out, int w, int h, const RefineParams& p,
                TileRunner& runner, RefineStats* stats, const std::function<bool()>& abort,
                std::vector<float>* trimapOut) {
  RefineStats st;
  const auto t0 = Clock::now();
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  if (n == 0) {
    if (stats) *stats = st;
    return;
  }
  const double band = std::max(p.bandPx, 0.5), outer = std::max(p.outerPx, 0.0);
  std::vector<float> sd, tri;
  buildTrimap(matte, w, h, band, &sd, &tri, outer);
  long long unknown = 0;
  for (float v : tri) unknown += v == 0.5f;
  st.bandFraction = double(unknown) / n;
  // A frame smaller than the tile gets the smallest multiple of 32 that holds it.
  const int fit = (std::max(w, h) + 31) / 32 * 32;
  const int T = std::max(32, std::min(p.tile / 32 * 32, fit));
  st.tileSize = T;
  const int overlap = std::min(std::max(p.overlap, 0), T / 2);
  const std::vector<Tile> tiles = selectTiles(tri, w, h, T, overlap);
  st.tiles = int(tiles.size());
  st.trimapMs = msSince(t0);

  std::vector<float> acc(n, 0.0f), wsum(n, 0.0f), input(size_t(4) * T * T), alpha(size_t(T) * T);
  Workers wk;  // for prepareTile; joined when refineEdge returns
  for (const Tile& t : tiles) {
    if (abort && abort()) {
      st.aborted = true;
      break;
    }
    auto ts = Clock::now();
    prepareTile(rgb, tri.data(), w, h, t, T, input.data(), &wk);
    st.prepMs += msSince(ts);
    ts = Clock::now();
    runner.run(input.data(), T, alpha.data(), t.x, t.y);
    st.runMs += msSince(ts);
    ts = Clock::now();
    accumulateTile(alpha.data(), t, T, overlap, w, h, acc.data(), wsum.data());
    st.postMs += msSince(ts);
    ++st.tilesRun;
  }
  const auto tb = Clock::now();
  if (st.aborted) {
    if (out != matte) std::memcpy(out, matte, n * sizeof(float));
  } else {
    for (size_t i = 0; i < n; ++i) acc[i] = wsum[i] > 0 ? acc[i] / wsum[i] : -1.0f;
    st.pixelsChanged = blendBack(matte, acc.data(), sd.data(), n, band, std::min(std::max(p.mix, 0.0), 1.0),
                                 p.onlyGrey, out, outer);
  }
  st.blendMs = msSince(tb);
  st.totalMs = msSince(t0);
  if (trimapOut) trimapOut->swap(tri);
  if (stats) *stats = st;
}

}  // namespace ml
}  // namespace mr
