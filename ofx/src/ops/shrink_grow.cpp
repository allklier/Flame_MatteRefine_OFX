#include "ops/shrink_grow.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {

namespace {

// Profiles up to this far each side of the contour move rigidly (edges up to 12 px wide).
constexpr double kProfilePx = 6.0;
// Larger amounts move in passes: the band, and with it the cost, grows with the step.
constexpr double kMaxStepPx = 4.0;

// The body of src: src without its strands (see shrinkGrow), and below = body < 0.5. The
// opening is solid = src >= 0.5 eroded by the 3x3 box and dilated by the 5x5 one, beyond the
// frame the nearest pixel (the frame border is no edge). Row bands with their halo rows
// recomputed, so it is one pass over the frame; byte loops on plain locals so they vectorise
// (a uint8_t store may alias anything reached through a pointer).
void bodyOf(const float* src, int w, int h, float* body, uint8_t* below, Workers* wk) {
  constexpr int kRows = 32;
  dynamicFor<std::vector<uint8_t>>(wk, (h + kRows - 1) / kRows, [=](int b, std::vector<uint8_t>& buf) {
    const int W = w, H = h, y0 = b * kRows, y1 = std::min(H, y0 + kRows);
    const int lo = std::max(0, y0 - 3), hi = std::min(H - 1, y1 + 2);  // solid rows needed
    const int rows = hi - lo + 1;
    buf.resize(size_t(4) * (kRows + 6) * W);
    uint8_t* solid = buf.data();                       // rows lo..hi
    uint8_t* rowMin = solid + size_t(kRows + 6) * W;    // rows lo..hi: 1 x 3 min
    uint8_t* eroded = rowMin + size_t(kRows + 6) * W;   // rows lo..hi (lo + 1..hi - 1 used): 3 x 3 min
    uint8_t* rowMax = eroded + size_t(kRows + 6) * W;   // 5 x 1 max of eroded
    for (int r = 0; r < rows; ++r) {
      const float* s = src + size_t(lo + r) * W;
      uint8_t* o = solid + size_t(r) * W;
      for (int x = 0; x < W; ++x) o[x] = s[x] >= 0.5f;
      uint8_t* m = rowMin + size_t(r) * W;
      m[0] = std::min(o[0], o[W > 1 ? 1 : 0]);
      for (int x = 1; x + 1 < W; ++x) m[x] = std::min(std::min(o[x - 1], o[x]), o[x + 1]);
      if (W > 1) m[W - 1] = std::min(o[W - 2], o[W - 1]);
    }
    // Eroded rows a = max(0, y0 - 2) .. min(H - 1, y1 + 1), from rowMin rows a - 1..a + 1 clamped.
    const int ea = std::max(0, y0 - 2), eb = std::min(H - 1, y1 + 1);
    for (int a = ea; a <= eb; ++a) {
      const uint8_t* u = rowMin + size_t(std::max(a - 1, 0) - lo) * W;
      const uint8_t* c = rowMin + size_t(a - lo) * W;
      const uint8_t* d = rowMin + size_t(std::min(a + 1, H - 1) - lo) * W;
      uint8_t* e = eroded + size_t(a - lo) * W;
      for (int x = 0; x < W; ++x) e[x] = std::min(std::min(u[x], c[x]), d[x]);
      uint8_t* m = rowMax + size_t(a - lo) * W;
      auto clamped = [&](int x) {
        uint8_t v = e[x];
        for (int k = -2; k <= 2; ++k) v = std::max(v, e[std::min(std::max(x + k, 0), W - 1)]);
        m[x] = v;
      };
      for (int x = 0; x < std::min(2, W); ++x) clamped(x);
      for (int x = 2; x + 2 < W; ++x)
        m[x] = std::max(std::max(std::max(e[x - 2], e[x - 1]), std::max(e[x], e[x + 1])), e[x + 2]);
      for (int x = std::max(2, W - 2); x < W; ++x) clamped(x);
    }
    for (int y = y0; y < y1; ++y) {
      const uint8_t* r[5];
      for (int k = 0; k < 5; ++k) r[k] = rowMax + size_t(std::min(std::max(y + k - 2, 0), H - 1) - lo) * W;
      const uint8_t* so = solid + size_t(y - lo) * W;
      const float* s = src + size_t(y) * W;
      float* bo = body + size_t(y) * W;
      uint8_t* bl = below + size_t(y) * W;
      uint8_t* strand = rowMin + size_t(y - lo) * W;  // rowMin's row y - lo is no longer read
      for (int x = 0; x < W; ++x) {
        const uint8_t core = std::max(std::max(std::max(r[0][x], r[1][x]), std::max(r[2][x], r[3][x])), r[4][x]);
        strand[x] = so[x] & (core ^ 1);  // solid && !core (both 0 / 1)
      }
      for (int x = 0; x < W; ++x) {  // v = strand ? +0 : s[x], as bits (a select kept a branch)
        uint32_t bits;
        std::memcpy(&bits, &s[x], 4);
        bits &= uint32_t(strand[x]) - 1u;
        float v;
        std::memcpy(&v, &bits, 4);
        bo[x] = v;
        bl[x] = v < 0.5f;
      }
    }
  });
}

}  // namespace

// Each pass: the body = the matte without its strands. Parts of the >= 0.5 area that an opening
// by the 3x3 box (radius 1.5) removes, under ~3 px wide, are strands: detail that keeps its
// shape (moving them too made every strand of hair above 0.5 a 2 x Amount wide band, and their
// contours pulled the head's edge into lumps at the roots). out(p) = the body's profile at
// sd(p) + step along the normal (edge_field shiftAlongNormal's resampling in distance space).
// The step is full within kProfile + |step| of the contour and tapers to 0 over |step| + 1 px
// beyond, so detail inside near the edge (a grey face) is carried along without a seam and the
// mapping stays monotone. Grow: max with the input (strands stay, nothing gets lower); shrink:
// min with the input, except where the matte is more than the body (strands, hair).
void shrinkGrow(const float* in, float* out, int w, int h, const ShrinkGrowParams& p,
                ShrinkGrowStats* stats) {
  ShrinkGrowStats st;
  const size_t n = size_t(std::max(w, 0)) * std::max(h, 0);
  const double amount = p.amountPx;
  const int passes = amount == 0.0 ? 0 : int(std::ceil(std::fabs(amount) / kMaxStepPx));
  const double step = passes ? amount / passes : 0.0, a = std::fabs(step);
  const double rigid = kProfilePx + a, band = rigid + a + 1.0;
  if (passes == 0) {
    std::atomic<long long> changed(0);
    parallelFor(h, 16, [&](int y0, int y1) {
      long long c = 0;
      for (size_t i = size_t(y0) * w; i < size_t(y1) * w; ++i) c += in[i] != in[i];
      changed += c;
      if (in != out) std::copy(in + size_t(y0) * w, in + size_t(y1) * w, out + size_t(y0) * w);
    });
    st.pixelsChanged = changed;
    if (stats) *stats = st;
    return;
  }
  Workers wk;
  // Buffers for all passes. Each pass reads src and writes dst: the last one out, the others
  // alternate between two buffers. The last pass reads src only at the pixel it writes, so
  // out may be in; it also counts the changes against in (still as given until then).
  std::unique_ptr<float[]> body(new float[n]), sdBuf(new float[n]);
  std::unique_ptr<uint8_t[]> below(new uint8_t[n]);
  std::unique_ptr<float[]> passBuf[2];
  for (int k = 0; k < std::min(passes - 1, 2); ++k) passBuf[k].reset(new float[n]);
  float* sd = sdBuf.get();
  // Pixels farther than this from every contour pixel are beyond the band (sd = +-(band + 1)):
  // they keep their value.
  const int reach = int(std::ceil(band + 1.5)) + 1;
  std::atomic<long long> changed(0);
  const float* src = in;
  for (int pass = 0; pass < passes; ++pass) {
    const bool last = pass + 1 == passes;
    float* dst = last ? out : passBuf[pass % 2].get();
    bodyOf(src, w, h, body.get(), below.get(), &wk);
    const std::vector<int> contour = contourPixels(below.get(), w, h, &wk);
    signedEdgeDistanceExact(body.get(), below.get(), contour, w, h, 0.5f, band, sd, &wk);
    // Outside the contour the body's profile ends where the edge's own falloff does (0.5 /
    // its slope at the contour, + 1 px) and only falls going out (no pixel more than any nearer
    // it along the normal): so hair below 0.5 next to the edge stays detail and is not carried
    // along (it was, as a second copy Amount px out). Below 0.5: sd stays as it is.
    std::vector<float> falloffs(contour.size());
    parallelFor(&wk, int(contour.size()), 256, [&](int c0, int c1) {
      for (int c = c0; c < c1; ++c) {
        const int x = contour[size_t(c)] % w, y = contour[size_t(c)] / w;
        const double slope = contourSlope(body.get(), sd, w, h, x, y);
        falloffs[size_t(c)] = slope > 0.0 ? float(0.5 / slope) : -1.0f;
      }
    });
    const ContourIndex index(contour, w, h);
    const NearTiles near(contour, w, h, reach);
    // The new body values are collected first, per band of rows, and written after: they are
    // read from the old. Bands handed out one at a time: the work is where the contour is.
    constexpr int kRows = 8;
    const int bands = (h + kRows - 1) / kRows;
    std::vector<std::vector<std::pair<int, float>>> monoParts(static_cast<size_t>(bands));
    dynamicFor<int>(&wk, bands, [&](int b, int&) {
      std::vector<std::pair<int, float>>& part = monoParts[size_t(b)];
      const float* bd = body.get();
      for (int y = b * kRows; y < std::min(h, (b + 1) * kRows); ++y)
        near.runs(y, [&](int x0, int x1, bool isNear) {
          if (!isNear) return;
          for (int x = x0; x < x1; ++x) {
            const size_t i = size_t(y) * w + x;
            double v = bd[i], nx, ny, falloff;
            if (v > 0.0 && sd[i] < 0.0 && sd[i] > -band && edgeNormal(sd, w, h, x, y, nx, ny)) {
              const auto measure = [&](int qx, int qy) -> const float* {
                const int c = index.at(qx, qy);
                return c < 0 ? nullptr : &falloffs[size_t(c)];
              };
              if (footMeanAt(1, w, h, x - sd[i] * nx, y - sd[i] * ny, &falloff, measure) > 0.0 &&
                  -sd[i] > falloff + 1.0)
                v = 0.0;
              for (double t = 0.5; t < -sd[i] && v > 0.0; t += 0.5)
                v = std::min(v, double(sampleBilinear(bd, w, h, x + t * nx, y + t * ny)));
              part.emplace_back(int(i), float(v));
            }
          }
        });
    });
    parallelFor(&wk, bands, 1, [&](int b0, int b1) {
      for (int b = b0; b < b1; ++b)
        for (const auto& c : monoParts[size_t(b)]) body[size_t(c.first)] = c.second;
    });
    const float* bd = body.get();
    dynamicFor<int>(&wk, bands, [&](int b, int&) {
      long long c = 0;
      for (int y = b * kRows; y < std::min(h, (b + 1) * kRows); ++y)
        near.runs(y, [&](int x0, int x1, bool isNear) {
          const size_t r = size_t(y) * w;
          if (!isNear) {
            if (last)
              for (int x = x0; x < x1; ++x) c += src[r + x] != in[r + x];
            if (dst != src) std::copy(src + r + x0, src + r + x1, dst + r + x0);
            return;
          }
          for (int x = x0; x < x1; ++x) {
            const size_t i = r + x;
            const float s0 = src[i], orig = in[i];
            float o = s0;
            const double s = sd[i], k = std::min(1.0, (band - std::fabs(s)) / (a + 1.0));
            double nx, ny;
            if (k > 0.0 && edgeNormal(sd, w, h, x, y, nx, ny)) {
              const float v = profileAt(bd, sd, w, h, x, y, nx, ny, s + step * k);
              o = step > 0 ? std::max(s0, v) : std::min(s0, std::max(v, s0 - bd[i]));
            }
            dst[i] = o;
            if (last) c += o != orig;
          }
        });
      changed += c;
    });
    src = dst;
  }
  st.pixelsChanged = changed;
  st.passes = passes;
  if (stats) *stats = st;
}

}  // namespace mr
