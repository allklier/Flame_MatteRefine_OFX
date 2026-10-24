#include "merge_morph.h"

#include <algorithm>
#include <cstring>

#include "ops/morphology.h"

namespace mr {
namespace merge {

namespace {

// 1-D squared distance transform of f (n samples, kFar = no site) into d (Felzenszwalb).
// v, z: scratch of n and n + 1.
void edt1d(const float* f, int n, float* d, int* v, float* z) {
  int k = 0;
  int first = -1;
  for (int q = 0; q < n; ++q)
    if (f[q] < kFar) {
      first = q;
      break;
    }
  if (first < 0) {
    for (int q = 0; q < n; ++q) d[q] = kFar;
    return;
  }
  v[0] = first;
  z[0] = -1e30f;
  z[1] = 1e30f;
  for (int q = first + 1; q < n; ++q) {
    if (f[q] >= kFar) continue;
    for (;;) {
      const int p = v[k];
      const float s = float(((double(f[q]) + double(q) * q) - (double(f[p]) + double(p) * p)) / (2.0 * (q - p)));
      if (s <= z[k]) {
        if (k == 0) {  // q dominates everything left of it
          v[0] = q;
          z[0] = -1e30f;
          z[1] = 1e30f;
          break;
        }
        --k;
        continue;
      }
      ++k;
      v[k] = q;
      z[k] = s;
      z[k + 1] = 1e30f;
      break;
    }
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    const float dq = float(q - v[k]);
    d[q] = dq * dq + f[v[k]];
  }
}

// van Herk / Gil-Werman along one axis, for `lanes` independent lines at once: sample j of lane l
// is at in[j * step + l] (rows: lanes = 1, step = 1 per line; columns: lanes = a strip of x, step =
// w, so the inner loop runs along a row and vectorises). Blocks of b = 2r + 1 samples: g = running
// op from each block's start, hb = from each block's end (or the last sample). A window [lo, hi]
// (clipped to [0, n - 1]) spans at most two blocks: op(hb[lo], g[hi]); within one block it starts
// at the block start (g[hi]) or ends at the last sample (hb[lo]).
// The same along one row (one lane), written scalar.
// No division per sample (it cost more than the filter): blocks are walked, and lo's position in
// its block (loMod) is counted along.
template <bool Min>
void vhgwRow(const float* in, float* out, int n, int r, float* g, float* hb) {
  const int b = 2 * r + 1;
  auto op = [](float a, float c) { return Min ? (c < a ? c : a) : (c > a ? c : a); };
  for (int s = 0; s < n; s += b) {
    const int e = std::min(n, s + b);
    g[s] = in[s];
    for (int j = s + 1; j < e; ++j) g[j] = op(g[j - 1], in[j]);
    hb[e - 1] = in[e - 1];
    for (int j = e - 2; j >= s; --j) hb[j] = op(hb[j + 1], in[j]);
  }
  int loMod = 0;  // lo % b
  for (int j = 0; j < n; ++j) {
    if (j - r > 0 && ++loMod == b) loMod = 0;  // lo advanced by one
    const int lo = std::max(0, j - r), hi = std::min(n - 1, j + r);
    // lo / b != hi / b  <=>  hi >= (lo's block start) + b
    out[j] = hi >= lo - loMod + b ? op(hb[lo], g[hi]) : loMod == 0 ? g[hi] : hb[lo];
  }
}

template <bool Min>
void vhgw(const float* in, float* out, int n, size_t step, int lanes, int r, float* g, float* hb) {
  const int b = 2 * r + 1;
  auto op = [](float a, float c) { return Min ? (c < a ? c : a) : (c > a ? c : a); };
  for (int j = 0; j < n; ++j) {
    const float* src = in + j * step;
    float* gj = g + size_t(j) * lanes;
    if (j % b == 0)
      for (int l = 0; l < lanes; ++l) gj[l] = src[l];
    else
      for (int l = 0; l < lanes; ++l) gj[l] = op(gj[l - lanes], src[l]);
  }
  for (int j = n - 1; j >= 0; --j) {
    const float* src = in + j * step;
    float* hj = hb + size_t(j) * lanes;
    if (j == n - 1 || (j + 1) % b == 0)
      for (int l = 0; l < lanes; ++l) hj[l] = src[l];
    else
      for (int l = 0; l < lanes; ++l) hj[l] = op(hj[l + lanes], src[l]);
  }
  for (int j = 0; j < n; ++j) {
    const int lo = std::max(0, j - r), hi = std::min(n - 1, j + r);
    float* dst = out + j * step;
    const float* gh = g + size_t(hi) * lanes;
    const float* hl = hb + size_t(lo) * lanes;
    if (lo / b != hi / b)
      for (int l = 0; l < lanes; ++l) dst[l] = op(hl[l], gh[l]);
    else if (lo % b == 0)
      for (int l = 0; l < lanes; ++l) dst[l] = gh[l];
    else  // same block, not from its start: the window is clipped at the end (hi = n - 1)
      for (int l = 0; l < lanes; ++l) dst[l] = hl[l];
  }
}

template <bool Min>
// tmpBuf: w*h floats for the row pass, or nullptr (allocated here).
void filter2d(const float* in, int w, int h, int r, float* out, Workers* wk, float* tmpBuf) {
  if (r <= 0) {
    std::memcpy(out, in, sizeof(float) * size_t(w) * h);
    return;
  }
  std::vector<float> own;
  if (!tmpBuf) own.resize(size_t(w) * h), tmpBuf = own.data();
  float* const tmp = tmpBuf;
  parallelFor(wk, h, 16, [&](int y0, int y1) {
    std::vector<float> g(static_cast<size_t>(w)), hb(static_cast<size_t>(w));
    for (int y = y0; y < y1; ++y)
      vhgwRow<Min>(in + size_t(y) * w, tmp + size_t(y) * w, w, r, g.data(), hb.data());
  });
  // Columns: strips of 64 columns, each processed as rows of 64 lanes (in place in the frame).
  const int strip = 64, strips = (w + strip - 1) / strip;
  parallelFor(wk, strips, 1, [&](int s0, int s1) {
    std::vector<float> g(size_t(h) * strip), hb(size_t(h) * strip);
    for (int s = s0; s < s1; ++s) {
      const int x0 = s * strip, lanes = std::min(strip, w - x0);
      vhgw<Min>(tmp + x0, out + x0, h, size_t(w), lanes, r, g.data(), hb.data());
    }
  });
}

}  // namespace

void distanceSquared(const uint8_t* set, int w, int h, float* d2, Workers* wk, std::vector<float>* scratch) {
  std::vector<float> own;
  std::vector<float>& colPass = scratch ? *scratch : own;
  if (colPass.size() < size_t(w) * h) colPass.resize(size_t(w) * h);
  // Columns in strips of 16: reads and writes run along rows (a column at a time touched one
  // cache line per row).
  constexpr int kStrip = 16;
  const int strips = (w + kStrip - 1) / kStrip;
  parallelFor(wk, strips, 1, [&](int s0, int s1) {
    std::vector<float> f(size_t(h) * kStrip), d(size_t(h) * kStrip), z(static_cast<size_t>(h) + 1);
    std::vector<int> v(static_cast<size_t>(h));
    for (int s = s0; s < s1; ++s) {
      const int x0 = s * kStrip, lanes = std::min(kStrip, w - x0);
      for (int y = 0; y < h; ++y) {
        const uint8_t* row = set + size_t(y) * w + x0;
        for (int c = 0; c < lanes; ++c) f[size_t(c) * h + y] = row[c] ? 0.0f : kFar;
      }
      for (int c = 0; c < lanes; ++c) edt1d(&f[size_t(c) * h], h, &d[size_t(c) * h], v.data(), z.data());
      for (int y = 0; y < h; ++y) {
        float* row = colPass.data() + size_t(y) * w + x0;
        for (int c = 0; c < lanes; ++c) row[c] = d[size_t(c) * h + y];
      }
    }
  });
  parallelFor(wk, h, 8, [&](int y0, int y1) {
    std::vector<float> z(static_cast<size_t>(w) + 1);
    std::vector<int> v(static_cast<size_t>(w));
    for (int y = y0; y < y1; ++y) {
      edt1d(colPass.data() + size_t(y) * w, w, d2 + size_t(y) * w, v.data(), z.data());
      for (int x = 0; x < w; ++x)
        if (d2[size_t(y) * w + x] > kFar) d2[size_t(y) * w + x] = kFar;
    }
  });
}

static float* scratchOf(std::vector<float>* scratch, size_t n) {
  if (!scratch) return nullptr;
  if (scratch->size() < n) scratch->resize(n);
  return scratch->data();
}

void minFilter(const float* in, int w, int h, int r, float* out, Workers* wk, std::vector<float>* scratch) {
  filter2d<true>(in, w, h, r, out, wk, scratchOf(scratch, size_t(w) * h));
}

void maxFilter(const float* in, int w, int h, int r, float* out, Workers* wk, std::vector<float>* scratch) {
  filter2d<false>(in, w, h, r, out, wk, scratchOf(scratch, size_t(w) * h));
}

void opening(const float* in, int w, int h, int r, float* out, Workers* wk, std::vector<float>* scratch) {
  const size_t n = size_t(w) * h;
  std::vector<float> own;
  std::vector<float>& sc = scratch ? *scratch : own;
  if (sc.size() < 2 * n) sc.resize(2 * n);
  float* e = sc.data();  // the erosion; the second half: the filters' row pass
  filter2d<true>(in, w, h, r, e, wk, sc.data() + n);
  filter2d<false>(e, w, h, r, out, wk, sc.data() + n);
}

}  // namespace merge
}  // namespace mr
