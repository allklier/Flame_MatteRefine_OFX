#include "merge_picreg.h"

#include <algorithm>
#include <cmath>

#include "ops/edge_field.h"
#include "ops/morphology.h"

namespace mr {
namespace merge {

namespace {

float finite(float v) { return std::isfinite(v) ? v : 0.0f; }

// Everything below runs per block on small windows, on the calling thread (a block is one work
// item): only the blocks near the zone with detail in them cost anything.

// Normalised Gaussian kernel, radius ceil(3 sigma).
std::vector<float> gaussKernel(double sigma) {
  const int r = std::max(1, int(std::ceil(3 * sigma)));
  std::vector<float> k(2 * r + 1);
  float sum = 0;
  for (int j = -r; j <= r; ++j) sum += k[j + r] = float(std::exp(-0.5 * j * j / (sigma * sigma)));
  for (float& e : k) e /= sum;
  return k;
}

// out = blur(in) on a w*h window (clamped at its border); tmp: scratch.
void blurInto(const float* in, float* out, int w, int h, const std::vector<float>& k, std::vector<float>& tmp) {
  const int r = int(k.size() / 2);
  tmp.resize(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float a = 0;
      for (int j = -r; j <= r; ++j) a += k[j + r] * in[size_t(y) * w + std::min(std::max(x + j, 0), w - 1)];
      tmp[size_t(y) * w + x] = a;
    }
  for (int y = 0; y < h; ++y) {
    float* o = out + size_t(y) * w;
    for (int x = 0; x < w; ++x) o[x] = 0;
    for (int j = -r; j <= r; ++j) {
      const float* t = &tmp[size_t(std::min(std::max(y + j, 0), h - 1)) * w];
      for (int x = 0; x < w; ++x) o[x] += k[j + r] * t[x];
    }
  }
}

// v -= blur(v): only local structure is left (tried 2026-10-24: a difference of Gaussians that
// also removes structure near the pixel scale made the busy scene worse, 0.37 -> up to 2.8 px).
void bandPass(float* v, int w, int h, const std::vector<float>& k, std::vector<float>& tmp, std::vector<float>& b) {
  b.resize(size_t(w) * h);
  blurInto(v, b.data(), w, h, k, tmp);
  for (size_t i = 0; i < size_t(w) * h; ++i) v[i] -= b[i];
}

// Min (Min = true) or max over a (2r + 1) square, window clipped to the buffer: naive separable.
template <bool Min>
void filterSq(const float* in, float* out, std::vector<float>& tmp, int w, int h, int r) {
  tmp.resize(size_t(w) * h);
  auto op = [](float a, float b) { return Min ? std::min(a, b) : std::max(a, b); };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float m = in[size_t(y) * w + x];
      for (int xx = std::max(0, x - r); xx <= std::min(w - 1, x + r); ++xx) m = op(m, in[size_t(y) * w + xx]);
      tmp[size_t(y) * w + x] = m;
    }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float m = tmp[size_t(y) * w + x];
      for (int yy = std::max(0, y - r); yy <= std::min(h - 1, y + r); ++yy) m = op(m, tmp[size_t(yy) * w + x]);
      out[size_t(y) * w + x] = m;
    }
}

struct Block {
  int bx0 = 0, by0 = 0;          // base px
  int qx0 = 0, qy0 = 0, qw = 0, qh = 0;  // the detail window (detail px)
  std::vector<float> D;          // band-passed thin detail in the window
  std::vector<float> pv, ps;     // band-passed picture V, S over the block, mean removed
  double vv = 0, ss = 0;         // their sums of squares
  bool cand = false;
  // this round's result
  bool valid = false;
  double sx = 0, sy = 0, quality = 0;
};

struct Scratch {
  std::vector<float> a, b, c, t1, t2, W, cv, cs;
};

// Sub-pixel offset of a peak from three samples (c0 the largest), in [-0.5, 0.5].
double parabola(double cm, double c0, double cp) {
  const double d = cm - 2 * c0 + cp;
  if (d >= 0) return 0.0;
  return std::max(-0.5, std::min(0.5, 0.5 * (cm - cp) / d));
}

// Largest eigenvalue of a symmetric n x n matrix (n <= 3), by power iteration.
double largestEigen(const double M[3][3], int n) {
  double v[3] = {1, 0.7, 0.4}, lam = 0;
  for (int it = 0; it < 200; ++it) {
    double u[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) u[i] += M[i][j] * v[j];
    double len = 0;
    for (int i = 0; i < n; ++i) len += u[i] * u[i];
    len = std::sqrt(len);
    if (len <= 0) return 0;
    lam = len;
    for (int i = 0; i < n; ++i) v[i] = u[i] / len;
  }
  return lam;
}

// Inverse of a symmetric n x n matrix (n <= 3), Gauss-Jordan; false if singular.
bool invert(const double A[3][3], int n, double out[3][3]) {
  double M[3][6];
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) M[i][j] = A[i][j], M[i][n + j] = i == j;
  for (int c = 0; c < n; ++c) {
    int p = c;
    for (int r = c + 1; r < n; ++r)
      if (std::fabs(M[r][c]) > std::fabs(M[p][c])) p = r;
    for (int j = 0; j < 2 * n; ++j) std::swap(M[c][j], M[p][j]);
    if (std::fabs(M[c][c]) < 1e-12) return false;
    const double d = M[c][c];
    for (int j = 0; j < 2 * n; ++j) M[c][j] /= d;
    for (int r = 0; r < n; ++r)
      if (r != c) {
        const double m = M[r][c];
        for (int j = 0; j < 2 * n; ++j) M[r][j] -= m * M[c][j];
      }
  }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) out[i][j] = M[i][n + j];
  return true;
}

}  // namespace

PictureFit registerToPicture(const float* detail, const float* picture, int w, int h, int openRadius,
                             const Registration& init, bool translationOnly, int searchPx, const uint8_t* zone,
                             Workers* wk) {
  PictureFit fit;
  fit.reg = init;
  fit.reg.delta = 0;
  fit.reg.ok = false;
  const double k = h / 1080.0;
  const int B = std::max(16, int(std::lround(48 * k)));
  if (w < 2 * B || h < 2 * B) return fit;
  const int S0 = std::max(2, searchPx);
  const int openR = std::max(1, openRadius);
  const std::vector<float> kern = gaussKernel(std::max(1.0, 1.5 * k));
  const int rb = int(kern.size() / 2);

  // The block grid (base px), centred; blocks with no zone pixel are skipped.
  const int bxN = w / B, byN = h / B, ox = (w - bxN * B) / 2, oy = (h - byN * B) / 2;
  std::vector<Block> blocks;
  for (int by = 0; by < byN; ++by)
    for (int bx = 0; bx < bxN; ++bx) {
      Block b;
      b.bx0 = ox + bx * B;
      b.by0 = oy + by * B;
      bool any = !zone;
      for (int y = b.by0; y < b.by0 + B && !any; y += 2)
        for (int x = b.bx0; x < b.bx0 + B; x += 2)
          if (zone[size_t(y) * w + x]) {
            any = true;
            break;
          }
      if (any) blocks.push_back(std::move(b));
    }

  // Per block: the detail window covering the block moved by init (+- the search and a margin for
  // the opening and the blur), its thin structures (detail minus its opening) band-passed; the
  // picture block band-passed. A block is a candidate with at least 3 % of its pixels above 0.1 in
  // the thin detail (sampled every 2nd pixel).
  dynamicFor<Scratch>(wk, int(blocks.size()), [&](int bi, Scratch& s) {
    Block& b = blocks[bi];
    double ax, ay, bx, by;
    init.apply(b.bx0 - S0 - 2, b.by0 - S0 - 2, &ax, &ay);
    init.apply(b.bx0 + B + S0 + 2, b.by0 + B + S0 + 2, &bx, &by);
    const int m = 2 * openR + rb + 2;
    b.qx0 = std::max(0, int(std::floor(std::min(ax, bx))) - m);
    b.qy0 = std::max(0, int(std::floor(std::min(ay, by))) - m);
    const int qx1 = std::min(w - 1, int(std::ceil(std::max(ax, bx))) + m);
    const int qy1 = std::min(h - 1, int(std::ceil(std::max(ay, by))) + m);
    if (qx1 - b.qx0 < 4 || qy1 - b.qy0 < 4) return;
    b.qw = qx1 - b.qx0 + 1;
    b.qh = qy1 - b.qy0 + 1;
    const size_t qn = size_t(b.qw) * b.qh;
    // Quick reject: the window has no grey (all near 0 or all near 1).
    float lo = 1, hi = 0;
    for (int y = 0; y < b.qh; y += 2)
      for (int x = 0; x < b.qw; x += 2) {
        const float v = detail[size_t(b.qy0 + y) * w + b.qx0 + x];
        lo = std::min(lo, v), hi = std::max(hi, v);
      }
    if (hi < 0.1f || lo > 0.9f) return;
    s.a.resize(qn);
    for (int y = 0; y < b.qh; ++y)
      std::copy(detail + size_t(b.qy0 + y) * w + b.qx0, detail + size_t(b.qy0 + y) * w + b.qx0 + b.qw, &s.a[size_t(y) * b.qw]);
    s.b.resize(qn);
    s.c.resize(qn);
    filterSq<true>(s.a.data(), s.b.data(), s.t1, b.qw, b.qh, openR);
    filterSq<false>(s.b.data(), s.c.data(), s.t1, b.qw, b.qh, openR);
    b.D.resize(qn);
    for (size_t i = 0; i < qn; ++i) b.D[i] = std::max(0.0f, s.a[i] - s.c[i]);
    int cnt = 0;
    for (int y = b.by0; y < b.by0 + B; y += 2)
      for (int x = b.bx0; x < b.bx0 + B; x += 2) {
        double qx, qy;
        init.apply(x, y, &qx, &qy);
        cnt += sampleBilinear(b.D.data(), b.qw, b.qh, qx - b.qx0, qy - b.qy0) > 0.1f;
      }
    if (cnt * 4 < 0.03 * B * B) {
      std::vector<float>().swap(b.D);
      return;
    }
    bandPass(b.D.data(), b.qw, b.qh, kern, s.t1, s.c);
    // The picture block with a blur margin.
    const int px0 = std::max(0, b.bx0 - rb), py0 = std::max(0, b.by0 - rb);
    const int px1 = std::min(w - 1, b.bx0 + B - 1 + rb), py1 = std::min(h - 1, b.by0 + B - 1 + rb);
    const int pw = px1 - px0 + 1, ph = py1 - py0 + 1;
    s.a.resize(size_t(pw) * ph);
    s.b.resize(size_t(pw) * ph);
    for (int y = 0; y < ph; ++y)
      for (int x = 0; x < pw; ++x) {
        const float* p = picture + (size_t(py0 + y) * w + px0 + x) * 3;
        const float r = finite(p[0]), g = finite(p[1]), bl = finite(p[2]);
        s.a[size_t(y) * pw + x] = (r + g + bl) / 3.0f;
        s.b[size_t(y) * pw + x] = std::max(r, std::max(g, bl)) - std::min(r, std::min(g, bl));
      }
    bandPass(s.a.data(), pw, ph, kern, s.t1, s.c);
    bandPass(s.b.data(), pw, ph, kern, s.t1, s.c);
    const double N = double(B) * B;
    double mv = 0, ms = 0;
    for (int y = 0; y < B; ++y)
      for (int x = 0; x < B; ++x) {
        const size_t j = size_t(b.by0 + y - py0) * pw + b.bx0 + x - px0;
        mv += s.a[j], ms += s.b[j];
      }
    mv /= N;
    ms /= N;
    b.pv.resize(size_t(B) * B);
    b.ps.resize(size_t(B) * B);
    b.vv = b.ss = 0;
    for (int y = 0; y < B; ++y)
      for (int x = 0; x < B; ++x) {
        const size_t j = size_t(b.by0 + y - py0) * pw + b.bx0 + x - px0, o = size_t(y) * B + x;
        b.pv[o] = float(s.a[j] - mv);
        b.ps[o] = float(s.b[j] - ms);
        b.vv += double(b.pv[o]) * b.pv[o];
        b.ss += double(b.ps[o]) * b.ps[o];
      }
    b.cand = true;
  });
  std::vector<int> cand;
  for (int i = 0; i < int(blocks.size()); ++i)
    if (blocks[i].cand) cand.push_back(i);
  fit.blocks = int(cand.size());
  if (fit.blocks < (translationOnly ? 4 : 8)) return fit;

  const double cx = init.cx, cy = init.cy, R = std::max(w, h) / 2.0;
  const int np = translationOnly ? 2 : 3;
  std::vector<double> wt(cand.size());
  Registration reg = fit.reg;
  for (int round = 0; round < 3; ++round) {
    const int S = round == 0 ? S0 : 2;
    dynamicFor<Scratch>(wk, int(cand.size()), [&](int ci, Scratch& s) {
      Block& br = blocks[cand[ci]];
      br.valid = false;
      const int RW = B + 2 * S;
      s.W.resize(size_t(RW) * RW);
      for (int y = 0; y < RW; ++y)
        for (int x = 0; x < RW; ++x) {
          double qx, qy;
          reg.apply(br.bx0 - S + x, br.by0 - S + y, &qx, &qy);
          s.W[size_t(y) * RW + x] = sampleBilinear(br.D.data(), br.qw, br.qh, qx - br.qx0, qy - br.qy0);
        }
      const double N = double(B) * B;
      const int SW = 2 * S + 1;
      s.cv.assign(size_t(SW) * SW, 0.0f);
      s.cs.assign(size_t(SW) * SW, 0.0f);
      for (int sy = -S; sy <= S; ++sy)
        for (int sx = -S; sx <= S; ++sx) {
          double nv = 0, ns = 0, sw = 0, sww = 0;
          for (int y = 0; y < B; ++y) {
            const float* wr = &s.W[size_t(y + S + sy) * RW + S + sx];
            const float* a = &br.pv[size_t(y) * B];
            const float* b = &br.ps[size_t(y) * B];
            float fv = 0, fs = 0, fw = 0, fww = 0;
            for (int x = 0; x < B; ++x) {
              const float q = wr[x];
              fv += q * a[x];
              fs += q * b[x];
              fw += q;
              fww += q * q;
            }
            nv += fv, ns += fs, sw += fw, sww += fww;
          }
          const double var = sww - sw * sw / N;
          if (var <= 1e-12) continue;
          const size_t j = size_t(sy + S) * SW + sx + S;
          s.cv[j] = br.vv > 1e-20 ? float(nv / std::sqrt(var * br.vv)) : 0.0f;
          s.cs[j] = br.ss > 1e-20 ? float(ns / std::sqrt(var * br.ss)) : 0.0f;
        }
      // The channel and shift with the largest |NCC|; its sign is the polarity.
      int best = -1;
      bool useS = false;
      float bestAbs = 0;
      for (int j = 0; j < SW * SW; ++j) {
        if (std::fabs(s.cv[j]) > bestAbs) bestAbs = std::fabs(s.cv[j]), best = j, useS = false;
        if (std::fabs(s.cs[j]) > bestAbs) bestAbs = std::fabs(s.cs[j]), best = j, useS = true;
      }
      if (best < 0) return;
      const int jx = best % SW, jy = best / SW;
      if (jx == 0 || jy == 0 || jx == SW - 1 || jy == SW - 1) return;  // peak at the search border
      const std::vector<float>& c = useS ? s.cs : s.cv;
      const double sgn = c[best] > 0 ? 1.0 : -1.0;
      auto at = [&](int x, int y) { return sgn * c[size_t(y) * SW + x]; };
      br.sx = jx - S + parabola(at(jx - 1, jy), at(jx, jy), at(jx + 1, jy));
      br.sy = jy - S + parabola(at(jx, jy - 1), at(jx, jy), at(jx, jy + 1));
      br.quality = bestAbs;
      br.valid = bestAbs >= 0.25f;
    });

    // Robust fit of s(p) = dt + du (p - c) / R to the block shifts (IRLS, Tukey biweight on the
    // shift residual, scale 1.4826 x the median residual, floor 0.1 px), weights quality^2.
    auto resid = [&](const Block& br, const double* d, double* rx, double* ry) {
      const double pxc = br.bx0 + (B - 1) / 2.0, pyc = br.by0 + (B - 1) / 2.0;
      *rx = br.sx - (d[0] + d[2] * (pxc - cx) / R);
      *ry = br.sy - (d[1] + d[2] * (pyc - cy) / R);
    };
    auto addNormal = [&](const Block& br, double ww, double A[3][3], double* rhs) {
      const double pxc = br.bx0 + (B - 1) / 2.0, pyc = br.by0 + (B - 1) / 2.0;
      const double jx[3] = {1, 0, (pxc - cx) / R}, jy[3] = {0, 1, (pyc - cy) / R};
      for (int i = 0; i < np; ++i) {
        for (int j = 0; j < np; ++j) A[i][j] += ww * (jx[i] * jx[j] + jy[i] * jy[j]);
        if (rhs) rhs[i] += ww * (jx[i] * br.sx + jy[i] * br.sy);
      }
    };
    double d[3] = {0, 0, 0}, cut = 1e9;
    int nValid = 0;
    for (int ci : cand) nValid += blocks[ci].valid;
    if (nValid < (translationOnly ? 4 : 8)) return fit;
    for (int it = 0; it < 12; ++it) {
      double A[3][3] = {{0}}, rhs[3] = {0, 0, 0};
      for (size_t c = 0; c < cand.size(); ++c) {
        const Block& br = blocks[cand[c]];
        wt[c] = 0;
        if (!br.valid) continue;
        double rx, ry;
        resid(br, d, &rx, &ry);
        const double t = std::hypot(rx, ry) / cut;
        const double tw = it == 0 ? 1.0 : (t < 1 ? (1 - t * t) * (1 - t * t) : 0.0);
        wt[c] = br.quality * br.quality * tw;
        if (wt[c] > 0) addNormal(br, wt[c], A, rhs);
      }
      double Ai[3][3];
      if (!invert(A, np, Ai)) return fit;
      double nd[3] = {0, 0, 0};
      for (int i = 0; i < np; ++i)
        for (int j = 0; j < np; ++j) nd[i] += Ai[i][j] * rhs[j];
      for (int i = 0; i < 3; ++i) d[i] = nd[i];
      std::vector<double> rr;
      for (int ci : cand)
        if (blocks[ci].valid) {
          double rx, ry;
          resid(blocks[ci], d, &rx, &ry);
          rr.push_back(std::hypot(rx, ry));
        }
      std::nth_element(rr.begin(), rr.begin() + rr.size() / 2, rr.end());
      cut = std::max(0.5, 4.685 * std::max(0.1, 1.4826 * rr[rr.size() / 2]));
    }
    // The block shifts are relative to the current move: A(p + s(p)) = A(p) + scale s(p).
    reg.tx += reg.scale * d[0];
    reg.ty += reg.scale * d[1];
    if (!translationOnly) reg.scale = std::max(0.95, std::min(1.05, reg.scale * (1.0 + d[2] / R)));

    if (round == 2) {
      // Statistics of the last round (its fit is the last, small correction).
      double A[3][3] = {{0}}, swr = 0, sw = 0;
      int inl = 0;
      for (size_t c = 0; c < cand.size(); ++c) {
        if (wt[c] <= 0) continue;
        double rx, ry;
        resid(blocks[cand[c]], d, &rx, &ry);
        swr += wt[c] * (rx * rx + ry * ry);
        sw += wt[c];
        ++inl;
        addNormal(blocks[cand[c]], wt[c], A, nullptr);
      }
      fit.inliers = inl;
      fit.rmsPx = sw > 0 ? std::sqrt(swr / (2 * sw)) : 0.0;
      // sigma0^2 = sum w r^2 / (observations - parameters), a block of weight w having variance
      // sigma0^2 / w; floored at (0.05 px)^2 for a block of mean weight: block shifts share small
      // systematic errors that their scatter does not show.
      const double meanW = sw / std::max(1, inl);
      const double var = std::max(0.0025 * meanW, swr / std::max(1.0, 2.0 * inl - np));
      double Ai[3][3];
      fit.uncertaintyPx = 1e9;
      if (inl > 0 && invert(A, np, Ai)) {
        for (int i = 0; i < np; ++i)
          for (int j = 0; j < np; ++j) Ai[i][j] *= var;
        fit.uncertaintyPx = std::sqrt(largestEigen(Ai, np));
      }
      fit.ok = inl >= (translationOnly ? 4 : 8);
    }
  }
  reg.delta = 0;
  reg.ok = fit.ok;
  reg.samples = fit.inliers;
  reg.rmsPx = fit.rmsPx;
  reg.sensitivity = 0;
  fit.reg = reg;
  return fit;
}

}  // namespace merge
}  // namespace mr
